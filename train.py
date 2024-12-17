# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.datasets.mixtera_datasets import build_mixtera_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.models.llama.model import PerDomainLoss
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.utils import device_module, device_type

from mixtera.torch import MixteraTorchDataset
from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import InferringMixture, StaticMixture, MixtureKey
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.algo.ado.ado import AdoDynamicMixing
from mixtera.utils.feedback import handle_mixtera_feedback

# Query execution in Mixtera takes long, and NCCL would time out otherwise.
os.environ["NCCL_TIMEOUT"] = str(30 * 60 * 1000)

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        dp_group = dp_mesh.get_group()
    else:
        dp_degree, dp_rank = 1, 0
        dp_group = None

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    if parallel_dims.tp_enabled:
        tp_mesh = world_mesh["tp"]
        tp_rank = tp_mesh.get_local_rank()
    else:
        tp_rank = 0

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(world_mesh, device, job_config)
    model_name = job_config.model.name

    # build tokenizer
    # tokenizer_type = model_name_to_tokenizer[model_name]
    # tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
    
    # Mixtera setup (todo: make this config parameter)
    client = MixteraClient.from_remote("localhost", 8888)
    job_id = "torchtitan_test"
    chunk_size = 512
    tunnel_via_server = False
    chunk_reading_degree_of_parallelism = 1
    num_workers = 16
    tokenizer = "EleutherAI/gpt-neox-20b"

    coordinate = world_mesh.get_coordinate()
    mesh_dim_names = world_mesh.mesh_dim_names
    mesh_sizes = world_mesh.size_per_dim()

    # Map dimension names to indices and sizes
    coord_dict = dict(zip(mesh_dim_names, coordinate))
    size_dict = dict(zip(mesh_dim_names, mesh_sizes))

    # Get dp_group_id (data parallel group ID)
    dp_group_id = coord_dict.get("dp", 0)  # Default to 0 if dp not enabled
    assert dp_group_id == dp_rank

    # Compute node_id (unique index within model parallel group)
    non_dp_dims = [dim for dim in mesh_dim_names if dim != "dp"]
    node_id = 0
    multiplier = 1
    for dim_name in reversed(non_dp_dims):
        idx = coord_dict[dim_name]
        size = size_dict[dim_name]
        node_id += idx * multiplier
        multiplier *= size

    # Compute nodes_per_dp_group
    nodes_per_dp_group = 1
    for dim_name in non_dp_dims:
        nodes_per_dp_group *= size_dict[dim_name]

    assert nodes_per_dp_group == world_size // dp_degree
    assert nodes_per_dp_group == parallel_dims.non_data_parallel_size
    assert nodes_per_dp_group * dp_degree == world_size

    logger.info(f"dp_group_id: {dp_group_id}, dp_degree: {dp_degree}, node_id: {node_id}, nodes_per_dp_group: {nodes_per_dp_group}")

    mixture_pile_static = StaticMixture(chunk_size=chunk_size, mixture={
        MixtureKey({"pile_set_name": ["FreeLaw"]}): 0.04493927695030662,
        MixtureKey({"pile_set_name": ["Enron Emails"]}): 0.000998021865918546,
        MixtureKey({"pile_set_name": ["Github"]}): 0.12267758913758665,
        MixtureKey({"pile_set_name": ["OpenSubtitles"]}): 0.015835745965429738,
        MixtureKey({"pile_set_name": ["PubMed Central"]}): 0.12148621531516873,
        MixtureKey({"pile_set_name": ["OpenWebText2"]}): 0.10960682218906206,
        MixtureKey({"pile_set_name": ["StackExchange"]}): 0.049107965728456646,
        MixtureKey({"pile_set_name": ["Pile-CC"]}): 0.1824984780261193,
        MixtureKey({"pile_set_name": ["ArXiv"]}): 0.08862621733009907,
        MixtureKey({"pile_set_name": ["USPTO Backgrounds"]}): 0.02616577419097875,
        MixtureKey({"pile_set_name": ["Books3"]}): 0.10458626728299704,
        MixtureKey({"pile_set_name": ["Wikipedia (en)"]}): 0.04016661238580172,
        MixtureKey({"pile_set_name": ["PubMed Abstracts"]}): 0.02212837481440004,
        MixtureKey({"pile_set_name": ["NIH ExPorter"]}): 0.0018685647881937016,
        MixtureKey({"pile_set_name": ["BookCorpus2"]}): 0.006327357399975309,
        MixtureKey({"pile_set_name": ["EuroParl"]}): 0.008072738376112661,
        MixtureKey({"pile_set_name": ["HackerNews"]}): 0.004731183407655429,
        MixtureKey({"pile_set_name": ["DM Mathematics"]}): 0.019084626704901235,
        MixtureKey({"pile_set_name": ["YoutubeSubtitles"]}): 0.004027438721554198,
        MixtureKey({"pile_set_name": ["PhilPapers"]}): 0.0026731438901686708,
        MixtureKey({"pile_set_name": ["Ubuntu IRC"]}): 0.004850316881507234,
        MixtureKey({"pile_set_name": ["Gutenberg (PG-19)"]}): 0.0195412686476066,
    })

    mixture_ado = DynamicMixture(chunk_size=chunk_size, initial_mixture=mixture_pile_static, mixing_alg=AdoDynamicMixing(variant="vanilla"))
    mixture = mixture_ado

    query_execution_args = QueryExecutionArgs(mixture=mixture, dp_groups=dp_degree, nodes_per_group=nodes_per_dp_group, num_workers=num_workers)
    streaming_args = ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=tunnel_via_server, 
                                         chunk_reading_degree_of_parallelism=chunk_reading_degree_of_parallelism,
                                         chunk_reading_mixture_type="token", chunk_reading_tokenizer=tokenizer, chunk_reading_sequence_len=job_config.training.seq_len,
                                         chunk_reading_token_overlapping=False, chunk_reading_eos=True, chunk_reading_bos=True)

    query = Query.for_job(job_id).select(None)

    raw_dataset = MixteraTorchDataset(client, query, query_execution_args, streaming_args, checkpoint_path=None)

    # build dataloader
    data_loader = build_mixtera_data_loader(
        raw_dataset, job_config.training.batch_size, num_workers
    )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    #def loss_fn(pred, labels):
    #    return torch.nn.functional.cross_entropy(
    #        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    #    )

    #if job_config.training.compile:
    #    loss_fn = torch.compile(loss_fn)

    per_domain_loss_module = PerDomainLoss(device=device) 

    if job_config.training.compile:
        per_domain_loss_module = torch.compile(per_domain_loss_module)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        raise NotImplementedError("do not support per domain loss with pp yet")
        # apply PT-D Pipeline Parallel
        pp_schedule, model_parts = models_pipelining_fns[model_name](
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            m.init_weights(buffer_device=buffer_device)
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        model.init_weights(buffer_device=buffer_device)
        model.train()

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            input_ids, labels, key_ids = batch
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            key_ids = key_ids.to(device_type)
            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels, model.freqs_cis],
                    cp_seq_dims=[1, 1, 0],
                    cp_no_restore_buffers={input_ids, labels},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # todo mixtera support
                raise NotImplementedError("no mixtera support for pp yet")
                # Pipeline Parallel forward / backward inside step() call
                is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                with train_context(optional_context_parallel_ctx):
                    if pp_mesh.get_local_rank() == 0:
                        pp_schedule.step(input_ids)
                    elif is_last_stage:
                        losses = []
                        pp_schedule.step(target=labels, losses=losses)
                    else:
                        pp_schedule.step()

                # accumulate losses across pipeline microbatches
                loss = (
                    torch.mean(torch.stack(losses))
                    if is_last_stage
                    else torch.Tensor([-1.0])
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    pred = model(input_ids)
                    loss = per_domain_loss_module(pred, labels, key_ids)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

        if per_domain_loss_module.has_per_domain_loss:
            # First, all-reduce max_domain_id to get global_max_domain_id
            max_domain_id_tensor = per_domain_loss_module.max_domain_id.clone()
            if parallel_dims.dp_enabled:
                torch.distributed.all_reduce(max_domain_id_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
            global_max_domain_id = max_domain_id_tensor.item()

            # Resize losses_tensor and counts_tensor if needed
            num_domains = global_max_domain_id + 1
            if num_domains > per_domain_loss_module.losses_tensor.size(0):
                extra_size = num_domains - per_domain_loss_module.losses_tensor.size(0)
                per_domain_loss_module.losses_tensor = torch.cat([
                    per_domain_loss_module.losses_tensor,
                    torch.zeros(extra_size, dtype=per_domain_loss_module.losses_tensor.dtype, device=per_domain_loss_module.device)
                ], dim=0)
                per_domain_loss_module.counts_tensor = torch.cat([
                    per_domain_loss_module.counts_tensor,
                    torch.zeros(extra_size, dtype=per_domain_loss_module.counts_tensor.dtype, device=per_domain_loss_module.device)
                ], dim=0)

            # Now perform all-reduce on losses_tensor and counts_tensor
            if parallel_dims.dp_enabled:
                torch.distributed.all_reduce(per_domain_loss_module.losses_tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
                torch.distributed.all_reduce(per_domain_loss_module.counts_tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)

            losses_per_domain_np = per_domain_loss_module.losses_tensor.cpu().numpy()
            counts_per_domain_np = per_domain_loss_module.counts_tensor.cpu().numpy()

            handle_mixtera_feedback(
                raw_dataset,
                train_state.step,
                losses_per_domain_np,
                counts_per_domain_np,
                dp_rank,
                tp_rank,
            )

            # Reset per-domain loss module
            per_domain_loss_module.reset_per_domain_stats()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, dp_mesh),
                        utils.dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
