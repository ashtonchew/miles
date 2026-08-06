import asyncio
import logging
import os
from typing import TypeVar

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _await_task_terminal(task: asyncio.Future[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _await_task_before_cancellation(task: asyncio.Future[_T]) -> _T:
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await _await_task_terminal(task)
        except BaseException as terminal_error:
            raise cancellation from terminal_error
        raise


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    async def update_weights_with_admission_hold(rollout_id: int | None) -> None:
        acquire_ref = rollout_manager.acquire_train_admission_hold.remote()
        acquire_task = asyncio.ensure_future(acquire_ref)
        try:
            hold_id = await asyncio.shield(acquire_task)
        except asyncio.CancelledError as cancellation:
            try:
                hold_id = await _await_task_terminal(acquire_task)
                release_task = asyncio.ensure_future(rollout_manager.release_train_admission_hold.remote(hold_id))
                await _await_task_terminal(release_task)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        # Once published, failures retain the hold because rollout or weight state may be indeterminate.
        await rollout_manager.wait_train_admission_hold.remote(hold_id)
        if rollout_id is None:
            await actor_model.update_weights()
        else:
            await actor_model.update_weights(rollout_id=rollout_id)

        async def commit_weight_update() -> None:
            await rollout_manager.record_train_weight_update.remote(hold_id)
            await rollout_manager.release_train_admission_hold.remote(hold_id)

        # A successful update records its boundary and releases admission as one cancellation-safe commit.
        commit_task = asyncio.create_task(commit_weight_update())
        await _await_task_before_cancellation(commit_task)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights_with_admission_hold(None)

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    rollout_data_curr_ref = None
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    async def sync_prefetched_rollout() -> None:
        nonlocal rollout_data_curr_ref, rollout_data_next_future
        if rollout_data_next_future is None:
            return
        rollout_data_curr_ref = await rollout_data_next_future
        rollout_data_next_future = None

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        await sync_prefetched_rollout()

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await sync_prefetched_rollout()
            await rollout_manager.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            await sync_prefetched_rollout()
            await update_weights_with_admission_hold(rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
