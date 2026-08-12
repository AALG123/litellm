import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS,
    SPEND_LOG_CLEANUP_BATCH_SIZE,
    SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS,
    SPEND_LOG_CLEANUP_JOB_NAME,
    SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES,
    SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP,
    SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS,
    SPEND_LOG_RUN_LOOPS,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import (
    RunOutcome,
    SpendLogCleanupMetrics,
)
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import (
    SpendLogsPartitionManager,
)
from litellm.proxy.utils import PrismaClient

StopReason: TypeAlias = Literal["exhausted", "budget_exhausted", "batch_cap_reached", "aborted"]


@dataclass(frozen=True, slots=True)
class TableCleanupResult:
    """Outcome of pruning one table, so the caller can report why a run ended."""

    rows_deleted: int
    stop_reason: StopReason


class _RemainingRow(BaseModel):
    """One row of the capped outstanding-rows probe, validated out of prisma's untyped result."""

    remaining: int


_REMAINING_ROWS: Final = TypeAdapter(list[_RemainingRow])


class SpendLogCleanup:
    """
    Handles cleaning up old spend logs based on maximum retention period.

    When LiteLLM_SpendLogs is range-partitioned, expired data is reclaimed by
    dropping whole partitions (instant, frees disk immediately). Otherwise it
    falls back to deleting logs in batches.
    Uses PodLockManager to ensure only one pod runs cleanup in multi-pod deployments.

    Every run is bounded three ways so it can never monopolise the database: a
    wall-clock budget shared across all tables, a per-table batch cap, and a
    Postgres statement/lock timeout on each individual delete. A run that hits a
    bound stops cleanly and the next run resumes from where it left off, because
    the cutoff is recomputed and deleted rows are gone.
    """

    def __init__(
        self,
        general_settings=None,
        redis_cache: RedisCache | None = None,
        partition_manager: SpendLogsPartitionManager | None = None,
    ):
        self.retention_seconds: int | None = None
        self.partition_manager = partition_manager or SpendLogsPartitionManager()
        from litellm.proxy.proxy_server import general_settings as default_settings

        self.general_settings = general_settings or default_settings
        self.batch_size = self._positive_int_setting(
            "maximum_spend_logs_cleanup_batch_size", SPEND_LOG_CLEANUP_BATCH_SIZE
        )
        self.max_batches = self._positive_int_setting("maximum_spend_logs_cleanup_max_batches", SPEND_LOG_RUN_LOOPS)
        self.run_budget_seconds = self._duration_setting(
            "maximum_spend_logs_cleanup_run_budget", SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS
        )
        self.batch_timeout_seconds = self._duration_setting(
            "maximum_spend_logs_cleanup_batch_timeout", SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS
        )
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
        self.pod_lock_manager = pod_lock_manager
        verbose_proxy_logger.info(
            "SpendLogCleanup initialized: batch_size=%s max_batches=%s run_budget=%ss batch_timeout=%ss",
            self.batch_size,
            self.max_batches,
            self.run_budget_seconds,
            self.batch_timeout_seconds,
        )

    def _positive_int_setting(self, setting_name: str, default: int) -> int:
        """
        Read a positive-integer knob, falling back to the default when unset or unusable.
        """
        raw: Final = self.general_settings.get(setting_name)
        if raw is None:
            return default
        try:
            parsed: Final = int(raw)
        except (TypeError, ValueError):
            verbose_proxy_logger.warning("Invalid %s value: %s, using default %s", setting_name, raw, default)
            return default
        if parsed <= 0:
            verbose_proxy_logger.warning("%s must be positive, got %s, using default %s", setting_name, parsed, default)
            return default
        return parsed

    def _duration_setting(self, setting_name: str, default_seconds: float) -> float:
        """
        Read a duration knob (e.g. '5m'), falling back to the default when unset or unusable.

        The knob must never be able to remove the bound it exists to enforce, so
        anything the parser rejects (including the non-finite spellings 'inf' and
        'nan') and anything non-positive falls back rather than being honoured.
        """
        raw: Final = self.general_settings.get(setting_name)
        if raw is None:
            return default_seconds
        try:
            parsed: Final = float(duration_in_seconds(str(raw)))
        except (ValueError, TypeError) as e:
            verbose_proxy_logger.warning(
                "Invalid %s value: %s (%s), using default %ss", setting_name, raw, e, default_seconds
            )
            return default_seconds
        if parsed <= 0:
            verbose_proxy_logger.warning(
                "%s must be a positive duration, got %s, using default %ss", setting_name, raw, default_seconds
            )
            return default_seconds
        return parsed

    def _retention_seconds_for(self, setting_name: str) -> int | None:
        """
        Parse one retention setting into seconds, or None when unset or invalid.
        """
        retention_setting = self.general_settings.get(setting_name)
        verbose_proxy_logger.info("Checking %s: %s", setting_name, retention_setting)

        if retention_setting is None:
            return None

        try:
            if isinstance(retention_setting, int):
                verbose_proxy_logger.warning(
                    "%s is an integer (%s); treating as days. Use a string like '3d' to be explicit.",
                    setting_name,
                    retention_setting,
                )
                retention_setting = f"{retention_setting}d"
            retention_seconds: Final = duration_in_seconds(retention_setting)
        except ValueError as e:
            verbose_proxy_logger.warning("Invalid %s value: %s, error: %s", setting_name, retention_setting, e)
            return None
        verbose_proxy_logger.info("%s set to %s seconds", setting_name, retention_seconds)
        return retention_seconds

    def _should_delete_spend_logs(self) -> bool:
        """
        Determines if logs should be deleted based on the max retention period in settings.
        """
        self.retention_seconds = self._retention_seconds_for("maximum_spend_logs_retention_period")
        return self.retention_seconds is not None

    async def _execute_delete_batch(
        self, prisma_client: PrismaClient, delete_sql: str, cutoff_date: datetime
    ) -> int | None:
        """
        Run one delete batch under a Postgres statement and lock timeout.

        The timeouts are what actually bound the work: a Prisma transaction
        timeout cannot interrupt a statement that is already executing, so
        without these a single batch blocked behind a lock would hold its
        connection, and the row locks it already took, indefinitely. SET LOCAL
        scopes both to this transaction so the pooled connection is unaffected.

        Returns the row count, or None when the driver returned something that
        is not a row count. That is a contract violation rather than a transient
        fault, so the caller stops instead of retrying.
        """
        # An int cannot carry SQL, and SET does not accept a bind parameter.
        timeout_ms: Final = max(1, int(self.batch_timeout_seconds * 1000))
        async with prisma_client.db.tx() as tx:
            await tx.execute_raw(f"SET LOCAL statement_timeout = {timeout_ms}")
            await tx.execute_raw(f"SET LOCAL lock_timeout = {timeout_ms}")
            deleted_result: Final = await tx.execute_raw(delete_sql, cutoff_date, self.batch_size)
        return deleted_result if isinstance(deleted_result, int) else None

    async def _count_remaining(
        self, prisma_client: PrismaClient, cutoff_date: datetime, table_name: str, time_column: str
    ) -> int | None:
        """
        Count expired rows still outstanding, stopping at a cap.

        An uncapped COUNT(*) over an expired backlog would itself be the kind of
        long scan this job exists to avoid, so the probe reads at most
        SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP index entries. A result equal to
        the cap means "at least this many".
        """
        count_sql: Final = f"""
            SELECT count(*)::int AS remaining FROM (
                SELECT 1 FROM "{table_name}"
                WHERE "{time_column}" < $1::timestamptz
                LIMIT $2
            ) capped
            """
        try:
            rows: Final = _REMAINING_ROWS.validate_python(
                await prisma_client.db.query_raw(count_sql, cutoff_date, SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP)
            )
        except Exception as e:  # noqa: BLE001 - an observability probe must never fail the cleanup run
            verbose_proxy_logger.warning("Could not count remaining %s rows: %s", table_name, e)
            return None
        return rows[0].remaining if rows else None

    async def _delete_old_rows_batched(
        self,
        prisma_client: PrismaClient,
        cutoff_date: datetime,
        table_name: str,
        key_columns: tuple[str, ...],
        time_column: str,
        deadline: float,
    ) -> TableCleanupResult:
        """
        Delete a table's rows older than the cutoff in batches.

        Stops at whichever bound is reached first: the backlog running out, the
        shared wall-clock deadline, the per-table batch cap, or too many
        consecutive batch failures.
        """
        key_list: Final = ", ".join(f'"{col}"' for col in key_columns)
        delete_sql: Final = f"""
            DELETE FROM "{table_name}"
            WHERE ({key_list}) IN (
                SELECT {key_list} FROM "{table_name}"
                WHERE "{time_column}" < $1::timestamptz
                LIMIT $2
            )
            """
        total_deleted = 0
        run_count = 0
        consecutive_failures = 0
        while True:
            if time.monotonic() >= deadline:
                verbose_proxy_logger.info(
                    "Run budget exhausted during %s cleanup after %d rows; the next run resumes from here",
                    table_name,
                    total_deleted,
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "budget_exhausted"
                )
            if run_count >= self.max_batches:
                verbose_proxy_logger.info(
                    "Max batches reached for %s cleanup, remaining rows will be deleted in next run", table_name
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "batch_cap_reached"
                )
            # Find rows and delete them in one go without fetching to application
            batch_started_at = time.monotonic()
            try:
                batch_result = await self._execute_delete_batch(prisma_client, delete_sql, cutoff_date)
            except Exception as batch_exc:
                # A single batch failure (e.g. Prisma/DB timeout) must not abort
                # the whole run — subsequent batches may still succeed.
                consecutive_failures += 1
                SpendLogCleanupMetrics.record_batch_failure(table_name)
                verbose_proxy_logger.exception(
                    "%s cleanup batch failed "
                    "(run_count=%d, consecutive_failures=%d, batch_size=%d, "
                    "cutoff=%s, total_deleted_so_far=%d): %s: %s",
                    table_name,
                    run_count,
                    consecutive_failures,
                    self.batch_size,
                    cutoff_date.isoformat(),
                    total_deleted,
                    type(batch_exc).__name__,
                    batch_exc,
                )
                if consecutive_failures >= SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES:
                    verbose_proxy_logger.error(
                        "Aborting %s cleanup after %d consecutive batch failures; total deleted before abort: %d",
                        table_name,
                        consecutive_failures,
                        total_deleted,
                    )
                    return await self._finish_table(
                        prisma_client, cutoff_date, table_name, time_column, total_deleted, "aborted"
                    )
                await asyncio.sleep(SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS)
                continue

            if batch_result is None:
                verbose_proxy_logger.error(
                    "Unexpected execute_raw return type for %s cleanup; aborting cleanup to avoid infinite loop",
                    table_name,
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "aborted"
                )

            consecutive_failures = 0
            deleted_count = batch_result
            SpendLogCleanupMetrics.record_batch(table_name, deleted_count, time.monotonic() - batch_started_at)
            verbose_proxy_logger.info("Deleted %s %s rows in this batch", deleted_count, table_name)

            if deleted_count == 0:
                verbose_proxy_logger.info("No more %s rows to delete. Total deleted: %s", table_name, total_deleted)
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "exhausted"
                )

            total_deleted += deleted_count
            run_count += 1

            # Add a small sleep to prevent overwhelming the database
            await asyncio.sleep(0.1)

    async def _finish_table(
        self,
        prisma_client: PrismaClient,
        cutoff_date: datetime,
        table_name: str,
        time_column: str,
        rows_deleted: int,
        stop_reason: StopReason,
    ) -> TableCleanupResult:
        """
        Publish how much of this table is still outstanding, then report the run's result.
        """
        remaining: Final = await self._count_remaining(prisma_client, cutoff_date, table_name, time_column)
        if remaining is not None:
            SpendLogCleanupMetrics.set_rows_remaining(table_name, remaining)
        return TableCleanupResult(rows_deleted=rows_deleted, stop_reason=stop_reason)

    async def _delete_old_logs(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogs",
            key_columns=("request_id", "startTime"),
            time_column="startTime",
            deadline=deadline,
        )

    async def _delete_old_tool_index_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        # SpendLogToolIndex rows are derived from spend logs, so they expire on the
        # same cutoff; rows older than retention point at already-deleted logs.
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogToolIndex",
            key_columns=("request_id", "tool_name"),
            time_column="start_time",
            deadline=deadline,
        )

    async def _delete_old_autorouter_session_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_AutoRouterSession",
            key_columns=("api_key", "session_id", "router_name"),
            time_column="last_turn_at",
            deadline=deadline,
        )

    async def _clean_spend_log_tables(
        self, prisma_client: PrismaClient, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune the spend logs and the tool index rows derived from them.

        When the table is range-partitioned, whole expired partitions are dropped
        first because that reclaims disk immediately. Expired rows can still sit in
        the DEFAULT partition (backfill, coverage gaps) or in a partition that spans
        the cutoff, so retention still deletes those stragglers row-wise.
        """
        cutoff_date: Final = datetime.now(timezone.utc) - timedelta(seconds=float(self.retention_seconds or 0))
        verbose_proxy_logger.info("Removing logs older than %s", cutoff_date.isoformat())

        if self.general_settings.get(
            "use_spend_logs_partitioning", False
        ) and await self.partition_manager.is_partitioned(prisma_client):
            await self.partition_manager.ensure_partitions(prisma_client)
            dropped: Final = await self.partition_manager.drop_partitions_older_than(prisma_client, cutoff_date)
            verbose_proxy_logger.info("Dropped %d expired spend-log partitions: %s", len(dropped), dropped)

        logs_result: Final = await self._delete_old_logs(prisma_client, cutoff_date, deadline)
        verbose_proxy_logger.info("Deleted %s logs", logs_result.rows_deleted)

        index_result: Final = await self._delete_old_tool_index_rows(prisma_client, cutoff_date, deadline)
        verbose_proxy_logger.info("Deleted %s expired tool index rows", index_result.rows_deleted)
        return (logs_result, index_result)

    async def _clean_session_rollup(
        self, prisma_client: PrismaClient, retention_seconds: int, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune auto-router session rollup rows, which carry their own retention horizon.
        """
        session_cutoff: Final = datetime.now(timezone.utc) - timedelta(seconds=float(retention_seconds))
        sessions_result: Final = await self._delete_old_autorouter_session_rows(prisma_client, session_cutoff, deadline)
        verbose_proxy_logger.info("Deleted %s expired auto-router session rollup rows", sessions_result.rows_deleted)
        return (sessions_result,)

    @staticmethod
    def _run_outcome(results: tuple[TableCleanupResult, ...]) -> RunOutcome:
        """
        Report the most operationally significant reason the run stopped.

        A bound that was hit matters more than a table that simply ran dry, so
        those win over "completed", and an abort wins over everything.
        """
        reasons: Final = frozenset(result.stop_reason for result in results)
        if "aborted" in reasons:
            return "aborted"
        if "budget_exhausted" in reasons:
            return "budget_exhausted"
        if "batch_cap_reached" in reasons:
            return "batch_cap_reached"
        return "completed"

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        If pod_lock_manager is available, ensures only one pod runs cleanup.
        If no pod_lock_manager, runs cleanup without distributed locking.
        """
        lock_acquired = False
        try:
            verbose_proxy_logger.info("Cleanup job triggered at %s", datetime.now())

            delete_spend_logs: Final = self._should_delete_spend_logs()
            autorouter_retention_seconds: Final = self._retention_seconds_for(
                "maximum_autorouter_session_retention_period"
            )
            if not delete_spend_logs and autorouter_retention_seconds is None:
                SpendLogCleanupMetrics.record_run("skipped_disabled")
                return

            if delete_spend_logs and self.retention_seconds is None:
                verbose_proxy_logger.error("Retention seconds is None, cannot proceed with cleanup")
                SpendLogCleanupMetrics.record_run("skipped_disabled")
                return

            # If we have a pod lock manager, try to acquire the lock
            if self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                lock_acquired = (
                    await self.pod_lock_manager.acquire_lock(
                        cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME,
                    )
                    or False
                )
                verbose_proxy_logger.info(
                    "Lock acquisition attempt: %s  at %s", "successful" if lock_acquired else "failed", datetime.now()
                )

                if not lock_acquired:
                    verbose_proxy_logger.info("Another pod is already running cleanup")
                    SpendLogCleanupMetrics.record_run("skipped_locked")
                    return

            deadline: Final = time.monotonic() + self.run_budget_seconds

            spend_log_results: Final = (
                await self._clean_spend_log_tables(prisma_client, deadline)
                if delete_spend_logs and self.retention_seconds is not None
                else ()
            )
            session_results: Final = (
                await self._clean_session_rollup(prisma_client, autorouter_retention_seconds, deadline)
                if autorouter_retention_seconds is not None
                else ()
            )

            SpendLogCleanupMetrics.record_run(self._run_outcome(spend_log_results + session_results))

        except Exception as e:
            # .exception() captures the traceback; str(e) alone on a Prisma/DB
            # timeout is often empty and gives operators no signal to diagnose.
            verbose_proxy_logger.exception(
                "Error during spend log cleanup: %s: %s",
                type(e).__name__,
                e,
            )
            SpendLogCleanupMetrics.record_run("aborted")
            return  # Return after error handling
        finally:
            # Only release the lock if it was actually acquired
            if lock_acquired and self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info("Released cleanup lock")
