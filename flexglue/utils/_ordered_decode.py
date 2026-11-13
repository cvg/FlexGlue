"""Ordered bounded CPU loading, no pixel/model policy.

Uses two CPU workers and at most two pending reads. A caller retaining its
previous item while advancing can transiently keep four decoded items alive.
Close the context on early exit so pending work is cancelled and workers joined.
"""
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def ordered_decode(rows, loader, *, enabled, stats=None):
    """Yield (row, loaded) in order; loader alone executes in CPU threads.

    Failures surface at their original ordinal. Already-running successor
    reads may finish during shutdown; no new work is submitted after failure.
    Worker startup, initial reads and join occur inside this context.
    """
    stats = stats if stats is not None else {}
    stats.update(
        submitted=0,
        consumed=0,
        maxPending=0,
        consumerWaitSeconds=0.0,
        joinSeconds=0.0,
        workersJoined=False,
    )
    if not enabled:

        def read_row(row):
            stats["submitted"] += 1
            value = loader(row)
            stats["consumed"] += 1
            return row, value

        def serial():
            for row in rows:
                yield read_row(row)

        iterator = serial()
        try:
            yield iterator
        finally:
            iterator.close()
            stats["workersJoined"] = True
        return
    executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="flex-ordered-decode"
    )
    pending = deque()
    source = iter(rows)

    def submit():
        nonlocal source
        if source is None:
            return
        try:
            row = next(source)
        except StopIteration:
            source = None
            return
        except Exception as error:
            source = None
            row, future = None, Future()
            # Report source failures through the same ordered result path.
            future.set_exception(error)
        else:
            future = executor.submit(loader, row)
            stats["submitted"] += 1
        pending.append((row, future))
        stats["maxPending"] = max(stats["maxPending"], len(pending))
        if len(pending) > 2:
            raise RuntimeError("prefetch bound exceeded")

    def receive():
        row, future = pending.popleft()
        waited = time.perf_counter()
        value = future.result()
        stats["consumerWaitSeconds"] += time.perf_counter() - waited
        del future
        stats["consumed"] += 1
        submit()
        return row, value

    def ordered():
        submit()
        submit()
        while pending:
            yield receive()

    iterator = ordered()
    try:
        yield iterator
    finally:
        iterator.close()
        for _, future in pending:
            future.cancel()
        started = time.perf_counter()
        executor.shutdown(wait=True, cancel_futures=True)
        stats["joinSeconds"] = time.perf_counter() - started
        pending.clear()
        stats["workersJoined"] = True
