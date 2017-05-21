# -*- coding: utf-8 -*-
import time
from concurrent.futures import ProcessPoolExecutor


class IteratorExecutor(ProcessPoolExecutor):

    def __init__(self):
        super().__init__()
        # self._result_queue = mp.Queue(maxsize=1000)
        # self._result_queue = queue.Queue(maxsize=1000)
        # self._work_ids = queue.Queue(maxsize=10000)

    def map(self, fn, *iterables, timeout=None, chunksize=1, submit_chunk_size=1):
        """Returns an iterator equivalent to map(fn, iter).
        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if timeout is not None:
            end_time = timeout + time.time()

        # fs = (self.submit(fn, *args) for args in zip(*iterables))
        fs = self._submission_iterator(fn, iterables[0], submit_chunk_size=submit_chunk_size)

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                for future in fs:
                    if timeout is None:
                        yield future.result()
                    else:
                        yield future.result(end_time - time.time())
            finally:
                for future in fs:
                    future.cancel()

        # results = result_iterator()
        # return itertools.chain.from_iterable(results)
        yield from result_iterator()

    def _submission_iterator(self, fn, iterables, submit_chunk_size=1):
        submission_batch = []
        for args in iterables:
            if len(submission_batch) > submit_chunk_size:
                for f in submission_batch:
                    yield f
                submission_batch = []
            submission_batch.append(self.submit(fn, *args))

        for f in submission_batch:
            yield f
