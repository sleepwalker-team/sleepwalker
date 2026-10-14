import asyncio
from collections import deque
from threading import Lock

from ..stores.job_store import Job


class JobSubscription:
    """Bounded mailbox between job worker threads and one async consumer."""

    def __init__(self, initial: Job) -> None:
        self._loop = asyncio.get_running_loop()
        self._initial: Job | None = initial
        self._updates: deque[Job | None] = deque(maxlen=64)
        self._wakeup = asyncio.Event()
        self._lock = Lock()
        self._notification_pending = False
        self._closed = False

    def publish(self, job: Job | None) -> None:
        with self._lock:
            if self._closed:
                return

            self._updates.append(job)
            if self._notification_pending:
                return
            self._notification_pending = True

        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            # The consumer's event loop has already shut down.
            self.close()

    async def get(self) -> Job | None:
        while True:
            with self._lock:
                if self._initial is not None:
                    initial = self._initial
                    self._initial = None
                    return initial
                if self._updates:
                    return self._updates.popleft()
                if self._closed:
                    return None

                self._wakeup.clear()
                self._notification_pending = False

            await self._wakeup.wait()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._initial = None
            self._updates.clear()

        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            pass
