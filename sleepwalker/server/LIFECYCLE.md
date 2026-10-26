# Temporary sessions and server storage

Sessions remain accessible while they have a queued, running or cancelling
job, even if their previous inactivity deadline has elapsed. Before a job
becomes succeeded, failed or cancelled, its session deadline is renewed for
the full TTL. This is an automatic deadline renewal; `last_accessed_at` still
records client access. Explicit session deletion is never undone by job
completion. Imports additionally renew their deadline during restoration.

On startup, the server exclusively locks `.server.lock` in its dedicated
temporary-storage root. A second server using that same directory refuses to
start. The lock is held for the server's lifespan and released by the OS if
the process exits. The lock file remains on disk; its existence alone does
not mean another server is active. This local V1 uses Unix `flock`.

While holding the lock, startup removes only unrecognized directories whose
names match the server's generated `sess_<UUID>` format. Directories already
represented in the injected session store are retained for normal expiry
handling. Symlinks and unrelated entries are excluded from orphan cleanup.
This reclaims uploads and exports left by an earlier process; in-memory
sessions themselves are not restored. Save a workspace ZIP to retain a study.

Periodic cleanup and startup file deletion run off the event loop. Server
tests isolate default storage in their test temporary directory so they
cannot claim or clean a running server's actual storage.
