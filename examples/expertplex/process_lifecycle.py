"""Owned service shutdown shared by validation and serving launchers."""

SUPERVISOR_GRACE_SECONDS = 90
CHILD_GRACE_SECONDS = 10


def service_shutdown_budget(service_count):
    """Allow every supervisor and its bounded fallback to finish in order."""
    if type(service_count) is not int or service_count < 0:
        raise ValueError("Service count must be a nonnegative integer")
    return service_count * (SUPERVISOR_GRACE_SECONDS + 2 * CHILD_GRACE_SECONDS) + 10


def stop_owned_services(
    processes, grace_timeout=SUPERVISOR_GRACE_SECONDS, child_timeout=CHILD_GRACE_SECONDS
):
    """Stop clients before E and allow each supervisor to release its workers.

    Snapshot process identities before a parent can exit and orphan children.
    Descendant termination is a bounded fallback, after the supervisor's own
    stop protocol; killing E workers first bypasses executor.stop().
    """
    import psutil

    trees = []
    for proc in processes:
        try:
            trees.append((proc, proc.children(recursive=True)))
        except psutil.NoSuchProcess:
            trees.append((proc, []))
    remaining = []
    for proc, descendants in reversed(trees):
        try:
            proc.terminate()
            try:
                proc.wait(timeout=grace_timeout)
            except psutil.TimeoutExpired:
                proc.kill()
        except psutil.NoSuchProcess:
            pass
        owned = [proc, *descendants]
        for child in descendants:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=child_timeout)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=child_timeout)
        for child in alive:
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(child.pid)
            except psutil.NoSuchProcess:
                pass
    return remaining
