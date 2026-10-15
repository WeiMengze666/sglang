"""Local, generation-scoped exchange of CUDA IPC handles.

This is a startup control plane only. Activations, readiness, and completion
never pass through these files. A fresh run ID prevents importing stale IPC
allocations after an expert worker restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from .synthetic_weights import weight_identity

PROTOCOL_VERSION = 2


def config_digest(config) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_fingerprint(model_path: str) -> str:
    """Fingerprint configuration and shard identity without reading 960 GB.

    Checkpoint files are required to remain immutable for the deployment's
    lifetime. Size/mtime are startup guards, not cryptographic weight hashes.
    """
    root = Path(model_path).expanduser().resolve()
    digest = hashlib.sha256((root / "config.json").read_bytes())
    for path in sorted(root.glob("*.safetensors*")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def begin_run(config) -> dict:
    identity = weight_identity(config)
    root = Path(config.rendezvous_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # No automatic stale-lock takeover: two launchers must never export buffers
    # for the same slots. Use a fresh rendezvous directory after a crash.
    lock = root / "owner.lock"
    try:
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        os.chmod(lock, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"ExpertPlex rendezvous is already owned: {lock}. "
            "Choose a fresh rendezvous_dir for a new run."
        ) from exc
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "run_id": uuid.uuid4().hex,
        "config_digest": config_digest(config),
        "checkpoint_fingerprint": checkpoint_fingerprint(config.model_path),
        "weight_identity": identity,
        "pid": os.getpid(),
        "state": "starting",
    }
    atomic_json(root / "run.json", manifest)
    return manifest


def claim_client(config, manifest: dict, client_id: int) -> None:
    """A slot admits a single attention process for the whole run lifetime."""
    config.client(client_id)
    path = Path(config.rendezvous_dir) / f"client-{client_id}.lock"
    try:
        with path.open("x") as stream:
            os.chmod(path, 0o600)
            json.dump({"run_id": manifest["run_id"], "pid": os.getpid()}, stream)
    except FileExistsError as exc:
        raise RuntimeError(
            f"ExpertPlex attention slot {client_id} already has an owner. "
            "Restart the deployment with a fresh rendezvous_dir to replace a client."
        ) from exc


def wait_for_workers(config) -> tuple[dict, list[dict]]:
    root = Path(config.rendezvous_dir)
    deadline = time.monotonic() + config.startup_timeout_s
    last_reason = "no run manifest"
    expected_checkpoint = checkpoint_fingerprint(config.model_path)
    expected_weights = weight_identity(config)
    while time.monotonic() < deadline:
        try:
            manifest = json.loads((root / "run.json").read_text())
            if manifest["protocol"] != PROTOCOL_VERSION:
                raise RuntimeError("ExpertPlex IPC protocol version mismatch")
            if manifest["config_digest"] != config_digest(config):
                raise RuntimeError("ExpertPlex worker/client configuration mismatch")
            if manifest.get("weight_identity") != expected_weights:
                raise RuntimeError(
                    "ExpertPlex worker/client weight initialization mismatch"
                )
            if manifest["checkpoint_fingerprint"] != expected_checkpoint:
                raise RuntimeError("ExpertPlex checkpoint changed after worker startup")
            if not pid_alive(manifest["pid"]):
                raise RuntimeError("ExpertPlex worker supervisor has exited")
            if manifest["state"] in ("failed", "stopped"):
                raise RuntimeError(f"ExpertPlex workers are {manifest['state']}")
            peers = []
            for rank, device in enumerate(config.expert_devices):
                peer = json.loads((root / f"expert-{rank}.json").read_text())
                if peer["run_id"] != manifest["run_id"]:
                    raise FileNotFoundError("worker descriptor is from an old run")
                if peer["rank"] != rank or peer["device"] != device:
                    raise RuntimeError("ExpertPlex rank/device descriptor mismatch")
                if not pid_alive(peer["pid"]):
                    raise RuntimeError(f"ExpertPlex expert rank {rank} has exited")
                peers.append(peer)
            if manifest["state"] == "ready":
                return manifest, peers
            last_reason = "workers are loading weights or compiling CUDA"
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            last_reason = str(exc)
        time.sleep(0.1)
    raise TimeoutError(
        f"ExpertPlex workers did not become ready within "
        f"{config.startup_timeout_s}s: {last_reason}"
    )
