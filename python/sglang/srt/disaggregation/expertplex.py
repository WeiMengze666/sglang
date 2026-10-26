"""Independent ExpertPlex MoE service, reached through sglang.launch_server.

P/D are ordinary SGLang PD servers connecting to this service's bootstrap URL.
Only expert ranks are children of this process; attention servers and the
router have independent lifetimes and are never spawned here.
"""

import tempfile
from dataclasses import replace
from pathlib import Path


def launch_server(cfg):
    from sglang.srt.expertplex.integration import expert_node_config
    from sglang.srt.expertplex.rendezvous import atomic_json
    from sglang.srt.expertplex.worker import supervise

    node = expert_node_config(cfg)
    if cfg.expertplex_config:
        # Preserve explicitly configured rendezvous for existing file clients.
        # New attention servers still discover this pool through HTTP.
        supervise(
            cfg.expertplex_config, bootstrap_host=cfg.host, bootstrap_port=cfg.port
        )
        return
    # Every MoE process incarnation gets new IPC allocations and descriptors.
    # This directory is private to E; HTTP clients do not need a shared path.
    with tempfile.TemporaryDirectory(prefix="sglang-expertplex-") as directory:
        node = replace(node, rendezvous_dir=directory)
        config_path = Path(directory) / "node.json"
        atomic_json(config_path, node.to_dict())
        supervise(str(config_path), bootstrap_host=cfg.host, bootstrap_port=cfg.port)
