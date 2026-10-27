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
    # Pass the resolved node to workers even when it came from JSON: CLI policy
    # overrides must reach both bootstrap and executors. Never rewrite the user's
    # source config. Keep its rendezvous path for explicitly configured clients.
    with tempfile.TemporaryDirectory(prefix="sglang-expertplex-") as directory:
        if not cfg.expertplex_config:
            # HTTP clients do not need a shared path. Each pool incarnation
            # gets a private directory and fresh IPC allocations/descriptors.
            node = replace(node, rendezvous_dir=directory)
        config_path = Path(directory) / "node.json"
        atomic_json(config_path, node.to_dict())
        supervise(str(config_path), bootstrap_host=cfg.host, bootstrap_port=cfg.port)
