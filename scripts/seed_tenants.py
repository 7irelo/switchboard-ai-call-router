"""Provisions the five fictional tenants through the admin API. Prints each tenant's key to a local gitignored file."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx2 as httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    api = os.environ.get("SWITCHBOARD_API", "http://127.0.0.1:8080")
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#"))
    admin = os.environ.get("SWITCHBOARD_ADMIN_KEY", env.get("SWITCHBOARD_ADMIN_KEY", ""))
    keys_path = ROOT / "runs/tenant_keys.json"
    keys = json.loads(keys_path.read_text(encoding="utf-8")) if keys_path.exists() else {}
    for path in sorted((ROOT / "configs/tenants").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        response = httpx.post(f"{api}/admin/tenants", json=config, headers={"Authorization": f"Bearer {admin}"}, timeout=30)
        if response.status_code == 409:
            print(f"{config['tenant']}: already provisioned")
            continue
        if response.status_code != 200:
            print(f"{config['tenant']}: FAILED {response.status_code} {response.text}")
            return 1
        keys[config["tenant"]] = response.json()["api_key"]
        print(f"{config['tenant']}: provisioned at version {response.json()['version']}")
    keys_path.parent.mkdir(exist_ok=True)
    keys_path.write_text(json.dumps(keys, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
