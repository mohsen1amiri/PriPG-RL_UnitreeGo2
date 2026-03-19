import os
import sys
from pathlib import Path

# ---- choose portable root (must be set BEFORE SimulationApp starts) ----
il_root = os.environ.get("IL_ROOT", "")
portable_root = os.environ.get("ISAAC_PORTABLE_ROOT", "")

if not portable_root:
    if il_root:
        portable_root = str(Path(il_root) / "kit_portable")
    else:
        portable_root = str(Path.cwd() / "kit_portable")

portable_root = str(Path(portable_root).expanduser().resolve())

# Keep Kit/Omniverse from touching $HOME for global config discovery.
# Kit looks for omniverse.toml under OMNI_CONFIG_PATH if set. :contentReference[oaicite:3]{index=3}
omni_config_path = str(Path(portable_root) / "omni_config")
os.environ["OMNI_CONFIG_PATH"] = omni_config_path

# Optional: also redirect XDG roots away from $HOME (Omniverse defaults rely on XDG_*). :contentReference[oaicite:4]{index=4}
os.environ.setdefault("XDG_CACHE_HOME", str(Path(portable_root) / "xdg_cache"))
os.environ.setdefault("XDG_DATA_HOME",  str(Path(portable_root) / "xdg_data"))
os.environ.setdefault("XDG_CONFIG_HOME",str(Path(portable_root) / "xdg_config"))

Path(omni_config_path).mkdir(parents=True, exist_ok=True)

# Create omniverse.toml so any non-portable/global tokens also point to portable_root. :contentReference[oaicite:5]{index=5}
(Path(omni_config_path) / "omniverse.toml").write_text(
    "[paths]\n"
    f"data_root = \"{portable_root}/ov_data\"\n"
    f"cache_root = \"{portable_root}/ov_cache\"\n"
    f"logs_root = \"{portable_root}/ov_logs\"\n"
)

print("===VERIFY: INPUTS===")
print("portable_root =", portable_root)
print("OMNI_CONFIG_PATH =", os.environ["OMNI_CONFIG_PATH"])
print("XDG_CACHE_HOME =", os.environ["XDG_CACHE_HOME"])
print("XDG_DATA_HOME  =", os.environ["XDG_DATA_HOME"])
print("XDG_CONFIG_HOME=", os.environ["XDG_CONFIG_HOME"])
print()

# ---- START KIT (MUST HAPPEN BEFORE importing omni/isaaclab that depends on Kit) ----
# Omniverse plugins can't be imported unless the Toolkit (Kit) is running. :contentReference[oaicite:6]{index=6}
from isaacsim.simulation_app import SimulationApp

app = SimulationApp({
    "headless": True,
    "extra_args": [
        "--portable-root", portable_root,   # relocates logs/data/cache under portable root :contentReference[oaicite:7]{index=7}
        "-v",                               # INFO logs help spot token/caching messages
    ],
})

try:
    import carb.tokens
    import carb.settings

    resolve = carb.tokens.get_tokens_interface().resolve
    settings = carb.settings.get_settings()

    print("===VERIFY: KIT TOKENS / PATHS===")
    token_vals = {
        "${kit}": resolve("${kit}"),
        "${cache}": resolve("${cache}"),
        "${data}": resolve("${data}"),
        "${logs}": resolve("${logs}"),
        "${omni_cache}": resolve("${omni_cache}"),
        "${omni_data}": resolve("${omni_data}"),
        "${omni_logs}": resolve("${omni_logs}"),
        "${omni_config}": resolve("${omni_config}"),
    }
    for k, v in token_vals.items():
        print(f"{k:14s} -> {v}")

    # Heuristic: fail if any important resolved path is still under /isaac-sim/kit/cache
    bad = [v for v in token_vals.values() if "/isaac-sim/kit/cache" in v]
    print()
    print("===VERIFY: CACHE ROOT CHECK===")
    if bad:
        print("FAIL: Found paths still pointing into /isaac-sim/kit/cache:")
        for v in bad:
            print("  ", v)
        sys.exit(2)
    else:
        print("OK: No resolved token path points to /isaac-sim/kit/cache")

    # Quick writability test
    print()
    print("===VERIFY: WRITABLE TEST===")
    testfile = Path(portable_root) / "_write_test.txt"
    testfile.parent.mkdir(parents=True, exist_ok=True)
    testfile.write_text("ok\n")
    print("OK: wrote", str(testfile))

    # ---- import IsaacLab AFTER Kit is up ----
    print()
    print("===VERIFY: ISAACLAB IMPORT===")
    import isaaclab
    print("isaaclab imported OK")

    # ---- gym registry: importing isaaclab_tasks registers envs :contentReference[oaicite:8]{index=8} ----
    print()
    print("===VERIFY: GYM ENV REGISTRY (GO2) ===")
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 (this triggers environment registration) :contentReference[oaicite:9]{index=9}

    reg = gym.envs.registry
    ids = sorted(list(reg.keys())) if hasattr(reg, "keys") else sorted([spec.id for spec in reg.values()])

    go2_ids = [i for i in ids if "go2" in i.lower()]
    vel_go2_ids = [i for i in go2_ids if "velocity" in i.lower()]

    print(f"Total envs in registry: {len(ids)}")
    print(f"Envs containing 'go2'   : {len(go2_ids)}")
    print(f"Go2 velocity envs       : {len(vel_go2_ids)}")
    print("Go2 velocity env IDs (up to 50):")
    for i in vel_go2_ids[:50]:
        print("  ", i)

    # Also print any Go2 env IDs (up to 50), helpful if naming differs
    if not vel_go2_ids and go2_ids:
        print()
        print("Other Go2 env IDs (up to 50):")
        for i in go2_ids[:50]:
            print("  ", i)

    print()
    print("===ALL CHECKS PASSED===")

finally:
    # Always close Kit
    app.close()
