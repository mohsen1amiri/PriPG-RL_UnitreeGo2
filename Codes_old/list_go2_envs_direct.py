import os
from pathlib import Path

portable_root = os.environ.get("PORTABLE_ROOT")
assert portable_root, "PORTABLE_ROOT env var not set"
portable_root = str(Path(portable_root).resolve())

# Keep everything under portable root (avoid HOME + read-only /isaac-sim/kit/cache)
os.environ["OMNI_CONFIG_PATH"] = str(Path(portable_root) / "omni_config")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(portable_root) / "xdg_cache"))
os.environ.setdefault("XDG_DATA_HOME",  str(Path(portable_root) / "xdg_data"))
os.environ.setdefault("XDG_CONFIG_HOME",str(Path(portable_root) / "xdg_config"))
Path(os.environ["OMNI_CONFIG_PATH"]).mkdir(parents=True, exist_ok=True)

# Start Kit first (required before omni/isaaclab pieces that depend on Kit)
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({
    "headless": True,
    "extra_args": [
        "--portable-root", portable_root,
        "--no-window",
    ],
})

try:
    import carb.tokens
    resolve = carb.tokens.get_tokens_interface().resolve
    print("TOKENS:")
    print("${kit}   ->", resolve("${kit}"))
    print("${cache} ->", resolve("${cache}"))
    print("${data}  ->", resolve("${data}"))
    print("${logs}  ->", resolve("${logs}"))
    print()

    print("IMPORTS:")
    import isaaclab
    print("isaaclab OK")

    # This triggers gym env registration
    import isaaclab_tasks  # noqa: F401
    print("isaaclab_tasks OK")
    print()

    import gymnasium as gym
    reg = gym.envs.registry

    # gymnasium registry differs by version; support both dict-like and spec objects
    if hasattr(reg, "keys"):
        ids = sorted(list(reg.keys()))
    else:
        ids = sorted([spec.id for spec in reg.values()])

    go2 = [i for i in ids if "go2" in i.lower()]
    print(f"Total envs: {len(ids)}")
    print(f"Go2 envs  : {len(go2)}")
    for i in go2:
        print(i)

finally:
    app.close()
