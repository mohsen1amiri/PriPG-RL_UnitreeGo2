cat > "$IL_ROOT/patch_rsl_rl_disable_git.py" <<'PY'
from pathlib import Path
import rsl_rl

pkg = Path(rsl_rl.__file__).resolve().parent
logger_py = pkg / "utils" / "logger.py"

text = logger_py.read_text()

if "PATCHED_DISABLE_GITPYTHON" in text:
    print("Already patched:", logger_py)
    raise SystemExit(0)

if "import git" not in text:
    print("Did not find 'import git' in:", logger_py)
    raise SystemExit(1)

patch = """# PATCHED_DISABLE_GITPYTHON
try:
    import git  # GitPython (optional)
except Exception:
    git = None
"""

# replace ONLY first occurrence
text = text.replace("import git", patch, 1)

# add a very safe fallback: if later code uses git.Repo, it won't crash
if "git.Repo" in text and "if git is None" not in text:
    text = text.replace(
        "git.Repo",
        "(git.Repo if git is not None else (lambda *a, **k: None))",
    )

logger_py.write_text(text)
print("Patched:", logger_py)
PY