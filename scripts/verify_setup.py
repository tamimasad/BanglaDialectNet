"""
Environment verification for BanglaDialectNet.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys


def check(label, condition, detail=""):
    """Print PASS/FAIL for one check and return whether it passed."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    all_ok = True

    # 1. Python version must be 3.10+
    v = sys.version_info
    all_ok &= check(
        "Python >= 3.10", v >= (3, 10), f"found {v.major}.{v.minor}.{v.micro}"
    )

    # 2. Confirming running inside the venv, not system Python
    all_ok &= check("Running inside virtual environment", sys.prefix != sys.base_prefix)

    # 3. Cache env vars must point to E:, not the default C: user folder
    for var in ["PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME"]:
        value = os.environ.get(var, "")
        all_ok &= check(
            f"{var} points to E:", value.upper().startswith("E:"), value or "NOT SET"
        )

    # 4. PyTorch + GPU
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "none"
        all_ok &= check("PyTorch installed", True, f"v{torch.__version__}")
        all_ok &= check("CUDA available to PyTorch", cuda_ok, f"GPU: {gpu}")
    except ImportError:
        all_ok &= check("PyTorch installed", False, "missing")

    # 5. Core NLP libraries
    for pkg in [
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "evaluate",
        "huggingface_hub",
    ]:
        try:
            mod = __import__(pkg)
            all_ok &= check(
                f"{pkg} installed", True, f"v{getattr(mod, '__version__', '?')}"
            )
        except ImportError:
            all_ok &= check(f"{pkg} installed", False, "missing")

    # 6. Real functional test: download a tiny model and confirm it

    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        hf_home = os.environ.get("HF_HOME", "")
        landed_on_e = os.path.isdir(hf_home) and len(os.listdir(hf_home)) > 0
        all_ok &= check(
            "Hugging Face download landed in E: cache", landed_on_e, hf_home
        )
    except Exception as e:
        all_ok &= check("Hugging Face download test", False, repr(e))

    # 7. Git repo sanity
    all_ok &= check(".git folder exists", os.path.isdir(".git"))
    all_ok &= check("requirements.txt exists", os.path.isfile("requirements.txt"))

    print()
    if all_ok:
        print("ALL CHECKS PASSED -- ready for Phase 1.")
    else:
        print("SOME CHECKS FAILED -- fix the FAIL lines above before continuing.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
