"""Smoke tests for the scripts in examples/.

Each example runs in a fresh interpreter from an unrelated working directory,
the way a reader would run it, and its output must contain the result the
example exists to demonstrate.
"""

import os
import shutil
import subprocess
import sys

import pytest

_EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")
)

# Script -> text its stdout must contain.
_EXPECTED_OUTPUT = {
    "basic_usage.py": [
        "Decoded: CC(=O)Oc1ccccc1C(=O)O",
    ],
    "batch_processing.py": [
        "Call interface",
        "Speedup:",
    ],
    "train_tokenizer.py": [
        "Token IDs identical after save()/from_file(): True",
    ],
    "compare_tokenizers.py": [
        "AtomBPETokenizer is SmilesTokenizer: True",
        "ByteBPETokenizer never emits <unk>: True",
    ],
    "persistence_and_interop.py": [
        "Token IDs identical after save()/from_file(): True",
        "Pickle round trip identical: True",
        "Multiprocessing results identical: True",
        "Wrote HuggingFace tokenizer.json",
    ],
}


def _run(script_path, cwd, *args):
    return subprocess.run(
        [sys.executable, script_path, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_every_example_is_smoke_tested():
    scripts = sorted(f for f in os.listdir(_EXAMPLES_DIR) if f.endswith(".py"))
    assert scripts == sorted(_EXPECTED_OUTPUT)


@pytest.mark.parametrize("script", sorted(_EXPECTED_OUTPUT))
def test_example_runs_from_any_directory(script, tmp_path):
    result = _run(os.path.join(_EXAMPLES_DIR, script), tmp_path)

    assert result.returncode == 0, result.stderr
    for marker in _EXPECTED_OUTPUT[script]:
        assert marker in result.stdout, f"{marker!r} missing from {script} output"


@pytest.mark.parametrize("script", ["basic_usage.py", "batch_processing.py"])
def test_missing_vocabulary_gives_clear_message(script, tmp_path):
    """A copied example without the repository's data/ explains what to do."""
    copied = tmp_path / script
    shutil.copy(os.path.join(_EXAMPLES_DIR, script), copied)

    result = _run(str(copied), tmp_path)

    assert result.returncode == 1
    assert "chembl36_vocab.txt" in result.stderr
    assert "https://github.com/HFooladi/rustmolbpe" in result.stderr
    assert "Traceback" not in result.stderr
