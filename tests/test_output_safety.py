"""Repeated runs must not overwrite an existing evaluation or checkpoint."""
from pathlib import Path
import subprocess
import sys


def test_cli_preserves_existing_run_outputs(tmp_path):
    output = tmp_path / 'existing-run'
    output.mkdir()
    evaluation = output / 'evaluation.json'
    evaluation.write_text('{"fixture": "previous run"}\n')
    original = evaluation.read_bytes()
    script = Path(__file__).resolve().parents[1] / 'src/pipeline.py'
    result = subprocess.run([sys.executable, str(script), '--epochs', '1', '--output-dir', str(output)], capture_output=True, text=True, timeout=40, cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert evaluation.read_bytes() == original
    assert set(p.name for p in output.iterdir()) == {'evaluation.json'}


def test_existing_destinations_rejected_before_data_loading(tmp_path):
    from types import SimpleNamespace
    import pytest
    from src.training import run

    directory = tmp_path / 'empty'
    directory.mkdir()
    file = tmp_path / 'file'
    file.write_text('previous artifact')
    link = tmp_path / 'link'
    link.symlink_to(tmp_path / 'missing-target')
    for destination in (directory, file, link):
        args = SimpleNamespace(epochs=1, threads=1, output_dir=destination)
        with pytest.raises(ValueError, match='Output destination already exists'):
            run(args, enhanced=False)
