"""Malformed market data fails at ingestion with actionable CLI errors."""
from pathlib import Path
import subprocess
import sys

import pytest

from src.training import load_stock


@pytest.mark.parametrize('text,expected', [
    ('Name,Date,Close\n01.Bank,01-01-2022,10\n', 'Missing required columns: Volume'),
    ('Name,Date,Close,Volume\n01.Bank,,10,100\n', 'Stock dates must not be missing'),
])
def test_invalid_csv_cli_is_actionable_and_writes_no_artifacts(tmp_path, text, expected):
    data = tmp_path/'invalid.csv'
    data.write_text(text)
    output = tmp_path/'artifacts'
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'src/pipeline.py'), '--data', str(data), '--output-dir', str(output), '--epochs', '1'], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert expected in result.stderr
    assert 'Traceback' not in result.stderr
    assert not output.exists()


def test_load_stock_requires_all_columns_before_filtering(tmp_path):
    data = tmp_path/'invalid.csv'
    data.write_text('Date,Close\n01-01-2022,10\n')
    with pytest.raises(ValueError, match='Missing required columns: Name, Volume'):
        load_stock(data, '01.Bank')
