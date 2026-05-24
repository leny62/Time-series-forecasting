from datetime import date
from pathlib import Path

from mtraffic.io.paths import date_from_filename, partition_path


def test_date_from_filename_parses() -> None:
    d = date_from_filename(Path("sms-call-internet-mi-2013-11-01.txt"))
    assert d == date(2013, 11, 1)


def test_date_from_filename_rejects_other() -> None:
    assert date_from_filename(Path("notes.txt")) is None
    assert date_from_filename(Path("sms-call-internet-mi-2013-13-01.txt")) is None


def test_partition_path_layout(tmp_path: Path) -> None:
    p = partition_path(tmp_path, date(2013, 12, 16))
    assert p.parent.name == "year_month=2013-12"
    assert p.name == "day=2013-12-16.parquet"
    assert p.suffix == ".parquet"
