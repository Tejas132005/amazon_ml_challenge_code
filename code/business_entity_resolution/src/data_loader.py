"""Strict, memory-efficient TSV loading with PyArrow.

Design choices
--------------
* PyArrow's multithreaded CSV reader parses into compact Arrow string columns
  (contiguous UTF-8 buffer + int32 offsets, roughly 1.1x the raw bytes) instead
  of one Python ``str`` object per cell (~50+ bytes overhead each in pandas).
* ``quote_char=False``: the files are TSV, so quotes are literal characters.
  Treating them as quotes could silently merge rows if a name contains ``"``.
* No NA inference: empty cells stay ``""``, and strings like ``NA``/``null``
  (valid business names or country codes) are never turned into missing values.
* ``ignore_empty_lines=False`` + an invalid-row handler + a raw newline count
  guarantee that parsed rows == physical data lines, so nothing is dropped silently.

Memory: one full source file ~= file size x ~1.1 in RAM (e.g. a 700 MB file ~ 0.8 GB).
Callers load one file at a time and pass ``columns`` to read only what they need.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .config import CFG, GT_COLUMNS, SOURCE_COLUMNS, SOURCE_PREFIXES, Config

LOG = logging.getLogger("ber.data_loader")


class DataFormatError(ValueError):
    """Raised when a file does not match the expected TSV format or schema."""


@dataclass
class ReadReport:
    path: str
    rows: int
    data_lines: int
    invalid_rows: int = 0
    invalid_examples: list[str] = field(default_factory=list)


def read_header(path: Path) -> list[str]:
    """Return header fields; raise DataFormatError if the file is not tab-separated."""
    with open(path, "rb") as fh:
        first = fh.readline()
    line = first.decode("utf-8-sig").rstrip("\r\n")
    if "\t" not in line:
        hint = " It looks comma-separated." if "," in line else ""
        raise DataFormatError(f"{path}: header contains no tab character.{hint} Header={line!r}")
    return line.split("\t")


def count_data_lines(path: Path, chunk_size: int = 16 * 2**20) -> int:
    """Count physical lines after the header by streaming bytes (O(file) time, O(1) memory)."""
    newlines, last = 0, b""
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    total_lines = newlines + (1 if last and last != b"\n" else 0)
    return max(total_lines - 1, 0)


def read_tsv(path: Path, expected_columns: Sequence[str], columns: Sequence[str] | None = None,
             expected_rows: int | None = None, cfg: Config = CFG) -> tuple[pa.Table, ReadReport]:
    """Read a TSV into an Arrow table of non-null UTF-8 strings.

    Args:
        path: file to read.
        expected_columns: exact header, in order.
        columns: subset of columns to materialise (default: all).
        expected_rows: known row count; mismatch raises when ``cfg.strict_row_count``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    header = read_header(path)
    if header != list(expected_columns):
        raise DataFormatError(f"{path}: header {header} != expected {list(expected_columns)}")
    columns = list(columns or expected_columns)
    unknown = set(columns) - set(expected_columns)
    if unknown:
        raise ValueError(f"Unknown columns requested: {sorted(unknown)}")

    invalid: list[str] = []

    def _on_invalid(row: pacsv.InvalidRow) -> str:
        invalid.append(f"line={row.number} expected={row.expected_columns} "
                       f"actual={row.actual_columns} text={row.text[:200]!r}")
        return "skip"

    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=cfg.arrow_block_size, encoding="utf8", use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False,
                                         newlines_in_values=False, ignore_empty_lines=False,
                                         invalid_row_handler=_on_invalid),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in expected_columns},
            include_columns=columns, strings_can_be_null=False, quoted_strings_can_be_null=False,
            null_values=[]),
    )
    report = ReadReport(str(path), table.num_rows, count_data_lines(path), len(invalid), invalid[:5])

    problems = []
    if report.invalid_rows:
        problems.append(f"{report.invalid_rows} malformed rows skipped, e.g. {report.invalid_examples}")
    if report.rows != report.data_lines:
        problems.append(f"parsed rows {report.rows} != physical data lines {report.data_lines}")
    if expected_rows is not None and report.rows != expected_rows:
        problems.append(f"parsed rows {report.rows} != expected {expected_rows}")
    if problems:
        msg = f"{path.name}: " + "; ".join(problems)
        if cfg.strict_row_count:
            raise DataFormatError(msg)
        LOG.warning(msg)
    LOG.info("Read %s: %d rows x %d cols (%.0f MB in Arrow)", path.name, table.num_rows,
             table.num_columns, table.nbytes / 2**20)
    return table, report


def load_source(split: str, source_no: int, columns: Sequence[str] | None = None,
                cfg: Config = CFG) -> tuple[pa.Table, ReadReport]:
    if source_no not in SOURCE_PREFIXES:
        raise ValueError(f"source_no must be one of {list(SOURCE_PREFIXES)}")
    path = cfg.source_path(split, source_no)
    return read_tsv(path, SOURCE_COLUMNS, columns, cfg.expected_rows.get(path.stem), cfg)


def load_ground_truth(split: str = "train", cfg: Config = CFG) -> tuple[pa.Table, ReadReport]:
    path = cfg.gt_path(split)
    return read_tsv(path, GT_COLUMNS, None, cfg.expected_rows.get(path.stem), cfg)


def id_format_mask(ids: pa.Array | pa.ChunkedArray, source_no: int) -> pa.ChunkedArray:
    """True where the ID looks like ``S{n}-<digits>``."""
    return pc.match_substring_regex(ids, pattern=f"^{SOURCE_PREFIXES[source_no]}-[0-9]+$")


def explode_ground_truth(gt: pa.Table) -> pa.Table:
    """One row per (source1 entity, matched id): columns ``s1_id``, ``tgt_id``, ``src`` (int8 2/3).

    Singletons contribute no rows. Raises on IDs that are not S2-/S3- formatted.
    """
    s1 = gt["source1_entity_id"].combine_chunks()
    lists = pc.split_pattern(gt["matched_entity_ids"].combine_chunks(), ",")
    parent = pc.list_parent_indices(lists)
    ids = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    keep = pc.not_equal(ids, "")
    ids, parent = ids.filter(keep), parent.filter(keep)
    is2, is3 = id_format_mask(ids, 2), id_format_mask(ids, 3)
    bad = pc.sum(pc.invert(pc.or_(is2, is3))).as_py() or 0
    if bad:
        raise DataFormatError(f"{bad} matched ids are not S2-/S3- formatted")
    src = pc.if_else(is2, pa.scalar(2, pa.int8()), pa.scalar(3, pa.int8()))
    t = pa.table({"s1_id": s1.take(parent), "tgt_id": ids, "src": src})
    return t.group_by(["s1_id", "tgt_id", "src"]).aggregate([]).select(["s1_id", "tgt_id", "src"])
