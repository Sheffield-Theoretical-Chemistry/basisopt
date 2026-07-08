"""Tests for BasisOptimizationLogger (opt/opt_logging.py).

The logger writes a per-composition CSV of energy + exponents during an
optimization. These lock the enable/disable flag, header/row schema,
composition-change flush, and resume-from-existing-file behaviour after the
switch away from the internal object-dtype ``.npy`` buffer.
"""

import csv
import glob
import os

from basisopt.opt.opt_logging import BasisOptimizationLogger
from tests.data.factories import make_basis


def _csv_files(log_dir):
    return sorted(glob.glob(os.path.join(log_dir, "*.csv")))


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_disabled_logger_writes_nothing(tmp_path):
    basis = make_basis("h", config=(("s", (1.0, 2.0)),))
    with BasisOptimizationLogger(
        basis=basis, element="h", strategy_name="test", log_dir=str(tmp_path), enabled=False
    ) as logger:
        logger.log(-1.0, basis, "h", cbs_limit=0.0)

    assert _csv_files(str(tmp_path)) == []
    # a disabled logger must not touch its counters either
    assert not hasattr(logger, "total_eval_counter")


def test_no_npy_files_created(tmp_path):
    # regression: the logger used to keep an object-dtype .npy loaded with
    # allow_pickle=True; it should now only ever write CSV.
    basis = make_basis("h", config=(("s", (1.0, 2.0)),))
    with BasisOptimizationLogger(
        basis=basis, element="h", strategy_name="test", log_dir=str(tmp_path), session_id="s1"
    ) as logger:
        logger.log(-1.0, basis, "h", cbs_limit=0.0)

    assert glob.glob(os.path.join(str(tmp_path), "*.npy")) == []
    assert len(_csv_files(str(tmp_path))) == 1


def test_header_and_rows(tmp_path):
    basis = make_basis("h", config=(("s", (5.0, 1.0)), ("p", (0.3,))))
    with BasisOptimizationLogger(
        basis=basis,
        element="h",
        strategy_name="mystrat",
        log_dir=str(tmp_path),
        session_id="s2",
    ) as logger:
        logger.log(-2.0, basis, "h", cbs_limit=-1.5)
        logger.log(-2.5, basis, "h", cbs_limit=-1.5)

    (path,) = _csv_files(str(tmp_path))
    rows = _read_csv(path)
    # header: bookkeeping columns + one column per exponent (2 s + 1 p)
    assert rows[0] == ["eval_num", "strategy", "energy", "dE_CBS", "s1", "s2", "p1"]
    assert len(rows) == 3  # header + 2 evaluations

    first = rows[1]
    assert first[0] == "1"
    assert first[1] == "mystrat"
    assert float(first[2]) == -2.0
    # dE_CBS = energy - cbs_limit
    assert abs(float(first[3]) - (-2.0 - -1.5)) < 1e-12
    # the three exponents follow
    assert [float(x) for x in first[4:]] == [5.0, 1.0, 0.3]
    assert rows[2][0] == "2"


def test_composition_change_flushes_and_opens_new_file(tmp_path):
    small = make_basis("h", config=(("s", (1.0, 2.0)),))
    grown = make_basis("h", config=(("s", (1.0, 2.0, 3.0)),))
    with BasisOptimizationLogger(
        basis=small, element="h", strategy_name="s", log_dir=str(tmp_path), session_id="s3"
    ) as logger:
        logger.log(-1.0, small, "h", cbs_limit=0.0)
        logger.log(-1.0, grown, "h", cbs_limit=0.0)  # composition change -> new file

    files = _csv_files(str(tmp_path))
    assert len(files) == 2
    # one file has the 2s composition, the other the 3s composition
    assert any("_2s_" in f for f in files)
    assert any("_3s_" in f for f in files)


def test_resume_continues_eval_counter(tmp_path):
    basis = make_basis("h", config=(("s", (1.0, 2.0)),))
    # first session writes two evaluations
    with BasisOptimizationLogger(
        basis=basis, element="h", strategy_name="s", log_dir=str(tmp_path), session_id="resume"
    ) as logger:
        logger.log(-1.0, basis, "h", cbs_limit=0.0)
        logger.log(-1.0, basis, "h", cbs_limit=0.0)

    # a second logger sharing the session_id (hence the same file) resumes
    with BasisOptimizationLogger(
        basis=basis, element="h", strategy_name="s", log_dir=str(tmp_path), session_id="resume"
    ) as logger:
        logger.log(-1.0, basis, "h", cbs_limit=0.0)

    (path,) = _csv_files(str(tmp_path))
    rows = _read_csv(path)
    # header + 3 data rows, eval_num counting 1, 2, 3 (not restarting at 1)
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]


def test_flush_interval(tmp_path):
    basis = make_basis("h", config=(("s", (1.0,)),))
    logger = BasisOptimizationLogger(
        basis=basis,
        element="h",
        strategy_name="s",
        log_dir=str(tmp_path),
        session_id="flush",
        flush_interval=2,
    )
    logger.log(-1.0, basis, "h", cbs_limit=0.0)
    # below the interval: buffered, not yet written
    (path,) = _csv_files(str(tmp_path))
    assert len(_read_csv(path)) == 1  # header only
    logger.log(-1.0, basis, "h", cbs_limit=0.0)
    # hitting the interval flushes both buffered rows
    assert len(_read_csv(path)) == 3
    logger.finalize()
