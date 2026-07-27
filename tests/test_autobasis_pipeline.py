"""Tests for the auto-basis pipeline engine (config merge, manifest, resume,
input resolution, skip-if-done, artifact persistence).

These use lightweight dummy steps injected via a registry override, so the
orchestration is exercised without any quantum-chemistry backend.
"""

import json

import pytest
import yaml

from basisopt.autobasis import config as cfgmod
from basisopt.autobasis import load_basis, run_pipeline
from basisopt.autobasis.config import ConfigError, deep_merge, load_config, parse_config
from basisopt.autobasis.state import RunState, StepResult
from tests.data.factories import make_basis


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_deep_merge_override_wins_and_is_recursive():
    base = {"a": 1, "m": {"x": 1, "y": 2}, "list": [1, 2]}
    over = {"a": 9, "m": {"y": 20, "z": 30}, "list": [3]}
    merged = deep_merge(base, over)
    assert merged == {"a": 9, "m": {"x": 1, "y": 20, "z": 30}, "list": [3]}
    # inputs untouched
    assert base["m"] == {"x": 1, "y": 2}


def test_parse_config_requires_keys():
    with pytest.raises(ConfigError):
        parse_config({"name": "x", "workdir": "w", "element": "N"})  # no steps


def test_parse_config_rejects_unknown_step():
    with pytest.raises(ConfigError):
        parse_config({"name": "x", "workdir": "w", "element": "N", "steps": ["bogus"]})


def test_load_config_merges_extends(tmp_path, monkeypatch):
    # a fake preset directory so the test doesn't depend on the bundled presets
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "demo.yaml").write_text(
        yaml.safe_dump(
            {
                "tier": "demo",
                "defaults": {"method": {"psi4": {"functional": "BHHLYP"}}},
                "pruning": {"backend": "molpro", "energy_target": 2.0e-4},
            }
        )
    )
    monkeypatch.setattr(cfgmod, "PRESETS_DIR", preset_dir)

    user = tmp_path / "run.yaml"
    user.write_text(
        yaml.safe_dump(
            {
                "extends": "demo",
                "name": "N-demo",
                "workdir": str(tmp_path / "wd"),
                "element": "N",
                "steps": ["pruning"],
                "pruning": {"energy_target": 5.0e-4},  # override the preset value
            }
        )
    )
    cfg = load_config(user)
    assert cfg.tier == "demo"
    assert cfg.step_backend("pruning") == "molpro"  # from preset
    assert cfg.step_config("pruning")["energy_target"] == 5.0e-4  # user override wins
    assert cfg.method_params("psi4")["functional"] == "BHHLYP"  # from preset defaults


# --------------------------------------------------------------------------- #
# dummy steps + driver
# --------------------------------------------------------------------------- #
def _dummy_registry(calls):
    """A registry whose steps record their resolved input and emit a marker basis."""

    def make(step_name):
        def run(state: RunState, step_cfg: dict) -> StepResult:
            calls.append((step_name, state.input_basis is not None))
            basis = make_basis("h", config=(("s", (1.0, 2.0)),))
            return StepResult(
                basis=basis,
                record={"step": step_name, "had_input": state.input_basis is not None},
                exports={"basis.molpro.txt": f"! {step_name} export\n"},
            )

        return run

    from basisopt.autobasis.config import CANONICAL_STEPS

    return {name: make(name) for name in CANONICAL_STEPS}


def _write_config(path, workdir, steps, **extra):
    doc = {"name": "run", "workdir": str(workdir), "element": "H", "steps": steps}
    doc.update(extra)
    path.write_text(yaml.safe_dump(doc))
    return path


def test_driver_runs_steps_and_writes_artifacts(tmp_path):
    calls = []
    wd = tmp_path / "wd"
    cfgfile = _write_config(tmp_path / "run.yaml", wd, ["primitives", "reduction"])
    manifest = run_pipeline(cfgfile, registry=_dummy_registry(calls), timestamp="T0")

    assert [c[0] for c in calls] == ["primitives", "reduction"]
    # primitives had no input (generative); reduction picked up primitives' output
    assert calls[0][1] is False
    assert calls[1][1] is True

    # artifacts on disk with stable canonical numbering
    assert (wd / "01_primitives" / "basis.json").exists()
    assert (wd / "01_primitives" / "basis.molpro.txt").exists()
    assert (wd / "01_primitives" / "record.json").exists()
    assert (wd / "02_reduction" / "basis.json").exists()
    assert (wd / "manifest.json").exists()
    assert (wd / "config.resolved.yaml").exists()

    assert manifest.has("primitives") and manifest.has("reduction")
    rec = json.loads((wd / "02_reduction" / "record.json").read_text())
    assert rec["step"] == "reduction" and rec["had_input"] is True


def test_driver_resumes_across_invocations(tmp_path):
    wd = tmp_path / "wd"
    # first invocation: steps 1-3
    f1 = _write_config(tmp_path / "a.yaml", wd, ["primitives", "reduction", "contraction"])
    run_pipeline(f1, registry=_dummy_registry([]), timestamp="T0")

    # second invocation, same workdir: only uncontraction — must pick up
    # contraction's output from the manifest (not regenerate)
    calls = []
    f2 = _write_config(tmp_path / "b.yaml", wd, ["uncontraction"])
    run_pipeline(f2, registry=_dummy_registry(calls), timestamp="T1")

    assert calls == [("uncontraction", True)]  # resolved an input from the manifest
    assert (wd / "04_uncontraction" / "basis.json").exists()


def test_driver_skips_completed_unless_forced(tmp_path):
    wd = tmp_path / "wd"
    cfgfile = _write_config(tmp_path / "run.yaml", wd, ["primitives"])
    run_pipeline(cfgfile, registry=_dummy_registry([]), timestamp="T0")

    calls = []
    run_pipeline(cfgfile, registry=_dummy_registry(calls), timestamp="T1")
    assert calls == []  # already done -> skipped

    run_pipeline(cfgfile, registry=_dummy_registry(calls), force=True, timestamp="T2")
    assert calls == [("primitives", False)]  # forced rerun


def test_driver_input_from_explicit_path(tmp_path):
    wd = tmp_path / "wd"
    # stash a canonical basis file to be used as an explicit input
    ext = tmp_path / "external.json"
    from basisopt.autobasis import save_basis

    save_basis(make_basis("h", config=(("s", (3.0,)),)), ext)

    calls = []
    cfgfile = _write_config(
        tmp_path / "run.yaml", wd, ["uncontraction"], uncontraction={"input": str(ext)}
    )
    run_pipeline(cfgfile, registry=_dummy_registry(calls), timestamp="T0")
    assert calls == [("uncontraction", True)]  # loaded from the explicit path


def test_basis_json_roundtrip(tmp_path):
    basis = make_basis("h", config=(("s", (5.0, 1.0, 0.2)), ("p", (1.5, 0.3))))
    from basisopt.autobasis import save_basis

    save_basis(basis, tmp_path / "b.json")
    restored = load_basis(tmp_path / "b.json")
    assert set(restored) == {"h"}
    assert [len(s.exps) for s in restored["h"]] == [3, 2]


# --------------------------------------------------------------------------- #
# bundled presets + chemistry step guards + CLI
# --------------------------------------------------------------------------- #
def test_extends_by_path(tmp_path):
    # presets are not shipped; `extends` can point at a preset file by path
    preset = tmp_path / "fast.yaml"
    preset.write_text(
        yaml.safe_dump(
            {
                "tier": "fast",
                "defaults": {"method": {"psi4": {"functional": "BHHLYP"}}},
                "primitives": {"backend": "psi4"},
                "pruning": {"backend": "molpro", "energy_target": 2.0e-4},
            }
        )
    )
    user = tmp_path / "run.yaml"
    user.write_text(
        yaml.safe_dump(
            {
                "extends": "fast.yaml",  # path relative to this config file
                "name": "N-fast",
                "workdir": str(tmp_path / "wd"),
                "element": "N",
                "steps": ["primitives", "reduction"],
                "reference": {"cbs_limit": -54.5, "target": 2.0e-3},
                "pruning": {"energy_target": 5.0e-4},  # user override
            }
        )
    )
    cfg = load_config(user)
    assert cfg.tier == "fast"
    assert cfg.step_backend("primitives") == "psi4"
    assert cfg.step_config("pruning")["energy_target"] == 5.0e-4
    assert cfg.method_params("psi4")["functional"] == "BHHLYP"


def _chem_state(input_basis=None, **reference):
    cfg = parse_config(
        {
            "name": "t",
            "workdir": "w",
            "element": "N",
            "steps": ["primitives"],
            "reference": reference,
        }
    )
    return RunState(config=cfg, element="N", input_basis=input_basis)


@pytest.mark.parametrize(
    "step_name", ["reduction", "contraction", "uncontraction", "purification", "pruning"]
)
def test_steps_require_input_before_touching_backend(step_name):
    from basisopt.autobasis import get_step

    state = _chem_state(input_basis=None, cbs_limit=-1.0, target=1.0e-3)
    with pytest.raises(ValueError, match="needs an input basis"):
        get_step(step_name)(state, {})


def test_primitives_requires_cbs_limit():
    from basisopt.autobasis import get_step

    state = _chem_state(input_basis=None)  # no cbs_limit
    with pytest.raises(ValueError, match="cbs_limit"):
        get_step("primitives")(state, {})


def test_cli_list_presets_reports_none_when_unset(capsys, monkeypatch):
    from basisopt.autobasis.cli import main

    monkeypatch.setattr(cfgmod, "PRESETS_DIR", None)
    assert main(["list-presets"]) == 0
    assert "No presets found" in capsys.readouterr().out


def test_cli_list_presets_lists_from_configured_dir(tmp_path, capsys, monkeypatch):
    from basisopt.autobasis.cli import main

    (tmp_path / "fast.yaml").write_text("tier: fast\n")
    (tmp_path / "min.yaml").write_text("tier: min\n")
    monkeypatch.setattr(cfgmod, "PRESETS_DIR", tmp_path)
    assert main(["list-presets"]) == 0
    out = capsys.readouterr().out
    assert "fast" in out and "min" in out


def test_run_energy_uses_energy_key(dummy_backend):
    # regression: _run_energy looked up get_value(mol.method) (e.g. "linear"),
    # which is never a stored key, so it always returned None.
    from basisopt.autobasis.chemistry import _run_energy
    from tests.data.factories import make_molecule

    mol = make_molecule(("H", "H"), method="linear")
    assert _run_energy(mol, {}) == -2.0  # Dummy linear energy = -natoms, not None


def test_purification_records_real_energy(tmp_path):
    # end-to-end via the driver: with evaluate_energy on and a (dummy) backend,
    # the purified single-point energy must be recorded as a real number, not None.
    import numpy as np

    from basisopt.autobasis import load_basis, run_pipeline, save_basis
    from basisopt.containers import Shell

    shell = Shell()
    shell.l = "s"
    shell.exps = np.array([10.0, 3.0, 1.0, 0.3])
    shell.coefs = [np.array([0.7, 0.2, 0.1, 0.0]), np.array([0.0, 0.1, 0.3, 0.9])]
    contracted = tmp_path / "contracted.json"
    save_basis({"n": [shell]}, contracted)

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "N-purify",
                "workdir": str(wd),
                "element": "N",
                "backend": {"default": "dummy", "tmp_dir": str(tmp_path / "scratch")},
                "steps": ["purification"],
                "purification": {"input": str(contracted), "evaluate_energy": True},
            }
        )
    )

    run_pipeline(cfgfile, timestamp="T0")
    rec = json.loads((wd / "05_purification" / "record.json").read_text())
    assert rec["purified_energy"] == -1.0  # single N atom, dummy linear = -natoms
    assert set(load_basis(wd / "05_purification" / "basis.json")) == {"n"}


def test_contraction_generates_naos_with_backend(tmp_path):
    # native NAO contraction via the (dummy) backend, end-to-end through the driver
    from basisopt.autobasis import load_basis, run_pipeline, save_basis
    from tests.data.factories import make_basis

    uncontracted = make_basis("n", (("s", (10.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    infile = tmp_path / "uncontracted.json"
    save_basis(uncontracted, infile)

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "N-nao",
                "workdir": str(wd),
                "element": "N",
                "backend": {"default": "dummy", "tmp_dir": str(tmp_path / "scratch")},
                "steps": ["contraction"],
                "contraction": {
                    "input": str(infile),
                    "generate": True,
                    "n_keep": {"s": 2, "p": 1},
                    "evaluate_energy": True,
                },
            }
        )
    )

    run_pipeline(cfgfile, timestamp="T0")

    shells = load_basis(wd / "03_contraction" / "basis.json")["n"]
    assert shells[0].l == "s" and len(shells[0].coefs) == 2  # kept 2 s NAOs
    assert shells[1].l == "p" and len(shells[1].coefs) == 1  # kept 1 p NAO
    assert len(shells[0].coefs[0]) == 4  # each NAO spans the 4 s primitives

    rec = json.loads((wd / "03_contraction" / "record.json").read_text())
    assert rec["occupations"] == {"s": [4.0, 3.0], "p": [2.0]}
    assert rec["contraction_error_mEh"] == 0.0  # dummy energy is basis-independent


def test_real_purification_step_end_to_end(tmp_path):
    """Drive the *real* purification step (pure linear algebra, no backend)
    through the driver from an external input file - exercises the full path
    config -> driver -> registered chemistry step -> artifacts."""
    import numpy as np

    from basisopt.autobasis import run_pipeline, save_basis
    from basisopt.containers import Shell

    # a small contracted basis to purify
    shell = Shell()
    shell.l = "s"
    shell.exps = np.array([10.0, 3.0, 1.0, 0.3])
    shell.coefs = [np.array([0.7, 0.2, 0.1, 0.0]), np.array([0.0, 0.1, 0.3, 0.9])]
    contracted = tmp_path / "contracted.json"
    save_basis({"n": [shell]}, contracted)

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "N-purify",
                "workdir": str(wd),
                "element": "N",
                "steps": ["purification"],
                "purification": {"input": str(contracted), "evaluate_energy": False},
            }
        )
    )

    manifest = run_pipeline(cfgfile, timestamp="T0")  # real registry, no override
    assert manifest.has("purification")
    out_basis = load_basis(wd / "05_purification" / "basis.json")
    assert set(out_basis) == {"n"}
    assert (wd / "05_purification" / "record.json").exists()


# --------------------------------------------------------------------------- #
# Step 7: polarisation (multi-molecule, spectator bases)
# --------------------------------------------------------------------------- #
def test_step_polarisation_grows_shells_end_to_end(tmp_path):
    """Grow d/f polarisation shells onto an sp basis via the dummy backend.

    The dummy energy is basis-independent, so the loss is constant: each shell
    'stalls' immediately and the strategy advances d -> f -> stop (max_l). This
    exercises the whole step (combined basis, reference molecule, strategy,
    record) end-to-end through the driver, without a real backend.
    """
    from basisopt.autobasis import save_basis

    sp = make_basis("o", (("s", (10.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    infile = tmp_path / "sp.json"
    save_basis(sp, infile)

    geom = tmp_path / "o2.xyz"
    geom.write_text("2\nO2\nO 0.0 0.0 0.0\nO 0.0 0.0 1.2\n")

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "O-pol",
                "workdir": str(wd),
                "element": "O",
                "backend": {"default": "dummy", "tmp_dir": str(tmp_path / "scratch")},
                "steps": ["polarisation"],
                "polarisation": {
                    "input": str(infile),
                    "min_l": 2,
                    "max_l": 3,
                    "target": 1e-9,  # unreachable with constant dummy energy
                    "stall_tol": 1e-6,  # constant energy -> stall -> advance l
                    "opt_params": {"options": {"maxiter": 20}},
                    "molecules": [
                        {"geometry": str(geom), "cbs_limit": -2.5, "multiplicity": 3},
                    ],
                },
            }
        )
    )

    run_pipeline(cfgfile, timestamp="T0")

    shells = load_basis(wd / "07_polarisation" / "basis.json")["o"]
    ls = [sh.l for sh in shells]
    assert ls[:2] == ["s", "p"]  # sp preserved, untouched
    assert "d" in ls and "f" in ls  # polarisation shells added

    rec = json.loads((wd / "07_polarisation" / "record.json").read_text())
    assert rec["stop_reason"] == "max_l"
    assert set(rec["polarisation_shells"]) == {"d", "f"}
    assert list(rec["per_molecule_error"])  # one entry per reference molecule


def test_spectator_basis_from_published_name():
    """A spectator can pull a published basis (default pc-seg-4) by name."""
    from basisopt.autobasis.chemistry import _spectator_basis

    el, shells = _spectator_basis({"element": "H", "basis": "sto-3g"})
    assert el == "h"  # normalised to lower-case
    assert len(shells) >= 1


def test_spectator_basis_from_file(tmp_path):
    """A spectator can point at a prior-built basis file."""
    from basisopt.autobasis import save_basis
    from basisopt.autobasis.chemistry import _spectator_basis

    f = tmp_path / "h.json"
    save_basis(make_basis("h", (("s", (1.0, 0.3)),)), f)
    el, shells = _spectator_basis({"element": "H", "basis_file": str(f)})
    assert el == "h"
    assert len(shells) == 1


def test_parallel_settings_step_block_overrides_global(monkeypatch):
    """A per-step `parallel` block wins over global backend.parallel; absent, it
    falls back to global; absent both, serial. (Ray-free: set_parallel is stubbed.)"""
    from types import SimpleNamespace

    import basisopt.api as api
    from basisopt.autobasis.chemistry import _parallel_settings

    calls = {}
    monkeypatch.setattr(api, "set_parallel", lambda value, n: calls.update(value=value, n=n))
    backend_cfg = SimpleNamespace(parallel={"n_cores": 2, "threads_per_job": 1}, tmp_dir="./tmp")
    state = SimpleNamespace(config=SimpleNamespace(backend=backend_cfg))

    parallel, rp = _parallel_settings(
        state, "psi4", {"parallel": {"n_cores": 8, "threads_per_job": 2, "n_workers": 3}}
    )
    assert parallel is True and calls["n"] == 8  # the step block, not the global 2
    assert rp == {"backend": "psi4", "tmp_dir": "./tmp", "threads_per_job": 2, "n_workers": 3}

    parallel2, rp2 = _parallel_settings(state, "psi4", {})  # fall back to the global block
    assert parallel2 is True and calls["n"] == 2 and "n_workers" not in rp2

    serial_state = SimpleNamespace(
        config=SimpleNamespace(backend=SimpleNamespace(parallel=None, tmp_dir="./tmp"))
    )
    assert _parallel_settings(serial_state, "psi4", None) == (False, None)


def test_resolve_polarisation_target_absolute_and_reference_loss():
    """Absolute target passes through; target_ratio scales an explicit reference_loss;
    a target_ratio with no reference errors clearly."""
    from basisopt.autobasis.chemistry import _resolve_polarisation_target

    t, rec = _resolve_polarisation_target({"target": 1e-3}, [], None, "mean", False, None)
    assert t == 1e-3 and rec == {"mode": "absolute"}

    t, rec = _resolve_polarisation_target(
        {"target_ratio": 0.1, "reference_loss": 4e-3}, [], None, "mean", False, None
    )
    assert t == pytest.approx(4e-4)
    assert rec == {
        "mode": "relative",
        "ratio": 0.1,
        "reference_loss": 4e-3,
        "source": "reference_loss",
    }

    with pytest.raises(ValueError, match="reference_loss"):
        _resolve_polarisation_target({"target_ratio": 0.1}, [], None, "mean", False, None)


def test_reference_basis_requires_every_element(tmp_path):
    """A reference basis must cover every atom in the molecules (no silent fallback)."""
    from basisopt.autobasis import save_basis
    from basisopt.autobasis.chemistry import _reference_basis

    f = tmp_path / "o_only.json"
    save_basis(make_basis("o", (("s", (5.0, 1.0)),)), f)
    with pytest.raises(ValueError, match="no entry for element 'h'"):
        _reference_basis({"basis_file": str(f)}, {"o", "h"})


def test_step_polarisation_spectator_default_applied(tmp_path):
    """A heteronuclear molecule with no per-molecule spectator picks up the
    step-level `spectator_basis` default (so the coverage check is satisfied)."""
    from basisopt.autobasis import save_basis

    sp = make_basis("o", (("s", (10.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    infile = tmp_path / "sp.json"
    save_basis(sp, infile)

    geom = tmp_path / "oh.xyz"
    geom.write_text("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\n")

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "O-pol",
                "workdir": str(wd),
                "element": "O",
                "backend": {"default": "dummy", "tmp_dir": str(tmp_path / "scratch")},
                "steps": ["polarisation"],
                "polarisation": {
                    "input": str(infile),
                    "min_l": 2,
                    "max_l": 2,
                    "target": 1e-9,  # unreachable with constant dummy energy
                    "stall_tol": 1e-6,
                    "spectator_basis": "sto-3g",  # default fixed basis for the H
                    "opt_params": {"options": {"maxiter": 10}},
                    "molecules": [
                        {"geometry": str(geom), "cbs_limit": -2.5, "multiplicity": 2},
                    ],
                },
            }
        )
    )
    run_pipeline(cfgfile, timestamp="T0")

    rec = json.loads((wd / "07_polarisation" / "record.json").read_text())
    assert rec["spectator_basis"] == "sto-3g"
    assert rec["loss"] == "mean_per_electron"
    # the H acquired a basis from the default (else the coverage check would raise),
    # and a d polarisation shell was grown on O
    shells = load_basis(wd / "07_polarisation" / "basis.json")["o"]
    assert "d" in [sh.l for sh in shells]


# --------------------------------------------------------------------------- #
# config-driven export (global end-of-flow + per-step)
# --------------------------------------------------------------------------- #
def test_export_format_resolution():
    """Per-step format wins; else the global export format; else the default."""
    from basisopt.autobasis.pipeline import _export_format, _normalize_export

    assert _normalize_export("x.mpro") == {"path": "x.mpro"}
    assert _normalize_export({"path": "x", "format": "psi4"}) == {"path": "x", "format": "psi4"}
    assert _normalize_export(None) is None
    assert _export_format({"format": "psi4"}, {"format": "molpro"}) == "psi4"  # step wins
    assert _export_format({"path": "x"}, {"format": "nwchem"}) == "nwchem"  # falls to global
    assert _export_format({"path": "x"}, None) == "molpro"  # falls to the built-in default


def test_global_export_at_end_of_flow(tmp_path):
    """A global `export` writes the final step's basis once, at the end."""
    wd = tmp_path / "wd"
    out = tmp_path / "final.json"
    cfgfile = _write_config(
        tmp_path / "run.yaml",
        wd,
        ["primitives", "reduction"],
        export={"path": str(out), "format": "json"},
    )
    run_pipeline(cfgfile, registry=_dummy_registry([]), timestamp="T0")
    assert out.exists()  # the final (reduction) basis exported at end of flow
    assert set(load_basis(out)) == {"h"}  # reloadable internal JSON


def test_per_step_export_without_global(tmp_path):
    """A per-step export fires with NO global export set, and is recorded."""
    wd = tmp_path / "wd"
    prim_out = tmp_path / "prim.json"
    cfgfile = _write_config(
        tmp_path / "run.yaml",
        wd,
        ["primitives", "reduction"],
        primitives={"export": {"path": str(prim_out), "format": "json"}},
    )
    run_pipeline(cfgfile, registry=_dummy_registry([]), timestamp="T0")
    assert prim_out.exists()
    rec = json.loads((wd / "01_primitives" / "record.json").read_text())
    assert rec["export"] == {"path": str(prim_out), "format": "json"}
    # only the configured step exported; reduction (no export) wrote nothing extra
    assert not (tmp_path / "red.json").exists()


def test_per_step_export_reexports_on_skip(tmp_path):
    """On a resume where the step is skipped, its export still refreshes."""
    wd = tmp_path / "wd"
    out = tmp_path / "prim.json"
    cfgfile = _write_config(
        tmp_path / "run.yaml",
        wd,
        ["primitives"],
        primitives={"export": {"path": str(out), "format": "json"}},
    )
    run_pipeline(cfgfile, registry=_dummy_registry([]), timestamp="T0")
    assert out.exists()
    out.unlink()
    run_pipeline(cfgfile, registry=_dummy_registry([]), timestamp="T1")  # primitives skipped
    assert out.exists()  # export re-fired from the manifest basis


def test_export_basis_molpro_roundtrip(tmp_path):
    """A molpro export round-trips through load_basis -- i.e. it can seed a resume
    input in a non-JSON format."""
    from basisopt.autobasis import export_basis
    from basisopt.bse_wrapper import fetch_basis

    basis = {k.lower(): v for k, v in fetch_basis("sto-3g", "o").items()}
    out = tmp_path / "o.mpro"
    export_basis(basis, out, "molpro")
    assert out.exists() and out.read_text().strip()
    reloaded = load_basis(out, "molpro")  # the resume path (load_basis reads molpro)
    assert any(k.lower() == "o" for k in reloaded)


# --------------------------------------------------------------------------- #
# per-step multiplicity / charge overrides (e.g. H atom doublet vs H2 singlet)
# --------------------------------------------------------------------------- #
def test_build_atom_multiplicity_and_charge_override():
    from basisopt.autobasis.config import parse_config
    from basisopt.autobasis.state import RunState

    cfg = parse_config(
        {
            "name": "x",
            "workdir": "w",
            "element": "H",
            "steps": ["primitives"],
            "reference": {"multiplicity": 2, "charge": 0},
        }
    )
    state = RunState(config=cfg, element="H")
    assert state.build_atom("dft").multiplicity == 2  # reference default (H doublet)
    assert state.build_atom("dft", multiplicity=1).multiplicity == 1  # per-step override wins
    m = state.build_atom("dft", multiplicity=1, charge=-1)
    assert (m.multiplicity, m.charge) == (1, -1)


def test_build_geometry_molecule_multiplicity_override(tmp_path):
    from basisopt.autobasis.config import parse_config
    from basisopt.autobasis.state import RunState

    geom = tmp_path / "h2.xyz"
    geom.write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    cfg = parse_config(
        {
            "name": "x",
            "workdir": "w",
            "element": "H",
            "steps": ["uncontraction"],
            "reference": {"multiplicity": 2, "geometry": str(geom)},
        }
    )
    state = RunState(config=cfg, element="H")
    assert state.build_geometry_molecule("dft").multiplicity == 2  # reference default
    assert state.build_geometry_molecule("dft", multiplicity=1).multiplicity == 1  # H2 singlet


def test_mult_charge_parses_step_overrides():
    from basisopt.autobasis.chemistry import _mult_charge

    assert _mult_charge({}) == (None, None)
    assert _mult_charge({"multiplicity": "1", "charge": "-1"}) == (1, -1)
    assert _mult_charge({"multiplicity": 3}) == (3, None)


def test_step_multiplicity_override_reaches_record(tmp_path):
    """A per-step `multiplicity` overrides the reference for that stage only."""
    from basisopt.autobasis import save_basis

    uncontracted = make_basis("n", (("s", (10.0, 3.0, 1.0, 0.3)), ("p", (1.5, 0.4))))
    infile = tmp_path / "u.json"
    save_basis(uncontracted, infile)

    wd = tmp_path / "wd"
    cfgfile = tmp_path / "run.yaml"
    cfgfile.write_text(
        yaml.safe_dump(
            {
                "name": "N-nao",
                "workdir": str(wd),
                "element": "N",
                "backend": {"default": "dummy", "tmp_dir": str(tmp_path / "scratch")},
                "reference": {"multiplicity": 4},  # atomic default
                "steps": ["contraction"],
                "contraction": {
                    "input": str(infile),
                    "generate": True,
                    "n_keep": {"s": 2, "p": 1},
                    "multiplicity": 2,  # override just this stage
                },
            }
        )
    )
    run_pipeline(cfgfile, timestamp="T0")
    rec = json.loads((wd / "03_contraction" / "record.json").read_text())
    assert rec["multiplicity"] == 2  # the per-step override, not the reference's 4
