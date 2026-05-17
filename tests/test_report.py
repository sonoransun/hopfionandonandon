"""Report generation: render Markdown, write figures."""
from pathlib import Path

from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.report import write_report
from hopfion.pipeline.runner import run


def test_report_renders_with_qc_and_figs(tmp_path):
    rc = RecipeConfig.from_dict({
        "name": "smoke_report",
        "grid": {"nx": 24, "ny": 24, "nz": 24, "dx": 0.4, "bc": "periodic"},
        "material": {"A_ex": 1.0, "D": 1.5, "Ku": 0.7,
                     "easy_axis": [0, 0, 1], "H_ext": [0, 0, 0]},
        "initial": {"kind": "hopfion", "R": 1.5, "p": 1, "q": 1},
        "run": [{"kind": "relax", "n_steps": 20, "dt": 0.003}],
        "qc": {"fail_on": [
            {"energy_monotonicity": {"tol": 1e-6}},
            {"norm_drift_below": {"tol": 1e-10}},
        ]},
        "io": {"out": str(tmp_path / "out")},
        "seed": 0,
    })
    run(rc, write=True)
    report = write_report(tmp_path / "out")
    text = Path(report).read_text()
    assert "Verdict" in text
    assert "ACCEPT" in text or "FAIL" in text
    assert (tmp_path / "out" / "figs" / "final_xy.png").exists()
    assert (tmp_path / "out" / "figs" / "energy.png").exists()
