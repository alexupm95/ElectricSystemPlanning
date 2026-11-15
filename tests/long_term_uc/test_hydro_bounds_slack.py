import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import pypsa
import pytest

from src.include.hydro_bounds_preprocessor import (
    relax_single_hydro_reservoir_bounds_lp,
    build_preprocessed_hydro_bounds_df,
    export_preprocessed_hydro_bounds_to_csv,
    preprocess_all_hydro_soc_bounds,
)


def test_hydro_bounds_slack_initial_soc_conflict():
    """
    Test that an initial SoC (e.g. 0.80) conflicting with an overly strict
    first target Vmax (e.g. 0.70) is minimally relaxed to permit feasibility.
    """
    n_ts = 24
    energy_capa = 10000.0  # MWh
    p_nom = 100.0          # MW (can discharge at most 100 MWh in 1 hour = 0.01 p.u.)
    soc_init = 0.80 * energy_capa  # starts at 80%
    inflow = np.zeros(n_ts)

    # Impose Vmax = 0.70 at t = 0
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_max_pu[0] = 0.70

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_hydro",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.02
    )

    # In 1 hour, max discharge is p_nom = 100 MWh -> SoC at t=0 cannot be lower than 8000 - 100 = 7900 MWh (0.79 p.u.)
    # Therefore, relaxed Vmax at t=0 must be raised to at least 0.79
    assert relaxed_max[0] >= 0.79 - 1e-5
    # For unconstrained hours, Vmax remains 1.0
    assert relaxed_max[1] == pytest.approx(1.0, abs=1e-5)


def test_hydro_bounds_slack_out_of_scale_scandinavia():
    """
    Test that out-of-scale bounds (> 1.0, e.g. Scandinavia with max ~ 4.79)
    are properly normalized and relaxed rather than clipped to 100% constant.
    """
    n_ts = 168
    energy_capa = 40000.0
    p_nom = 1000.0
    soc_init = 0.80 * energy_capa
    inflow = np.full(n_ts, 10.0)

    # Raw ERAA-like numbers for Scandinavia: week 1 min=1.199, max=4.397
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 1.199
    soc_max_pu[0] = 4.397

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="scandinavia_reservoir",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.05
    )

    # Normalized by scale factor 5: min ~ 0.24, max ~ 0.88
    # Since soc_init is 0.80, it lies well within [0.24, 0.88]!
    assert relaxed_min[0] <= 0.80 + 1e-5
    assert relaxed_max[0] >= 0.80 - 1e-5
    # Bounds must be within [0, 1]
    assert np.all(relaxed_min >= 0.0)
    assert np.all(relaxed_max <= 1.0)


def test_hydro_bounds_slack_already_feasible_unchanged():
    """
    Test that when bounds are easily reachable and feasible,
    the preprocessor preserves them without distortion.
    """
    n_ts = 24
    energy_capa = 10000.0
    p_nom = 500.0
    soc_init = 0.50 * energy_capa
    inflow = np.full(n_ts, 50.0)

    # Feasible corridor around 50%
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 0.40
    soc_max_pu[0] = 0.60

    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_feasible",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=0.05
    )

    assert relaxed_min[0] == pytest.approx(0.40, abs=1e-5)
    assert relaxed_max[0] == pytest.approx(0.60, abs=1e-5)


def test_hydro_bounds_slack_bandwidth_enforcement():
    """
    Test that when Vmin and Vmax are extremely close, the preprocessor
    expands them to guarantee a minimum operational bandwidth for dispatch.
    """
    n_ts = 24
    energy_capa = 10000.0
    p_nom = 500.0
    soc_init = 0.50 * energy_capa
    inflow = np.full(n_ts, 50.0)

    # Pinpoint bounds Vmin = 0.50, Vmax = 0.50
    soc_min_pu = np.zeros(n_ts)
    soc_max_pu = np.ones(n_ts)
    soc_min_pu[0] = 0.50
    soc_max_pu[0] = 0.50

    min_bandwidth = 0.06
    relaxed_min, relaxed_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="test_pinpoint",
        soc_min_pu=soc_min_pu,
        soc_max_pu=soc_max_pu,
        energy_capa=energy_capa,
        p_nom=p_nom,
        soc_init=soc_init,
        inflow=inflow,
        min_bandwidth_pu=min_bandwidth
    )

    width = relaxed_max[0] - relaxed_min[0]
    assert width >= min_bandwidth - 1e-5


def test_build_preprocessed_hydro_bounds_df():
    """
    Test that build_preprocessed_hydro_bounds_df formats weekly hydro levels
    with columns zone;week;min_value;max_value matching the ERAA input structure.
    """
    n_ts = 336  # 2 weeks
    dates = pd.date_range("1900-01-01 00:00:00", periods=n_ts, freq="h")
    net = pypsa.Network(snapshots=dates)
    net.add("Bus", "fra")
    net.add("Bus", "ibe")
    net.add("StorageUnit", "fra_hydro_reservoir", bus="fra", p_nom=1000, max_hours=10)
    net.add("StorageUnit", "ibe_hydro_reservoir", bus="ibe", p_nom=2000, max_hours=10)

    # Week 1 (t=0) and Week 2 (t=168)
    soc_min = {
        "fra_hydro_reservoir": np.zeros(n_ts),
        "ibe_hydro_reservoir": np.zeros(n_ts),
    }
    soc_max = {
        "fra_hydro_reservoir": np.ones(n_ts),
        "ibe_hydro_reservoir": np.ones(n_ts),
    }
    soc_min["fra_hydro_reservoir"][0] = 0.320
    soc_max["fra_hydro_reservoir"][0] = 0.790
    soc_min["fra_hydro_reservoir"][168] = 0.306
    soc_max["fra_hydro_reservoir"][168] = 0.772

    soc_min["ibe_hydro_reservoir"][0] = 0.278
    soc_max["ibe_hydro_reservoir"][0] = 0.725
    soc_min["ibe_hydro_reservoir"][168] = 0.278
    soc_max["ibe_hydro_reservoir"][168] = 0.734

    df = build_preprocessed_hydro_bounds_df(
        network=net,
        soc_min_processed_pu=soc_min,
        soc_max_processed_pu=soc_max,
        decimal_digits=3
    )

    assert list(df.columns) == ["zone", "week", "min_value", "max_value"]
    assert set(df["zone"].unique()) == {"france", "iberian-peninsula"}
    assert list(df[df["zone"] == "france"]["week"]) == [1, 2]
    assert df[(df["zone"] == "france") & (df["week"] == 1)]["min_value"].values[0] == pytest.approx(0.320)
    assert df[(df["zone"] == "france") & (df["week"] == 1)]["max_value"].values[0] == pytest.approx(0.790)


def test_export_preprocessed_hydro_bounds_to_csv(tmp_path):
    """
    Test that export_preprocessed_hydro_bounds_to_csv creates a valid CSV file
    with ';' separator, correct headers and rows.
    """
    n_ts = 168  # 1 week
    dates = pd.date_range("1900-01-01 00:00:00", periods=n_ts, freq="h")
    net = pypsa.Network(snapshots=dates)
    net.add("Bus", "sca")
    net.add("StorageUnit", "sca_hydro_reservoir", bus="sca", p_nom=1000, max_hours=10)

    soc_min = {"sca_hydro_reservoir": np.zeros(n_ts)}
    soc_max = {"sca_hydro_reservoir": np.ones(n_ts)}
    soc_min["sca_hydro_reservoir"][0] = 0.240
    soc_max["sca_hydro_reservoir"][0] = 0.879

    out_file = str(tmp_path / "PECD-hydro-weekly-reservoir-min-max-levels_preprocessed.csv")
    df = export_preprocessed_hydro_bounds_to_csv(
        network=net,
        soc_min_processed_pu=soc_min,
        soc_max_processed_pu=soc_max,
        output_filepath=out_file,
        decimal_digits=3
    )

    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        lines = f.readlines()

    assert lines[0].strip() == "zone;week;min_value;max_value"
    assert lines[1].strip() == "scandinavia;1;0.240;0.879"

    # Verify reload with pandas using ';' separator
    loaded_df = pd.read_csv(out_file, sep=";")
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["zone"] == "scandinavia"
    assert loaded_df.iloc[0]["week"] == 1
    assert loaded_df.iloc[0]["min_value"] == pytest.approx(0.240)
    assert loaded_df.iloc[0]["max_value"] == pytest.approx(0.879)

