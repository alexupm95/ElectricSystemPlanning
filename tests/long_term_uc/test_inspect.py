import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.read import read_usage_params, read_and_check_uc_run_params
from src.common.constants.usage_params_json import EnvPhaseNames
from my_little_europe_lt_uc import get_needed_eraa_data, set_fuel_sources_from_raw_params_json
from src.include.hydro_bounds_preprocessor import (
    relax_single_hydro_reservoir_bounds_lp,
    build_preprocessed_hydro_bounds_df,
)
import pypsa


def test_inspect_hydro_data():
    usage_params = read_usage_params()
    eraa_descr, uc_params = read_and_check_uc_run_params(
        phase_name=EnvPhaseNames.multizones_uc_model,
        usage_params=usage_params,
    )
    uc_params.set_target_years_for_capa_data(
        use_first_year_capas_as_default=usage_params.use_first_year_capas_as_default,
        init_target_year=min(eraa_descr.available_target_years)
    )
    fuel_sources = set_fuel_sources_from_raw_params_json()
    eraa_dataset = get_needed_eraa_data(
        uc_run_params=uc_params,
        eraa_data_descr=eraa_descr,
        fuel_sources=fuel_sources,
        get_optional_const_from_data_folder=usage_params.get_optional_const_from_data_folder,
    )
    min_pu, max_pu, capa = eraa_dataset.get_hydro_params_for_extr_levels_const()
    fra_min = min_pu["fra_hydro_reservoir"]
    fra_max = max_pu["fra_hydro_reservoir"]

    rel_min, rel_max = relax_single_hydro_reservoir_bounds_lp(
        unit_name="fra_hydro_reservoir",
        soc_min_pu=fra_min,
        soc_max_pu=fra_max,
        energy_capa=capa["fra_hydro_reservoir"],
        p_nom=8500.0,
        soc_init=0.8 * capa["fra_hydro_reservoir"],
        inflow=np.zeros(len(fra_min)),
        efficiency_dispatch=1.0,
        min_bandwidth_pu=0.05
    )

    net = pypsa.Network(snapshots=pd.date_range("1900-01-01 00:00:00", periods=len(fra_min), freq="h"))
    net.add("Bus", "fra")
    net.add("StorageUnit", "fra_hydro_reservoir", bus="fra", p_nom=8500.0, max_hours=1000)

    df_pre = build_preprocessed_hydro_bounds_df(
        network=net,
        soc_min_processed_pu={"fra_hydro_reservoir": rel_min},
        soc_max_processed_pu={"fra_hydro_reservoir": rel_max},
    )

    assert len(df_pre) == 52
    assert list(df_pre.columns) == ["zone", "week", "min_value", "max_value"]
    assert df_pre.iloc[0]["zone"] == "france"
    assert df_pre.iloc[0]["week"] == 1
    assert df_pre.iloc[0]["min_value"] == pytest.approx(0.320)
    assert df_pre.iloc[0]["max_value"] == pytest.approx(0.799)
