import os
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.common.constants.links import set_link_name
from src.common.constants.uc_json_inputs import EuropeJsonParamNames, EUR_JSON_PARAM_TYPES_FOR_CHECK
from src.common.long_term_uc_io import FigNamesPrefix, get_output_figure, set_full_lt_uc_output_folder
from src.include.uc_postprocessing import UCOptimalSolution
from src.utils.type_checker import CheckerNames, check_bool, apply_data_type_check


@pytest.fixture
def clean_test_output_folder():
    test_tag = "test_sat_unit"
    yield test_tag
    # Teardown: clean up any generated test folders
    for folder_type in ("figures", "data"):
        try:
            folder_path = set_full_lt_uc_output_folder(folder_type=folder_type, country="europe", output_tag=test_tag)
            parent_folder = Path(folder_path).parent
            if parent_folder.exists() and test_tag in str(parent_folder):
                shutil.rmtree(parent_folder, ignore_errors=True)
            elif os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
        except Exception:
            pass


def test_type_checker_bool():
    """Verify that is_bool checker correctly validates booleans and rejects other types."""
    assert check_bool(True) is True
    assert check_bool(False) is True
    assert check_bool(1) is False
    assert check_bool(0) is False
    assert check_bool("True") is False
    assert check_bool(None) is False
    assert check_bool([]) is False

    # Check via apply_data_type_check
    assert apply_data_type_check(CheckerNames.is_bool, True) is True
    assert apply_data_type_check(CheckerNames.is_bool, False) is True
    assert apply_data_type_check(CheckerNames.is_bool, "yes") is False

    # Check EUR_JSON_PARAM_TYPES_FOR_CHECK mapping
    assert EuropeJsonParamNames.plot_link_capa_saturation in EUR_JSON_PARAM_TYPES_FOR_CHECK
    assert EUR_JSON_PARAM_TYPES_FOR_CHECK[EuropeJsonParamNames.plot_link_capa_saturation] == CheckerNames.is_bool


def test_plot_link_capa_saturation_normal_diurnal(clean_test_output_folder):
    """Verify diurnal sunburst plotting with link saturation."""
    tag = clean_test_output_folder
    uc_sol = UCOptimalSolution(network_name="test_network")

    # 48 snapshots (2 full days)
    snapshots = pd.date_range("2025-01-01 00:00", periods=48, freq="h")
    link_name = set_link_name("fra", "ger", carrier_name="ac")

    # Flow is 1000 at hours 2 and 3 (00h-06h diurnal block), -1000 at hours 14 and 15 (12h-18h diurnal block)
    flows = np.zeros(48)
    flows[2] = 1000.0
    flows[3] = 1000.0
    flows[14] = -1000.0
    flows[15] = -1000.0

    uc_sol.link_flow_direct = pd.DataFrame({link_name: flows}, index=snapshots)
    link_capas = {link_name: 1000.0}

    year = 2025
    climatic_year = 1989
    start_horizon = datetime(2025, 1, 1)

    uc_sol.plot_link_capa_saturation(
        link_capacities=link_capas,
        year=year,
        climatic_year=climatic_year,
        start_horizon=start_horizon,
        output_tag=tag,
        n_time_blocks=4,
        time_label_mode="diurnal"
    )

    png_filepath = get_output_figure(
        fig_name=FigNamesPrefix.link_capa_saturation,
        country="europe",
        year=year,
        climatic_year=climatic_year,
        start_horizon=start_horizon,
        output_tag=tag,
    )
    html_filepath = f"{os.path.splitext(png_filepath)[0]}.html"

    # HTML interactive file must be generated and non-empty
    assert os.path.exists(html_filepath), f"Expected HTML output file at {html_filepath}"
    assert os.path.getsize(html_filepath) > 0


def test_plot_link_capa_saturation_no_saturation(clean_test_output_folder):
    """Verify that execution completes gracefully without error when there is no saturation."""
    tag = clean_test_output_folder
    uc_sol = UCOptimalSolution(network_name="test_network")

    snapshots = pd.date_range("2025-01-01 00:00", periods=24, freq="h")
    link_name = set_link_name("fra", "ger", carrier_name="ac")

    # Flow well below threshold
    flows = np.ones(24) * 100.0
    uc_sol.link_flow_direct = pd.DataFrame({link_name: flows}, index=snapshots)
    link_capas = {link_name: 1000.0}

    # Should not raise ValueError
    uc_sol.plot_link_capa_saturation(
        link_capacities=link_capas,
        year=2025,
        climatic_year=1989,
        start_horizon=datetime(2025, 1, 1),
        output_tag=tag,
    )


def test_plot_link_capa_saturation_zero_capacity():
    """Verify that links with 0 or negative capacity are safely ignored."""
    uc_sol = UCOptimalSolution(network_name="test_network")

    snapshots = pd.date_range("2025-01-01 00:00", periods=24, freq="h")
    link_name = set_link_name("fra", "ger", carrier_name="ac")

    uc_sol.link_flow_direct = pd.DataFrame({link_name: np.zeros(24)}, index=snapshots)
    link_capas = {link_name: 0.0, "unknown_link": -10.0}

    # Should not divide by zero or raise exception
    uc_sol.plot_link_capa_saturation(
        link_capacities=link_capas,
        year=2025,
        climatic_year=1989,
        start_horizon=datetime(2025, 1, 1),
    )


def test_plot_link_capa_saturation_empty_dataframe():
    """Verify that empty or None DataFrame is handled gracefully."""
    uc_sol = UCOptimalSolution(network_name="test_network")
    uc_sol.link_flow_direct = pd.DataFrame()

    uc_sol.plot_link_capa_saturation(
        link_capacities={"fra-ger_ac": 100.0},
        year=2025,
        climatic_year=1989,
        start_horizon=datetime(2025, 1, 1),
    )

    uc_sol.link_flow_direct = None
    uc_sol.plot_link_capa_saturation(
        link_capacities={"fra-ger_ac": 100.0},
        year=2025,
        climatic_year=1989,
        start_horizon=datetime(2025, 1, 1),
    )


def test_plot_link_capa_saturation_nonstandard_link_names(clean_test_output_folder):
    """Verify resilience when link names do not conform to the standard regex."""
    tag = clean_test_output_folder
    uc_sol = UCOptimalSolution(network_name="test_network")

    snapshots = pd.date_range("2025-01-01 00:00", periods=24, freq="h")
    # Non-standard link names (e.g. dc link, or non-3-char bus names)
    non_std_link = "france-germany_dc"
    flows = np.zeros(24)
    flows[0] = 500.0
    uc_sol.link_flow_direct = pd.DataFrame({non_std_link: flows}, index=snapshots)

    # Should safely parse bus names without unpacking error
    uc_sol.plot_link_capa_saturation(
        link_capacities={non_std_link: 500.0},
        year=2025,
        climatic_year=1989,
        start_horizon=datetime(2025, 1, 1),
        output_tag=tag,
    )


def test_plot_link_capa_saturation_chronological_modes(clean_test_output_folder):
    """Verify support for blocks and dates chronological aggregation modes."""
    tag = clean_test_output_folder
    uc_sol = UCOptimalSolution(network_name="test_network")

    snapshots = pd.date_range("2025-01-01 00:00", periods=48, freq="h")
    link_name = set_link_name("fra", "ger", carrier_name="ac")
    flows = np.zeros(48)
    flows[5] = 1000.0
    uc_sol.link_flow_direct = pd.DataFrame({link_name: flows}, index=snapshots)

    for mode in ("blocks", "dates"):
        uc_sol.plot_link_capa_saturation(
            link_capacities={link_name: 1000.0},
            year=2025,
            climatic_year=1989,
            start_horizon=datetime(2025, 1, 1),
            output_tag=tag,
            time_label_mode=mode,
        )
