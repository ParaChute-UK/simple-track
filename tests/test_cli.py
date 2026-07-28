import sys

import pytest
import yaml

from run_simple_track import run_tracking

from .conftest import generate_mwe_files


def test_cli_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["simpletrack", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        run_tracking()
    assert excinfo.value.code == 0


def test_cli_no_configs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["simpletrack"])
    with pytest.raises(SystemExit) as excinfo:
        run_tracking()
    assert excinfo.value.code != 0


def test_cli_with_mwe_config(monkeypatch, tmp_path):
    # Generate MWE files
    generate_mwe_files(tmp_path)

    mwe_config = {
        "INPUT": {
            "path": f"{str(tmp_path)}/*.field",
            "loader": "./tests/mwe_loader.py|load_mwe",
        },
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    # Run tracking with MWE config
    mwe_config_path = tmp_path / "mwe_config.yaml"
    with open(mwe_config_path, "w") as f:
        yaml.dump(mwe_config, f)

    monkeypatch.setattr(sys, "argv", ["simpletrack", str(mwe_config_path)])
    run_tracking()


def test_cli_with_mwe_config_and_cli_input(monkeypatch, tmp_path):
    # Generate MWE files
    generate_mwe_files(tmp_path)

    mwe_config = {
        "INPUT": {
            "loader": "./tests/mwe_loader.py|load_mwe",
        },
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    input_path = f"{str(tmp_path)}/*.field"

    # Run tracking with MWE config
    mwe_config_path = tmp_path / "mwe_config.yaml"
    with open(mwe_config_path, "w") as f:
        yaml.dump(mwe_config, f)

    monkeypatch.setattr(
        sys, "argv", ["simpletrack", str(mwe_config_path), "--path", input_path]
    )
    run_tracking()


def test_cli_with_mwe_config_and_verbose_loader_input(monkeypatch, tmp_path):
    # Generate MWE files
    generate_mwe_files(tmp_path)

    mwe_config = {
        "INPUT": {
            "path": f"{str(tmp_path)}/*.field",
            "iterate_over_array": False,
        },
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    # Run tracking with MWE config and loader input
    mwe_config_path = tmp_path / "mwe_config_loader_input.yaml"
    with open(mwe_config_path, "w") as f:
        yaml.dump(mwe_config, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simpletrack",
            str(mwe_config_path),
            "--loader",
            "./tests/mwe_loader.py|load_mwe",
        ],
    )
    run_tracking()


def test_cli_with_mwe_config_and_short_loader_input(monkeypatch, tmp_path):
    # Generate MWE files
    generate_mwe_files(tmp_path)

    mwe_config = {
        "INPUT": {
            "path": f"{str(tmp_path)}/*.field",
            "iterate_over_array": False,
        },
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    # Run tracking with MWE config and loader input
    mwe_config_path = tmp_path / "mwe_config_loader_input.yaml"
    with open(mwe_config_path, "w") as f:
        yaml.dump(mwe_config, f)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simpletrack",
            str(mwe_config_path),
            "-l",
            "./tests/mwe_loader.py|load_mwe",
        ],
    )
    run_tracking()


@pytest.fixture()
def setup_mwe_config(tmp_path):
    # Generate MWE files
    generate_mwe_files(tmp_path)

    mwe_config = {
        "INPUT": {
            "path": f"{str(tmp_path)}/*.field",
            "loader": "./tests/mwe_loader.py|load_all_mwe",
            "iterate_over_array": True,
            "iterating_dim": 0,
        },
        "FEATURE": {
            "threshold": 0.5,
            "under_threshold": False,
        },
        "FLOW_SOLVER": {
            "overlap_threshold": 0.3,
            "subdomain_size": 20,
        },
        "TRACKING": {"overlap_nbhood": 5, "overlap_threshold": 0.3},
    }

    # Run tracking with MWE config
    mwe_config_path = tmp_path / "mwe_config_iterate.yaml"
    with open(mwe_config_path, "w") as f:
        yaml.dump(mwe_config, f)
    return mwe_config_path


def test_cli_with_mwe_config_and_valid_verbose_iterate_over_array_inputs(
    monkeypatch, setup_mwe_config
):
    mwe_config_path = setup_mwe_config
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simpletrack",
            str(mwe_config_path),
            "--iterate_over_array",
            "--iterating_dim",
            "0",
        ],
    )
    run_tracking()


def test_cli_with_mwe_config_and_valid_short_iterate_over_array_inputs(
    monkeypatch, setup_mwe_config
):
    mwe_config_path = setup_mwe_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simpletrack",
            str(mwe_config_path),
            "-ia",
            "-dim",
            "0",
        ],
    )
    run_tracking()


def test_cli_with_mwe_config_and_iterate_over_array_but_no_iterating_dim(
    monkeypatch, setup_mwe_config
):
    mwe_config_path = setup_mwe_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simpletrack",
            str(mwe_config_path),
            "--iterate_over_array",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        run_tracking()
    assert excinfo.value.code != 0
