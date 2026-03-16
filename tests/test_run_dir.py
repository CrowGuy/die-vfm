from __future__ import annotations

import pytest

from die_vfm.utils.run_dir import create_run_dir


def test_create_run_dir(tmp_path) -> None:
    """Tests that the run directory and expected subdirectories are created."""
    run_directory = create_run_dir(
        output_root=str(tmp_path),
        run_name="unit_test_run",
    )

    assert run_directory.exists()
    assert run_directory.name == "unit_test_run"
    assert (run_directory / "checkpoints").exists()
    assert (run_directory / "embeddings").exists()
    assert (run_directory / "logs").exists()


def test_create_run_dir_raises_if_exists(tmp_path) -> None:
    """Tests that creating the same run directory twice raises an error."""
    create_run_dir(
        output_root=str(tmp_path),
        run_name="unit_test_run",
    )

    with pytest.raises(FileExistsError):
        create_run_dir(
            output_root=str(tmp_path),
            run_name="unit_test_run",
        )