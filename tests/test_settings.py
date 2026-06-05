from __future__ import annotations

from reviewradar.config.settings import build_paths, get_project_root


def test_project_root_resolves_to_reviewradar_repo() -> None:
    root = get_project_root()

    assert root.name == "Review-radar"
    assert (root / "pyproject.toml").exists()


def test_data_paths_are_inside_project_root() -> None:
    paths = build_paths()

    assert paths.data_dir == paths.project_root / "data"
    assert paths.raw_dir == paths.project_root / "data" / "raw"
