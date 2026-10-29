#!/usr/bin/env python3

import os
from pathlib import Path


def looks_like_package_root(path):
    return (
        (path / "config").is_dir()
        and (path / "urdf").is_dir()
        and (path / "scripts").is_dir()
    )


def get_package_root():
    env_root = os.environ.get(
        "ROBOT_DRAWING_ROOT"
    )

    if env_root:
        candidate = (
            Path(env_root)
            .expanduser()
            .resolve()
        )

        if not looks_like_package_root(candidate):
            raise RuntimeError(
                "ROBOT_DRAWING_ROOT is not a "
                "valid package root: "
                + str(candidate)
            )

        return candidate

    source_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    if looks_like_package_root(source_root):
        return source_root

    try:
        from ament_index_python.packages import (
            get_package_share_directory,
        )

        candidate = Path(
            get_package_share_directory(
                "robot_drawing"
            )
        )

        if not looks_like_package_root(candidate):
            raise RuntimeError(
                "Installed robot_drawing package "
                "is incomplete: "
                + str(candidate)
            )

        return candidate

    except Exception as error:
        raise RuntimeError(
            "Could not resolve robot_drawing "
            "package root"
        ) from error


PACKAGE_ROOT = get_package_root()
CONFIG_DIR = PACKAGE_ROOT / "config"
URDF_DIR = PACKAGE_ROOT / "urdf"
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
