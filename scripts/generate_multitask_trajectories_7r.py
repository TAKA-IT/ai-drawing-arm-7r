#!/usr/bin/env python3

import importlib.util
from pathlib import Path

from package_paths import PACKAGE_ROOT

import numpy as np


BASE = PACKAGE_ROOT

FK_PATH = BASE / "scripts/fk7_validator.py"
JAC_PATH = BASE / "scripts/jacobian7_validator.py"

CIRCLE_PATH = (
    BASE / "config/circle7_trajectory.npz"
)

LINE_PATH = (
    BASE / "config/line7_trajectory.npz"
)

SQUARE_PATH = (
    BASE / "config/square7_trajectory.npz"
)

JOINT_NAMES = [
    f"joint_{index}"
    for index in range(1, 8)
]

LOWER = np.array([
    -3.14159,
    -2.61799,
    -2.61799,
    -3.14159,
    -2.61799,
    -3.14159,
    -3.14159,
])

UPPER = np.array([
    3.14159,
    2.61799,
    2.61799,
    3.14159,
    2.61799,
    3.14159,
    3.14159,
])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Module load failed: " + str(path)
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def rotation_vector(rotation):
    cosine = float(
        np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = float(np.arccos(cosine))

    if angle < 1e-10:
        return np.zeros(3)

    return (
        angle
        * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        / (2.0 * np.sin(angle))
    )


def pose_error(target, current):
    position = (
        target[:3, 3]
        - current[:3, 3]
    )

    orientation = rotation_vector(
        target[:3, :3]
        @ current[:3, :3].T
    )

    return np.concatenate([
        position,
        orientation,
    ])


def solve(
    fk_module,
    jacobian_module,
    chain,
    target,
    seed,
):
    q = seed.copy()

    for iteration in range(500):
        current = fk_module.calculate_fk(
            chain,
            dict(zip(JOINT_NAMES, q)),
        )

        error = pose_error(
            target,
            current,
        )

        if (
            np.linalg.norm(error[:3]) < 1e-5
            and np.linalg.norm(error[3:]) < 1e-5
        ):
            return q, iteration + 1

        jacobian = (
            jacobian_module.numerical_jacobian(
                fk_module,
                chain,
                q,
                1e-6,
            )
        )

        sigma_min = float(
            np.linalg.svd(
                jacobian,
                compute_uv=False,
            )[-1]
        )

        damping = (
            0.002
            if sigma_min >= 0.02
            else (
                0.002
                + 0.028
                * (
                    1.0
                    - sigma_min / 0.02
                ) ** 2
            )
        )

        delta = (
            jacobian.T
            @ np.linalg.solve(
                jacobian @ jacobian.T
                + damping * damping * np.eye(6),
                error,
            )
        )

        maximum = float(
            np.max(np.abs(delta))
        )

        if maximum > 0.06:
            delta *= 0.06 / maximum

        base_error = float(
            np.linalg.norm(error)
        )

        accepted = False

        for scale in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ):
            candidate = np.clip(
                q + scale * delta,
                LOWER,
                UPPER,
            )

            candidate_pose = (
                fk_module.calculate_fk(
                    chain,
                    dict(
                        zip(
                            JOINT_NAMES,
                            candidate,
                        )
                    ),
                )
            )

            candidate_error = float(
                np.linalg.norm(
                    pose_error(
                        target,
                        candidate_pose,
                    )
                )
            )

            if candidate_error < base_error:
                q = candidate
                accepted = True
                break

        if not accepted:
            break

    raise RuntimeError("IK did not converge")


def interpolate(start, end, count):
    return np.linspace(
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
        count,
    )


def square_points(center, half_side):
    corners = [
        center + np.array([
            0.0,
            -half_side,
            -half_side,
        ]),
        center + np.array([
            0.0,
            half_side,
            -half_side,
        ]),
        center + np.array([
            0.0,
            half_side,
            half_side,
        ]),
        center + np.array([
            0.0,
            -half_side,
            half_side,
        ]),
        center + np.array([
            0.0,
            -half_side,
            -half_side,
        ]),
    ]

    result = []

    for index in range(4):
        segment = interpolate(
            corners[index],
            corners[index + 1],
            17,
        )

        if index > 0:
            segment = segment[1:]

        result.extend(segment)

    return np.asarray(result)


def solve_path(
    name,
    points,
    reference_pose,
    seed,
    output_path,
    fk_module,
    jacobian_module,
    chain,
):
    q = seed.copy()
    positions = []
    actual_tcp = []
    orientation_errors = []
    iterations = []

    for point in points:
        target = reference_pose.copy()
        target[:3, 3] = point

        q, count = solve(
            fk_module,
            jacobian_module,
            chain,
            target,
            q,
        )

        actual = fk_module.calculate_fk(
            chain,
            dict(zip(JOINT_NAMES, q)),
        )

        positions.append(q.copy())
        actual_tcp.append(
            actual[:3, 3].copy()
        )

        orientation_errors.append(
            np.degrees(
                np.linalg.norm(
                    rotation_vector(
                        reference_pose[:3, :3]
                        @ actual[:3, :3].T
                    )
                )
            )
        )

        iterations.append(count)

    positions = np.asarray(positions)
    actual_tcp = np.asarray(actual_tcp)

    position_errors = np.linalg.norm(
        actual_tcp - points,
        axis=1,
    ) * 1000.0

    joint_steps = np.linalg.norm(
        np.diff(positions, axis=0),
        axis=1,
    )

    maximum_position_error = float(
        np.max(position_errors)
    )

    maximum_orientation_error = float(
        np.max(orientation_errors)
    )

    maximum_joint_step = float(
        np.max(joint_steps)
    )

    limits_ok = bool(
        np.all(positions >= LOWER)
        and np.all(positions <= UPPER)
    )

    np.savez(
        output_path,
        joint_names=np.array(JOINT_NAMES),
        positions=positions,
        actual_tcp=actual_tcp,
        target_tcp=points,
        task_name=np.array(name),
    )

    print()
    print("task=" + name)
    print("point_count=" + str(len(points)))
    print(
        "max_position_error_mm="
        + format(
            maximum_position_error,
            ".9f",
        )
    )
    print(
        "max_orientation_error_deg="
        + format(
            maximum_orientation_error,
            ".9f",
        )
    )
    print(
        "max_joint_step_rad="
        + format(
            maximum_joint_step,
            ".9f",
        )
    )
    print(
        "max_ik_iterations="
        + str(max(iterations))
    )
    print(
        "joint_limits_respected="
        + str(limits_ok)
    )
    print("output=" + str(output_path))

    if maximum_position_error > 0.02:
        raise RuntimeError(
            name + " position validation failed"
        )

    if maximum_orientation_error > 0.001:
        raise RuntimeError(
            name + " orientation validation failed"
        )

    if maximum_joint_step > 0.15:
        raise RuntimeError(
            name + " continuity validation failed"
        )

    if not limits_ok:
        raise RuntimeError(
            name + " joint limit validation failed"
        )

    print(name + "_TRAJECTORY=PASS")


def main():
    fk_module = load_module(
        FK_PATH,
        "fk_multitask",
    )

    jacobian_module = load_module(
        JAC_PATH,
        "jacobian_multitask",
    )

    chain = fk_module.load_chain()

    with np.load(
        CIRCLE_PATH,
        allow_pickle=False,
    ) as data:
        circle_positions = np.asarray(
            data["positions"],
            dtype=float,
        )
        center = np.asarray(
            data["center"],
            dtype=float,
        )

    seed = circle_positions[0].copy()

    reference_pose = (
        fk_module.calculate_fk(
            chain,
            dict(
                zip(
                    JOINT_NAMES,
                    seed,
                )
            ),
        )
    )

    # 40 mm straight line in the Y direction.
    line_points = interpolate(
        center + np.array([
            0.0,
            -0.020,
            0.0,
        ]),
        center + np.array([
            0.0,
            0.020,
            0.0,
        ]),
        41,
    )

    # 35 mm square, inside the validated R25 mm circle.
    square = square_points(
        center,
        0.0175,
    )

    solve_path(
        "LINE",
        line_points,
        reference_pose,
        seed,
        LINE_PATH,
        fk_module,
        jacobian_module,
        chain,
    )

    solve_path(
        "SQUARE",
        square,
        reference_pose,
        seed,
        SQUARE_PATH,
        fk_module,
        jacobian_module,
        chain,
    )

    print()
    print(
        "7R_MULTITASK_TRAJECTORIES=PASS"
    )


if __name__ == "__main__":
    main()
