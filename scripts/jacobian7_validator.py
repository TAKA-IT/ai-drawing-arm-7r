#!/usr/bin/env python3

from pathlib import Path
from package_paths import SCRIPTS_DIR

import importlib.util
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


FK_PATH = str(
    SCRIPTS_DIR / "fk7_validator.py"
)

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


def load_fk_module():
    spec = importlib.util.spec_from_file_location(
        "fk7_validator",
        FK_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load FK module specification"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_functions = [
        "load_chain",
        "calculate_fk",
    ]

    missing = [
        name
        for name in required_functions
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "FK module missing functions: {}".format(
                missing
            )
        )

    return module


def rotation_vector(rotation):
    cosine = float(np.clip(
        (np.trace(rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    ))

    angle = float(np.arccos(cosine))

    if angle < 1.0e-10:
        return np.zeros(3)

    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))

    return axis * angle


def numerical_jacobian(
    fk_module,
    chain,
    joint_values,
    step,
):
    jacobian = np.zeros((6, 7))

    for index in range(7):
        plus = joint_values.copy()
        minus = joint_values.copy()

        plus[index] += step
        minus[index] -= step

        plus_positions = dict(zip(
            JOINT_NAMES,
            plus,
        ))

        minus_positions = dict(zip(
            JOINT_NAMES,
            minus,
        ))

        transform_plus = fk_module.calculate_fk(
            chain,
            plus_positions,
        )

        transform_minus = fk_module.calculate_fk(
            chain,
            minus_positions,
        )

        jacobian[:3, index] = (
            transform_plus[:3, 3]
            - transform_minus[:3, 3]
        ) / (2.0 * step)

        relative_rotation = (
            transform_plus[:3, :3]
            @ transform_minus[:3, :3].T
        )

        jacobian[3:, index] = (
            rotation_vector(relative_rotation)
            / (2.0 * step)
        )

    return jacobian


class JointStateReader(Node):
    def __init__(self):
        super().__init__("jacobian_validator_7r")

        self.message = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.receive,
            10,
        )

    def receive(self, message):
        self.message = message


def main():
    rclpy.init()
    node = JointStateReader()

    try:
        deadline = time.monotonic() + 10.0

        while (
            node.message is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

        if node.message is None:
            raise RuntimeError(
                "Timeout waiting for /joint_states"
            )

        state = dict(zip(
            node.message.name,
            node.message.position,
        ))

        missing = [
            name
            for name in JOINT_NAMES
            if name not in state
        ]

        if missing:
            raise RuntimeError(
                "Missing joint states: {}".format(
                    missing
                )
            )

        joint_values = np.array([
            state[name]
            for name in JOINT_NAMES
        ])

        fk_module = load_fk_module()
        chain = fk_module.load_chain()

        jacobian_1e6 = numerical_jacobian(
            fk_module,
            chain,
            joint_values,
            1.0e-6,
        )

        jacobian_1e7 = numerical_jacobian(
            fk_module,
            chain,
            joint_values,
            1.0e-7,
        )

        step_difference = float(np.max(np.abs(
            jacobian_1e6 - jacobian_1e7
        )))

        singular_values = np.linalg.svd(
            jacobian_1e6,
            compute_uv=False,
        )

        rank_tolerance = (
            max(jacobian_1e6.shape)
            * singular_values[0]
            * np.finfo(float).eps
        )

        rank = int(np.sum(
            singular_values > rank_tolerance
        ))

        nullity = 7 - rank

        if singular_values[-1] <= rank_tolerance:
            condition_number = float("inf")
        else:
            condition_number = float(
                singular_values[0]
                / singular_values[-1]
            )

        np.set_printoptions(
            precision=9,
            suppress=True,
        )

        print(
            "joint_positions_rad="
            + np.array2string(
                joint_values,
                separator=", ",
            )
        )

        print()
        print("jacobian_6x7=")
        print(jacobian_1e6)

        print()
        print(
            "step_consistency_max_abs="
            "{:.12e}".format(step_difference)
        )

        print(
            "singular_values="
            + np.array2string(
                singular_values,
                separator=", ",
            )
        )

        print("rank={}".format(rank))
        print("nullity={}".format(nullity))

        print(
            "condition_number={}".format(
                condition_number
            )
        )

        if jacobian_1e6.shape != (6, 7):
            raise RuntimeError(
                "Unexpected Jacobian shape: {}".format(
                    jacobian_1e6.shape
                )
            )

        if not np.isfinite(jacobian_1e6).all():
            raise RuntimeError(
                "Jacobian contains NaN or infinity"
            )

        if step_difference > 1.0e-5:
            raise RuntimeError(
                "Finite-difference consistency failed: "
                "{:.12e}".format(step_difference)
            )

        print()
        print("jacobian_shape_validation=PASS")
        print("jacobian_finite_values=PASS")
        print("finite_difference_consistency=PASS")
        print("7R_NUMERICAL_JACOBIAN=PASS")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
