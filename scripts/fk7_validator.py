from pathlib import Path
from package_paths import URDF_DIR
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


URDF = URDF_DIR / "generic_7r.urdf"


def vector(text):
    return np.array([
        float(value)
        for value in (text or "0 0 0").split()
    ])


def transform(rotation=None, translation=None):
    result = np.eye(4)

    if rotation is not None:
        result[:3, :3] = rotation

    if translation is not None:
        result[:3, 3] = translation

    return result


def rpy_rotation(values):
    roll, pitch, yaw = values

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return rz @ ry @ rx


def axis_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis

    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c

    return np.array([
        [
            x * x * v + c,
            x * y * v - z * s,
            x * z * v + y * s,
        ],
        [
            y * x * v + z * s,
            y * y * v + c,
            y * z * v - x * s,
        ],
        [
            z * x * v - y * s,
            z * y * v + x * s,
            z * z * v + c,
        ],
    ])


def quaternion_rotation(values):
    x, y, z, w = values
    norm = math.sqrt(x*x + y*y + z*z + w*w)

    x, y, z, w = (
        x / norm,
        y / norm,
        z / norm,
        w / norm,
    )

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ],
    ])


def load_chain():
    root = ET.parse(URDF).getroot()
    joints_by_parent = {}

    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        parent = element.find("parent").attrib["link"]

        joints_by_parent[parent] = {
            "name": element.attrib["name"],
            "type": element.attrib.get("type", "fixed"),
            "child": element.find("child").attrib["link"],
            "xyz": vector(
                origin.attrib.get("xyz")
                if origin is not None else None
            ),
            "rpy": vector(
                origin.attrib.get("rpy")
                if origin is not None else None
            ),
            "axis": vector(
                axis.attrib.get("xyz")
                if axis is not None else None
            ),
        }

    chain = []
    link = "world"

    while link != "tcp":
        if link not in joints_by_parent:
            raise RuntimeError(
                "URDF chain stops at {}".format(link)
            )

        joint = joints_by_parent[link]
        chain.append(joint)
        link = joint["child"]

    return chain


def calculate_fk(chain, positions):
    result = np.eye(4)

    for joint in chain:
        result = (
            result
            @ transform(
                rpy_rotation(joint["rpy"]),
                joint["xyz"],
            )
        )

        if joint["type"] in ("revolute", "continuous"):
            result = (
                result
                @ transform(
                    axis_rotation(
                        joint["axis"],
                        positions[joint["name"]],
                    )
                )
            )

    return result


class Validator(Node):
    def __init__(self):
        super().__init__("fk7_checked")

        self.joint_state = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.receive_joint_state,
            10,
        )

    def receive_joint_state(self, message):
        self.joint_state = message


def main():
    rclpy.init()
    node = Validator()

    try:
        deadline = time.monotonic() + 10.0

        while (
            node.joint_state is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.joint_state is None:
            raise RuntimeError("joint_states timeout")

        positions = dict(zip(
            node.joint_state.name,
            node.joint_state.position,
        ))

        required = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        missing = [
            name
            for name in required
            if name not in positions
        ]

        if missing:
            raise RuntimeError(
                "Missing joints: {}".format(missing)
            )

        fk_result = calculate_fk(
            load_chain(),
            positions,
        )

        tf_message = None
        deadline = time.monotonic() + 10.0

        while (
            tf_message is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)

            try:
                tf_message = node.tf_buffer.lookup_transform(
                    "world",
                    "tcp",
                    Time(),
                )
            except Exception:
                pass

        if tf_message is None:
            raise RuntimeError("world to tcp TF timeout")

        translation = tf_message.transform.translation
        quaternion = tf_message.transform.rotation

        tf_position = np.array([
            translation.x,
            translation.y,
            translation.z,
        ])

        tf_rotation = quaternion_rotation([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ])

        position_error = float(
            np.linalg.norm(
                fk_result[:3, 3] - tf_position
            )
        )

        relative_rotation = (
            fk_result[:3, :3].T @ tf_rotation
        )

        cosine = float(np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        ))

        orientation_error = math.degrees(
            math.acos(cosine)
        )

        print(
            "independent_fk_translation="
            "[{:+.9f}, {:+.9f}, {:+.9f}]".format(
                fk_result[0, 3],
                fk_result[1, 3],
                fk_result[2, 3],
            )
        )

        print(
            "ros_tf_translation="
            "[{:+.9f}, {:+.9f}, {:+.9f}]".format(
                tf_position[0],
                tf_position[1],
                tf_position[2],
            )
        )

        print(
            "position_error_mm={:.9f}".format(
                position_error * 1000.0
            )
        )

        print(
            "orientation_error_deg={:.9f}".format(
                orientation_error
            )
        )

        if position_error > 1.0e-6:
            raise RuntimeError(
                "FK position comparison failed"
            )

        if orientation_error > 1.0e-5:
            raise RuntimeError(
                "FK orientation comparison failed"
            )

        print("independent_fk_validation=PASS")
        print("7R_FK_MVP=PASS")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
