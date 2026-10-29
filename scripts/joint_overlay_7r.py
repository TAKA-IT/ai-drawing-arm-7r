#!/usr/bin/env python3

import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray


URDF_PATH = URDF_DIR / "generic_7r.urdf"


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

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


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c

    return np.array([
        [
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
        ],
        [
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
        ],
        [
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ],
    ])


def transform(xyz, rotation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = xyz
    return value


def parse_vector(text, default):
    if text is None:
        return np.asarray(default, dtype=float)

    return np.asarray(
        [float(value) for value in text.split()],
        dtype=float,
    )


class JointOverlay(Node):
    def __init__(self):
        super().__init__("joint_overlay_7r")

        self.joints = self.load_joints()
        self.positions = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            "/ai_task_ui/markers",
            10,
        )

        self.create_timer(
            0.10,
            self.publish_overlay,
        )

        self.get_logger().info(
            "7R joint overlay ready"
        )

    def load_joints(self):
        root = ET.parse(URDF_PATH).getroot()
        result = []

        for index in range(1, 8):
            name = "joint_" + str(index)
            joint = None

            for candidate in root.findall("joint"):
                if candidate.get("name") == name:
                    joint = candidate
                    break

            if joint is None:
                raise RuntimeError(
                    "Missing joint: " + name
                )

            origin = joint.find("origin")
            axis = joint.find("axis")

            xyz = parse_vector(
                origin.get("xyz")
                if origin is not None else None,
                [0, 0, 0],
            )

            rpy = parse_vector(
                origin.get("rpy")
                if origin is not None else None,
                [0, 0, 0],
            )

            axis_xyz = parse_vector(
                axis.get("xyz")
                if axis is not None else None,
                [1, 0, 0],
            )

            result.append({
                "name": name,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis_xyz,
            })

        return result

    def joint_state_callback(self, message):
        state = dict(
            zip(message.name, message.position)
        )

        if all(
            joint["name"] in state
            for joint in self.joints
        ):
            self.positions = {
                name: float(value)
                for name, value in state.items()
            }

    def joint_world_positions(self):
        current = np.eye(4)
        result = []

        for joint in self.joints:
            origin_rotation = rpy_matrix(
                joint["rpy"]
            )

            current = (
                current
                @ transform(
                    joint["xyz"],
                    origin_rotation,
                )
            )

            result.append(
                current[:3, 3].copy()
            )

            angle = self.positions[
                joint["name"]
            ]

            current = (
                current
                @ transform(
                    [0, 0, 0],
                    axis_angle_matrix(
                        joint["axis"],
                        angle,
                    ),
                )
            )

        return result

    def color(self, index):
        colors = {
            1: (0.20, 0.68, 1.00),
            2: (0.20, 0.68, 1.00),
            3: (0.25, 1.00, 0.40),
            4: (0.72, 0.42, 1.00),
            5: (1.00, 0.52, 0.12),
            6: (1.00, 0.52, 0.12),
            7: (1.00, 0.52, 0.12),
        }

        return colors[index]

    def publish_overlay(self):
        if self.positions is None:
            return

        points = self.joint_world_positions()
        markers = MarkerArray()

        for index, position in enumerate(
            points,
            start=1,
        ):
            color = self.color(index)

            sphere = Marker()
            sphere.header.frame_id = "world"
            sphere.header.stamp = (
                self.get_clock().now().to_msg()
            )
            sphere.ns = "joint_overlay_points"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD

            sphere.pose.position.x = float(position[0])
            sphere.pose.position.y = float(position[1])
            sphere.pose.position.z = float(position[2])
            sphere.pose.orientation.w = 1.0

            sphere.scale.x = 0.032
            sphere.scale.y = 0.032
            sphere.scale.z = 0.032

            sphere.color.r = color[0]
            sphere.color.g = color[1]
            sphere.color.b = color[2]
            sphere.color.a = 0.95

            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = "world"
            label.header.stamp = (
                self.get_clock().now().to_msg()
            )
            label.ns = "joint_overlay_labels"
            label.id = 100 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.position.x = float(position[0])
            label.pose.position.y = float(position[1])
            label.pose.position.z = float(
                position[2] + 0.055
            )
            label.pose.orientation.w = 1.0

            label.scale.z = 0.033

            label.color.r = color[0]
            label.color.g = color[1]
            label.color.b = color[2]
            label.color.a = 1.0

            label.text = "J" + str(index)

            markers.markers.append(label)

        self.publisher.publish(markers)


def main():
    rclpy.init()
    node = JointOverlay()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
