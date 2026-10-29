#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import re
import threading
import time
import unicodedata

import numpy as np
import torch
import rclpy

from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


from package_paths import PACKAGE_ROOT

BASE = str(PACKAGE_ROOT)

MODEL_PATH = BASE + "/config/task_intent_7r.pt"
TRAIN_SCRIPT = BASE + "/scripts/train_task_intent_7r.py"

TRAJECTORY_PATHS = {
    "DRAW_CIRCLE": (
        BASE + "/config/circle7_trajectory.npz"
    ),
    "DRAW_LINE": (
        BASE + "/config/line7_trajectory.npz"
    ),
    "DRAW_SQUARE": (
        BASE + "/config/square7_trajectory.npz"
    ),
}

TASK_LABELS = {
    "DRAW_CIRCLE": "CIRCLE_R25_73PTS",
    "DRAW_LINE": "LINE_40MM_41PTS",
    "DRAW_SQUARE": "SQUARE_35MM_65PTS",
}

CONFIDENCE_THRESHOLD = 0.80


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "intent_model_definition",
        TRAIN_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Model definition load failed"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class AIMultiTaskExecutor(Node):
    def __init__(self):
        super().__init__("ai_task_executor_7r")

        model_module = load_training_module()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
        )

        self.labels = checkpoint["labels"]
        self.feature_dim = checkpoint["feature_dim"]
        self.model_type = checkpoint["model_type"]

        self.model = model_module.IntentClassifier()
        self.model.load_state_dict(
            checkpoint["state_dict"]
        )
        self.model.eval()

        self.trajectories = {}

        for intent, path in TRAJECTORY_PATHS.items():
            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                names = data["joint_names"].tolist()
                positions = np.asarray(
                    data["positions"],
                    dtype=float,
                )

                if "actual_tcp" in data.files:
                    tcp = np.asarray(
                        data["actual_tcp"],
                        dtype=float,
                    )
                else:
                    tcp = np.asarray(
                        data["target_tcp"],
                        dtype=float,
                    )

            self.trajectories[intent] = {
                "joint_names": names,
                "positions": positions,
                "tcp": tcp,
            }

        # Safe retract pose on the robot side of the drawing plane.
        with np.load(
            BASE + "/config/safe_home7.npz",
            allow_pickle=False,
        ) as safe_home_data:
            self.home_positions = np.asarray(
                safe_home_data["positions"],
                dtype=float,
            )

            self.safe_home_tcp = np.asarray(
                safe_home_data["actual_tcp"],
                dtype=float,
            )

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        self.create_subscription(
            String,
            "/input_text",
            self.input_callback,
            10,
        )

        qos = QoSProfile(depth=1)
        qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "/ai_task_ui/markers",
            qos,
        )

        self.state_publisher = self.create_publisher(
            String,
            "/ai_task_state",
            qos,
        )

        self.create_timer(
            0.25,
            self.publish_ui,
        )

        self.instruction = "WAITING"
        self.intent = "NONE"
        self.confidence = 0.0
        self.execution_state = "READY"
        self.result = "NOT_STARTED"
        self.selected_intent = None
        self.path_visible = False
        self.busy = False

        self.get_logger().info(
            "7R multitask AI executor ready"
        )

    def normalize(self, text):
        text = unicodedata.normalize(
            "NFKC",
            text,
        )
        text = text.lower().strip()
        return re.sub(r"\s+", " ", text)

    def grams(self, text):
        padded = "^" + self.normalize(text) + "$"
        values = []

        for size in (1, 2, 3):
            for index in range(
                len(padded) - size + 1
            ):
                values.append(
                    padded[index:index + size]
                )

        return values

    def feature_index(self, token):
        digest = hashlib.sha256(
            token.encode("utf-8")
        ).digest()

        return int.from_bytes(
            digest[:8],
            "little",
        ) % self.feature_dim

    def vectorize(self, text):
        vector = torch.zeros(
            self.feature_dim,
            dtype=torch.float32,
        )

        for token in self.grams(text):
            vector[
                self.feature_index(token)
            ] += 1.0

        norm = torch.linalg.vector_norm(vector)

        if norm > 0:
            vector /= norm

        return vector

    def infer(self, text):
        with torch.no_grad():
            logits = self.model(
                self.vectorize(text).unsqueeze(0)
            )[0]

            probabilities = torch.softmax(
                logits,
                dim=0,
            )

            confidence, index = torch.max(
                probabilities,
                dim=0,
            )

        return (
            self.labels[int(index)],
            float(confidence),
        )

    def display_instruction(self):
        values = {
            "DRAW_CIRCLE": "DRAW_CIRCLE",
            "DRAW_LINE": "DRAW_LINE",
            "DRAW_SQUARE": "DRAW_SQUARE",
            "STOP": "STOP",
            "UNKNOWN": "UNKNOWN",
            "NONE": "WAITING",
        }

        return values.get(
            self.intent,
            "OTHER",
        )

    def input_callback(self, message):
        text = message.data.strip()

        if not text:
            return

        if self.busy:
            self.execution_state = "BUSY"
            self.result = "REJECTED"
            return

        # 新しい指示が届いた時点で、前の軌道を消す。
        self.path_visible = False
        self.selected_intent = None
        self.execution_state = "AI_INFERENCE"
        self.result = "PENDING"

        intent, confidence = self.infer(text)

        self.instruction = text
        self.intent = intent
        self.confidence = confidence

        self.get_logger().info(
            json.dumps(
                {
                    "instruction": text,
                    "intent": intent,
                    "confidence": confidence,
                },
                ensure_ascii=False,
            )
        )

        if confidence < CONFIDENCE_THRESHOLD:
            self.execution_state = "LOW_CONFIDENCE"
            self.result = "NOT_EXECUTED"
            return

        if intent not in self.trajectories:
            self.execution_state = "UNSUPPORTED"
            self.result = "NOT_EXECUTED"
            return

        # AIがタスクを選択した後にだけ軌道を表示する。
        self.selected_intent = intent
        self.path_visible = True
        self.execution_state = "INFERENCE_PASS"
        self.result = "AUTHORIZED"
        self.busy = True

        threading.Thread(
            target=self.execute_selected_task,
            daemon=True,
        ).start()

    def create_goal(self, positions, times):
        trajectory = self.trajectories[
            self.selected_intent
        ]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(
            trajectory["joint_names"]
        )

        for q, elapsed in zip(
            positions,
            times,
        ):
            point = JointTrajectoryPoint()
            point.positions = [
                float(value)
                for value in q
            ]

            seconds = float(elapsed)
            point.time_from_start.sec = int(
                seconds
            )

            point.time_from_start.nanosec = int(
                round(
                    (
                        seconds
                        - int(seconds)
                    ) * 1e9
                )
            )

            goal.trajectory.points.append(point)

        return goal

    def wait_future(self, future, timeout):
        end = time.monotonic() + timeout

        while (
            not future.done()
            and time.monotonic() < end
        ):
            time.sleep(0.05)

        if not future.done():
            raise RuntimeError("Action timeout")

        return future.result()

    def send_goal(
        self,
        positions,
        times,
        state,
        timeout,
    ):
        self.execution_state = state

        if not self.action_client.wait_for_server(
            timeout_sec=10.0
        ):
            raise RuntimeError(
                "Action server unavailable"
            )

        goal = self.create_goal(
            positions,
            times,
        )

        handle = self.wait_future(
            self.action_client.send_goal_async(
                goal
            ),
            15.0,
        )

        if handle is None or not handle.accepted:
            raise RuntimeError(
                state + " rejected"
            )

        wrapped = self.wait_future(
            handle.get_result_async(),
            timeout,
        )

        if wrapped is None:
            raise RuntimeError(
                state + " result unavailable"
            )

        if wrapped.result.error_code != 0:
            raise RuntimeError(
                state
                + " failed: "
                + wrapped.result.error_string
            )

    def execute_selected_task(self):
        try:
            trajectory = self.trajectories[
                self.selected_intent
            ]

            positions = trajectory["positions"]

            self.send_goal(
                positions[:1],
                [4.0],
                "MOVE_TO_START",
                15.0,
            )

            time.sleep(1.0)

            point_count = len(positions)

            if self.selected_intent == "DRAW_CIRCLE":
                step = 0.20
            elif self.selected_intent == "DRAW_SQUARE":
                step = 0.17
            else:
                step = 0.16

            times = step * np.arange(
                1,
                point_count + 1,
            )

            self.send_goal(
                positions,
                times,
                "DRAWING",
                30.0,
            )

            time.sleep(1.0)

            self.send_goal(
                [self.home_positions],
                [4.0],
                "RETURN_TO_HOME",
                15.0,
            )

            time.sleep(1.0)

            self.execution_state = "COMPLETE"
            self.result = "PASS"

        except Exception as error:
            self.execution_state = "ERROR"
            self.result = (
                type(error).__name__
                + ":"
                + str(error)
            )

            self.get_logger().error(
                self.result
            )

        finally:
            self.busy = False

    def text_marker(
        self,
        marker_id,
        text,
        z,
        scale,
        color,
    ):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = "ai_task_text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = 0.72
        marker.pose.position.y = 0.00
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.z = scale

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.text = text

        return marker

    def path_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = "selected_path"
        marker.id = 100
        marker.pose.orientation.w = 1.0

        if (
            not self.path_visible
            or self.selected_intent is None
        ):
            marker.action = Marker.DELETE
            return marker

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.008

        colors = {
            "DRAW_CIRCLE": (
                0.15,
                1.00,
                0.35,
            ),
            "DRAW_LINE": (
                0.20,
                0.72,
                1.00,
            ),
            "DRAW_SQUARE": (
                1.00,
                0.55,
                0.15,
            ),
        }

        color = colors[
            self.selected_intent
        ]

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        tcp = self.trajectories[
            self.selected_intent
        ]["tcp"]

        for xyz in tcp:
            point = Point()
            point.x = float(xyz[0])
            point.y = float(xyz[1])
            point.z = float(xyz[2])
            marker.points.append(point)

        return marker

    def publish_ui(self):
        markers = MarkerArray()

        task_text = (
            TASK_LABELS[
                self.selected_intent
            ]
            if self.selected_intent
            in TASK_LABELS
            else "PATH_HIDDEN"
        )

        short_state = self.execution_state

        user_text = {
            "DRAW_CIRCLE": "USER_JP_DRAW_A_CIRCLE",
            "DRAW_LINE": "USER_JP_DRAW_A_LINE",
            "DRAW_SQUARE": "USER_JP_DRAW_A_SQUARE",
            "STOP": "USER_JP_STOP",
            "UNKNOWN": "USER_JP_UNKNOWN_COMMAND",
            "NONE": "USER_WAITING",
        }.get(
            self.intent,
            "USER_OTHER_COMMAND",
        )

        inference_text = (
            "INFER_"
            + self.intent
            + "_"
            + format(
                self.confidence * 100.0,
                ".2f",
            )
            + "%"
        )

        plan_text = (
            "PLAN_" + task_text
            if self.selected_intent
            else "PLAN_NOT_SELECTED"
        )

        status_text = (
            "STATUS_"
            + short_state
            + "_"
            + self.result
        )

        lines = [
            (
                1,
                "AI_DRAWING_ARM_7R",
                1.01,
                0.031,
                (0.25, 0.82, 1.00, 1.00),
            ),
            (
                2,
                user_text,
                0.96,
                0.024,
                (1.00, 0.84, 0.25, 1.00),
            ),
            (
                3,
                "MODEL_CHAR_NGRAM_MLP",
                0.91,
                0.022,
                (0.72, 0.82, 1.00, 1.00),
            ),
            (
                4,
                inference_text,
                0.86,
                0.023,
                (0.78, 0.90, 1.00, 1.00),
            ),
            (
                5,
                plan_text,
                0.81,
                0.022,
                (0.58, 1.00, 0.68, 1.00),
            ),
            (
                6,
                status_text,
                0.76,
                0.024,
                (
                    0.35,
                    1.00,
                    0.45,
                    1.00,
                )
                if self.result == "PASS"
                else (
                    1.00,
                    0.62,
                    0.25,
                    1.00,
                ),
            ),
        ]

        for (
            marker_id,
            text,
            z,
            scale,
            color,
        ) in lines:
            markers.markers.append(
                self.text_marker(
                    marker_id,
                    text,
                    z,
                    scale,
                    color,
                )
            )

        markers.markers.append(
            self.path_marker()
        )

        self.marker_publisher.publish(
            markers
        )

        state = String()
        state.data = json.dumps(
            {
                "instruction": self.instruction,
                "intent": self.intent,
                "confidence": self.confidence,
                "selected_task": (
                    self.selected_intent
                ),
                "path_visible": (
                    self.path_visible
                ),
                "state": (
                    self.execution_state
                ),
                "result": self.result,
            },
            ensure_ascii=False,
        )

        self.state_publisher.publish(state)


def main():
    rclpy.init()
    node = AIMultiTaskExecutor()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
