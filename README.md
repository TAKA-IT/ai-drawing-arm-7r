# AI Drawing Arm 7R

A ROS 2 based 7-DOF drawing robot that classifies short natural-language commands and executes validated TCP trajectories.

## Motivation

This project explores how a digital instruction can be converted into a physical drawing action.

One possible future application is asynchronous communication between people living in different locations. For example, a child could request a simple shape and a robot located with grandparents could draw it on paper. The drawing could then be extended with handwritten notes or additional artwork.

This scenario is an application hypothesis. The project has not been evaluated with users or deployed in a real household.

## Supported commands

- `円を描いて` / `draw a circle`
- `直線を描いて` / `draw a straight line`
- `四角を描いて` / `draw a square`

## Runtime AI

The system uses a PyTorch character 1-3 gram MLP classifier.

Supported intents:

- `DRAW_CIRCLE`
- `DRAW_LINE`
- `DRAW_SQUARE`
- `STOP`
- `UNKNOWN`

The model classifies the task intent and selects a validated trajectory. It does not directly generate arbitrary trajectories.

## Drawing tasks

| Task | Geometry | Points |
|---|---:|---:|
| Circle | Radius 25 mm | 73 |
| Line | Length 40 mm | 41 |
| Square | Side 35 mm | 65 |

## Implemented features

- Single-arm 7-revolute-joint robot
- ROS 2 Humble and Gazebo integration
- Forward kinematics validation
- Numerical 6x7 Jacobian
- Damped least-squares inverse kinematics
- Multiple IK solutions
- Fixed-orientation TCP trajectories
- Dynamic J1-J7 visualization
- Unsupported-command safety gate
- Safe retract motion after each task
- Conceptual bearing-layout visualization

## AI and control pipeline

```text
Natural-language command
  -> character n-gram features
  -> PyTorch MLP intent classification
  -> validated task selection
  -> DLS-IK trajectory
  -> ROS 2 FollowJointTrajectory
```

## Safe retract motion

The drawing plane is approximately `X = 0.974996714 m`.

The shared safe-home position is approximately:

```text
X = 0.924996714 m
Y = 0.000000000 m
Z = 0.599990597 m
```

This provides approximately 50 mm of clearance from the drawing plane.

## Build

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/TAKA-IT/ai-drawing-arm-7r.git robot_drawing
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/robot_drawing/requirements.txt
colcon build --packages-select robot_drawing
source install/setup.bash
```

Replace `https://github.com/TAKA-IT/ai-drawing-arm-7r.git` after creating the public GitHub repository.

## Launch

```bash
ros2 launch robot_drawing robot_gazebo_7r.launch.py
```

In another terminal:

```bash
ros2 run robot_drawing ai_task_executor_7r.py
```

Send a command:

```bash
ros2 topic pub --once /input_text std_msgs/msg/String "{data: 円を描いて}"
```

## Validation

```bash
ros2 run robot_drawing fk7_validator.py
ros2 run robot_drawing jacobian7_validator.py
ros2 run robot_drawing multiple_ik7_validator.py
```

## Limitations

- This is a simulation-oriented project, not a production robot.
- The intent classifier is task-specific, not a general-purpose language model.
- The bearing geometry is conceptual.
- Force control, collision detection and functional-safety certification are not implemented.
- Motors, reducers, brakes, encoders and cable routing are not fully packaged.
- The remote-family communication scenario has not undergone user testing.

## Privacy

This repository is intended to contain no personal names, organizational affiliations, personal email addresses, credentials or access tokens.

The maintainer value in `package.xml` uses the reserved `.invalid` domain and is not a contact address.

## License

No license is granted at this time. Public availability does not itself grant permission to copy, modify or redistribute the code.
