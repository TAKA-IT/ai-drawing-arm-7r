cmake_minimum_required(VERSION 3.8)
project(robot_drawing)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/ai_task_executor_7r.py
    scripts/circle7_validator.py
    scripts/dls_ik7_solver.py
    scripts/fk7_validator.py
    scripts/generate_multitask_trajectories_7r.py
    scripts/jacobian7_validator.py
    scripts/joint_overlay_7r.py
    scripts/multiple_ik7_validator.py
    scripts/train_task_intent_7r.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    scripts/package_paths.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    config
    launch
    scripts
    urdf
    worlds
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
