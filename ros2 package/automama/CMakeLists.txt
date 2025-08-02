cmake_minimum_required(VERSION 3.5)
project(automama)

find_package(ament_cmake REQUIRED)

# Add this line to ensure Python scripts are installed
install(
  PROGRAMS
    scripts/test.py
    scripts/throttle.py
  DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY launch worlds scripts
    DESTINATION share/${PROJECT_NAME}/
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

ament_package()
