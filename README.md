## 🖥️ Simulation Environment – Automama Campus World (Gazebo)

To enable safe, repeatable, and low-cost testing of the autonomous system, we developed a **custom Gazebo simulation environment** that replicates our university campus. This simulated world is used to test and validate Automama’s full autonomous stack, including **point-to-point navigation**, **control**, and **perception modules**.

---

### 🏙️ Environment Overview

- ✅ **Custom campus world** built in Gazebo using blueprints, maps, and satellite references
- ✅ **Automama vehicle model** with realistic physics, kinematics, and sensor integrations
- ⚠️ **Under development**: Static campus layout mostly complete; environment still lacks:
  - Dynamic actors (pedestrians, traffic vehicles)
  - Advanced road elements (signals, curbs, signage)
  - Photorealistic textures and materials

---

### 🚗 Vehicle Simulation

- The AGV Automama is modeled as a **4-wheeled differential or ackermann-drive vehicle**
- Integrated with:
  - Realistic **vehicle physics**
  - Simulated **rotary encoder** and **brake actuator**
  - Optional **camera, IMU, and LiDAR** plugins for perception testing
- Tuned for accurate **steering, acceleration, and braking response**

---

### 🧠 Stack Integration

The **control and perception stacks** of Automama have been successfully deployed and tested in this Gazebo world.

#### Tested Modules:
- ✅ Low-level control (steering, motor, braking)
- ✅ Sensor fusion (IMU, encoder)
- ✅ Point-to-point navigation planner
- ✅ Basic obstacle avoidance logic (using simulated LiDAR or depth input)

---

### 🚧 Roadmap & To-Dos

| Feature                        | Status         |
|-------------------------------|----------------|
| Static campus layout           | ✅ Completed    |
| Realistic AGV model            | ✅ Completed    |
| Perception testing             | ✅ In progress  |
| Traffic actors (cars, people)  | 🔜 Pending      |
| Road furniture & signage       | 🔜 Pending      |
| Photorealistic material maps   | 🔜 Pending      |
| ROS 2 integration              | ✅ Functional   |

---

### 🖼️ Screenshots & Demo Videos

> *(Include screen captures or short GIFs here from Gazebo showing the Automama navigating the world)*

---

### 📂 Folder Structure (Suggested)

automama_simulation/
├── models/
│ ├── automama_vehicle/
│ └── campus_assets/
├── worlds/
│ └── automama_campus.world
├── launch/
│ └── simulation_launch.py
├── config/
│ └── vehicle_params.yaml
├── rviz/
│ └── automama_nav.rviz
└── README.md
---

---

### 📚 Dependencies

- Gazebo (compatible with Ignition or Classic)
- ROS 2 Humble / Iron
- `gazebo_ros_pkgs`
- Custom plugins (for actuator/sensor simulation)

---

### 👥 Contributors

- Sim development by: [Your Simulation Team or Names]
- World modeling & physics: [Contributors]
- Vehicle model & URDF: [Contributors]

---

