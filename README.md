# GPU Fluid Simulator

> Author: Laura Wang Qiu

## About

> [!IMPORTANT] To try it out you will need a NVIDIA GPU!

Enjoy this 2D particles simulation made in C++ with CUDA. You can change parameters to visualize the behavior of the particles depending on the applied forces:

- Gravity
- Collision restitution
- Smoothed-Particle Hydrodynamics

Preview:

<div style="text-align: center;">

![Gpu-Fluid-Simulator-Laura-Wang-Qiu](https://github.com/user-attachments/assets/a839cadc-af04-4bbe-bc23-9fe45cadc733)

</div>

## State of the Art

Fluid systems are widely used in games, industrial simulations, and scientific research in astrophysics, among other fields. These systems can be modeled using two main approaches:

- **Eulerian**: This approach focuses on fixed points in space and observes how the fluid flows through these points. It is commonly used for grid-based simulations, where the fluid's properties (e.g., velocity, pressure) are calculated at discrete points in a fixed grid.

- **Lagrangian**: This approach tracks individual particles as they move through space, making it ideal for particle-based simulations. It provides a more intuitive representation of fluid dynamics by following the motion of the particles themselves.

This project adopts the **Lagrangian** approach, using Smoothed-Particle Hydrodynamics (SPH) to simulate the behavior of fluids. SPH is a particle-based method that calculates fluid properties by averaging over neighboring particles, ensuring smooth and realistic simulations.

## Tools

- [SDL2](https://www.libsdl.org/): window and graphics
- [ImGui](https://www.dearimgui.com/): interface
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): as the GPU programming language used
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html): used SYCLomatic to migrate CUDA code to SYCL
