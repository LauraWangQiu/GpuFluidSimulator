#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.h"

__device__ float poly6Kernel(float r, float h);

__global__ void computeDensityPressure(Particle* particles, int numParticles, float restDensity, float h, float k);

__device__ float2 spikyGradient(float dx, float dy, float r, float h);

__global__ void computePressureViscosityForces(Particle* particles, int numParticles, float deltaTime, float h,
                                               float viscosity);

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce);

__global__ void applyDamping(Particle* particles, int numParticles, float damping);

__global__ void applyCollisions(Particle* particles, int numParticles, int windowWidth, int windowHeight,
                                float restitution);

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY);
