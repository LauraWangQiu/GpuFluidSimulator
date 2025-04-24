#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.h"

__global__ void applyBasics(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight);

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, int windowWidth,
                                  int windowHeight, float gravityForce);

__global__ void applyXForce(Particle* particles, int numParticles, float deltaTime, int windowWidth,
                            int windowHeight/*, XParams params*/);

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY);
