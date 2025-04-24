#include "kernel.cuh"
#include <math.h>

__global__ void applyBasics(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    particles[idx].timeLeft -= deltaTime;

    // Limits
    if (particles[idx].posY > windowHeight - particles[idx].radius - 1) {
        particles[idx].posY = windowHeight - particles[idx].radius - 1;
        particles[idx].velY = 0.0f;
    }
    if (particles[idx].posX < particles[idx].radius + 1) {
        particles[idx].posX = particles[idx].radius + 1;
        particles[idx].velX = 0.0f;
    }
    if (particles[idx].posX > windowWidth - particles[idx].radius - 1) {
        particles[idx].posX = windowWidth - particles[idx].radius - 1;
        particles[idx].velX = 0.0f;
    }

    // Collisions with other particles
    for (int i = 0; i < numParticles; ++i) {
        if (i == idx) continue;

        float dx = particles[idx].posX - particles[i].posX;
        float dy = particles[idx].posY - particles[i].posY;
        float distanceSquared = dx * dx + dy * dy;
        float minDistance = 2.0f * particles[idx].radius;

        if (distanceSquared < minDistance * minDistance) {
            float distance = sqrtf(distanceSquared);
            float overlap = minDistance - distance;

            float nx = dx / distance;
            float ny = dy / distance;

            particles[idx].posX += nx * overlap * 0.5f;
            particles[idx].posY += ny * overlap * 0.5f;

            particles[i].posX -= nx * overlap * 0.5f;
            particles[i].posY -= ny * overlap * 0.5f;

            float relativeVelX = particles[idx].velX - particles[i].velX;
            float relativeVelY = particles[idx].velY - particles[i].velY;
            float dotProduct = (relativeVelX * nx + relativeVelY * ny);

            particles[idx].velX -= dotProduct * nx;
            particles[idx].velY -= dotProduct * ny;

            particles[i].velX += dotProduct * nx;
            particles[i].velY += dotProduct * ny;
        }
    }
}

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, int windowWidth,
                                  int windowHeight, float gravityForce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    particles[idx].velY += gravityForce * deltaTime;
    particles[idx].posX += particles[idx].velX * deltaTime;
    particles[idx].posY += particles[idx].velY * deltaTime;
}

__global__ void applyXForce(Particle* particles, int numParticles, float deltaTime, int windowWidth,
                            int windowHeight/*, XParams params*/) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // ...
}

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY) {
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    applyGravityForce<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, windowWidth,
                                                          windowHeight, forces.gravityParams.gravityForce);
    applyBasics<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, windowWidth, windowHeight);

    cudaMemcpy(particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
}
