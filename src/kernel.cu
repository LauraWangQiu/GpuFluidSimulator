#include "kernel.cuh"
#include <math.h>

__device__ float poly6Kernel(float r, float h) {
    if (r >= 0 && r <= h) {
        float hr2 = h * h - r * r;
        return (315.0f / (64.0f * M_PI * powf(h, 9))) * hr2 * hr2 * hr2;
    }
    return 0.0f;
}

__global__ void computeDensityPressure(Particle* particles, int numParticles, float restDensity, float h, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    Particle& p = particles[i];
    p.density = 0.0f;

    for (int j = 0; j < numParticles; j++) {
        float dx = p.posX - particles[j].posX;
        float dy = p.posY - particles[j].posY;
        float r = sqrtf(dx * dx + dy * dy);

        p.density += particles[j].mass * poly6Kernel(r, h);
    }

    p.pressure = k * (p.density - restDensity);
}

__device__ float2 spikyGradient(float dx, float dy, float r, float h) {
    float factor = -45.0f / (M_PI * powf(h, 6)) * powf(h - r, 2);
    return make_float2(factor * dx / r, factor * dy / r);
}

__global__ void computeForces(Particle* particles, int numParticles, float h, float viscosity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];
    float fx = 0.0f, fy = 0.0f;

    for (int j = 0; j < numParticles; j++) {
        if (idx == j) continue;

        float dx = p.posX - particles[j].posX;
        float dy = p.posY - particles[j].posY;
        float r = sqrtf(dx * dx + dy * dy);

        if (r < h && r > 0.0001f) {
            // Pressure
            float2 grad = spikyGradient(dx, dy, r, h);
            float pressureTerm = (p.pressure + particles[j].pressure) / (2.0f * particles[j].density);
            fx += -particles[j].mass * pressureTerm * grad.x;
            fy += -particles[j].mass * pressureTerm * grad.y;

            // Viscosity (Laplacian of velocity)
            float velDiffX = particles[j].velX - p.velX;
            float velDiffY = particles[j].velY - p.velY;
            float laplacian = 45.0f / (M_PI * powf(h, 6)) * (h - r);
            fx += viscosity * particles[j].mass * velDiffX / particles[j].density * laplacian;
            fy += viscosity * particles[j].mass * velDiffY / particles[j].density * laplacian;
        }
    }

    p.velX += fx / p.density;
    p.velY += fy / p.density;
}

__global__ void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];
    p.velY += gravityForce * deltaTime;   // v=v0​+a⋅Δt, a=gravityForce
    p.posX += p.velX * deltaTime;
    p.posY += p.velY * deltaTime;
}

__global__ void applyCollisions(Particle* particles, int numParticles, int windowWidth, int windowHeight,
                                float restitution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];

    // Collision with bottom edge
    if (p.posY > windowHeight - p.radius - 1) {
        p.posY = windowHeight - p.radius - 1;
        p.velY *= -restitution;
    }

    // Collision with top edge
    if (p.posY < p.radius + 1) {
        p.posY = p.radius + 1;
        p.velY *= -restitution;
    }

    // Collision with left edge
    if (p.posX < p.radius + 1) {
        p.posX = p.radius + 1;
        p.velX *= -restitution;
    }

    // Collision with right edge
    if (p.posX > windowWidth - p.radius - 1) {
        p.posX = windowWidth - p.radius - 1;
        p.velX *= -restitution;
    }

    // Collisions with other particles
    for (int i = 0; i < numParticles; ++i) {
        if (i == idx) continue;

        float dx = p.posX - particles[i].posX;
        float dy = p.posY - particles[i].posY;
        float distanceSquared = dx * dx + dy * dy;
        float minDistance = p.radius + particles[i].radius;

        if (distanceSquared < minDistance * minDistance) {
            float distance = sqrtf(distanceSquared);
            float overlap = minDistance - distance;

            float nx = dx / distance;
            float ny = dy / distance;

            p.posX += nx * overlap * 0.5f;
            p.posY += ny * overlap * 0.5f;

            particles[i].posX -= nx * overlap * 0.5f;
            particles[i].posY -= ny * overlap * 0.5f;

            float relativeVelX = p.velX - particles[i].velX;
            float relativeVelY = p.velY - particles[i].velY;
            float dotProduct = (relativeVelX * nx + relativeVelY * ny);

            float massSum = p.mass + particles[i].mass;

            p.velX -= (2.0f * particles[i].mass / massSum) * dotProduct * nx;
            p.velY -= (2.0f * particles[i].mass / massSum) * dotProduct * ny;

            particles[i].velX += (2.0f * p.mass / massSum) * dotProduct * nx;
            particles[i].velY += (2.0f * p.mass / massSum) * dotProduct * ny;
        }
    }
}

__global__ void applyDamping(Particle* particles, int numParticles, float damping) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    Particle& p = particles[idx];
    p.velX *= damping;
    p.velY *= damping;
}

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY) {
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // https://es.wikipedia.org/wiki/Smoothed-particle_hydrodynamics
    computeDensityPressure<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, forces.sphParams.restDensity,
                                                               forces.sphParams.h, forces.sphParams.k);
    computeForces<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, forces.sphParams.h,
                                                      forces.sphParams.viscosity);

    applyGravityForce<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime,
                                                          forces.gravityParams.gravityForce);

    applyDamping<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, forces.damping);

    applyCollisions<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, windowWidth, windowHeight,
                                                        forces.collisionParams.restitution);
    
    cudaMemcpy(particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
}
