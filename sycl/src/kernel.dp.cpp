#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kernel.dp.hpp"
#include <math.h>

// Kernel function (smoothing function)
// Only for density calculations
float poly6Kernel(float r, float h) {
    // W(r, h) = (315 / (64 ⋅ π * h^9)) * (h^2 - r^2)^3
    if (r >= 0 && r <= h) {
        float hr2 = h * h - r * r;
        return (315.0f / (64.0f * M_PI * powf(h, 9))) * hr2 * hr2 * hr2;
    }
    return 0.0f;
}

void computeDensityPressure(Particle* particles, int numParticles, float restDensity, float h, float k,
                            const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= numParticles) return;

    Particle& p_i = particles[i];
    p_i.density = 0.0f;

    // Calculate current particle density
    // ρ_i = p(r_i) = sum_j(m_j ⋅ ρ_j / ρ_j ⋅ W(|r_i - r_j|, h) = sum_j(m_j ⋅ W(|r_i - r_j|, h)
    for (int j = 0; j < numParticles; j++) {
        Particle& p_j = particles[j];

        // Calculate distance between current particle and others
        float dx = p_i.posX - p_j.posX;
        float dy = p_i.posY - p_j.posY;
        float r = sycl::sqrt(dx * dx + dy * dy);

        // Apply kernel function
        p_i.density += p_j.mass * poly6Kernel(r, h);
    }

    // Calculate current particle pressure
    // P_i = k ⋅ (ρ_i - ρ_0)
    p_i.pressure = k * (p_i.density - restDensity);
}

// Only for pressure force calculations
sycl::float2 spikyGradient(float dx, float dy, float r, float h) {
    float factor = -45.0f / (M_PI * powf(h, 6)) * powf(h - r, 2);
    return sycl::float2(factor * dx / r, factor * dy / r);
}

void computePressureViscosityForces(Particle* particles, int numParticles, float h, float viscosity,
                                    const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= numParticles) return;

    Particle& p_i = particles[i];
    float fx = 0.0f, fy = 0.0f;

    for (int j = 0; j < numParticles; j++) {
        if (i == j) continue;

        Particle& p_j = particles[j];

        // Calculate distance between current particle and others
        float dx = p_i.posX - p_j.posX;
        float dy = p_i.posY - p_j.posY;
        float r = sycl::sqrt(dx * dx + dy * dy);

        if (r < h && r > 1e-6f) {
            // Pressure force
            sycl::float2 grad = spikyGradient(dx, dy, r, h);
            float pressureTerm = (p_i.pressure + p_j.pressure) / (2.0f * p_j.density);
            fx += -p_j.mass * pressureTerm * grad.x();
            fy += -p_j.mass * pressureTerm * grad.y();

            // Viscosity force (Laplacian of velocity)
            float velDiffX = p_j.velX - p_i.velX;
            float velDiffY = p_j.velY - p_i.velY;
            float laplacian = 45.0f / (M_PI * powf(h, 6)) * (h - r);
            fx += viscosity * p_j.mass * velDiffX / p_j.density * laplacian;
            fy += viscosity * p_j.mass * velDiffY / p_j.density * laplacian;
        }
    }

    p_i.velX += (fx / p_i.density);
    p_i.velY += (fy / p_i.density);
}

void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce,
                       const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= numParticles) return;

    Particle& p_i = particles[i];
    p_i.velY += gravityForce * deltaTime;   // v = v0​ + a ⋅ Δt, a = gravityForce
    p_i.posX += p_i.velX * deltaTime;
    p_i.posY += p_i.velY * deltaTime;
}

void applyCollisions(Particle* particles, int numParticles, int windowWidth, int windowHeight,
                                float restitution, const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= numParticles) return;

    Particle& p_i = particles[i];

    // Collision with bottom edge
    if (p_i.posY > windowHeight - p_i.radius - 1) {
        p_i.posY = windowHeight - p_i.radius - 1;
        p_i.velY *= -restitution;
    }

    // Collision with top edge
    if (p_i.posY < p_i.radius + 1) {
        p_i.posY = p_i.radius + 1;
        p_i.velY *= -restitution;
    }

    // Collision with left edge
    if (p_i.posX < p_i.radius + 1) {
        p_i.posX = p_i.radius + 1;
        p_i.velX *= -restitution;
    }

    // Collision with right edge
    if (p_i.posX > windowWidth - p_i.radius - 1) {
        p_i.posX = windowWidth - p_i.radius - 1;
        p_i.velX *= -restitution;
    }

    // Collisions with other particles
    for (int j = 0; j < numParticles; ++j) {
        if (i == j) continue;

        Particle& p_j = particles[j];

        float dx = p_i.posX - p_j.posX;
        float dy = p_i.posY - p_j.posY;
        float distanceSquared = dx * dx + dy * dy;
        float minDistance = p_i.radius + p_j.radius;

        if (distanceSquared < minDistance * minDistance) {
            float distance = sycl::sqrt(distanceSquared);
            float overlap = minDistance - distance;

            float nx = dx / distance;
            float ny = dy / distance;

            p_i.posX += nx * overlap * 0.5f;
            p_i.posY += ny * overlap * 0.5f;

            p_j.posX -= nx * overlap * 0.5f;
            p_j.posY -= ny * overlap * 0.5f;

            float relativeVelX = p_i.velX - p_j.velX;
            float relativeVelY = p_i.velY - p_j.velY;
            float dotProduct = (relativeVelX * nx + relativeVelY * ny);

            float massSum = p_i.mass + p_j.mass;

            p_i.velX -= (2.0f * p_j.mass / massSum) * dotProduct * nx;
            p_i.velY -= (2.0f * p_j.mass / massSum) * dotProduct * ny;

            p_j.velX += (2.0f * p_i.mass / massSum) * dotProduct * nx;
            p_j.velY += (2.0f * p_i.mass / massSum) * dotProduct * ny;
        }
    }
}

void applyDamping(Particle* particles, int numParticles, float damping, const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= numParticles) return;

    Particle& p = particles[i];
    p.velX *= damping;
    p.velY *= damping;
}

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.in_order_queue();
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    q_ct1.memcpy(d_particles, particles, numParticles * sizeof(Particle)).wait();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    float maxStep = 0.035f;
    int substeps = (int)ceil(deltaTime / maxStep);
    float subDeltaTime = deltaTime / substeps;

    for (int step = 0; step < substeps; ++step) {
        // https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics
        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                computeDensityPressure(d_particles, numParticles, forces.sphParams.restDensity, forces.sphParams.h,
                                       forces.sphParams.k, item_ct1);
            });

        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                computePressureViscosityForces(d_particles, numParticles, forces.sphParams.h,
                                               forces.sphParams.viscosity, item_ct1);
            });

        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                applyGravityForce(d_particles, numParticles, subDeltaTime, forces.gravityParams.gravityForce, item_ct1);
            });

        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                applyDamping(d_particles, numParticles, forces.damping, item_ct1);
            });

        for (int i = 0; i < 4; ++i) {
            /*
            DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
            */
            q_ct1.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock),
                                  sycl::range<3>(1, 1, threadsPerBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    applyCollisions(d_particles, numParticles, windowWidth, windowHeight,
                                    forces.collisionParams.restitution, item_ct1);
                });
        }
    }

    q_ct1.memcpy(particles, d_particles, numParticles * sizeof(Particle)).wait();
    dpct::dpct_free(d_particles, q_ct1);
}
