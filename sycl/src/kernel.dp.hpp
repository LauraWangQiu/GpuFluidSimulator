#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils.h"

float poly6Kernel(float r, float h);

void computeDensityPressure(Particle* particles, int numParticles, float restDensity, float h, float k,
                            const sycl::nd_item<3> &item_ct1);

sycl::float2 spikyGradient(float dx, float dy, float r, float h);

void computePressureViscosityForces(Particle* particles, int numParticles, float h, float viscosity,
                                    const sycl::nd_item<3> &item_ct1);

void applyGravityForce(Particle* particles, int numParticles, float deltaTime, float gravityForce,
                       const sycl::nd_item<3> &item_ct1);

void applyDamping(Particle* particles, int numParticles, float damping, const sycl::nd_item<3> &item_ct1);

void applyCollisions(Particle* particles, int numParticles, int windowWidth, int windowHeight,
                                float restitution, const sycl::nd_item<3> &item_ct1);

void updateParticles_kernels(Particle* particles, int numParticles, float deltaTime, int windowWidth, int windowHeight,
                             Forces forces, int lastMouseX, int lastMouseY);
