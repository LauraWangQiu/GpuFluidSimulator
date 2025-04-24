#pragma once
#include <SDL_stdinc.h>

/**
* @brief Structure to store r, g, b and a channels for the color
*/
struct Color {
    Uint8 r, g, b, a;
};

/**
* @brief Structure to store particle information like position,
* velocity, color, time left
*/
struct Particle {
    float posX, posY;   // position
    float velX, velY;   // velocity
    float radius;       // radius
    Color color;        // color
    float timeLeft;     // time left until disappear
};

/**
* @brief Structure to store gravity parameters
*/
struct GravityParams {
    float gravityForce;
};

/**
* @brief Structure to store different forces information
*/
struct Forces {
    GravityParams gravityParams;
    // ...
};