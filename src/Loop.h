#pragma once
#include "utils.h"
#include <vector>

struct SDL_Window;
struct SDL_Renderer;
struct ImGuiContext;

/**
* @brief Main Loop
*/
class Loop {
private:
    const char* windowTitle;  // Window title
    int windowWidth;          // Window width
    int windowHeight;         // Window height
    SDL_Window* window;       // Reference to the SDL Window
    SDL_Renderer* renderer;   // Reference to the graphics interface

    ImGuiContext* imguiContext;   // ImGui context
    bool imguiInit;               // Flag to check if ImGui is initialized
    bool imguiInitRender;         // Flag to check if ImGui Renderer is initialized

    bool exit;          // Condition value for the continuous execution of the main loop
    Uint32 lastTime;    // Time from last time
    float deltaTime;    // Time from last frame

    bool isLeftClickMousePressed;   // If left click of mouse is pressed
    bool isRightClickMousePressed;  // If right click of mouse is pressed
    int lastMouseX;                 // Last click mouse position x
    int lastMouseY;                 // Last click mouse position y

    /**
    * @brief Finishes the main loop establishing "exit" boolean to false value
    */
    void quit();

    /**
    * @brief Manages keyboard, mouse, window events
    */
    void handleEvents();
    /**
    * @brief Updates the current alive entities
    */
    void update();
    /**
    * @brief Removes the entities marked as not alive
    */
    void refresh();
    /**
    * @brief Renders on screen the current alive entities
    */
    void render();

private:
    const float METERS_TO_PIXELS = 100.0f; // 1 m = 100 pixels
    const float GRAVITY = 9.8f;
    const float DAMPING = 0.98;
    const float RESTITUTION = 0.8f;
    const float H = 0.1f;
    const float K = 0.0f;
    const float REST_DENSITY = 0.0f;
    const float VISCOSITY = 0.0f;
    const float PARTICLES_MIN_RESTITUTION = 1.0f;
    const float PARTICLES_MAX_RESTITUTION = 0.0f;
    const int PARTICLES_MIN_TIME_LEFT = 1;
    const int PARTICLES_MAX_TIME = 10;
    const float MIN_GRAVITY = -20.0f;
    const float MAX_GRAVITY = 20.0f;
    const float MIN_RESTITUTION = 0.0f;
    const float MAX_RESTITUTION = 1.0f;
    const float SPH_MIN_H = 0.1f;
    const float SPH_MAX_H = 10.0f;
    const float SPH_MIN_K = 0.0f;
    const float SPH_MAX_K = 100.0f;
    const float SPH_MIN_REST_DENSITY = 0.0f;
    const float SPH_MAX_REST_DENSITY = 0.1f;
    const float SPH_MIN_VISCOSITY = 0.0f;
    const float SPH_MAX_VISCOSITY = 0.1f;

    Color backgroundCol;             // Background color
    float particleMass;              // Particles mass
    float particleDensity;           // Particles density
    float particlePressure;          // Particles pressure
    Color particleCol;               // Particles color
    float particleTimeLeft;          // Particles alive time
    std::vector<Particle> particles; // All particles

    int brushSize;          // Diameter which determines the quantity of particles

    Forces forces;  // Applied forces

    /**
    * @brief Renders simulation
    */
    void renderSimulation();

    /**
    * @brief Renders interface
    */
    void renderInterface();

public:
    /**
    * Constructor of the main loop
    */
    Loop();
    /**
    * Destructor of the main loop. Destroys the window, renderer, entities...
    */
    ~Loop();
    /**
    * Inits SDL Window and Renderer
    */
    bool init();
    /**
    * Runs the main loop
    */
    void run();
};
