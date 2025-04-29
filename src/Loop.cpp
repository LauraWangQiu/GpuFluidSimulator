#include "Loop.h"
#include "defs.h"
#include "kernel.cuh"
#include <string>
#include <SDL.h>
#include <SDL_render.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
#include <cuda_runtime.h>
using namespace std;

Loop::Loop()
    : window(nullptr), renderer(nullptr), imguiContext(nullptr), imguiInit(false), imguiInitRender(false), exit(false),
      lastTime(0), deltaTime(0.0f), isLeftClickMousePressed(false), isRightClickMousePressed(false), lastMouseX(0),
      lastMouseY(0) {

    // Window
    windowTitle = "Simulacion de fluidos - Laura Wang Qiu";
    windowWidth = 640;
    windowHeight = 480;

    // Background Color
    backgroundCol.r = 0;
    backgroundCol.g = 0;
    backgroundCol.b = 0;
    backgroundCol.a = 255;

    // Particles Mass
    particleMass = 1.0f;
    // Particles Density
    particleDensity = 1000.0f;
    // Particles Pressure
    particlePressure = 0.0f;

    // Particles Color
    particleCol.r = 255;
    particleCol.g = 255;
    particleCol.b = 255;
    particleCol.a = 255;
    // Particles Time Alive
    particleTimeLeft = 1.0f;

    // Size of pen to draw particles
    brushSize = 2;

    // Forces constants
    forces.damping = DAMPING;
    forces.gravityParams.gravityForce = GRAVITY * METERS_TO_PIXELS;
    forces.collisionParams.restitution = RESTITUTION;
    forces.sphParams.h = H;
    forces.sphParams.k = K;
    forces.sphParams.restDensity = REST_DENSITY;
    forces.sphParams.viscosity = VISCOSITY;
}

Loop::~Loop() {
    if (renderer != nullptr) SDL_DestroyRenderer(renderer);
    if (window != nullptr) SDL_DestroyWindow(window);

    if (imguiInitRender) ImGui_ImplSDLRenderer2_Shutdown();
    if (imguiInit) ImGui_ImplSDL2_Shutdown();
    if (imguiContext != nullptr) ImGui::DestroyContext();

    SDL_Quit();

    window = nullptr;
    renderer = nullptr;
}

bool Loop::init() {

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        debugLog(("SDL_Init Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    Uint32 windowFlags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    window = SDL_CreateWindow(windowTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight,
                              windowFlags);
    if (window == nullptr) {
        debugLog(("SDL_CreateWindow Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    Uint32 rendererFlags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC;
    renderer = SDL_CreateRenderer(window, -1, rendererFlags);
    if (renderer == nullptr) {
        debugLog(("SDL_CreateRenderer Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    IMGUI_CHECKVERSION();
    imguiContext = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.WantCaptureKeyboard = true;
    io.WantCaptureMouse = true;
    ImGui::StyleColorsDark();
    imguiInit = ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    if (!imguiInit) {
        debugLog("ImGui_ImplSDL2_InitForSDLRenderer Error");
        return false;
    }
    imguiInitRender = ImGui_ImplSDLRenderer2_Init(renderer);
    if (!imguiInit) {
        debugLog("ImGui_ImplSDLRenderer2_Init Error");
        return false;
    }

    return true;
}

void Loop::run() {
    lastTime = SDL_GetTicks();

    while (!exit) {
        Uint32 currentTime = SDL_GetTicks();
        deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        handleEvents();
        update();
        refresh();
        render();
    }
}

void Loop::handleEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        ImGui_ImplSDL2_ProcessEvent(&event);

        ImGuiIO& io = ImGui::GetIO();
        if ((io.WantCaptureMouse &&
             (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP || event.type == SDL_MOUSEMOTION)) ||
            (io.WantCaptureKeyboard && (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP))) {
            continue;
        }

        switch (event.type) {
        case SDL_WINDOWEVENT:
            switch (event.window.event) {
            case SDL_WINDOWEVENT_CLOSE: quit(); break;
            case SDL_WINDOWEVENT_RESIZED:
                windowWidth = event.window.data1;
                windowHeight = event.window.data2;
                break;
            default: break;
            }
            break;
        case SDL_KEYDOWN:
            switch (event.key.keysym.sym) {
            case SDLK_ESCAPE: quit(); break;
            default: break;
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                isLeftClickMousePressed = true;
                lastMouseX = event.button.x;
                lastMouseY = event.button.y;
            }
            else if (event.button.button == SDL_BUTTON_RIGHT) {
                isRightClickMousePressed = true;
                lastMouseX = event.button.x;
                lastMouseY = event.button.y;
            }
            break;
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                isLeftClickMousePressed = false;
            }
            else if (event.button.button == SDL_BUTTON_RIGHT) {
                isRightClickMousePressed = false;
            }
            break;
        case SDL_MOUSEMOTION:
            lastMouseX = event.motion.x;
            lastMouseY = event.motion.y;
            break;
        case SDL_QUIT: quit(); break;
        default: break;
        }
    }
}

void Loop::update() {
    for (auto& p : particles) {
        if (p.timeLeft != PARTICLES_MAX_TIME) {
            p.timeLeft -= deltaTime;
        }
    }

    if (isLeftClickMousePressed) {
        Particle p;
        p.posX = lastMouseX;
        p.posY = lastMouseY;
        p.velX = 0.0f;
        p.velY = 0.0f;
        p.radius = brushSize / 2;
        p.mass = particleMass;
        p.density = particleDensity;
        p.color = particleCol;
        p.timeLeft = particleTimeLeft;

        particles.push_back(p);
    }

    if (isRightClickMousePressed) {
        for (auto& p : particles) {
            float dx = p.posX - lastMouseX;
            float dy = p.posY - lastMouseY;
            if ((dx * dx + dy * dy) < (p.radius * p.radius)) {
                p.timeLeft = -1;
            }
        }
    }

    updateParticles_kernels(particles.data(), (int)particles.size(), deltaTime, windowWidth, windowHeight, forces,
                            lastMouseX, lastMouseY);
}

void Loop::refresh() {
    particles.erase(std::remove_if(particles.begin(), particles.end(),
                                   [this](const Particle& p) {
                                       return (p.timeLeft <= 0 && p.timeLeft != PARTICLES_MAX_TIME) || p.posX < 0 ||
                                           p.posX >= windowWidth || p.posY < 0 || p.posY >= windowHeight;
                                   }),
                    particles.end());
}

void Loop::render() {
    if (renderer != nullptr) {
        SDL_SetRenderDrawColor(renderer, backgroundCol.r, backgroundCol.g, backgroundCol.b, backgroundCol.a);
        SDL_RenderClear(renderer);

        renderSimulation();

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        renderInterface();

        ImGui::Render();

        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }
}

void Loop::quit() { exit = true; }

void Loop::renderSimulation() {
    for (auto it = particles.begin(); it != particles.end(); ++it) {
        SDL_SetRenderDrawColor(renderer, it->color.r, it->color.g, it->color.b, it->color.a);

        for (int y = (int)-it->radius; y <= (int)it->radius; ++y) {
            for (int x = (int)-it->radius; x <= (int)it->radius; ++x) {
                int radiusSquared = it->radius * it->radius;
                if (x * x + y * y <= radiusSquared) {
                    SDL_RenderDrawPointF(renderer, it->posX + x, it->posY + y);
                }
            }
        }
    }
}

void Loop::renderInterface() {
    ImGui::SetNextWindowPos(ImVec2((float)windowWidth - windowWidth / 4, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2((float)windowWidth / 4, (float)windowHeight), ImGuiCond_Always);

    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    ImGui::PushTextWrapPos();

    ImGui::Text("Particles: %d", (int)particles.size());
    ImGui::Text("FPS: %.1f", 1.0f / deltaTime);
    ImGui::Separator();

    if (ImGui::Button("Clear")) {
        particles.clear();
    }

    if (ImGui::Button("Generate Particle Sea")) {
        generateParticleSea();
    }

    if (ImGui::CollapsingHeader("General Settings")) {
        ImGui::Text("Background Color");
        static float color[4] = {backgroundCol.r / 255.0f, backgroundCol.g / 255.0f, backgroundCol.b / 255.0f,
                                 backgroundCol.a / 255.0f};
        if (ImGui::ColorPicker4("Color ", color)) {
            backgroundCol.r = static_cast<Uint8>(color[0] * 255);
            backgroundCol.g = static_cast<Uint8>(color[1] * 255);
            backgroundCol.b = static_cast<Uint8>(color[2] * 255);
            backgroundCol.a = static_cast<Uint8>(color[3] * 255);
        }
    }

    if (ImGui::CollapsingHeader("Particle Settings")) {
        ImGui::Text("Particles Color");
        static float pColor[4] = {particleCol.r / 255.0f, particleCol.g / 255.0f, particleCol.b / 255.0f,
                                  particleCol.a / 255.0f};
        if (ImGui::ColorPicker4("Color", pColor)) {
            particleCol.r = static_cast<Uint8>(pColor[0] * 255);
            particleCol.g = static_cast<Uint8>(pColor[1] * 255);
            particleCol.b = static_cast<Uint8>(pColor[2] * 255);
            particleCol.a = static_cast<Uint8>(pColor[3] * 255);
        }

        ImGui::Separator();

        ImGui::Text("Brush Size");
        ImGui::SliderInt("  ", &brushSize, 2, 50, "%d");

        ImGui::Separator();

        ImGui::Text("Particle Lifetime");
        ImGui::SliderFloat(" ", &particleTimeLeft, 0.1f, PARTICLES_MAX_TIME, "%.1f seconds");
        if (particleTimeLeft == PARTICLES_MAX_TIME) {
            ImGui::SameLine();
            ImGui::Text("inf");
        }
    }

    if (ImGui::CollapsingHeader("Gravity Params")) {
        ImGui::Text("Gravity Force");
        ImGui::SliderFloat("   ", &forces.gravityParams.gravityForce, MIN_GRAVITY * METERS_TO_PIXELS, MAX_GRAVITY * METERS_TO_PIXELS, "%.1f pixels/s2");
    }

    if (ImGui::CollapsingHeader("Collision Params")) {
        ImGui::Text("Restitution");
        ImGui::SliderFloat("    ", &forces.collisionParams.restitution, MIN_RESTITUTION, MAX_RESTITUTION, "%.1f");
    }

    if (ImGui::CollapsingHeader("SPH Params")) {
        ImGui::Text("Smoothing radius");
        ImGui::SliderFloat("     ", &forces.sphParams.h, SPH_MIN_H, SPH_MAX_H, "%.3f");

        ImGui::Separator();

        ImGui::Text("Stiffness");
        ImGui::SliderFloat("       ", &forces.sphParams.k, SPH_MIN_K, SPH_MAX_K, "%.3f");

        ImGui::Separator();

        ImGui::Text("Rest Density");
        ImGui::SliderFloat("      ", &forces.sphParams.restDensity, SPH_MIN_REST_DENSITY, SPH_MAX_REST_DENSITY, "%.3f");

        ImGui::Separator();

        ImGui::Text("Viscosity");
        ImGui::SliderFloat("        ", &forces.sphParams.viscosity, SPH_MIN_VISCOSITY, SPH_MAX_VISCOSITY, "%.3f");
    }

    ImGui::PopTextWrapPos();
    ImGui::End();
}

#define MAX_PARTICLES_SEA 4000 // Adjust this value depending on your GPU

void Loop::generateParticleSea() {
    particles.clear();

    float diameter = (float)brushSize;
    int columns = static_cast<int>(windowWidth / diameter);
    int maxRows = static_cast<int>((windowHeight / 5.0f) / diameter);

    int rows = std::min(maxRows, MAX_PARTICLES_SEA / columns);

    float startX = diameter / 2.0f;
    float startY = (windowHeight - rows * diameter) / 2.0f;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
            if ((int)particles.size() >= MAX_PARTICLES_SEA) break;

            Particle p;
            p.posX = startX + x * diameter;
            p.posY = startY + y * diameter;
            p.velX = 0.0f;
            p.velY = 0.0f;
            p.radius = brushSize / 2.0f;
            p.mass = particleMass;
            p.density = particleDensity;
            p.color = particleCol;
            p.timeLeft = PARTICLES_MAX_TIME;

            particles.push_back(p);
        }
    }
}
