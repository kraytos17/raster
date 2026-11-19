#pragma once

#include <SDL3/SDL.h>

namespace raster::platform {
    [[nodiscard]]
    SDL_AppResult application_init(void** appstate, int argc, char* argv[]);

    [[nodiscard]]
    SDL_AppResult application_event(void* appstate, SDL_Event* event);

    [[nodiscard]]
    SDL_AppResult application_iterate(void* appstate);

    void application_quit(void* appstate, SDL_AppResult result);
} // namespace raster::platform
