#define SDL_MAIN_USE_CALLBACKS

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "raster/platform/application.hpp"

// SDL owns the callback loop; all application state and behavior lives in the
// platform application controller.
SDL_AppResult SDLCALL SDL_AppInit(void** appstate, int argc, char* argv[]) {
    return raster::platform::application_init(appstate, argc, argv);
}

SDL_AppResult SDLCALL SDL_AppEvent(void* appstate, SDL_Event* event) {
    return raster::platform::application_event(appstate, event);
}

SDL_AppResult SDLCALL SDL_AppIterate(void* appstate) {
    return raster::platform::application_iterate(appstate);
}

void SDLCALL SDL_AppQuit(void* appstate, SDL_AppResult result) {
    raster::platform::application_quit(appstate, result);
}
