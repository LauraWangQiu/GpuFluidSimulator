@echo off
set SRC_DIR=src
set OUT_DIR=bin
set EXECUTABLE=project.exe

set INCLUDE_DIRS=/I"..\dependencies\SDL2-2.30.4\include" ^
                 /I"..\dependencies\imgui-1.90.8" ^
                 /I"..\dependencies\imgui-1.90.8\backends" ^
                 /I"..\dependencies\imgui-1.90.8\misc\cpp" ^
                 /I"..\dependencies\imgui-1.90.8\misc\fonts" ^
                 /I"..\dependencies\imgui-1.90.8\misc\freetype" ^
                 /I"..\dependencies\imgui-1.90.8\misc\single_file"

set LIB_DIRS=/L"..\dependencies\SDL2-2.30.4\lib\x64"

set LIBS="..\dependencies\SDL2-2.30.4\lib\x64\SDL2.lib" ^
         "..\dependencies\SDL2-2.30.4\lib\x64\SDL2main.lib" ^
         "..\dependencies\SDL2-2.30.4\lib\x64\SDL2test.lib"

set IMGUI_FILES=..\dependencies\imgui-1.90.8\imgui.cpp ^
                ..\dependencies\imgui-1.90.8\imgui_demo.cpp ^
                ..\dependencies\imgui-1.90.8\imgui_draw.cpp ^
                ..\dependencies\imgui-1.90.8\backends\imgui_impl_sdl2.cpp ^
                ..\dependencies\imgui-1.90.8\backends\imgui_impl_sdlrenderer2.cpp ^
                ..\dependencies\imgui-1.90.8\imgui_tables.cpp ^
                ..\dependencies\imgui-1.90.8\imgui_widgets.cpp

if not exist %OUT_DIR% (
    mkdir %OUT_DIR%
)

icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fuse-ld=lld %SRC_DIR%\*.cpp %IMGUI_FILES% -o %OUT_DIR%\%EXECUTABLE% %INCLUDE_DIRS% %LIB_DIRS% %LIBS%

pause
