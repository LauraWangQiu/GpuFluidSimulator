#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Loop.h"
#include "checkML.h"
#include "kernel.cuh"
#include <iostream>

int main(int argc, char** argv) {
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    Loop* loop = new Loop();
    if (loop->init()) loop->run();
    delete loop;

    return 0;
}
