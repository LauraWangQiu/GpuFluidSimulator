#ifdef _DEBUG
#include <iostream>
#endif

inline void debugLog(const char* message) {
#ifdef _DEBUG
    std::cout << message << '\n';
#endif
}
