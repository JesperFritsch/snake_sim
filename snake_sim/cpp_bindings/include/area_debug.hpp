
#pragma once


#ifdef DEBUG
#define DEBUG_PRINT(x) do { x; } while (0)
#else
#define DEBUG_PRINT(x) do {} while (0)
#endif