
#pragma once


#ifdef DEBUG
#define DEBUG_ONLY(x) do { x; } while (0)
#else
#define DEBUG_ONLY(x) do {} while (0)
#endif