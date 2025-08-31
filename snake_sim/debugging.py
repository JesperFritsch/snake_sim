import inspect

ENABLED_DEBUG_FUNCTIONS = set()

def debug_print_func(*args, **kwargs):
    frame = inspect.currentframe()
    outer_frame = frame.f_back
    func_name = outer_frame.f_code.co_name
    if func_name in ENABLED_DEBUG_FUNCTIONS:
        print(f"[{func_name}]", *args, **kwargs)

def debug_print(*args, **kwargs):
    pass

def activate_debug():
    global debug_print
    debug_print = debug_print_func

def enable_debug_for(*func_names):
    """Enable debugging for the given function names."""
    ENABLED_DEBUG_FUNCTIONS.update(func_names)
