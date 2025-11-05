import inspect
import sys

ENABLED_DEBUG_FUNCTIONS = set()
ALL_ENABLED = False

def debug_print_func(*args, **kwargs):
    frame = inspect.currentframe()
    outer_frame = frame.f_back
    func_name = outer_frame.f_code.co_name
    class_name = None
    if 'self' in outer_frame.f_locals:
        class_name = outer_frame.f_locals['self'].__class__.__name__
    # Enable if function or class is in the set
    if ALL_ENABLED or func_name in ENABLED_DEBUG_FUNCTIONS or (class_name and class_name in ENABLED_DEBUG_FUNCTIONS):
        prefix = f"[{class_name + '.' if class_name else ''}{func_name}]"
        print(prefix, *args, **kwargs)
        sys.stdout.flush()

def debug_print(*args, **kwargs):
    pass

def enable_debug_for_all():
    global ALL_ENABLED
    ALL_ENABLED = True

def activate_debug():
    global debug_print
    debug_print = debug_print_func

def enable_debug_for(*func_names_or_classes):
    """Enable debugging for the given function or class names."""
    ENABLED_DEBUG_FUNCTIONS.update(func_names_or_classes)
