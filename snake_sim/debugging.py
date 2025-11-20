import inspect
import sys

ENABLED_DEBUG_FUNCTIONS = set()
ALL_ENABLED = False

def _get_caller_info():
    frame = inspect.currentframe()
    outer_frame = frame.f_back.f_back
    func_name = outer_frame.f_code.co_name
    class_name = None
    if 'self' in outer_frame.f_locals:
        class_name = outer_frame.f_locals['self'].__class__.__name__
    return class_name, func_name

def _get_caller_tree():
    frame = inspect.currentframe()
    outer_frame = frame.f_back.f_back
    call_tree = []
    while outer_frame:
        func_name = outer_frame.f_code.co_name
        class_name = None
        if 'self' in outer_frame.f_locals:
            class_name = outer_frame.f_locals['self'].__class__.__name__
        call_tree.append((class_name, func_name))
        outer_frame = outer_frame.f_back
    return call_tree

def debug_print_func(*args, **kwargs):
    call_tree = _get_caller_tree()
    if _is_active_here(call_tree):
        class_name, func_name = call_tree[0] if call_tree else (None, None)
        prefix = f"[{class_name + '.' if class_name else ''}{func_name}]"
        print(prefix, *args, **kwargs)
        sys.stdout.flush()

def _is_active_here(call_tree: list[tuple[str | None, str]]):
    return ALL_ENABLED or any(
        (class_name in ENABLED_DEBUG_FUNCTIONS) or
        (func_name in ENABLED_DEBUG_FUNCTIONS)
        for class_name, func_name in call_tree
    )

def _is_debug_active():
    call_tree = _get_caller_tree()
    return _is_active_here(call_tree)

def is_debug_active():
    return False

def debug_print(*args, **kwargs):
    pass

def enable_debug_for_all():
    global ALL_ENABLED
    ALL_ENABLED = True

def activate_debug():
    global debug_print
    debug_print = debug_print_func
    global is_debug_active
    is_debug_active = _is_debug_active

def enable_debug_for(*func_names_or_classes):
    """Enable debugging for the given function or class names."""
    ENABLED_DEBUG_FUNCTIONS.update(func_names_or_classes)
