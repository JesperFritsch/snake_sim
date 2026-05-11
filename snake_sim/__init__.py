try:
    import evdev
except ImportError:
    class _MissingEvdev:
        def __getattr__(self, name):
            raise ImportError(
                "evdev is not installed. Manual input mode requires the "
                "'game' extra: pip install 'snake_sim[game]'"
            )
    evdev = _MissingEvdev()