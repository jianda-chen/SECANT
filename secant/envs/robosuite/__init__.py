def is_installed():
    try:
        import robosuite

        return True
    except ImportError:
        return False


if is_installed():
    from .core import make_robosuite, ALL_TASKS, make_robosuite_random_sample

else:
    _ERROR_MSG = """
Robosuite is not installed. Please follow the docs to install it.
    """.strip()
