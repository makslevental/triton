def hip_bindings_not_installed():
    try:
        import hip

        # don't skip
        return False

    except ImportError:
        # skip
        return True
