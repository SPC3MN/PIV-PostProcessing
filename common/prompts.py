def ask_yes_no(prompt, default=False):
    """Prompt for a yes/no answer. Blank input accepts `default`."""
    suffix = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def ask_float(prompt, default=None):
    """Prompt for a number. Blank input accepts `default` if given, else re-prompts."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return default
            print("A value is required.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def ask_int(prompt, default=None):
    """Prompt for a whole number. Blank input returns `default` (no forced re-prompt)."""
    suffix = f" [{default}]" if default is not None else " [blank = none]"
    raw = input(f"{prompt}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Not a whole number; leaving unset.")
        return default


def ask_text(prompt, default=None):
    """Prompt for free text. Blank input returns `default`."""
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else default
