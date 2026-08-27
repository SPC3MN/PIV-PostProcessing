import os
import glob


def discover_case_dirs(root, required_glob):
    """Return {case_name: case_dir} for every immediate subdirectory of `root`
    that contains at least one file matching `required_glob` (a path relative
    to the case directory, e.g. "*.npz" or
    os.path.join("Ensemble_Averages", "Averages.npz")).

    This is the folder-of-subfolders convention both decomposition (pointed at
    a root of raw npz case folders) and analysis (pointed at decomposition's
    output root) discover cases from, instead of a hand-maintained case list.
    """
    cases = {}
    for entry in sorted(os.listdir(root)):
        case_dir = os.path.join(root, entry)
        if not os.path.isdir(case_dir):
            continue
        if glob.glob(os.path.join(case_dir, required_glob)):
            cases[entry] = case_dir
    return cases


def discover_case_dirs_or_root(root, required_glob):
    """Like discover_case_dirs, but also accepts being pointed directly at a
    single case folder: if `root` itself contains files matching
    `required_glob`, it's returned alone as one case (named after its own
    directory) instead of being scanned for subfolders. Lets a caller accept
    either one bare snapshot folder or a parent of many case folders."""
    if glob.glob(os.path.join(root, required_glob)):
        return {os.path.basename(os.path.normpath(root)): root}
    return discover_case_dirs(root, required_glob)
