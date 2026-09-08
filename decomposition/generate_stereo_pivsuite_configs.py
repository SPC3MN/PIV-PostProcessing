"""
Generate one .pivproj config per case for the custom PIV Suite (piv-suite
CLI, stereo mode), for cases that have no DaVis vector output at all and
need correlation run from raw camera images.

Everything except correlation.passes and project.input_path/output_dir is
taken verbatim from TEMPLATE_PIVPROJ -- same calibration, same
preprocess/validation/postprocess settings, same performance.n_workers.
That template must already exist (build one with the piv-suite GUI or CLI
against any one .set in this calibration session first; calibration is
shared across every recording in one campaign, so any case's template
works for all of them).

The only deviation from the template's own defaults: this repo's
convention for a case with no DaVis result at all is 64px@50% + TWO 32px@
75% passes, not the usual 64px@50% + THREE 32px@75% passes -- see
run_batch_stereo_pivsuite.py's docstring for why (it changes the raw
correlation grid from 384x735 to 380x731, and TRIM_PTS there is set to 6,
not 8, to land on the same target shape after trimming).

Usage:
    python generate_stereo_pivsuite_configs.py
    python generate_stereo_pivsuite_configs.py --only-case 6.0-0.7-0.3-stereo
"""
import argparse
import json
import os

TEMPLATE_PIVPROJ = r"C:\Users\Germiel\Desktop\PIV_Suite_testing\stereo_full.pivproj"
RAW_ROOT = r"J:\Final_Stereo\Swirl"          # holds "<raw case>.set" files
OUT_ROOT = r"J:\PIV_PostProc\_npz_staging\CustomSuite"
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "pivproj_configs")

PASSES_2X32 = [
    {"window_size": 64, "overlap_fraction": 0.5},
    {"window_size": 32, "overlap_fraction": 0.75},
    {"window_size": 32, "overlap_fraction": 0.75},
]

# case_name -> raw DaVis recording folder/set name (without ".set")
CASES = {
    "3.0-1.5-0.7-stereo": "On Time=3.0_Burst On Time=1.5_Burst Off Time=0.7",
    "3.0-1.5-1.5-stereo": "On Time=3.0_Burst On Time=1.5_Burst Off Time=1.5",
    "3.0-1.5-3.0-stereo": "On Time=3.0_Burst On Time=1.5_Burst Off Time=3.0",
    "6.0-0.7-0.3-stereo": "On Time=6.0_Burst On Time=0.7_Burst Off Time=0.3",
    "6.0-0.7-0.7-stereo": "On Time=6.0_Burst On Time=0.7_Burst Off Time=0.7",
    "6.0-0.7-3.0-stereo": "On Time=6.0_Burst On Time=0.7_Burst Off Time=3.0",
}


def generate(case_name, raw_name, template):
    cfg = json.loads(json.dumps(template))  # deep copy
    cfg["project"]["input_path"] = os.path.join(RAW_ROOT, f"{raw_name}.set")
    cfg["project"]["output_dir"] = os.path.join(OUT_ROOT, case_name)
    cfg["correlation"]["passes"] = PASSES_2X32
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-case", type=str, default=None)
    ap.add_argument("--template", type=str, default=TEMPLATE_PIVPROJ,
                     help="Path to an existing .pivproj to copy calibration/settings from")
    args = ap.parse_args()

    if not os.path.isfile(args.template):
        raise FileNotFoundError(
            f"Template config not found: {args.template!r}\n"
            "Build one first with the piv-suite GUI or CLI against any one .set file "
            "in this calibration session -- calibration is shared across every "
            "recording in one campaign, so any case's template works for all of them.")

    with open(args.template, encoding="utf-8") as f:
        template = json.load(f)

    cases = dict(CASES)
    if args.only_case:
        cases = {k: v for k, v in cases.items() if k == args.only_case}
        if not cases:
            raise ValueError(f"{args.only_case!r} not in CASES: {sorted(CASES)}")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    for case_name, raw_name in sorted(cases.items()):
        cfg = generate(case_name, raw_name, template)
        out_path = os.path.join(CONFIG_DIR, f"{case_name}.pivproj")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"{case_name}")
        print(f"  input_path:  {cfg['project']['input_path']}")
        print(f"  output_dir:  {cfg['project']['output_dir']}")
        print(f"  passes:      {cfg['correlation']['passes']}")
        print(f"  -> {out_path}")

    print(f"\n{len(cases)} config(s) written to {CONFIG_DIR}")
