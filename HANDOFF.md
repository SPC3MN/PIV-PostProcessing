# Swirl PIV Processing — Handoff

Everything needed to reproduce this pipeline on a fresh machine and finish the one job still
outstanding: the **stereo decomposition and analysis**.

The trim and crop settings in [The settings](#the-settings) are the part that must not be guessed.

---

## Status

| Dataset | Variant | Decomposition | Analysis | Location |
|---|---|---|---|---|
| Planar (28) | full width, untrimmed | done | done | `D:\Final_NPZ\Swirl_PLANAR_PROCESSED\<case>_PROC` |
| Planar (28) | 10 cm, untrimmed | done | done | `C:\Users\Germiel\Desktop\Swirl_Planar_Cropped\<case>` |
| Planar (28) | full width, trimmed | done | done | `D:\Final_NPZ\Swirl_PLANAR_TRIMMED\FullFOV` |
| Planar (28) | 10 cm, trimmed | done | done | `D:\Final_NPZ\Swirl_PLANAR_TRIMMED\Crop10cm` |
| **Stereo (7)** | full width, trimmed | **TO DO** | **TO DO** | `D:\Final_NPZ\Swirl_STEREO_TRIMMED\Stereo_FullFOV` |
| **Stereo (7)** | 10 cm, trimmed | **TO DO** | **TO DO** | `D:\Final_NPZ\Swirl_STEREO_TRIMMED\Stereo_Crop10cm` |

Result workbooks written so far:

- `D:\Final_NPZ\Swirl_PLANAR_PROCESSED\Analysis_Results\Planar_Results.xlsx`
- `C:\Users\Germiel\Desktop\Swirl_Planar_Cropped\Analysis_Results\Planar_Results_10cm.xlsx`
- `D:\Final_NPZ\Swirl_PLANAR_TRIMMED\Analysis_Results\Planar_Results_FullFOV_trim7.xlsx`
- `D:\Final_NPZ\Swirl_PLANAR_TRIMMED\Analysis_Results\Planar_Results_Crop10cm_trim7.xlsx`

The stereo run was started once and stopped after ~1 minute. It wrote nothing — both stereo output
folders are empty and there is no partial state to clean up.

---

## Inputs

Every case folder holds 1000 `snap_*.npz` files. Each file carries `X`, `Y` and the velocity
components — `U`, `V` for planar; `U`, `V`, `W` for stereo — plus a `mask` array the pipeline does
not use.

| Set | Cases | Path | Layout |
|---|---|---|---|
| Planar | 23 | `J:\PostProc_NPZ` | 3 recording folders, each holding case subfolders |
| Planar | 5 | `D:\Final_NPZ\Swirl_PLANAR` | flat: case subfolders named `<case>-planar` |
| Stereo | 7 | `D:\Final_NPZ\Swirl_STEREO` | flat: case subfolders named `<case>-stereo` |

Case names encode the actuation as `<mean on>-<burst on>-<burst off>` in seconds.

The 7 stereo cases: `3.0-1.5-0.3`, `6.0-0.0-0.0`, `6.0-1.5-0.3`, `6.0-1.5-0.7`, `6.0-1.5-1.5`,
`6.0-3.0-0.7`, `6.0-3.0-1.5` — each suffixed `-stereo`.

---

## The settings

**Order matters.** The edge trim is applied first, to the raw field; the 10 cm window is taken from
the already-trimmed field. Doing it this way is what makes the full-width and cropped variants share
an identical height, so the two are directly comparable.

The trim exists to drop the outermost interrogation windows, where PIV correlation is unreliable.
The two datasets use a different number of points because their grids differ in spacing — both were
sized to remove the same **~6.3% of FOV area**.

| | Planar | Stereo |
|---|---|---|
| Raw grid (ny × nx) | 379 × 514 | 384 × 735 |
| dx = dy (mm) | 0.41191 | 0.44640 |
| Raw x range (mm) | −97.764 … 113.544 | −156.408 … 171.253 |
| Raw y range (mm) | −85.582 … 70.119 | −93.790 … 77.182 |
| **TRIM points per edge** | **7** | **8** |
| — border removed (mm) | 2.883 | 3.571 |
| — area removed | 6.32% | 6.25% |
| After trim (ny × nx) | 365 × 500 | 368 × 719 |
| — x range (mm) | −94.88 … 110.66 | −152.84 … 167.68 |
| — y range (mm) | −82.70 … 67.24 | −90.22 … 73.61 |
| **CROP half-width (points)** | **121** | **112** |
| After crop (ny × nx) | 365 × 242 | 368 × 224 |
| — crop x range (mm) | −49.98 … 49.29 | −50.16 … 49.38 |
| — crop width (mm) | 99.68 | 99.99 |
| Height, both variants (mm) | 149.93 | 163.83 |

Every case within a dataset shares one grid — verified across all 28 planar and all 7 stereo cases —
so a single setting applies to the whole set.

> **The crop is centred on x = 0, the coordinate axis — not the centre of the field of view.**
> The two are far apart here: the planar FOV centre sits near x = 7.9 mm and the stereo centre near
> x = 7.4 mm. Using the FOV centre instead would shift the window off the swirl axis and quietly
> invalidate the comparison.

### Authoritative implementation

```python
TRIM_PTS      = 7      # planar; use 8 for stereo
CROP_WIDTH_MM = 100.0

# 1. TRIM first, on the raw field
t  = TRIM_PTS
sl = (slice(t, -t), slice(t, -t))
X  = X_raw[sl];  Y = Y_raw[sl]
U  = U_raw[sl];  V = V_raw[sl]        # + W for stereo

# 2. CROP second, on the already-trimmed x coordinates
x    = X[0]
dx   = np.median(np.diff(x))
zero = int(np.argmin(np.abs(x)))                   # column nearest x = 0
half = int(round((CROP_WIDTH_MM / 2) / abs(dx)))   # 121 planar, 112 stereo
left, right = zero - half, zero + half             # 2*half columns

X_c = X[:, left:right]                             # height untouched
U_c = U[:, :, left:right]
```

`half` is derived from `dx` at runtime rather than hard-coded, so the same code gives 242 columns on
planar and 224 on stereo without editing.

---

## Setup

1. **Install Python 3.11 and clone the repo.** Python 3.11.9 is what this was run on.

   ```bash
   git clone https://github.com/SPC3MN/PIV-PostProcessing.git
   cd PIV-PostProcessing
   ```

2. **Create the environment.**

   ```bash
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   That pulls numpy, pandas, matplotlib, scipy and openpyxl. **lvpyio is not needed** — it only
   matters if you start from raw DaVis `.vc7` files, and it is not on PyPI (it ships with the DaVis
   Python add-ons). This pipeline starts from `.npz`, so skip it.

3. **Confirm the parallelisation commit is present.**

   ```bash
   git log --oneline -1 -- common/decomposition_stats.py
   ```

   You want `671ff76` or later. Without it the per-snapshot loops run single-threaded and a case takes
   **~21 minutes instead of ~3**.

4. **Confirm the batch runners came with the clone.**

   ```bash
   git log --oneline -1 -- decomposition/run_batch_stereo_trimmed.py
   ```

   You want `ecdca79` or later. They were untracked local files until that commit.

5. **Fix the paths at the top of each runner.** Each hard-codes `RAW_ROOT` / `RAW_ROOTS` and
   `OUT_ROOT` as absolute Windows paths belonging to the acquisition machine. **Change these first** —
   they are committed as-is and will not resolve anywhere else.

---

## The batch runners

In `decomposition/`, committed as of `ecdca79`. These drive the `common/*` functions directly,
because the repo's own `Planar_Decomposition.py` / `Stereo_Decomposition.py` are interactive,
prompt-driven, process one root at a time, and have no trim option.

| Runner | Scope |
|---|---|
| `run_batch_trimmed.py` | planar, 28 cases, 7 pts/edge trim, both variants |
| `run_batch_stereo_trimmed.py` | stereo, 7 cases, 8 pts/edge trim, both variants, 3 components + Lumley |
| `run_batch_planar.py` | superseded — nested case roots, no trim |
| `run_batch_planar_flat.py` | superseded — flat case root, no trim |
| `run_batch_planar_cropped.py` | superseded — 10 cm crop about x=0, no trim |

The superseded three are kept because they produced the untrimmed results already on disk.

Both current runners **load each case's 1000 snapshots once with the edge trim applied, then derive
the full-width and 10 cm variants from that single read**. Doing both variants per load rather than
in two separate passes saves roughly 20 minutes of I/O across a full planar batch, and guarantees the
two variants come from byte-identical input.

Flags: `--cutoff N` (limit snapshots, for a smoke test), `--only-case NAME`, `--variant NAME`.

The planar runner reads from two roots with different shapes: `J:\PostProc_NPZ` nests case folders
one level deeper (inside recording folders) while `D:\Final_NPZ\Swirl_PLANAR` is flat. Its
`discover_all_cases()` handles both — it tries `discover_case_dirs_or_root` first and, only if that
finds nothing, descends one more level. The stereo runner needs no such handling; its root is flat.

---

## Running the outstanding stereo job

Expect roughly **5 minutes per case**, so about **35 minutes** for all seven. Peak memory ~8 GB.

```bash
# smoke test one case first - 12 snapshots, ~30 s
.venv\Scripts\python.exe -u decomposition\run_batch_stereo_trimmed.py ^
    --cutoff 12 --only-case 6.0-0.0-0.0-stereo

# then the real run: 7 cases x 2 variants
.venv\Scripts\python.exe -u decomposition\run_batch_stereo_trimmed.py
```

**Delete the smoke-test output before the real run** — a 12-snapshot result sitting in the output
tree is indistinguishable from a finished one.

Then the analysis, once per variant. Edit the `# Control` block at the top of
`analysis\Stereo_Analysis.py`: set `processed_root` to the variant folder, `results_path` to the
workbook, and `save = True`.

```bash
# MPLBACKEND=Agg is required - see the gotchas
set MPLBACKEND=Agg
.venv\Scripts\python.exe -u analysis\Stereo_Analysis.py
```

---

## Gotchas

**Use the batch runners, not the repo's own scripts.** `Planar_Decomposition.py` and
`Stereo_Decomposition.py` are interactive, process one root at a time, and have **no trim option** —
they cannot do this job unaided.

**`Stereo_Analysis.py` plots unconditionally.** `Planar_Analysis.py` has a `plotting` flag in its
control block; the stereo one does not, and calls `plt.show()` seven times per case. Run it headless
and it blocks forever. Set `MPLBACKEND=Agg` so `show()` becomes a no-op — that leaves the analysis
maths untouched. For the planar script, set `plotting = False` instead.

**Never patch the control block with a regex.** The control values are Windows paths full of
backslashes. Using `re.sub` to rewrite them throws `bad escape \F`, and if that error is swallowed
the script happily runs against the *previous* values and overwrites the wrong workbook. Use plain
string replacement on whole lines. This happened once; the file it clobbered was regenerated from
identical inputs so nothing was lost, but it is an easy way to silently destroy a result.

**Analysis rewrites the whole workbook each run.** `write_results_xlsx` regenerates the file from
scratch, so the workbook always reflects exactly the cases just processed. Point two variants at the
same `results_path` and the second silently replaces the first. Give each variant its own filename.

**Two outputs are not trustworthy as measurements.** The Taylor microscale `lambda_1` is pinned at
~4 grid spacings across every case and window — it is a PIV-noise artefact, not a length scale, and
neither it nor any `Re_lambda` derived from it should be reported. In the 10 cm variants `L_11` is
window-limited: most fitted values exceed the half-width of the window they were measured in, because
the model extrapolates past the available separations. Use the full-width `L_11`.

---

## Verification

- Each variant folder has **7 case folders**; each case has `Ensemble_Averages/` with 4 `.npz` files
  (`Averages`, `Bootstrapped_Statistics`, `Structure_Function`, `Autocorrelation_Function`) plus
  `Lumley_Statistics/Lumley_Statistics.npz`. Stereo produces **no** energy spectrum — that stage
  exists only in the planar script.
- `Averages.npz` must contain `W_mean`, `W_rms`, `uv`, `uw`, `vw`. If those are missing you ran the
  planar path by mistake.
- `Lumley_Statistics.npz` holds `R` shaped `(3, 3, ny, nx)` plus `eta2`, `xi`, `II`, `III`.
- Grid shapes are exactly **368 × 719** (full) and **368 × 224** (10 cm), and the **y range matches
  between the two** — that is the check that the trim ran before the crop.
- The workbook has **7 rows** and no `NaN` in `eps_11` or `L_11`. A collapsed length-scale fit shows
  up as a value near zero or negative.

---

## Pipeline summary

SPC3MN/PIV-PostProcessing. Reynolds decomposition, Dirichlet bootstrap confidence intervals,
second-order structure functions, spatial autocorrelations, 1-D energy spectra (planar only), and —
stereo only — Reynolds stresses with Lumley anisotropy invariants, over 1000 snapshots per case.

Relevant commits:

- `671ff76` — parallelised the per-snapshot bootstrap loops with a thread pool (~7–8× speedup)
- `ecdca79` — added the five `run_batch_*` runners
