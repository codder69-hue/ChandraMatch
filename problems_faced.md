# Problems Faced During Development

Each entry: what the problem was, why it was actually a problem, and how it was solved.

---

**1. Unknown PDS4 data key ("ARRAY_0" vs "IMAGE")**
Documentation and tutorials for the `pdr` (Planetary Data Reader) library
show pixel data accessed via a key called `"IMAGE"`. Real Chandrayaan-2
OHRC/TMC-2 files instead store it under `"ARRAY_0"`. Because the key
name is undocumented per-mission, the loading code crashed / returned
nothing until the correct key was known — guessing wrong wastes time
across an entire batch. **Fix:** built a `--inspect` diagnostic mode
that prints every key, shape, and dtype in a real file *before* writing
any loading logic against it, then made the loader try several known
key names automatically instead of hardcoding one.

**2. Full-resolution images risked exhausting available RAM**
A single real OHRC scene is roughly 101,075 × 12,000 pixels — about 1.2
billion pixels. Converting an array that size to floating point (needed
for contrast normalization) before doing anything else would require
approximately 10GB of RAM, on a laptop with 8GB total. This risked
crashing the whole pipeline partway through a run, not just running
slowly. **Fix:** downsample the image to a manageable size (e.g. max
dimension 4000px) *before* any float conversion, so peak memory stays
low regardless of the original file size.

**3. Windows blocked the setup script as an "Unknown Publisher"**
Double-clicking `setup_windows.bat` triggered a Windows SmartScreen
warning. This looked like something was wrong with the file. In fact,
this happens to *every* unsigned `.bat` file regardless of content —
signing a script costs money most student projects don't spend on.
**Fix:** verified the script's contents (just environment setup and
`pip install`, nothing else) and ran it anyway; confirmed this is
standard, expected behavior, not a defect.

**4. `streamlit` command ran from the wrong Python installation**
After installing Streamlit inside the project's virtual environment,
running `streamlit run streamlit_app.py` still failed with
`ModuleNotFoundError: No module named 'cv2'` — a package that was
definitely installed. Windows can have multiple Python installations,
and a bare command name resolves to whichever one is first on PATH,
which is not necessarily the active virtual environment. **Fix:**
always invoke via `python -m pip install ...` and
`python -m streamlit run ...`, which forces both commands to use
whichever Python the current environment is actually pointing at.

**5. Identical-looking filenames for completely different files**
Windows Explorer hides file extensions by default. This meant a PDS4
product's small `.xml` metadata/label file and its large `.img` binary
data file displayed with the exact same visible name, making it
impossible to tell them apart by looking. Picking the wrong one broke
the loader. **Fix:** identified the correct file by its file *size* and
*type* column instead of its name, and enabled "file name extensions"
in Explorer to remove the ambiguity going forward.

**6. Batch processing appeared to hang for 30+ minutes**
Running the pipeline across 10 downloaded scenes with both detectors
requires up to 90 individual pipeline executions. The first version of
the batch script reloaded every ~1GB scene from disk fresh, every time
it appeared in a pair — with no caching — multiplying disk I/O and
PDS4-parsing time far beyond what was necessary. It also only wrote
results to the output CSV at the very end, so interrupting a long run
lost everything computed so far. **Fix:** added an in-memory cache so
each unique scene is read from disk exactly once per run regardless of
how many pairs it appears in, and switched to writing each result to
CSV immediately after it's computed, so progress is never lost even if
the run is interrupted.

**7. Cross-scale simulation failed due to extreme image aspect ratio**
OHRC scenes are long, narrow strips — roughly an 8:1 ratio (very tall,
comparatively narrow). After the pipeline's own downsampling, the
narrow dimension was already reduced to ~474 pixels. Reducing that
*again* by ~20x (to simulate TMC-2's coarser resolution) left only ~24
pixels in that dimension — far too thin for any feature detector to
find reliable keypoints, so registration failed outright. **Fix:**
instead of shrinking an already-small image further, read a genuine
square crop directly from the *native full-resolution* data first
(6000×6000 pixels, well within the scene's actual 12,000px width), then
apply the 20x reduction to that square crop — leaving a comfortable
300×300 image with real content to match against.

**8. No real Chandrayaan-2 OHRC/TMC-2 pair could be found in the archive**
After building a working cross-sensor registration pipeline, no TMC-2
scene could be found that overlapped, either in location or in time,
with the OHRC data already collected — filtering TMC-2 by the same date
range returned zero results for the entire month. This is a genuine gap
in the public archive's coverage, not a bug in any code written for
this project. **Fix:** pivoted to a controlled validation instead: real
OHRC imagery already on hand was synthetically degraded to TMC-2's
documented resolution under a *known* applied transform, so registration
accuracy could be measured directly against ground truth — arguably a
stronger validation than an unpaired real dataset would have provided.

**9. RANSAC produced a visually distorted result on a low-overlap pair**
When two OHRC scenes with only a small genuine overlap were registered,
the output showed a fan-like, "shredded" distortion instead of a clean
alignment. Investigating the inlier-match visualization showed the
matched points were clustered in one small corner of the image — a
homography is only reliably accurate near the points that defined it,
so it extrapolates unpredictably everywhere else when the true overlap
is small. This is an inherent property of applying a single global
transform to a partially-overlapping pair, not a code defect. **Fix:**
this was documented explicitly as a known limitation, and used as a
deliberate example of the pipeline correctly reporting a low-confidence
result rather than fabricating a convincing but wrong one.

**10. A type mismatch crashed image resizing**
`cv2.resize()` failed with an argument error because the computed
target size was a Python float rather than an integer — a side effect
of one input parameter being parsed as a floating-point command-line
argument. OpenCV's underlying implementation requires exact integer
pixel dimensions and does not automatically convert floats. **Fix:**
explicitly wrapped every computed width/height value in `int()` before
passing it to `cv2.resize()`.

**11. A missing leading dot silently disabled `.gitignore`**
A file intended to tell Git which folders to exclude from
version control was saved as `gitignore` instead of `.gitignore`. Git
only recognizes the exact filename `.gitignore` — without the leading
dot, it is treated as an ordinary, meaningless text file, so none of
its exclusion rules applied. Large folders (virtual environment,
downloaded data, cache files) kept appearing as trackable instead of
being ignored. **Fix:** renamed the file correctly and confirmed via
`git status` that the intended folders disappeared from the untracked
list.

**12. Risk of committing multi-gigabyte data files to version control**
Each downloaded Chandrayaan-2 product is 750MB to ~1.2GB. GitHub
rejects any single file larger than 100MB, so committing the raw
downloaded data folder as-is would have made the repository impossible
to push at all. **Fix:** excluded the data folder from version control
entirely and documented in the project README exactly where and how to
re-download the same data instead of storing it in the repository.
