# USP and Answers to Panel Questions

## Unique Selling Points

**1. Built for Chandrayaan-2's actual multi-sensor reality, not a generic demo.**
OHRC (~0.25m/pixel), TMC-2 (~5m/pixel), and IIRS (hyperspectral) are
three genuinely different resolutions and modalities from one mission.
This pipeline is architected to handle correspondence across all three,
not tuned for a single convenient sensor pair.

**2. Classical, explainable methods — not a black box.**
ORB/SIFT + RANSAC + homography instead of a trained deep model. This
is a deliberate choice: no labeled training data required, no GPU
dependency, and every stage is inspectable — which specific matches
were accepted or rejected, and why, can be shown directly, not just a
final answer with no reasoning behind it. For a scientific/archival
tool, that is arguably more valuable than higher raw accuracy from a
model nobody can audit.

**3. Proven on real, messy ISRO data — not sample/toy images.**
Real undocumented quirks were found and solved: the PDS4 key name
being `"ARRAY_0"` instead of the documented `"IMAGE"`, real files being
~1.2 billion pixels and needing deliberate memory handling, extreme
image aspect ratios breaking naive scale simulation. Most "concept"
submissions never encounter these because they demo on prepared sample
images instead of the actual public archive.

**4. Built-in quantitative evaluation, not just visual results.**
Every run logs inlier ratio, reprojection RMSE, and processing time
automatically. This is what turns "it looks aligned" into a measurable,
reproducible, comparable number — and is what enabled a real batch
study (42 registrations across 21 real pairs) rather than a handful of
cherry-picked examples.

**5. A working, usable tool, not just a script.**
A local web interface where two images can be uploaded and registered
live, with results savable and comparable side by side. That is the
difference between "we wrote some code" and "a tool that could
actually be used to check work."

---

## Direct answers to questions raised

### "Why build this when similar projects already exist?"

The closest comparable project is **MoonMetaSync / SyncVision**
(github.com/ashu1069/MoonMetaSync), a research package that also
performs OHRC/TMC-2 registration and introduces its own matching
method ("IntFeat") alongside ORB/SIFT.

What it has that this project doesn't: a novel matching algorithm and
a published paper — a genuine research contribution.

What this project has that it doesn't, as far as its public repository
shows: a working interactive web interface (their tool is command-line
only), a real batch evaluation across 21 scene pairs with statistical
analysis (correlation, distributions, detector comparison), automatic
geometry-based overlap discovery (no manual guessing which scenes to
pair), and ground-truth cross-scale validation with directly measured
accuracy. Their repository is 99%+ Jupyter notebooks, typical of
exploratory research code rather than a hardened, reproducible tool.

**Positioning:** this project does not claim to have invented a better
matching algorithm. It claims a more complete, validated, and *usable
system* built on established methods — a different and equally valid
kind of contribution.

### "Doesn't using less RAM (downsampling) reduce quality?"

Yes, honestly — downsampling does discard fine pixel-level detail in
the final output image. This is an explicit, disclosed tradeoff made
for a working prototype on 8GB-RAM consumer hardware with no GPU, not
something hidden.

Two important points limit how much this actually matters here:

- The tradeoff affects *how much fine detail survives into the final
  registered image*, not whether the *method itself* is correct. The
  same algorithm (ORB/SIFT + RANSAC + homography) applies identically
  whether images are full-resolution or downsampled — accuracy metrics
  obtained at reduced resolution (e.g. sub-2-pixel reprojection error,
  under 1% scale-recovery error in ground-truth testing) demonstrate
  the *method* generalizes; it is a resource/time constraint of this
  prototype, not a limitation of the underlying approach.
- Full-resolution processing was identified from the start as a
  planned next step (via tiling — splitting large scenes into
  overlapping chunks and processing each at full detail, then merging
  results), not something believed to be unsolvable. It was
  deprioritized in favor of breadth (more scenes, more validation)
  given the project timeline, not because it's out of reach.

### "Why does cross-sensor matching matter?"

OHRC and TMC-2 trade off resolution against coverage: OHRC sees
extremely fine detail (individual boulders, small craters) but only
a ~3km-wide strip per pass; TMC-2 sees much less detail but covers a
far wider area per pass. Neither instrument alone gives both fine
detail *and* wide context.

Registering the two means a small, highly-detailed OHRC patch can be
placed at its exact correct location within the broader TMC-2 map —
turning two separately-useful but disconnected datasets into one
coherent picture.

### "What's the benefit of a combined output from both sensors?"

A correctly registered combined output means every pixel from OHRC and
every pixel from TMC-2 that correspond to the same physical location on
the Moon are known to align. Concretely, this enables:

- **Detail-in-context:** viewing a fine OHRC feature (e.g. a boulder
  field or small crater) with correct positioning relative to the
  wider TMC-2 terrain around it, instead of needing to manually
  cross-reference two unaligned images by eye.
- **Cross-validation:** a feature spotted in one instrument's data can
  be checked against the other at the same registered location,
  reducing false detections from artifacts specific to a single
  sensor.
- **A foundation for further fusion:** once two sensors are registered
  into one common frame, a third data layer (e.g. IIRS hyperspectral
  bands) can, in principle, be added into the same frame later —
  building toward combined "what is it, and exactly what shape/size is
  it" analysis that no single instrument provides alone.
- **Practical use for review or mission planning:** anyone assessing a
  region (e.g. a candidate site) benefits from seeing both fine detail
  and wide-area context together rather than switching between two
  separate, unaligned datasets.
