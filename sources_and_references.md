# Sources and References

## Data source

Chandrayaan-2 OHRC and TMC-2 imagery: Indian Space Science Data Centre
(ISSDC), ISRO — Pradan portal.
https://pradan.issdc.gov.in/ch2/

Instrument specifications (ground sampling distance, swath, orbit
altitude) referenced from ISSDC's own Chandrayaan-2 mission
documentation on the same portal.

PDS4 (Planetary Data System, version 4) — the data format Chandrayaan-2
products are distributed in:
https://pds.nasa.gov/datastandards/documents/

## Software / libraries used

- OpenCV (feature detection, matching, homography estimation,
  image processing) — https://docs.opencv.org/
  Specifically consulted: "Feature Matching + Homography to find
  Objects" tutorial (docs.opencv.org) for the standard
  SIFT/ORB + RANSAC + homography pipeline structure.
- `pdr` (Planetary Data Reader), for reading PDS4-format files in
  Python — https://pdr.readthedocs.io/
- Streamlit, for the web interface — https://docs.streamlit.io/
- NumPy, Matplotlib, Pandas — standard scientific Python libraries.

## Algorithms — original sources

The core pipeline is built on established computer vision methods, not
a novel algorithm. Original references:

- **SIFT**: Lowe, D.G. (2004). "Distinctive Image Features from
  Scale-Invariant Keypoints." *International Journal of Computer
  Vision*, 60(2), 91-110.
- **ORB**: Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011).
  "ORB: An efficient alternative to SIFT or SURF." *2011 International
  Conference on Computer Vision (ICCV)*, 2564-2571.
- **RANSAC**: Fischler, M.A., & Bolles, R.C. (1981). "Random Sample
  Consensus: A Paradigm for Model Fitting with Applications to Image
  Analysis and Automated Cartography." *Communications of the ACM*,
  24(6), 381-395.
- **Homography / multi-view geometry**: Hartley, R., & Zisserman, A.
  (2004). *Multiple View Geometry in Computer Vision* (2nd ed.).
  Cambridge University Press.

## Related work consulted

- Kumar, A. et al. "MoonMetaSync: Lunar Image Registration Analysis."
  arXiv:2410.11118. https://arxiv.org/abs/2410.11118
  Associated code: https://github.com/ashu1069/MoonMetaSync
  Consulted for context on prior lunar image registration approaches
  and typical OHRC/TMC-2 resolution figures used for sanity-checking
  this project's own geometry-derived GSD estimates. No code from this
  repository is reused in this project.

## Idea / problem framing

The multi-modal, multi-scale image correspondence problem (matching
across OHRC, TMC-2, and IIRS) follows the general theme of past ISRO
public data-science hackathon problem statements involving Chandrayaan-2
imagery (e.g. crater/feature detection from OHRC, spectral analysis
from IIRS) — cited here as the general problem context this project's
own problem statement was framed within, not as a specific reused
source.
