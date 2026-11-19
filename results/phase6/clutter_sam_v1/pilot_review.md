# Phase-6 pilot review

The user completed preparation and the ten-image pilot on NVIDIA L40S. The pilot
completed technically; the user subsequently accepted the candidate previews and
authorized the final run with the frozen default recipe on 2026-11-19.
This note records the pilot-stage observations. The subsequent full 1,626-image
analysis has completed; its comparison is in `results/phase6_report.md` and still
awaits final user verification.

The prepared source hashes match the current analysis code. All fetched preview
checksums match `pilot.json`. The inventory contains exactly 1,626 unique original
image identities. The four compared checkpoints have the expected seeds, saved
bf16 settings and three/four input channels; stage 1 is the shared seed-0 RGB model.

| Pilot measurement | Observed value |
|---|---:|
| Images | 10 (six in-domain, four distinct YouTube videos) |
| SAM proposals | 996 |
| Retained candidate regions | 670 |
| Raw-area rejections | 142 |
| Foreground-overlap rejections | 124 |
| Any-ignore-overlap rejections | 25 |
| Additional mask-NMS duplicates | 35 |
| Retained regions with some foreground pixels clipped | 101 |
| Median SAM generation time | 2.60 s/image |
| Maximum SAM generation time | 7.03 s/image (4K image) |
| Total SAM generation time | 31.06 s |
| Pilot wall time, including rendering and other work | 316.24 s |
| Maximum PyTorch allocated GPU memory | 8.45 GB (decimal) |
| Maximum PyTorch reserved GPU memory | 26.41 GB (decimal) |

All retained audit rows pass the configured raw/eligible area and overlap
thresholds, contain no ignored pixels, and have positive hand distance. All ten
pilot images have labelled hands. The six distance bins contain
120 / 16 / 61 / 73 / 181 / 219 retained regions respectively; these are pilot
coverage counts, not false-positive measurements or full-dataset denominators.

## Visual assessment

The assistant inspected all ten overviews and selected retained detail sheets.
Plausible untouched kitchen objects are retained, and labelled hands/directly
interacting objects are excluded or clipped as designed. The set also visibly
includes floors, cabinet panels, walls, clothing, sink/counter surfaces and
nested object parts. It is therefore a proposal-level background-region proxy,
not a clean count of independent untouched object instances.

Concrete examples for the user's review:

- EPIC retained sheet 1, IDs 0 and 4: overlapping floor regions. Both survive the
  agreed area and mask-IoU thresholds; this is an interpretation limitation.
- THU retained sheet 2, IDs 5 and 6: wall/floor regions; ID 8 is a small loose item.
- Ego4D kitchen retained sheet 1: distinct kitchen items and a large surface patch.
- Katsudon retained sheet 1: plate/bowl/cup candidates alongside a background patch.

Open `pilot/index.html` for every overview and paginated proposal sheet. Green
regions passed the rules; green does not certify semantic object identity.
Retaining this recipe is defensible if the eventual result is explicitly reported
as SAM background-proposal false positives. A claim limited to untouched object
instances would require a revised, model-independent candidate-selection policy
and another user review before the full run.

## Accepted full-run plan

The pilot's median-based SAM-only projection is 1.17 GPU-hours. Applying its slowest
observed per-image time to every image gives 3.17 GPU-hours, or about 47.7 minutes
with four evenly loaded workers. These are extrapolations, not bounds or measured
full-run runtimes. Startup, CPU filtering, model inference, commits and I/O add time.
The full job does not render hundreds of preview sheets per image.

Keep the proposed four L40S workers and current two-hour per-worker ceiling as a
conservative execution bound; do not interpret the ceiling as a cost estimate.
The user subsequently persisted approval on the Volume and completed all four
shards. See `results/phase6_report.md` for the actual final comparison; the pilot
projections above are retained as planning history.
