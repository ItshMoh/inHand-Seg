# Four-case prediction preview review

The user completed and fetched the four-case preview on 2026-11-19. The assistant
checked the real full-frame and matched-crop panels for all four cases. User
acceptance is still pending; no acceptance record has been created by the assistant.

## Numeric and artifact audit

- Exact selected coverage: four preview cases, eight PNG panels, four sealed
  case records and one HTML page. All 13 referenced file checksums validate;
  all eight PNGs decode.
- All 16 model/image records (four runs on four images) exactly reproduce the
  saved per-class image IoUs. Their confusion-matrix row sums agree with the
  original-label scan, and object FP/FN counts recompute from those matrices.
  Original image/label identities, seed labels, case fingerprints and crop bounds
  match the current selection.
- All eight fractions for the two SAM cases match the original Phase-6 candidate
  records, including exact mask ID, eligible area, bounding box and hand distance.
  The frozen rendering workflow checks cached raw/eligible-mask content hashes
  before producing these panels; no SAM proposals were regenerated.
- The Phase-7 code identity and frozen Phase-6 source hashes still match. Both
  cascades retain the same seed-0 hand model, with equal per-case hand IoUs.
- The latest preview summary says zero new prediction cases and `gpu: null`
  because that invocation reused completed case records. It is not a measurement
  of zero GPU use for the original inference. The individual case records contain
  their original prediction/render timing; billing is not inferred.

## What the actual panels show

All model comparisons in the PNGs display seed 0. Caption means and the HTML's
both-seed scores remain distinct from these displayed-seed values.

| example | displayed seed-0 comparison | observed visual evidence |
|---|---|---|
| own-video book | object IoU 21.99% baseline → 5.52% cascade | Both systems include background around the book; the cascade labels much more of the bedsheet as object. The matched crop confirms the same coordinates. The effect reverses for seed 1, as already disclosed in the caption. |
| YouTube ingredients/tray | object IoU 0.47% → 67.37% | The baseline misses almost all labelled object foreground. The cascade recovers much more of the tray/contents region, while still missing parts of its boundary and interior. |
| Ego4D headlamp part, SAM ID 60 | candidate object fraction 0% → 100% | The magenta eligible region is a small reflective part within the headlamp assembly. It remains GT background. The cascade paints the region and much of the surrounding assembly as object; the baseline leaves that selected region as background. This is an object-part proxy example, not a certified independent untouched object. |
| EPIC packaging region, SAM ID 18 | candidate object fraction 85.26% → 12.26% | The magenta outline covers a packaging-looking background region near the lower image edge, separate from the held square item. The baseline overpredicts this region and misses the held item; the cascade reduces the selected-region error and recovers much more of the held item's GT. Some misses and false positives remain. |

Layout observations: six aligned tiles per view; consistent original-coordinate
full frames and matched crops; blue hands, yellow objects, red object FP, cyan
object FN and magenta eligible-region outlines. Class/error legends and run/seed
labels are readable in the original PNGs. Ignore pixels remain outside scoring;
the displayed background proposals do not overlap GT hands or direct objects.

The current captions are valid descriptive preview captions. Wording about a
candidate's semantic extent awaiting its outline records the pre-render draft;
the two rendered regions have now been visually assessed above. The remaining
two selected SAM examples still await full-gallery rendering. No cause of a
failure is isolated by these selected examples.

## Next gate

The assistant's visual/technical review passes. The user must personally inspect
`gallery/preview.html` and accept this exact preview before running
`scripts/phase7_report.py approve-preview` and the full render command. Full
rendering can reuse the four checked cases and generate the remaining 12.
Final gallery/category/source-report verification remains a separate gate.
