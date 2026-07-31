# Known issues

## Tick labels can disagree with tick positions on a pinned axis

**Status:** open. Understood, reproducible, not yet fixed.

**Symptom.** An axis pinned with `--xlim`/`--ylim`/`--zlim` can be labelled with
only the first part of its range, stretched over its whole length, so the grid
looks like it stops early:

| limits | labelled | should be |
|---|---|---|
| `--ylim 0 4000` | `0 … 2000` | `0 … 4000` |
| `--ylim 0 6000` | `0 … 3000` | `0 … 6000` |
| `--ylim 1000 6000` | `1000 … 3500` | `1000 … 6000` |
| `--ylim 0 7000` | `0 … 7000` (correct) | — |

**Cause.** PyVista builds the label *strings* — `n_xlabels` values spread evenly
across the axis — while `vtkCubeAxesActor` decides how many ticks to actually
draw. It draws `min(our count, its own count)`:

- Ask for **fewer** labels than VTK's own count and it uses ours, spacing that
  many ticks evenly across the axis. The labels land where they claim to. This
  is the common case and it is correct.
- Ask for **more** and VTK falls back to its own count and its own spacing, then
  paints our strings onto those ticks in order and stops when it runs out. The
  axis is left carrying a prefix of the intended values stretched over its whole
  length.

So the defect is one-sided: only an axis where we request more labels than VTK
wants is wrong. `--ylim 0 7000` is correct because our 8 happens to match VTK's
8; `--ylim 0 4000` asks for 9 against VTK's 5.

VTK's count is deterministic from the range — identical at two different window
sizes, so it is not driven by screen geometry. Empirically its step is the
nearest 1/2/5 ×10^k to `span / 5`, checked against every native-labelled render
taken so far.

**The tell** is the last label, not the first. Both schemes start at the box
minimum, so tick 0 always agrees and the axis looks fine at the origin end. If
the highest label does not equal the axis maximum, the axis is compressed.

**Audit at the time of writing.** Correct: the default auto-fitted showcase and
single-trajectory scenes, on all three axes, and `--ylim 0 7000`. Wrong:
`aileron_roll.csv` (Y, 11 vs 5), `--demo spiral` (Z), the README showcase with
`--xlim` (X), `--ylim 0 4000` and `0 6000` (X and Y), `combat_example` (Y).

**Fix plan.** Clamp the requested label count to VTK's, since asking for fewer is
always honoured — predict VTK's count from the range with the rule above and cap
`_ticks_for_bounds` / `_nice_bounds` at it. That keeps the round steps and the
sparse short axes as they are, and only ever thins an axis that is currently
lying. Validate across a sweep of spans and window sizes before shipping, since
a wrong model of VTK's count would put labels back out of step.

Handing labelling to VTK entirely (dropping our strings so VTK labels its own
ticks) was tried and rejected: every value comes out correct, but VTK then labels
*every* tick it draws, and a short axis — a 1000 m altitude range beside a 6000 m
ground track — collects around six labels that overlap into an unreadable blob.

**Workaround meanwhile.** Leave axes unpinned where you can; the automatic fit
usually asks for fewer labels than VTK's count and is correct. When pinning, check
that the top label equals the limit you typed.
