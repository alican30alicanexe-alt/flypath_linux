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

**Cause.** PyVista and VTK split the job and never compare notes. PyVista builds
the label *strings* — `n_xlabels` values spread evenly across the axis — while
`vtkCubeAxesActor` independently decides the tick *positions* from the range,
using its own step. When the two counts disagree, VTK paints our strings onto
its ticks in order and stops when it runs out, so the axis ends up carrying a
prefix of the intended values spread across its full length.

`0 7000` looks right only by luck: both sides happened to pick a 1000 step.

The count VTK picks is deterministic from the range — it is identical at two
different window sizes, so it is not driven by screen geometry — and appears to
be a 1/2/5 ×10^k step targeting roughly five intervals.

**Why it is not simply fixed.** `vtkCubeAxesActor` exposes no public way to set
its tick count, and no way to read back the count it chose. Two workarounds were
tried:

- *Hand labelling back to VTK* (drop our strings, let VTK label its own ticks).
  Every value comes out correct, but VTK then labels **every** tick it draws, and
  a short axis — a 1000 m altitude range beside a 6000 m ground track — collects
  around six labels that overlap into an unreadable blob. Rejected for that.
- *Matching VTK's count*, then blanking alternate labels to thin a short axis.
  This is the real fix: correct values and a readable short axis. It needs VTK's
  step rule reverse-engineered and validated across a sweep of ranges, since a
  wrong model puts the labels back out of step.

**Note on what looks fine today.** A short axis currently looks *good* because of
this bug, not despite it: we send three labels, VTK draws about six ticks, so
only three are painted — sparse and legible, but sitting at the wrong positions.
A `--zlim 0 1000` axis reading `0 / 500 / 1000` has those labels at roughly
z = 0 / 200 / 400.

**Workaround.** Pin limits whose span divides by the step VTK is likely to pick —
in practice a span that is a round multiple of 1000 at kilometre scale, e.g.
`--ylim 0 7000` — or leave the axis unpinned, since the automatic fit chooses
round bounds that usually agree with VTK.

**Fix plan.** Replicate VTK's step selection, emit exactly that many label
strings, and blank out alternates on axes that are short on screen. Validate
against a sweep of spans and window sizes before shipping.
