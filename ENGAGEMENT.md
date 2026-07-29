# Engagement visualization — 3DOF PPO missile sim

Turns pursuer/evader engagement runs from the external 3DOF simulation
(`~/Desktop/TDTP/TDTP_13Jul/3DOF_v3`) into pytrajectory scenes: stills, animated
GIFs, and a **live window that renders in-process while the simulation
computes**.

Nothing under `3DOF_v3` is modified. Its modules are imported read-only, and the
one loop that needed a per-step hook was copied here rather than edited in place.

---

## Live mode

Runs the simulation and draws it as it goes, in one process.

```bash
cd /home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3
venv/bin/python /home/mmsgb/Desktop/pytrajectory/examples/live_eval.py \
    --calcgrid --realtime
```

**Use the simulation's `venv/bin/python`.** That is the interpreter with
torch/SB3 *and* pyvista. The pytrajectory venv has no torch and will fail on
import.

PPO and PNG are stepped in lockstep in two separate environments seeded from the
same initial states. The evader's control depends only on time and its own
state, so both produce an identical target track and the two pursuers can be
shown racing one target — the live equivalent of the PPO-vs-PNG comparison plot.

Expected output for the default seed:

```
seed 2026 | controllers ['PPO', 'PNG'] | dt 0.01s | calcgrid
PPO HIT (7.2m)   /   PNG MISS (5682.2m)
  PPO  HIT      min_R      7.16 m  1469 steps
  PNG  MISS     min_R   5682.25 m   931 steps
  grid [0, 6000, -1000, 7000, 10000, 13000]
```

Those figures match `ep001_seed2026.npz` exactly — if you see them, the physics
path is intact and the viewer is only watching.

### Grid modes — exactly one is required

| mode | when the box is decided | live? |
|---|---|---|
| `--calcgrid` | solves the run first, fits the box, then plays back | playback |
| `--xlim/--ylim/--zlim` | you supply it; used exactly as given, never widened | yes |
| `--autogrid` | grows to fit as samples arrive | yes |

`--calcgrid` and explicit limits give a grid that is final from the first frame.
`--autogrid` re-lays out as the run develops (4 times on the default seed) — use
it only when the extent genuinely is not knowable up front.

Explicit limits that are too small are honoured, not silently widened: the run
is drawn outside the box and a warning names the axes.

### Options

```bash
--seed N              episode seed (default 2026)
--realtime            pace playback to simulation time (~15 s for seed 2026)
--render-every N      redraw every Nth step (default 10; raise if it stutters)
--controllers PPO     default PPO,PNG
--maneuver-mode M     none | fixed | random
--model PATH.zip      default runs/ppo_missile_v3_final.zip
--line-width W        trail thickness (default 400)
--model-scale S       body size multiplier (default 1.0)
--headless --out P    render off-screen to a PNG
--record OUT.gif      also write the rendered frames to a GIF
```

`--line-width` is **not pixels**: tube radius is `scene_range × 7e-6 ×
line_width`, so trails look the same whether the engagement spans 5 km or 50 km.
`--model-scale` multiplies a base of 6% of the scene's longest axis
(`MODEL_SIZE_FRACTION`) — bodies are deliberately oversized, since a true 3 m
missile is sub-pixel at kilometre scale.

---

## Rendering saved runs

```python
import pytrajectory as pt

eng = pt.load_engagement('EvalTrajectories/20260728_151647/ep001_seed2026.npz')
pt.render_engagement(eng, 'out.png')                     # still
pt.animate_engagement(eng, 'out.gif', duration=8.0)      # animation

for row in eng.verify():                                 # geometry vs recorded
    print(row['label'], row['recorded'], row['recomputed'], row['delta'])
```

`load_episode_dir(path)` loads a whole export directory.

Stills and animations share one axis helper, so a still and a frame of its
animation carry identical titles and ticks. The live view converges on the same
box for the same run — verified on all 16 episodes.

---

## Export schema (`.npz`)

| key | shape | meaning |
|---|---|---|
| `pos_t` | (N, 3) | evader/target positions |
| `pos_m0..k` | (M, 3) | one array per pursuer; M may differ per pursuer and from N |
| `labels` | (k+1,) | object array of pursuer labels, ordered to match `pos_m*` |
| `meta` | () | JSON string: case, source, seed, episode, `runs[]` |
| `version` | () | schema version int |

Positions are metres in a **Z-up** frame, which is what pytrajectory renders, so
they pass through without a sign flip. Pursuer arrays are found by scanning
`pos_m<i>` upward rather than trusting `len(labels)`, so a truncated label list
still yields every exported track. A pursuer's run stops at intercept while the
target flies on — unequal lengths are expected and are held, never trimmed.

---

## Design rules

- **True 1:1:1 aspect, always.** No vertical exaggeration, so a 5° dive looks
  like 5°. Wasted space is dealt with by tightening the box, not by stretching.
- **Round tick values.** Axes are widened to land every tick on a round number
  (1/2/2.5/5 × 10ⁿ), choosing the smallest step that fits the label budget so
  padding stays minimal. The box only ever grows — no sample is ever cropped.
- **Visual matches data.** `Engagement.verify()` recomputes closest approach
  from the positions actually drawn and checks it against the recorded `min_R`.
- **Colours follow the controller, not track order.** PPO is tab:blue and PNG is
  tab:green, matching the simulation's own matplotlib output, because different
  exporters write the controllers in different orders.
- **Originals untouched.** `models/missile.mat` is never modified; the widened
  copy needed for legibility (the stock mesh is a 62:1 needle) is generated
  alongside it on first use.

---

## Tests

```bash
trajectory.venv/bin/python tests/engagement_axes_check.py
trajectory.venv/bin/python tests/live_replay_check.py
```

- `engagement_axes_check` — no axis crops data, every tick is an exact multiple
  of its step, `min_R` matches the recording, across all 16 episodes.
- `live_replay_check` — replaying a finished run through the live API reaches
  the same box as the static renderer, the renderer's bounds stay pinned to it
  so ticks cannot drift, and the ranges stay exact.

Both skip with a message if the export directory is absent.

---

## Environment

Two interpreters, both Python 3.11.2 with matching numpy/scipy:

| | `3DOF_v3/venv` | `pytrajectory/trajectory.venv` |
|---|---|---|
| torch / SB3 / gymnasium | yes | no |
| pyvista / vtk | **added** | yes |
| pandas | no | yes |

pyvista, vtk and imageio were installed into the simulation venv so it can
render in-process. `pandas` is imported lazily in `pytrajectory/io.py` (it is
only needed for CSV input), so the package imports fine without it.

Rendering is software GL on this machine: about 52 ms/frame at 900×900. Raise
`--render-every` if the simulation outruns the renderer.
