"""
Engagement-simulation ingest for pytrajectory.

Loads pursuer/evader engagement runs produced by an external 3DOF simulation
and turns them into pytrajectory scenes. The loader is driven by the file's own
schema rather than a fixed controller list, so a run with one pursuer and a run
with five load the same way.

Export schema (.npz)
--------------------
pos_t       (N, 3)  evader/target positions
pos_m0..k   (M, 3)  one array per pursuer; M may differ per pursuer and from N,
                    because a pursuer's run stops at intercept
labels      (k+1,)  object array of pursuer labels, ordered to match pos_m*
meta        ()      JSON string: case, source, seed, episode, runs[]
version     ()      schema version int

Positions are metres in a Z-up frame, which is what pytrajectory renders, so
they pass through without a sign flip.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# Matched to the simulation's own matplotlib output (tab:blue / tab:green) so a
# controller keeps its colour across both toolchains.
LABEL_COLORS = {
    'PPO': '#1f77b4',
    'PNG': '#2ca02c',
    'ZERO': '#7f7f7f',
    'ZEM': '#9467bd',
}
FALLBACK_PALETTE = ('#9467bd', '#8c564b', '#17becf', '#bcbd22')
TARGET_COLOR = '#ff0000'
TARGET_LABEL = 'Target'

PANE_COLOR = '#e8e8e8'

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
TARGET_MODEL = 'f-16.mat'
PURSUER_MODEL = 'missile.mat'

# The stock missile mesh is 46.8 x 0.75 x 0.75 — a 62:1 needle. Sized to a
# multi-kilometre engagement its cross-section lands well under one pixel and
# the body is invisible. A widened copy is generated on first use; the original
# mesh is never modified.
MISSILE_ASPECT = 6.0

# Model length as a fraction of the scene's longest axis. An engagement spans
# kilometres, so a true-to-life 3 m missile would be invisible; bodies are drawn
# oversized to stay legible, as they are in the simulation's own animations.
MODEL_SIZE_FRACTION = 0.06


def color_for_label(label, index=0):
    """Stable colour for a controller label, independent of track order.

    Keyed on the label rather than position because different exporters write
    the controllers in different orders; an order-based palette would swap
    colours between two views of the same engagement.
    """
    up = str(label).upper()
    for key, color in LABEL_COLORS.items():
        if key in up:
            return color
    return FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)]


@dataclass
class Track:
    """One body's path through an engagement."""

    label: str
    points: np.ndarray          # (N, 3) metres, Z up
    color: str
    role: str                   # 'pursuer' or 'target'
    outcome: str = None         # 'HIT' / 'MISS', pursuers only
    min_r: float = None         # recorded closest approach, metres
    steps: int = None

    def __len__(self):
        return len(self.points)


@dataclass
class Engagement:
    """A single engagement: one target and one or more pursuers."""

    target: Track
    pursuers: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    source_path: Path = None

    @property
    def tracks(self):
        """All tracks, pursuers first then the target."""
        return list(self.pursuers) + [self.target]

    @property
    def case(self):
        return self.meta.get('case', 'Engagement')

    @property
    def n_steps(self):
        return max(len(t) for t in self.tracks)

    def title(self, sep=' | '):
        """Headline matching the simulation's own plot title.

        e.g. "PPO HIT (6.2m) | PNG MISS (7143.4m)"

        `sep` is adjustable because VTK's built-in title font draws '|' as
        whitespace, which silently runs the two outcomes together on screen.
        """
        parts = [
            f'{p.label} {p.outcome} ({p.min_r:.1f}m)'
            for p in self.pursuers
            if p.outcome is not None and p.min_r is not None
        ]
        return sep.join(parts) if parts else self.case

    def range_series(self, pursuer):
        """Range from `pursuer` to the target at each shared step, in metres.

        A pursuer's run ends at intercept while the target flies on, so the
        series is truncated to the shorter of the two.
        """
        n = min(len(pursuer), len(self.target))
        return np.linalg.norm(pursuer.points[:n] - self.target.points[:n], axis=1)

    def verify(self, tol=1e-6):
        """Recompute closest approach from geometry and compare with the export.

        The rendered picture and the printed numbers must come from the same
        data; this recomputes min range from the positions actually being drawn
        and checks it against the `min_r` the simulation recorded.

        Returns
        -------
        list of dict
            One entry per pursuer with 'label', 'recorded', 'recomputed',
            'delta' and 'ok'.
        """
        report = []
        for p in self.pursuers:
            recomputed = float(self.range_series(p).min())
            recorded = p.min_r
            delta = None if recorded is None else abs(recomputed - recorded)
            report.append({
                'label': p.label,
                'recorded': recorded,
                'recomputed': recomputed,
                'delta': delta,
                'ok': delta is not None and delta <= max(tol, abs(recorded) * 1e-6),
            })
        return report


def load_npz(path):
    """Load an engagement from the simulation's .npz export.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Engagement
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Engagement export not found: {path}')

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if 'pos_t' not in keys:
            raise ValueError(
                f"{path.name} has no 'pos_t' (target track). "
                f'Found: {sorted(keys)}'
            )

        version = int(z['version']) if 'version' in keys else SCHEMA_VERSION
        if version > SCHEMA_VERSION:
            raise ValueError(
                f'{path.name} is schema version {version}, but this build '
                f'understands up to {SCHEMA_VERSION}.'
            )

        meta = json.loads(str(z['meta'])) if 'meta' in keys else {}

        # Pursuer arrays are discovered by scanning pos_m<i> upward rather than
        # trusting len(labels), so a truncated or missing label list still
        # yields every track that was actually exported.
        pursuer_arrays = []
        i = 0
        while f'pos_m{i}' in keys:
            pursuer_arrays.append(np.asarray(z[f'pos_m{i}'], dtype=float))
            i += 1

        labels = [str(x) for x in z['labels']] if 'labels' in keys else []
        target_points = np.asarray(z['pos_t'], dtype=float)

    if not pursuer_arrays:
        raise ValueError(f'{path.name} contains no pos_m* pursuer tracks.')

    runs = {str(r.get('label')): r for r in meta.get('runs', [])}

    pursuers = []
    for idx, pts in enumerate(pursuer_arrays):
        label = labels[idx] if idx < len(labels) else f'Pursuer {idx}'
        run = runs.get(label, {})
        pursuers.append(Track(
            label=label,
            points=_validate_points(pts, label),
            color=color_for_label(label, idx),
            role='pursuer',
            outcome=run.get('result'),
            min_r=run.get('min_r'),
            steps=run.get('steps'),
        ))

    target = Track(
        label=TARGET_LABEL,
        points=_validate_points(target_points, TARGET_LABEL),
        color=TARGET_COLOR,
        role='target',
    )

    return Engagement(target=target, pursuers=pursuers, meta=meta,
                      source_path=path)


def load_engagement(path):
    """Load an engagement from any supported export format.

    Currently dispatches on suffix; .npz is the only format the simulation
    writes today.
    """
    path = Path(path)
    if path.suffix.lower() == '.npz':
        return load_npz(path)
    raise ValueError(
        f'Unsupported engagement format: {path.suffix!r}. Expected .npz'
    )


def load_episode_dir(directory):
    """Load every episode export in a directory, sorted by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f'Not a directory: {directory}')
    files = sorted(directory.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No .npz exports found in {directory}')
    return [load_engagement(f) for f in files]


def render_engagement(eng, out_path=None, window_size=(900, 900), view='iso',
                      line_width=400, off_screen=True, show_markers=True,
                      title=None, n_labels=10, show_panes=False,
                      show_models=True, model_scale=1.0, zoom=0.85):
    """Render an engagement as a static 3D scene.

    Mirrors the simulation's own comparison plot: one coloured path per
    controller, the target in red, a black dot at the launch point and a cross
    at each track's final position, titled with the outcome of every pursuer.

    Unlike the simulation's matplotlib figure, the axes here keep a true 1:1:1
    aspect ratio, so the geometry is not visually distorted.

    Parameters
    ----------
    eng : Engagement
    out_path : str or Path, optional
        Where to write a PNG. Required when `off_screen` is True.
    view : str, default 'iso'
        Camera preset passed through to the renderer.
    line_width : int, default 400
        Trail thickness. This is *not* pixels: the renderer sets tube radius to
        ``global_range * 7e-6 * line_width``, a scale-independent ratio. An
        engagement spans kilometres, so small values are sub-pixel and the
        trails disappear entirely.
    show_markers : bool, default True
        Draw launch-point and endpoint markers.
    n_labels : int, default 10
        Tick-label budget for the longest axis. Shorter axes get proportionally
        fewer, and the box is widened to land every tick on a round number, so
        the actual count per axis varies.
    show_panes : bool, default False
        Fill the three rear faces of the box with a light grey. Off by default:
        the panes are coplanar with the grid and swallow segments of the black
        box lines, which reads as a rendering fault.
    show_models : bool, default True
        Place a 3D model at each track's final position: an F-16 for the
        target, a missile for every pursuer.
    model_scale : float, default 1.0
        Multiplier on model size relative to the scene extent.
    zoom : float, default 0.85
        Camera zoom applied after framing. Below 1.0 pulls back to leave room
        for the axis tick labels, which sit outside the bounding box.

    Returns
    -------
    Path or pyvista.Plotter
        The written PNG path, or the live plotter when `off_screen` is False.
    """
    from .core import flypath3d_multi

    trajectories = [
        {'data': t.points, 'color': t.color, 'label': t.label,
         'line_width': line_width}
        for t in eng.tracks
    ]

    all_pts = np.vstack([t.points for t in eng.tracks])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    scale = float(np.max(hi - lo)) or 1.0
    box, ticks = _nice_box(lo, hi, n_labels)

    plotter = flypath3d_multi(
        trajectories,
        title=title if title is not None else eng.title(sep='   /   '),
        show_grid=False,          # replaced below with metre-labelled axes
        show_axes=False,
        off_screen=off_screen,
        return_plotter=True,
        view=view,
        window_size=tuple(window_size),
        face='path',
        # Frame the rounded box, not the raw data extent, so the drawn grid is
        # fully in view rather than clipped at the edges.
        xlim=(box[0], box[1]), ylim=(box[2], box[3]), zlim=(box[4], box[5]),
    )

    if show_models:
        _add_models(plotter, eng, scale * MODEL_SIZE_FRACTION * model_scale)

    if show_markers:
        # Launch point: the pursuers start together, so the first pursuer's
        # first sample is the shared origin.
        _add_sphere(plotter, eng.pursuers[0].points[0], scale * 0.008, 'black')
        # A model already marks where its track ended, and a cross drawn at the
        # same point simply covers it up, so the two are mutually exclusive.
        if not show_models:
            for t in eng.tracks:
                _add_cross(plotter, t.points[-1], scale * 0.018, t.color)

    if show_panes:
        _add_panes(plotter, box, PANE_COLOR)

    plotter.show_grid(bounds=list(box), grid=True, location='outer',
                      bold=True, font_size=10, **_grid_kwargs(ticks))
    plotter.add_axes()
    # The grid's tick labels are drawn outside the box, so a camera framed
    # exactly on the bounds clips them at the window edge.
    plotter.camera.zoom(zoom)

    if not off_screen:
        return plotter

    if out_path is None:
        raise ValueError('out_path is required when off_screen=True')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def animate_engagement(eng, out_path, window_size=(900, 900), view='iso',
                       line_width=400, duration=8.0, model_scale=1.0,
                       title=None, n_labels=10, zoom=0.85):
    """Render an engagement as an animated GIF with growing trails.

    Every body advances on one shared frame clock, so the relative timing you
    see is the simulation's own. Tracks of unequal length — a pursuer that
    intercepted early against a target that flew on — hold at their final
    sample rather than being trimmed, so no run is cropped.

    The axes match `render_engagement` exactly — same rounded metre ticks, same
    1:1:1 aspect — so a still and a frame of its animation are directly
    comparable.

    Parameters
    ----------
    duration : float, default 8.0
        Animation length in seconds.

    Returns
    -------
    Path to the written GIF.
    """
    from .core import flypath3d_multi

    trajectories = [
        {'data': t.points, 'color': t.color, 'label': t.label,
         'line_width': line_width}
        for t in eng.tracks
    ]

    # Sizes are expressed relative to the renderer's own 0.05 model fraction.
    rel_scale = (MODEL_SIZE_FRACTION / 0.05) * model_scale
    models = []
    for i, track in enumerate(eng.tracks):
        path = MODEL_DIR / TARGET_MODEL if track.role == 'target' \
            else wide_missile_path()
        if Path(path).exists():
            models.append({'path': str(path), 'trajectory_index': i,
                           'color': track.color, 'scale': rel_scale})

    all_pts = np.vstack([t.points for t in eng.tracks])
    box, ticks = _nice_box(all_pts.min(axis=0), all_pts.max(axis=0), n_labels)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flypath3d_multi(
        trajectories,
        models=models or None,
        title=title if title is not None else eng.title(sep='   /   '),
        animate=True,
        save_animation=str(out_path),
        off_screen=True,
        speed=duration,
        trail=True,
        view=view,
        window_size=tuple(window_size),
        show_grid=True,
        show_axes=True,
        xlim=(box[0], box[1]), ylim=(box[2], box[3]), zlim=(box[4], box[5]),
        grid_kwargs=_grid_kwargs(ticks),
        zoom=zoom,
        # A 3DOF point-mass sim carries no attitude state, so bodies are aimed
        # along their velocity tangent. face='data' would also index columns
        # 3/4/5 for pitch/yaw/roll, which plain (N,3) tracks do not have.
        face='path',
    )
    return out_path


def _grid_kwargs(ticks):
    """Axis styling shared by the static and animated paths.

    Both go through the same helper so a still and a frame of its animation
    carry identical titles and tick positions.
    """
    return {
        'xtitle': 'X (m)', 'ytitle': 'Y (m)', 'ztitle': 'Z (m)',
        'n_xlabels': ticks[0], 'n_ylabels': ticks[1], 'n_zlabels': ticks[2],
        'fmt': '%.0f',
    }


STEP_MANTISSAS = (1.0, 2.0, 2.5, 5.0)


def _candidate_steps(span):
    """Round tick intervals worth considering for an axis of `span`, ascending.

    Restricted to 1, 2, 2.5 and 5 times a power of ten so ticks read as 2000 /
    2500 / 3000 rather than the raw data extent divided into equal parts.
    """
    if span <= 0:
        return [1.0]
    top = int(np.floor(np.log10(span)))
    return sorted(m * 10.0 ** e
                  for e in range(top - 2, top + 2)
                  for m in STEP_MANTISSAS)


def _fit_axis(lo, hi, max_div):
    """Snap one axis outward to round ticks, using the least padding possible.

    Picks the *smallest* step whose division count still fits `max_div`. A finer
    step can only tighten the snapped bounds, so minimising the step minimises
    the empty space around the data — while `max_div` stops a short axis from
    collecting more grid lines than it has room to label.

    Returns (lo, hi, n_labels); the interval only ever grows, so no sample is
    ever cropped.
    """
    span = float(hi) - float(lo)
    best = None
    for step in _candidate_steps(max(span, 1e-9)):
        a = np.floor(lo / step) * step
        b = np.ceil(hi / step) * step
        if b - a < step / 2:              # degenerate axis (a flat trajectory)
            b = a + step
        n_div = int(round((b - a) / step))
        if n_div < 1 or n_div > max_div:
            continue
        if best is None or (b - a) < (best[1] - best[0]):
            best = (float(a), float(b), n_div + 1)
    if best is None:
        # Every candidate overshot max_div: fall back to the coarsest.
        step = _candidate_steps(max(span, 1e-9))[-1]
        a = float(np.floor(lo / step) * step)
        b = float(np.ceil(hi / step) * step)
        best = (a, max(b, a + step), 2)
    return best


def _nice_box(lo, hi, target_n):
    """Expand a data extent outward until every axis ends on a round tick.

    Returns
    -------
    (box, ticks)
        box   : [xmin, xmax, ymin, ymax, zmin, zmax]
        ticks : per-axis label count that lands exactly on the round steps

    The box only ever grows, so no sample is cropped, and each axis is snapped
    independently — the 1:1:1 aspect is untouched because nothing is rescaled,
    the drawn bounds just move out to the next round number.

    Label density is shared out in proportion to each axis's span. Under a true
    1:1:1 aspect a short axis is physically short on screen, so giving every
    axis the same tick count crowds the short one into overlapping text.
    """
    spans = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    longest = max(float(spans.max()), 1e-9)

    box, ticks = [], []
    for axis in range(3):
        share = float(spans[axis]) / longest
        max_div = max(3, int(round(target_n * share)))
        a, b, n_labels = _fit_axis(float(lo[axis]), float(hi[axis]), max_div)
        box.extend([a, b])
        ticks.append(n_labels)
    return box, ticks


def _add_panes(plotter, box, color):
    """Fill the three rear faces of the bounding box with a flat colour.

    Only the far faces are drawn (x-min, y-min, z-min), which are the ones
    behind the data for the default isometric camera — the same three walls
    matplotlib shades. Each pane is nudged very slightly outward so it does not
    z-fight with the grid lines drawn on the same plane.
    """
    import pyvista as pv

    x0, x1, y0, y1, z0, z1 = box
    eps = max(x1 - x0, y1 - y0, z1 - z0) * 1e-3

    def quad(corners):
        return pv.PolyData(np.asarray(corners, dtype=float),
                           np.array([4, 0, 1, 2, 3]))

    panes = [
        quad([(x0 - eps, y0, z0), (x0 - eps, y1, z0),      # x-min wall
              (x0 - eps, y1, z1), (x0 - eps, y0, z1)]),
        quad([(x0, y0 - eps, z0), (x1, y0 - eps, z0),      # y-min wall
              (x1, y0 - eps, z1), (x0, y0 - eps, z1)]),
        quad([(x0, y0, z0 - eps), (x1, y0, z0 - eps),      # floor
              (x1, y1, z0 - eps), (x0, y1, z0 - eps)]),
    ]
    for pane in panes:
        plotter.add_mesh(pane, color=color, lighting=False)


def wide_missile_path(aspect=MISSILE_ASPECT):
    """Path to a widened copy of the missile mesh, generated on first use.

    The stock mesh is too slender to see at engagement scale. Only the two
    minor axes are inflated, so length — the dimension that carries the
    orientation cue — stays true. The original models/missile.mat is untouched.
    """
    out = MODEL_DIR / f'missile-wide-{aspect:g}.mat'
    if out.exists():
        return out

    from scipy.io import loadmat, savemat

    mat = loadmat(MODEL_DIR / PURSUER_MODEL)
    V = np.asarray(mat['V'], dtype=float)
    extents = np.ptp(V, axis=0)
    long_axis = int(np.argmax(extents))
    minor = [i for i in range(3) if i != long_axis]
    width = max(extents[minor].max(), 1e-9)
    factor = (extents[long_axis] / aspect) / width

    if factor > 1.0:                    # only ever fatten, never slim
        centre = V.mean(axis=0)
        for i in minor:
            V[:, i] = centre[i] + (V[:, i] - centre[i]) * factor

    savemat(out, {'V': V, 'F': np.asarray(mat['F'])})
    return out


def _add_models(plotter, eng, target_size):
    """Place a 3D model at the final position of every track.

    Each model is aimed along its own final heading. The mesh is anchored at
    its tail, matching the package convention that a trail terminates where the
    body begins, so the model reads as continuing forward out of its trail.
    """
    from .core import _alignment_rotation, _prepare_model_mesh

    for track in eng.tracks:
        path = MODEL_DIR / TARGET_MODEL if track.role == 'target' \
            else wide_missile_path()
        if not Path(path).exists():
            continue

        # Average the last few segments: a single final difference can be
        # near-zero at intercept, which would leave the model unrotated.
        tail = track.points[-min(5, len(track.points)):]
        direction = tail[-1] - tail[0]

        matrix = np.eye(4)
        matrix[:3, :3] = _alignment_rotation(direction)
        matrix[:3, 3] = track.points[-1]

        actor = plotter.add_mesh(_prepare_model_mesh(str(path), target_size),
                                 color=track.color)
        actor.user_matrix = matrix


def _add_sphere(plotter, center, radius, color):
    """Small solid marker at a point."""
    import pyvista as pv
    plotter.add_mesh(pv.Sphere(radius=radius, center=center), color=color,
                     smooth_shading=True)


def _add_cross(plotter, center, size, color):
    """Three orthogonal bars marking a track endpoint.

    A 3D cross rather than a flat 'x' so it stays legible from any camera
    angle, where a screen-aligned glyph would vanish edge-on.
    """
    import pyvista as pv
    center = np.asarray(center, dtype=float)
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = size
        plotter.add_mesh(
            pv.Line(center - d, center + d).tube(radius=size * 0.18),
            color=color,
        )


def _validate_points(pts, label):
    """Check a track is (N, 3) and finite before it reaches the renderer."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"track '{label}' must be (N, 3), got {pts.shape}")
    if len(pts) < 2:
        raise ValueError(f"track '{label}' needs at least 2 points, got {len(pts)}")
    if not np.isfinite(pts).all():
        bad = int((~np.isfinite(pts)).any(axis=1).sum())
        raise ValueError(f"track '{label}' has {bad} non-finite sample(s)")
    return pts
