"""
Live engagement view — a scene that is fed one simulation step at a time.

`render_engagement` needs the whole run up front. This builds the same picture
incrementally instead: a simulation pushes each step as it computes it and the
window updates in place, so an engagement can be watched as it happens rather
than replayed afterwards.

The look matches `render_engagement` deliberately — same colours, same models,
same rounded metre axes, same true 1:1:1 aspect — so a live session and a still
of the same engagement are directly comparable.

    view = LiveEngagementView([{'label': 'PPO', 'role': 'pursuer'},
                               {'label': 'Target', 'role': 'target'}])
    view.start()
    while running:
        view.push(t, {'PPO': pos_m, 'Target': pos_t})
    view.finish('PPO HIT (6.8m)')

Axes cannot be fixed up front the way they can for a finished run, because the
extent is not yet known. They are recomputed as the engagement grows, but only
ever outward and always snapped to round numbers, so the grid re-lays out a
handful of times rather than on every frame.
"""

import time
from pathlib import Path

import numpy as np

from .engagement import (
    MODEL_DIR, MODEL_SIZE_FRACTION, TARGET_COLOR, TARGET_MODEL,
    _grid_kwargs, _nice_box, _ticks_for_bounds, color_for_label,
    wide_missile_path,
)

# Trail vertices kept for display. The tube is rebuilt from these on every
# refresh, so this caps that cost no matter how long the run gets.
TRAIL_POINTS = 300


def _no_bounds(actor):
    """Exclude an actor from the renderer's bounds calculation.

    Trails and models move and grow every frame. Left in the bounds they would
    drag the axes around continuously; excluded, the scene extent is decided
    solely by the pinned corners of the rounded box.
    """
    actor.SetUseBounds(False)
    return actor


class LiveEngagementView:
    """An engagement scene updated in place as the simulation advances.

    Parameters
    ----------
    tracks : list of dict
        One per body, in draw order. Keys:
        - 'label' : str, also the key used in `push()`
        - 'role'  : 'pursuer' or 'target'
        - 'color' : str, optional (defaults to the label's standard colour)
    off_screen : bool, default False
        True renders headlessly, for verification or frame capture.
    render_every : int, default 1
        Redraw only every Nth pushed step. Samples are always recorded; this
        trades visual smoothness for speed, which matters when the renderer is
        slower than the simulation.
    realtime : bool, default False
        Pace redraws against the pushed timestamps so the engagement plays at
        wall-clock speed instead of as fast as it can compute.
    bounds : (lo, hi), optional
        Known scene extent as two length-3 sequences. Supplying it keeps the
        axes fixed for the whole run; without it they grow to fit, which does
        mean the grid re-lays out once or twice early on.
    snap_bounds : bool, default True
        Whether `bounds` may be widened to land on round ticks. True suits an
        extent measured from data; False honours hand-picked limits exactly.
    """

    def __init__(self, tracks, *, window_size=(900, 900), off_screen=False,
                 line_width=400, model_scale=1.0, n_labels=10, zoom=0.85,
                 title=None, bounds=None, snap_bounds=True, show_models=True,
                 show_hud=True, render_every=1, realtime=False,
                 trail_points=TRAIL_POINTS):
        if not tracks:
            raise ValueError('at least one track is required')

        self.tracks = []
        for i, spec in enumerate(tracks):
            label = spec['label']
            role = spec.get('role', 'pursuer')
            color = spec.get('color') or (
                TARGET_COLOR if role == 'target' else color_for_label(label, i))
            self.tracks.append({'label': label, 'role': role, 'color': color})

        self.labels = [t['label'] for t in self.tracks]
        self.window_size = tuple(window_size)
        self.off_screen = off_screen
        self.line_width = line_width
        self.model_scale = model_scale
        self.n_labels = n_labels
        self.zoom = zoom
        self.title = title
        self.show_models = show_models
        self.show_hud = show_hud
        self.render_every = max(1, int(render_every))
        self.realtime = realtime
        self.trail_points = max(2, int(trail_points))

        self._points = {label: [] for label in self.labels}
        self._times = []
        self._trail_actors = {}
        self._model_actors = {}
        self._box = None
        self._pin_actor = None
        self._hud_actor = None
        self._title_actor = None
        self._base_view_angle = None
        self._pushes = 0
        self._wall_start = None
        self._sim_start = None
        self._status = None
        self.plotter = None

        self.snap_bounds = snap_bounds
        self._fixed_bounds = None
        if bounds is not None:
            lo, hi = bounds
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            if np.any(hi <= lo):
                raise ValueError(f'bounds must increase on every axis, '
                                 f'got lo={lo.tolist()} hi={hi.tolist()}')
            self._fixed_bounds = (lo, hi)

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Open the render window and build the (still empty) scene."""
        import pyvista as pv

        self.plotter = pv.Plotter(off_screen=self.off_screen,
                                  window_size=self.window_size)
        self.plotter.set_background('white')

        if self.title:
            self._title_actor = self.plotter.add_text(
                self.title, position='upper_edge', font_size=14,
                color='black', name='title', render=False)
        self.plotter.add_axes()
        self._base_view_angle = self.plotter.camera.view_angle

        # The whole scene is built before the window is mapped, with every add
        # passing render=False. show() maps the window and paints once, and it
        # leaves the camera alone, so that single first paint is already the
        # finished grid under its final framing — no empty frame, and no grid
        # drawn at a size it is about to change from.
        if self._fixed_bounds is not None:
            if self.snap_bounds:
                self._apply_box(*self._fixed_bounds)
            else:
                self._apply_exact_box(*self._fixed_bounds)

        if not self.off_screen:
            # Non-blocking: returns immediately, so the caller keeps control of
            # the simulation loop and drives the frames itself.
            self.plotter.show(interactive_update=True, auto_close=False)
        return self

    def close(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- feeding -----------------------------------------------------------

    def push(self, t, positions, status=None):
        """Record one simulation step and, on schedule, redraw.

        Parameters
        ----------
        t : float
            Simulation time in seconds.
        positions : dict
            label -> (3,) position. Labels absent from the dict are treated as
            finished and simply hold their last sample, which is what happens
            when a pursuer intercepts while the target flies on.
        status : str, optional
            Extra line for the HUD, e.g. a phase or outcome.
        """
        if self.plotter is None:
            raise RuntimeError('call start() before push()')

        self._times.append(float(t))
        if status is not None:
            self._status = status
        for label in self.labels:
            pos = positions.get(label)
            if pos is not None:
                self._points[label].append(np.asarray(pos, dtype=float))

        self._pushes += 1
        if self._pushes % self.render_every == 0:
            self.refresh()
            if self.realtime:
                self._pace(t)

    def finish(self, status=None):
        """Force a final redraw so the last step is always shown."""
        if status is not None:
            self._status = status
        if self.plotter is not None:
            self.refresh()

    def refresh(self):
        """Redraw the scene from everything pushed so far."""
        if self.plotter is None:
            return
        self._update_bounds()
        for track in self.tracks:
            pts = self._array(track['label'])
            if len(pts) >= 2:
                self._update_trail(track, pts)
                if self.show_models:
                    self._update_model(track, pts)
        if self.show_hud:
            self._update_hud()

        if self.off_screen:
            self.plotter.render()
        else:
            self.plotter.update()

    def set_title(self, text):
        """Replace the headline — the outcome is only known once a run ends."""
        self.title = text
        if self.plotter is None:
            return
        if self._title_actor is None:
            self._title_actor = self.plotter.add_text(
                text, position='upper_edge', font_size=14, color='black',
                name='title', render=False)
        else:
            self._title_actor.set_text('upper_edge', text)

    def capture(self):
        """Current frame as an RGB array, for assembling a recording."""
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    def screenshot(self, path):
        """Write the current frame to a PNG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.render()
        self.plotter.screenshot(str(path))
        return path

    # -- internals ---------------------------------------------------------

    def _array(self, label):
        pts = self._points[label]
        return np.asarray(pts, dtype=float) if pts else np.empty((0, 3))

    def _all_points(self):
        chunks = [self._array(l) for l in self.labels]
        chunks = [c for c in chunks if len(c)]
        return np.vstack(chunks) if chunks else np.empty((0, 3))

    def _scale(self):
        """Longest axis of the current box — the reference for every size."""
        if self._box is None:
            return 1.0
        b = self._box
        return max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) or 1.0

    def _update_bounds(self):
        """Re-fit the axes, but only when the data has outgrown the box."""
        if self._fixed_bounds is not None and self._box is not None:
            return
        pts = self._all_points()
        if len(pts) < 2:
            return
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if self._box is not None:
            b = self._box
            inside = all(b[2 * i] <= lo[i] and hi[i] <= b[2 * i + 1]
                         for i in range(3))
            if inside:
                return
        self._apply_box(lo, hi)

    def _apply_box(self, lo, hi):
        """Fit a rounded box around an extent and install it."""
        box, ticks = _nice_box(lo, hi, self.n_labels)
        self._set_box(box, ticks)

    def _apply_exact_box(self, lo, hi):
        """Install caller-supplied bounds verbatim, without widening them."""
        box = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        self._set_box(box, _ticks_for_bounds(box, self.n_labels))

    def _set_box(self, box, ticks):
        import pyvista as pv

        self._box = box

        # VTK re-fits the cube axes to the renderer's bounds on every render
        # (Renderer.update_bounds_axes), which would drag the ticks off their
        # round values as the trails grow. Every moving actor is excluded from
        # the bounds calculation (see _no_bounds) and an invisible point is
        # pinned at each corner of the box, so the renderer's bounds *are* the
        # round box and that re-fit becomes a no-op. This also keeps the live
        # box identical to the one render_engagement computes for the same run.
        corners = np.array([[box[i], box[2 + j], box[4 + k]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                           dtype=float)
        if self._pin_actor is None:
            self._pin_actor = self.plotter.add_mesh(pv.PolyData(corners),
                                                    opacity=0.0,
                                                    reset_camera=False,
                                                    render=False)
        else:
            self._pin_actor.mapper.dataset = pv.PolyData(corners)

        # Replacing the axes actor rather than adding a second one; the same
        # redraw that keeps the recorded animation's axes honest.
        self.plotter.remove_bounds_axes()
        self.plotter.show_grid(bounds=list(box), grid=True, location='outer',
                               bold=True, font_size=10, render=False,
                               **_grid_kwargs(ticks))

        # The axes actor's own extent includes the tick marks and labels drawn
        # just outside the box. Left in the bounds it enlarges them, VTK re-fits
        # the axes to that, and the overhang grows again on the next render —
        # ticks visibly creep off their round values over a long run. Excluding
        # it leaves the pinned corners as the only thing setting the extent.
        axes_actor = self.plotter.renderer.cube_axes_actor
        if axes_actor is not None:
            _no_bounds(axes_actor)

        self._frame_camera()

    def _frame_camera(self):
        """Point the camera at the current box, isometric and true 1:1:1."""
        box = self._box
        if box is None or self.plotter is None:
            return
        centre = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2,
                           (box[4] + box[5]) / 2])
        span = self._scale()
        self.plotter.camera.focal_point = centre
        self.plotter.camera.position = centre + span * 1.5
        self.plotter.camera.view_up = (0, 0, 1)
        self.plotter.camera.clipping_range = (0.01, span * 10)
        # zoom() is cumulative on the camera, and the box may be re-fitted
        # several times over a run, so the angle is restored first — otherwise
        # every re-fit multiplies the zoom again and the scene shrinks away.
        self.plotter.camera.view_angle = self._base_view_angle
        self.plotter.camera.zoom(self.zoom)

    def _trail_mesh(self, pts):
        import pyvista as pv
        if len(pts) > self.trail_points:
            idx = np.linspace(0, len(pts) - 1, self.trail_points, dtype=int)
            pts = pts[idx]
        radius = self._scale() * 7e-6 * self.line_width
        # A polyline tube rather than a fitted spline: this is rebuilt on every
        # frame, and spline fitting a growing path is far too slow to keep up.
        return pv.lines_from_points(pts).tube(radius=radius)

    def _update_trail(self, track, pts):
        label = track['label']
        mesh = self._trail_mesh(pts)
        actor = self._trail_actors.get(label)
        if actor is None:
            actor = self.plotter.add_mesh(mesh, color=track['color'],
                                          smooth_shading=True,
                                          reset_camera=False, render=False)
            _no_bounds(actor)
            self._trail_actors[label] = actor
        else:
            actor.mapper.dataset = mesh

    def _update_model(self, track, pts):
        from .core import _alignment_rotation, _prepare_model_mesh

        label = track['label']
        actor = self._model_actors.get(label)
        if actor is None:
            path = (MODEL_DIR / TARGET_MODEL if track['role'] == 'target'
                    else wide_missile_path())
            if not Path(path).exists():
                return
            size = self._scale() * MODEL_SIZE_FRACTION * self.model_scale
            mesh = _prepare_model_mesh(str(path), size)
            actor = self.plotter.add_mesh(mesh, color=track['color'],
                                          reset_camera=False, render=False)
            _no_bounds(actor)
            self._model_actors[label] = actor

        tail = pts[-min(5, len(pts)):]
        matrix = np.eye(4)
        matrix[:3, :3] = _alignment_rotation(tail[-1] - tail[0])
        matrix[:3, 3] = pts[-1]
        actor.user_matrix = matrix

    def _update_hud(self):
        lines = [f't = {self._times[-1]:6.2f} s']
        target = next((t for t in self.tracks if t['role'] == 'target'), None)
        if target is not None:
            tp = self._array(target['label'])
            for track in self.tracks:
                if track['role'] == 'target':
                    continue
                pp = self._array(track['label'])
                n = min(len(pp), len(tp))
                if n:
                    r = float(np.linalg.norm(pp[n - 1] - tp[n - 1]))
                    lines.append(f'{track["label"]:<6s} R = {r:9.1f} m')
        if self._status:
            lines.append(self._status)
        text = '\n'.join(lines)

        # add_text() goes through add_actor(), and pyvista's add_actor forces a
        # full render_window.Render() on the spot. Re-adding the HUD every frame
        # therefore doubled the renders and made the whole scene — grid included
        # — visibly flicker under software GL. The actor is created once and its
        # string is swapped in place after that; the caller's own render at the
        # end of refresh() is what puts it on screen.
        if self._hud_actor is None:
            self._hud_actor = self.plotter.add_text(
                text, position='lower_left', font_size=10, color='black',
                name='hud', render=False)
        else:
            self._hud_actor.set_text('lower_left', text)

    def _pace(self, t):
        """Sleep so playback tracks simulation time."""
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start, self._sim_start = now, t
            return
        target = self._wall_start + (t - self._sim_start)
        delay = target - now
        if delay > 0:
            time.sleep(delay)


def replay(engagement, view=None, **kwargs):
    """Feed a finished engagement through the live view, one step at a time.

    Exercises exactly the incremental path a simulation drives, which is what
    makes the live view testable without a simulation attached.
    """
    n = engagement.n_steps
    tracks = [{'label': t.label, 'role': t.role} for t in engagement.tracks]
    if view is None:
        view = LiveEngagementView(
            tracks, title=engagement.title(sep='   /   '), **kwargs)
        view.start()

    for i in range(n):
        positions = {}
        for track in engagement.tracks:
            if i < len(track.points):
                positions[track.label] = track.points[i]
        view.push(float(i), positions)

    view.finish()
    return view
