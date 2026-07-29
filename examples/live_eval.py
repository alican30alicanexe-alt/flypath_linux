"""Watch an eval_v4 engagement render live, in-process, as it is computed.

    # from the simulation's own venv, which now has pyvista:
    cd /home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3
    P=/home/mmsgb/Desktop/pytrajectory/examples/live_eval.py

    venv/bin/python $P --calcgrid --realtime
    venv/bin/python $P --xlim 0 6000 --ylim -1000 7000 --zlim 10000 13000
    venv/bin/python $P --autogrid

Exactly one of three grid modes is required:

  --calcgrid          solve the engagement first, fit the box to the result,
                      then play it back. Nothing is drawn until the physics is
                      done, so the box is its final size from the first frame
                      and never moves. --realtime still paces the playback to
                      simulation time.

  --xlim/--ylim/--zlim  you supply the limits, so nothing has to be measured.
                      Renders as it computes — genuinely live, with a grid that
                      is fixed because you fixed it. Limits are used exactly as
                      given, never widened.

  --autogrid          the grid grows to fit as samples arrive. Also live, and
                      the only option when the extent truly is not known ahead
                      of time — but it does re-lay out as the run develops.

PPO and PNG are stepped in lockstep in two separate environments seeded from the
same initial states. The evader's control depends only on time and its own
state, so both environments produce an identical target track and the two
pursuers can be shown racing the same target in one scene — the live equivalent
of the PPO-vs-PNG comparison plot.

The simulation's own sources are imported read-only; nothing under 3DOF_v3 is
modified. The per-step loop below mirrors eval_v4._run_episode_direct, which
cannot be reused directly because it runs to completion with no per-step hook.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SIM_ROOT = Path('/home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3')
PYTRAJ_ROOT = Path(__file__).resolve().parent.parent

for root in (str(PYTRAJ_ROOT), str(SIM_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)


def build_runners(cfg, init_states, maneuver, model_path, controllers):
    """One independent environment per controller, all from the same start."""
    from environment import CustomEnvironment
    from eval_v4 import _make_evader_fc, MAX_STEPS
    from pursuer_fc import PursuerFC

    max_time = MAX_STEPS * cfg.simulation.dt_sim
    runners = []
    for label in controllers:
        if label == 'PPO':
            pfc = PursuerFC('two_phase', cfg, model_path=model_path,
                            max_time=max_time)
        else:
            pfc = PursuerFC('png', cfg, max_time=max_time)
        pfc.reset()

        efc = _make_evader_fc(maneuver)
        efc.reset()

        env = CustomEnvironment(cfg)
        state = env.reset(init_states)
        runners.append({
            'label': label, 'env': env, 'pfc': pfc, 'efc': efc,
            'state': state, 'done': False, 'outcome': 'TIMEOUT',
            'min_r': float(state['true_r']), 'r0': float(state['true_r']),
            'rdot_pos': 0, 'steps': 0,
        })
    return runners


def advance(runner, step_idx, cfg):
    """One simulation step, with eval_v4's termination rules."""
    from eval_v4 import CPA_MIN_STEPS, MAX_RANGE_FACTOR, RDOT_DEBOUNCE

    state = runner['state']
    acc_m = runner['pfc'].compute(state)
    acc_t = runner['efc'].compute(step_idx * cfg.simulation.dt_sim, state)
    done_hit, state = runner['env'].step(acc_m, acc_t)
    runner['state'] = state
    runner['steps'] = step_idx + 1

    R = float(state['true_r'])
    runner['min_r'] = min(runner['min_r'], R)
    alt_m = float(state['true_pos_m'][2])
    speed_m = float(np.linalg.norm(state['true_vel_m']))

    if done_hit:
        runner['done'], runner['outcome'] = True, 'HIT'
    elif alt_m < 0.0 or speed_m < 1.0:
        runner['done'], runner['outcome'] = True, 'CRASH'
    elif step_idx + 1 >= CPA_MIN_STEPS and float(state['true_r_dot']) > 0.0:
        runner['rdot_pos'] += 1
        if runner['rdot_pos'] >= RDOT_DEBOUNCE:
            runner['done'], runner['outcome'] = True, 'MISS'
    else:
        runner['rdot_pos'] = 0

    if not runner['done'] and R > MAX_RANGE_FACTOR * runner['r0']:
        runner['done'], runner['outcome'] = True, 'MISS'

    return state


def simulate(cfg, runners, max_steps, dt, on_step=None):
    """Step every runner in lockstep until all are done.

    Returns the recorded [(t, {label: position}), ...]. `on_step` is called with
    each step as it is produced, which is what makes the same loop serve both
    the render-as-you-go and the solve-then-play paths.
    """
    recorded = []
    for step_idx in range(max_steps):
        positions = {}
        for runner in runners:
            if runner['done']:
                continue
            state = advance(runner, step_idx, cfg)
            positions[runner['label']] = state['true_pos_m'].copy()
            # Every environment integrates the same evader, so whichever run is
            # still going supplies the one shared target track.
            positions['Target'] = state['true_pos_t'].copy()

        if not positions:
            break

        t = (step_idx + 1) * dt
        recorded.append((t, positions))
        if on_step is not None:
            on_step(t, positions)
        if all(r['done'] for r in runners):
            break
    return recorded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(SIM_ROOT / 'runs' /
                                           'ppo_missile_v3_final.zip'))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--maneuver-mode', default='random')
    ap.add_argument('--controllers', default='PPO,PNG')
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--render-every', type=int, default=10)
    ap.add_argument('--line-width', type=float, default=400,
                    help='trail thickness (default 400); scale-independent, '
                         'not pixels — see LiveEngagementView._trail_mesh')
    ap.add_argument('--model-scale', type=float, default=1.0,
                    help='body size multiplier (default 1.0); 1.0 draws each '
                         'model at 6%% of the scene s longest axis')
    ap.add_argument('--realtime', action='store_true')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--out', default='Results_viz/live_final.png')
    ap.add_argument('--record', default=None,
                    help='also write the rendered frames to this GIF')
    ap.add_argument('--calcgrid', action='store_true',
                    help='solve the run first, then fit a fixed grid to it')
    ap.add_argument('--autogrid', action='store_true',
                    help='grow the grid to fit as the run develops')
    for axis in 'xyz':
        ap.add_argument(f'--{axis}lim', nargs=2, type=float,
                        metavar=('MIN', 'MAX'),
                        help=f'{axis.upper()} grid limits in metres')
    args = ap.parse_args()

    limits = [args.xlim, args.ylim, args.zlim]
    chosen = [name for name, on in (('--calcgrid', args.calcgrid),
                                    ('--autogrid', args.autogrid),
                                    ('--x/y/zlim', any(v is not None
                                                       for v in limits)))
              if on]
    if len(chosen) != 1:
        ap.error('pick exactly one grid mode: --calcgrid, --autogrid, or '
                 f'--xlim/--ylim/--zlim (got {", ".join(chosen) or "none"})')
    if chosen[0] == '--x/y/zlim':
        missing = [f'--{a}lim' for a, v in zip('xyz', limits) if v is None]
        if missing:
            ap.error(f'explicit limits need every axis: missing '
                     f'{", ".join(missing)}')

    from eval_v4 import _sample_episode, make_eval_config
    from pytrajectory.live import LiveEngagementView

    cfg = make_eval_config(noise_enabled=False)
    controllers = [c.strip().upper() for c in args.controllers.split(',')
                   if c.strip()]
    init_states, maneuver = _sample_episode(args.seed, cfg, args.maneuver_mode)
    runners = build_runners(cfg, init_states, maneuver, args.model, controllers)

    tracks = [{'label': r['label'], 'role': 'pursuer'} for r in runners]
    tracks.append({'label': 'Target', 'role': 'target'})

    frames = []
    dt = cfg.simulation.dt_sim
    mode = {'--calcgrid': 'calcgrid', '--autogrid': 'autogrid'}.get(
        chosen[0], 'fixed limits')
    print(f'seed {args.seed} | controllers {controllers} | dt {dt}s | {mode}')

    def make_view(bounds, snap):
        view = LiveEngagementView(
            tracks,
            title=f'seed {args.seed} — running',
            off_screen=args.headless,
            render_every=args.render_every,
            realtime=args.realtime,
            bounds=bounds,
            snap_bounds=snap,
            line_width=args.line_width,
            model_scale=args.model_scale,
        )
        return view.start()

    def on_step(view, t, positions):
        view.push(t, positions)
        if args.record and view._pushes % view.render_every == 0:
            frames.append(view.capture())

    if args.calcgrid:
        # Solve in full before opening the window, so the box is known and
        # fixed before the first frame is drawn.
        steps = simulate(cfg, runners, args.max_steps, dt)
        pts = np.vstack([p for _, positions in steps
                         for p in positions.values()])
        view = make_view((pts.min(axis=0), pts.max(axis=0)), snap=True)
        for t, positions in steps:
            on_step(view, t, positions)
    elif args.autogrid:
        # No bounds up front: the view fits the box to whatever has arrived and
        # re-fits outward as the run develops.
        view = make_view(None, snap=True)
        simulate(cfg, runners, args.max_steps, dt,
                 on_step=lambda t, p: on_step(view, t, p))
    else:
        lo = [args.xlim[0], args.ylim[0], args.zlim[0]]
        hi = [args.xlim[1], args.ylim[1], args.zlim[1]]
        view = make_view((lo, hi), snap=False)
        simulate(cfg, runners, args.max_steps, dt,
                 on_step=lambda t, p: on_step(view, t, p))

    title = '   /   '.join(f'{r["label"]} {r["outcome"]} ({r["min_r"]:.1f}m)'
                           for r in runners)
    view.set_title(title)
    view.finish()
    print(title)
    for r in runners:
        print(f'  {r["label"]:4s} {r["outcome"]:8s} min_R {r["min_r"]:9.2f} m  '
              f'{r["steps"]} steps')
    print(f'  grid {[int(b) for b in view._box]}')

    # Fixed limits are used exactly as given, so a run that leaves the box is
    # the user's to know about rather than silently cropped.
    pts = np.vstack([view._array(l) for l in view.labels])
    box = view._box
    outside = [ax for i, ax in enumerate('XYZ')
               if pts[:, i].min() < box[2 * i] or pts[:, i].max() > box[2 * i + 1]]
    if outside:
        print(f'  WARNING: track leaves the grid on {", ".join(outside)} — '
              f'widen the limits to see all of it')

    if args.headless:
        print('wrote', view.screenshot(args.out))
    if args.record:
        try:
            import imageio
        except ImportError:
            print('--record needs imageio; install it or drop the flag')
        else:
            frames.append(view.capture())
            imageio.mimsave(args.record, frames, fps=20, loop=0)
            print(f'wrote {args.record} ({len(frames)} frames)')

    if not args.headless:
        print('\nclose the window to exit')
        view.plotter.show(auto_close=False)
    view.close()


if __name__ == '__main__':
    main()
