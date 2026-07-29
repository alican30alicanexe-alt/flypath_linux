"""Watch an eval_v4 engagement render live, in-process, as it is computed.

    # from the simulation's own venv, which now has pyvista:
    cd /home/mmsgb/Desktop/TDTP/TDTP_13Jul/3DOF_v3
    venv/bin/python /home/mmsgb/Desktop/pytrajectory/examples/live_eval.py --realtime

    --headless          render off-screen and write a PNG (and GIF with --record)
    --realtime          pace playback against simulation time
    --seed N            episode seed (default 2026, matching the exports)
    --render-every N    redraw every Nth step; raise it if rendering lags

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(SIM_ROOT / 'runs' /
                                           'ppo_missile_v3_final.zip'))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--maneuver-mode', default='random')
    ap.add_argument('--controllers', default='PPO,PNG')
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--render-every', type=int, default=10)
    ap.add_argument('--realtime', action='store_true')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--out', default='Results_viz/live_final.png')
    ap.add_argument('--record', default=None,
                    help='also write the rendered frames to this GIF')
    args = ap.parse_args()

    from eval_v4 import _sample_episode, make_eval_config
    from pytrajectory.live import LiveEngagementView

    cfg = make_eval_config(noise_enabled=False)
    controllers = [c.strip().upper() for c in args.controllers.split(',') if c.strip()]
    init_states, maneuver = _sample_episode(args.seed, cfg, args.maneuver_mode)
    runners = build_runners(cfg, init_states, maneuver, args.model, controllers)

    tracks = [{'label': r['label'], 'role': 'pursuer'} for r in runners]
    tracks.append({'label': 'Target', 'role': 'target'})

    view = LiveEngagementView(
        tracks,
        title=f'seed {args.seed} — running',
        off_screen=args.headless,
        render_every=args.render_every,
        realtime=args.realtime,
    )
    view.start()

    frames = []
    dt = cfg.simulation.dt_sim
    print(f'seed {args.seed} | controllers {controllers} | dt {dt}s')

    for step_idx in range(args.max_steps):
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

        view.push((step_idx + 1) * dt, positions)
        if args.record and view._pushes % view.render_every == 0:
            frames.append(view.capture())

        if all(r['done'] for r in runners):
            break

    title = '   /   '.join(f'{r["label"]} {r["outcome"]} ({r["min_r"]:.1f}m)'
                           for r in runners)
    view.set_title(title)
    view.finish()
    print(title)
    for r in runners:
        print(f'  {r["label"]:4s} {r["outcome"]:8s} min_R {r["min_r"]:9.2f} m  '
              f'{r["steps"]} steps')

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
