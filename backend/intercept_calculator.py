import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax.experimental.optimizers import adam
from dataclasses import dataclass

@dataclass
class InterceptSettings:
    min_duration: float = 0.02
    min_duration_final: float = 0.05
    duration: float = 0
    distance: float = 1.0
    fix_initial_angle: bool = True
    n_segments: int = 2
    n_steps: int = 20_000
    n_runs: int = 50
    lr: float = 5e-3
    attack_bearing: float = 0
    attack_position_angle: float = np.pi/2
    fix_attack_side: bool = False


def get_rot_matrix(phi):
    s,c = jnp.sin(phi), jnp.cos(phi)
    return jnp.array([[c,-s],[s,c]])


def get_duration(duration_param):
    return jnp.exp(duration_param)

def move(params, state, speed):
    angle = params[0]
    duration = get_duration(params[1])
    x, y, old_angle, time = state
    dx = jnp.cos(angle) * speed * duration
    dy = jnp.sin(angle) * speed * duration
    return jnp.array([x + dx, y + dy, angle, time + duration])


def calc_route(params, initial_state, speed):
    waypoints = [initial_state]
    state = initial_state
    for p in params:
        state = move(p, state, speed)
        waypoints.append(state)
    return jnp.stack(waypoints)


def softabs(x, scale=0.1):
    return jnp.sqrt(x**2 + scale**2)


def loss_func(route, target_pos, target_angle, target_speed, cfg: InterceptSettings):
    final_waypoint = route[-1]
    pos = final_waypoint[:2]
    angle = final_waypoint[2]
    durations = jnp.diff(route[:, 3])
    total_duration = route[-1, 3] - route[0, 3]

    e_target = jnp.array([jnp.cos(target_angle), jnp.sin(target_angle)])
    final_target_pos = target_pos + e_target * target_speed * total_duration

    delta_r = final_target_pos - pos
    distance = jnp.linalg.norm(delta_r)
    e_interceptor = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    e_delta_r = delta_r / (distance + 1e-6)

    loss_dist = softabs(distance - cfg.distance)
    if cfg.fix_attack_side:
        loss_attack_position = 1 - jnp.dot(e_delta_r, get_rot_matrix(cfg.attack_position_angle) @ e_target)
    else:
        loss_attack_position = 1 - jnp.dot(e_delta_r, get_rot_matrix(cfg.attack_position_angle) @ e_target)**2
    loss_attack_bearing = 1 - jnp.dot(e_delta_r, get_rot_matrix(-cfg.attack_bearing) @ e_interceptor)
    loss_total_duration = softabs(total_duration - cfg.duration, 0.5)
    loss_final_run_duration = jax.nn.sigmoid(-(durations[-1] / (0.25 * cfg.min_duration_final)))
    loss_durations = jnp.sum(jax.nn.sigmoid(-(durations / (0.25 * cfg.min_duration))))

    return (loss_dist +
            loss_attack_position +
            loss_attack_bearing +
            0.05 * loss_total_duration +
            loss_final_run_duration +
            loss_durations
            )

calc_route_batch = jax.vmap(calc_route, in_axes=[0, None, None], out_axes=0)
loss_batch = jax.vmap(loss_func, in_axes=[0, None, None, None, None], out_axes=0)


def build_value_and_grad(initial_state, speed, target_pos, target_angle, target_speed, cfg):
    def loss_from_param(param):
        route = calc_route(param, initial_state, speed)
        return loss_func(route, target_pos, target_angle, target_speed, cfg)

    return jax.vmap(jax.value_and_grad(loss_from_param))


def plot_routes(routes, losses, initial_target_pos, target_angle, target_speed, show_legend=False, axis=None):
    axis = axis or plt.gca()
    e_target = np.array([np.cos(target_angle), np.sin(target_angle)])
    arrow_size = 0.2

    colors = [f'C{i}' for i in range(10)]
    for i, (route, l) in enumerate(zip(routes, losses)):
        color = colors[i % len(colors)]
        time = route[-1][3]
        target_pos = initial_target_pos + e_target * time * target_speed
        axis.plot(route[:, 0], route[:, 1], label=f"t={time:.2f}, loss={l:.5f}", color=color, marker='o', ms=2)
        axis.arrow(*(target_pos - e_target * 0.5 * arrow_size), *(e_target * arrow_size), head_width=0.2, color=color)
    axis.arrow(*(initial_target_pos - e_target * 0.5 * arrow_size), *(e_target * arrow_size), head_width=0.2, color='k')
    if show_legend:
        axis.legend()
    axis.grid(alpha=0.5)
    axis.set_aspect('equal', adjustable='box')

def route_to_list_of_dicts(route, speed, target_pos, target_angle, target_speed):
    keys = ['x', 'y', 'angle', 't']
    output = [{k: float(v) for k, v in zip(keys, waypoint)} for waypoint in route]

    for i, wp in enumerate(output):
        if i < len(output) - 1:
            wp['new_angle'] = output[i+1]['angle']
        else:
            wp['new_angle'] = wp['angle']
        wp['duration'] = wp['t'] - route[0][3]
        wp['target_x'] = target_pos[0] + np.cos(target_angle) * target_speed * wp['duration']
        wp['target_y'] = target_pos[1] + np.sin(target_angle) * target_speed * wp['duration']
        dx, dy = wp['target_x'] - wp['x'], wp['target_y'] - wp['y']
        wp['target_dist'] = np.sqrt(dx**2 + dy**2)
        rel_angle = np.arctan2(wp['target_y'] - wp['y'], wp['target_x'] - wp['x']) - wp['angle']
        wp['target_bearing'] = rel_angle % (2 * np.pi)
        wp['speed'] = speed
        output[i] = {k: float(v) for k,v in wp.items()}
    return output


def calculate_intercept(initial_pos, initial_angle, initial_speed, target_pos, target_angle, target_speed, initial_time, cfg: InterceptSettings):
    initial_pos = np.array(initial_pos)
    target_pos = np.array(target_pos)
    initial_state = jnp.array([initial_pos[0], initial_pos[1], initial_angle, initial_time])
    target_params = (np.array(target_pos), target_angle, target_speed)

    initial_params = np.stack([np.random.uniform(0, 2 * np.pi, [cfg.n_runs, cfg.n_segments]),
                               np.random.uniform(-1, 1, [cfg.n_runs, cfg.n_segments])], axis=-1)
    grad_mask = np.ones_like(initial_params)

    opt_init, opt_update, opt_get_params = adam(lambda t: cfg.lr / (1+t/5000))
    val_grad_func = build_value_and_grad(initial_state, initial_speed, *target_params, cfg)

    if cfg.fix_initial_angle:
        initial_params[..., 0, 0] = initial_angle
        grad_mask[..., 0, 0] = 0

    @jax.jit
    def opt_step(step_nr, _opt_state):
        p = opt_get_params(_opt_state)
        _losses, g = val_grad_func(p)
        g = g * grad_mask
        return opt_update(step_nr, g, _opt_state), _losses

    opt_state = opt_init(initial_params)

    all_losses = []
    for n in range(cfg.n_steps):
        opt_state, epoch_losses = opt_step(n, opt_state)
        all_losses.append(epoch_losses)

    params = opt_get_params(opt_state)
    routes = calc_route_batch(params, initial_state, initial_speed)
    losses = loss_batch(routes, *target_params, cfg)
    ind_best = np.nanargmin(losses)
    route = routes[ind_best]
    loss = losses[ind_best]
    route_list = route_to_list_of_dicts(route, initial_speed, *target_params)

    return dict(route=route_list,
                loss=float(loss)), (routes, losses, all_losses)


if __name__ == '__main__':
    # initial_target_pos = [6, 2]
    initial_target_pos = np.random.uniform(-5,5,2)
    target_angle = np.random.uniform(0, 2 * np.pi)
    # target_angle = 3.0
    target_speed = 1.0

    cfg = InterceptSettings(fix_initial_angle=False, n_runs=6, n_segments=2,
                            attack_position_angle=-np.pi/2)
    intercept, (routes, losses, all_losses) = calculate_intercept(initial_pos=[0, 0],
                                                                initial_angle=0,
                                                                initial_speed=1.5,
                                                                target_pos=initial_target_pos,
                                                                target_angle=target_angle,
                                                                target_speed=target_speed,
                                                                initial_time=0,
                                                                cfg=cfg)
    plt.close("all")
    fig, (ax_map, ax_loss) = plt.subplots(1,2)
    plot_routes(routes, losses, initial_target_pos, target_angle, target_speed, False, axis=ax_map)
    all_losses = np.array(all_losses)
    ax_loss.plot(all_losses)
    ax_loss.set_ylim([0,1])
