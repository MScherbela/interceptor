import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax.experimental.optimizers import adam


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


def loss_distance(distance):
    min_distance = 0.2
    return 0.1 * (distance ** 2) + jnp.exp(-(distance / min_distance) ** 2)


def loss_func(route, target_pos, target_angle, target_speed):
    min_duration = 0.02
    min_duration_final = 0.05

    final_waypoint = route[-1]
    pos = final_waypoint[:2]
    angle = final_waypoint[2]
    durations = jnp.diff(route[:, 3])
    total_duration = route[-1, 3] - route[0, 3]

    e_target = jnp.array([jnp.cos(target_angle), jnp.sin(target_angle)])
    target_pos = target_pos + e_target * target_speed * total_duration

    delta_r = target_pos - pos
    distance = jnp.linalg.norm(delta_r)
    e_interceptor = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    e_delta_r = delta_r / (distance + 1e-6)

    loss_dist = loss_distance(distance)
    loss_bearing = 1 - jnp.dot(e_delta_r, e_interceptor)
    loss_angle = jnp.dot(e_delta_r, e_target) ** 2
    loss_total_duration = 0.05 * total_duration
    loss_final_run_duration = jax.nn.sigmoid(-(durations[-1] / (0.25 * min_duration_final)))
    loss_durations = jnp.sum(jax.nn.sigmoid(-(durations / (0.25 * min_duration))))

    return (loss_dist +
            loss_bearing +
            loss_angle +
            0.05 * loss_total_duration +
            loss_final_run_duration +
            loss_durations
            )


calc_route_batch = jax.vmap(calc_route, in_axes=[0, None, None], out_axes=0)
loss_batch = jax.vmap(loss_func, in_axes=[0, None, None, None], out_axes=0)


def build_value_and_grad(initial_state, speed, target_pos, target_angle, target_speed):
    def loss_from_param(param):
        route = calc_route(param, initial_state, speed)
        return loss_func(route, target_pos, target_angle, target_speed)

    return jax.vmap(jax.value_and_grad(loss_from_param))


def plot_routes(routes, losses, initial_target_pos, target_angle, target_speed, axis=None):
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
    axis.legend()
    axis.grid(alpha=0.5)
    axis.set_aspect('equal', adjustable='box')

def route_to_list_of_dicts(route):
    keys = ['x', 'y', 'angle', 't']
    output = [{k: float(v) for k, v in zip(keys, waypoint)} for waypoint in route]
    for i, wp in enumerate(output[:-1]):
        wp['angle'] = output[i+1]['angle']
    return output

def calculate_intercept(initial_pos, initial_angle, initial_speed, target_pos, target_angle, target_speed, initial_time=0, n_segments=3,
                        n_runs=10, lr=1e-2, n_steps=5000):
    initial_pos = np.array(initial_pos)
    target_pos = np.array(target_pos)
    initial_state = jnp.array([initial_pos[0], initial_pos[1], initial_angle, initial_time])
    target_params = (np.array(target_pos), target_angle, target_speed)

    initial_params = np.stack([np.random.uniform(0, 2 * np.pi, [n_runs, n_segments]),
                               np.random.uniform(-1, 1, [n_runs, n_segments])], axis=-1)
    opt_init, opt_update, opt_get_params = adam(lr)
    val_grad_func = build_value_and_grad(initial_state, initial_speed, *target_params)

    @jax.jit
    def opt_step(step_nr, _opt_state):
        p = opt_get_params(_opt_state)
        _losses, g = val_grad_func(p)
        return opt_update(step_nr, g, _opt_state), _losses

    opt_state = opt_init(initial_params)

    all_losses = []
    for n in range(n_steps):
        opt_state, epoch_losses = opt_step(n, opt_state)
        all_losses.append(epoch_losses)

    params = opt_get_params(opt_state)
    routes = calc_route_batch(params, initial_state, initial_speed)
    losses = loss_batch(routes, *target_params)
    print(losses)
    ind_best = np.argmin(losses)
    route = routes[ind_best]
    loss = losses[ind_best]
    # params = losses[ind_best]
    duration = (route[-1, 3] - route[0, 3])
    final_target_pos = target_pos + np.array([np.cos(target_angle), np.sin(target_angle)]) * target_speed * duration
    final_distance = np.linalg.norm(final_target_pos - route[-1][:2])

    e_target = np.array([np.cos(target_angle), np.sin(target_angle)])
    delta_t = route[:,3] - route[0,3]
    distances_to_target = np.linalg.norm(route[:,:2] - (target_pos + e_target * target_speed * delta_t[:, None]), axis=1)

    route_list = route_to_list_of_dicts(route)
    for wp, d in zip(route_list, distances_to_target):
        wp['target_dist'] = float(d)

    return dict(route=route_list,
                loss=float(loss),
                duration=float(duration),
                final_target_pos=dict(x=float(final_target_pos[0]), y=float(final_target_pos[1])),
                final_distance=float(final_distance)), (routes, losses, all_losses)


if __name__ == '__main__':
    initial_target_pos = [1, 2]
    target_angle = 3.0
    target_speed = 1.0
    intercept, (routes, losses, all_losses) = calculate_intercept(initial_pos=[0, 0],
                                                                initial_angle=0,
                                                                initial_speed=2,
                                                                target_pos=initial_target_pos,
                                                                target_angle=target_angle,
                                                                target_speed=target_speed,
                                                                n_segments=3,
                                                                n_runs=5)
    plt.close("all")
    plot_routes(routes, losses, initial_target_pos, target_angle, target_speed, axis=None)
    plt.figure()
    all_losses = np.array(all_losses)
    plt.plot(all_losses)
