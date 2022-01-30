import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax.experimental.optimizers import adam


def get_duration(duration_param):
    return jnp.exp(duration_param)

def move(state, params):
    angle = params[0]
    duration = get_duration(params[1])
    speed = 1.0
    x, y, old_angle, time = state
    dx = jnp.cos(angle) * speed * duration
    dy = jnp.sin(angle) * speed * duration
    return jnp.array([x + dx, y + dy, angle, time + duration])


def calc_route(initial_state, params):
    waypoints = [initial_state]
    state = initial_state
    for p in params:
        state = move(state, p)
        waypoints.append(state)
    return jnp.stack(waypoints)

def loss_distance(distance):
    min_distance = 0.2
    return 0.1 * (distance ** 2) + jnp.exp(-(distance/min_distance) ** 2)


def loss_func(route, target_pos, target_angle, target_speed):
    min_duration = 0.2
    min_duration_final = 1.0

    final_waypoint = route[-1]
    pos = final_waypoint[:2]
    angle = final_waypoint[2]
    durations = jnp.diff(route[:,3])
    total_duration = route[-1, 3] - route[0, 3]

    e_target = jnp.array([jnp.cos(target_angle), jnp.sin(target_angle)])
    target_pos = target_pos + e_target * target_speed * total_duration

    delta_r = target_pos - pos
    distance = jnp.linalg.norm(delta_r)
    e_interceptor = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    e_delta_r = delta_r / (distance + 1e-6)

    loss_dist = loss_distance(distance)
    loss_bearing = 1 - jnp.dot(e_delta_r, e_interceptor)
    loss_angle = jnp.dot(e_delta_r, e_target)**2
    loss_total_duration = 0.05*total_duration
    loss_final_run_duration = jax.nn.sigmoid(-(durations[-1] / (0.25*min_duration_final)))
    loss_durations = jnp.sum(jax.nn.sigmoid(-(durations / (0.25*min_duration))))

    return loss_dist + loss_bearing + loss_angle + 0.05*loss_total_duration + loss_final_run_duration + loss_durations


calc_route_batch = jax.vmap(calc_route, in_axes=[None, 0], out_axes=0)
loss_batch = jax.vmap(loss_func, in_axes=[0,None,None,None], out_axes=0)


def build_value_and_grad(initial_state, target_pos, target_angle, target_speed):
    def loss_from_param(param):
        route = calc_route(initial_state, param)
        return loss_func(route, target_pos, target_angle, target_speed)
    return jax.vmap(jax.value_and_grad(loss_from_param))

def plot_routes(routes, losses, initial_target_pos, target_angle, target_speed, axis=None):
    axis = axis or plt.gca()
    e_target = np.array([np.cos(target_angle), np.sin(target_angle)])
    arrow_size = 0.2

    colors = [f'C{i}' for i in range(10)]
    for i, (route, l) in enumerate(zip(routes, losses)):
        color = colors[i%len(colors)]
        time = route[-1][3]
        target_pos = initial_target_pos + e_target * time * target_speed
        axis.plot(route[:,0], route[:,1], label=f"t={time:.2f}, loss={l:.5f}", color=color, marker='o', ms=2)
        axis.arrow(*(target_pos - e_target * 0.5 * arrow_size), *(e_target * arrow_size), head_width=0.2, color=color)
    axis.arrow(*(initial_target_pos - e_target * 0.5 * arrow_size), *(e_target * arrow_size), head_width=0.2, color='k')
    axis.legend()
    axis.grid(alpha=0.5)
    axis.set_aspect('equal', adjustable='box')

def calculate_intercept(initial_pos, initial_angle, initial_speed, target_pos, target_angle, target_speed, initial_time = 0, n_segments=3, n_runs=10, lr=1e-2, n_steps=1000)
    initial_state = jnp.array([initial_pos[0], initial_pos[1], initial_angle, initial_time])
    target_params = (target_pos, target_angle, target_speed)

    initial_params = np.stack([np.random.uniform(0, 2*np.pi, [n_runs, n_segments]),
                             np.random.uniform(-1, 1, [n_runs, n_segments])], axis=-1)
    opt_init, opt_update, opt_get_params = adam(lr)
    val_grad_func = build_value_and_grad(initial_state, *target_params)

    @jax.jit
    def opt_step(step_nr, _opt_state):
        p = opt_get_params(_opt_state)
        _losses, g = val_grad_func(p)
        return opt_update(step_nr, g, opt_state), _losses

    opt_state = opt_init(initial_params)

    for n in range(n_steps):
        opt_state, _ = opt_step(n, opt_state)
    params = opt_get_params(opt_state)
    routes = calc_route_batch(initial_state, params)
    losses = loss_batch(routes, *target_params)
    ind_routes = np.argsort(losses)
    params = params[ind_routes]
    routes = routes[ind_routes]
    losses = losses[ind_routes]

    return [dict(route=r, segments=p, loss=l) for r,p,l in zip(routes, params, losses)]

