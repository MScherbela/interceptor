from flask import Flask, request, redirect
from flask_cors import CORS
from intercept_calculator import calculate_intercept, InterceptSettings

app = Flask('app')
CORS(app)
PI = 3.14159265359

def heading_to_angle(heading):
    return (90 - heading) * PI / 180


def angle_to_heading(angle):
    return (90 - (angle * 180 / PI)) % 360


@app.route("/")
def static_index():
    return redirect('static/index.html')

@app.route("/api", methods=['GET'])
def api():
    """
    Input/output units: positions in sm, headings in deg from north, speeds in kn, times in seconds
    """
    args = request.args
    intercept_params = dict(
        initial_pos=[float(args['x']), float(args['y'])],
        initial_angle=heading_to_angle(float(args['heading'])),
        initial_speed=float(args['speed']),
        target_pos=[float(args['target_x']), float(args['target_y'])],
        target_angle=heading_to_angle(float(args['target_heading'])),
        target_speed=float(args['target_speed']),
        initial_time=float(args.get('time', 0)) / 3600,
    )

    cfg = InterceptSettings(duration=float(args.get('desired_duration', 0)) / 3600,
                            distance=float(args.get('desired_distance', 1.0)),
                            fix_initial_angle=(args.get('fix_initial_angle', 'true') == 'true'),
                            n_segments=int(args.get('n_segments', 3)),
                            attack_bearing=float(args.get('attack_bearing', 0.0)) * PI / 180,
                            attack_position_angle=float(args.get('attack_position_angle', 90)) * PI/180,
                            fix_attack_side=(args.get('fix_attack_side', 'false') == 'true'))

    result,_ = calculate_intercept(**intercept_params, cfg=cfg)
    for wp in result['route']:
        wp['t'] *= 3600
        wp['duration'] *= 3600
        wp['heading'] = angle_to_heading(wp['angle'])
        wp['new_heading'] = angle_to_heading(wp['new_angle'])
        wp['target_bearing'] = (-wp['target_bearing'] * 180 / PI) % 360
    print(result)

    return result


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
