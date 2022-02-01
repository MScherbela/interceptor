from flask import Flask, request, send_file
from flask_cors import CORS
from intercept_calculator import calculate_intercept

app = Flask('app')
CORS(app)
PI = 3.14159265359

def heading_to_angle(heading):
    return (90 - heading) * PI / 180


def angle_to_heading(angle):
    return (90 - (angle * 180 / PI)) % 360


@app.route("/")
def static_index():
    return send_file('static/index.html')

#
# @app.route("/js/<filename>")
# def static_js(filename):
#     return send_from_directory('static/js', filename)
#
#
# @app.route("/css/<filename>")
# def static_css(filename):
#     return send_from_directory('static/css', filename)


@app.route("/api", methods=['GET'])
def api():
    args = request.args
    intercept_params = dict(
        initial_pos=[float(args['x']), float(args['y'])],
        initial_angle=heading_to_angle(float(args['heading'])),
        initial_speed=float(args['speed']),
        target_pos=[float(args['target_x']), float(args['target_y'])],
        target_angle=heading_to_angle(float(args['target_heading'])),
        target_speed=float(args['target_speed']),
        initial_time=float(args.get('time', 0)) / 3600,
        n_segments=int(args.get('n_segments', 3))
    )

    result,_ = calculate_intercept(**intercept_params)
    result['duration'] *= 3600
    for wp in result['route']:
        wp['t'] *= 3600
        wp['heading'] = angle_to_heading(wp['angle'])

    return result


if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
