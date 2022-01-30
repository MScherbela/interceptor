from flask import Flask, send_from_directory

app = Flask('app')

@app.route("/")
def static_index():
    return send_from_directory('static', 'index.html')

@app.route("/js/<filename>")
def static_js(filename):
    return send_from_directory('static/js', filename)

@app.route("/css/<filename>")
def static_css(filename):
    return send_from_directory('static/css', filename)

@app.route("/api", methods=['GET', 'POST'])
def api():
    return dict(a=1)

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)