from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route("/")
@app.route("/about")
def about():
    return render_template("index.html", title="Brian Frechette - About")

@app.route("/brian_ai")
def brian_ai():
    return render_template("brian_ai.html", title="Brian.AI - ChatBot")


if __name__ == "__main__":
    app.run(debug=True)