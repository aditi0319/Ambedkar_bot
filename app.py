from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_pipeline import create_chain

app = Flask(__name__)
CORS(app)
qa_chain = create_chain()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    response = qa_chain.invoke({"query": question})
    return jsonify({"answer": response["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

