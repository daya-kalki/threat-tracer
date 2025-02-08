from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows frontend to communicate with Flask

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend access

# API Route to process input and return a decision
@app.route('/decision', methods=['POST'])
def process_decision():
    data = request.json  # Get JSON input from frontend

    # Sample decision logic: Approve if value > 50, otherwise Reject
    input_value = data.get("value", 0)
    decision = "Approved" if input_value > 50 else "Rejected"

    return jsonify({"decision": decision})  # Send decision output to frontend

if __name__ == '__main__':
    app.run(debug=True)  # Run Flask server at http://127.0.0.1:5000
