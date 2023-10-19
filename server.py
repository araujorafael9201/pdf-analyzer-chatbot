import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import worker

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  
    print('user_message', user_message)

    bot_response = worker.process_prompt(user_message)

    return jsonify({
        "botResponse": bot_response
    }), 200

# Route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try "
                           "again. If the problem persists, try using a different file"
        }), 400

    file = request.files['file']
    file_path = file.filename
    file.save(file_path)

    worker.process_document(file_path)  # Process the document using the worker module

    return jsonify({
        "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any "
                       "questions regarding it!"
    }), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, port=os.environ.get('PORT'), host=os.environ.get('HOST'))
