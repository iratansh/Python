"""
This program recieves input from a webpage and returns a GPT generated response back to the frontend to be displayed 
Author: Ishaan Ratanshi
Club Project for the Google Development Student Club
"""

from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from PyPDF2 import PdfReader
from openai import OpenAI

OPENAI_API_KEY = 'INPUT API KEY'
client = OpenAI(api_key=OPENAI_API_KEY)
model_id = 'gpt-3.5-turbo-instruct'

app = Flask(__name__)
CORS(app)

extracted_text = None  

def extract_text(file):
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        if reader.is_encrypted:
            return jsonify({'error': 'Encrypted PDFs are not supported'}), None

        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()

        if not text.strip():
            return jsonify({'error': 'No text found in the PDF'}), None

        return None, text  # Return None for error and text for success

    elif file.filename.endswith('.txt'):
        try:
            text = file.read().decode('utf-8')
            return None, text  # Return None for error and text for success

        except UnicodeDecodeError:
            return jsonify({'error': 'Unable to decode text file. Ensure it is UTF-8 encoded.'}), None

    else:
        return jsonify({'error': 'Unsupported file type'}), None

@app.route('/', methods=['POST', 'GET'])
def upload():
    global extracted_text  

    if request.method == 'GET':
        return render_template("upload.html")

    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file provided'}), 400

    error_response, text = extract_text(file)
    if error_response:
        return error_response

    extracted_text = text if text else ""  

    return jsonify({'text': text})

@app.route('/chat', methods=['POST'])
def process_message():
    global extracted_text  

    data = request.get_json()
    user_message = data.get('message')

    try:
        prompt = extracted_text + "\n" + user_message if extracted_text else user_message

        response = client.completions.create(
            model=model_id,
            prompt=prompt,
            max_tokens=100
        )
        bot_reply = response.choices[0].text.strip()
        print("Bot Reply:", bot_reply)

        return jsonify({'reply': bot_reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)