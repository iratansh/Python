import flask_cors
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import requests
import os
import sys

app = Flask('App')
CORS(app)

@app.route('/api', methods=['GET'])
def api():
    return jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    pass