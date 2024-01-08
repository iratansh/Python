# DocumentGPT
DocumentGPT is an AI-powered document analysis and summarization tool designed to assist users in efficiently extracting key insights from a wide range of documents, including reports, articles, research papers, and more.

## Installation (Backend)
To run, create a Python virtual environment (venv) by running:

```
python3 -m venv .flaskenv
```
and activate it by running 
```
. .flaskenv/bin/activate 
```

Now, install flask using: (Flask installation docs - https://flask.palletsprojects.com/en/3.0.x/installation/)

```
pip install flask
```
to run a flask development server:

```
python -m flask
flask --app <filename> run 
```

## Current Functionalities 
1. Lets the user upload a .pdf or .txt file
2. Returns the text output in json format
