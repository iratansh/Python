"""
HelpBot: A multi-purpose chatbot application
"""

import spacy
import requests
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from datetime import datetime

nlp = spacy.load("en_core_web_md")
API_KEY = '725fc8d670f044b4b6cab371441f9d59'
help_bot = ChatBot('help_bot')
last_result = None  # Store the last_result for future calculations / responses

def get_weather(city, state=None, country=None):
    """
    Access weather data
    Inputs: city, state, country
    Returns: weather data
    """
    print(f"Location: {city}, State: {state}, Country: {country}")
    base_url = 'https://api.weatherbit.io/v2.0/current'
    params = {'city': city, 'key': API_KEY}
    if state:
        params['state'] = state
    if country:
        params['country'] = country

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            list_of_locations = data.get('data', [])

            if len(list_of_locations) > 1:
                return "Error: More than 1 location matches the input."
            elif len(list_of_locations) == 1:
                city_weather = list_of_locations[0]
                return f"In {city}, {state if state else ''} {country if country else ''}, the current weather is: {city_weather['weather']['description']}, temperature: {city_weather['temp']}°C."
            else:
                return "Location not found."
        else:
            return "Failed to get weather data."
    except Exception as e:
        print(e)
        return "An error occurred while fetching the weather data."

def train_for_weather(statement):
    """
    Train HelpBot for weather
    Input: statement
    Returns: weather
    """
    min_similarity = 0.5
    weather = nlp('Current Weather Details in a City')
    statement = nlp(statement)
    trainer = ListTrainer(help_bot)

    if weather.similarity(statement) >= min_similarity:
        city = None
        state = None
        country = None
        for ent in statement.ents:
            if ent.label_ == "GPE":
                if not city:
                    city = ent.text
                elif not state:
                    state = ent.text
                else:
                    country = ent.text
        if city:
            return get_weather(city, state, country)
        else:
            return "You need to tell me a city to check."
    else:
        return "Sorry, I don't understand that. Please rephrase your statement."

def simple_math_calculations(statement):
    """
    Perform simple math calculations
    Input: statement
    Returns: calculations
    """
    global last_result 
    doc = nlp(statement)
    
    numbers = [token for token in doc if token.like_num or token.text.lower() == "last"]
    operations = [token for token in doc if token.text in ('plus', 'minus', 'times', 'divided', 'divide', '+', '-', '*', '/')]
    
    if len(numbers) < 2 or len(operations) < 1:
        return "Please provide a valid mathematical expression."

    numbers = [float(last_result) if token.text.lower() == "last" else float(token.text) for token in numbers]
    
    operation = operations[0].text
    if operation in ('plus', '+'):
        result = numbers[0] + numbers[1]
    elif operation in ('minus', '-'):
        result = numbers[0] - numbers[1]
    elif operation in ('times', '*'):
        result = numbers[0] * numbers[1]
    elif operation in ('divided', 'divide', '/'):
        result = numbers[0] / numbers[1]
    else:
        return "Operation not supported."
    
    last_result = result  # Update the last result
    return f"The result is {result}."

def get_current_time():
    """
    Access current time
    Returns: current data
    """
    global last_result  
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    last_result = current_time  
    return f"The current time is {current_time}."

def respond_to_user(statement):
    """
    Respond back to the user
    Inputs: statement
    Returns: response
    """
    doc = nlp(statement)
    if "time" in statement.lower() and not any(token.text in ['times', 'multiply'] for token in doc):
        return get_current_time()
    elif any(token.lemma_ in ["weather", "temperature"] for token in doc):
        return train_for_weather(statement)
    elif any(token.lemma_ in ["plus", "minus", "times", "divided", "divide", "+", "-", "*", "/"] for token in doc):
        return simple_math_calculations(statement)
    else:
        return "I'm not sure how to help with that."

def main():
    """
    Main program function
    """
    print("Welcome to HelpBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = respond_to_user(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
