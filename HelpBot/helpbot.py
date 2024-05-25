import spacy
import requests
import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

nlp = spacy.load("en_core_web_md")
API_KEY = '725fc8d670f044b4b6cab371441f9d59'
help_bot = ChatBot('help_bot')

def get_weather(location):
    print(location)
    base_url = 'https://api.weatherbit.io/v2.0/current'
    params = {'city': city_name, 'country': country, 'state': state, 'key': API_KEY}

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        list_of_locations = data.get('data', [])

        # If more than one location matches the input
        if len(list_of_locations) > 1:
            print("Error: More than 1 location matches the Input.")
        # If there is only one location that matches the input       
        else:
            response = requests.get(base_url, params=params)
            data = response.json()
            if response.status_code == 200:
                return data['data'][0]
    except Exception as e:
        print(e)
        return None
    

def train_for_weather(statement):
    min_similarity = 0.5
    weather = nlp('Current Weather Details in a City')
    statement = nlp(statement)
    trainer = ListTrainer(help_bot)
    if weather.similarity(statement) >= min_similarity:
        for ent in statement.ents:
            print(ent.label_)
        for ent in statement.ents:
            if ent.label_ == "GPE": 
                location = ent.text
                break
            else:
                return "You need to tell me a city to check."
        city_weather = get_weather(location)
        if city_weather is not None:
            return "In " + location + ", the current weather is: " + city_weather
        else:
            return "Something went wrong."
    else:
        return "Sorry I don't understand that. Please rephrase your statement."


# city_name, temperature, datetime, gust, precip, pressure, snow, sunrise, sunset, uv, weather, wind_speed = get_weather('Edmonton', 'Canada', 'AB')
response = train_for_weather("Is it going to rain in Edmonton, Alberta, Candada today?")
print(response)