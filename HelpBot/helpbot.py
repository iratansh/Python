"""
HelpBot: A multi-purpose chatbot application
"""

import spacy
import requests, PyDictionary
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from datetime import datetime
import yfinance as yf

class HelpBot:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.help_bot = ChatBot('help_bot')
        self.trainer = ListTrainer(self.help_bot)
        self.API_KEY = '725fc8d670f044b4b6cab371441f9d59'
        self.last_result = None

    def get_weather(self, city, state=None, country=None):
        base_url = 'https://api.weatherbit.io/v2.0/current'
        params = {'city': city, 'key': self.API_KEY}
        if state:
            params['state'] = state
        if country:
            params['country'] = country
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            city_weather = data['data'][0]  # Assuming only one location is returned
            
            return f"In {city}, {state if state else ''} {country if country else ''}, the current weather is: {city_weather['weather']['description']}, temperature: {city_weather['temp']}°C."
        
        except Exception as e:
            print(e)
            return "Failed to get weather data."

    def train_weather_queries(self):
        self.trainer.train(["Current Weather Details in a City"])

    def respond_to_weather_query(self, statement):
        doc = self.nlp(statement)
        min_similarity = 0.5
        weather = self.nlp('Current Weather Details in a City')

        if weather.similarity(doc) >= min_similarity:
            city = None
            state = None
            country = None
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    if not city:
                        city = ent.text
                    elif not state:
                        state = ent.text
                    else:
                        country = ent.text

            if city:
                return self.get_weather(city, state, country)
            else:
                return "You need to tell me a city to check."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."

    def respond_to_user(self, statement):
        doc = self.nlp(statement)
        if "time" in statement.lower() and not any(token.text in ['times', 'multiply'] for token in doc):
            return self.get_current_time()
        elif any(token.lemma_ in ["weather", "temperature"] for token in doc):
            return self.respond_to_weather_query(statement)
        elif any(token.lemma_ in ["plus", "minus", "times", "divided", "divide", "+", "-", "*", "/"] for token in doc):
            return self.simple_math_calculations(statement)
        elif "date" in statement.lower():
            return self.get_current_date()
        elif "webpage" in statement.lower():
            return self.train_for_webpage_content(statement)
        elif "wikipedia" in statement.lower():
            return self.get_wikipedia_content(statement)
        elif "stock" in statement.lower():
            return self.train_for_stock_information(statement)
        elif "dictionary" in statement.lower():
            return self.get_dictionary_definition(statement)
        else:
            return "I'm not sure how to help with that."

    def get_current_time(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}."

    def get_current_date(self):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        return f"The current date is {current_date}."

    def access_webpage(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                return "Failed to access the webpage."
        except Exception as e:
            print(e)
            return "An error occurred while accessing the webpage."
    
    def get_webpage_content(self, url):
        doc = self.nlp(url)
        if any(token.text in ['webpage', 'web', 'page'] for token in doc):
            return self.access_webpage(url)
        else:
            return "I'm not sure how to help with that."

    def get_wikipedia_summary(self, topic):
        try:
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}")
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', "No summary found.")
            else:
                return "Failed to get the Wikipedia summary."
        except Exception as e:
            print(e)
            return "An error occurred while fetching the Wikipedia summary."
        
    def get_wikipedia_search_results(self, topic):
        try:
            response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={topic}&format=json")
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('query', {}).get('search', [])
                if search_results:
                    return search_results[0].get('snippet', "No search results found.")
                else:
                    return "No search results found."
            else:
                return "Failed to get the Wikipedia search results."
        except Exception as e:
            print(e)
            return "An error occurred while fetching the Wikipedia search results."
    
    def get_wikipedia_content(self, topic):
        doc = self.nlp(topic)
        if any(token.text in ['summary', 'summarize'] for token in doc):
            return self.get_wikipedia_summary(topic)
        elif any(token.text in ['search', 'find'] for token in doc):
            return self.get_wikipedia_search_results(topic)
        else:
            return "I'm not sure how to help with that."
    
    def get_current_stock_information(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            stock_info = stock.info
            return f"Current price of {stock_info['shortName']} ({symbol}): {stock_info['currency']} {stock_info['regularMarketPrice']}"
        except Exception as e:
            print(e)
            return "An error occurred while fetching the stock information."

    def simple_math_calculations(self, statement):
        doc = self.nlp(statement)
        numbers = [token for token in doc if token.like_num or token.text.lower() == "last"]
        operations = [token for token in doc if token.text in ('plus', 'minus', 'times', 'divided', 'divide', '+', '-', '*', '/')]
        
        if len(numbers) < 2 or len(operations) < 1:
            return "Please provide a valid mathematical expression."

        numbers = [float(self.last_result) if token.text.lower() == "last" else float(token.text) for token in numbers]
        
        operation = operations[0].text
        if operation in ('plus', '+'):
            result = numbers[0] + numbers[1]
        elif operation in ('minus', '-'):
            result = numbers[0] - numbers

    def search_internet_for_educational_resources(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_news(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_entertainment(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_sports(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_technology(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_health(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_business(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_travel(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_shopping(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_food(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_social_media(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_videos(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_images(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_music(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_movies(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def search_internet_for_books(self, topic):
        return "I'm sorry, I'm not able to help with that."

    def get_dictionary_definition(self, word):
        dictionary = PyDictionary.PyDictionary()
        definition = dictionary.meaning(word)
        if definition:
            return f"The definition of {word} is: {definition['Noun'][0]}"
        else:
            return "I'm sorry, I'm not able to help with that."
        
    def summarize_information(self, text):
        doc = self.nlp(text)
        word_dict = dict()
        for word in doc:
            word = word.text.lower()

            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        sentences = []

        sentence_score = 0
        for i, sentence in enumerate(doc.sents):
            for word in sentence:
                word = word.text.lower()
                sentence_score += word_dict[word]
            sentences.append((i, sentence.text.replace('\n', ''), sentence_score/len(sentence)))
        
        sorted_sentences = sorted(sentences, key=lambda x: -x[2], reverse=True)
        top_three = sorted(sorted_sentences[:3], key=lambda x: x[0])
        summary = ""
        for sentence in top_three:
            summary += sentence[1] + " "
        return summary

    def train_for_stock_information(self, statement):
        min_similarity = 0.5
        stock = self.nlp('Current Stock Information for a Symbol')
        doc = self.nlp(statement)

        if stock.similarity(doc) >= min_similarity:
            symbol = None
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    symbol = ent.text

            if symbol:
                return self.get_current_stock_information(symbol)
            else:
                return "You need to tell me a stock symbol to check."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."

    def train_for_webpage_content(self, statement):
        min_similarity = 0.5
        webpage = self.nlp('Access Webpage Content')
        doc = self.nlp(statement)

        if webpage.similarity(doc) >= min_similarity:
            url = None
            for ent in doc.ents:
                if ent.label_ == "URL":
                    url = ent.text

            if url:
                return self.get_webpage_content(url)
            else:
                return "You need to tell me a URL to check."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."

    def train_for_searching_internet(self, statement):
        min_similarity = 0.5
        search = self.nlp('Search Internet for Educational Resources')
        doc = self.nlp(statement)

        if search.similarity(doc) >= min_similarity:
            topic = None
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    topic = ent.text

            if topic:
                return self.search_internet_for_educational_resources(topic)
            else:
                return "You need to tell me a topic to search for."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
    
    def train_for_wikipedia(self, statement):
        min_similarity = 0.5
        wikipedia = self.nlp('Get Wikipedia Summary')
        doc = self.nlp(statement)

        if wikipedia.similarity(doc) >= min_similarity:
            topic = None
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    topic = ent.text

            if topic:
                return self.get_wikipedia_content(topic)
            else:
                return "You need to tell me a topic to search for."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
        
    def train_for_dictionary(self, statement):
        min_similarity = 0.5
        dictionary = self.nlp('Get Dictionary Definition')
        doc = self.nlp(statement)

        if dictionary.similarity(doc) >= min_similarity:
            word = None
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    word = ent.text

            if word:
                return self.get_dictionary_definition(word)
            else:
                return "You need to tell me a word to check."
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
    
    def train_for_time(self, statement):
        min_similarity = 0.5
        time = self.nlp('Get Current Time')
        doc = self.nlp(statement)

        if time.similarity(doc) >= min_similarity:
            return self.get_current_time()
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
    
    def train_for_date(self, statement):
        min_similarity = 0.5
        date = self.nlp('Get Current Date')
        doc = self.nlp(statement)

        if date.similarity(doc) >= min_similarity:
            return self.get_current_date()
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
        
    def train_for_weather(self, statement):
        min_similarity = 0.5
        weather = self.nlp('Get Current Weather Details')
        doc = self.nlp(statement)

        if weather.similarity(doc) >= min_similarity:
            return self.respond_to_weather_query(statement)
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
        
    def train_for_math(self, statement):
        min_similarity = 0.5
        math = self.nlp('Simple Math Calculations')
        doc = self.nlp(statement)

        if math.similarity(doc) >= min_similarity:
            return self.simple_math_calculations(statement)
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
    
    def train_for_summary(self, statement):
        min_similarity = 0.5
        summary = self.nlp('Summarize Information')
        doc = self.nlp(statement)

        if summary.similarity(doc) >= min_similarity:
            return self.summarize_information(statement)
        else:
            return "Sorry, I don't understand that. Please rephrase your statement."
    
def main():
    help_bot = HelpBot()
    help_bot.train_weather_queries()
    
    print("Welcome to HelpBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = help_bot.respond_to_user(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
