"""
Helpbot - A multi-purpose chatbot application
"""

import spacy
import requests
from datetime import datetime
from PyDictionary import PyDictionary
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import yfinance as yf
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
import heapq
import re
import logging
from collections import deque

class HelpBot:
    def __init__(self):
        # Initialize Spacy model and other components
        self.nlp = spacy.load("en_core_web_md")
        self.help_bot = ChatBot('help_bot')
        self.trainer = ListTrainer(self.help_bot)
        self.API_KEY = '725fc8d670f044b4b6cab371441f9d59'
        self.last_result = None
        self.summarization_pipeline = pipeline("summarization")
        self.train_conversations_from_file('conversations.txt')
        logging.basicConfig(level=logging.INFO)
        self.context = deque(maxlen=10)  # Memory mechanism to keep track of context

    def train_conversations_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                conversations = [line.strip().split('|') for line in file.readlines()]
                for conversation in conversations:
                    if len(conversation) == 2:
                        self.trainer.train(conversation)
        except Exception as e:
            logging.error(f"Error training conversations from file: {e}")

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
            logging.error(f"Error fetching weather data: {e}")
            return "Failed to get weather data."

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
        # Check for basic conversational prompts
        response = self.help_bot.get_response(statement)
        if response.confidence > 0.5:
            return str(response)
        self.context.append(statement)  # Store context for personalization and continuity
        
        # Handle other types of queries
        doc = self.nlp(statement)
        if "time" in statement.lower() and not any(token.text in ['times', 'multiply'] for token in doc):
            return self.get_current_time()
        elif any(token.lemma_ in ["weather", "temperature"] for token in doc):
            return self.respond_to_weather_query(statement)
        elif any(token.lemma_ in ["plus", "minus", "times", "divided", "divide", "+", "-", "*", "/", "x"] for token in doc):
            return self.simple_math_calculations(statement)
        elif "date" in statement.lower():
            return self.get_current_date()
        elif "webpage" in statement.lower() or self.contains_url(statement):
            return self.train_for_webpage_content(statement)
        elif "wikipedia" in statement.lower():
            return self.train_for_wikipedia(statement)
        elif "stock" in statement.lower() or self.contains_stock_ticker(doc):
            return self.train_for_stock_information(statement)
        elif "dictionary" in statement.lower():
            return self.train_for_dictionary(statement)
        elif "summarize" in statement.lower():
            return self.train_for_summary(statement)
        else:
            return "I'm not sure how to help with that."

    def contains_stock_ticker(self, doc):
        # Simple heuristic to check if a statement might contain a stock ticker symbol
        stock_symbols = ["AAPL", "GOOG", "MSFT", "TSLA"] 
        return any(token.text in stock_symbols for token in doc)

    def contains_url(self, statement):
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return re.search(url_pattern, statement) is not None

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
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error accessing webpage: {e}")
            return "An error occurred while accessing the webpage."
    
    def get_webpage_content(self, statement):
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        url = re.search(url_pattern, statement)
        if url:
            return self.access_webpage(url.group(0))
        else:
            return "Please provide a valid URL."

    def get_wikipedia_summary(self, topic):
        try:
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}")
            response.raise_for_status()
            data = response.json()
            return data.get('extract', "No summary found.")
        except Exception as e:
            logging.error(f"Error fetching Wikipedia summary: {e}")
            return "An error occurred while fetching the Wikipedia summary."
        
    def get_wikipedia_search_results(self, topic):
        try:
            response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={topic}&format=json")
            response.raise_for_status()
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            if search_results:
                return search_results[0].get('snippet', "No search results found.")
            else:
                return "No search results found."
        except Exception as e:
            logging.error(f"Error fetching Wikipedia search results: {e}")
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
            logging.error(f"Error fetching stock information: {e}")
            return "An error occurred while fetching the stock information."
        
    def process_user_input(self, user_input):
        tokens = nltk.word_tokenize(user_input)
        synonyms = set()

        for token in tokens:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())

        return ' '.join(synonyms)

    def simple_math_calculations(self, statement):
        doc = self.nlp(statement)
        numbers = [token for token in doc if token.like_num or token.text.lower() == "last"]
        operations = [token for token in doc if token.text in ('plus', 'minus', 'times', 'divided', 'divide', '+', '-', '*', '/', 'x')]

        if len(numbers) < 2 or len(operations) < 1:
            return "Please provide a valid mathematical expression."

        numbers = [float(self.last_result) if token.text.lower() == "last" else float(token.text) for token in numbers]
        
        operation = operations[0].text
        if operation in ('plus', '+'):
            result = numbers[0] + numbers[1]
        elif operation in ('minus', '-'):
            result = numbers[0] - numbers[1]
        elif operation in ('times', '*', 'x'):
            result = numbers[0] * numbers[1]
        elif operation in ('divided', 'divide', '/'):
            result = numbers[0] / numbers[1]
        else:
            return "Unknown operation."

        self.last_result = result
        return f"The result is {result}."

    def summarize_information(self, text):
        # Simple summarization method
        doc = self.nlp(text)
        word_freq = {}

        for word in doc:
            if word.is_stop or word.is_punct:
                continue
            word_text = word.text.lower()
            if word_text not in word_freq:
                word_freq[word_text] = 1
            else:
                word_freq[word_text] += 1

        max_freq = max(word_freq.values())

        for word in word_freq.keys():
            word_freq[word] = word_freq[word] / max_freq

        sentence_scores = {}
        for sent in doc.sents:
            for word in sent:
                word_text = word.text.lower()
                if word_text in word_freq.keys():
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word_text]
                    else:
                        sentence_scores[sent] += word_freq[word_text]

        summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
        summary = ' '.join([sent.text for sent in summary_sentences])
        return summary

    def train_for_webpage_content(self, statement):
        return self.get_webpage_content(statement)

    def train_for_wikipedia(self, statement):
        topic = statement.split("wikipedia")[-1].strip()
        return self.get_wikipedia_content(topic)

    def train_for_stock_information(self, statement):
        symbol = self.extract_stock_symbol(statement)
        if symbol:
            return self.get_current_stock_information(symbol)
        else:
            return "Please provide a valid stock ticker symbol."

    def extract_stock_symbol(self, statement):
        doc = self.nlp(statement)
        for token in doc:
            if token.text.isupper() and len(token.text) <= 5:
                return token.text
        return None

    def train_for_dictionary(self, statement):
        dictionary = PyDictionary()
        word = statement.split("dictionary")[-1].strip()
        meaning = dictionary.meaning(word)
        if meaning:
            return f"The meaning of {word} is: {meaning}"
        else:
            return f"Could not find the meaning of {word}."
    
    def train_for_summary(self, statement):
        text = statement.split("summarize")[-1].strip()
        return self.summarize_information(text)

def main():
    bot = HelpBot()
    while True:
        statement = input("You: ")
        if statement.lower() == "exit":
            print("Goodbye!")
            break
        response = bot.respond_to_user(statement)
        print("HelpBot:", response)

if __name__ == "__main__":
    main()
