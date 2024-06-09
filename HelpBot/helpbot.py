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
        """
        Initialize the HelpBot with necessary components.
        Input: None
        Output: None
        """
        # Initialize Spacy model and other components
        self.nlp = spacy.load("en_core_web_md")
        self.help_bot = ChatBot('help_bot')
        self.trainer = ListTrainer(self.help_bot)
        self.API_KEY = '725fc8d670f044b4b6cab371441f9d59'
        self.last_result = None
        self.summarization_pipeline = pipeline("summarization")
        self.generation_pipeline = pipeline("text-generation", model="gpt2")
        self.translation_pipeline = pipeline("translation_en_to_fr")  # Add translation pipeline
        self.question_answering_pipeline = pipeline("question-answering")  # Add question answering pipeline
        self.train_conversations_from_file('conversations.txt')
        logging.basicConfig(level=logging.INFO)
        self.context = deque(maxlen=10)  # Keep track of context

    def train_conversations_from_file(self, filename):
        """
        Train the bot using conversations from a file.
        Input: filename (str)
        Output: None
        """
        try:
            with open(filename, 'r') as file:
                conversations = [line.strip().split('|') for line in file.readlines()]
                for conversation in conversations:
                    if len(conversation) == 2:
                        # Convert to uppercase
                        upper_conversation = [phrase.upper() for phrase in conversation]
                        self.trainer.train(upper_conversation)
        except Exception as e:
            logging.error(f"Error training conversations from file: {e}")

    def get_weather(self, city, state=None, country=None):
        """
        Get the current weather for a given city using the Weatherbit API.
        Input: city (str), state (str), country (str)
        Output: weather information (str)
        """
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
        """
        Respond to a weather query based on the user's input.
        Input: statement (str)
        Output: weather information (str)
        """
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
        """
        Respond to the user's input based on the type of query.
        Input: statement (str)
        Output: response (str)
        """
        # Check for basic conversational prompts
        upper_statement = statement.upper()  # Convert user input to uppercase
        response = self.help_bot.get_response(upper_statement)
        if response.confidence > 0.5:
            return str(response)
        
        # Store context for future reference
        self.context.append(statement)
        
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
        elif "complete" in statement.lower():
            return self.train_for_completion(statement)
        elif "paraphrase" in statement.lower():
            return self.train_for_paraphrasing(statement)
        elif "translate" in statement.lower():
            return self.train_for_translation(statement)
        elif "question" in statement.lower() and "answer" in statement.lower():
            return self.train_for_question_answering(statement)
        else:
            return self.generate_response(statement)

    def contains_stock_ticker(self, doc):
        """
        Check if a statement contains a stock ticker symbol.
        Input: doc (spacy Doc object)
        Output: True if stock ticker is found, False otherwise
        """
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        return any(ticker_pattern.match(token.text) for token in doc)

    def contains_url(self, statement):
        """
        Check if a statement contains a URL.
        Input: statement (str)
        Output: True if URL is found, False otherwise
        """
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return re.search(url_pattern, statement) is not None

    def get_current_time(self):
        """
        Get the current time.
        Input: None
        Output: Current time (str)
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}."

    def get_current_date(self):
        """
        Get the current date.
        Input: None
        Output: Current date (str)
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        return f"The current date is {current_date}."

    def access_webpage(self, url):
        """
        Access a webpage and return its content.
        Input: url (str)
        Output: Webpage content (str)
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error accessing webpage: {e}")
            return "An error occurred while accessing the webpage."

    def get_webpage_content(self, statement):
        """
        Get the content of a webpage based on the URL provided in the user's statement.
        Input: statement (str)
        Output: Webpage content (str)
        """
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        url = re.search(url_pattern, statement)
        if url:
            return self.access_webpage(url.group(0))
        else:
            return "Please provide a valid URL."

    def get_wikipedia_summary(self, topic):
        """
        Get a summary of a given topic from Wikipedia.
        Input: topic (str)
        Output: Wikipedia summary (str)
        """
        try:
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}")
            response.raise_for_status()
            data = response.json()
            return data.get('extract', "No summary found.")
        except Exception as e:
            logging.error(f"Error fetching Wikipedia summary: {e}")
            return "An error occurred while fetching the Wikipedia summary."

    def get_wikipedia_search_results(self, topic):
        """
        Get search results for a given topic from Wikipedia.
        Input: topic (str)
        Output: Wikipedia search results (str)
        """
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

    def get_summary(self, text, max_length=50):
        """
        Summarize the given text.
        Input: text (str), max_length (int)
        Output: Summary of the text (str)
        """
        try:
            summary = self.summarization_pipeline(text, max_length=max_length, min_length=25, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error summarizing text: {e}")
            return "An error occurred while summarizing the text."

    def get_stock_information(self, ticker):
        """
        Get stock information for a given ticker symbol.
        Input: ticker (str)
        Output: Stock information (str)
        """
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            return f"Stock: {stock_info['shortName']}\nPrice: {stock_info['currentPrice']}\nMarket Cap: {stock_info['marketCap']}"
        except Exception as e:
            logging.error(f"Error fetching stock information: {e}")
            return "An error occurred while fetching the stock information."

    def get_dictionary_definition(self, word):
        """
        Get the dictionary definition of a word.
        Input: word (str)
        Output: Dictionary definition (str)
        """
        dictionary = PyDictionary()
        try:
            meaning = dictionary.meaning(word)
            if meaning:
                return "\n".join([f"{part}: {', '.join(definitions)}" for part, definitions in meaning.items()])
            else:
                return "No definition found."
        except Exception as e:
            logging.error(f"Error fetching dictionary definition: {e}")
            return "An error occurred while fetching the dictionary definition."

    def generate_response(self, statement):
        """
        Generate a response using a generative AI model.
        Input: statement (str)
        Output: Generated response (str)
        """
        try:
            generated_text = self.generation_pipeline(statement, max_length=100, num_return_sequences=1)
            return generated_text[0]['generated_text']
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "An error occurred while generating a response."

    def train_for_webpage_content(self, statement):
        """
        Train the bot to provide content from a webpage.
        Input: statement (str)
        Output: Webpage content (str)
        """
        return self.get_webpage_content(statement)

    def train_for_wikipedia(self, statement):
        """
        Train the bot to provide Wikipedia information.
        Input: statement (str)
        Output: Wikipedia information (str)
        """
        doc = self.nlp(statement)
        topic = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']])
        if topic:
            return self.get_wikipedia_summary(topic)
        else:
            return "Please provide a topic to search on Wikipedia."

    def train_for_stock_information(self, statement):
        """
        Train the bot to provide stock information.
        Input: statement (str)
        Output: Stock information (str)
        """
        doc = self.nlp(statement)
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        ticker = [token.text for token in doc if ticker_pattern.match(token.text)]
        if ticker:
            return self.get_stock_information(ticker[0])
        else:
            return "Please provide a valid stock ticker symbol."

    def train_for_dictionary(self, statement):
        """
        Train the bot to provide dictionary definitions.
        Input: statement (str)
        Output: Dictionary definition (str)
        """
        doc = self.nlp(statement)
        word = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
        if word:
            return self.get_dictionary_definition(word)
        else:
            return "Please provide a word to look up."

    def train_for_summary(self, statement):
        """
        Train the bot to provide text summaries.
        Input: statement (str)
        Output: Text summary (str)
        """
        doc = self.nlp(statement)
        text = " ".join([token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
        if text:
            return self.get_summary(text)
        else:
            return "Please provide text to summarize."

    def train_for_completion(self, statement):
        """
        Train the bot to complete a given text.
        Input: statement (str)
        Output: Completed text (str)
        """
        try:
            completion = self.generation_pipeline(statement, max_length=100, num_return_sequences=1)
            return completion[0]['generated_text']
        except Exception as e:
            logging.error(f"Error completing text: {e}")
            return "An error occurred while completing the text."

    def train_for_paraphrasing(self, statement):
        """
        Train the bot to paraphrase a given text.
        Input: statement (str)
        Output: Paraphrased text (str)
        """
        try:
            paraphrase = self.generation_pipeline(statement, max_length=100, num_return_sequences=1)
            return paraphrase[0]['generated_text']
        except Exception as e:
            logging.error(f"Error paraphrasing text: {e}")
            return "An error occurred while paraphrasing the text."

    def train_for_translation(self, statement):
        """
        Train the bot to translate text from English to French.
        Input: statement (str)
        Output: Translated text (str)
        """
        try:
            translation = self.translation_pipeline(statement)
            return translation[0]['translation_text']
        except Exception as e:
            logging.error(f"Error translating text: {e}")
            return "An error occurred while translating the text."

    def train_for_question_answering(self, statement):
        """
        Train the bot to answer questions based on a provided context.
        Input: statement (str)
        Output: Answer to the question (str)
        """
        context = " ".join(self.context)
        question = statement
        try:
            answer = self.question_answering_pipeline(question=question, context=context)
            return answer['answer']
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            return "An error occurred while answering the question."
