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
from nltk.corpus import wordnet
import re
import logging
from collections import deque
from google_trans_new import google_translator
from MultiDataSetTrainer import MultiDatasetTrainer

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
        self.train_conversations_from_file('conversations.txt')
        logging.basicConfig(level=logging.INFO)
        self.context = deque(maxlen=10)  # Keep track of context
        self.translator = google_translator()
        self.trainer = MultiDatasetTrainer(model_name='gpt2')
        datasets = ['hotpotqa/hotpot_qa', "tau/commonsense_qa", "allenai/break_data"] 
        self.trainer.train(datasets)

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

    def load_and_train_dataset(self, filepath):
        """
        Load a dataset from a file and train the bot.
        Input: filepath (str)
        Output: None
        """
        try:
            with open(filepath, 'r') as file:
                conversations = file.readlines()
                cleaned_conversations = [line.strip().upper() for line in conversations if line.strip()]
                self.trainer.train(cleaned_conversations)
                logging.info(f"Training completed with dataset from {filepath}")
        except Exception as e:
            logging.error(f"Error loading and training dataset: {e}")

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
        
        # Store context for personalization and continuity
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
        elif self.contains_ticker_symbol(doc):
            ticker = self.extract_ticker_symbol(doc)
            if ticker:
                return self.get_ticker_information(ticker)
        elif "define" in statement.lower() or "meaning" in statement.lower() or "definition" in statement.lower():
            return self.train_for_dictionary(statement)
        elif "summarize" in statement.lower():
            return self.train_for_summary(statement)
        elif "translate" in statement.lower():
            return self.train_for_translation(statement)
        else:
            return self.generate_response(statement)

    def contains_ticker_symbol(self, doc):
        """
        Check if a statement contains a ticker symbol.
        Input: doc (spacy Doc object)
        Output: True if ticker symbol is found, False otherwise
        """
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        return any(ticker_pattern.match(token.text) for token in doc)

    def extract_ticker_symbol(self, doc):
        """
        Extract the ticker symbol from a statement.
        Input: doc (spacy Doc object)
        Output: ticker symbol (str) or None
        """
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        for token in doc:
            if ticker_pattern.match(token.text):
                return token.text
        return None

    def get_ticker_information(self, ticker):
        """
        Get information for a given ticker symbol.
        Input: ticker (str)
        Output: Ticker information (str)
        """
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            if stock_info['quoteType'] == 'ETF':
                return self.get_etf_information(ticker)
            else:
                return self.get_stock_information(ticker)
        except Exception as e:
            logging.error(f"Error fetching ticker information: {e}")
            return "An error occurred while fetching the ticker information."

    def get_stock_information(self, ticker):
        """
        Get stock information for a given ticker symbol.
        Input: ticker (str)
        Output: stock information (str)
        """
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            
            stock_data = {
                'Stock Name': stock_info.get('shortName', 'N/A'),
                'Current Price': stock_info.get('regularMarketPrice', 'N/A'),
                'Previous Close': stock_info.get('previousClose', 'N/A'),
                'Open': stock_info.get('open', 'N/A'),
                'Day Low': stock_info.get('dayLow', 'N/A'),
                'Day High': stock_info.get('dayHigh', 'N/A'),
                'Volume': stock_info.get('volume', 'N/A'),
                'Market Cap': stock_info.get('marketCap', 'N/A'),
                '52 Week Low': stock_info.get('fiftyTwoWeekLow', 'N/A'),
                '52 Week High': stock_info.get('fiftyTwoWeekHigh', 'N/A'),
                'Dividend Yield': stock_info.get('dividendYield', 'N/A')
            }
            
            return "\n".join([f"{key}: {value}" for key, value in stock_data.items()])
        except Exception as e:
            logging.error(f"Error fetching stock information: {e}")
            return "An error occurred while fetching the stock information."

    def get_etf_information(self, ticker):
        """
        Get ETF information for a given ticker symbol.
        Input: ticker (str)
        Output: ETF information (str)
        """
        try:
            etf = yf.Ticker(ticker)
            etf_info = etf.info

            etf_data = {
                'ETF Name': etf_info.get('shortName', 'N/A'),
                'Current Price': etf_info.get('regularMarketPrice', 'N/A'),
                'Previous Close': etf_info.get('previousClose', 'N/A'),
                'Open': etf_info.get('open', 'N/A'),
                'Day Low': etf_info.get('dayLow', 'N/A'),
                'Day High': etf_info.get('dayHigh', 'N/A'),
                'Volume': etf_info.get('volume', 'N/A'),
                'Market Cap': etf_info.get('marketCap', 'N/A'),
                '52 Week Low': etf_info.get('fiftyTwoWeekLow', 'N/A'),
                '52 Week High': etf_info.get('fiftyTwoWeekHigh', 'N/A'),
                'Net Assets': etf_info.get('totalAssets', 'N/A'),
                'Expense Ratio': etf_info.get('annualReportExpenseRatio', 'N/A')
            }

            return "\n".join([f"{key}: {value}" for key, value in etf_data.items()])
        except Exception as e:
            logging.error(f"Error fetching ETF information: {e}")
            return "An error occurred while fetching the ETF information."

    def simple_math_calculations(self, statement):
        """
        Perform simple math calculations based on user's input.
        Input: statement (str)
        Output: calculation result (str)
        """
        tokens = self.nlp(statement)
        numbers = [float(token.text) for token in tokens if token.like_num]
        operators = [token.text for token in tokens if token.text in ['plus', 'minus', 'times', 'divided', '+', '-', '*', '/', 'x']]

        # Check if we have two numbers and one operator
        if len(numbers) == 2 and len(operators) == 1:
            num1, num2 = numbers
            operator = operators[0]

            # Perform the calculation based on the operator
            if operator in ['plus', '+']:
                result = num1 + num2
            elif operator in ['minus', '-']:
                result = num1 - num2
            elif operator in ['times', 'x', '*']:
                result = num1 * num2
            elif operator in ['divided', '/']:
                result = num1 / num2
            else:
                return "Invalid operator."
            
            return f"The result of {num1} {operator} {num2} is {result}."
        else:
            return "Please provide a valid mathematical expression with two numbers and one operator."

    def generate_response(self, statement):
        """
        Generate a response to the user's input using a text generation model.
        Input: statement (str)
        Output: generated response (str)
        """
        try:
            result = self.generation_pipeline(statement, max_length=100, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I'm not sure how to respond to that."

    def get_current_time(self):
        """
        Get the current time.
        Input: None
        Output: current time (str)
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}"

    def get_current_date(self):
        """
        Get the current date.
        Input: None
        Output: current date (str)
        """
        today = datetime.today()
        current_date = today.strftime("%B %d, %Y")
        return f"Today's date is {current_date}"

    def train_for_webpage_content(self, statement):
        """
        Train the bot to handle requests for accessing webpage content.
        Input: statement (str)
        Output: webpage content (str)
        """
        url = re.search("(?P<url>https?://[^\s]+)", statement).group("url")
        if url:
            return self.access_webpage(url)
        else:
            return "Please provide a valid URL."

    def train_for_wikipedia(self, statement):
        """
        Train the bot to handle requests for Wikipedia information.
        Input: statement (str)
        Output: Wikipedia summary (str)
        """
        try:
            search_query = statement.lower().replace('wikipedia', '').strip()
            summary = self.summarization_pipeline(search_query, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error summarizing Wikipedia content: {e}")
            return "I couldn't retrieve the Wikipedia summary."

    def train_for_dictionary(self, statement):
        """
        Train the bot to handle dictionary requests.
        Input: statement (str)
        Output: definition (str)
        """
        try:
            dictionary = PyDictionary()
            word = statement.lower().replace('define', '').replace('definition', '').replace('meaning', '').strip()
            meaning = dictionary.meaning(word)
            if meaning:
                return f"The definition of {word} is:\n{meaning}"
            else:
                return f"Sorry, I couldn't find the definition of {word}."
        except Exception as e:
            logging.error(f"Error fetching definition: {e}")
            return "An error occurred while fetching the definition."

    def train_for_summary(self, statement):
        """
        Train the bot to handle text summarization requests.
        Input: statement (str)
        Output: summary (str)
        """
        try:
            text = statement.lower().replace('summarize', '').strip()
            summary = self.summarization_pipeline(text, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error summarizing text: {e}")
            return "I couldn't summarize the text."

    def train_for_translation(self, statement):
        """
        Train the bot to handle translation requests.
        Input: statement (str)
        Output: translation (str)
        """
        try:
            parts = statement.lower().replace('translate', '').strip().split('to')
            text_to_translate = parts[0].strip()
            target_language = parts[1].strip() if len(parts > 1) else 'en'
            translation = self.translator.translate(text_to_translate, lang_tgt=target_language)
            return translation
        except Exception as e:
            logging.error(f"Error translating text: {e}")
            return "I couldn't translate the text."

    def contains_url(self, text):
        """
        Check if the given text contains a URL.
        Input: text (str)
        Output: True if URL is found, False otherwise
        """
        url_pattern = re.compile(r'https?://[^\s]+')
        return url_pattern.search(text) is not None
