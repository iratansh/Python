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
from google_trans_new import google_translator

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
        self.context = deque(maxlen=10)  # Memory mechanism to keep track of context
        self.translator = google_translator()

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
                'Previous Close': etf_info.get('previousClose', 'N/A'),
                'Open': etf_info.get('open', 'N/A'),
                'Day Low': etf_info.get('dayLow', 'N/A'),
                'Day High': etf_info.get('dayHigh', 'N/A'),
                '52 Week Low': etf_info.get('fiftyTwoWeekLow', 'N/A'),
                '52 Week High': etf_info.get('fiftyTwoWeekHigh', 'N/A'),
                'Dividend Yield': etf_info.get('yield', 'N/A'),
                'Total Assets': etf_info.get('totalAssets', 'N/A'),
                'NAV Price': etf_info.get('navPrice', 'N/A'),
                'YTD Return': etf_info.get('ytdReturn', 'N/A'),
                'Fund Family': etf_info.get('fundFamily', 'N/A')
            }

            return "\n".join([f"{key}: {value}" for key, value in etf_data.items()])
        except Exception as e:
            logging.error(f"Error fetching ETF information: {e}")
            return "An error occurred while fetching the ETF information."

    def get_current_time(self):
        """
        Get the current time.
        Input: None
        Output: current time (str)
        """
        now = datetime.now()
        return f"The current time is {now.strftime('%H:%M:%S')}."

    def get_current_date(self):
        """
        Get the current date.
        Input: None
        Output: current date (str)
        """
        now = datetime.now()
        return f"Today's date is {now.strftime('%Y-%m-%d')}."

    def contains_url(self, statement):
        """
        Check if the statement contains a URL.
        Input: statement (str)
        Output: True if URL is found, False otherwise
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return re.search(url_pattern, statement) is not None


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
        
    def simple_math_calculations(self, statement):
        """
        Perform simple math calculations based on user input.
        Input: statement (str)
        Output: result of calculation (str)
        """
        # Extract numbers and operators from the statement
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

    def train_for_dictionary(self, statement):
        """
        Train the bot for dictionary queries.
        Input: statement (str)
        Output: response (str)
        """
        dictionary = PyDictionary()
        words = statement.split()
        if len(words) > 1:
            word = words[1]
            meaning = dictionary.meaning(word)
            if meaning:
                return f"The meaning of {word} is: {meaning}"
            else:
                return "Sorry, I couldn't find the meaning of that word."
        else:
            return "Please provide a word to look up."

    def train_for_summary(self, statement):
        """
        Train the bot for summarization queries.
        Input: statement (str)
        Output: response (str)
        """
        # Extract the text to be summarized
        text = statement.replace("summarize", "").strip()
        summary = self.summarization_pipeline(text)
        return summary[0]['summary_text']
    
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

    def train_for_translation(self, statement):
        """
        Train the bot for translation queries.
        Input: statement (str)
        Output: response (str)
        """
        # Extract the text and target language
        match = re.search(r'translate "(.*?)" to (.*)', statement, re.IGNORECASE)
        if match:
            text_to_translate = match.group(1)
            target_language = match.group(2)
            translation = self.translator.translate(text_to_translate, lang_tgt=target_language)
            return translation
        else:
            return "Please provide the text to translate and the target language."

    def generate_response(self, statement):
        """
        Generate a response using the text generation pipeline.
        Input: statement (str)
        Output: response (str)
        """
        response = self.generation_pipeline(statement, max_length=100)
        return response[0]['generated_text']
    
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
