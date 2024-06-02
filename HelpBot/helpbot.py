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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline

class HelpBot:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.help_bot = ChatBot('help_bot')
        self.trainer = ListTrainer(self.help_bot)
        self.API_KEY = '725fc8d670f044b4b6cab371441f9d59'
        self.last_result = None
        self.summarization_pipeline = pipeline("summarization")
        self.train_basic_conversations()

    def train_basic_conversations(self):
        conversations = [
            "Hello", "Hi there!",
            "Hi", "Hello!",
            "How are you?", "I'm good, thank you! How can I help you today?",
            "What's your name?", "I'm HelpBot, your assistant.",
            "Goodbye", "Goodbye! Have a nice day!",
            "See you later", "See you later! Take care!",
            "What can you do?", "I can help you with various tasks like checking the time, date, weather, doing simple math calculations, summarizing text, and more. Just ask me anything!",
            "Thanks", "You're welcome!",
            "Thank you", "You're welcome!",
            "What's up?", "Not much, just here to help you!",
            "How can you help me?", "I can help you with various tasks like checking the time, date, weather, doing simple math calculations, summarizing text, and more. Just ask me anything!",
            "Can you help me?", "Of course! Just tell me what you need help with.",
            "Who are you?", "I'm HelpBot, your assistant.",
            "What are you?", "I'm HelpBot, your assistant.",
            "What do you do?", "I can help you with various tasks like checking the time, date, weather, doing simple math calculations, summarizing text, and more. Just ask me anything!",
            "What is your purpose?", "I'm here to help you with anything you need!",
            "Who made you?", "I was created by a developer named Ishaan.",
            "Who created you?", "I was created by a developer named Ishaan.",
            "Who is your creator?", "I was created by a developer named Ishaan.",
            "Who is your developer?", "I was created by a developer named Ishaan.",
            "Who is your owner?", "I was created by a developer named Ishaan.",
            "Who is your master?", "I have no master, I'm here to help you!",
            "What's new?" "Nothing much, I'm here to help you with anything you need!",
            "How's your day?" "I'm a bot, I don't have days, but I'm here to help you!",
            "Howdy" "Hello! How can I help you today?",
            "Hey" "Hello! How can I help you today?",
            "Yo" "Hello! How can I help you today?",
            "What's happening?" "Nothing much, I'm here to help you with anything you need!",
            "What's going on?" "Nothing much, I'm here to help you with anything you need!",
            "Good morning" "Good morning! How can I help you today?",
            "Good afternoon" "Good afternoon! How can I help you today?",
            "Good evening" "Good evening! How can I help you today?",
            "Good night" "Good night! Have a nice day!",
            "Howdy partner" "Hello! How can I help you today?",
            "Hey there" "Hello! How can I help you today?",
            "Hey buddy" "Hello! How can I help you today?",
            "Hey friend" "Hello! How can I help you today?",
            "Hey pal" "Hello! How can I help you today?",
            "Hey dude" "Hello! How can I help you today?",
            "Nice to meet you" "Nice to meet you too! How can I help you today?",
            "Pleased to meet you" "Pleased to meet you too! How can I help you today?",
            "Nice meeting you" "Nice meeting you too! How can I help you today?",
            "Pleased meeting you" "Pleased meeting you too! How can I help you today?",
            "How have you been?" "I'm a bot, I don't have feelings, but I'm here to help you!",
            "How's life?" "I'm a bot, I don't have a life, but I'm here to help you!",
            "How's everything?" "Everything is good! How can I help you today?",
            "How's it going?" "Everything is good! How can I help you today?",
            "How's the weather?" "I can check the weather for you! Just tell me the city.",
            "What's the weather like?" "I can check the weather for you! Just tell me the city.",
            "What's the temperature?" "I can check the weather for you! Just tell me the city.",
            "Greetings" "Hello! How can I help you today?",
            "Salutations" "Hello! How can I help you today?",
            "Hello there" "Hello! How can I help you today?",
            "Hello friend" "Hello! How can I help you today?",
            "Hello buddy" "Hello! How can I help you today?",
            "Hello pal" "Hello! How can I help you today?",
            "Hello dude" "Hello! How can I help you today?",
            "Hello mate" "Hello! How can I help you today?",
            "Hello sir" "Hello! How can I help you today?",
            "Hello ma'am" "Hello! How can I help you today?",
            "Hello miss" "Hello! How can I help you today?",
            "Hello mister" "Hello! How can I help you today?",
            "Hello madam" "Hello! How can I help you today?",
            "What's the news?" "I can't provide news updates, but I can help you with other things!",
            "What's the latest?" "I can't provide news updates, but I can help you with other things!",
            "Long time no see" "I'm a bot, I don't have eyes, but I'm here to help you!",
            "It's been a while" "I'm a bot, I don't have a sense of time, but I'm here to help you!",
            "How's the family?" "I'm a bot, I don't have a family, but I'm here to help you!",
            "How's work?" "I'm a bot, I don't have a job, but I'm here to help you!",
            "How's school?" "I'm a bot, I don't go to school, but I'm here to help you!",
            "How's the job?" "I'm a bot, I don't have a job, but I'm here to help you!",
        ]
        self.trainer.train(conversations)

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
        # Check for basic conversational prompts
        response = self.help_bot.get_response(statement)
        if response.confidence > 0.5:
            return str(response)
        
        # Handle other types of queries
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
            return self.train_for_wikipedia(statement)
        elif "stock" in statement.lower():
            return self.train_for_stock_information(statement)
        elif "dictionary" in statement.lower():
            return self.train_for_dictionary(statement)
        elif "summarize" in statement.lower():
            return self.train_for_summary(statement)
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
            result = numbers[0] - numbers[1]
        elif operation in ('times', '*'):
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
        for word in word_freq:
            word_freq[word] /= max_freq

        sentence_scores = {}
        for sent in doc.sents:
            for word in sent:
                word_text = word.text.lower()
                if word_text in word_freq:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word_text]
                    else:
                        sentence_scores[sent] += word_freq[word_text]

        from heapq import nlargest
        summarized_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
        final_summary = ' '.join([sent.text for sent in summarized_sentences])
        return final_summary

    def advanced_summarize(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # Summarize the document with 3 sentences
        return ' '.join([str(sentence) for sentence in summary])
    
    def transformer_summarize(self, text):
        return self.summarization_pipeline(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']

    def train_for_summary(self, text):
        if "advanced" in text:
            return self.advanced_summarize(text)
        elif "transformer" in text:
            return self.transformer_summarize(text)
        else:
            return self.summarize_information(text)
    
    def train_for_webpage_content(self, text):

        return "Webpage content training is not implemented yet."

    def train_for_wikipedia(self, text):

        return "Wikipedia content training is not implemented yet."

    def train_for_stock_information(self, text):
        
        return "Stock information training is not implemented yet."

    def train_for_dictionary(self, text):

        return "Dictionary training is not implemented yet."

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
