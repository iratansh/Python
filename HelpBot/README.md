This program is a conversational bot designed to assist users with various tasks.

Functionality:
* Conversational Interface: Responds to user queries and prompts for assistance.
* Weather Information: Retrieves current weather conditions for a given city using the Weatherbit API.
* Time and Date: Provides current time and date.
* asic Calculations: Performs simple math calculations.
* Webscraping: Accesses and returns the content of a webpage.
* Wikipedia: Provides summaries or search results from Wikipedia.
* Stock Information: Retrieves current stock prices or ETF Net Asset Values using the Yahoo Finance API.
* Dictionary: Defines words and provides their meanings.
* Text Summarization: Summarizes text using the Hugging Face Transformers library.
* Context Management: Maintains context for personalization and continuity across interactions.

Technologies Used:
* Python: The main programming language.

* Libraries and APIs:
* Spacy: For natural language processing (NLP) tasks such as entity recognition and similarity analysis.
* ChatterBot: Provides the conversational interface with training capabilities.
* PyDictionary: Accesses word definitions and meanings.
* Requests: Makes HTTP requests to external APIs.
* YFinance: Retrieves stock market data.
* Transformers (Hugging Face): Utilized for text summarization.
* NLTK: Provides tools for NLP tasks such as tokenization and word synonyms.
* Data Storage: Utilizes files for training data (conversations.txt).
* Logging: Records errors and informational messages for debugging.
* Data Structures: Uses deque for memory mechanism to track context.
