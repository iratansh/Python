This program defines a chatbot named `HelpBot` that can perform a variety of tasks, including:

1. **Conversational Responses**: Utilizes the ChatterBot library to provide conversational responses trained from a file containing conversations.
2. **Weather Information**: Fetches current weather details for a given city using the Weatherbit API.
3. **Time and Date**: Provides the current time and date.
4. **Webpage Content Retrieval**: Fetches and displays the content of a given webpage.
5. **Wikipedia Summaries**: Retrieves summaries or search results from Wikipedia.
6. **Stock Information**: Provides current stock prices using Yahoo Finance.
7. **Dictionary Definitions**: Retrieves definitions of words using the PyDictionary library.
8. **Text Summarization**: Summarizes a given text using a pre-trained summarization pipeline from the transformers library.
9. **Simple Math Calculations**: Performs basic arithmetic operations.
10. **Contextual Memory**: Maintains context of recent conversations for better interaction.

### Technologies Used:
1. **Spacy**: For natural language processing tasks such as entity recognition and similarity checks.
2. **Requests**: For making HTTP requests to external APIs (Weatherbit, Wikipedia).
3. **PyDictionary**: For fetching word meanings.
4. **ChatterBot**: For creating and training the chatbot.
5. **yFinance**: For retrieving stock information.
6. **Transformers (HuggingFace)**: For text summarization.
7. **NLTK**: For tokenizing and finding synonyms.
8. **Heapq**: For ranking sentences based on frequency scores during summarization.
9. **Logging**: For error handling and logging messages.
10. **Collections (Deque)**: For maintaining a limited-size context of past conversations.
