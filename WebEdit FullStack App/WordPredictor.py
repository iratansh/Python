import nltk
from nltk.corpus import words
from Trie import Trie

class WordFinisher:
    def __init__(self):
        self.trie = Trie()

    def insert_words_from_nltk(self):
        nltk.download('words')  
        word_list = words.words()
        for word in word_list:
            self.trie.insert(word.lower())

    def insert_word(self, word):
        self.trie.insert(word.lower()) 

    def predict_words(self, prefix, k=3):
        return self.trie.search(prefix.lower(), k) 

word_finisher = WordFinisher()
word_finisher.insert_words_from_nltk()
prefix = 'probabil'
predictions = word_finisher.predict_words(prefix)
print(f"Predictions for '{prefix}': {predictions}")

    
