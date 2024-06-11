from BinarySearchTree import BinarySearchTree
from PyDictionary import PyDictionary

class WordPredictor:
    def __init__(self):
        self.dictionary = PyDictionary()
        self.tree = BinarySearchTree()
        self._build_tree()

    def _build_tree(self):
        with open('words.txt') as f:
            for line in f:
                word = line.strip()
                self.tree.insert(word, None)

    def predict(self, prefix):
        return [node.key for node in self.tree if node.key.startswith(prefix)]
    
    def __contains__(self, word):
        return word in self.tree
    
    def __getitem__(self, word):
        return self.tree.search(word)
    
    def __setitem__(self, word, value):
        self.tree.insert(word, value)

    def __len__(self):
        return len(self.tree)
    
    def __str__(self):
        return str(self.tree)
    