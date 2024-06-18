"""
This module contains the implementation of a Trie data structure.
The Trie data structure is used to store words and complete a partial word based on a given prefix.
"""

from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the Trie.
        Input: word (str) - The word to be inserted into the Trie.
        Output: None
        """
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end_of_word = True

    def batch_insert_from_file(self, batch_size=2000):
        """
        Inserts words from a file into the Trie in batches.
        Input: batch_size (int) - The number of words to insert in each batch.
        Output: None
        """
        try:
            with open('words.txt', 'r', encoding='utf-8') as file:
                words_batch = []
                for line in file:
                    word = line.strip()
                    words_batch.append(word.lower())
                    if len(words_batch) == batch_size:
                        self.batch_insert(words_batch)
                        words_batch = []
                if words_batch:
                    self.batch_insert(words_batch)
        except FileNotFoundError:
            print(f"Error: File 'words.txt' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred while reading file 'words.txt': {e}")

    def batch_insert(self, words):
        """
        Inserts a batch of words into the Trie.
        Input: words (list) - A list of words to be inserted into the Trie.
        Output: None
        """
        try:
            for word in words:
                self.insert(word)
        except Exception as e:
            print(f"Error: An unexpected error occurred during batch insertion: {e}")

    def search(self, prefix):
        """
        Searches for words in the Trie based on a given prefix.
        Input: prefix (str) - The prefix for which words need to be searched.
        Output: list - A list of words that match the prefix.
        """
        try:
            node = self._traverse(prefix)
            if not node:
                return []
            return self._get_words_from_node(node, prefix)
        except Exception as e:
            print(f"Error: An unexpected error occurred during search: {e}")
            return []

    def _traverse(self, prefix):
        """
        Traverses the Trie based on the given prefix.
        Input: prefix (str) - The prefix to traverse the Trie.
        Output: TrieNode - The node at the end of the prefix.
        """
        try:
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return None
                node = node.children[char]
            return node
        except Exception as e:
            print(f"Error: An unexpected error occurred during traversal: {e}")
            return None

    def _get_words_from_node(self, node, prefix):
        """
        Gets words from a given node in the Trie.
        Input: node (TrieNode) - The node from which to fetch words.
               prefix (str) - The prefix of the words.
        Output: list - A list of words from the node.
        """
        try:
            results = []
            if node.is_end_of_word:
                results.append(prefix)
            for char in node.children:
                results.extend(self._get_words_from_node(node.children[char], prefix + char))
            return results
        except Exception as e:
            print(f"Error: An unexpected error occurred during fetching words: {e}")
            return []

    
    
