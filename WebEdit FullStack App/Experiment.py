"""
Compare the search times of Trie and Hashmap data structures for searching words with a given prefix.
"""

from collections import defaultdict
from Trie import Trie
import time

class Hashmap:
    def __init__(self):
        self.hash_map = defaultdict(list)

    def insert(self, word):
        self.hash_map[word[0]].append(word)

    def search(self, prefix):
        """
        Returns a list of words from the hashmap that start with the given prefix.
        """
        prefix = prefix.lower()  # Convert prefix to lowercase for consistency
        results = []
        for word in self.hash_map.get(prefix[0], []):
            if word.startswith(prefix):
                results.append(word)
        return results


def run_experiment():
    trie = Trie()
    hashmap = Hashmap()

    # Read words from 'words.txt' and insert into both data structures
    with open('words.txt', 'r', encoding='utf-8') as file:
        words = [line.strip().lower() for line in file]

    for word in words:
        trie.insert(word)
        hashmap.insert(word)

    # Define prefixes for search operations
    prefixes = ['pro', 'test', 'abc']

    # Measure search times for Trie
    trie_search_times = []
    for prefix in prefixes:
        start_time = time.time()
        trie.search(prefix)
        end_time = time.time()
        trie_search_times.append(end_time - start_time)
    average_trie_search_time = sum(trie_search_times) / len(trie_search_times)

    # Measure search times for Hashmap
    hashmap_search_times = []
    for prefix in prefixes:
        start_time = time.time()
        hashmap.search(prefix)
        end_time = time.time()
        hashmap_search_times.append(end_time - start_time)
    average_hashmap_search_time = sum(hashmap_search_times) / len(hashmap_search_times)

    # Print results
    print(f"Average search time for Trie: {average_trie_search_time:.6f} seconds")
    print(f"Average search time for Hashmap: {average_hashmap_search_time:.6f} seconds")

    # Calculate reduction in search time
    if average_hashmap_search_time != 0:
        reduction_percentage = ((average_hashmap_search_time - average_trie_search_time) / average_hashmap_search_time) * 100
        print(f"Reduction in search time using Trie: {reduction_percentage:.2f}%")
    else:
        print("Cannot calculate reduction percentage: Hashmap search time is zero.")

if __name__ == "__main__":
    run_experiment()
