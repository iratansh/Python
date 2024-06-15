class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word_count += 1

    def search(self, prefix, k=3):
        node = self._traverse(prefix)
        if not node:
            return []
        return self._get_words_from_node(node, prefix, k)

    def _traverse(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _get_words_from_node(self, node, prefix, k):
        results = []
        if node.is_end_of_word:
            results.append(prefix)
        if len(results) >= k:
            return results
        for char in sorted(node.children.keys()):  
            results.extend(self._get_words_from_node(node.children[char], prefix + char, k))
            if len(results) >= k:
                break
        return results

    
    
