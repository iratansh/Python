class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0
    
    def insert(self, key, value):
        if self.root is None:
            self.root = Node(key, value)
        else:
            self._insert(self.root, key, value)
        self.size += 1
    
    def _insert(self, node, key, value):
        if key < node.key:
            if node.left is None:
                node.left = Node(key, value)
            else:
                self._insert(node.left, key, value)
        elif key > node.key:
            if node.right is None:
                node.right = Node(key, value)
            else:
                self._insert(node.right, key, value)
        else:
            node.value = value

    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, node, key):
        if node is None:
            return None
        if key < node.key:
            return self._search(node.left, key)
        elif key > node.key:
            return self._search(node.right, key)
        else:
            return node.value
        
    def __len__(self):
        return self.size
    
    def __contains__(self, key):
        return self.search(key) is not None
    
    def __getitem__(self, key):
        return self.search(key)
    
    def __setitem__(self, key, value):
        self.insert(key, value)

    def __iter__(self):
        return self.root.__iter__()
    
    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return str(self.root)
    
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
    
    def __iter__(self):
        if self.left is not None:
            for node in self.left:
                yield node
        yield self
        if self.right is not None:
            for node in self.right:
                yield node
    
    def __str__(self):
        return f'{self.key}: {self.value}'
    
    def __repr__(self):
        return f'{self.key}: {self.value}'
    
    