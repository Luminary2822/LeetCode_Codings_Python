class MyHashSet(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashSet = []

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        if key not in self.hashSet:
            self.hashSet.append(key)

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        if key in self.hashSet:
            self.hashSet.remove(key)


    def contains(self, key):
        """
        Returns true if this set contains the specified element
        :type key: int
        :rtype: bool
        """
        if key in self.hashSet:
            return True
        return False