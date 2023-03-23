import collections
class Node(object):
    def __init__(self):
        # 26个英文字符，{字符：Node}
        self.children = collections.defaultdict(Node)
        # isWord 表示从根节点到当前节点为止，该路径是否形成了一个有效的字符串
        self.isword = False

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        current = self.root
        # 拆解关键词，每个字符按照其在 'a' ~ 'z' 的序号，放在对应的 chidren 里面。下一个字符是当前字符的子节点
        for w in word:
            current = current.children[w]
        current.isword = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        current = self.root
        # 依次遍历该关键词所有字符，在前缀树中找出这条路径。
        for w in word:
            current = current.children.get(w)
            if current == None:
                return False
        return current.isword

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        current = self.root
        for w in prefix:
            current = current.children.get(w)
            if current == None:
                return False
        return True

