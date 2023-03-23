class MyHashMap(object):
# 超大数组：用空间换时间
    def __init__(self):
        self.data = [-1] * (10**6 + 1)

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def remove(self, key):
        self.data[key] = -1
