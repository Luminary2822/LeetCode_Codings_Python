class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        if len(coordinates) <= 2:
            return True
        x0, y0 = coordinates[0]
        x1, y1 = coordinates[1]
        for x, y in coordinates[2:]:
            if (y0 - y1) * (x1 - x) != (x0 - x1) * (y1 - y):
                return False
        return True