class ParkingSystem(object):
    
    def __init__(self, big, medium, small):
        """
        :type big: int
        :type medium: int
        :type small: int
        """
        # 利用数组保存剩余车位书目，下标获取对应车型停车位数目
        self.parkArea = [0, big, medium, small]


    def addCar(self, carType):
        """
        :type carType: int
        :rtype: bool
        """
        # 利用车型索引到空余车位不为0的情况就停车进去，车位减一
        if self.parkArea[carType]:
            self.parkArea[carType] -= 1
            return True
        return False
