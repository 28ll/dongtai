from typing import List

class Vehicle:
    def __init__(self, vid: str, vtype: str):
        self.id = vid
        self.type = vtype
        if vtype == 'small':
            self.capacity = 3
            self.fill_time = 5   # 分钟
        elif vtype == 'medium':
            self.capacity = 8
            self.fill_time = 8
        elif vtype == 'large':
            self.capacity = 16
            self.fill_time = 12
        else:
            self.capacity = 0
            self.fill_time = 0
        self.status = 'idle'
        self.position = None

class FireStation:
    def __init__(self, sid: str, position: str):
        self.id = sid
        self.position = position
        self.vehicles: List[Vehicle] = []

class WaterSource:
    def __init__(self, wid: str, position: str):
        self.id = wid
        self.position = position

class FireEvent:
    def __init__(self, fid: str, position: str, level: int):
        self.id = fid
        self.position = position
        self.level = level

WATER_DEMAND = {1: 5, 2: 35, 3: 70}
DISCHARGE_RATE = 1.0