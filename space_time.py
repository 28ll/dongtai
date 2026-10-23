from typing import Tuple

class SpaceTimeGrid:
    def __init__(self):
        self.occupancy = {}

    def get_next_free_time(self, edge: Tuple[int, int], start: float, duration: float) -> float:
        if edge not in self.occupancy:
            return start
        intervals = self.occupancy[edge]
        current = start
        for s, e in intervals:
            if current + duration <= s:
                return current
            current = max(current, e)
        return current

    def reserve(self, edge: Tuple[int, int], start: float, end: float):
        if edge not in self.occupancy:
            self.occupancy[edge] = []
        intervals = self.occupancy[edge]
        new_interval = [start, end]
        merged = []
        for i in sorted(intervals + [new_interval]):
            if not merged or i[0] >= merged[-1][1]:
                merged.append(i)
            else:
                merged[-1][1] = max(merged[-1][1], i[1])
        self.occupancy[edge] = merged

    def get_congestion(self, edge: Tuple[int, int]) -> float:
        if edge not in self.occupancy:
            return 1.0
        return 1.0 + 0.2 * len(self.occupancy[edge])