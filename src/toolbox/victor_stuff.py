from typing import Tuple, List

class Video:pass
class Image:pass

class Line:
    def __init__(
            self,
            start_point : List[int],
            end_point : List[int],
            axis : int,
            half_part : int,
            count_value: int
    ):
        self._start_point = start_point
        self._end_point = end_point
        self.count_value = count_value
        if axis not in [0, 1]:
            raise ValueError("axis must be 0 or 1")
        else:
            self.axis = axis  # 0 for vertical, 1 for vertical
        if half_part not in [0, 1]:
            raise ValueError("half_part must be 0 or 1")
        else:
            self.half_part = half_part  # 0 for first part, 1 for second
        self.is_valid = None
        self.a, self.b = self.compute_a_and_b()


    @property
    def end_point(self):
        return self._end_point


    @end_point.setter
    def end_point(self, value):
        self._end_point = value
        self.a, self.b = self.compute_a_and_b()

    @property
    def start_point(self):
        return self._start_point


    @start_point.setter
    def start_point(self, value):
        self._start_point = value
        self.a, self.b = self.compute_a_and_b()


    def compute_a_and_b(self):
        # y = ax + b
        if self._end_point[0] != self._start_point[0]:
            a = (self._end_point[1] - self._start_point[1]) / (self._end_point[0] - self._start_point[0])
        else:
            a = 10000000
        b = self._start_point[1] - a * self._start_point[0]
        return a, b

class Rectangle:
    def __init__(self, start_x, start_y, width=None, height=None, end_x=None, end_y=None):
        if not height is None:
            end_y = start_y+height
        if not width is None:
            end_x = start_x+width
        if None in (start_x, start_y, end_x, end_y):
            raise ValueError("Rectangle init does not specify a rectangle")
        (self.start_x, self.start_y, self.end_x, self.end_y) = (start_x, start_y, end_x, end_y)

    @property
    def width(self):
        return self.end_x - self.start_x

    @property
    def height(self):
        return self.end_y - self.start_y

    def __str__(self):
        return f"Rectangle([{self.start_x}, {self.start_y}], [{self.end_x}, {self.end_y}])"

    def __repr__(self):
        return self.__str__(self)
