from __future__ import annotations
from typing import Tuple, List, Union
import numpy as np, pandas as pd, tqdm, logging
import collections
from toolbox.ressource_manager import mk_loader_with_error, mk_json_loader, RessourceLoader

logger = logging.getLogger(__name__)

Image = np.ndarray

class Video(collections.abc.Sequence):
    def __init__(self, path=None, copy: Video=None):
        if not copy is None:
            self.vid = copy.vid
            self.transformations=copy.transformations.copy()
            self.width=copy.width
            self.height=copy.height
            self.source_path = copy.source_path
        elif not path is None:
            import cv2
            self.vid = cv2.VideoCapture(str(path))
            self.source_path = str(path)
            self.transformations=[]
            self.width = self[0].shape[1]
            self.height = self[0].shape[0]

    @property
    def fps(self):
        import cv2
        return self.vid.get(cv2.CAP_PROP_FPS) 
    
    @property
    def nb_frames(self):
        import cv2
        return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def duration(self):
        return self.nb_frames/self.fps
    
    # @property
    # def height(self):
    #     return self[0].shape[0]
    
    # @property
    # def width(self):
    #     return self[0].shape[1]
    

    def __str__(self):
        return f"Video(#frames{self.nb_frames}, w={self.width}, h={self.height}, fps={self.fps}, duration={self.duration})"
    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return Video(copy=self)
    
    def __len__(self):
        return self.nb_frames
    
    def __getitem__(self, pos):
        import cv2
        if isinstance(pos, int):
            if pos >= self.nb_frames:
                raise IndexError(f"Video has only {self.nb_frames} frames. Trying to access frame {pos}")
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame =  self.vid.read()
            if not ret is True:
                raise RuntimeError(f"Unable to read frame. Read returned: {ret}")
            else:
                return self._transform(frame, pos)
            
        # if isinstance(pos, slice):
        #     frames=[]
        #     for curr_pos in range(slice.start, min(slice.stop, self.nb_frames), slice.step):
        #         self.vid.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)
        #         ret, frame =  self.vid.read()
        #         if not ret is True:
        #             raise RuntimeError(f"Unable to read frame. Read returned: {ret}")
        #         else:
        #             frames.append(self._transform(frame, curr_pos))
        #     return frames
        else:
            raise NotImplementedError(f"Unhandled indexing type {type(pos)} for video")
        
    def __iter__(self):
        import cv2
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.iterpos=0
        return self
    
    def __next__(self):
        if self.iterpos < self.nb_frames:
            ret, frame = self.vid.read()
            res = self._transform(frame, self.iterpos)
            self.iterpos+=1
            return res
        else:
            raise StopIteration
    # @property
    # def position(self):
    #     import cv2
    #     return int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    # @position.setter
    # def position(self, pos:int):
    #     import cv2
    #     self.vid.set(cv2.CAP_PROP_POS_FRAMES, pos)

    # def __copy__(self):
    #     cp = Video(self.source_path)
    #     cp.transformations = self.transformations
    #     cp.position = self.position
    #     return cp
    
    # def copy(self):
    #     return self.__copy__()

    def crop(self, crop: Union[Rectangle, pd.DataFrame], copy=True):
        if copy:
            vid = self.copy()
        else:
            vid = self
            
        if isinstance(crop, Rectangle):
            vid.width = crop.width
            vid.height = crop.height
            vid.transformations.append(("map", lambda frame: frame[crop.start_y:crop.end_y, crop.start_x:crop.end_x]))
        if isinstance(crop, pd.DataFrame):
            vid.width = crop["end_x"].iat[0] - crop["start_x"].iat[0]
            vid.height = crop["end_y"].iat[0] - crop["start_y"].iat[0]
            vid.transformations.append(("imap", lambda i, frame: frame[crop["start_y"].iat[i]:crop["end_y"].iat[i], crop["start_x"].iat[i]:crop["end_x"].iat[i]]))
        return vid

    # def get_frame(self, f: int = None):
    #     if f is None:
    #         f = self.position
    #         ret, frame =  self.vid.read()
    #     else:
    #         pos = self.position
    #         self.position = f
    #         ret, frame = self.vid.read()
    #         self.position=pos
    #     if not ret is True:
    #         raise RuntimeError(f"Unable to read frame. Read returned: {ret}")
    #     return self._transform(frame, f)

    def save(self, path, tqdm = tqdm.tqdm):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
        for f in tqdm(self, desc = f"Writing video {path}"):
            if self.width != f.shape[1] or self.height != f.shape[0]:
                logger.warning(f"Problem when writting frame. Dimensions do not match. Video size ({self.width}, {self.height}), frame size ({f.shape[1]}, {f.shape[0]})")
            out.write(f)
        out.release()

    def _transform(self, frame: Image, i):
        if self.transformations is []:
            return frame
        f = frame.copy()
        for t in self.transformations:
            if t[0] == "map":
                f = t[1](f)
            elif t[0] == "imap":
                f = t[1](i, f)
            else:
                raise NotImplementedError("Transformation type not implemented")
        return f

def mk_video_loader():
    def load(path):
        return Video(path)
    def save(path, vid):
        vid.save(path)
    return RessourceLoader(".mp4", load, save)

video_loader = mk_loader_with_error(mk_video_loader())

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
        (self.start_x, self.start_y, self.end_x, self.end_y) = tuple([int(x) for x in (start_x, start_y, end_x, end_y)])

    @property
    def width(self):
        return self.end_x - self.start_x

    @property
    def height(self):
        return self.end_y - self.start_y

    def __str__(self):
        return f"Rectangle([{self.start_x}, {self.start_y}], [{self.end_x}, {self.end_y}])"

    def __repr__(self):
        return self.__str__()
    

def mk_rectangle_loader() -> RessourceLoader:
    def save(path, r: Rectangle):
        mk_json_loader().save(path, {"start_x": r.start_x, "start_y": r.start_y, "end_x": r.end_x, "end_y": r.end_y})
    def load(path):
        return Rectangle(**mk_json_loader().load(path))
    return RessourceLoader(mk_json_loader().extension, load, save)


rectangle_loader = mk_loader_with_error(mk_rectangle_loader())