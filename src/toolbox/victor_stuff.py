from __future__ import annotations
from typing import Tuple, List, Union
import numpy as np, pandas as pd, tqdm, logging
import collections, pathlib
from toolbox.ressource_manager import mk_loader_with_error, mk_json_loader, RessourceLoader

logger = logging.getLogger(__name__)

Image = np.ndarray

class Video(collections.abc.Sequence):
    def __init__(self, path=None, copy: Video=None):
        self.is_from_img_list=False
        if not copy is None:
            self.vid = copy.vid
            self.transformations=copy.transformations.copy()
            self.width=copy.width
            self.height=copy.height
            self.source_path = copy.source_path
            self.is_from_img_list=copy.is_from_img_list
            self.img_list = [img for img in copy.img_list] if not copy.img_list is None else None
            self.start_frame=copy.start_frame
            self.end_frame = copy.end_frame
        elif isinstance(path, str) or isinstance(path, pathlib.Path):
            import cv2
            self.vid = cv2.VideoCapture(str(path))
            self.source_path = str(path)
            self.transformations=[]
            self.start_frame=0
            self.end_frame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = self[0].shape[1]
            self.height = self[0].shape[0]
            self.is_from_img_list=False
            self.img_list = None
            
        else:
            self.vid= None
            self.source_path = "from imgs"
            self.is_from_img_list=True
            self.img_list = path
            self.start_frame=0
            self.end_frame = len(self.img_list)
            self.width = self[0].shape[1]
            self.height = self[0].shape[0]
            self.transformations=[]
            

    @property
    def fps(self):
        if self.is_from_img_list:
            return 30
        import cv2
        return self.vid.get(cv2.CAP_PROP_FPS) 
    
    @property
    def nb_frames(self):
        return self.end_frame-self.start_frame
    
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
    
    def cut(self, start, end):
        v = self.copy()
        v.start_frame = v.start_frame+start
        v.end_frame = v.start_frame +end
        return v
    
    def __getitem__(self, pos):
        import cv2
        if isinstance(pos, int):
            if pos >= self.nb_frames:
                raise IndexError(f"Video has only {self.nb_frames} frames. Trying to access frame {pos}")
            pos = pos - self.start_frame
            if self.is_from_img_list:
                return self.img_list[pos]
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame =  self.vid.read()
            if not ret is True:
                raise RuntimeError(f"Unable to read frame. Read returned: {ret}")
            else:
                mret =  self._transform(frame, pos)
                return mret
            
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
        if not self.is_from_img_list:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.iterpos=0
        return self
    
    def __next__(self):
        if self.iterpos < self.nb_frames:
            if self.is_from_img_list:
                ret, frame = None, self.img_list[self.iterpos]
            else:
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
    
    def add_text(self, text: Union[str, List[str]], position: Tuple[int, int] = None, color = (0, 0, 255), copy=True):
        import cv2
        if copy:
            vid = self.copy()
        else:
            vid = self
            
        if position is None:
            position = (int(self.width/10), int(self.height/10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if isinstance(position, str):
            vid.transformations.append(("map", lambda frame: cv2.putText(frame.copy(), text, position, font, 2, color, 2, cv2.LINE_AA)))
        else:
            vid.transformations.append(("imap", lambda i, frame: cv2.putText(frame.copy(), text[i], position, font, 2, color, 2, cv2.LINE_AA)))
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
        if frame is None:
            raise BaseException("No frame")
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
    
    def as_dict(self):
        return {"start_x": self.start_x, "start_y": self.start_y, "end_x": self.end_x, "end_y": self.end_y}
    
    

def mk_rectangle_loader() -> RessourceLoader:
    def save(path, r: Rectangle):
        mk_json_loader().save(path, {"start_x": r.start_x, "start_y": r.start_y, "end_x": r.end_x, "end_y": r.end_y})
    def load(path):
        return Rectangle(**mk_json_loader().load(path))
    return RessourceLoader(mk_json_loader().extension, load, save)


rectangle_loader = mk_loader_with_error(mk_rectangle_loader())

import collections.abc
class MatPlotLibObject:
    def __init__(self, show_func, n=None, subplots: Union[Tuple[int, int], List[Tuple[int, int]]]=(1, 1), text="mplo"):
        if len(subplots) > 0 and isinstance(subplots[0], collections.abc.Sequence):
            if not n is None and len(subplots!=n):
                raise ValueError("Subplot length does not match number of requested figures")
            self.subplots = subplots
        elif not n is None: 
            self.subplots = [subplots for i in range(n)]
        else: 
            self.subplots = [subplots for i in range(1)]

        for s in self.subplots:
            if not len(s) == 2:
                raise ValueError("Subplot requires two dimensions")
            
        self.show_func = show_func
        self.text = text

    def show(self, rtab):
        from toolbox import mk_result_tab
        for n, s in enumerate(self.subplots):
            tab, mpls = mk_result_tab(s[0], s[1])
            import PyQt5.QtWidgets as QtWidgets
            # tab = QtWidgets.QWidget()
            for i in range(s[0]):
                for j in range(s[1]):
                    self.show_func(n, i, j, mpls[i, j].canvas.ax)
            rtab.addTab(tab, self.text)
        return tab


def mk_mplo_loader() -> RessourceLoader:
    def save(path, r: Rectangle):
        raise NotImplementedError("mplo loader save")
    def load(path):
        raise NotImplementedError("mplo loader load")
    return RessourceLoader(".mplo", load, save)

mplo_loader = mk_loader_with_error(mk_mplo_loader())
