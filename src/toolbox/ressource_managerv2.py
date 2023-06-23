from __future__ import annotations
from typing import Literal, List, Tuple, Union, Dict, Any
import psutil

class Delayed:
    def __init__(self, f):
        self.f = f

    def get(self, tqdm = None): 
        self.f()


class Storage:
    def mk_storage_id(self, rec: Memoized) -> Any: pass
    def save_ressource(self, value: Any, storage_id, allow_updates=False) -> None: pass
    def load_ressource(self, storage_id: Any): pass
    def has_ressource(self, storage_id) -> bool: pass
    def remove_ressource(self, storage_id): pass
    def available_space_info(self) -> float: pass

class MetadataStorage(Storage):
    def get_all_ressources(self):pass



def MemoryStorage(Storage):
    def __init__(self):
        self.d= {}

    def mk_storage_id(self, rec: Memoized) -> Any: 
        return rec.persistent_id

    def save_ressource(self, value: Any, storage_id, allow_updates=False):
        if storage_id in self.d and not allow_updates:
            raise RuntimeError(f"Ressource with id {storage_id} already stored in memory")
        self.d[storage_id] = value

    def load_ressource(self, storage_id: Any):
        if not storage_id in self.d:
            raise RuntimeError(f"No ressource stored in memory at {storage_id}")
        return self.d[storage_id]
    
    def has_ressource(self, storage_id) -> bool:
        return storage_id in self.d
    
    def available_space_info(self) -> float: 
        mem = psutil.virtual_memory()
        return mem.available/mem.total
    
class DiskStorage(Storage):
    def __init__(self, storage_base_path="cache"):
        self.storage_base_path = storage_base_path

    def mk_storage_id(self, rec: Memoized) -> Any: 
        rec.get_initial_params_dict()


    def save_ressource(self, value: Any, storage_id, allow_updates=False) -> None: pass
    def load_ressource(self, storage_id: Any): pass
    def has_ressource(self, storage_id) -> bool: pass
    def remove_ressource(self, storage_id): pass
    def available_space_info(self) -> float: pass
    

    

def make_hashable(obj):
    if isinstance(obj, list):
        return tuple([make_hashable(o) for o in obj])
    if isinstance(obj, dict):
        return tuple(frozenset([make_hashable(o) for o in obj.items()]))

class Memoized(Delayed):
    def __init__(self, name, f, kwargs = {}, result_output_computation = "return_value", storage = None, metadata_storage = None, speed=None):
        super().__init__(f)
        self.kwargs = kwargs
        self.name = name
        self.persistent_id = hash((name, make_hashable(kwargs.items())))
        self.is_computed = False
        self.childs = []
        #append childs to other ressources
        #add_to_list_of_existing ressources_to_manager

    def __hash__(self):
        return self.persistent_id
    def __eq__(self, other):
        return (self.name, frozenset(self.kwargs.items())) == (other.name, frozenset(other.kwargs.items()))
    
    def get(self, tqdm = None):pass



        