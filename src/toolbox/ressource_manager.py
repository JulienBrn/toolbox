from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, List, Literal, IO, Callable
import pathlib
import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)

class RessourceLoader: #Interface from which one may inherit
  extension: str
  load: Callable[[pathlib.Path], Any]
  save: Callable[[pathlib.Path, Any], None]

  def __init__(self, name: str, ext: str, load, save):
    self.name = name
    self.extention=ext
    self.load = load
    self.save = save

def mk_df_loader():
  def load(path):
    return pd.read_csv(str(path), sep="\t")
  
  def save(path, df):
    df.to_csv(str(path), sep="\t")

  return RessourceLoader("tsv", load, save)

def mk_numpy_loader():
  def load(path):
    return np.load(str(path), allow_pickle=False)
  
  def save(path, a):
    return np.save(str(path), a, allow_pickle=False)

  return RessourceLoader("npy", load, save)

df_loader = mk_df_loader()
np_loader = mk_numpy_loader()

class Manager:
  IdType = str
  base_folder: pathlib.Path


  ######### USER API ##############################

  def __init__(self, base_folder): pass

  def create_value_ressource(self, value, path, loader, storage: List[str], name="ValueRessource") -> RessourceHandle: 
    """ 
      Path must not exist !
      Path must be compatible with extension
      Will be saved on disk on call
      May be removed from memory on call (depends on unload)
    """
    
    raise NotImplementedError("This function is probably useless")
  


  def declare_ressource(self, path, loader, name="ValueRessource") -> RessourceHandle:
    """ 
      Path must exist !
      Path must be compatible with extension
      Will not be removed from Disk
      May be loaded when necessary in memory
    """ 
    if not pathlib.Path(path).exists():
      raise ValueError("file {} must exist to declare a ressource from that location".format(path))
    if pathlib.Path(path).suffix != loader.extension:
      raise ValueError("file {} must have extension {} (defined by loader)".format(path, loader.extension))
    
    ressource = Manager.Ressource(self, path, name, loader, path)
    self.d[path] = ressource
    return ressource.handle

  def declare_computable_ressource(self, function, params: Dict[str, Any], loader, name, save: bool | None = None) -> RessourceHandle:
    """
      Path is auto-computed
      Path may already exist or not on disk
      May or may not be saved on disk (after computation), depends on save
    """
    def single_key_func(*args, **kwargs):
      return {0 : function(*args, **kwargs)}
    return self.declare_computable_ressources(single_key_func, params, {0: (loader, id, save)})[0]

  def declare_computable_ressources(self, function, params: Dict[str, Any], out: Dict[Any, Tuple[RessourceLoader, str, bool | None]]) -> Dict[str, RessourceHandle]:
    """
      Same, but this time the function can compute several ressources
      the set of results is identified by: {key: function(params)[key] for key in out.keys()}
      Returns a set of ressources indexed by name (second argument of tuple)
    """
    def mk_id(loader, name, params):
      return "{}_{}".format(name, params)
    
    new_out = {key: (loader, mk_id(loader, name, params), save) for key, (loader, name, save) in out.items()}

    computer = Manager.Computer(function, params, new_out)
    def mk_ressource(loader, name, save):
      id = mk_id(loader, name, params)
      ressource = Manager.Ressource(self, id, name, loader, self.base_folder, computer)
      self.d[id] = ressource
      return ressource.handle

    return {name: mk_ressource(loader, name, save) for _,(loader, name, save) in out.items()}



  ########### API for Handles #######################

  def get_disk_path(self, id):
    return self.d[id].get_disk_path()

  def get_result(self, id): 
    """
      Fetches result for ressource in the following priority: Memory, Disk, Compute.
      May compute other ressources while doing so.
    """
    ressource = self.d[id]
    if "Memory" in ressource.storage_locations: #For now, do not additionaly save on disk
      if ressource.value is None:
        logger.error("Value is None but should be stored in memory. Probably a bug in the ressource manager.")
      return ressource.value
    if "Disk" in ressource.storage_locations: #For now, we keep in memory
      ressource.disk_to_memory()
      return ressource.value
    if ressource.storage_locations != []:
      logger.error("Unknown storage location")
    if ressource.computer is None:
      raise BaseException("Ressource is not saved but cannot be either computed...")
    else:
      real_params = {key:val if not isinstance(val, RessourceHandle) else val.get_result() for key, val in ressource.computer.params.items()}
      res = ressource.computer.func(**real_params)
      for key, (loader, childid, save) in ressource.computer.out.items():
        rec = self.d[childid]
        if "Memory" not in rec.storage_locations:
          rec.value = res[key]
          rec.storage_locations.append("Memory")
          if "Disk" not in rec.storage_locations and save:
            rec.memory_to_disk(self.base_folder)
      if "Memory" not in ressource.storage_locations:
        raise BaseException("Ressource should have been computed but is not")
      if ressource.value is None:
        logger.error("Value is None but should be stored in memory. Probably a bug in the ressource manager.")
      return ressource.value

  def save_on_disk(self, id):
    ressource = self.d[id]
    self.get_result(id)
    if "Disk" in ressource.storage_locations:
      return
    if "Memory" in ressource.storage_locations:
      ressource.memory_to_disk(self.base_folder)
    else:
      raise BaseException("Ressource should have been computed but is not")
    

  def is_stored(self, id) -> bool:
    return self.d[self.id].storage_locations != []

  def is_in_memory(self, id) -> bool: 
    return "Memory" in self.d[id].storage_locations
  
  def is_saved_on_disk(self, id) -> bool:
    return "Disk" in self.d[self.id].storage_locations

  ######## internals ################
  d: Dict[IdType, Ressource]

  class Ressource:
    base_storage: pathlib.Path
    handle: RessourceHandle
    loader: RessourceLoader
    value: Any | None = None
    computer: None | Manager.Computer = None   #If it has a computer is may be removed from disk depending on its save param
    storage_locations: List[str] = [] #Memory or Disk

    def __init__(self, manager, id, name, loader, base_storage, computer = None):
      self.handle = RessourceHandle(manager, id, name)
      self.loader = loader
      self.base_storage = base_storage
      self.computer = computer

    def get_disk_path(self) -> pathlib.Path:
      pass

    def memory_to_disk(self): pass

    def disk_to_memory(self): pass

  class Computer:
    func: Callable
    params: Dict[str, Any]
    out: Dict[Any, Tuple[RessourceLoader, Manager.IdType, bool | None]]

    def __init__(self, func, params, out: Dict[Any, Tuple[RessourceLoader, Manager.IdType, bool | None]]):
      self.func = func
      self.params = params
      self.out = out


class RessourceHandle:
  id: Manager.IdType   # A (short) unique id (probably of type string). Only used because may be simpler to save than path.
  name: str #Used for nice printing
  manager : Manager

  def __init__(self, manager, id, name):
    self.id = id
    self.manager = manager
    self.name = name

  def get_result(self) -> Any: 
    self.manager.get_result(self.id)

  def get_disk_path(self) -> pathlib.Path:
    return self.manager.get_disk_path(self.id)
  
  def is_in_memory(self) -> bool: 
    return self.manager.is_in_memory(self.id)
  
  def is_saved_on_disk(self) -> bool:
    return self.manager.is_saved_on_disk(self.id)
  
  def is_stored(self) -> bool:
    return self.manager.is_stored(self.id)
  
  def save(self) -> bool: 
    self.manager.save_on_disk(self.id)



