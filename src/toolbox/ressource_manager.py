from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, List, Literal, IO, Callable
import pathlib
import logging
import numpy as np
import sys
import toolbox
import scipy
import mat73
import os
import psutil, hashlib
import numpy as np
import traceback

logger = logging.getLogger(__name__)

class RessourceLoader: #Interface from which one may inherit
  extension: str
  load: Callable[[pathlib.Path], Any]
  save: Callable[[pathlib.Path, Any], None]

  def __init__(self, ext: str, load, save):
    self.extension=ext
    self.load = load
    self.save = save

def mk_df_loader():
  def load(path):
    return pd.read_csv(str(path), sep="\t")
  
  def save(path, df: pd.DataFrame):
    df.to_csv(str(path), sep="\t", index=False)

  return RessourceLoader(".tsv", load, save)

def mk_numpy_loader():
  def load(path):
    return np.load(str(path), allow_pickle=False)
  
  def save(path, a):
    return np.save(str(path), a, allow_pickle=False)

  return RessourceLoader(".npy", load, save)

def mk_float_loader():
  def load(path):
    with open(path, "r") as f:
      return float(f.read())
  
  def save(path, val: float):
    with open(path, "w") as f:
      f.write(str(val))

  return RessourceLoader(".txt", load, save)

def mk_matlab_loader():
  def load(path):
    try:
      return scipy.io.loadmat(path)
    except NotImplementedError:
      return mat73.loadmat(path)
  
  def save(path, d: Dict[str:Any]):
    scipy.io.savemat(path, d)

  return RessourceLoader(".mat", load, save)

def mk_matlab73_loader():
  def load(path):
    return mat73.loadmat(path)
  
  def save(path, d: Dict[str:Any]):
    scipy.io.savemat(path, d)

  return RessourceLoader(".mat", load, save)

import json

def mk_json_loader():
  def load(path):
     with open(str(path), "r") as fp:
      return json.load(fp)
  
  def save(path, d):
    with open(str(path), "w") as fp:
      json.dump(d , fp, indent=4) 

  return RessourceLoader(".mat", load, save)

class Error:
  def __init__(self, e, tb=None):
    self.e = e
    self.tb = tb
  def __str__(self):
    if self.tb is None:
      return "Error({})".format(self.e)
    else:
      return "Error({}).\n{}".format(self.e, self.tb)
  def __repr__(self):
    return self.__str__()
  
def mk_loader_with_error(loader):
  def save(path, d):
    if isinstance(d, Error):
      with open(str(path), 'w') as f:
        f.write('__Error:{}'.format(d.e))
    else:
      loader.save(path, d)

  def load(path):
    with open(str(path), 'r') as f:
      try:
        s = f.read(8)
        if s == '__Error:':
          return Error(BaseException(f.read()))
      except:
        pass
    return loader.load(path)
  return RessourceLoader(loader.extension, load, save)
    


df_loader = mk_loader_with_error(mk_df_loader())
np_loader = mk_loader_with_error(mk_numpy_loader())
float_loader = mk_loader_with_error(mk_float_loader())
matlab_loader = mk_loader_with_error(mk_matlab_loader())
matlab73_loader = mk_loader_with_error(mk_matlab73_loader())
json_loader = mk_loader_with_error(mk_json_loader())



class Manager:
  IdType = str
  base_folder: pathlib.Path


  ######### USER API ##############################

  def __init__(self, base_folder): 
    self.base_folder = base_folder
    pathlib.Path(base_folder).mkdir(parents=True, exist_ok=True)

  def create_value_ressource(self, value, path, loader, storage: List[str], name="ValueRessource") -> RessourceHandle: 
    """ 
      Path must not exist !
      Path must be compatible with extension
      Will be saved on disk on call
      May be removed from memory on call (depends on unload)
    """

    raise NotImplementedError("This function is probably useless")
  


  def declare_ressource(self, path, loader, name="ValueRessource", id: str | None = None, check = True) -> RessourceHandle:
    """ 
      Path must exist !
      Path must be compatible with extension
      Will not be removed from Disk
      May be loaded when necessary in memory
    """ 
    path = pathlib.Path(path)
    if id is None:
      id = str(path)
    if check:
      if not path.exists():
        raise ValueError("file {} must exist to declare a ressource from that location".format(path))
    if path.suffix != loader.extension:
      raise ValueError("file {} must have extension {} (defined by loader). Current extension is {}".format(path, loader.extension, path.suffix))
    
    ressource = Manager.Ressource(self, id, name, loader, path, storage_locations=["Disk"])
    self.d[id] = ressource
    return ressource.handle

  def declare_computable_ressource(self, function, params: Dict[str, Any], loader, name, save: bool | None = None, error_method="propagate") -> RessourceHandle:
    """
      Path is auto-computed
      Path may already exist or not on disk
      May or may not be saved on disk (after computation), depends on save
    """
    def single_key_func(*args, **kwargs):
      logger.debug("Single key for ressource {} called with {} {}".format(name, args, kwargs.keys()))
      return {0 : function(*args, **kwargs)}
    return self.declare_computable_ressources(single_key_func, params, {0: (loader, name, save)}, error_method=error_method)[name]

  def declare_computable_ressources(self, function, params: Dict[str, Any], out: Dict[Any, Tuple[RessourceLoader, str, bool | None]], error_method="propagate") -> Dict[str, RessourceHandle]:
    """
      Same, but this time the function can compute several ressources
      the set of results is identified by: {key: function(params)[key] for key in out.keys()}
      Returns a set of ressources indexed by name (second argument of tuple)
    """
    def mk_id(loader, name, params):
      return "{}_{}".format(name, params)
    
    new_out = {key: (loader, mk_id(loader, name, params), save) for key, (loader, name, save) in out.items()}

    computer = Manager.Computer(function, params, new_out, error_method=error_method)
    def mk_ressource(key, loader, name, save):
      id = mk_id(loader, name, params)
      ressource = Manager.Ressource(self, id, name, loader, self.base_folder, (computer, key))
      self.d[id] = ressource
      return ressource.handle

    return {name: mk_ressource(key, loader, name, save) for key,(loader, name, save) in out.items()}



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
      if isinstance(ressource.value, Error):
        real_params = {key:val for key, val in ressource.computer.params.items()}
        logger.error("Ressource value in memory is error for {}({}). Error is {}".format(ressource.handle.name,str(real_params), ressource.value))
      return ressource.value
    if "Disk" in ressource.storage_locations: #For now, we keep in memory
      ressource.disk_to_memory()
      ret = ressource.value
      # if isinstance(ressource.value, Error):
      #   real_params = {key:val for key, val in ressource.computer.params.items()}
      #   logger.error("Ressource value on disk is error for {}({}). Error is {}".format(ressource.handle.name,str(real_params), ressource.value))
      if not isinstance(ressource.value, Error):
        self.unload_memory_if_necessary()
        return ret
      else:
        ressource.remove_from_memory()
        ressource.remove_from_disk()
    if ressource.storage_locations != []:
      logger.error("Unknown storage location in {}".format(ressource.storage_locations))
    if ressource.computer is None:
      raise BaseException("Ressource is not saved but cannot be either computed...")
    else:
      real_params = {key:val if not isinstance(val, RessourceHandle) else val.get_result() for key, val in ressource.computer.params.items()}
      error_params = {key:val for key, val in real_params.items() if isinstance(val, Error)}
      if ressource.computer.error_method !="filter" and len(error_params) > 0:
        # print("Err method for ressource:", ressource.handle.name, ressource.computer.error_method)
        logger.error("Error while computing ressource {}({}). Error is is due to paramters {}".format(ressource.handle.name,str(real_params), error_params))
        res = Error(BaseException("Error in parameters {}".format(error_params)))
        # if 
        # return 
      else:
        logger.info("Ressource {} is being computed".format(ressource.get_disk_path()))
        try:
          res = ressource.computer.func(**{k:v for k,v in real_params.items() if not isinstance(v, Error)})
        except KeyboardInterrupt as e:
          raise e
        except BaseException as e:
          tb = traceback.format_exc()
          logger.error("Error while computing ressource {}({}). Error is {}. \n\nTraceback:\n{}".format(ressource.handle.name,str(real_params), e, tb))
          res = Error(e, tb)
      for key, (loader, childid, save) in ressource.computer.out.items():
        rec = self.d[childid]
        if "Memory" not in rec.storage_locations:
          rec.value = res[key] if not isinstance(res, Error) else res
          rec.storage_locations.append("Memory")
          if "Disk" not in rec.storage_locations and save:
            rec.memory_to_disk()
      if "Memory" not in ressource.storage_locations:
        raise BaseException("Ressource should have been computed but is not")
      if ressource.value is None:
        logger.error("Value is None but should be stored in memory. Probably a bug in the ressource manager.")
      ret = ressource.value
      self.unload_memory_if_necessary()
      return ret
    
  def unload_memory_if_necessary(self):
    mem = psutil.virtual_memory()
    if mem.available/1000000000 < 4: #25Gb
      logger.warning("Memory usage was {}, unloading".format({k:v/1000000000 for k, v in mem._asdict().items()}))
      for r in self.d.values():
        handle: RessourceHandle = r.handle
        if handle.is_saved_on_disk():
          handle.unload()
      mem = psutil.virtual_memory()
      if mem.available/1000000000 <4: #20Gb
        logger.warning("Memory usage was {} even after unloading saved results, unloading fully".format({k:v/1000000000 for k, v in mem._asdict().items()}))
        for r in self.d.values():
          handle: RessourceHandle = r.handle
          handle.unload()
      mem = psutil.virtual_memory()
      logger.warning("Finishing with memory {}".format({k:v/1000000000 for k, v in mem._asdict().items()}))
      if mem.available/1000000000 <4: #20Gb
        logger.error("Memory problem. Unable to guarantee memory space")
    

  def save_on_disk(self, id):
    ressource = self.d[id]
    self.get_result(id)
    if "Disk" in ressource.storage_locations:
      return
    if "Memory" in ressource.storage_locations:
      ressource.memory_to_disk()
    else:
      raise BaseException("Ressource should have been computed but is not")
    

  def is_stored(self, id) -> bool:
    return self.d[id].storage_locations != []

  def invalidate_all(self, id):
    ressource = self.d[id]
    if not ressource.computer is None:
      ressource.remove_from_memory()
      ressource.remove_from_disk()
    for id,rec in self.d.items():
      if rec.computer and ressource in rec.computer.params.values():
        rec.invalidate_all()

  def is_in_memory(self, id) -> bool: 
    return "Memory" in self.d[id].storage_locations
  
  def is_saved_on_disk(self, id) -> bool:
    return "Disk" in self.d[id].storage_locations
  
  def unload(self, id):
    ressource = self.d[id]
    if "Disk" in ressource.storage_locations or not ressource.computer is None:
      ressource.remove_from_memory()

  ######## internals ################
  d: Dict[IdType, Ressource] = {}

  class Ressource:
    base_storage: pathlib.Path #For non computable ressources is simply the path to ressource. 
      # Otherwise the base folder in which to store computation results
    handle: Manager.RessourceHandle
    loader: RessourceLoader
    value: Any | None = None
    computer: None | Manager.Computer = None   #If it has a computer is may be removed from disk depending on its save param
    computer_key: None | Any = None
    storage_locations: List[str] #Memory or Disk
        
    def __init__(self, manager, id, name, loader, base_storage, computing = (None, None), storage_locations = []):
      self.handle = RessourceHandle(manager, id, name)
      # if "Disk" in self.storage_locations:
      #   logger.error("Ressource {} is on Disk. Locations: {}".format(self.handle.name, self.storage_locations))
      self.loader = loader
      self.base_storage = base_storage
      self.computer = computing[0]
      self.computer_key = computing[1]
      self.storage_locations = storage_locations.copy()
      # if "Disk" in self.storage_locations:
      #   logger.error("Ressource {} is on Disk. Locations: {}".format(self.handle.name, self.storage_locations))
      path = self.get_disk_path()
      # logger.debug("Path is {}".format(path))
      # if path.exists():
      if not "Disk" in self.storage_locations:
        if os.path.isfile(str(path)) and pathlib.Path(path).exists():
          self.storage_locations.append("Disk")
        # logger.debug("Path {} exists".format(path))
      # else:
      #   logger.debug("Path {} does not exist".format(path))
      # if "Disk" in self.storage_locations:
      #   logger.debug("Ressource {} is on Disk".format(self.handle.name))


    def get_disk_path(self) -> pathlib.Path:
      if hasattr(self, "disk_path"):
        return self.disk_path
      # if "Disk" in self.storage_locations:
      #   logger.debug("Ressource {} is on Disk".format(self.handle.name))
      if self.computer is None:
        self.disk_path = self.base_storage
        self.old_disk_path = self.base_storage
      else:
        (loader, id, save) = self.computer.out[self.computer_key]
        ext = loader.extension
        def compute_param_str_old(ressource):
          # logger.debug("Ressource is {}".format(ressource.handle.name))
          name = ressource.handle.name
          if ressource.computer is None:
            return ressource.handle.id
          static_params = {key:p for key, p in ressource.computer.params.items() if not isinstance(p, RessourceHandle)}
          ressourceParams = {key:p for key, p in ressource.computer.params.items() if isinstance(p, RessourceHandle)}
          return "{}_{}/{}".format(
            name, 
            toolbox.clean_filename(str(static_params)), 
            "/".join([compute_param_str_old(ressource.handle.manager.d[p.id]) for p in ressourceParams.values()]))
        def compute_param_str(ressource):
          # logger.debug("Ressource is {}".format(ressource.handle.name))
          name = ressource.handle.name
          if ressource.computer is None:
            return ressource.handle.id
          
          def flatten_dict_of_list(d):
            res= {}
            for key, v in d.items():
              if isinstance(v, list):
                nd= {("{}_{}".format(key, i)):vi for i,vi in enumerate(v)}
              else:
                nd = {key:v}
              res = dict(res, **nd)
            return res
          tmp_params = flatten_dict_of_list(ressource.computer.params)
          static_params = {key:p for key, p in tmp_params.items() if not isinstance(p, RessourceHandle)}
          ressourceParams = {key:p for key, p in tmp_params.items() if isinstance(p, RessourceHandle)}

          # old_core_path = "{}_{}/{}".format(
          #   name, 
          #   str(static_params), 
          #   ".".join([str(ressource.handle.manager.d[p.id].old_disk_path) for p in ressourceParams.values()]))
          
          # old_core_path = "{}_{}/{}".format(
          #   name, 
          #   str(dict(sorted(static_params.items()))), 
          #   ".".join([str(ressource.handle.manager.d[p.id].old_disk_path) for p in dict(sorted(ressourceParams.items())).values()]))
          old_core_path = None

          new_core_path = "{}_{}/{}".format(
            name, 
            str(dict(sorted(static_params.items()))), 
            ".".join([str(ressource.handle.manager.d[p.id].get_disk_path()) for p in dict(sorted(ressourceParams.items())).values()]))
          
          return (new_core_path, old_core_path)
        core_path, old_core_path = compute_param_str(self)
        if self.handle.name == "":
          logger.warning("name is empty")
        new_disk_path = self.base_storage / pathlib.Path(self.handle.name) / pathlib.Path("hash") / pathlib.Path("{}{}".format(hashlib.md5(core_path.encode()).hexdigest(), ext))
        if old_core_path:
          old_disk_path = self.base_storage / pathlib.Path("hash") / pathlib.Path("{}{}".format(hashlib.md5(old_core_path.encode()).hexdigest(), ext))
          if (old_disk_path != new_disk_path) and old_disk_path.exists():
            new_disk_path.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(old_disk_path), str(new_disk_path))
        else:
          old_disk_path=""
        self.disk_path = new_disk_path
        self.old_disk_path= old_disk_path
      # if "Disk" in self.storage_locations:
      #   logger.debug("Ressource {} is on Disk".format(self.handle.name))
      return self.disk_path


    def memory_to_disk(self): #Here to add json file if wanted
      if "Disk" in self.storage_locations:
        logger.debug("Unnecessary Memory to disk called")
        return 
      logger.info("Ressource {} is being saved on disk".format(self.get_disk_path()))
      if not "Memory" in self.storage_locations:
        raise BaseException("Bug in Ressource Manager. Ressource should have been loaded.")
      if self.value is None:
        logger.warning("Value is supposed to be loaded but has Value None. Strange...")
      self.get_disk_path().parent.mkdir(parents=True, exist_ok=True)
      # logger.debug("Folder {} created".format(self.get_disk_path().parent))
      self.loader.save(self.get_disk_path(), self.value)
      # logger.info("Writing to disk")
      self.storage_locations.append("Disk")
      # if "Disk" in self.storage_locations:
      #   logger.debug("Ressource {} is on Disk".format(self.handle.name))

    def disk_to_memory(self):
      if "Memory" in self.storage_locations:
        logger.debug("Unnecessary disk to memory called")
        return 
      logger.info("Ressource {} is being loaded into memory".format(self.get_disk_path()))
      if not "Disk" in self.storage_locations:
        raise BaseException("Bug in Ressource Manager. Ressource should be on disk at path {} but is not.".format(self.get_disk_path()))
      if not self.get_disk_path().exists():
        # logger.error("Bug in Ressource Manager. Incoherent location state.")
        raise BaseException("Bug in Ressource Manager. Incoherent location state. Ressource should be on disk at path {} but is not.".format(self.get_disk_path()))
      self.value = self.loader.load(self.get_disk_path())
      self.storage_locations.append("Memory")
      # if "Disk" in self.storage_locations:
      #   logger.debug("Ressource {} is on Disk".format(self.handle.name))


    def remove_from_memory(self):
      if "Memory" in self.storage_locations:
        logger.info("Ressource {} is being removed from memory".format(self.get_disk_path()))
        self.value = None
        self.storage_locations.remove("Memory")

    def remove_from_disk(self):
      if "Disk" in self.storage_locations:
        logger.info("Ressource {} is being removed from disk".format(self.get_disk_path()))
        self.get_disk_path().unlink()
        self.storage_locations.remove("Disk")

  class Computer:
    func: Callable
    params: Dict[str, Any]
    out: Dict[Any, Tuple[RessourceLoader, Manager.IdType, bool | None]]

    def __init__(self, func, params, out: Dict[Any, Tuple[RessourceLoader, Manager.IdType, bool | None]], error_method="propagate"):
      self.func = func
      self.params = params
      self.out = out
      self.error_method = error_method


class RessourceHandle:
  id: Manager.IdType   # A (short) unique id (probably of type string). Only used because may be simpler to save than path.
  name: str #Used for nice printing
  manager : Manager

  def __init__(self, manager, id, name):
    self.id = id
    self.manager = manager
    self.name = name

  def get_result(self) -> Any: 
    return self.manager.get_result(self.id)

  def get_disk_path(self) -> pathlib.Path:
    return self.manager.get_disk_path(self.id)
  
  def is_in_memory(self) -> bool: 
    return self.manager.is_in_memory(self.id)
  
  def is_saved_on_disk(self) -> bool:
    return self.manager.is_saved_on_disk(self.id)
  
  def is_stored(self) -> bool:
    return self.manager.is_stored(self.id)
  
  def is_saved_on_compute(self) -> bool:
    r = self.manager.d[self.id]
    if r.computer:
      if r.computer.out[r.computer_key][2]:
        return True
    return False

  def save(self): 
    self.manager.save_on_disk(self.id)

  def unload(self):
    self.manager.unload(self.id)

  def invalidate_all(self):
    self.manager.invalidate_all(self.id)

  def __str__(self):
    if self.is_in_memory():
      res = self.get_result()
      if isinstance(res, float):
        return str(res)
      if isinstance(res, int):
        return str(res)
      if isinstance(res, np.int64):
        return str(res)
      if isinstance(res, toolbox.Video):
        return str(res)
      # print(type(res))
      if hasattr(res, "shape"):
          if res.size == 0:
            return "Rec(None)"
          if res.size < 5:
            return "Rec({})".format(res)
          return "Rec(shape{})".format(res.shape)
      elif hasattr(res, "__len__"):
         if len(res) < 5:
            return "Rec({})".format(res)
         return "Rec({}_of_{}_elements)".format(type(res), len(res))
      else:
          return "Rec({})".format(str(res)[0:50])
    elif self.is_stored():
      return "StoredRec({})".format(self.manager.d[self.id].storage_locations)
    else:
      return "UncomputedRec"


def get(x):
  if isinstance(x, RessourceHandle):
    return x.get_result()
  else:
    return x