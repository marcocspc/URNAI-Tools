import os.path, pkgutil
import importlib
import inspect
from .error import ClassNotFoundError

def get_modules(top_pkg_str, pkg_str):
    pkg_str_path = os.path.dirname(importlib.import_module(top_pkg_str + "." + pkg_str).__file__)
    
    return [name for _, name, _ in pkgutil.walk_packages([pkg_str_path])
             if not name.endswith('__')]

def get_classes(top_pkg_str, pkg_str):
    class_dict = {}

    for module_str in get_modules(top_pkg_str, pkg_str):
        try:
            aux_str = top_pkg_str + "." + pkg_str + "." + module_str
            aux_mod = importlib.import_module(aux_str)
            md = aux_mod.__dict__
            for key in md:
                if isinstance(md[key], type): 
                    if aux_str in str(md[key]):
                        class_dict[key] = module_str 
        except ModuleNotFoundError:
            pass
        except NameError:
            pass

    return class_dict 

def get_cls(top_pkg_str, pkg_str, classname):
    class_dict = get_classes(top_pkg_str, pkg_str)

    if classname in class_dict.keys():
        cls_import_lst = [top_pkg_str,pkg_str,class_dict[classname]]
        mod = importlib.import_module('.'.join(cls_import_lst))
        cls = getattr(mod, classname)
        return cls
    else:
        raise ClassNotFoundError("Class " + classname + " was not found in " + top_pkg_str + "." + pkg_str)
