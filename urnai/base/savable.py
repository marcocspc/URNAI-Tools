from abc import ABC, abstractmethod

class Savable(ABC):

    '''
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename to save on disk.
    '''

    def __init__(self):
        '''
        This init method is here
        to instantiate self.pickle_obj
        '''
        self.pickle_obj = []

    def get_default_filename(self):
        '''
        This method returns the default
        file name that should be used while
        persisting the object.
        '''
        return self.__class__.__name__ + "_"

    def save_pickle(self, persist_path):
        '''
        This method saves a list instance
        saved by pickle.

        self.pickle_obj should be populated in __init__ method.
        '''
        with open(self.get_full_persistance_pickle_path(), "wb") as pickle_out: 
            pickle.dump(self.pickle_obj, pickle_out)
    
    def save_extra(self, persist_path):
        '''
        This method should be implemented when
        some extra persistence is to be saved.
        '''
        pass

    def load_pickle(self, persist_path):
        '''
        This method loads a list instance
        saved by pickle.
        '''
            #Check if pickle file exists
            exists_pickle = os.path.isfile(self.get_full_persistance_pickle_path(persist_path))
            #If yes, load it
            if exists_pickle:
                with open(self.get_full_persistance_pickle_path(persist_path), "wb") as pickle_in: 
                    self.pickle_obj = pickle.load(pickle_in)
            else:
                #Else, raise exception
                raise FileNotFoundError(self.get_full_persistance_tensorflow_path(persist_path) + " was not found.")

    def load_extra(self, persist_path):
        '''
        This method should be implemented when
        some extra persistence is to be loaded. 

        Also it should put self.pickle_obj data
        into the right attributes.
        i.e.:
        self.a = pickle_obj[0]
        self.b = pickle_obj[1]
        '''
        pass

    def get_full_persistance_pickle_path(self, persist_path):
        '''
        This method returns the default persistance pickle path. 
        '''
        return persist_path + self.file_name + os.path.sep + self.get_default_filename() + self.file_name + ".pkl"

    def get_full_persistance_tensorflow_path(self, persist_path):
        '''
        This method returns the default persistance tensorflow path. 
        '''
        return persist_path + os.path.sep + self.get_default_filename + "tensorflow_" + self.file_name

    def save(self, savepath):
        '''
        This method saves self.pickle_obj
        and extra stuff needed
        '''
        self.save_pickle(savepath)
        self.save_extra(savepath)

    def load(self, loadpath):
        '''
        This method loads self.pickle_obj
        and extra stuff needed
        '''
        self.load_pickle(loadpath)
        self.load_extra(loadpath)
