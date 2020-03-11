from abc import ABC, abstractmethod

class Savable(ABC):

    '''
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename to save on disk.
    '''
    
    '''
    This method should:
    1) Get a save path as parameter,
    2) Use its default filename to save itself as a file inside the given path.
    '''
    def step(self, savepath):
        raise NotImplementedError 


    '''
    This method should:
    1) Get a load path as parameter,
    2) Use its default filename to load itself from inside the given path.
    '''
    def play(self, loadpath):
        raise NotImplementedError 
