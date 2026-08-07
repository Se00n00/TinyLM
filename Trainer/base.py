from abc import ABC, abstractmethod

class Trainer(ABC):
    
    @abstractmethod
    def get_batch(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

