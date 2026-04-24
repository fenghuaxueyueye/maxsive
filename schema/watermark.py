from abc import ABC, abstractmethod

class watermark(ABC):
    
    @abstractmethod
    def watermark_injection(self):
        pass

    @abstractmethod
    def detection(self):
        pass
