from abc import ABC, abstractmethod


class SparkDistributable(ABC):

    @abstractmethod
    def init(self):
        """
        initialize the object to be used on in a spark worker
        """
        pass

    @abstractmethod
    def deinit(self):
        """
        deinitialize the object, closing resources (e.g. file handles)
        """
        pass

    @abstractmethod
    def to_spark(self):
        """
        send the obj to the spark cluster to be used on spark workers
        """
        pass
