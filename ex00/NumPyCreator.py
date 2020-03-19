import numpy as np


class NumPyCreator():
    @staticmethod
    def from_list(lst, dtype=None):
        return np.array(list(lst), dtype=dtype)

    @staticmethod
    def from_tuple(tpl, dtype=None):
        return np.array(tuple(tpl), dtype=dtype)

    @staticmethod
    def from_iterable(itr, dtype=None):
        return np.array(list(itr), dtype=dtype)

    @staticmethod
    def from_shape(shape, value=0, dtype=None):
        return np.full(shape, value, dtype=dtype)

    @staticmethod
    def random(shape):
        return np.random.random_sample(shape)

    @staticmethod
    def identity(n, dtype=None):
        return np.eye(n, dtype=dtype)


def print_repr(*args):
    print(repr(*args))


if __name__ == "__main__":
    # From list
    lst = [[1, 2, 3], [6, 3, 4]]
    print_repr(NumPyCreator.from_list(lst))

    # From List with type
    str_lst = ["10", "11", "12"]
    print_repr(NumPyCreator.from_list(str_lst, dtype=np.float))

    # From tuple
    tpl = ("a", "b", "c")
    print_repr(NumPyCreator.from_tuple(tpl))

    # From string iterable
    string = "Toto"
    print_repr(NumPyCreator.from_iterable(iter(string)))
    # From list iterable
    print_repr(NumPyCreator.from_iterable(iter(lst)))
    # From range
    itr = range(5)
    print_repr(NumPyCreator.from_iterable(itr))

    # From shape
    print_repr(NumPyCreator.from_shape((3, 5)))
    print_repr(NumPyCreator.from_shape((2, 2), value=42, dtype="<U2"))

    # Random
    print_repr(NumPyCreator.random((3, 5)))

    # Identity
    print_repr(NumPyCreator.identity(4))
