import numpy as np


class Zone:
    def __init__(self, shape: str, low=None, up=None, center=None, r=None):
        self.shape = shape
        if shape == 'ball':
            self.center = np.array(center, dtype=np.float32)
            self.r = r  # radius squared
        elif shape == 'box':
            self.low = np.array(low, dtype=np.float32)
            self.up = np.array(up, dtype=np.float32)
            self.center = (self.low + self.up) / 2
        else:
            raise ValueError(f'There is no area of such shape!')


class Example:
    def __init__(self, n, D_zones, f, name):
        self.n = n  # number of variables
        self.D_zones = D_zones  # local condition
        self.f = f  # differential equation
        self.name = name  # name or identifier


examples = {
    0: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1000**2),
        f=[
            lambda x: -x[0] + x[0] * x[1],
            lambda x: -x[1]
        ],
        name='C1'
    ),
    1: Example(
        n=2,
        D_zones=Zone('ball', center=[0]*2, r=1000**2),
        f=[
            lambda x: -x[0] + x[0] * x[1],
            lambda x: -x[1]
        ],
        name='C2'

    ),
    2: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=1000**2),
        f=[
            lambda x: -x[0],
            lambda x: -2 * x[1] + 0.1 * x[0] * x[1] ** 2 + x[2],
            lambda x: -x[2] - 1.5 * x[1]
        ],
        name='C3'
    ),
    3: Example(
        n=4,
        D_zones=Zone(shape='box', low=[-2, -2, -2], up=[2, 2, 2]),
        f=[lambda x: x[0],
           lambda x: x[1],
           lambda x: x[2],
           lambda x: - 3980 * x[3] - 4180 * x[2] - 2400 * x[1] - 576 * x[0]
           ],
        name='hi_ord_4'
    ),
    4: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=2000**2),
        f=[
            lambda x: -3 * x[0] - 0.1 * x[0] * x[1] ** 3,
            lambda x: -x[1] + x[2],
            lambda x: -x[2]
        ],
        name='C4'
    ),
    5: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1500**2),
        f=[
            lambda x: -x[0],
            lambda x: -x[1]
        ],
        name='C5'
    ),
    6: Example(
        n=2,
        D_zones=Zone(shape='box', low=[-1.5, -1.5], up=[1.5, 1.5]),
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[0] * x[1] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c3'
    ),
    7: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=1000**2),
        f=[
            lambda x: -x[0] ** 3 - x[0] * x[2] ** 2,
            lambda x: -x[1] - x[0] ** 2 * x[1],
            lambda x: -x[2] + 3 * x[0] ** 2 * x[2] - 3 * x[2]
        ],
        name='C6'
    ),
    8: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1300**2),
        f=[
            lambda x: -x[0] ** 3 + x[1],
            lambda x: -x[0] - x[1],
        ],
        name='C7'
    ),
    9: Example(
        n=2,
        D_zones=Zone(shape='box', low=[-1, -1], up=[1, 1]),
        f=[lambda x: -2 * x[0] + x[0] * x[0] + x[1],
           lambda x: x[0] - 2 * x[1] + x[1] * x[1]
           ],
        name='emsoft_c6'
    ),
    10: Example(
        n=2,
        D_zones=Zone(shape='box', low=[-2, -2], up=[2, 2]),
        f=[lambda x: -x[0] + x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c7'
    ),
    11: Example(
        n=2,
        D_zones=Zone(shape='box', low=[-2, -2], up=[2, 2]),
        f=[lambda x: -x[0] + 2 * x[0] * x[0] * x[1],
           lambda x: -x[1]
           ],
        name='emsoft_c8'
    ),
    12: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1500**2),
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
        ],
        name='C8'

    ),
    13: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1500**2),
        f=[
            lambda x: -x[0] - 1.5 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 0.5 * x[0] ** 3 * x[1] ** 2
        ],
        name='C9'
    ),
    14: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=10**2),
        f=[
            lambda x: x[1],
            lambda x: -x[0] - x[1] + 1 / 3.0 * x[0] ** 3
        ],
        name='barr_2'
    ),
    15: Example(
        n=4,
        D_zones=Zone(shape='box', low=[-10, -10, -10, -10], up=[10, 10, 10, 10]),
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
            lambda x: -x[2] ** 3 + x[3],
            lambda x: -x[2] - x[3],
        ],
        name='poly5'
    ),
    16: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=10 ** 2),
        f=[
            lambda x: -x[0] ** 3 - x[0] * x[2] ** 2,
            lambda x: -x[1] - x[0] ** 2 * x[1],
            lambda x: -x[2] + 3 * x[0] ** 2 * x[2] - 3 * x[2]
        ],
        name='poly_1'
    ),
    17: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
        f=[
            lambda x: -2*x[0] + x[0] * x[1],
            lambda x: -x[1] + x[0] * x[1],
        ],
        name='C11'
    ),
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError(f'The example {name} was not found.')


if __name__ == '__main__':
    print(get_example_by_id(1))
