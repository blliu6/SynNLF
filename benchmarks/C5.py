from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
from benchmarks.Exampler_V import get_example_by_name


def main():
    # SQUARE MUL SKIP
    activations = ['SQUARE']
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('C5')
    start = timeit.default_timer()

    example.D_zones.r = pow(50, 2)

    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "BATCH_SIZE": 300,
        "LEARNING_RATE": 0.01,
        "LOSS_WEIGHT": (1.0, 1.0),
        "SPLIT_D": False,
        'BIAS': False,
        'DEG': [4, 2, 2],
        'max_iter': 20,
        'counter_nums': 130,
        'ellipsoid': True,
    }
    Config = CegisConfig(**opts)
    c = Cegis(Config)
    c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))
    if example.n == 2:
        from plots.plot import Draw
        draw = Draw(c.ex, c.Learner.net.get_lyapunov())
        draw.plot_benchmark_2d()


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
