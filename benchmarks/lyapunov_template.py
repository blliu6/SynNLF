from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
from benchmarks.Exampler_V import get_example_by_name
from plots.plot_lyap import plot_benchmark_2d


# barr_2,emsoft_c3,emsoft_c6,emsoft_c7,emsoft_c8,nonpoly0,nonpoly2,nonpoly1,nonpoly3

def main():
    activations = ['SKIP']
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('nonpoly1')
    start = timeit.default_timer()
    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "BATCH_SIZE": 500,
        "LEARNING_RATE": 0.1,
        "LOSS_WEIGHT": (1.0, 1.0),
        "SPLIT_D": True,
        'BIAS': False,
        'DEG': [4, 4],
        'max_iter': 10,
        'counter_nums': 50,
        'Global_Optimization': False,
        'ellipsoid': True
    }
    Config = CegisConfig(**opts)
    c = Cegis(Config)
    c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))
    # if example.n == 2:
    #     plot_benchmark_2d(c.ex, c.Learner.net.get_lyapunov())


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
