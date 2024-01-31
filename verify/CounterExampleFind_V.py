from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import sympy as sp
from benchmarks.Exampler_V import Example, Zone, get_example_by_id
import torch


def split_bounds(bounds, n):
    """
    Divide an n-dimensional cuboid into 2^n small cuboids, and output the upper and lower bounds of each small cuboid.

    parameter: bounds: An array of shape (n, 2), representing the upper and lower bounds of each dimension of an
    n-dimensional cuboid.

    return:
        An array with a shape of (2^n, n, 2), representing the upper and lower bounds of the divided 2^n small cuboids.
    """

    if n == bounds.shape[0]:
        return bounds.reshape((-1, *bounds.shape))
    else:
        # Take the middle position of the upper and lower bounds of the current dimension as the split point,
        # and divide the cuboid into two small cuboids on the left and right.
        mid = (bounds[n, 0] + bounds[n, 1]) / 2
        left_bounds = bounds.copy()
        left_bounds[n, 1] = mid
        right_bounds = bounds.copy()
        right_bounds[n, 0] = mid
        # Recursively divide the left and right small cuboids.
        left_subbounds = split_bounds(left_bounds, n + 1)
        right_subbounds = split_bounds(right_bounds, n + 1)
        # Merge the upper and lower bounds of the left and right small cuboids into an array.
        subbounds = np.concatenate([left_subbounds, right_subbounds])
        return subbounds


class CounterExampleFinder:
    def __init__(self, example: Example, config):
        self.n = example.n
        self.inv = example.D_zones
        self.f = example.f
        self.eps = config.eps
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.config = config
        self.nums = config.counter_nums

    def find_counterexamples(self, V):
        print('_______________________________________')
        res = []

        expr1 = V
        vis1, x1 = self.get_extremum_scipy(self.inv, expr1)
        if vis1:
            x1 = self.generate_sample(x1, expr1)
            res.extend(x1)

        x = self.x
        expr2 = -sum([sp.diff(V, x[i]) * self.f[i](x) for i in range(self.n)])
        if self.config.SPLIT_D:
            bounds = self.split_zone(self.inv)
        else:
            bounds = [self.inv]

        for bound in bounds:
            vis2, x2 = self.get_extremum_scipy(bound, expr2)
            if vis2:
                print(vis2)
                x2 = self.generate_sample(x2, expr2)
                res.extend(x2)

        return res

    def generate_sample(self, x, expr):
        eps = self.eps
        nums = self.nums
        result = [x]
        for i in range(nums - 1):
            rd = (np.random.random(self.n) - 0.5) * eps
            rd = rd + x
            result.append(rd)
        fun = sp.lambdify(self.x, expr)
        result = [e for e in result if fun(*e) < 0]
        result = [e for e in result if self.check(e)]
        return result

    def check(self, x):
        zone = self.inv

        if zone.shape == 'ball':
            return sum((zone.center - x) ** 2) <= zone.r
        else:
            vis = True
            low, up = zone.low, zone.up
            for i in self.n:
                vis = vis and (low[i] <= x[i] <= up[i])
            return vis

    def get_extremum_scipy(self, zone: Zone, expr):
        x_ = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        opt = sp.lambdify(x_, expr)
        result = None
        if zone.shape == 'box':
            bound = tuple(zip(zone.low, zone.up))
            res = minimize(lambda x: opt(*x), np.zeros(self.n), bounds=bound)
            if res.fun < 0 and res.success:
                # print(f'Counterexample found:{res.x}')
                result = res.x
        elif zone.shape == 'ball':
            poly = zone.r
            for i in range(self.n):
                poly = poly - (x_[i] - zone.center[i]) ** 2
            poly_fun = sp.lambdify(x_, poly)
            con = {'type': 'ineq', 'fun': lambda x: poly_fun(*x)}
            res = minimize(lambda x: opt(*x), np.zeros(self.n), constraints=con)
            if res.fun < 0 and res.success:
                # print(f'Counterexample found:{res.x}')
                result = res.x
        if result is None:
            return False, []
        else:
            return True, result

    def split_zone(self, zone: Zone):
        bound = list(zip(zone.low, zone.up))
        bounds = split_bounds(np.array(bound), 0)
        ans = [Zone(shape='box', low=e.T[0], up=e.T[1]) for e in bounds]
        return ans


if __name__ == '__main__':
    """

    test code!!

    """
    ex = get_example_by_id(2)
