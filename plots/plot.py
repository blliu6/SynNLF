from benchmarks.Exampler_V import Example, Zone
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import sympy as sp


class Draw:
    def __init__(self, ex: Example, V):
        self.ex = ex
        self.V = V

    def plot_benchmark_2d(self):
        ex = self.ex
        V = self.V
        fig = plt.figure()
        ax = plt.gca()
        zone = self.draw_zone(ex.D_zones, 'black', 'ROA')

        r = np.sqrt(ex.D_zones.r)
        self.plot_contour(V, r)
        # self.plot_vector_field(r, ex.f)
        ax.add_patch(zone)
        ax.set_xlim(-2 * r, 2 * r)
        ax.set_ylim(-2 * r, 2 * r)
        ax.set_aspect(1)
        plt.savefig(f'img/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_benchmark_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        r = np.sqrt(self.ex.D_zones.r)
        self.plot_barrier_3d(ax, r, self.V)

        domain = self.draw_zone(self.ex.D_zones, color='g', label='domain')
        ax.add_patch(domain)
        art3d.pathpatch_2d_to_3d(domain, z=0, zdir="z")
        plt.savefig(f'img/{self.ex.name}_3d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_barrier_3d(self, ax, r, v):
        r = 2 * r
        x = np.linspace(-r, r, 1000)
        y = np.linspace(-r, r, 1000)
        X, Y = np.meshgrid(x, y)
        s_x = sp.symbols(['x1', 'x2'])
        lambda_b = sp.lambdify(s_x, v, 'numpy')
        plot_b = lambda_b(X, Y)
        ax.plot_surface(X, Y, plot_b, rstride=5, cstride=5, alpha=0.5, cmap='cool')

    def draw_zone(self, zone: Zone, color, label, fill=False):
        if zone.shape == 'ball':
            circle = Circle(zone.center, np.sqrt(zone.r), color=color, label=label, fill=fill, linewidth=1.5)
            return circle
        else:
            w = zone.up[0] - zone.low[0]
            h = zone.up[1] - zone.low[1]
            box = Rectangle(zone.low, w, h, color=color, label=label, fill=fill, linewidth=1.5)
            return box

    def plot_contour(self, hx, r):
        r = 2 * r
        x = np.linspace(-r, r, 1000)
        y = np.linspace(-r, r, 1000)

        X, Y = np.meshgrid(x, y)

        s_x = sp.symbols(['x1', 'x2'])
        fun_hx = sp.lambdify(s_x, hx, 'numpy')
        value = fun_hx(X, Y)
        plt.contourf(X, Y, value, alpha=0.5, cmap=plt.cm.jet)

    def plot_vector_field(self, r, f, color='white'):
        r = 2 * r
        xv = np.linspace(-r, r, 100)
        yv = np.linspace(-r, r, 100)
        Xd, Yd = np.meshgrid(xv, yv)

        DX, DY = f[0]([Xd, Yd]), f[1]([Xd, Yd])
        DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
        DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)

        plt.streamplot(Xd, Yd, DX, DY, linewidth=0.3,
                       density=0.8, arrowstyle='-|>', arrowsize=1, color=color)
