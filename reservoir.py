from __future__ import division

import numpy as np
import seaborn as sns
from itertools import product


class Well(object):
    def __init__(self, start_time, end_time, flow_rate, min_pressure=500):
        self.start_time = start_time
        self.end_time = end_time
        self.flow_rate = flow_rate
        self.min_pressure = min_pressure
        self.well_index = None

    def establish(self, kx, ky, vol):
        self.well_index = 0.006328 * np.sqrt(kx * ky) * vol

    def get_flow_rate(self, pressure, time):
        if self.start_time <= time < self.end_time:
            if pressure >= self.min_pressure:
                return self.flow_rate
            else:
                return -self.well_index * (pressure - self.min_pressure)
        return 0


class Grid(object):
    def __init__(self, width, length, height):
        self.width = width
        self.length = length
        self.height = height
        self.locations = list(product(*(range(width), range(length), range(height))))
        self.shape = (width, length, height)
        self.size = width * height * length
        self.dimensions = np.zeros(shape=(width, length, height, 3))

    def _get_adjacents(self, x, y, z, d=1):
        x_range = range(max(0, x - d), min(self.shape[0], x + d))
        y_range = range(max(0, y - d), min(self.shape[1], y + d))
        z_range = range(max(0, z - d), min(self.shape[2], z + d))
        adjacents = list(product(x_range, y_range, z_range))
        adjacents.remove((x, y, z))
        return adjacents

    def _get_five_point(self, x, y, z):
        adjacents = []
        for i, v in enumerate((x, y, z)):
            if v - 1 >= 0:
                loc = [x, y, z]
                loc[i] = v - 1
                adjacents.append(loc)
            if v + 1 < self.shape[i]:
                loc = [x, y, z]
                loc[i] = v + 1
                adjacents.append(loc)
        return adjacents

    def _get_distances(self, a, b):
        ax, ay, az = a
        bx, by, bz = b
        a_dims = self.dimensions[ax, ay, az]
        b_dims = self.dimensions[bx, by, bz]
        dis = [0, 0, 0]
        for i in range(len(dis)):
            di = a[i] - b[i]
            if di != 0:
                dis[i] = (a_dims[i]/2 + b_dims[i]/2 + abs(di)) * (di / abs(di))
        # print 'distances:', a, b, '->', dis
        return dis

    def node_size(self, x, y, z):
        w, l, h = self.dimensions[x, y, z]
        return w*l*h


class Reservoir(Grid):
    def __init__(self, width=4, length=4, height=1):
        super(Reservoir, self).__init__(width, length, height)
        self.wells = {}
        self.exchanges = np.zeros(shape=(width, length, height, 4))
        self.reservoir_time = 0.00
        # Rock Properties
        self.permeabilities = np.zeros(shape=(width, length, height, 3))
        self.porosities = np.zeros(shape=self.shape)
        # Fluid Properties
        self.viscosity = 0.00
        self.density = 0.00
        self.pressures = np.zeros(shape=self.shape)

    def set_fluid(self, density, viscosity, pressure=None):
        self.density = density
        self.viscosity = viscosity
        if pressure is not None:
            self.pressures.fill(pressure)

    def set_media(self, dimension=None, permeability=None, porosity=None):
        if dimension is not None:
            self.dimensions[:, :, :] = dimension
        if permeability is not None:
            self.permeabilities[:, :, :] = permeability
        if porosity is not None:
            self.porosities.fill(porosity)

    def create_well(self, x, y, z, start_time, end_time, flow, pressure):
        well = Well(start_time, end_time, flow_rate=flow, min_pressure=pressure)
        index = self.locations.index((x, y, z))
        permeability = self.permeabilities[x, y, z]
        well.establish(permeability[0], permeability[1], self.node_size(x, y, z))
        self.wells[index] = well

    def well_flow_rate(self, index, pressure):
        if index in self.wells.keys():
            return self.wells[index].get_flow_rate(pressure, self.reservoir_time)
        return 0

    def time_step(self, step):
        self.reservoir_time += step
        a_mat = np.zeros(shape=(self.size, self.size))
        r_mat = np.zeros(shape=(self.size, 1))
        for n, location in enumerate(self.locations):
            lx, ly, lz = location
            adjacents = self._get_five_point(lx, ly, lz)
            e_val = 0  # total node flux
            for adjacent in adjacents:
                adj_x, adj_y, adj_z = adjacent  # adjacent node grid location
                gdx, gdy, gdz = self._get_distances(location, adjacent)  # distance to adjacent node
                kx, ky, kz = self.permeabilities[adj_x, adj_y, adj_z]  # adjacent node permeabilities
                adx, ady, adz = self.dimensions[adj_x, adj_y, adj_z]  # adjacent node dimensions
                if gdx != 0:  # distance along x axis
                    adj_flux = 0.006328 * kx * ady * adz / adx
                elif gdy != 0:  # distance along y axis
                    adj_flux = 0.006328 * ky * adx * adz / ady
                else:  # distance along z axis
                    adj_flux = 0.006328 * kz * adx * ady / adz
                e_val += adj_flux
                adj_index = self.locations.index((adj_x, adj_y, adj_z))
                a_mat[n, adj_index] = adj_flux
            ndx, ndy, ndz = self.dimensions[lx, ly, lz]  # node dimensions
            volume = ndx * ndy * ndz * self.porosities[lx, ly, lz]  # node effective volume
            a_mat[n, n] = -e_val - volume / step  # adjusted flux
            n_pressure = self.pressures[lx, ly, lz]  # node pressure
            r_mat[n, 0] = -n_pressure * volume / step - self.well_flow_rate(n, n_pressure)  #
        a_mat = np.linalg.inv(a_mat)
        p_mat = np.dot(a_mat, r_mat)
        self.pressures = p_mat[:, 0].reshape(self.shape)

    def run_steps(self, steps, step):
        for _ in range(steps):
            self.time_step(step)


def trial():
    res = Reservoir(width=5, length=7, height=1)
    res.set_fluid(1, 0.25, 4000)
    res.set_media((2, 2, 1), (1, 1, 0.4), 0.08)
    res.create_well(1, 2, 0, 0, 100, flow=-5.615*11, pressure=500)
    res.create_well(3, 4, 0, 0, 100, flow=5.615*10, pressure=500)
    res.run_steps(1000, 0.1)  # 2231
    print res.pressures.mean()

    sns.heatmap(res.pressures[:, :, 0])
    sns.plt.show()

    return res

trial()
