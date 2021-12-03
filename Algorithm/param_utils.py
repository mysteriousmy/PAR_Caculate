import math
import laspy
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename


class nxParameter(object):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y


class nxParameter_xyz(nxParameter):
    def __init__(self, data_x, data_y, gridsize=0, secgrid=0):
        super().__init__(data_x, data_y)
        self.cells = data_x
        self.cells_y = data_y
        self.arr = (max(self.cells) - min(self.cells)) / gridsize
        self.arr_y = (max(self.cells_y) - min(self.cells_y)) / gridsize
        self.gridsize = gridsize
        self.secgrid = secgrid

    def newarr(self):
        self.arr = (max(self.cells) - min(self.cells)) / self.gridsize
        self.arr_y = (max(self.cells_y) - min(self.cells_y)) / self.gridsize


class nxParameter_ball(nxParameter):
    def __init__(self, data_x, data_y, theta_grid, phi_grid, data_r, r_grid=0, max_t=90, max_p=360):
        super().__init__(data_x, data_y)
        self.data_r = data_r
        self.r_grid = r_grid
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
        self.arr_t = max_t / theta_grid
        self.arr_p = max_p / phi_grid
        self.arr_r = max(data_r) / r_grid


class lasfileAction(object):
    def __init__(self, filepath=""):
        self.filepath = filepath
        self.lasfile = object

    def get_filedata(self):
        try:
            self.lasfile_data = laspy.read(self.filepath)
            return self.lasfile_data
        except:
            print("Error: file is not found or file defined")
            sys.exit(2)

    def get_xyz(self):
        if self.lasfile_data is not None:
            self.x, self.y, self.z = self.lasfile_data.x, self.lasfile_data.y, self.lasfile_data.z
            return True
        return False

    def getLasfile(self):
        while len(self.filepath) == 0:
            Tk().withdraw()
            self.filepath = askopenfilename()
        self.lasfile = self.get_filedata()
        return self.lasfile


def atan2(y, x):
    # radiant
    if x > 0 and y > 0:  # 1
        return math.pi/2 - math.atan(y/x)
    elif x < 0 and y > 0:  # 2
        return math.pi*3/2 - math.atan(y/x)  # 减去负值
    elif x < 0 and y < 0:
        return math.pi*3/2 - math.atan(y/x)  # 加正值
    elif x > 0 and y < 0:
        return math.pi/2 - math.atan(y/x)
    elif x == 0 and y > 0:
        return 0
    elif x == 0 and y < 0:
        return math.pi
    elif y == 0 and x > 0:
        return math.pi/2
    elif y == 0 and x < 0:
        return math.pi*3/2
    elif x == 0 and y == 0:
        return 0
