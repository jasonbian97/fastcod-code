import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange,reduce,repeat
import pandas as pd
import os
import sys

file_dir = os.path.dirname(__file__)

class FsLut(object):
    def __init__(self):
        with open(f'{file_dir}/FreeSurferColorLUT.txt') as f:
            lines = [line.rstrip() for line in f]
            lines = [line for line in lines if not line.startswith("#")]
            lines = [line for line in lines if len(line)>1]
            lines = [line.split() for line in lines]

        lut = pd.DataFrame(data=lines,columns=["No","Label","R","G","B","A"])
        lut[["No","R","G","B","A"]] = lut[["No","R","G","B","A"]].apply(pd.to_numeric)
        self.lut = lut

    def get_rgb(self, no):
        return self.lut.loc[self.lut.No==no,"R":"B"].to_numpy().astype('int')

    def parc2rbg(self,parc):
        """map each entry of parc to rbg arrays accoriding to lut.
        parc : parc is a 2D image with integer values at each entry
        return: a 3d tensor, with the last channel being rbg.
        Usecase:
            lut = FsLut()
            a = np.array([[1,2], [3,4]])
            out = lut.parc2rbg(a)
            plt.imshow(out); plt.show()
        """
        h, w = parc.shape
        a_fl = parc.flatten()
        a_vals = np.unique(parc)
        N = a_vals.shape[0]
        rgbimg = np.zeros((N, 3)).astype('int')
        for val in a_vals:
            rgbimg[a_fl == val] = self.get_rgb(val)
        rgbimg = rearrange(rgbimg, "(h w) c -> h w c", c=3, h=h, w=w)
        return rgbimg

    def parc2rbg_3d(self, parc):
        """similar to parc2rbg.
        parc : parc is a 3D image with integer values at each entry
        return: a 4d tensor, with the last channel being rbg.
        Usecase:
            lut = FsLut()
            a = np.random.randint(0,4,(2,2,2))
            out = lut.parc2rbg_3d(a)
            plt.imshow(out[:,:,0]); plt.show()
        """
        h, w, d = parc.shape
        a_fl = parc.flatten()
        a_vals = np.unique(parc)
        N = a_fl.shape[0]
        rgbimg = np.zeros((N, 3)).astype('int')
        for val in a_vals:
            rgbimg[a_fl == val] = self.get_rgb(val)
        rgbimg = rearrange(rgbimg, "(h w d) c -> h w d c", c=3, h=h, w=w)
        return rgbimg


