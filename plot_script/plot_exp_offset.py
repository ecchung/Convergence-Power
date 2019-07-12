import numpy as np
import matplotlib.pyplot as plt

pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
def bottom_offset(self, bboxes, bboxes2):
    bottom = self.axes.bbox.ymin
    self.offsetText.set(va="top", ha="left") 
    oy = bottom - pad * self.figure.dpi / 40.0
    self.offsetText.set_position((1, oy))
    
def top_offset(self, bboxes, bboxes2):
    top = self.axes.bbox.ymax
    self.offsetText.set(va="top", ha="left") 
    oy = top - pad * self.figure.dpi / 1600.0
    self.offsetText.set_position((1, oy))
    
