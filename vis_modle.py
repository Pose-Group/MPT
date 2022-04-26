import time
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class plot_h36m(object):

    def __init__(self, prediction_data, GT_data):
        print ('prediction_data',type(prediction_data))
        print ('GT_data',type(GT_data))
        self.joint_xyz = GT_data
        self.nframes = prediction_data.shape[0]
        self.joint_xyz_f = prediction_data

        # set up the axes
        xmin = -800
        xmax = 800
        ymin = -800
        ymax = 800
        zmin = -800
        zmax = 800

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        
        self.chain = [np.array([0, 1, 2, 3]),
                      np.array([0, 4, 5, 6]),
                      np.array([0, 7, 8, 9, 10]),
                      np.array([8, 11, 12, 13]),
                      np.array([8, 14, 15, 16])]
        print (type(self.chain))
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])
        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][ :],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth
            
    def plot(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        ani.save('./data/WalkDog/vis_0.gif',writer='pillow')
        # ani.save('./data/WalkDog/vis_1.gif',writer='pillow')
        # ani.save('./data/WalkDog/vis_2.gif',writer='pillow')
        # ani.save('./data/WalkDog/vis_3.gif',writer='pillow')
        plt.show()
        
        
if __name__ == '__main__':
    config = yaml.load(open('config.yml'))
    use_node = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,20,21,22])

    #load GT_data
    base_path = './data/WalkDog'

    test_save_path = os.path.join(base_path, 'GT_0.npy')
    # test_save_path = os.path.join(base_path, 'GT_1.npy')
    # test_save_path = os.path.join(base_path, 'GT_2.npy')
    # test_save_path = os.path.join(base_path, 'GT_3.npy')
    GT_data = np.load(test_save_path)
    GT_data = GT_data[0]
    #load prediction_data
    prediction_data_path = os.path.join(base_path, 'vis_0.npy')
    # prediction_data_path = os.path.join(base_path, 'vis_1.npy')
    # prediction_data_path = os.path.join(base_path, 'vis_2.npy')
    # prediction_data_path = os.path.join(base_path, 'vis_3.npy')
    prediction_data = np.load(prediction_data_path)
    # prediction_data = prediction_data[0]
    
    print('prediction_data:\n',prediction_data.shape)
    print('GT_data:\n',GT_data.shape)

    nframes = prediction_data.shape[0]


    prediction_data = prediction_data[:,use_node,:]
    prediction_data = prediction_data.reshape(-1,17,3)
    GT_data = GT_data[:,use_node,:]
    print (GT_data.shape)
    
    
    
    predict_plot = plot_h36m(prediction_data, GT_data)
    predict_plot.plot()
    
    
    
























'''

fig = plt.figure()
ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

chain = [np.array([0, 1, 2, 3]),
         np.array([0, 4, 5, 6]),
         np.array([0, 7, 8, 9, 10]),
         np.array([7, 11, 12, 13]),
         np.array([7, 14, 15, 16])]

scats = []
lns = []
filename = filename

def update(frame):
    for scat in scats:
        scat.remove()
    for ln in lns:
        ax.lines.pop(0)

    scats = []
    lns = []

    xdata = np.squeeze(GT_data[frame, :, 0])
    ydata = np.squeeze(GT_data[frame, :, 1])
    zdata = np.squeeze(GT_data[frame, :, 2])

    xdata_f = np.squeeze(prediction_data[frame, :, 0])
    ydata_f = np.squeeze(prediction_data[frame, :, 1])
    zdata_f = np.squeeze(prediction_data[frame, :, 2])

    for i in range(len(chain)):
        lns.append(ax.plot3D(xdata_f[chain[i][:],], ydata_f[chain[i][:],], zdata_f[chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
        lns.append(ax.plot3D(xdata[chain[i][:],], ydata[chain[i][:],], zdata[chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

ani = FuncAnimation(fig, update, frames=nframes, interval=40, repeat=False)
plt.title(filename, fontsize=16)

plt.show()
'''


