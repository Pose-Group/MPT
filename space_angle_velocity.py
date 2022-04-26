import os
import yaml
import math 
import numpy as np
import torch
import torch.nn.functional as F
import data_utils



def reconstruction_motion(prediction_distance, prediction_angle, current_frame, node_num):
    reconstruction_coordinate = torch.zeros([node_num,3], dtype = torch.float32)
    for i in range (current_frame.shape[0]):

        x = current_frame[i,0] + prediction_distance[i]*prediction_angle[i,0]
        y = current_frame[i,1] + prediction_distance[i]*prediction_angle[i,1]
        z = current_frame[i,2] + prediction_distance[i]*prediction_angle[i,2]
        current_joint_coordinates = torch.tensor([x, y, z])
        reconstruction_coordinate[i] = current_joint_coordinates

    return reconstruction_coordinate

def mpjpe(input, target):
    return torch.mean(torch.norm(input - target, 2, 1))
