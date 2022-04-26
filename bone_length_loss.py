import torch
import yaml
from torch import nn

def bone_length_loss(predicted_shot, out_shot):    
    predicted_shot = predicted_shot.reshape(-1,17,3)
    out_shot = out_shot.reshape(-1,17,3)
    skeleton_bone_length_sum  = torch.sum(out_shot[0])

    skeleton_bone_length = torch.tensor([])
    skeleton_parent = out_shot.index_select(1,torch.tensor([  0,1,2,3,4,0,6,7,8, 9, 0,12,13,14,13,17,18,19,19,13,25,26,27,27]))
    skeleton_children = out_shot.index_select(1,torch.tensor([1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,22,21,25,26,27,30,29]))
    for i in range (skeleton_parent.shape[1]):
        skeleton_dist_matrix = torch.sqrt(torch.sum((skeleton_parent[0,i]-skeleton_children[0,i])**2))
        skeleton_dist_matrix = torch.unsqueeze(skeleton_dist_matrix, 0)
        skeleton_bone_length = torch.cat((skeleton_bone_length, skeleton_dist_matrix), dim=0)
    skeleton_bone_length_sum  = torch.sum(skeleton_bone_length)

    
    prediction_parent  =  predicted_shot.index_select(1,torch.tensor([0,1,2,3,4,0,6,7,8, 9, 0,12,13,14,13,17,18,19,19,13,25,26,27,27]))
    prediction_children = predicted_shot.index_select(1,torch.tensor([1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,22,21,25,26,27,30,29]))
    prediction_bone_length = torch.tensor([])
    for a in range (predicted_shot.shape[0]):
        for i in range (prediction_parent.shape[1]):
            prediction_dist_matrix = torch.sqrt(torch.sum((prediction_parent[a,i]-prediction_children[a,i])**2))
            prediction_dist_matrix = torch.unsqueeze(prediction_dist_matrix, 0)
            prediction_bone_length = torch.cat((prediction_bone_length, prediction_dist_matrix), dim=0)
    out_shot_parent  =  out_shot.index_select(1,torch.tensor([0,1,2,3,4,0,6,7,8, 9, 0,12,13,14,13,17,18,19,19,13,25,26,27,27]))
    out_shot_children = out_shot.index_select(1,torch.tensor([1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,22,21,25,26,27,30,29]))
    out_shot_bone_length = torch.tensor([])
    out_shot_bone_ratio = torch.tensor([])
    for a in range (out_shot.shape[0]):
        for i in range (out_shot_parent.shape[1]):
            out_shot_dist_matrix = torch.sqrt(torch.sum((out_shot_parent[a,i]-out_shot_children[a,i])**2))
            out_shot_dist_matrix = torch.unsqueeze(out_shot_dist_matrix, 0)
            out_bone_ratio = out_shot_dist_matrix/skeleton_bone_length_sum
            out_shot_bone_ratio = torch.cat((out_shot_bone_ratio, out_bone_ratio), dim=0)
            out_shot_bone_length = torch.cat((out_shot_bone_length, out_shot_dist_matrix), dim=0)

    prediction_bone_length = prediction_bone_length.mul(out_shot_bone_ratio)

    out_shot_bone_length = out_shot_bone_length.mul(out_shot_bone_ratio)

    mse = nn.MSELoss(reduction='mean')
    bone_length_loss = mse(prediction_bone_length,out_shot_bone_length)

    return bone_length_loss