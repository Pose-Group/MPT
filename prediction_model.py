import yaml
import h5py
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataloader import DataLoader
import data_utils
import space_angle_velocity
import bone_length_loss
import model_4GRU


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = yaml.load(open('./config.yml'),Loader=yaml.FullLoader)


node_num = config['node_num']
input_n=config['input_n']
output_n=config['output_n']
base_path = './data/WalkDog'
input_size = config['in_features']
hidden_size = config['hidden_size']
output_size = config['out_features']
batch_size = config['batch_size']

use_node = np.array([ 0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22])

#load data
s80 = []
s160 = []
s320 = []
s400 = []
s560 = []
s720 = []
s1000 = []

# Enter the start frame for testing.
index_list=[]

for i in range (len(index_list)):
    
    #test_save_path = os.path.join(base_path, 'WalkDog.npy')
    test_save_path = os.path.join(base_path, 'WalkDog_1.npy')
    test_save_path = test_save_path.replace("\\","/")
    dataset = np.load(test_save_path,allow_pickle = True)
    dataset = torch.tensor(dataset, dtype = torch.float32,requires_grad=False)
    
    # Compared to the original sequence, our input sequence is one frame shorter.
    dataset = dataset[index_list[i]-1:,:,:].cuda()
    
    input_dataset = dataset[0:input_n]
    output_dataset = dataset[input_n:input_n+output_n]
    input_dataset = input_dataset.expand(batch_size,input_dataset.shape[0],input_dataset.shape[1],input_dataset.shape[2])
    output_dataset = output_dataset.expand(batch_size,output_dataset.shape[0],output_dataset.shape[1],output_dataset.shape[2])
    
    input_dataset = input_dataset
    output_dataset = output_dataset
    
    
    total_samples = 0
    total_mse = 0
    total_mpjpe = 0
    
    model_x = torch.load(os.path.join(base_path, 'generator_x_4GRU.pkl')).to(device)
    model_y = torch.load(os.path.join(base_path, 'generator_y_4GRU.pkl')).to(device)
    model_z = torch.load(os.path.join(base_path, 'generator_z_4GRU.pkl')).to(device)
    model_v = torch.load(os.path.join(base_path, 'generator_v_4GRU.pkl')).to(device)
    
    
    input_angle = input_dataset[:, 1:, :, :3]
    input_velocity = input_dataset[:, 1:, :, 3].permute(0, 2, 1)
    
    
    target_angle = output_dataset[:, :, :, :3]
    target_velocity = output_dataset[:, :, :, 3]
    #read velocity
    input_velocity = input_velocity.float()
    target_velocity = target_velocity.float()
    #read angle_x
    input_angle_x = input_angle[:,:,:,0].permute(0, 2, 1).float()
    target_angle_x = target_angle[:,:,:,0].float()
    #read angle_y
    input_angle_y = input_angle[:,:,:,1].permute(0, 2, 1).float()
    target_angle_y = target_angle[:,:,:,1].float()
    #read angle_z
    input_angle_z = input_angle[:,:,:,2].permute(0, 2, 1).float()
    target_angle_z = target_angle[:,:,:,2].float()
    #read 3D data
    input_3d_data = input_dataset[:, :, :, 4:]
    target_3d_data =output_dataset[:, :, :, 4:]
    
    output_v, _ = model_v(input_velocity,  hidden_size)
    output_v = output_v.view(target_velocity.shape[0],target_velocity.shape[2],output_size)
    
    output_x, _ = model_x(input_angle_x, hidden_size)
    output_x = output_x.view(target_angle_x.shape[0],target_angle_x.shape[2],output_size)
    
    output_y, _ = model_y(input_angle_y, hidden_size)
    output_y = output_y.view(target_angle_y.shape[0],target_angle_y.shape[2],output_size)
    
    output_z, _ = model_z(input_angle_z, hidden_size)
    output_z = output_z.view(target_angle_z.shape[0],target_angle_z.shape[2],output_size)
    
    angle_x = output_x.permute(0, 2, 1)
    angle_y = output_y.permute(0, 2, 1)
    angle_z = output_z.permute(0, 2, 1)
    pred_v = output_v.permute(0, 2, 1)
    
    pred_angle_set = torch.stack((angle_x,angle_y,angle_z),3)
    
    pred_angle_set = pred_angle_set.reshape(pred_angle_set.shape[0],pred_angle_set.shape[1],-1,3)
    
    #reconstruction_loss
    input_pose = torch.zeros((target_velocity.shape[0], output_n, input_3d_data.shape[-2], input_3d_data.shape[-1]))
    
    for a in range(input_pose.shape[0]):
        input_pose[a,0,:,:] = input_3d_data[a,input_n-1,:,:]
    re_data = torch.FloatTensor([])
    for b in range (target_3d_data.shape[0]):
        for c in range (target_3d_data.shape[1]):
            reconstruction_coordinate = space_angle_velocity.reconstruction_motion(pred_v[b,c,:,], pred_angle_set[b, c,:,:], input_pose[b, c, :, :],node_num)
            re_data = torch.cat([re_data,reconstruction_coordinate],dim=0)
            reconstruction_coordinate = reconstruction_coordinate
            if c+1<target_3d_data.shape[1]:
                input_pose[b,c+1,:,:] = reconstruction_coordinate
            else:
                continue
    re_data = re_data.view(target_3d_data.shape[0],-1,node_num,3)
    
    frame_re_data = re_data[0]
    frame_target_3d_data = target_3d_data[0]
    
    # For fair, following "Learning dynamic relationships for 3d human motion prediction, CVPR, 2020", each pose is represented as a skeleton of 17 joints.
    mpjpe_set = []
    for e in range (frame_re_data.shape[0]):
        frame_re_data = frame_re_data.to(device)
        frame_target_3d_data = frame_target_3d_data.to(device)
        frame_rec_loss = torch.mean(torch.norm(frame_re_data[e,use_node,:] - frame_target_3d_data[e,use_node,:], 2, 1))
        
        mpjpe_set.append(frame_rec_loss)
    
    #save vis data
    frame_target_3d_data = frame_target_3d_data.cpu()
    frame_re_data = frame_re_data.cpu()
    frame_target_3d_data = np.array(frame_target_3d_data[0])
    mpjpe_set = torch.Tensor(mpjpe_set)
    mpjpe_set = np.array(mpjpe_set)
    # vis_mpjpe_save_path = os.path.join(base_path, 'vis_mpjpe.npy')
    # np.save(vis_save_path, frame_re_data)
    # np.save(vis_mpjpe_save_path, mpjpe_set)
    
    vis_name = 'vis_' + str(i) + '.npy'
    GT_name = 'GT_' + str(i) + '.npy'
    
    vis_save_path = os.path.join(base_path, vis_name)
    GT_save_path = os.path.join(base_path, GT_name)
    
    vis_mpjpe_save_path = os.path.join(base_path, 'vis_mpjpe.npy')
    np.save(vis_save_path, frame_re_data)
    np.save(GT_save_path, target_3d_data.cpu())
    np.save(vis_mpjpe_save_path, mpjpe_set)

    #print ('-------------------') 
    '''
    print ('mpjpe_set',mpjpe_set)   
    print ('frame_re_data.shape:\n',frame_re_data.shape)
    
    print('80ms:\n',mpjpe_set[1])
    print('160ms:\n',mpjpe_set[3])
    print('320ms:\n',mpjpe_set[7])
    print('400ms:\n',mpjpe_set[9])
    print('560ms:\n',mpjpe_set[13])
    print('720ms:\n',mpjpe_set[17])
    print('100ms:\n',mpjpe_set[24])
    '''
    s80.append(mpjpe_set[1])
    s160.append(mpjpe_set[3])
    s320.append(mpjpe_set[7])
    s400.append(mpjpe_set[9])
    s560.append(mpjpe_set[13])
    s720.append(mpjpe_set[17])
    s1000.append(mpjpe_set[24])
print('-----------------------')
print('80ms:\n', torch.mean(torch.Tensor(s80)))
print('160ms:\n',torch.mean(torch.Tensor(s160)))
print('320ms:\n',torch.mean(torch.Tensor(s320)))
print('400ms:\n',torch.mean(torch.Tensor(s400)))
print('560ms:\n',torch.mean(torch.Tensor(s560)))
print('720ms:\n',torch.mean(torch.Tensor(s720)))
print('1000ms:\n',torch.mean(torch.Tensor(s1000)))































