import json
import numpy as np
import torch
import cv2
from typing import List, Any, Dict, Callable
import imageio
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

import os
from datetime import datetime
import pytz

from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between

def log_all(agent, path, q1_loss, policy_loss, rew, traj_len, episode):
    '''
    # This function is used to save loss/reward histories and weights under structure:
    # - path
    #     - plots
    #         - mean_reward_history
    #         - qf_loss
    #         - policy_loss
    #     - models
    #         - transfer_function / policy_net
    #         - qf1e
    #         - value
    '''

        # TODO: save both vf losses (maybe with arg)
    def save_plot(loss_history, name:str, path=f'{os.getcwd()}/evaluation/transfer_function/one-sided/', figure_size: Tuple[int,int] = (20, 10)):
        def remove_outliers_iqr(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = np.array(data)
            return data[(data >= lower_bound) & (data <= upper_bound)]
        
        def moving_average(data, window_size=10):
            index_array = np.arange(1, len(data) + 1)
            data = pd.Series(data, index = index_array)
            return data.rolling(window=window_size).mean()
        def format_label(label):
            words = label.split('_')
            return ' '.join(word.capitalize() for word in words)
        
        # TODO: save both vf losses (maybe with arg)
        os.makedirs(path, exist_ok=True)
        fig, axs = plt.subplots(1, figsize=figure_size)
        smooth_plot(axs, np.arange(len(loss_history)), loss_history, label=format_label(name))
        axs.legend()
        axs.set_xlabel('Train epochs')

        fig.savefig(os.path.join(path,name+'.png'))

        plt.close()

    # Save networks
    curr_path = path + '/models/policy_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.policy_network.cpu(), save_path)
    curr_path = path + '/models/vf1_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.q_value_network1.cpu(), save_path)
    curr_path = path + '/models/vf2_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.q_value_network2.cpu(), save_path)
    curr_path = path + '/models/value_model/'
    os.makedirs(os.path.dirname(curr_path), exist_ok=True)
    save_path = curr_path + f'epoch_{episode}.pth'
    if episode % 100 == 0:
        torch.save(agent.value_network.cpu(), save_path)
    agent.q_value_network1.cuda() 
    agent.q_value_network2.cuda()
    agent.value_network.cuda()
    agent.policy_network.cuda() 

    # Save plots
    path_plots = path + '/plots/'
    save_plot(q1_loss, name='vf_loss', path=path_plots)
    save_plot(rew, name='reward_history', path=path_plots)
    save_plot(policy_loss, name='policy_loss', path=path_plots)
    save_plot(traj_len, name='traj_len', path=path_plots)


def _frames_to_gif(frames: List[np.ndarray], info, gif_path, transform: Callable = None):
    """ Write collected frames to video file """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
        for i, frame in enumerate(frames):
            frame = frame.astype(np.uint8)  # Ensure the frame is of type uint8
            frame = np.ascontiguousarray(frame)
            cv2.putText(frame, 'reward: ' + str(info['reward'][i]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'obs: ' + str(info['obs'][i]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'action: ' + str(info['action'][i]), (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'task: ' + str(info['base_task'][i]), (0, 75), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            cv2.putText(frame, 'task: ' + str(info['task'][i]), (0, 95), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
            # Apply transformation if any
            if transform is not None:
                frame = transform(frame)
            else:
                # Convert color space if no transformation provided
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            writer.append_data(frame)
