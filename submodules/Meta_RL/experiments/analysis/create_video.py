"""
This file helps to create videos of meta-RL agents. 

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-04-06
"""

from smrl.experiment.analysis import result_loader, load_results
from mrl_analysis.video.video_creator import VideoCreator

video_creator = VideoCreator()
video_creator.fps = 30
resolution_factor = 0.5

paths = [
    "/home/ubuntu/juan/Meta-RL/data/toy1d_rand_Base-config_2023-11-14_12-27-45",
]

for name, r, _, _ in result_loader(paths):
    env = r['eval_env']
    video_creator.create_video(
        r, 
        video_length=25.0,
        save_as=f"./videos/{name}.mp4", 
        env_reset_interval=250, 
        width=int(env.screen_width * resolution_factor), 
        height=int(env.screen_height * resolution_factor),
    )
