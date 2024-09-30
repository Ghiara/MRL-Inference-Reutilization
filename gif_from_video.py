import imageio
import os
from pathlib import Path
from PIL import Image
import numpy as np

def video_to_gif(path_without_extension, fps=10):
    """
    Convert a video file to a GIF.
    
    Parameters:
    - video_path: Path to the input video file.
    - gif_path: Path where the output GIF should be saved.
    - fps: Frames per second (how many frames of the video to include in each second of the GIF).
    """
    video_path = path_without_extension + '.mp4'
    gif_path = path_without_extension + '_gif.mp4'

    # Create a reader for the video file
    reader = imageio.get_reader(video_path)
    
    # Get video metadata to adjust GIF fps if necessary
    video_fps = reader.get_meta_data()['fps']
    frame_skip = max(1, round(video_fps / fps))
    
    # Create a writer for saving GIF, adjust 'fps' to your liking
    writer = imageio.get_writer(gif_path, fps=fps)
    
    # Read and write frames
    for i, frame in enumerate(reader):
        if i % frame_skip == 0:  # Skip frames to adjust GIF fps
            frame = add_padding(frame)
            writer.append_data(frame)
    
    # Close the writer to finish writing the GIF file
    writer.close()

    with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
        for i, frame in enumerate(reader):
            writer.append_data(frame)
    
    print(f"Converted {video_path} to {gif_path}")

def add_padding(frame):
    """Add black padding to the frame to reach the target size."""
    img = Image.fromarray(frame)
    old_size = img.size
    target_size = (max(old_size), max(old_size))
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size])
    
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    new_img = Image.new("RGB", target_size)
    new_img.paste(img, ((target_size[0]-new_size[0])//2,
                        (target_size[1]-new_size[1])//2))
    
    return np.array(new_img)

    
if __name__ == "__main__":
    directories_path = [
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_14_17_38_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_13_13_51_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_22_13_11_43_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_08_39_25_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_08_44_11_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_21_39_11_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_20_34_55_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_10_00_39_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_10_02_02_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_11_04_15_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/cheetah-multi-task/2024_02_20_10_32_49_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_23_13_45_47_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_21_18_19_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_09_08_00_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_15_46_22_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_22_07_02_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_13_07_18_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_07_52_46_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_16_15_31_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_06_14_23_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_02_20_44_03_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_45_24_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_16_16_28_default_true_gmm'       
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_42_22_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_39_59_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_04_12_56_41_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_04_15_39_22_default_true_gmm', 
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_10_15_27_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_10_38_05_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_18_49_39_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_09_11_42_26_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_09_19_28_19_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_10_13_56_50_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_11_09_25_22_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy2d-multi-task/2024_04_12_15_34_15_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_15_20_42_34_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_20_51_34_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_20_51_34_default_true_gmm'
        # '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/half_cheetah_multi',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_23_12_27_55_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_22_15_43_33_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_26_16_36_36_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_27_09_46_39_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_01_10_44_33_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_03_21_10_34_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_05_18_37_31_default_true_gmm', 
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_04_15_07_45_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_43_53_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_39_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_10_10_26_13_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_10_13_15_03_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_11_10_25_45_default_true_gmm'
        # '/home/ubuntu/juan/SAC/output/toy1d-multi-task/2024_05_11_14_28_38_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_11_16_59_40_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_20_51_34_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_15_10_43_10_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_23_14_27_54_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_09_00_44_46_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_41_00_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_51_58_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_52_42_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_17_04_04_default_true_gmm', 
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_17_25_37_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_06_00_28_43_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_05_23_23_38_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_13_01_28_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_13_04_42_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_08_10_17_04_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_01_08_41_47_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_25_12_38_14_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_01_20_35_45_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_21_06_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_21_50_default_true_gmm', 
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_07_29_57_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_09_13_11_12_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_10_08_03_29_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_11_49_42_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm'
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_20_31_24_default_true_gmm',
        '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_13_00_07_13_default_true_gmm'
        
        ]
    for directory in directories_path:
        video_dir = os.path.join(directory, Path('videos'))
        # video_dir = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_30_10_59_18_default_true_gmm/videos/task_goal_forward'
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                file_path = os.path.join(root, file)  # Construct absolute path
                path_without_extension, extension = os.path.splitext(file_path)
                if os.path.exists(path_without_extension+'_gif.mp4') or path_without_extension.endswith('_gif'):
                    continue
                video_to_gif(path_without_extension, fps=10)