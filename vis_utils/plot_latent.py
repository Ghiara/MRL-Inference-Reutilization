import sys, os, re, time, platform
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import matplotlib
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib.colors as mcolors



PLOT_LIST = []

MARKERS = ['.', '^', 's', 'p', '*', 'X', 'h', 'd', '+', 'P']


def main(path, run_name=None, show_='last', save=True, use_tsne=True, DIM_RED=2, hardcoded=f'{os.getcwd()}/output/toy1d-multi-task/2024_07_11_15_51_58_default_true_gmm/walker_full_30_06'):
   
    data_dir = os.path.join(path, 'tensorboard')
    # data_dir = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_51_58_default_true_gmm'
    fig_dir = os.path.join(path,'log', 'encodings')
    # run_name = DATA_DIR
    # if run_name is not None and os.path.exists(os.path.join(run_name, 'tensorboard')):
    #         DATA_DIR = os.path.join(run_name, 'tensorboard')
    # else:
    #     raise FileNotFoundError('Not a valid path %s' % run_name)
    run_name = Path(data_dir).parent
    # run_name = Path(data_dir)

    fig_folder = os.path.join(fig_dir, f'{os.path.split(data_dir)[-1]}')
    if not os.path.isdir(fig_folder) and save:
        os.makedirs(fig_folder, exist_ok=True)

    epoch_dict = {}
    if hardcoded:
        fig_folder = os.path.join(hardcoded)
        step_ = '0'
        data_dir = hardcoded
        with open(os.path.join(data_dir, 'metadata.tsv'), newline='') as f:
            r = csv.reader(f, delimiter='\t')
            metadata = [row_ for row_ in r]
        with open(os.path.join(data_dir, 'tensors.tsv'), newline='') as f:
            r = csv.reader(f, delimiter='\t')
            data = [row_ for row_ in r]

        # Convert to np
        metadata = np.array(metadata)
        data = np.array(data, dtype=float)

        # Bring data to 3D
        pcad = ''
        if data.shape[1] <= 2:
            temp = np.zeros([data.shape[0], 3])
            temp[:, 0:data.shape[1]] = data
            data = temp

        elif data.shape[1] > 2:
            if use_tsne:
                print(f'Performing T-SNE from {data.shape[1]} DIM to {DIM_RED}')
                pcad = f' (Using T-SNE From {data.shape[1]} To {DIM_RED} Dimensions)'
                true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
                pattern = r'\[([-+]?\d*\.\d+|\d+)\]'
                specs = np.array([float(re.findall(pattern, row[0])[0]) for row in metadata])
                true_tasks[true_tasks=='1']='2'
                true_tasks[(specs<0) & (true_tasks=='0')]='1'
                true_tasks[(specs<0) & (true_tasks=='2')]='3'
                unique_tasks = np.sort(np.unique(true_tasks))
                data = TSNE(n_components=DIM_RED, init='pca').fit_transform(data)
            else:
                print(f'Performing PCA from {data.shape[1]} DIM to {DIM_RED}')
                pcad = f' (Using PCA From {data.shape[1]} To {DIM_RED} Dimensions)'
                data = perform_pca(data)

        epoch_dict[step_] = [metadata, data, pcad]
    else:
        folders_ = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        maxlen = max([len(f) for f in folders_])
        folders_ = sorted([f.zfill(maxlen) for f in folders_])
        for folder in (folders_ if show_ == 'all' else [folders_[-1]]):
            if re.match('[0-9]+', folder):

                step_ = re.findall('([0-9]+)', folder)[0]

                with open(os.path.join(data_dir, folder, 'default', 'metadata.tsv'), newline='') as f:
                    r = csv.reader(f, delimiter='\t')
                    metadata = [row_ for row_ in r]
                with open(os.path.join(data_dir, folder, 'default', 'tensors.tsv'), newline='') as f:
                    r = csv.reader(f, delimiter='\t')
                    data = [row_ for row_ in r]

                # Convert to np
                metadata = np.array(metadata)
                data = np.array(data, dtype=float)

                # Bring data to 3D
                pcad = ''
                if data.shape[1] <= 2:
                    temp = np.zeros([data.shape[0], 3])
                    temp[:, 0:data.shape[1]] = data
                    data = temp

                elif data.shape[1] > 2:
                    if use_tsne:
                        print(f'Performing T-SNE from {data.shape[1]} DIM to {DIM_RED}')
                        pcad = f' (Using T-SNE From {data.shape[1]} To {DIM_RED} Dimensions)'
                        true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
                        unique_tasks = np.sort(np.unique(true_tasks))
                        data = TSNE(n_components=DIM_RED, init='pca').fit_transform(data)
                    else:
                        print(f'Performing PCA from {data.shape[1]} DIM to {DIM_RED}')
                        pcad = f' (Using PCA From {data.shape[1]} To {DIM_RED} Dimensions)'
                        data = perform_pca(data)

                epoch_dict[step_] = [metadata, data, pcad]

        m_l = max([len(k) for k in epoch_dict.keys()])

    if os.path.exists(os.path.join(run_name, 'task_dict.json')):
        with open(os.path.join(run_name, 'task_dict.json'), 'r') as f:
            # Copy so not both changed during updates
            task_dict = json.load(f)
            task_dict = {str(el): key for key, el in task_dict.items()} if task_dict is not None else None

    # Map
    class_map = {'flip_forward': 'Front flip', 'jump': 'Jump',
                 'goal_backward': 'Goal in back', 'goal_forward': 'Goal in front',
                 'stand_back': 'Back stand', 'stand_front': 'Front stand',
                 'velocity_backward': 'Run backward', 'velocity_forward': 'Run forward',
                 'direction_backward': 'Run forward', 'direction_forward': 'Run backward',
                 'reach-v2': 'reach-v2', 'push-v2': 'push-v2', 
                 'velocity_up': 'Velocity Up', 'velocity_down': 'Velocity Down', 'velocity_right': 'Velocity Right', 'velocity_left': 'Velocity Left',
                 'goal_up': 'Goal Up', 'goal_down': 'Goal Down', 'goal_right': 'Goal Right', 'goal_left': 'Goal Left'}

    # Plotting

    plt.style.use('seaborn-v0_8')

    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    size_ = 31
    plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize

    for step_ in sorted(epoch_dict.keys()):

        metadata, values, pcad = epoch_dict[step_]

        true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
        # pattern = r'\[([-+]?\d*\.\d+|\d+)\]'
        # specs = np.array([float(re.findall(pattern, row[0])[0]) for row in metadata])
        # true_tasks[true_tasks=='1']='2'
        # true_tasks[(specs<0) & (true_tasks=='0')]='1'
        # true_tasks[(specs<0) & (true_tasks=='2')]='3'
        unique_tasks = np.sort(np.unique(true_tasks))
        max_len_task = max([len(t) for t in unique_tasks])
        specs = np.array([float(re.findall('[0-9]+[.]*[0-9]*', s[0])[1]) for s in metadata])
        if len(np.array([(re.findall('[0-9]+[.]*[0-9]*', s[0])) for s in metadata])[0]) == 4:
            string_tuple = [re.findall('[0-9]+[.]*[0-9]*', s[0])[1:3] for s in metadata]
            specs = np.zeros(len(string_tuple))
            for i, tp in enumerate(string_tuple):
                idx = np.nonzero([float(s) for s in tp])
                specs[i] = float(tp[idx[0].item()])
        predicted_tasks = np.array([s[0].split('->')[1].strip() for s in metadata])

        fig = plt.figure()

        if values.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('auto')
            # ax.set_title(f'Latent Encodings{pcad} For GMM Training Step {int(step_)}', y=1.08)
        else:
            ax = plt.gca()
            # ax.set_aspect('auto')
            ax.set_title(f'Walker 4-Task')

        legend_elements = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors += ['#db8233', '#8c564b', '#e377c2']
        # for i, target_class in enumerate(unique_tasks):
        #     true_match_values = values[(true_tasks == target_class)]
        #     non_match_values = values[(true_tasks == target_class) & (true_tasks != predicted_tasks)]

        #     brightness = specs_normalized[i]
        #     # Convert the base color to HSV
        #     rgb = mcolors.to_rgb(colors[i % len(colors)])
        #     hsv = mcolors.rgb_to_hsv(rgb)

        #     # Adjust the brightness (V value in HSV)
        #     hsv[2] = brightness * 0.5 + 0.5  # Adjust scaling as needed for visibility     
        #     # Convert back to RGB
        #     new_rgb = mcolors.hsv_to_rgb(hsv)
    
        #     if true_match_values.shape[1] == 3:
        #         el = ax.scatter(true_match_values[:, 0], true_match_values[:, 1], true_match_values[:, 2],
        #                         label=class_map[task_dict[target_class]],
        #                         marker=MARKERS[0],
        #                         color=new_rgb)
        #         # el1 = ax.scatter(non_match_values[:, 0], non_match_values[:, 1], non_match_values[:, 2],
        #         #                 label='wrongly classified',
        #         #                 marker=MARKERS[0],
        #         #                 color='#000000')
        #     else:
        #         el = ax.scatter(true_match_values[:, 0], true_match_values[:, 1],
        #                         label=class_map[task_dict[target_class]],
        #                         marker=MARKERS[0],
        #                         color=new_rgb)
                # el1 = ax.scatter(non_match_values[:, 0], non_match_values[:, 1],
                #                 label='wrongly classified',
                #                 marker=MARKERS[0],
                #                 color='#000000')


        for i, target_class in enumerate(unique_tasks):

            # true_match_values = values[(true_tasks == target_class)]
            # Find indices where true_tasks matches target_class
            matching_indices = np.where(true_tasks == target_class)[0]

            # Use indices to get the matching values
            true_match_values = values[matching_indices]
            specs_filtered = specs[matching_indices]
            specs_filtered = specs_filtered / np.max(specs_filtered)
            for j, true_match_value in enumerate(true_match_values):

                non_match_values = values[(true_tasks == target_class) & (true_tasks != predicted_tasks)]

                brightness = specs_filtered[j]
                # Convert the base color to HSV
                rgb = mcolors.to_rgb(colors[i % len(colors)])
                hsv = mcolors.rgb_to_hsv(rgb)

                # Adjust the brightness (V value in HSV)
                hsv[2] = brightness * 0.5 + 0.5  # Adjust scaling as needed for visibility     
                # Convert back to RGB
                new_rgb = mcolors.hsv_to_rgb(hsv)
        
                if true_match_values.shape[1] == 3:
                    el = ax.scatter(true_match_values[:, 0], true_match_values[:, 1], true_match_values[:, 2],
                                    label=class_map[task_dict[target_class]],
                                    marker=MARKERS[0],
                                    color=new_rgb)
                    # el1 = ax.scatter(non_match_values[:, 0], non_match_values[:, 1], non_match_values[:, 2],
                    #                 label='wrongly classified',
                    #                 marker=MARKERS[0],
                    #                 color='#000000')
                else:
                    el = ax.scatter(true_match_value[0], true_match_value[1],
                                    label=class_map[task_dict[target_class]],
                                    marker=MARKERS[0],
                                    color=new_rgb)

            legend_elements.append(el)
            # legend_elements.append(el1)


        if values.shape[1] == 3:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_zlabel('Latent Dim 3')
            # Equal scaling
            x_lim_min, x_lim_max = ax.get_xlim()
            y_lim_min, y_lim_max = ax.get_ylim()
            z_lim_min, z_lim_max = ax.get_zlim()
            max_ = np.array([(x_lim_max - x_lim_min), (y_lim_max - y_lim_min), (z_lim_max - z_lim_min)]).max()
            ax.set_xlim(x_lim_min - (max_ - (x_lim_max - x_lim_min)) / 2, x_lim_max + (max_ - (x_lim_max - x_lim_min)) / 2)
            ax.set_ylim(y_lim_min - (max_ - (y_lim_max - y_lim_min)) / 2, y_lim_max + (max_ - (y_lim_max - y_lim_min)) / 2)
            ax.set_zlim(z_lim_min - (max_ - (z_lim_max - z_lim_min)) / 2, z_lim_max + (max_ - (z_lim_max - z_lim_min)) / 2)
        else:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')

            ts = ax.get_xticks()
            if len(ts) < 5:
                ax.set_xticks(np.linspace(min(ts), max(ts), 5))
            ts = ax.get_yticks()
            if len(ts) < 5:
                ax.set_yticks(np.linspace(min(ts), max(ts), 5))

        # ax.view_init(-90, 90)

        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), markerscale=1.5)

        fig.set_size_inches(10., 7.1) #8 task

        if save:
            plt.savefig(os.path.join(fig_folder, f'encodings.pdf'), format='pdf', dpi=100, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        print(f'Created plot for {data_dir}')


def perform_pca(values):

    # sample points equally for all gaussians
    x = np.copy(values)

    # centering the data
    x -= np.mean(x, axis=0)

    cov = np.cov(x, rowvar=False)

    evals, evecs = LA.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx[:3]]

    return np.dot(x, evecs)
if __name__ == '__main__':

    paths = [
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_22_13_11_43_default_true_gmm',
            #  '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_22_13_45_29_default_true_gmm',
            #  '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_23_12_07_04_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_23_13_45_47_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_12_29_51_default_true_gmm',
            # "/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_14_07_17_default_true_gmm",
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_14_25_52_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_15_50_38_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/cheetah-multi-task/2024_02_27_16_53_13_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_17_21_28_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_27_22_08_00_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_06_39_55_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_07_15_53_default_true_gmm',
            # "/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_07_59_30_default_true_gmm",
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_09_18_43_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_09_47_54_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_10_19_49_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_10_35_51_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_11_23_16_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_11_58_32_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_12_49_28_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_13_54_22_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_16_39_34_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_17_59_02_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_18_33_12_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_19_38_08_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_20_34_55_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_28_21_39_11_default_true_gmm'
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_08_39_25_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_02_29_08_44_11_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_01_13_02_32_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_01_13_32_17_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_01_13_48_07_default_true_gmm',
            # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_08_31_28_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_10_00_39_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_10_02_02_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_11_04_15_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_17_42_52_default_true_gmm', # Try new_position as input instead of action
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_19_17_58_default_true_gmm', # latent_dim = 8
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_19_47_02_default_true_gmm', # gamma = 7
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_20_17_16_default_true_gmm', # gamma = 3
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_20_20_29_default_true_gmm', # gamma = 5, latent_dim = 2
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_20_33_38_default_true_gmm', # latent_dim = 6
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_21_34_23_default_true_gmm', # latent_dim = 8, change velocity to pos diff
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_21_35_23_default_true_gmm', # change vel back, time steps = 5
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_22_08_03_default_true_gmm', # time steps = 3
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_02_22_12_02_default_true_gmm', # time steps = 5, max_path_length = 10, max traj = 40
        # '/home/ubuntu/juan/melts/output/cheetah-multi-task/2024_02_20_10_32_49_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_08_00_25_default_true_gmm', # continue training latent_dim = 8
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_08_01_56_default_true_gmm', # time steps = 10, max_path_length = 10
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_08_05_33_default_true_gmm', # increase reconstruction time steps to 256 
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_09_22_39_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_09_22_54_default_true_gmm', #back to scaled path
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_09_27_21_default_true_gmm', # change m_startlap to 3 (no merges)
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_11_46_32_default_true_gmm', # hard assignment
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_12_07_01_default_true_gmm', # hard assignment m_startlap = 2
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_12_11_27_default_true_gmm', #small actions
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_13_15_34_default_true_gmm', # velocity is summed too.
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_13_24_03_default_true_gmm', # all  values set to small values
        # '/home/ubuntu/juan/melts/output/cheetah-multi-task/2024_03_03_14_31_36_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_19_26_44_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_19_23_15_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_21_18_19_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_09_08_00_default_true_gmm',     # only use action_space 4
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_10_08_25_default_true_gmm',     # increase sac_alpha to 0.5 for more exploration
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_15_46_22_default_true_gmm', # smaller actions
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_22_07_02_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_07_52_46_default_true_gmm', # set not max actions to zero
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_08_12_19_default_true_gmm', # Same with slightly different parameters
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_13_07_18_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_16_15_31_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_06_16_15_31_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_06_14_23_default_true_gmm',
        # "/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_08_38_48_default_true_gmm",
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_09_28_13_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_09_43_13_default_true_gmm'
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_09_50_00_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_09_51_32_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_10_02_57_default_true_gmm',
        # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_07_10_40_43_default_true_gmm',
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_02_18_35_27_default_true_gmm',     # old model without exploration, but with task, toy, new toy
    # '/home/ubuntu/juan/melts/output/cheetah-multi-task/2024_04_02_18_49_45_default_true_gmm',   # same but with cheetah, see if task is problem
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_02_20_44_03_default_true_gmm',     # four tasks
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_32_07_default_true_gmm',     # reduce time_steps to 10 from 64
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_39_59_default_true_gmm',     # add randomization, var=0.2
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_42_22_default_true_gmm',     # var = 0.1
    # # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_08_45_24_default_true_gmm',     # quita random, rdeuce gear to 50, increase steps to 500
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_03_16_16_28_default_true_gmm',     # increate time_step to 64, gear to 100
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_04_12_56_41_default_true_gmm',     # make again with setting position, velocity
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_04_15_39_22_default_true_gmm',     # reduce time_steps to 10, randomize setting of x_pos, vel
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_10_15_27_default_true_gmm',     # longer training
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_10_38_05_default_true_gmm',     # gear 200
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_08_18_49_39_default_true_gmm',     # encoder without action
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_09_11_42_26_default_true_gmm',     # deactivate normalization
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_09_19_28_19_default_true_gmm',     # different variant.json
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_10_13_56_50_default_true_gmm',     # continue training
    # '/home/ubuntu/juan/melts_copy/output/toy1d-multi-task/2024_04_11_09_21_29_default_true_gmm', # remove action from task decoder
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_11_09_25_22_default_true_gmm',     # add some randomness in the action
    # '/home/ubuntu/juan/melts/output/toy2d-multi-task/2024_04_16_08_11_01_default_true_gmm',     # toy2d
    # # # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_15_20_42_34_default_true_gmm'
    # # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_09_23_06_default_true_gmm',     # gear350, more randomization
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_17_20_51_34_default_true_gmm',     # reduce dt to 0.01 increase gear to 5000
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_19_06_31_18_default_true_gmm',     # gear 2000 dt 0.05
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_19_19_08_21_default_true_gmm',     # change to set positions again
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_19_15_53_45_default_true_gmm',     # more randomness, gear 1000, dt 0.05
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_19_23_09_04_default_true_gmm',     # set position with randomness
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_22_15_43_33_default_true_gmm',     # completely random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_23_12_27_55_default_true_gmm',     # no idea
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_26_16_36_36_default_true_gmm',     # time step 10, env.dt 0.2, gear 300, traj_len 100
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_27_05_36_11_default_true_gmm',     # dt 0.5, steps 50 look at videos
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_30_10_59_18_default_true_gmm',     # train for test with dt 0.01 step 20
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_01_10_44_33_default_true_gmm',     # gear 500, before 300
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_01_19_13_32_default_true_gmm',     # keep on training
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_03_07_19_34_default_true_gmm',     # keep on training with randomization
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_03_07_49_45_default_true_gmm',     # keep training with complete random actions
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_04_05_53_56_default_true_gmm',     # dt 0.01, completely random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_04_15_07_45_default_true_gmm',     # dt 0.01
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_06_06_00_56_default_true_gmm',     # remove exploration
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_06_10_03_55_default_true_gmm',     # completely random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_06_15_05_36_default_true_gmm',     # only some randomness around actions
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm',     # gear 1000, dt 0.1
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_39_default_true_gmm',     # gear 500, dt 0.1
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_43_53_default_true_gmm',     # gear 500, no randomization
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_10_13_15_03_default_true_gmm',     # keep training '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_04_30_10_59_18_default_true_gmm', dt 0.1, gear 500
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_11_07_10_54_default_true_gmm'
    # '/home/ubuntu/juan/SAC/output/toy1d-multi-task/2024_05_11_14_28_38_default_true_gmm'
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_15_10_43_10_default_true_gmm',     # new config, random 0.9, var
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_23_14_27_54_default_true_gmm',     # more tasks
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_09_00_44_46_default_true_gmm',     # keep on training from '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm', random = 0.9
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_09_12_00_22_default_true_gmm',     # keep on training from '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm', random = 0.6
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_41_00_default_true_gmm',     # keep on training from '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm', no randomization
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_51_58_default_true_gmm',     # keep on training from '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm', alpha=0, no random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_52_42_default_true_gmm',     # keep on training from '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_05_08_14_42_25_default_true_gmm', alpha = 0, random=0.8
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_17_04_04_default_true_gmm',     # new, probably alpha 0, random = 0.8
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_17_25_37_default_true_gmm',     # new, probably alpha 0, no random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_05_14_28_49_default_true_gmm',     # own simulation implementation, no randomness
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_05_14_45_53_default_true_gmm',     # more power (from 10 to 50)
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_05_23_10_46_default_true_gmm',     # improve
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_05_23_23_38_default_true_gmm',     # make it 50
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_06_00_28_43_default_true_gmm',     # dt=0.1
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_06_13_41_55_default_true_gmm',     # time_steps 32
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_01_07_34_default_true_gmm',     # alpha 0.2
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_13_04_42_default_true_gmm',     # continue /home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_01_07_34_default_true_gmm with timestep 200
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_13_01_28_default_true_gmm',     # " but with alpha 0
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_17_18_45_34_default_true_gmm',     # continue /home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_07_01_07_34_default_true_gmm with random 0.5
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_18_13_56_55_default_true_gmm',     # " var 0.2
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_18_13_57_09_default_true_gmm',     # var 0.05
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_19_01_52_11_default_true_gmm',     # bigger steps
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_19_23_25_51_default_true_gmm',     # no random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_20_11_11_23_default_true_gmm',     # var 0.2
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_08_25_12_38_14_default_true_gmm',     # var 0.5
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_01_20_35_45_default_true_gmm',     # all tasks, new toy
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm',     # dt 0.1 no random
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_21_06_default_true_gmm',     # var 0.2
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_21_50_default_true_gmm',     # var 0.5
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_07_29_57_default_true_gmm',     # all tasks, now right
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_09_13_11_12_default_true_gmm'
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_10_08_03_29_default_true_gmm'
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_10_08_03_29_default_true_gmm',
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm',     # different random, var=0.1
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_11_49_42_default_true_gmm',     # no random, 10 steps
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm',     # 32 steps, random var=0.2
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_20_31_24_default_true_gmm',     # var = 0.05
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_13_00_07_13_default_true_gmm',     # next is var 0.1, steps 10
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_14_20_58_59_default_true_gmm',     # var 0.1, multi
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_14_20_55_42_default_true_gmm',     # var 0.1, step 10
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_12_22_59_default_true_gmm',     # keep training var0.1 step 32, without random ??
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm',     # keep training var0.1 step 32, with var 0.1 ??
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_21_11_05_23_default_true_gmm',     # var 0.05
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_21_11_06_00_default_true_gmm',     # var 0.02
    '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_23_12_40_04_default_true_gmm',     # var 0.05


        ]
    for path in paths:
        main(path, *sys.argv[1:], hardcoded=None)
