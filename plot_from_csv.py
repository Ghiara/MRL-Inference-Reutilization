import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mrl_analysis.plots.plot_settings import *
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between
from typing import Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mrl_analysis.plots.plot_settings import *
from mrl_analysis.utility.data_smoothing import smooth_plot, smooth_fill_between, smooth_data
from typing import Tuple
import os



def save_plot(loss_histories, names, plot_name, path='/home/ubuntu/juan/melts/experiment_plots', figure_size: Tuple[int,int] = (20, 10), fill: bool = True, rolling_window: int = 50, fill_scaling_factor: float = 1.0, kernel_width: int = None, scale_param: int = None):
    def format_label(label):
        words = label.split('_')
        return ' '.join(word.capitalize() for word in words)

    os.makedirs(path, exist_ok=True)
    fig, axs = plt.subplots(1, figsize=figure_size)
    
    for loss_history, name in zip(loss_histories, names):
        x_values = np.arange(len(loss_history))

        # Generate the smoothed data using the smooth_plot logic
        if kernel_width is None: kernel_width = int(len(x_values) / 10)
        if scale_param is None: scale_param = len(x_values) / 50
        smoothed_values = smooth_data(loss_history, scale_param, kernel_width)

        # Initialize arrays for the rolling averages
        rolling_avg_upper = np.zeros(len(loss_history))
        rolling_avg_lower = np.zeros(len(loss_history))

        # Calculate rolling averages based on smoothed values
        for i in range(len(loss_history)):
            if i < rolling_window:
                # For the first few points, take the average of the seen points
                recent_data = loss_history[:i + 1]
            else:
                # For the rest, take the last 50 points (or rolling_window)
                recent_data = loss_history[i - rolling_window + 1:i + 1]
            
            # Split recent_data into above and below the smoothed value
            above_smooth_data = recent_data[recent_data >= smoothed_values[i]]
            below_smooth_data = recent_data[recent_data < smoothed_values[i]]

            # If current point is above the smoothed value, use rolling average above; otherwise use rolling average below
            if loss_history[i] >= smoothed_values[i]:
                rolling_avg_upper[i] = np.mean(above_smooth_data) if len(above_smooth_data) > 0 else smoothed_values[i]
                rolling_avg_lower[i] = smoothed_values[i]  # Use smoothed value for the lower boundary
            else:
                rolling_avg_lower[i] = np.mean(below_smooth_data) if len(below_smooth_data) > 0 else smoothed_values[i]
                rolling_avg_upper[i] = smoothed_values[i]  # Use smoothed value for the upper boundary

        # Plot the smoothed data (replacing plot_original=False)
        smooth_plot(axs, x_values, loss_history, label=format_label(name), plot_original=False)

        if fill:
            # Fill between the rolling averages
            smooth_fill_between(axs, x_values, rolling_avg_lower, rolling_avg_upper, label=None, alpha=0.3)

    # Set axis labels with font size 20
    axs.set_xlabel('Train epochs', fontsize=36, labelpad=15)
    axs.set_ylabel('Reward History', fontsize=36)

    # Adjust tick label size to 20
    axs.tick_params(axis='both', which='major', labelsize=36)
    axs.tick_params(axis='both', which='minor', labelsize=36)

    axs.legend(fontsize=24)
    plt.tight_layout()
    print(path, names)
    
    # Save plot as PNG and PDF
    fig.savefig(os.path.join(path, '_'.join(plot_name.split()) + '.png'))
    fig.savefig(os.path.join(path, '_'.join(plot_name.split()) + '.pdf'), bbox_inches='tight')
    plt.close()

# Example usage remains the same.


# The rest of your script remains the same.

def load_and_plot(csv_files, column_name, plot_name, rows_range=(0,2400)):
    loss_histories = []
    names = []
    
    for i, csv_file in enumerate(csv_files['paths']):
        # Load the data from the CSV file
        data = pd.read_csv(csv_file)

        # Check if the specified column exists
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file: {csv_file}")
        
        # Extract the column data
        filtered_data = data[column_name].dropna()

        # Adjust row range if it exceeds available data length
        if rows_range[1] > len(filtered_data):
            print(f"Requested range {rows_range} exceeds available data length {len(filtered_data)}. Adjusting to maximum available rows.")
            filtered_data = filtered_data.iloc[rows_range[0]:]
        else:
            filtered_data = filtered_data.iloc[rows_range[0]:rows_range[1]]

        # Convert to numpy array for calculations
        loss_history = filtered_data.values

        # Store the data and name for plotting
        loss_histories.append(loss_history)
        names.append(csv_files['names'][i])

        # Print the average of the data
        average_loss = np.mean(loss_history)
        print(f"The average of '{column_name}' in file '{csv_file}' is {average_loss}")

    # Call the save_plot function with all the extracted data
    save_plot(loss_histories, names, plot_name, path='/home/ubuntu/juan/melts/experiment_plots', figure_size=(20, 10), fill=True, rolling_window=50, fill_scaling_factor=1.0)
csv_file_paths = dict(
    paths = ['/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_02_15_23_09_default_true_gmm/cheetah_retrain_beta_0.1/progress.csv',
             '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_11_10_26_00_default_true_gmm/progress.csv',
             '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_12_09_17_42_default_true_gmm/cheetah_retrain_beta_0.1/progress.csv',],
    names = ['No random', 'var_0.1','var0.2']
)
column_name = 'Reward History'
plot_name = 'Randomness comparisson'
load_and_plot(csv_file_paths, column_name, plot_name)