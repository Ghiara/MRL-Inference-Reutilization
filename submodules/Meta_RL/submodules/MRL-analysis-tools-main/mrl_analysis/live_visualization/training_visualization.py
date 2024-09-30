from collections import deque
import numpy as np
import pandas as pd
import time
from typing import List, Any, Dict
import matplotlib.pyplot as plt

from smrl.experiment.analysis import load_results

from mrl_analysis.live_visualization.plots import Plot
from mrl_analysis.utility import interfaces


plt.ion()


class TrainVisualizer():
    def __init__(
        self,
        plots: List[Plot],
        log_path: str = "./data",
    ) -> None:

        self.log_path = log_path
        self.plots = plots

        self.last_update_time = time.time()
        self.last_env_reset_time = time.time()

        self.progress: pd.DataFrame
        self.env: interfaces.MetaEnv
        self.encoder: interfaces.MdpEncoder
        self.decoder: interfaces.MdpDecoder
        self.policy: interfaces.MetaRLPolicy
        self.variant: Dict[str, Any]

        self._load_results()
        self._update_plots()

        # Context data initialization
        self.observations = deque(
            [np.zeros((self.decoder.observation_dim)) for _ in range(self.encoder.context_size)],
            maxlen=self.encoder.context_size
        )
        self.actions = deque(
            [np.zeros((self.decoder.action_dim)) for _ in range(self.encoder.context_size)],
            maxlen=self.encoder.context_size
        )
        self.rewards = deque(
            [np.zeros((1)) for _ in range(self.encoder.context_size)],
            maxlen=self.encoder.context_size
        )
        self.next_observations = deque(
            [np.zeros((self.decoder.observation_dim)) for _ in range(self.encoder.context_size)],
            maxlen=self.encoder.context_size
        )
        self.terminals = deque(
            [np.zeros((1)) for _ in range(self.encoder.context_size)],
            maxlen=self.encoder.context_size
        )
        

    def _load_results(self):
        results = load_results(self.log_path, itr=None)

        self.progress = results['progress']
        self.env = results['env']
        self.encoder = results['encoder']
        self.decoder = results['decoder']
        self.policy = results['policy']
        self.variant = results['config']

        self.last_update_time = time.time()

    def _update_plots(self):
        for plot in self.plots:
            plot.update(self.progress)

    def visualize(self, update_every_seconds: float = 60, env_reset_interval: float = 15):
        obs = self.env.reset()
        while True:
            if time.time() - self.last_update_time > update_every_seconds:
                self._load_results()
                self._update_plots()
                print("Fetched data!")
            
            latent = self.encoder.get_encoding(
                np.array(self.observations),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.next_observations),
                np.array(self.terminals)
            )
            # latent = self.env.task['goal']    # OPTION: Can be used for debugging

            action, _ = self.policy.get_action(obs, latent, mode='mean')
            next_obs, reward, terminal, info = self.env.step(action)

            try:
                image = self.env.render()
            except SystemExit:
                break

            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(np.array([reward]))
            self.next_observations.append(next_obs)
            self.terminals.append(np.array([terminal]))
            obs = next_obs

            if time.time() - self.last_env_reset_time > env_reset_interval:
                obs = self.env.reset()
                self.last_env_reset_time = time.time()
                print("Environment reset!")



if __name__ == "__main__":
    plots = [
        Plot('Epoch', ['eval/Average Returns', 'eval/Returns Min', 'eval/Returns Max'], title = "Returns", x_label="Epoch", y_label=["Average returns", "Minimal returns", "Maximal returns"]),
        Plot('Epoch', ['trainer/Policy trainer/QF1 Loss', 'trainer/Policy trainer/QF2 Loss'], title = "Q function losses", x_label="Epoch", y_label=["Qf1 loss", "Qf2 loss"]),
        Plot('Epoch', 'trainer/Policy trainer/Policy Loss', title = "Policy loss", x_label="Epoch", y_label="Policy loss"),
        Plot('Epoch', 'trainer/Inference trainer/elbo', title="ELBO", x_label="Epoch", y_label="ELBO"),
        Plot('trainer/Inference trainer/num train calls', ['trainer/Inference trainer/reward prediction error', 'trainer/Inference trainer/state prediction error'], x_label="Num train calls", y_label=["Reward prediction error", "State prediction error"])
    ]
    visualizer = TrainVisualizer(plots, log_path="/home/julius/code/Symmetric-Meta-Reinforcement-Learning/data/Toy1D_Base-config_2023-01-03_20-43-57",)
    visualizer.visualize(update_every_seconds=20, env_reset_interval=5)
