import numpy as np
from submodules.meta_rand_envs.meta_rand_envs.toy2d_multi import Toy2dMultiTask
from gym import utils

from . import register_env


@register_env('toy2d-multi-task')
class Toy2dMultiTaskkWrappedEnv(Toy2dMultiTask):
    def __init__(self, *args, **kwargs):
        super(Toy2dMultiTaskkWrappedEnv, self).__init__(*args, **kwargs)
        self.train_tasks, self.test_tasks = self.sample_tasks([kwargs.get('n_train_tasks', 80), kwargs.get('n_eval_tasks', 40)])
        self.tasks = self.train_tasks + self.test_tasks

        self.name2number = {k : i for i, k in enumerate(self.task_variants)}

        # TODO: Decide if we want to have dynamically changing tasks
        self.change_mode = ''

        self.last_idx = None
        self.env_buffer = {}

        self.reset_task(0)

        utils.EzPickle.__init__(self, *args, **kwargs)


    def reset_task(self, idx, keep_buffered=False):

        self.last_idx = idx

        self._task = self.tasks[int(idx)]

        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self._goal = self._task['specification']
        self.color = self._task['color']

        # self.recolor()
        self.reset()

        if keep_buffered: self.env_buffer[idx] = self.sim.get_state()

        return self._get_obs()

    def set_task(self, idx):
        assert idx in self.env_buffer.keys()

        # TODO: In case of dynamic environments, the new task has to be saved as well
        if self.last_idx is not None: self.env_buffer[self.last_idx] = self.sim.get_state()

        self._task = self.tasks[int(idx)]

        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self._goal = self._task['specification']
        self.color = self._task['color']
        self.recolor()

        self.sim.reset()
        self.sim.set_state(self.env_buffer[idx])
        self.sim.forward()
        self.last_idx = idx


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def clear_buffer(self):
        self.env_buffer = {}