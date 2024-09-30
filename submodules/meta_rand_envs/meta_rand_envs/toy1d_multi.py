import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import mujoco_py
import os
from gym.spaces import Box
import re
import tempfile


class Toy1dMultiTask(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.simple_env_dt = 0.01
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', '')
        self.change_prob = kwargs.get('change_prob', 0.0)
        self.change_steps = kwargs.get('change_steps', 80)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0

        # velocity/position specifications in ranges [from, to]
        # self.velocity_x = [0.5, 1.0]
        # self.pos_x = [0.5, 1]
        # self.velocity_y = [0.5, 1]
        # self.pos_y = [0.5, 1]
        # self.velocity_z = [0.5, 1.]

        self.simple_dt = 0.1
        self.x, self.v, self.a = 0,0,0

        self.velocity_x = [0.5, 3.0]
        self.pos_x = [1.0, 25.0]
        self.velocity_y = [2. * np.pi, 4. * np.pi]
        self.pos_y = [np.pi / 5., np.pi / 2.]
        self.velocity_z = [1.5, 3.]

        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.base_task = 0
        self.task_specification = 1.0
        task_names = ['velocity_forward', 'velocity_backward',
                      'goal_forward', 'goal_backward',
                      'flip_forward',
                      'stand_front', 'stand_back',
                      'jump',
                      'direction_forward', 'direction_backward',
                      'velocity']
        self.task_variants = kwargs.get('task_variants', task_names)
        self.bt2t = {k: self.task_variants.index(k) if k in self.task_variants else -1 for k in task_names}

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        self.model_path = os.path.join(os.getcwd(), 'submodules', 'meta_rand_envs', 'meta_rand_envs',
                                       'toy1d.xml')
        if 'env_init' in kwargs:
            parameters = dict(
                dt = kwargs['env_init']['dt'],
                gear = kwargs['env_init']['gear'])
            self.model_path = self.update_xml(self.model_path, parameters)
            skip_frames = kwargs['env_init']['skip_frames'] 
        else: skip_frames = 1

        observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), self.model_path, 5, observation_space=observation_space, 
                           hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=[], skip_frames=skip_frames)
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def update_xml(self, path, parameters):
        from lxml import etree
        with open(path, 'r') as file:
            xml_string = file.read()
        root = etree.fromstring(xml_string)

        # def replace_match(match):
        #     placeholder = match.group(0)
        #     key = match.group(1)
        #     default_value = match.group(2) if match.group(2) else ''
        #     return str(parameters.get(key, default_value))
        
        # pattern = r'\{(\w+)(?::([^}]*))?\}'
        # modified_file = re.sub(pattern, replace_match, xml_string)
        params = {'/mujoco/option': parameters['dt'],
                      '/mujoco/actuator/motor':parameters['gear']}
        for xpath, new_value in params.items():
            elements = root.xpath(xpath)
            for elem in elements:
                if 'gear' in elem.attrib:
                    elem.attrib['gear'] = str(new_value)
                if 'timestep' in elem.attrib:
                    elem.attrib['timestep'] = str(new_value)

        # Convert modified XML back to a string
        modified_xml_string = etree.tostring(root, pretty_print=True).decode()

        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        modified_xml_path = os.path.join(temp_dir, 'modified_model.xml')
        with open(modified_xml_path, 'w') as file:
            file.write(modified_xml_string)
        return modified_xml_path


    def _step(self, action, random=False):
        # change task after some steps
        if self.change_mode == "time" and not self.initialize:

            if 'current_step' not in self.tasks[self.last_idx].keys():
                self.tasks[self.last_idx]['current_step'] = 0
            self.tasks[self.last_idx]['current_step'] += 1

            if 'changed_task_spec' in self.tasks[self.last_idx].keys():
                self.change_task(self.tasks[self.last_idx]['changed_task_spec'])
            if self.tasks[self.last_idx]['current_step'] % self.change_steps == 0:
                task_spec = np.random.choice(self.train_tasks if self.meta_mode == 'train' else self.test_tasks)
                self.tasks[self.last_idx]['changed_task_spec'] = {
                    'base_task': task_spec['base_task'],
                    'specification': task_spec['specification'],
                    'color': task_spec['color']
                }
                self.change_task(self.tasks[self.last_idx]['changed_task_spec'])
                self.tasks[self.last_idx]['current_step'] = 0
        ob_before = self._get_obs()
        xposbefore = np.copy(self.sim.data.qpos)  # Assuming 'box' is the name of the body
        xvelbefore = np.copy(self.sim.data.qvel)
        
        # self.sim.data.set_joint_qpos('boxslideZ', (np.random.rand()+1.1)/2)
        # Calculate new velocities required for the position and angle changes
        # Limiting factor is the rotation, 
        scaled_action = np.copy(action)
        scaled_action[0] = scaled_action[0]*self.velocity_x[1]
        # scaled_action[1] = scaled_action[1]*self.velocity_x[1]
        # scaled_action[2] = scaled_action[2]*self.pos_y[1]
        # scaled_action[3] = scaled_action[3] *self.velocity_x[1]
        # scaled_action[4] = scaled_action[4] *self.velocity_y[1]
        # # action[0] = action[0]*self.velocity_x[1]
        # scaled_action[3] = scaled_action[3]*self.velocity_y[1]
        # action[1] = action[1]*self.velocity_z[1]
        # action[2] = action[2]*self.pos_y[1]
        # scaled_action[4] = scaled_action[4]*self.velocity_y[1]
        # action[3] = action[3]*self.velocity_x[1]
        # action[4] = action[4]*self.velocity_y[1]
        # # velocity = action / self.simple_env_dt
        # new_position_x = xposbefore[0] + scaled_action[0] * self.simple_env_dt
        new_pos_x = xposbefore[0] + scaled_action[0]*self.simple_env_dt
        # new_pos_y = xposbefore[1] + action[0]
        # new_pos_y = new_pos_y.clip(-self.pos_y[1], self.pos_y[1])
        # new_vel_x = xvelbefore[0] + action[1]

        # new_pos_x = np.array(new_pos_x).clip(-self.pos_x[1], self.pos_x[1])
        # new_vel_x = np.array(new_vel_x).clip(-self.velocity_x[1], self.velocity_x[1])
        # new_pos_y = xposbefore[2] + action[2]
        # new_vel_y = action[2] / self.simple_env_dt
        # new_vel_z = action[1] * self.velocity_z [1]
        # if xposbefore[1]>0:
        #     new_position [1] = 0
        #     velocity[1] = (new_position[1] - xposbefore[1])/self.simple_env_dt

        # Assume only one non zero entry
        # TODO: change environment such that sef.dt=1 -> change self.simple_env_dt to self.dt
        # velocity = scaled_action[[3,1,4]] / self.simple_env_dt
        # velocity = action[[3,1,4]]
        # new_position = xposbefore + action[:3]
        # new_position[[0,2]] += action[3:] * self.simple_env_dt
        # velocity = xvelbefore + action[[3,1,4]]
        # if np.abs(xposbefore[1])>0:
        #     new_position[1] = 0
        #     velocity[1] = (new_position[1] - xposbefore[1])/self.simple_env_dt
        #     scaled_action[1] = self.simple_env_dt * velocity[1]
        #     action[1] = scaled_action[1] / self.velocity_z[1]

        # new_position[0] = np.clip(new_position[0], -self.pos_x[1], self.pos_x[1])
        # new_position[2] = np.clip(new_position[2], -self.pos_y[1], self.pos_y[1])

       

        # # Make the simulation aware of the new state
        # self.sim.forward()
        def simulate(action):
            x = self.sim.data.qpos[0]
            v_x = self.sim.data.qvel[0]
            if action.shape[0]>1:
                y = self.sim.data.qpos[2]
                z = self.sim.data.qpos[1]
                v_y = self.sim.data.qvel[2]
                v_z = self.sim.data.qvel[1]
            else:
                y = 0
                z = 0
                v_y = 0
                v_z = 0
            v_x_max = 5
            v_y_max = 4. * np.pi
            v_z_max = 3
            f_x_max = 50
            f_y_max = 50
            f_z_max = v_z_max/self.simple_dt + 9.81*self.simple_dt
            f_x = f_x_max * action[0]
            if action.shape[0]>1:
                f_y = f_y_max * action[1]
                f_z = f_z_max * np.abs(action[2])
            else:
                f_y = 0
                f_z = 0
            if z>0.3 or v_z > 0:
                f_z = 0
            c_x = f_x_max/v_x_max
            c_y = f_y_max/v_y_max
            # c_z = f_z_max/v_z_max
            # self.do_simulation(action, self.frame_skip)
            a_x = (f_x - c_x * v_x)
            a_y = (f_y - c_y * v_y)
            a_z = (f_z - 9.81)

            # Update velocity and position
            v_x = v_x + a_x * self.simple_dt
            v_x = np.clip(v_x, -v_x_max, v_x_max)
            v_y = v_y + a_y * self.simple_dt
            v_y = np.clip(v_y, -v_y_max, v_y_max)
            v_z = v_z + a_z * self.simple_dt
            v_z = np.clip(v_z, -v_z_max, v_z_max)
            x = x + v_x * self.simple_dt
            y = y + v_y * self.simple_dt
            y = np.clip(y, -np.pi/2, np.pi/2)
            z = z + v_z * self.simple_dt
            if z<0 and v_z < 0:
                v_z = 0
            z = np.clip(z, 0, 4)
            return x,y,z,v_x,v_y,v_z
         
        x,y,z,v_x,v_y,v_z = simulate(action)

        random = False
        if random:
            var = 0.08
            x = np.random.normal(loc=x, scale=var, size=1)
            v_x = np.random.normal(loc=v_x, scale=var, size=1)
            y = np.random.normal(loc=y, scale=var, size=1)
            v_y = np.random.normal(loc=v_y, scale=var, size=1)
            z = np.random.normal(loc=z, scale=var, size=1)
            v_z = np.random.normal(loc=v_z, scale=var, size=1)
        # Set the new positions and angles
        self.sim.data.set_joint_qpos('boxslideX', x)
        if action.shape[0]>1:
            self.sim.data.set_joint_qpos('boxrotateY', y)
            self.sim.data.set_joint_qpos('boxslideZ', z)
        # self.sim.data.set_joint_qpos('boxslideZ', np.random.rand())
        # self.sim.data.set_joint_qpos('boxrotateY', new_pos_y) # Only for visulaization purposes (update bar)

        # Directly set the velocities to match the position and rotation changes
        self.sim.data.set_joint_qvel('boxslideX', v_x)
        if action.shape[0]>1:
            self.sim.data.set_joint_qvel('boxrotateY', v_y)
            self.sim.data.set_joint_qvel('boxslideZ', v_z)
        # self.sim.data.set_joint_qvel('boxslideZ', scaled_action[1])
        # self.sim.data.set_joint_qvel('boxrotateY', scaled_action[4])

        xposafter = np.copy(self.sim.data.qpos)
        xvelafter = np.copy(self.sim.data.qvel)
        # penalty = xposafter+xvelafter

        ob = self._get_obs()

        if self.base_task in [self.bt2t['velocity_forward'], self.bt2t['velocity_backward']]:  # 'velocity'
            reward_run = - np.abs(xvelafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
            # reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['goal_forward'], self.bt2t['goal_backward']]:  # 'goal'
            reward_run = - np.abs(xposafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
            # reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['flip_forward']]:  # 'flipping'
            reward_run = - np.abs(xvelafter[2] - self.task_specification)
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
            # reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['stand_front'], self.bt2t['stand_back']]:  # 'stand_up'
            reward_run = - np.abs(xposafter[2] - self.task_specification)
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
            # reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['jump']]:  # 'jump'
            reward_run = - np.abs(np.abs(xvelafter[1]) - self.task_specification)
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)
            # reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['direction_forward'], self.bt2t['direction_backward']]:  # 'direction'
            reward_run = (xposafter[0] - xposbefore[0]) / self.dt * self.task_specification
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['velocity']]:
            forward_vel = (xposafter[0] - xposbefore[0]) / self.dt
            reward_run = -1.0 * np.abs(forward_vel - self.task_specification)
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward_ctrl = 0
            # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(penalty))
            # reward_ctrl = -0.5 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
        else:
            raise RuntimeError("base task not recognized")

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=self.base_task, specification=self.task_specification))

    # from pearl rlkit
    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()
        # obs[1] = self.rotation
        # obs[4] = self.rotation_vel
        return obs

    def reset_model(self):
        # reset changepoint
        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        # reset tasks
        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self.recolor()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=1, size=self.model.nv) * .1
        self.rotation = 0
        self.rotation_vel = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_image(self, width=256, height=256, camera_name=None):
        if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self.viewer_setup()
            self._viewers['rgb_array'] = self.viewer

        # use sim.render to avoid MJViewer which doesn't seem to work without display
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def viewer_setup(self):
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = 0

    def change_task(self, spec):
        self.base_task = spec['base_task']
        self.task_specification = spec['specification']
        self._goal = spec['specification']
        self.color = spec['color']
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value = self.color
        geom_rgba[1:, :3] = np.asarray(rgb_value)
        self.model.geom_rgba[:] = geom_rgba

    def sample_tasks(self, num_tasks_list):
        if type(num_tasks_list) != list: num_tasks_list = [num_tasks_list]

        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = [int(num_tasks / num_base_tasks) for num_tasks in num_tasks_list]
        num_tasks_per_subtask_cumsum = np.cumsum(num_tasks_per_subtask)

        tasks = [[] for _ in range(len(num_tasks_list))]
        # velocity tasks
        if 'velocity_forward' in self.task_variants:
            velocities = np.linspace(self.velocity_x[0], self.velocity_x[1], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_forward'], 'specification': velocity, 'color': np.array([1, 0, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'velocity_backward' in self.task_variants:
            velocities = np.linspace(-self.velocity_x[1], -self.velocity_x[0], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_backward'], 'specification': velocity, 'color': np.array([0, 1, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # goal
        if 'goal_forward' in self.task_variants:
            goals = np.linspace(self.pos_x[0], self.pos_x[1], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_forward'], 'specification': goal, 'color': np.array([1, 1, 0])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'goal_backward' in self.task_variants:
            goals = np.linspace(-self.pos_x[1], -self.pos_x[0], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_backward'], 'specification': goal, 'color': np.array([0, 1, 1])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # flipping
        if 'flip_forward' in self.task_variants:
            goals = np.linspace(self.velocity_y[0], self.velocity_y[1], num=sum(num_tasks_per_subtask))
            tasks_flipping = [
                {'base_task': self.bt2t['flip_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0])} for
                goal in goals]
            np.random.shuffle(tasks_flipping)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_flipping[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # stand_up
        if 'stand_front' in self.task_variants:
            goals = np.linspace(self.pos_y[0], self.pos_y[1], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_front'], 'specification': goal, 'color': np.array([1., 0, 0.5])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'stand_back' in self.task_variants:
            goals = np.linspace(-self.pos_y[1], -self.pos_y[0], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_back'], 'specification': goal, 'color': np.array([0.5, 0, 1.])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # jump
        if 'jump' in self.task_variants:
            goals = np.linspace(self.velocity_z[0], self.velocity_z[1], num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['jump'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # direction
        if 'direction_forward' in self.task_variants:
            goals = np.array([1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'direction_backward' in self.task_variants:
            goals = np.array([-1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_backward'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'velocity' in self.task_variants:
            goals = np.linspace(0.0, 3.0, num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['velocity'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # Return nested list only if list is given as input
        return tasks if len(num_tasks_list) > 1 else tasks[0]

    def set_meta_mode(self, mode):
        self.meta_mode = mode
