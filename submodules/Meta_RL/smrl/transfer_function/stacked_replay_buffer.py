import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from tqdm import tqdm
import ray
import copy


class StackedReplayBuffer:
    def __init__(self,
                 env,
                 agent,
                 remote=False,
                 memory_size = 1e+6,
                 sample_until_terminal=False,
                 max_path_length=100,
                 random_change_task=None,
                 observation_dim = 20,
                 action_dim = 6,
                 task_dim = 1,
                 permute_samples = False,
                 sampling_mode=None):
        self.results = dict(
            observations=[],
            rewards=[],
            tasks=[],
            actions=[],
            next_observations=[], 
            terminals=[],
            log_probs=[]
        )
        self.memory = dict(
            observations=[],
            rewards=[],
            tasks=[],
            actions=[],
            next_observations=[], 
            terminals=[]
        )
        self.env = env
        self.agent = agent
        self.max_path_length = max_path_length
        self.remote = remote
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_dim = task_dim
        self.memory_size = memory_size
        self.stats = None
        self.sample_until_terminal = sample_until_terminal
        self.random_change_task = random_change_task
        if self.random_change_task is not None:
            self.random_change_task_prob = random_change_task['prob']
            self.random_change_task_factor = random_change_task['factor']

    def store(self, obs, reward, done, action, next_obs, task, log_probs):
        self.results['observations'].append(obs)
        self.results['rewards'].append(np.array([reward]))
        self.results['terminals'].append(np.array([done]))
        self.results['actions'].append(action.detach().cpu().numpy())
        self.results['next_observations'].append(next_obs)
        self.results['tasks'].append(np.array([task]))
        self.results['log_probs'].append(log_probs.detach().cpu().numpy())
        if len(self.results['observations']) > self.memory_size:
            self.results['observations'].pop(0)
            self.results['rewards'].pop(0)
            self.results['terminals'].pop(0)
            self.results['actions'].pop(0)
            self.results['next_observations'].pop(0)
            self.results['tasks'].pop(0)
            self.results['log_probs'].pop(0)
        assert len(self.results['observations']) <= self.memory_size

    def fill_buffer(self, tasks):
        print('Collecting paths for different tasks')
        if self.remote:
            
            path_tasks = []
            for task in tasks:
                path_tasks.append(rollout_remote.remote(
                    task,
                    copy.deepcopy(self.env),
                    self.agent,
                    self.max_path_length,
                ))
            while len(path_tasks) > 0:
                ready, path_tasks = ray.wait(path_tasks)
                res = copy.deepcopy(ray.get(ready)) 
                self.results['observations'].append(res['observations'])
                self.results['tasks'].append(res['tasks'])
                self.results['rewards'].append(res['rewards'])
                self.results['terminals'].append(res['terminals'])
                self.results['actions'].append(res['actions'])
                self.results['next_observations'].append(res['next_results'])
        else:
            for index, task in tqdm(enumerate(tasks), total=len(tasks)):
                self.rollout(task, self.max_path_length)
            # self.stats = self.get_stats_dict()
            
    def get_stats_dict(self):
        stats_dict = dict(
            observations={},
            next_observations={},
            actions={},
            rewards={},
        )
        for key in stats_dict.keys():
            stats_dict[key]["max"] = self.results[key].max(axis=0)
            stats_dict[key]["min"] = self.results[key].min(axis=0)
            stats_dict[key]["mean"] = self.results[key].mean(axis=0)
            stats_dict[key]["std"] = self.results[key].std(axis=0)
        return stats_dict
    
    def normalize(self):
        for key in self.stats.keys():
            self.results[key] = (self.results[key] - self.stats[key]["mean"]) / (self.stats[key]["std"] + 1e-8)


    def delete_buffer(self):
        self.results = dict(
            observations=[],
            rewards=[],
            tasks=[],
            actions=[],
            next_observations=[], 
            terminals=[]
        )

    def rollout(self, task, max_path_length):
        obs = self.env.reset()[0]
        self.env.update_task(task)
        for j in range(max_path_length):
            action = self.agent.get_action(ptu.from_numpy(obs).cuda(), ptu.from_numpy(np.array(task)).cuda(), return_dist=False)
            result = self.env.step(action.detach().cpu().numpy())
            term = result[2]
            if j == max_path_length-1:
                term = True
            self.results['observations'].append(obs)
            self.results['tasks'].append(np.array(task))
            self.results['rewards'].append(np.array(result[1]))
            self.results['terminals'].append(term)
            self.results['actions'].append(action.detach().cpu().numpy())
            self.results['next_observations'].append(result[0])
            obs = result[0]
            # if j == 0:
            #     self.results['observations'].append(obs)
            #     self.results['tasks'].append(np.array(task))
            #     self.results['rewards'].append(np.array(result[1]))
            #     self.results['terminals'].append(result[2])
            #     self.results['actions'].append(action.detach().cpu().numpy())
            #     self.results['next_observations'].append(result[0])
            #     obs = result[0]
            # else:
            #     self.results['observations'][i] = np.append(self.results['observations'][i], obs)
            #     self.results['tasks'][i] = np.append(self.results['tasks'][i], np.array(task))
            #     self.results['rewards'][i] = np.append(self.results['rewards'][i], np.array(result[1]))
            #     self.results['terminals'][i] = np.append(self.results['terminals'][i], result[2])
            #     self.results['actions'][i] = np.append(self.results['actions'][i], action.detach().cpu().numpy())
            #     self.results['next_observations'][i] = np.append(self.results['next_observations'][i], result[0])
            #     obs = result[0]
            if j == self.max_path_length//2:
                print('mitad')
            if j == self.max_path_length//2:
                if self.random_change_task is not None:
                    change = True if np.random.rand() < self.random_change_task_prob else False
                    if change == True:
                        task = task + (np.random.rand()*self.random_change_task_factor*2 - self.random_change_task_factor)
                        self.env.update_task(task)
            if result[2] == 1:
                return
            
    def sample_batch(self, batch_size, max_traj_len):

        # Sample indices
        rng = np.random.default_rng()
        # points = len(self.results['observations'])
        # num_indices = 0
        # indices = []
        # while num_indices<batch_size:
        #     row = np.random.randint(0, points)
        #     if np.size(self.results['observations'][row])>0:
        #         column = np.random.randint(0, np.size(self.results['observations'][row])//self.observation_dim)
        #         lower_limit = column - max_traj_len
        #         if num_indices == 0:
        #             indices = np.array([row, lower_limit, column])[None,:]
        #         else:
        #             a = np.array([row, lower_limit, column])
        #             indices = np.concatenate((indices, a[None, :]), axis = 0)
        #         num_indices+=1

        
        # indices[:, 1] = np.clip(indices[:, 1], 0, np.inf)
        points = np.arange(len(self.results['observations']))
        if len(points) < batch_size: batch_size = len(points)
        indices = rng.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False)
        
        if self.sample_until_terminal:
            num_added = 0
            append_list = np.array([], dtype=int)
            for j, idx in enumerate(indices):
                # Remove all duplicate elements

                i = 0
                while self.results['terminals'][idx+i] != True and i<100 and idx+i+1<len(self.results['terminals']):
                    i+=1
                if i>1:
                    add = np.arange(i) + idx + 1
                    append_list = np.concatenate([append_list, add], dtype=int)
            indices = np.concatenate([indices, append_list])
            # unique_elements, unique_indices = np.unique(indices, return_index=True)
            # indices = indices[np.sort(unique_indices)]
        
        # all_indices = indices[:, None] + np.arange(-self.time_steps, 1)[None, :]
        # for key in self.results.keys():
        #     self.results[key] = np.array(self.results[key])
            
        # Take only indices bigger equak than 0
        # return  dict(observations=[self.results['observations'][row_idx][start_idx*self.observation_dim:end_idx*self.observation_dim] 
        #                                     for row_idx, start_idx, end_idx in indices],
        #              rewards=[self.results['rewards'][row_idx][start_idx:end_idx] 
        #                                for row_idx, start_idx, end_idx in indices],
        #              actions=[self.results['actions'][row_idx][start_idx*self.action_dim:end_idx*self.action_dim] 
        #                                for row_idx, start_idx, end_idx in indices],
        #              next_observations=[self.results['next_observations'][row_idx][start_idx*self.observation_dim:end_idx*self.observation_dim] 
        #                                          for row_idx, start_idx, end_idx in indices],
        #              terminals=[self.results['terminals'][row_idx][start_idx] 
        #                                  for row_idx, start_idx, end_idx in indices], 
        #              tasks=[self.results['tasks'][row_idx][start_idx] 
        #                              for row_idx, start_idx, end_idx in indices]
        #              )
        if 'log_probs' in self.results:
            return dict(observations=np.array(self.results['observations'])[indices],
                        rewards=np.array(self.results['rewards'])[indices],
                        actions=np.array(self.results['actions'])[indices],
                        next_observations=np.array(self.results['next_observations'])[indices],
                        terminals=np.array(self.results['terminals'])[indices], 
                        tasks=np.array(self.results['tasks'])[indices],
                        log_probs=np.array(self.results['log_probs'])[indices],
                        )

        return dict(observations=np.array(self.results['observations'])[indices],
                     rewards=np.array(self.results['rewards'])[indices],
                     actions=np.array(self.results['actions'])[indices],
                     next_observations=np.array(self.results['next_observations'])[indices],
                     terminals=np.array(self.results['terminals'])[indices], 
                     tasks=np.array(self.results['tasks'])[indices]
                     )

@ray.remote
def rollout_remote(task, env, agent, max_path_length):
    results = dict(
        observations=[],
        rewards=[],
        tasks=[],
        actions=[],
        next_observations=[], 
        terminals=[]
    )
    env.update_task(task)
    obs = env.reset()[0]
    for i in range(max_path_length):
        action = agent.get_action(ptu.from_numpy(obs).cuda(), ptu.from_numpy(np.array(task)).cuda())
        result = env.step(action.detach().cpu().numpy())
        results['observations'].append(obs.cpu())
        results['tasks'].append(task.cpu())
        results['rewards'].append(result[1].cpu())
        results['terminals'].append(result[2].cpu())
        results['actions'].append(action.detach().cpu().numpy())
        results['next_observations'].append(result[0].cpu())
        obs = result[0]
        if result[2] == 1:
            return
        
    return results