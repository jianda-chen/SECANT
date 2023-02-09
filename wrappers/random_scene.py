import gym
import random

class RandomScene(gym.Wrapper):
    def __init__(
        self,
        envs,
    ):
        """
        Args:
            mode: 'stack' creates a new dim, while 'concat' concatenates the leading dim
                e.g. stack([7,9]) twice -> [2, 7, 9]
                     concat([7,9]) twice -> [14, 9]
            stack_dim: which axis to stack
            include_keys: frame stack only the included keys, otherwise framestack all
        """
        assert type(envs) is list
        super().__init__(envs[0])
        self.envs_list = envs
        self.num_envs = len(envs)
        self.env = random.choice(self.envs_list)

        if hasattr(envs[0], "_max_episode_steps"):
            self._max_episode_steps = envs[0]._max_episode_steps
    
    def reset(self, **kwargs):
        self.env = random.choice(self.envs_list)
        
        obs = self.env.reset(**kwargs)
        return obs