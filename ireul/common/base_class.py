import numpy as np
import gym

class BaseAgent:
    def __init__(
            self,
            env: gym.Env,
            use_conv: bool=True,
            learning_rate: float=3e-4,
            gamma: float=0.99,
            tau: float=0.01,
            buffer_size: int=10000,
            eps_start: float=0.9,
            eps_end: float=0.05,
            eps_decay: float=200
        ):
        pass

    def get_action(self, state: np.ndarray, eps: float=0.20):
        raise NotImplementedError
    
    def update(self, batch_size: int) -> None:
        raise NotImplementedError
    
    def target_net_sync(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
