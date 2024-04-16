import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils import ReplayBuffer, get_env, run_episode

torch.autograd.set_detect_anomaly(True)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _activation_function(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class NeuralNetwork(nn.Module):
    """
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        hidden_layers: int,
        activation: str,
        final_activation: str,
    ):
        super(NeuralNetwork, self).__init__()

        activation_fun = _activation_function(activation)
        activation_fun_final = _activation_function(final_activation)

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation_fun)

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fun)

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
        layers.append(activation_fun_final)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, hidden_layers):
        super(QNetwork, self).__init__()

        self.network = NeuralNetwork(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation="relu",
            final_activation="identity",
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, hidden_layers):
        super(ActorNetwork, self).__init__()

        self.network = NeuralNetwork(
            input_dim=state_dim,
            output_dim=hidden_size,
            activation="relu",
            final_activation="relu",
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )
        self.output_mean = nn.Linear(hidden_size, action_dim)
        self.output_log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.network(state)
        return self.output_mean(x), self.output_log_std(x)


class Actor:
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        actor_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        self.network = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        """
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the action.
        """
        assert state.shape == (3,) or state.shape[1] == self.state_dim, (
            "State passed to this method has a wrong " "shape"
        )

        # Get policy distribution parameters
        # TODO Check this later
        mean, log_std = self.network(state)
        std = self.clamp_log_std(log_std).exp()
        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            action = F.tanh(mean)
            log_prob = torch.zeros((state.shape[0], self.action_dim))
        else:
            sample = normal.rsample()
            action = F.tanh(sample)
            # Reference: Original paper: https://arxiv.org/pdf/1801.01290.pdf (Appendix C)
            log_prob = normal.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)

        assert action.shape == (
            state.shape[0],
            self.action_dim,
        ), "Incorrect shape for action"
        assert log_prob.shape == (
            state.shape[0],
            self.action_dim,
        ), "Incorrect shape for log_prob."

        return action, log_prob


class Critic:
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        critic_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.base = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.target = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.target.load_state_dict(self.base.state_dict())

        self.optimizer = torch.optim.Adam(self.base.parameters(), lr=self.critic_lr)


class TrainableParameter:
    """
    This class could be used to define a trainable parameter in your method. You could find it
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    """

    def __init__(
        self,
        init_param: float,
        lr_param: float,
        train_param: bool,
        device: torch.device = torch.device("cpu"),
    ):
        self.log_param = torch.tensor(
            np.log(init_param), requires_grad=train_param, device=device
        )
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param

    def is_trainable(self) -> bool:
        return self.log_param.requires_grad


class Agent:
    def __init__(
        self,
        init_alpha: float = 0.2158,
        train_alpha: bool = True,
        lr: float = 0.003036,
        hidden_size: int = 256,
        hidden_layers_actor: int = 2,
        hidden_layers_critic: int = 2,
    ):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 1000000

        self.memory = ReplayBuffer(
            self.min_buffer_size, self.max_buffer_size, self.device
        )

        self.discount_factor = 0.99
        self.tau = 0.005

        # Reference: https://arxiv.org/pdf/1812.05905.pdf
        self.target_alpha = -np.prod(self.action_dim).item()
        self.alpha = TrainableParameter(
            init_param=init_alpha,
            lr_param=3e-4,
            train_param=train_alpha,
            device=self.device,
        )

        self.actor = Actor(
            hidden_size=hidden_size,
            hidden_layers=hidden_layers_actor,
            actor_lr=lr,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        q1 = Critic(
            hidden_size=hidden_size,
            hidden_layers=hidden_layers_critic,
            critic_lr=lr,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        q2 = Critic(
            hidden_size=hidden_size,
            hidden_layers=hidden_layers_critic,
            critic_lr=lr,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.qs = [q1, q2]

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode.
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        state = torch.FloatTensor(s.reshape(1, -1)).to(self.device)

        # Sample actions based on the policy network
        action, _ = self.actor.get_action_and_log_prob(state, deterministic=False)

        # Convert output action to the numpy array
        action = action.cpu().data.numpy().reshape(-1)

        assert action.shape == (self.action_dim,), "Incorrect action shape."
        assert isinstance(action, np.ndarray), "Action dtype must be np.ndarray"
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        """
        This function takes in a object containing trainable parameters and an optimizer,
        and using a given loss, runs one step of gradient update. If you set up trainable parameters
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        """
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(
        self,
        base_net: NeuralNetwork,
        target_net: NeuralNetwork,
        tau: float,
        soft_update: bool,
    ):
        """
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        """
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(
                    param_target.data * (1.0 - tau) + param.data * tau
                )
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        """
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer, and then updates the policy and critic networks
        using the sampled batch.
        """
        batch = self.memory.sample(self.batch_size)
        state, action, reward, next_state = batch
        state, action, reward, next_state = (
            state.to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_state.to(self.device),
        )

        alpha_value = self.alpha.get_param().detach()

        with torch.no_grad():
            next_action, next_state_log_prob = self.actor.get_action_and_log_prob(
                next_state, deterministic=False
            )
            next_q_values = [q.target(next_state, next_action) for q in self.qs]
            q_value_target = reward + self.discount_factor * (
                torch.min(*next_q_values) - alpha_value * next_state_log_prob
            )

        # Train the critic
        for q in self.qs:
            q_value_pred = q.base(state, action)
            q_loss = F.mse_loss(q_value_pred, target=q_value_target)
            self.run_gradient_update_step(q, q_loss)

        # Train the policy
        alt_action, alt_action_log_prob = self.actor.get_action_and_log_prob(
            state, deterministic=False
        )
        q_values = [q.base(state, alt_action) for q in self.qs]
        actor_loss = (torch.min(*q_values) - alpha_value * alt_action_log_prob).mean()
        # Perform gradient ascent
        self.run_gradient_update_step(self.actor, -1 * actor_loss)

        if self.alpha.is_trainable():
            alpha_value = self.alpha.get_param()
            alpha_loss = -(
                alpha_value * (alt_action_log_prob + self.target_alpha).detach()
            ).mean()
            self.alpha.optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha.optimizer.step()

        with torch.no_grad():
            for q in self.qs:
                self.critic_target_update(
                    base_net=q.base.network,
                    target_net=q.target.network,
                    tau=self.tau,
                    soft_update=True,
                )


# This main function is provided here to enable some basic testing.
# ANY changes here WON'T take any effect while grading.
if __name__ == "__main__":
    import wandb

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    wandb.init(
        project="pai-task4",
        entity="eugleo",
        config={
            "init_alpha": 0.21582,
            "train_alpha": True,
            "lr": 0.003,
            "hidden_size": 256,
            "hidden_layers_actor": 2,
            "hidden_layers_critic": 2,
        },
    )

    # You may set the save_video param to output the video of one of the evalution episodes, or
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent(
        init_alpha=wandb.config.init_alpha,
        train_alpha=wandb.config.train_alpha,
        lr=wandb.config.lr,
        hidden_size=wandb.config.hidden_size,
        hidden_layers_actor=wandb.config.hidden_layers_actor,
        hidden_layers_critic=wandb.config.hidden_layers_critic,
    )
    env = get_env(g=10.0, train=True)

    train_returns = []
    for EP in range(TRAIN_EPISODES):
        episode_return = run_episode(env, agent, None, verbose, train=True)
        train_returns.append(episode_return)
        wandb.log({"train/return": episode_return})
        if EP > 25 and sum(train_returns[-5:]) / 5 < -1400:
            break

    if verbose:
        print("\n")

    test_returns = []
    env = get_env(g=10.0, train=False)

    video_rec = None
    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
            wandb.log({"test/return": episode_return})
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))
    wandb.summary["avg_test_return"] = avg_test_return

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if video_rec is not None:
        video_rec.close()
