import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(obs)

    def noisy_action(self, obs):
        logits = self.architecture.architecture(obs)
        actions, _ = self.distribution.sample(logits)
        return actions

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, init_scale, device='cpu'):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn
        self.device = device

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [init_scale]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(init_scale)

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(init_scale)

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def sample(self, obs):

        return self.forward(obs)

    def forward(self, obs):

        return self.architecture(obs)

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.distribution = MultivariateNormal(logits, covariance)

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(logits, covariance)

        actions_log_prob = distribution.log_prob(outputs)
        entropy = distribution.entropy()

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()