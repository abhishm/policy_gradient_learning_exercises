import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import defaultdict


def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)

  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
  config['t'] += 1
  mhat = config['m']/(1-config['beta1']**config['t'])
  vhat = config['v']/(1-config['beta2']**config['t'])
  next_x = x + config['learning_rate']*mhat/(np.sqrt(vhat) + config['epsilon'])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return next_x, config

def one_hot_encoding(env, present_state):
    encoding = np.zeros(env.observation_space.n)
    encoding[present_state] = 1.0
    return encoding

class LinearFeatureBaseline_rllab(object):
    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self._coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + self._reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(returns)
        )[0]

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)

def do_episode(env, policy, max_pathlength, discount, path_baseline):
    """
    Simulate the env and policy for max_pathlength steps
    """
    present_state = env.reset()

    obs = []
    actions = []
    rewards = []
    is_Discrete = isinstance(env.observation_space, Discrete)

    for _ in xrange(max_pathlength):
        # One hot encoding the observations if the observation space is Discrete
        if is_Discrete:
            present_state = one_hot_encoding(env, present_state)
        action = policy.act(present_state)
        next_state, reward, done, _ = env.step(action)

        obs.append(present_state)
        actions.append(action)
        rewards.append(reward)

        present_state = next_state

        if done:
            break
    if path_baseline:
        baseline = path_baseline.predict({"observations" : np.array(obs), "actions" : np.array(actions), "rewards" : np.array(rewards)})
        advantages = []

    returns = []
    return_so_far = 0
    for t in xrange(len(rewards) - 1, -1, -1):
        return_so_far = rewards[t] + discount * return_so_far
        returns.append(return_so_far)
        if path_baseline:
            advantage = return_so_far - baseline[t]
            advantages.append(advantage)
    # The returns and advantages are stored backwards in time, so we need to revert it
    returns = returns[::-1]
    if path_baseline:
        advantages = advantages[::-1]

    if path_baseline:
        return {"observations" : np.array(obs), "actions" : np.array(actions), "rewards" : np.array(rewards),
                "returns" : np.array(returns), "advantages" : advantages }
    else:
        return {"observations" : np.array(obs), "actions" : np.array(actions), "rewards" : np.array(rewards),
                "returns" : np.array(returns)}

def rollouts(env, policy, max_pathlength, number_of_paths, discount, path_baseline):
    paths = []
    for _ in range(number_of_paths):
        path = do_episode(env, policy, max_pathlength, discount, path_baseline)
        paths.append(path)
    if path_baseline:
        path_baseline.fit(paths)
    return paths

class StochasticLinearPolicy():

    def __init__(self, ob_dim, ac_dim, **userconfig):
        """
        ob_space: observation space
        ac_space: action space
        """
        self.n_state = ob_dim
        self.n_action = ac_dim
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            }
        self.W = self.config["init_std"] * np.random.randn(self.n_state, self.n_action) + self.config["init_mean"]
        self.b = self.config["init_std"] * np.random.randn(1, self.n_action) + self.config["init_mean"]

    def softmax_prob(self, f_na):
        """
        Exponentiate f_na and normalize rows to have sum 1
        so each row gives a probability distribution over discrete
        action set
        """
        prob_nk = np.exp(f_na - f_na.max(axis=1,keepdims=True))
        prob_nk /= prob_nk.sum(axis=1,keepdims=True)
        return prob_nk

    def cat_sample(self, prob_nk):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        assert np.allclose(prob_nk.sum(axis=1,keepdims=True),1)
        N = prob_nk.shape[0]
        csprob_nk = np.cumsum(prob_nk, axis=1)
        out = np.zeros(N, dtype='i')
        for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
            for (k,csprob) in enumerate(csprob_k):
                if csprob > r:
                    out[n] = k
                    break
        return out

    def act(self, ob):
        """
        softmax probability
        """
        y = ob.dot(self.W) + self.b
        prob_nk = self.softmax_prob(y)
        acts_n = self.cat_sample(prob_nk)
        return acts_n[0]

    def softmax_prob_grad(self, observations, actions, returns):
        """
        return the gradient with respect to parameters
        """

        N = observations.shape[0]
        y = observations.dot(self.W) + self.b
        prob_nk = self.softmax_prob(y)

        grads = {}
        prob_grad = np.zeros((N, self.n_action))
        prob_grad[np.arange(N), actions] = 1
        prob_grad -= prob_nk

        grads["b"] = returns.dot(prob_grad)/float(N)
        grads["W"] = np.dot(observations.T, prob_grad*returns.reshape(-1,1))/float(N)

        return grads

class VPG(object):
    def __init__(self, env, policy, baseline, **userconfig):
        self.env = env
        self.policy = policy
        self.path_baseline = baseline
        self.config = {
            "learning_rate" : 0.1,
            "discount": 0.95,
            "batch_size": 100,             # Number of episodes
            "max_pathlength": 10000,       # Maximum number of transitions of states in a path
            "n_itr": 10,                   # number of updates
            "gradient_update": "sgd",      # gradient update formula
        }
        self.config.update(userconfig)


    def train(self):
        """
        train the policy
        """
        config = self.config
        training_reward = []
        for _ in xrange(config["n_itr"]):
            paths = rollouts(self.env, self.policy, config["max_pathlength"], config["batch_size"],
                             config["discount"], self.path_baseline)
            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            rewards = np.concatenate([p["rewards"] for p in paths])

            if self.path_baseline:
                advantages = np.concatenate([p["advantages"] for p in paths])
                advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-8)
                grads = self.policy.softmax_prob_grad(observations, actions, advantages)
            else:
                returns = np.concatenate([p["returns"] for p in paths])
                returns = (returns - np.mean(returns))/(np.std(returns) + 1e-8)
                grads = self.policy.softmax_prob_grad(observations, actions, returns)


            if config["gradient_update"] == "sgd":
                self.policy.W += config["learning_rate"]*grads["W"]
                self.policy.b += config["learning_rate"]*grads["b"]
            elif config["gradient_update"] == "adam":
                x = np.concatenate([self.policy.W.flatten(), self.policy.b.flatten()])
                dx = np.concatenate([grads["W"].flatten(), grads["b"].flatten()])
                next_x, self.config = adam(x, dx, self.config)
                self.policy.W = next_x[:self.policy.W.size].reshape(self.policy.W.shape)
                self.policy.b = next_x[self.policy.W.size:].reshape(1,-1)

            training_reward.append(np.sum(rewards)/float(config["batch_size"]))
        return training_reward

    def accuracy(self, n_trials):
        """
        compare the reward obtained by the policy
        """
        config = self.config
        paths = rollouts(self.env, self.policy, config["max_pathlength"], n_trials, config["discount"], self.path_baseline)
        mean_reward = np.mean([sum(p["rewards"]) for p in paths])
        return "The average reward for the policy is {0} in {1} trials".format(mean_reward, n_trials)

    def plot_episode(self, num_steps, render=True):
        """
        plot an episode
        """
        total_rew = 0
        frames = []
        ob = self.env.reset()
        for t in range(num_steps):
            a = self.policy.act(ob)
            (ob, reward, done, _info) = self.env.step(a)
            total_rew += reward
            if render and t%3==0: frames.append(self.env.render(mode = 'rgb_array'))
            if done: break
        self.env.render(close=True)
        return frames, total_rew
