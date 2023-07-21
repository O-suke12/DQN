import gym
from model import DQN
import torch
from torch import optim
from replay import ReplayMemory
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GANNMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LEARNING_RATE = 1e-4


environment_name = "CartPole-v0"
env = gym.make(environment_name)
n_actions = env.action_space.n
state = env.reset() #[Position, Velocity, Angle, Angular velocity]
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(10000)



steps_done = 0
def select_action(state):
  global steps_done
  sample = random.random()
  #As learning progresses, eps_thersold decreaseswhich leads to less greedy algorithm.
  eps_theresold = EPS_END + (EPS_START-EPS_END) * math.exp(-1.*steps_done/ EPS_DECAY) 
  steps_done += 1
  if sample > eps_theresold:
    with troch.no_grad():
      return policy_net(state).max(1)[1].view(1,1)
  else: 
    return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)



episode_durations = []
def plot_durations(show_result=False):
  plt.figure(1)
  duration_t = torch.tensor(episode_durations, dtype=torch.float)
  if show_result:
    plt.title("Result")
  else: 
    plt.clf()
    plt.title("Training")
  plt.xlabel("Episode")
  plt.ylabel("Duration")
  plt.plot(duration_t.numpy())

  if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

  plt.pause(0.001)  # pause a bit so that plots are updated
  if is_ipython:
      if not show_result:
          display.display(plt.gcf())
          display.clear_output(wait=True)
      else:
          display.display(plt.gcf())

def main():
  state = torch.tensor(state)
  print(policy_net(state).max(1))
  

if __name__ == "__main__":
  main()