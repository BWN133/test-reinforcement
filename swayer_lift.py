import numpy as np
import torch
import robosuite as suite
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
#TODO: Need Machenics to get states' reward



class NET(nn.Module):
    def __init__(self, input_length):
        super(NET, self).__init__()
        self.l1 = nn.Linear(input_length, 64)
        self.l2 = nn.Linear(64, 16)
        self.l3 = nn.Linear(16, 8)



    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x



class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def print_buffer(self):
        for i in range(len(self.buffer)):
            print(self.buffer[i])

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        #states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for idx in indices:
            info = self.buffer[idx]
            states.append(info[0])
            actions.append(info[1])
            rewards.append(info[2])
            dones.append(info[3])
            next_states.append(info[4])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
# create environment instance
def info_on_suite():
    print("All environment: ")
    print(suite.ALL_ENVIRONMENTS)
    print("\n")
    print("All robots:")
    print(suite.ALL_ROBOTS)
    print("\n")
    print("all controllers")
    print(suite.ALL_CONTROLLERS)
    print("\n")
    print("all griper")
    print(suite.ALL_GRIPPERS)
    print("\n")




def generate_random_action():
    #TODO:  provide negative actions too
    rand_ind = random.randint(0, env.robots[0].dof - 1)
    rand_np = np.random.randint(2)
    action = np.zeros(env.robots[0].dof)
    action[rand_ind] = action_size
    if rand_np == 1:
        action[rand_ind] = -action_size
    return action

def action_based_on_q_vals(action_values):
    max = -10000.0
    max_ind = -1
    for i in range(len(action_values)):
        if action_values[i] > max:
            max_ind = i
            max = action_values[i]
    action = torch.zeros(len(action_values))
    action[max_ind] = action_size
    action = action.detach()
    action = action.numpy()
    return action

#input is numpy
def distance_to_cube(cur_state):
    target = np.array([0,0,0])
    squared_dist = np.sum((cur_state - target) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def getting_reward(cur_state):
    dist = distance_to_cube(cur_state)

    if dist == 0:
        return 500
    result = 10 *  (1 / dist**2)
    # result = 3 * (1 - np.tanh(10.0 * dist))
    return result

def gain_experience(num, cur_state, net_work=None, use_random=False):
    #TODO: the action step need to be fix
    #TODO: need to figure out what value is the best for state. Can try obs returned from env.step() but seems too big as key for dict
    cur_state_c = cur_state
    for i in range(num):
        if use_random:
            action = generate_random_action()
        else:
            cur_state_v = torch.tensor(cur_state_c)
            action_values = net_work.forward(cur_state_v.float())
            action = action_based_on_q_vals(action_values)
        next_state, reward, done, info  = env.step(action)
        reward = getting_reward(next_state["gripper_to_cube_pos"])
        # print(reward)
        buffer.append([cur_state_c, action, reward, done, next_state["gripper_to_cube_pos"]])

        if done is True:
            env.reset()
            mask_action = np.zeros(env.robots[0].dof)
            next_state, reward, done, info = env.step(mask_action)
        #robot0_proprio-state
        #gripper_to_cube_pos
        cur_state_c = next_state["gripper_to_cube_pos"]
    return cur_state_c

def gain_q_value_for_action(qvals, actions):
    action_ind = actions.max(1)[1]
    result = torch.zeros(actions.size()[0])
    for i in range(actions.size()[0]):
        result[i] = qvals[i][action_ind[i]]
    return result



# reset the environment
#env.reset()
# def gen_experience(num):
#     for i in range(num):


#
# class DQN():
#     def

# for i in range(1):

#     action = np.random.randn(env.robots[0].dof) # sample random action
#
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     sampe = [obs, reward, done, info]
#     buffer = collections.deque(maxlen=2)
#     buffer.append(sampe)
#     sample2 = [obs, reward + 1, True, info]
#     obs1, reward1, done1, info1 = zip(*[sample2[idx] for idx in [0,1,2,3]])
#     # print("obs:")
#     # print(obs)
#     # print("\n reward")
#     # print(reward)
#     # print("\n done")
#     # print(done)
#     # print("\n info")
#     # print(info)
#     env.render()  # render on display


input_length = 3
env = suite.make(
    env_name="Lift",  # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
eps = 0.05
action_size = 1.5
epochs = 10000
gamma = 0.99
learning_rate = 1e-4
sync_target_frames = 1000

def testing_net(net_work, optimizer):
    cur_state = np.array([0.5,0.5,0.5])
    cur_state_t = torch.tensor(cur_state)
    q_vals = net_work.forward(cur_state_t.float())
    print(q_vals)
    action = np.array([0,0,0,0,action_size,0,0,0])
    next_state, reward, done, info = env.step(action)
    next_states_v = torch.tensor(next_state["gripper_to_cube_pos"])
    actions_v = torch.tensor(action)
    rewards = [10]
    rw = [100]
    rewards_v = torch.tensor(rewards)
    expected_state_action_values = rewards_v

    state_action_values = gain_q_value_for_action(q_vals, actions_v)
    loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()
    q_vals = net_work.forward(cur_state_t.float())
    print(q_vals)

#testing for pushing
if __name__ == "__main__":
    buffer = ExperienceReplay(10000)
    # info_on_suite()
    #net_work: current network, net_work target: previous network
    net_work = NET(input_length)

    net_work_target = NET(input_length)
    net_work_target.load_state_dict(net_work.state_dict())
    optimizer = torch.optim.Adam(net_work.parameters(), lr=learning_rate)
    obs, reward, done, info = env.step(np.zeros(env.robots[0].dof))

    cur_state = obs["gripper_to_cube_pos"]
    gain_experience(10000, cur_state,use_random=True)
    #Training neural network
    for i in range(epochs):
        #def gain_experience(num, cur_state, net_work=None, use_random=False):
        cur_state = gain_experience(4, cur_state, net_work=net_work)
        batch = buffer.sample(32)
        states, actions, rewards, dones, next_states = batch
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        states_v = torch.tensor(states)
        done_mask = torch.ByteTensor(dones)
        next_states_v = torch.tensor(next_states)
        q_vals = net_work.forward(states_v.float())
        next_q_val = net_work_target.forward(next_states_v.float()).max(1)[0]
        next_q_val = next_q_val.detach()
        next_q_val[done_mask] = 0.0
        expected_state_action_values = rewards_v + next_q_val * gamma
        state_action_values = gain_q_value_for_action(q_vals, actions_v)
        # state_action_values = state_action_values.detach()
        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        if epochs % sync_target_frames == 0:
            net_work_target.load_state_dict(net_work.state_dict())
    env.reset()
    # testing_net(net_work, optimizer)
    obs, reward, done, info = env.step(np.zeros(env.robots[0].dof))
    cur_state = obs["gripper_to_cube_pos"]
    unfinish = True
    shortest_dist = 100000
    while unfinish:
        q_vals = net_work_target(torch.tensor(cur_state).float()) # sample random action
        action = action_based_on_q_vals(q_vals)
        obs, reward, done, info = env.step(action)  # take action in the environment
        cur_state = obs["gripper_to_cube_pos"]
        cur_dist = distance_to_cube(cur_state)
        if cur_dist < shortest_dist:
            shortest_dist = cur_dist
            print(shortest_dist)
        env.render()  # render on display

        if done:
            unfinish = False








