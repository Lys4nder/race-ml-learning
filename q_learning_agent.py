import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from race_env import RaceEnv
from collections import deque
import heapq

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_distance(start, goal, track):
    if start == goal:
        return 0

    grid_size = len(track)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path_length = 0
            while current in came_from:
                current = came_from[current]
                path_length += 1
            return path_length

        for move in [(0, 1), (0, -1), (-1, 0), (1, 0)]:  # 4-neighborhood
            neighbor = (current[0] + move[0], current[1] + move[1])

            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue

            if track[neighbor[0], neighbor[1]] == 1:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return manhattan_distance(start, goal)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class EnhancedRaceEnv:
    def __init__(self, base_env):
        self.env = base_env
        self.previous_distance = None

    def reset(self):
        state = self.env.reset()
        self.previous_distance = astar_distance(tuple(state), self.env.finish_pos, self.env.track)
        direction = (self.env.finish_pos[0] - state[0], self.env.finish_pos[1] - state[1])
        direction_magnitude = max(abs(direction[0]) + abs(direction[1]), 1)
        normalized_direction = (direction[0] / direction_magnitude, direction[1] / direction_magnitude)
        return [state[0], state[1], self.previous_distance / self.env.grid_size, normalized_direction[0], normalized_direction[1]]

    def step(self, action):
        next_state, base_reward, done = self.env.step(action)

        if done:
            direction = (self.env.finish_pos[0] - next_state[0], self.env.finish_pos[1] - next_state[1])
            dm = max(abs(direction[0]) + abs(direction[1]), 1)
            nd = (direction[0] / dm, direction[1] / dm)
            enhanced_state = [
                next_state[0],
                next_state[1],
                0 if base_reward == 100 else self.previous_distance / self.env.grid_size,
                nd[0],
                nd[1]
            ]
            return enhanced_state, base_reward, done

        current_distance = astar_distance(tuple(next_state), self.env.finish_pos, self.env.track)
        distance_reward = (self.previous_distance - current_distance) * 2.0
        step_penalty = -0.5
        proximity_bonus = 0
        if current_distance <= 2:
            proximity_bonus = 5.0
        elif current_distance <= 4:
            proximity_bonus = 2.0

        staying_penalty = -2.0 if action == 0 else 0
        shaped_reward = distance_reward + step_penalty + proximity_bonus + staying_penalty

        self.previous_distance = current_distance

        direction = (self.env.finish_pos[0] - next_state[0], self.env.finish_pos[1] - next_state[1])
        direction_magnitude = max(abs(direction[0]) + abs(direction[1]), 1)
        normalized_direction = (direction[0] / direction_magnitude, direction[1] / direction_magnitude)

        enhanced_state = [next_state[0], next_state[1], current_distance / self.env.grid_size, normalized_direction[0], normalized_direction[1]]
        return enhanced_state, shaped_reward, done


def sample_batch(replay_buffer, batch_size, device):
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.from_numpy(np.array(states, dtype=np.float32)).to(device),
        torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
        torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
        torch.from_numpy(np.array(next_states, dtype=np.float32)).to(device),
        torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    )


def train_agent(episodes=10000, verbose=True, progress_callback=None):
    base_env = RaceEnv(render_mode=None)
    env = EnhancedRaceEnv(base_env)

    n_actions = 5  # stay, right, left, up, down
    input_dim = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_dim, n_actions).to(device)
    target_net = DQN(input_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=100000)
    batch_size = 128
    gamma = 0.98
    epsilon = 1.0
    epsilon_decay = 0.9997
    epsilon_min = 0.01
    target_update_freq = 500

    successes = 0
    recent_rewards = deque(maxlen=100)
    recent_success_rate = deque(maxlen=100)

    for ep in range(episodes):
        state = np.array(env.reset(), dtype=np.float32)
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state).to(device))
                    action = int(torch.argmax(q_values).item())

            next_state, reward, done = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = sample_batch(replay_buffer, batch_size, device)
                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        success = tuple(env.env.current_pos) == env.env.finish_pos
        if success:
            successes += 1
            recent_success_rate.append(1.0)
        else:
            recent_success_rate.append(0.0)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)

        if ep % target_update_freq == 0 and ep > 0:
            target_net.load_state_dict(policy_net.state_dict())

        if progress_callback:
            stats = {
                'episode': ep + 1,
                'total_episodes': episodes,
                'successes': successes,
                'avg_reward': float(np.mean(recent_rewards)) if recent_rewards else 0,
                'success_rate': float(np.mean(recent_success_rate)) if recent_success_rate else 0,
                'epsilon': float(epsilon)
            }
            progress_callback(ep + 1, stats)

    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'episodes': episodes,
        'successes': successes,
        'input_dim': input_dim
    }, 'trained_model.pth')

    return policy_net, target_net, successes, episodes
