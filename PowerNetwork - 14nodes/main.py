import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 放在最前面
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import copy
import torch
from environment import OPFEnv
from agent import Agent
from PhysicsInformedActor import VoltageInference, PhysicsActor
from policy import Critic
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pypower.api import runopf, ppoption, case14
import pandas as pd
import scipy.io as sio

def load_opf_dataset(mat_path):
    data = sio.loadmat(mat_path)

    X  = data['X']      # [N, dx]
    Y1 = data['Y1']     # [N, Ng]
    Y2 = data['Y2']     # [N, Ng]

    Pg = Y1[:, :5]  # Pg(t)
    Vm = Y1[:, 5:]  # Vm(t)
    Va = Y2[:, 5:]  # θ(t)

    return X, Pg, Vm, Va

# 这一部分是读取 case118的部分 其实可以放到actor里面去
def build_GB(case):
    bus = case["bus"]
    branch = case["branch"]
    nb = bus.shape[0]

    Ybus = np.zeros((nb, nb), dtype=complex)

    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r = br[2]
        x = br[3]
        b = br[4]

        z = complex(r, x)
        y = 1 / z
        b_shunt = complex(0, b / 2)

        Ybus[f, f] += y + b_shunt
        Ybus[t, t] += y + b_shunt
        Ybus[f, t] -= y
        Ybus[t, f] -= y

    G = Ybus.real
    B = Ybus.imag

    return G, B

def train_ddpg(env, agent,
               train_episodes=2000,
               batch_size=128,
               noise_std=0.01):

    episode_rewards = []
    episode_costs = []
    episode_v_violations = []
    episode_q_violations = []
    episode_branch_violations = []

    for episode in range(train_episodes):

        state = env.reset(episode_id=episode)
        done = False
        total_reward = 0
        total_cost = 0
        total_v_violation = 0
        total_q_violation = 0
        total_branch_violation = 0

        while not done:
            action = agent.select_action(state)

            # exploration noise
            action += np.random.normal(0, noise_std, size=action.shape)

            next_state, reward, done, cost, v_violation, q_violation, branch_violation= env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)

            state = next_state
            total_reward += reward
            total_cost += cost
            total_v_violation += v_violation
            total_q_violation += q_violation
            total_branch_violation += branch_violation


        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_v_violations.append(total_v_violation)
        episode_q_violations.append(total_q_violation)
        episode_branch_violations.append(total_branch_violation)

        print(f"Episode {episode} | Total Reward: {total_reward:.2f}")
        print(f" | Total cost: {total_cost:.2f}")

    return episode_rewards, episode_costs, episode_v_violations, episode_q_violations, episode_branch_violations

def evaluate_one_episode(env1, agent1, episode_id=0):

    state = env1.reset(episode_id=episode_id)
    done = False

    cost_list = []
    v_violations_list1 = []
    q_violations_list1 = []
    branch_violations_list1 = []

    pg_list = []

    while not done:

        action = agent1.select_action(state)  # 无噪声

        next_state, reward, done, cost1, v_violation, q_violation, branch_violation = env1.step(action)

        # 你现在 reward = - cost - penalty
        cost_list.append(cost1)
        v_violations_list1.append(v_violation)
        q_violations_list1.append(q_violation)
        branch_violations_list1.append(branch_violation)
        pg_list.append(action[env1.ng:])

        state = next_state

    total_cost = np.sum(cost_list)

    return total_cost, np.array(cost_list), np.array(pg_list), np.array(v_violations_list1), np.array(q_violations_list1), np.array(branch_violations_list1)


def precompute_opf_benchmark(env_b, episode_id, save_path="opf_cache.npy"):

    print("Precomputing OPF benchmark...")

    # ===== 创建缓存文件夹 =====
    os.makedirs(save_path, exist_ok=True)

    # ===== 生成文件路径 =====
    save_path = os.path.join(
        save_path,
        f"opf_ep_{episode_id}.npy"
    )

    env_b.reset(episode_id)
    X = env_b.X

    cost_list = []

    for t in range(env_b.episode_length):

        ppc = copy.deepcopy(env_b.base_ppc)

        # ===== 读取负荷 =====
        pd = X[t][env_b.ng:env_b.ng+env_b.nb]
        qd = X[t][env_b.ng+env_b.nb:]

        ppc['bus'][:,2] = pd
        ppc['bus'][:,3] = qd

        results = runopf(ppc, ppoption(VERBOSE=0, OUT_ALL=0))

        cost_list.append(results['f'])

    cost_array = np.array(cost_list)

    np.save(save_path, cost_array)

    print("OPF benchmark saved to:", save_path)

    return cost_array

def run_opf_benchmark(env_b,
                      episode_id=0,
                      cache_dir="opf_cache"):

    save_path = os.path.join(
        cache_dir,
        f"opf_ep_{episode_id}.npy"
    )

    if os.path.exists(save_path):

        cost_array = np.load(save_path)
        return np.sum(cost_array), cost_array

    cost_array = precompute_opf_benchmark(
        env_b,
        episode_id,
        cache_dir
    )

    return np.sum(cost_array), cost_array


def plot_cost_curve(cost_rl, cost_opf=None, vviolation=None, qviolation=None, branchviolation=None):

    time = np.arange(len(cost_rl))

    plt.figure(1)
    plt.plot(time, cost_rl)

    if cost_opf is not None:
        plt.figure(2)
        plt.plot(time, cost_opf)

    # if cost_opf is not None:
    #     plt.figure(3)
    #     plt.plot(time, vviolation)
    #
    # if cost_opf is not None:
    #     plt.figure(4)
    #     plt.plot(time, qviolation)
    #
    # if cost_opf is not None:
    #     plt.figure(5)
    #     plt.plot(time, branchviolation)

    plt.xlabel("Time Step (5 min resolution)")
    plt.ylabel("Generation Cost")
    plt.title("Cost Comparison Over 24 Hours")
    plt.draw()

def plot_training_curve(reward, cost2, vviolation=None, qviolation=None, branchviolation=None):

    episodes = np.arange(len(reward))
    plt.figure(0)
    plt.plot(episodes, reward)

    cost2 = [ x/1e6 for x in cost2]
    vviolation = [ x/2 for x in vviolation]
    qviolation = [x/1e5 for x in qviolation]
    branchviolation = [x / 1e5 for x in branchviolation]

    plt.figure(1)
    plt.plot(episodes, cost2,'r',label="cost")
    plt.plot(episodes, vviolation,'g',label="v_violation")
    plt.plot(episodes, qviolation,'y',label="q_violation")
    plt.plot(episodes, branchviolation,'b',label="branch")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Convergence Curve")
    plt.legend()
    plt.draw()


# ----------main------------
plt.ion()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 尝试读取  X其实就是state 不需要拆开 直接放进去就好
X, Pg, Vm, Va = load_opf_dataset('./Dataset/opf_dataset_array_case14.mat')

# 读取自己的case118
ppc = case14()

slack_idx = np.where(ppc['bus'][:,1] == 3)[0]
print(slack_idx)

# env
env = OPFEnv(X, Vm, Va, ppc)

G, B = build_GB(ppc)

### use the trained VoltagePredictors
# load the model Va and Vm
Va_predictor = VoltageInference("./Dataset/VaPredictor_14.pt", DEVICE)
Vm_predictor = VoltageInference("./Dataset/VmPredictor_14.pt", DEVICE)
gen_idx = ppc['gen'][:, 0].astype(int) - 1

actor = PhysicsActor(
    Va_predictor,
    Vm_predictor,
    G,
    B,
    gen_idx,
    slack_idx,
    ppc
)

state_dim = env.ng + 2*env.nb
action_dim = 2 * env.ng
critic = Critic(state_dim, action_dim)

agent = Agent(actor, critic, device=DEVICE)

# 感觉用不上这么多 虽然我有3000组数据 但是我感觉搞个300组差不多了
rewards, cost, v_violations_list, q_violations_list, branch_violations_list = train_ddpg(
    env,
    agent,
    train_episodes=50,
    batch_size=256
)

plot_training_curve(rewards,cost, v_violations_list, q_violations_list, branch_violations_list)

test_id = 90
# ===== 单日测试 =====
total_cost_rl, cost_curve_rl, pg_curve,_,_,_ = evaluate_one_episode(
    env, agent, episode_id=test_id
)

print("RL Total Cost:", total_cost_rl)


# ===== OPF benchmark =====
total_cost_opf, cost_curve_opf = run_opf_benchmark(
    env, episode_id=test_id
)

print("OPF Total Cost:", total_cost_opf)

# ===== 画论文图9 =====
# plot_cost_curve(cost_curve_rl, total_cost_rl*1e7/3, cost_curve_opf)
print("OPF Cost:", cost_curve_opf)

# 最后保持显示
plt.ioff()
plt.show()
