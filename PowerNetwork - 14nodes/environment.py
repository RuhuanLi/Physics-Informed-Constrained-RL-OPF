import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pypower.api import runpf, printpf, ppoption


def case118_pf_pypower(Pg, Vg, ppc, Pd=None, Qd=None):
    """
    IEEE 118 AC 潮流

    Pg : 发电机有功 (ng,)
    Vg : 发电机电压 (ng,)
    Pd : 可选负荷
    Qd : 可选负荷
    ppc: 读取算例框架
    """

    ppc['gen'][:, 1] = Pg
    ppc['gen'][:, 5] = Vg

    if Pd is not None:
        ppc['bus'][:, 2] = Pd
    if Qd is not None:
        ppc['bus'][:, 3] = Qd

    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
    results, success = runpf(ppc, ppopt)

    return results, success


class OPFEnv:
    """
    Physics-informed RL Environment
    State  : [Pg(t-1), Pd(t), Qd(t)]
    Action : Pg_set(t)
    Obs    : Voltage / Angle (via surrogate)
    episodes num: 300
    """

    def __init__(
        self,
        X,
        Vm,
        Va,
        base_ppc,
        episode_length=288,
    ):
        self.X_full = X
        self.Vm_full = Vm
        self.Va_full = Va

        self.episode_length = episode_length
        self.num_episodes = X.shape[0]//episode_length

        self.base_ppc = base_ppc
        gencost = self.base_ppc['gencost']
        self.ng = self.base_ppc['gen'].shape[0]
        self.nb = base_ppc['bus'].shape[0]
        self.a = gencost[:, 4]
        self.b = gencost[:, 5]
        self.c = gencost[:, 6]
        self.pg_max = self.base_ppc['gen'][:, 8]
        self.pg_min = self.base_ppc['gen'][:, 9]

        self.pg_max_original = self.base_ppc['gen'][:, 8].copy()

        self.reset()


    def reset(self, episode_id=None):

        if episode_id is None:
            episode_id = np.random.randint(self.num_episodes)

        self.t = 0

        start = episode_id * self.episode_length
        end   = start + self.episode_length

        self.X = self.X_full[start:end]
        self.pd = self.X[self.t][self.ng:self.ng+self.nb]
        self.qd = self.X[self.t][self.ng+self.nb:]

        self.pg_prev = self.base_ppc['gen'][:, 1].copy()

        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            self.pg_prev,
            self.pd,
            self.qd
        ])

    def step(self, action):
        """
        action = [Vg, Pg]
        """
        # 更新动作 把动作分成需要的两个量
        # ==========================
        pg = action[self.ng:]
        vg = action[:self.ng]

        # 限幅 其实爬坡约束也得限制 应该就是吧Pgt-1跟Pgt放一起 记得加 暂时先不管 这个地方插一个旗 看剪贴板是否有大影响
        pg = np.clip(pg, self.pg_min, self.pg_max)
        vg = np.clip(vg, 0.95, 1.05)

        # 爬坡我写了放在这里 到时候爬坡不爬都运行一下试试看 看到底是怎么个回事
        pg = np.clip(
            pg,
            self.pg_prev - 0.2 * self.pg_max,
            self.pg_prev + 0.2 * self.pg_max
        )
        # 更新当前负荷 此时就是一步一步的 这个仅为此处一步
        self.pd = self.X[self.t][self.ng:self.ng + self.nb]
        self.qd = self.X[self.t][self.ng + self.nb:]

        self.ppc = copy.deepcopy(self.base_ppc)

        # 潮流计算 以计算越限值 拿来当reward的一部分 这个地方理应将X也拿进来 为了防止上下不对应这里也标个点脑子清醒的时候修改
        # ==========================
        results, success = case118_pf_pypower(
            Pg=pg,
            Vg=vg,
            ppc=self.ppc,
            Pd=self.pd,
            Qd=self.qd
        )
        # 首先判断是否收敛
        # ==========================
        if not success:
            reward = -200
            done = True
            print("not success")
            return self._get_state(), reward, done, {}

        Vm = results['bus'][:, 7]
        Qg = results['gen'][:, 2]

        # 各参数越界检查（e的计算）
        # ==========================
        Qmax = results['gen'][:, 3]
        Qmin = results['gen'][:, 4]

        q_violation = np.sum(
            np.maximum(0, Qg - Qmax) +
            np.maximum(0, Qmin - Qg)
        )

        v_violation = np.sum(
            np.maximum(0, Vm - 1.05) +
            np.maximum(0, 0.95 - Vm)
        )

        S = results['branch'][:, 13]  # or PF
        rateA = results['branch'][:, 5]

        branch_violation = np.sum(
            np.maximum(0, np.abs(S) - rateA)
        )

        # 这里面的参数就很有说法了 需要根据opf里面的来 也需要修改
        cost = np.sum(self.a * pg ** 2 +
                      self.b * pg +
                      self.c)

        reward = - cost/1e6 \
                 - 0.5 * v_violation \
                 - q_violation/1e5 \
                 - branch_violation/1e5

        # 更新时间
        self.pg_prev = pg.copy()
        self.t += 1
        done = self.t >= self.episode_length

        return self._get_state(), reward, done, cost, v_violation, q_violation, branch_violation