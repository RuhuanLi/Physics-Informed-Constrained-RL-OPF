import numpy as np

import PowerNetwork.VoltagePredictor.VoltagePredictor as Model
import torch
import torch.nn as nn

class PhysicsActor(nn.Module):

    def __init__(self, Va_predictor, Vm_predictor, G, B, gen_idx, slack_idx, ppc):
        super().__init__()

        self.Va_predictor = Va_predictor
        self.Vm_predictor = Vm_predictor
        self.gen_idx = gen_idx

        branch = ppc["branch"]

        fbus = branch[:, 0].astype(int) - 1
        tbus = branch[:, 1].astype(int) - 1
        rateA = branch[:, 5]
        r = branch[:, 2]
        x = branch[:, 3]

        z = r + 1j * x
        y = 1 / z

        G_branch = np.real(y)
        B_branch = np.imag(y)

        gen = ppc["gen"]

        Qmax = gen[:, 3]
        Qmin = gen[:, 4]

        self.register_buffer("Qmax", torch.tensor(Qmax, dtype=torch.float32))
        self.register_buffer("Qmin", torch.tensor(Qmin, dtype=torch.float32))

        # 这个参数不参与梯度 被固定住 但是被模型所管理
        self.register_buffer("branch_from", torch.tensor(fbus))
        self.register_buffer("branch_to", torch.tensor(tbus))
        self.register_buffer("G", torch.tensor(G, dtype=torch.float32))
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("G_branch", torch.tensor(G_branch, dtype=torch.float32))
        self.register_buffer("B_branch", torch.tensor(B_branch, dtype=torch.float32))
        # ⚠ rateA 转为 p.u.
        rateA_pu = rateA / 100
        self.register_buffer("rateA", torch.tensor(rateA_pu, dtype=torch.float32))

        self.slack_idx = slack_idx
        self.ppc = ppc

    def forward(self, state):
        # state 必须是 tensor
        Va = self.Va_predictor(state)
        Vm = self.Vm_predictor(state)

        # Va = torch.tanh(Va)
        Vm = torch.tanh(Vm)

        # ===== 2️⃣ 映射到物理范围 =====
        Vm = (Vm + 1) / 2 * (1.05 - 0.95) + 0.95
        # Va = Va * 180

        Va[:, self.slack_idx] = 0.0

        P, Q = self.power_flow(Vm, Va)

        Pg = P[:, self.gen_idx]
        Vg = Vm[:, self.gen_idx]

        action = torch.cat([Vg, Pg], dim=1)

        return action

    def power_flow(self, V, theta):

        theta_diff = theta.unsqueeze(2) - theta.unsqueeze(1)

        P = V.unsqueeze(2) * V.unsqueeze(1) * (
            self.G * torch.cos(theta_diff) +
            self.B * torch.sin(theta_diff)
        ) * 100

        Q = V.unsqueeze(2) * V.unsqueeze(1) * (
            self.G * torch.sin(theta_diff) -
            self.B * torch.cos(theta_diff)
        ) * 100

        return P.sum(2), Q.sum(2)

    def compute_violations(self, state):
        # 重新算 P, Q
        Va = self.Va_predictor(state)
        Vm = self.Vm_predictor(state)

        Va[:, self.slack_idx] = 0.0

        P, Q = self.power_flow(Vm, Va)

        # ===== 举例：电压上下限 =====
        v_upper = Vm - 1.05
        v_lower = 0.95 - Vm
        v_violation = torch.maximum(v_upper, v_lower)

        # ===== Q限制 =====
        Qg = Q[:, self.gen_idx]
        q_upper = Qg - self.Qmax
        q_lower = self.Qmin - Qg
        q_violation = torch.maximum(q_upper, q_lower)

        # ===== 线路约束（示例）=====
        branch_violation = self.compute_branch_violation(Vm, Va)

        return v_violation, q_violation, branch_violation

    def compute_branch_violation(self, Vm, Va):
        Vi = Vm[:, self.branch_from]
        Vj = Vm[:, self.branch_to]

        theta_i = Va[:, self.branch_from]
        theta_j = Va[:, self.branch_to]

        theta_diff = theta_i - theta_j

        G = self.G_branch
        B = self.B_branch

        # from -> to 有功
        Pij = Vi ** 2 * G \
              - Vi * Vj * (G * torch.cos(theta_diff) +
                           B * torch.sin(theta_diff))

        # from -> to 无功
        Qij = -Vi ** 2 * B \
              - Vi * Vj * (G * torch.sin(theta_diff) -
                           B * torch.cos(theta_diff))

        Sij = torch.sqrt(Pij ** 2 + Qij ** 2 + 1e-8)

        violation = Sij - self.rateA

        return violation


class VoltageInference(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device)

        self.model = Model.VoltagePredictor(
            in_dim=ckpt["model_config"]["input_dim"],
            out_dim=ckpt["model_config"]["output_dim"],
            hidden_dim=256,
            depth=3
        ).to(device)

        self.model.load_state_dict(ckpt["model_state_dict"])

        self.X_mean = torch.tensor(
            ckpt["X_mean"],
            dtype=torch.float32,
            device=device
        )

        self.X_std = torch.tensor(
            ckpt["X_std"],
            dtype=torch.float32,
            device=device
        )

        self.Y_mean = torch.tensor(
            ckpt["Y_mean"],
            dtype=torch.float32,
            device=device
        )

        self.Y_std = torch.tensor(
            ckpt["Y_std"],
            dtype=torch.float32,
            device=device
        )

        self.device = device

    def forward(self, X):  # 这个用在RL的训练过程中 如果需要修改电压预测模型的模型 才需要保留梯度加入其中

        # 假设 X 已经是 tensor
        X = X.to(self.device)
        Xn = (X - self.X_mean) / self.X_std
        Yn = self.model(Xn)
        Y = Yn * self.Y_std + self.Y_mean

        return Y