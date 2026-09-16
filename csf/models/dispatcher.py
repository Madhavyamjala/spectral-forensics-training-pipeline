"""
Phase 2 - Budget-constrained RL dispatcher trained with Group Relative Policy Optimisation (GRPO).

The dispatcher sees the Phase-1 scanner verdict and the Llama-3.2-Vision "no-tool" state (the pooled
hidden state of the shared backbone before any evidence exists) and chooses a routing action:

    early_exit        - trust the frontline verdict (mean of scanner + no-tool backbone probabilities)
    spatial           - run spatial tools, then the arbiter
    spectral          - run spectral tools, then the arbiter
    latent            - run the DIRE latent tool, then the arbiter
    spatial_spectral  - run both cheap-ish tool groups, then the arbiter
    full_tri_domain   - run every tool, then the arbiter

Reward (proposal Sec. 4), with the tool-cost term normalised to [0, 1] so it acts as a tie-breaker
instead of flipping the sign of a correct answer:
    R = R_acc(y, y_hat) + alpha * R_attr(mIoU) - lambda_cost * cost(a) / max_cost - beta_miss * 1[y != Real and y_hat = Real]
R_attr is 0 because Chrono-TriClass-100k has no manipulation masks (alpha kept for completeness).
Cost(a) is MEASURED: median seconds of the tool groups (from the feature cache) + the arbiter pass.

Because routing outcomes are deterministic given a video, the arbiter's verdict for every action is
pre-computed once ("outcome table", csf.pipeline) and GRPO samples groups of actions against it:
advantages are group-normalised rewards, the loss is the PPO-clipped surrogate + exact categorical
KL to the frozen initial policy + an entropy bonus.

Input : outcome table arrays (state, scanner/no-tool probs, per-action probs, labels), profile, costs.
Output: trained `DispatcherPolicy` state_dict + training curve per ablation profile.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csf import LABEL2ID
from csf.logging_utils import get_logger

log = get_logger("models.dispatcher")

ACTIONS = ["early_exit", "spatial", "spectral", "latent", "spatial_spectral", "full_tri_domain"]
ACTION_GROUPS: Dict[str, List[str]] = {
    "early_exit": [], "spatial": ["spatial"], "spectral": ["spectral"], "latent": ["latent"],
    "spatial_spectral": ["spatial", "spectral"], "full_tri_domain": ["spatial", "spectral", "latent"],
}
REAL = LABEL2ID["real"]


@dataclass
class AblationProfile:
    name: str
    lambda_cost: float
    beta_miss: float
    allow_latent: bool = True


PROFILES: Dict[str, AblationProfile] = {
    "ultra_fast": AblationProfile("ultra_fast", lambda_cost=1.0, beta_miss=0.0, allow_latent=False),
    "balanced": AblationProfile("balanced", lambda_cost=0.2, beta_miss=0.5),
    "max_security": AblationProfile("max_security", lambda_cost=0.0, beta_miss=2.0),
}


def action_mask_array(action: str) -> np.ndarray:
    groups = ACTION_GROUPS[action]
    return np.array([g in groups for g in ("spatial", "spectral", "latent")], dtype=bool)


def action_costs(tool_costs: Dict[str, float], arbiter_seconds: float) -> np.ndarray:
    """Seconds per action on top of the always-paid scanner + dispatcher passes."""
    out = []
    for a in ACTIONS:
        groups = ACTION_GROUPS[a]
        if not groups:
            out.append(0.0)
            continue
        out.append(tool_costs.get("proposal", 0.0) + sum(tool_costs.get(g, 0.0) for g in groups) + arbiter_seconds)
    return np.asarray(out, dtype=np.float32)


def allowed_actions(profile: AblationProfile) -> torch.Tensor:
    return torch.tensor([profile.allow_latent or "latent" not in ACTION_GROUPS[a] for a in ACTIONS])


def compute_rewards(labels: torch.Tensor, preds: torch.Tensor, costs_norm: torch.Tensor,
                    profile: AblationProfile, alpha_attr: float = 0.0, miou: Optional[torch.Tensor] = None):
    r_acc = torch.where(preds == labels, 1.0, -1.0)
    r_attr = miou if miou is not None else torch.zeros_like(r_acc)
    missed = ((labels != REAL) & (preds == REAL)).float()
    return r_acc + alpha_attr * r_attr - profile.lambda_cost * costs_norm - profile.beta_miss * missed


class DispatcherPolicy(nn.Module):
    def __init__(self, state_dim: int, n_scalar: int, hidden: int = 256, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.state_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU())
        self.body = nn.Sequential(nn.Linear(hidden + n_scalar, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, n_actions))
        self.config = {"state_dim": state_dim, "n_scalar": n_scalar, "hidden": hidden, "n_actions": n_actions}

    def forward(self, state: torch.Tensor, scalars: torch.Tensor, allowed: Optional[torch.Tensor] = None):
        logits = self.body(torch.cat([self.state_proj(state.float()), scalars.float()], dim=-1))
        if allowed is not None:
            logits = logits.masked_fill(~allowed.to(logits.device), -1e9)
        return logits


def dispatcher_scalars(scanner_probs: np.ndarray, notool_probs: np.ndarray) -> np.ndarray:
    def entropy(p):
        return -(p * np.log(np.clip(p, 1e-9, 1))).sum(-1, keepdims=True)
    return np.concatenate([scanner_probs, notool_probs, entropy(scanner_probs), entropy(notool_probs)],
                          axis=-1).astype(np.float32)


class GRPOTrainer:
    def __init__(self, policy: DispatcherPolicy, profile: AblationProfile, costs: np.ndarray, cfg, device):
        self.policy = policy.to(device)
        self.ref = copy.deepcopy(policy).eval().requires_grad_(False)
        self.profile = profile
        self.cfg = cfg
        self.device = device
        self.costs_norm = torch.tensor(costs / max(costs.max(), 1e-9), device=device)
        self.allowed = allowed_actions(profile).to(device)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def _batch_rewards(self, actions, labels, action_preds):
        preds = action_preds.gather(1, actions)
        return compute_rewards(labels.unsqueeze(1).expand_as(actions), preds, self.costs_norm[actions], self.profile,
                               self.cfg.alpha_attr)

    def step(self, state, scalars, labels, action_preds) -> Dict[str, float]:
        g = self.cfg.group_size
        self.policy.eval()
        with torch.no_grad():
            old_logits = self.policy(state, scalars, self.allowed)
            dist = torch.distributions.Categorical(logits=old_logits)
            actions = dist.sample((g,)).T
            old_logp = dist.log_prob(actions.T).T
            rewards = self._batch_rewards(actions, labels, action_preds)
            adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
            adv = adv.clamp(-5, 5)
            ref_logits = self.ref(state, scalars, self.allowed)
        self.policy.train()
        stats = {}
        for _ in range(self.cfg.inner_epochs):
            logits = self.policy(state, scalars, self.allowed)
            logp_all = F.log_softmax(logits, -1)
            logp = logp_all.gather(1, actions)
            ratio = torch.exp(logp - old_logp)
            surrogate = torch.min(ratio * adv, ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv)
            p = logp_all.exp()
            allowed = self.allowed.unsqueeze(0).expand_as(p)
            ref_logp = F.log_softmax(ref_logits, -1)
            kl = (p * (logp_all - ref_logp)).masked_fill(~allowed, 0.0).sum(-1).mean()
            entropy = -(p * logp_all).masked_fill(~allowed, 0.0).sum(-1).mean()
            loss = -surrogate.mean() + self.cfg.kl_beta * kl - self.cfg.entropy_coef * entropy
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            stats = {"loss": loss.item(), "kl": kl.item(), "entropy": entropy.item(), "reward": rewards.mean().item(),
                     "clip_frac": ((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item()}
        return stats

    @torch.no_grad()
    def greedy_eval(self, state, scalars, labels, action_preds) -> Dict[str, float]:
        self.policy.eval()
        actions = self.policy(state, scalars, self.allowed).argmax(-1, keepdim=True)
        rewards = self._batch_rewards(actions, labels, action_preds)
        preds = action_preds.gather(1, actions).squeeze(1)
        return {"reward": float(rewards.mean()), "acc": float((preds == labels).float().mean()),
                "cost_norm": float(self.costs_norm[actions].mean()),
                "early_exit_rate": float((actions == ACTIONS.index("early_exit")).float().mean())}
