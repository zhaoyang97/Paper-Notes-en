---
title: >-
  [Paper Note] Safe Reinforcement Learning with Preference-Based Constraint Inference
description: >-
  [ICML 2026][Reinforcement Learning][Safe RL] Ours proposes PbCRL, which learns safety constraints from trajectory comparisons using an extended Bradley-Terry preference model with a "dead-zone." It incorporates signal-to…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Safe RL"
  - "Preference Learning"
  - "Bradley-Terry"
  - "Dead-zone Loss"
  - "SNR Regularization"
date: 2026-05-08
content_hash: d1d6ed28d6349d43
---

# Safe Reinforcement Learning with Preference-Based Constraint Inference

**Conference**: ICML 2026  
**arXiv**: [2603.23565](https://arxiv.org/abs/2603.23565)  
**Code**: None  
**Area**: Reinforcement Learning / Safe RL / Preference Learning  
**Keywords**: Safe RL, Preference Learning, Bradley-Terry, Dead-zone Loss, SNR Regularization  

## TL;DR
Ours proposes PbCRL, which learns safety constraints from trajectory comparisons using an extended Bradley-Terry preference model with a "dead-zone." It incorporates signal-to-noise ratio (SNR) regularization to prevent the cost function from flattening and implements a two-stage (offline pre-training + online few-shot fine-tuning) pipeline. Evaluated on Safety Gymnasium, autonomous driving, and LLM alignment tasks, PbCRL significantly reduces costs while maintaining returns.

## Background & Motivation

**Background**: Safe RL is typically formalized as a Constrained MDP (CMDP), aiming to maximize cumulative reward $\mathcal{J}^R(\pi)$ while ensuring that expected cumulative cost $\mathcal{J}^C(\pi)=\mathbb{E}_\pi[\sum_t \gamma^t c(s_t,a_t)]$ does not exceed a threshold $d$. In practice, safety constraints are complex, subjective, and often lack explicit formulas (e.g., "what constitutes a dangerous lane change" often requires human judgment), necessitating constraint **inference** from data.

**Limitations of Prior Work**: Inferring constraints from expert demonstrations (IRL / CBF / Robust Optimization) requires large volumes of dense, high-quality demonstrations, which is extremely costly. Using cheaper **preference data** (binary comparisons of trajectories) is an attractive alternative, but most existing preference-based methods simply adopt the Bradley-Terry (BT) model, simplifying constraint inference to a "which trajectory is safer" ranking problem.

**Key Challenge**: The authors point out two subtle flaws of the BT model in Safe RL. First, BT learns **relative rankings** and is indifferent to absolute values and distribution shapes—whereas real cost distributions are naturally **heavy-tailed** (a single collision often triggers subseqent incidents leading to long-tailed $C(\tau)$). The approximately symmetric distributions inferred by BT **systematically underestimate** expected costs, causing unsafe policies to be misclassified as safe. Second, most existing works focus only on prediction accuracy, ignoring whether the cost model "flattens" the cost landscape, which hinders subsequent policy learning.

**Goal**: To patch preference-driven constraint inference by ensuring the inferred cost distribution matches the real heavy-tailed shape and preserves sufficient cost variance for policy gradients.

**Key Insight**: By adding a **dead-zone** $\delta>0$ to the "unsafe" side of the BT safety loss, gradients can continually push predicted costs of unsafe trajectories further, **theoretically** guaranteeing a heavier right tail in the learned distribution. Concurrently, incorporating "cost variance / preference label entropy" as an SNR term into the loss explicitly encourages discriminative cost outputs.

**Core Idea**: Combining "dead-zone + SNR" dual regularization with standard BT safety loss, complemented by a two-stage training process (offline pre-training + online fine-tuning of dead-zone $\delta$). This ensures that constraint inference aligns with real safety semantics while providing informative cost gradients for policy optimization.

## Method

### Overall Architecture

PbCRL learns both the unknown cost function $c(s,a)$ and threshold $d$ in a CMDP, shifting the threshold to 0 such that the constraint is $\mathcal{J}^{\hat C}(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t\hat c(s_t,a_t)]\le 0$. Training occurs in two stages:

1. **Offline Pre-training Stage**: A cost network $c_\psi(s,a)$ is trained on a pre-collected preference dataset $\mathcal{D}=\{(\tau_1,\tau_2,\mu_1,\mu_2,\epsilon_1,\epsilon_2)\}$ (where $\mu$ are pairwise preference labels and $\epsilon$ are binary safety labels) using the loss $\mathcal{L}_{PbCI}=\mathcal{L}_{pair}+\mathcal{L}_{safe}^{DZ}+\mathcal{L}_{SNR}$.
2. **Online Policy Optimization Stage**: The learned $c_\psi$ serves as the CMDP cost function, and the policy is updated using a PPO-Lag style Lagrangian method. A small number of online trajectories are sampled every $K$ steps for human labeling to fine-tune the cost network and adaptively update the dead-zone parameter $\delta$.

### Key Designs

1. **Dead-zone Extended BT Safety Loss**:
    - **Function**: Ensures the inferred cost distribution has a heavier right tail than standard BT, correcting expected cost underestimation and preventing "confident violations."
    - **Mechanism**: Views "safety status" as a pairwise comparison with a virtual threshold trajectory $\tau_{th}$ (where true cost equals $d$ and estimated cost equals 0), i.e., $\hat{\mathbb{P}}(\tau\succ\tau_{th})=\sigma(-\hat C(\tau))$. While standard safety loss only requires $\hat C(\tau)>0$ for unsafe trajectories, the dead-zone version requires $\hat C(\tau)>\delta$, formulated as $\mathcal{L}_{safe}^{DZ}=-\mathbb{E}_\mathcal{D}\big[\epsilon\log\sigma(-\hat C(\tau))+(1-\epsilon)\log\sigma(\hat C(\tau)-\delta)\big]$. The authors provide a three-step proof: Lemma 3.1 shows the gradient is strictly more negative for unsafe trajectories; Theorem 3.2 uses induction to extend this to multiple steps; Corollary 3.3 translates instance-level offsets into distribution-level tail dominance $\mathbb{P}(\hat C^{DZ}\ge z)>\mathbb{P}(\hat C\ge z)$.
    - **Design Motivation**: Pure preference loss only concerns relative ranking and cannot transform a symmetric distribution into a heavy-tailed one. The dead-zone adds a minimum "push" to the unsafe side, representing the minimal change needed to encode distribution shape into the loss.

2. **Signal-to-Noise Ratio (SNR) Regularization**:
    - **Function**: Prevents the cost network from fitting all costs into a narrow range, which would hide signals from the policy gradient.
    - **Mechanism**: Treats cost variance as the signal and preference label entropy as noise within each batch: $\mathcal{L}_{SNR}=-\zeta\,\mathrm{Var}(\hat C(\tau))/\mathcal{H}(p(\mu))$. Minimizing this encourages higher $\mathrm{Var}(\hat C(\tau))$ while automatically relaxing the constraint on batches with noisy (high entropy) labels.
    - **Design Motivation**: Policy gradients are sensitive to cost "topography"; flat landscapes fail to drive policy movement. Separately modeling signal and noise allows regularization intensity to adapt to data noise levels more stably than simple variance penalties.

3. **Two-stage Training + Adaptive Dead-zone Calibration**:
    - **Function**: Shifts expensive online labeling to the offline stage while maintaining alignment between the cost model and safety semantics as the policy evolves.
    - **Mechanism**: Stage one optimizes $\mathcal{L}_{PbCI}$ on $\mathcal{D}$ with fixed $\delta$. Stage two uses a PPO-Lag target $\mathcal{L}(\psi,\theta,\lambda)=-[\mathcal{J}^R(\pi_\theta)-\lambda\mathcal{J}^{C_\psi}(\pi_\theta)]$. Every $K$ steps, a batch of online trajectories $\mathcal{B}$ is labeled to update $\delta$ via gradient descent on the violation rate mismatch $\mathcal{L}_\delta=\|\hat{\mathbb{P}}_{vio}-\mathbb{P}_{vio}\|^2$. Theorem 5.2 provides convergence guarantees for $(\psi,\theta,\lambda)$ to a local optimum under multi-timescale stochastic approximation.
    - **Design Motivation**: Purely online labeling is prohibitively expensive, while purely offline learning suffers from distribution drift. Using the violation rate as a proxy (independent of true costs) allows $\delta$ to compensate for distribution drift using a single scalar parameter.

### Loss & Training

The total loss is $\mathcal{L}_{PbCI}=\mathcal{L}_{pair}+\mathcal{L}_{safe}^{DZ}+\mathcal{L}_{SNR}$, where $\mathcal{L}_{pair}$ is the standard BT pairwise cross-entropy. The policy side uses PPO-Lag to optimize the Lagrangian objective, with learning rates satisfying the three-timescale separation condition $lr_\lambda=o(lr_\theta)=o(lr_\psi)$ to ensure convergence.

## Key Experimental Results

### Main Results

Evaluated on Safety Gymnasium against PPO-Lag (oracle upper bound using true costs) and preference-based baselines RLSF and PPO-BT. PbCRL maintains near-oracle returns while keeping costs near the threshold.

| Task (Threshold) | Metric | PPO-Lag (Oracle) | PbCRL (Ours) | RLSF | PPO-BT |
|--------------|------|------------------|--------------|------|--------|
| HalfCheetah (5) | Return | $2619\pm124$ | $\mathbf{2367\pm138}$ | $2084\pm126$ | $2494\pm195$ |
| HalfCheetah (5) | Cost | $4.82\pm0.91$ | $\mathbf{4.66\pm1.03}$ | $3.26\pm0.78$ | (Violation) |

### Ablation Study

Removing dead-zone and SNR regularization reveals their respective contributions to safety and performance.

| Configuration | Cost Constraint Met | Return Level | Description |
|------|--------------|----------|------|
| Full PbCRL | Yes | Near oracle | Dead-zone + SNR + Two-stage active |
| w/o Dead-zone | Systematic Violation | High | Reverts to standard BT; costs underestimated, policy aggressive |
| w/o SNR | Yes | Significant Drop | Cost landscape flattened; weak policy gradient signals |
| w/o Online Calibration | Intermittent Violation | Moderate | Offline $\delta$ mismatched with online trajectory distribution |

### Key Findings

- **Dead-zone manages "Safety," SNR manages "Performance"**: Removing the dead-zone leads to significant cost violations (validating the theory on BT underestimation), while removing SNR results in the largest return drop (validating that flat cost landscapes harm the policy).
- **Two-stage training drastically reduces labeling costs**: Compared to fully online baselines, PbCRL shifts most labeling offline, requiring only small online batches for $\delta$ calibration.
- **Cross-domain transferability**: Beyond robotics, gains were observed in autonomous driving and LLM alignment, demonstrating that the "preference + tail alignment + signal preservation" combination is not specific to Safety Gymnasium.

## Highlights & Insights
- **Strict Proof**: Rather than relying on intuition, the work formally proves (Lemma → Theorem → Corollary) why BT models fail to infer heavy tails, providing theoretical support for the "dead-zone" modification.
- **Cost Topology**: Using SNR (Cost Variance / Label Entropy) explicitly incorporates the cost network's impact on policy learning into the objective, offering a new perspective for evaluating cost model quality.
- **Adaptive Calibration**: Using "violation rate mismatch" as a ground-truth-independent proxy to calibrate $\delta$ is a lightweight yet effective design for handling distribution drift, applicable to other "online hyperparameter calibration" preference learning scenarios.

## Limitations & Future Work
- The optimal value of $\delta$ depends heavily on the tail shape of the true cost distribution. If violations are sparse (e.g., in high-margin industrial systems), the proxy signal variance might be too high for stable calibration.
- The paper assumes access to preference data with binary safety labels $\epsilon$. If a dataset only contains relative comparisons without absolute safety judgments, the dead-zone target is unavailable.
- Convergence proofs rely on multi-timescale separation and Lipschitz conditions, which may not strictly hold for deep non-linear cost networks in practice.
- The Lagrangian optimization currently only guarantees local optima; multi-constraint or multi-stage tasks may require more complex Lagrangian structures.

## Related Work & Insights
- **vs RLSF (Reddy Chirra et al., 2024)**: RLSF uses standard BT to learn binary costs. Ours proves this setup underestimates expected costs and uses a dead-zone to "prop up" the distribution shape—noting that RLSF's apparent safety in HalfCheetah often stems from a "self-deceiving" cost model.
- **vs Safe RLHF / PPO-BT (Dai et al., 2024)**: Safe RLHF adapts BT to language models but remains at the ranking level. PbCRL adds safety and SNR losses to the BT framework, minimally fixing "distribution shape" and "signal strength" issues.
- **vs PPO-Lag (Ray et al., 2019)**: PPO-Lag serves as the performance upper bound with known costs. PbCRL effectively narrows the return gap to within a few percentage points using only preference data.

## Rating
- Novelty: ⭐⭐⭐⭐ Clearly identifies and fixes two theoretical flaws of BT in Safe RL with minimal intervention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various domains (robotics, driving, LLM) with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain, strong correspondence between theorems and algorithms.
- Value: ⭐⭐⭐⭐ Provides a transferable loss template and convergence guarantees for preference-based constraint inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Safe In-Context Reinforcement Learning](safe_in-context_reinforcement_learning.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)
- [\[ICML 2026\] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)
- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](../../ICLR2026/reinforcement_learning/safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
