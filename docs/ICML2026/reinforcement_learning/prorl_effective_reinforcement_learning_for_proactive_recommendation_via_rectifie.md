---
title: >-
  [Paper Note] ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation
description: >-
  [ICML2026][Reinforcement Learning][Proactive Recommendation] Addressing the issue where naive policy gradients in "proactive recommendation" tasks collapse into "equal-length repetitive paths…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Proactive Recommendation"
  - "Policy Gradient"
  - "Length Shortcut"
  - "Position-Adaptive Advantage"
  - "Multi-Objective Reward"
date: 2026-05-08
content_hash: abc607128bb2ad79
---

# ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation

**Conference**: ICML2026  
**arXiv**: [2605.28293](https://arxiv.org/abs/2605.28293)  
**Code**: https://github.com/hongruhou89/ProRL  
**Area**: Reinforcement Learning / Recommender Systems  
**Keywords**: Proactive Recommendation, Policy Gradient, Length Shortcut, Position-Adaptive Advantage, Multi-Objective Reward

## TL;DR
Addressing the issue where naive policy gradients in "proactive recommendation" tasks collapse into "equal-length repetitive paths," the authors theoretically attribute this failure to the "length shortcut" and excessive variance induced by positive-mean step-level rewards after path-level reward decomposition. They propose ProRL: it employs Stepwise Reward Centering to subtract a constant baseline from the expected reward of each step to eliminate length bias, and Position-Specific Advantage Estimation to implement GRPO-style group baselines by step position for variance reduction. On three real-world datasets, ProRL outperforms heuristic, supervised, and LLM-based SOTA across four metrics: IoI, IoR, CTR, and Coherence.

## Background & Motivation

**Background**: Conventional recommender systems primarily reflect historical user preferences, but platforms often seek to proactively guide users toward specific target items (e.g., new arrivals, exclusive content, long-tail categories). This has led to Proactive Recommender Systems (PRS): given a user interaction sequence $S_u$ and a platform-specified target item $i_T$, the system generates a guidance path $L_u=(i_1,\ldots,i_L)$ of $L$ intermediate items. Each step must be accepted by the user (Path Feasibility, measured by CTR) and substantially increase user interest in the target item (Guidance Effectiveness, measured by IoI and IoR). Existing solutions fall into three categories: heuristics (IPG, ITMPRec) rely on greedy rules and often trap in local optima; LLM-based planning (LLM-IPP, T-PRA) is expensive and hard to deploy industrially; supervised methods (IRN) mimic historical paths, being limited by the training distribution and unable to discover superior paths.

**Limitations of Prior Work**: Fitting PRS into a reinforcement learning framework seems natural—one can optimize the policy using the path reward $R_\text{path}=\alpha\cdot\mathrm{IoI}+\beta\cdot\mathrm{IoR}+\gamma\cdot\mathrm{CTR}$ as an observed signal with a standard policy gradient estimator $\hat g_\text{std}=\frac{1}{nm}\sum_{i,j}\sum_{t=1}^{L^{(i,j)}}\nabla_\theta\log \pi_\theta^{(i,j,t)}\cdot R^{(i,j)}$. However, empirically, the policy often collapses within a few hundred steps into generating nearly identical paths of maximum length $L_\max$ for all users, losing diversity.

**Key Challenge**: The authors decompose this failure into two structural flaws. First, path-level rewards naturally decompose into step-level increments $r_t:=R(i_1,\ldots,i_t)-R(i_1,\ldots,i_{t-1})$. Experiments show $\mathbb E_\pi[r_t]$ is consistently positive for CTR, IoI, and IoR components, meaning $\mathbb E[R_\text{path}]$ grows linearly with path length—this implies that "extending the path" yields gains faster than "selecting better items" in early training, leading to a length shortcut. Theoretically, the authors prove that in a simplified model, the stop probability $p(s)$ monotonically drops to 0 at a rate of $O(1/s)$. Second, the standard estimator weights every step's log-probability by the total path reward $R^{(i,j)}$. Since the action at step $t$ only affects $r_t,\ldots,r_L$, including $r_1,\ldots,r_{t-1}$ injects irrelevant noise, resulting in high gradient variance.

**Goal**: Within a lightweight Transformer framework (using supervised pre-training + RL fine-tuning), redesign the policy gradient estimator such that: (1) the expected gain of extending a path is zero, forcing the model to allocate gradients toward "selecting better items"; and (2) the advantage estimation at each step is low-variance and unbiased.

**Key Insight**: Since the failure stems from the structural property of "non-zero mean step-level rewards," the most direct way to eliminate it is via *reward centering*—but it must be performed per step and paired with multi-objective normalization. Since the variance in standard estimation comes from weighting every step with the global reward, the GRPO idea of decomposing the critic into group baselines can be further refined to be *position-specific*.

**Core Idea**: By specializing reward centering and GRPO-style group baselines for the unique structures of "positive-mean step rewards" and "position-varying reward-to-go" in PRS, a critic-free, low-variance, and length-unbiased policy gradient estimator is derived.

## Method

### Overall Architecture
ProRL formulates PRS as an RL problem where an episode consists of multiple discrete selection steps: the policy $\pi_\theta(\cdot\mid S_u,i_T)$ autoregressively generates the next item ID from the vocabulary until it outputs EOS or reaches the length limit $L_\max=10$. The full pipeline is "supervised pre-training $\pi_0 \to$ policy gradient fine-tuning with the ProRL estimator." The RL objective function is $J(\theta)=\mathbb E_{L_u\sim\pi_\theta}[R_\text{path}]-\lambda\cdot D_\mathrm{KL}(\pi_\theta\|\pi_0)$. Since the KL term has an analytical gradient, the key lies entirely in estimating the reward term $\nabla_\theta\mathbb E_{\pi_\theta}[R]$. ProRL uses two independent but complementary mechanisms to solve the "length shortcut" and "gradient variance" problems respectively.

### Key Designs

1.  **Stepwise Reward Centering (SRC)**:
    *   **Function**: Subtracts a global constant baseline from each step-level reward so that the expected gradient of "extending one step" is zero, thereby eliminating the length shortcut.
    *   **Mechanism**: First decompose the path reward as $R=\sum_t r_t$. For a single reward, set $\tilde r_t=r_t-\bar r$, where $\bar r=\mathbb E_\pi[r_*]$ is the global expected step reward. By construction, $\mathbb E_\pi[\tilde r_t]=0$, so $\mathbb E[\sum_t\tilde r_t]$ no longer grows with $L$. In multi-objective scenarios, this is extended to per-component normalization $\tilde r_t=\sum_{i=1}^K w_i\cdot(r_t^{(i)}-\mu^{(i)})/\sigma^{(i)}$, where $\mu^{(i)},\sigma^{(i)}$ are estimated using rollouts in the first warm-up epoch and then frozen to prevent drift with $\pi$.
    *   **Design Motivation**: It was observed that the step-level expectations of CTR, IoI, and IoR are all positive and relatively stable (IoR declines slightly but remains positive). Thus, a single global baseline suffices to break the "length-reward coupling." Freezing statistics ensures reward estimation is not repeatedly distorted by policy improvements. Normalization makes gradient magnitudes across multi-objective terms comparable, preventing large-valued objectives (like IoR) from suppressing smaller ones (like IoI).

2.  **Position-Specific Advantage Estimation (PSAE)**:
    *   **Function**: Replaces the high-variance "total path reward × step log-prob" estimation with a position-adaptive low-variance baseline for each (input, step) pair without training a critic.
    *   **Mechanism**: First, use reward-to-go $G_t^{(i,j)}=\sum_{\ell=t}^{L^{(i,j)}}r_\ell^{(i,j)}$ to remove past $r_1,\ldots,r_{t-1}$ (irrelevant to the current action) from the weighting signal. Then, apply grouped baselines per position: $\bar G_{i,t}=\sum_{j:L^{(i,j)}\ge t}G_t^{(i,j)}/\sum_j \mathbb I[L^{(i,j)}\ge t]$, averaging across all rollouts for input $i$ that reach step $t$. The final advantage is $\hat A_t^{(i,j)}=G_t^{(i,j)}-\bar G_{i,t}$, and the rectified estimator is $\hat g_\text{rect}=\frac{1}{nm}\sum_{i,j}\sum_{t=1}^{L^{(i,j)}}\nabla_\theta\log\pi_\theta^{(i,j,t)}\cdot\hat A_t^{(i,j)}$.
    *   **Design Motivation**: GRPO uses the average path reward $\bar R_i$ as a shared baseline for all steps, ignoring that reward-to-go magnitudes naturally differ between early and late steps. Later steps have shorter accumulations ($G_t$ is smaller), so a global baseline is too low for early steps and too high for later steps, which may fail to reduce variance. Grouping by position naturally fits the PRS structure where $\mathbb E[G_t]$ varies with $t$, maintaining an unbiased estimate (per Williams, 1992) with significantly lower variance.

3.  **Joint Training of Multi-Objective Rewards and KL Constraints**:
    *   **Function**: Evaluates CTR/IoI/IoR components within a single centered, normalized, and KL-constrained objective to ensure RL fine-tuning does not deviate too far from $\pi_0$.
    *   **Mechanism**: The reward side injects $\tilde r_t$ into advantage calculations; the KL side uses $\pi_0$ as an anchor, and the analytical gradient of $\lambda\cdot D_\mathrm{KL}(\pi_\theta\|\pi_0)$ is added directly to $\hat g_\text{rect}$. The number of rollout samples $m$ controls the statistical validity of the PSAE baselines.
    *   **Design Motivation**: In the PRS reward combination $R_\text{path}=\alpha\cdot\mathrm{IoI}+\beta\cdot\mathrm{IoR}+\gamma\cdot\mathrm{CTR}$, the terms differ by orders of magnitude. Without normalization, terms like IoR (magnitude in hundreds) would dominate gradients. KL anchoring preserves linguistic/sequential priors from pre-training and prevents PSAE from drifting out-of-distribution, a common practice to avoid "reward hacking" in industrial deployment.

### Loss & Training
Training consists of two stages. **Pre-training stage**: Historical interaction sequences are truncated into (history, target, path) triplets, and $\pi_0$ is learned via seq2seq cross-entropy. **RL stage**: For each batch, $m$ paths are sampled from $\pi_\theta$. The first epoch uses rollouts to estimate $\mu^{(i)},\sigma^{(i)},\bar r$, which are then frozen. Subsequent epochs perform policy gradient ascent based on $\hat g_\text{rect}-\lambda\nabla_\theta D_\mathrm{KL}(\pi_\theta\|\pi_0)$. $L_\max=10$. The user simulator is a SASRec model trained on historical interactions, and the acceptance probability $P(i\mid S)$ is retrieved from the SASRec softmax output.

## Key Experimental Results

### Main Results
Three real-world datasets (MovieLens-1M, Steam, Amazon-Book) were tested using SASRec as the evaluator, compared against 9 baselines.

| Dataset | Metric | ProRL | Next Best Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| MovieLens-1M | CTR | 0.8543 | IRN 0.8398 | +1.7% |
| MovieLens-1M | IoI | 2.8504 | T-PRA 2.4867 | +14.6% |
| MovieLens-1M | IoR | 728.18 | LLM-IPP 662.52 | +9.9% |
| Steam | CTR | 0.5625 | Bert4Rec 0.4617 | +21.8% |
| Steam | IoI | 1.1188 | T-PRA 0.3339 | +235% |
| Amazon-Book | IoI | 2.9812 | T-PRA 1.7261 | +72.7% |
| Amazon-Book | IoR | 1383.41 | T-PRA 476.93 | +190% |

ProRL also dominantly outperforms in **Coherence** (semantic continuity, not included in rewards), e.g., MovieLens-1M 0.8422 vs LLM-IPP 0.6288, indicating the learning of high-quality paths rather than reward hacks.

### Ablation Study
On ML-1M, removing SRC and PSAE:

| Configuration | CTR | IoI | IoR | Note |
| :--- | :--- | :--- | :--- | :--- |
| Full ProRL | 0.8543 | 2.8504 | 728.18 | Full Model |
| w/o SRC | 0.9731 | 1.2373 | 649.96 | CTR is highest but IoI drops 56%; confirms length shortcut causes over-optimization for clicks. |
| w/o PSAE | 0.7456 | 2.5556 | 695.86 | All metrics drop; PSAE contributes significantly to CTR. |

Multi-objective reward ablation (ML-1M): removing any of CTR, IoI, or IoR leads to a significant drop in its respective primary metric, and in some cases, other metrics also degrade, proving the three objectives mutually reinforce each other.

### Key Findings
*   The true value of SRC lies in guidance rather than CTR: without SRC, CTR reaches 0.97 due to positive-mean bias but IoI is halved, indicating that the length shortcut essentially uses "short-term clicks" to crowd out "long-term guidance."
*   PSAE has a structural advantage over GRPO’s $\bar R_i$ baseline in PRS: as reward-to-go decreases with steps, position-based baselines achieve lower variance than global ones (Table 5 reports advantage variance significantly lower than reward-to-go and GRPO).
*   **Cross-evaluator analysis**: ProRL trained on SASRec consistently ranks first in IoI/IoR when evaluated by unseen models like GRU4Rec, LightSANs, and Bert4Rec (e.g., ML-1M IoI 2.4560 vs T-PRA 2.3167 under GRU4Rec), proving it learns a generalized guidance strategy rather than overfitting the reward model.

## Highlights & Insights
*   Elevating "training failure" to a "structural diagnosis" is the most compelling aspect: by decomposing $R=\sum r_t$ and empirically showing $\mathbb E[r_t]>0$, the length shortcut is transformed from an empirical phenomenon into a provable $O(1/s)$ result, giving the fix (centering) a clear objective—"make $\mathbb E[\tilde r_t]=0$."
*   Extending the GRPO-style group baseline to be position-specific is a generalizable trick for any task where "rewards are decomposable and step-level semantics vary," such as dialog generation, tool-use planning, or multi-step code agent generation.
*   Freezing warm-up epoch statistics ($\mu, \sigma, \bar r$) is an often-overlooked engineering detail: allowing the baseline to update alongside the policy can create a feedback loop where "policy improvement raises the baseline $\to$ decreases advantage $\to$ stalls further improvement."
*   Coherence improves despite not being in the reward, suggesting the "Reward + KL Anchor" combination naturally favors semantically continuous paths—a proof of existence that PRS can move beyond the training distribution while retaining semantic priors better than pure supervised methods.

## Limitations & Future Work
*   Using a scalar global baseline $\bar r$ to approximate expectations for all steps introduces residual bias for rewards where $\mathbb E[r_t]$ varies significantly (e.g., the slight decline in IoR); a more refined approach would estimate $\bar r_t$ per position, though at the cost of sample efficiency.
*   The user simulator is an offline SASRec model. While cross-evaluator experiments mitigate this, it remains "RL on a simulator," and real-world A/B testing gains have not been verified.
*   While SRC fixes the length shortcut for $L_\max=10$, approximation errors of a constant $\bar r$ may accumulate in much longer episodes (e.g., 100 steps); low-rank $\bar r_t$ or critic-based baselines might be needed.
*   The selection of the KL anchoring weight $\lambda$ was not systematically ablated in the main results; weak KL loses the $\pi_0$ prior while strong KL retards RL toward SFT, representing a hidden tuning cost for deployment.

## Related Work & Insights
*   **vs IRN (Zhu et al., 2023)**: IRN uses direct seq2seq imitation learning on historical paths and is locked to the training distribution; ProRL uses RL on the same lightweight Transformer backbone to explore out-of-distribution paths, achieving over 65% higher IoI than IRN.
*   **vs LLM-IPP / T-PRA**: While LLMs are strong at planning, they are costly and deployment-unfriendly; ProRL uses a small model + RL to surpass most metrics, especially on Steam where IoI improves from 0.33 to 1.12.
*   **vs GRPO (Shao et al., 2024)**: GRPO uses a single global baseline $\bar R_i$, whereas PSAE refines the baseline by position $t$, leveraging the structural signal of reward-to-go variation in PRS without requiring a critic.
*   **vs Classic REINFORCE + baseline (Williams, 1992)**: Classic baselines use a state-value $V(s)$ which requires training a separate critic; PSAE replaces this with the mean of multiple rollouts for the same input and position, requiring zero additional parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](../../NeurIPS2025/reinforcement_learning/robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](../../ACL2026/reinforcement_learning/ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)
- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)

</div>

<!-- RELATED:END -->
