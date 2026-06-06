---
title: >-
  [Paper Note] F-TIS: Harnessing Diverse Models in Collaborative GRPO
description: >-
  [ICML 2026][LLM Alignment][GRPO] F-TIS combines "Truncated Importance Sampling (TIS)" with the "filtering of negative advantage off-policy samples via KL threshold" into a unified GRPO loss. This allows multiple LLMs of…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "GRPO"
  - "Decentralized RL"
  - "Heterogeneous Model Collaboration"
  - "Importance Sampling"
  - "Truncation + Filtering"
date: 2026-05-08
content_hash: 470247bdffed422b
---

# F-TIS: Harnessing Diverse Models in Collaborative GRPO

**Conference**: ICML 2026  
**arXiv**: [2605.22537](https://arxiv.org/abs/2605.22537)  
**Code**: None  
**Area**: Reinforcement Learning / LLM Post-training / Decentralized Training  
**Keywords**: GRPO, Decentralized RL, Heterogeneous Model Collaboration, Importance Sampling, Truncation + Filtering

## TL;DR
F-TIS combines "Truncated Importance Sampling (TIS)" with the "filtering of negative advantage off-policy samples via KL threshold" into a unified GRPO loss. This allows multiple LLMs of different sizes, expertise, or even those with only partial trainable parameters to exchange samples during a single decentralized GRPO training run. The method achieves convergence comparable to pure on-policy training and yields up to +12% performance gains on OOD math tasks.

## Background & Motivation

**Background**: GRPO has become the de facto standard for LLM post-training, particularly for reasoning enhancement. It samples a group of $G$ completions for each prompt and uses group-normalized advantage $\hat{A}_i = (r_i - \mu_r)/\sigma_r$ to replace the PPO value model, saving memory and compute. However, the bottleneck of GRPO shift from backpropagation to autoregressive generation—sampling 8 or more completions per prompt is often prohibitive for a single GPU. Existing industry practices distribute the generation step across multiple nodes in parallel.

**Limitations of Prior Work**: Existing distributed GRPO frameworks (LlamaRL, Intellect2, GenRL) default to running the same model across all nodes to keep the generator and trainer distributions as close as possible, maintaining a "near on-policy" state. This assumption fails in "decentralized training" scenarios where models with different architectures, compute capacities, or preferences collaborate. Even when starting from the same checkpoint, exchanging completions causes models to drift due to floating-point non-associativity, generating destructive "pseudo-on-policy" noise that leads to policy collapse.

**Key Challenge**: GRPO is fundamentally an on-policy algorithm; clipped IS only tolerates "slight staleness." It fails when encountering truly heterogeneous off-policy samples (verified in the paper using Qwen2.5-1.5B + 3B joint training on GSM8K, where both models performed worse than when trained individually). To enable decentralization, GRPO must be able to **consume off-policy samples without collapsing** while maintaining low communication overhead.

**Goal**: Enable multiple heterogeneous models to feed each other samples in a single GRPO training run across three types of heterogeneity (model size, expertise, and trainable parameters), achieving on-policy-level convergence with communication restricted to ~8 bytes per token (log-prob + token id).

**Key Insight**: This work bridges two independent research lines: (1) TIS proposed by `yao2025efficient_rl_offpolicy` (clamping the importance ratio outside the token-level loss to a constant $C$); (2) the practice in DeepSeek-v3 and HTTT of "filtering negative advantage $\hat{A}_i < 0$ off-policy samples based on a KL threshold." The former minimizes gradient bias, while the latter prunes "negative advantage + distant policy" samples—those responsible for amplifying tokens the current model cannot generate and causing "gibberish" collapse. Remaining $\hat{A}_i > 0$ samples still provide meaningful direction signals even if they are off-policy.

**Core Idea**: F-TIS = Low-bias gradients from TIS + Stability from negative advantage filtering, serving as a unified loss for decentralized heterogeneous GRPO.

## Method

### Overall Architecture
F-TIS implements "vertical decentralized RL": each node receives a prompt, generates a complete group of $G$ completions locally, and broadcasts `(token_ids, per-token log-prob, reward)` to other nodes via all-gather. Upon receiving completions from all nodes, each node treats them as training samples for its local policy $\pi_\theta$, updating the model using the modified GRPO loss. Communication volume is only $8 \times |p|$ bytes per token (4-byte token id + 4-byte log-prob), making it viable over wide-area networks. Group advantages $\hat{A}_i$ are computed locally, ensuring they are always normalized relative to the local node's perspective.

The training consists of two phases:
- **Generation Phase**: Each node independently runs group sampling using its $\pi_{\theta_{gen}}$ and computes rewards and group advantages locally.
- **Training Phase**: Each node performs a forward pass using its current $\pi_\theta$ on all gathered completions to obtain $\pi_\theta(a_{i,t}|\cdot)$, which is then fed into the F-TIS loss along with the broadcasted $\pi_{\theta_{gen}}(a_{i,t}|\cdot)$.

### Key Designs

1.  **Truncated Importance Sampling (TIS) as Backbone**:
    - **Function**: Uses an importance ratio clamped to an upper bound $C$ instead of the standard token-level ratio to control the variance and bias introduced by heterogeneous generators.
    - **Mechanism**: Traditional GRPO is formulated at the token level as $\min[r_{i,t}\hat{A}_i, \text{clip}(r_{i,t}\hat{A}_i, 1\pm\epsilon)]$, where $r_{i,t}=\pi_\theta/\pi_{\theta_{gen}}$. TIS moves this ratio outside the token loss for overall magnitude limiting: $\min\big(\pi_\theta/\pi_{\theta_{gen}},\; C\big)\cdot \min(\mathcal{R}_{i,\theta}\hat{A}_i,\; \text{clip}(\mathcal{R}_{i,\theta}\hat{A}_i, 1-\epsilon, 1+\epsilon))$, where $\mathcal{R}_{i,\theta}=\pi_\theta/\pi_{\theta_{detach}}$ and $C=2$.
    - **Design Motivation**: Figure 2 shows that pure NoIS (treating heterogeneous samples as on-policy) results in performance drops for both 1.5B and 3B models. Token-level IS (VIS) works for 1.5B but causes degradation in 3B. TIS significantly outperforms VIS on 3B, suggesting that global ratio clamping is more stable than token-level IS for larger models.

2.  **KL Threshold-based Negative Advantage Filtering**:
    - **Function**: During the update phase, samples where $\hat{A}_i < 0$ and $\mathcal{D}_{KL}(\pi_\theta\Vert\pi_{\theta_{gen}}) > g$ have their advantage set to zero, so these tokens no longer contribute to the gradient (though they still participate in the mean/variance calculation for group advantage). Specifically, $\hat{A}_{t,i} = \hat{A}_i$ if $\hat{A}_i > 0$ or $\mathcal{D}_{KL} < g$, otherwise 0.
    - **Mechanism**: Samples with $\hat{A}_i > 0$ tell the model "this path is correct," which is useful even if off-policy. Off-policy samples with $\hat{A}_i < 0$ "punish tokens the current model wouldn't have generated," which pushes probability to other low-probability tokens and causes gibberish. The parameter $g$ controls the distance threshold (default $g=50$).
    - **Design Motivation**: Figure 3 shows that adding only this filtering (F-NoIS) prevents NoIS from collapsing, performing most of the stability work. F-TIS layers this on top of TIS for residual gains. Section 4.5 notes that smaller models prefer smaller $g$ (filtering earlier), while larger models prefer larger $g$ in later stages to leverage more off-policy exploration.

3.  **Vertical Collaboration Strategy**:
    - **Function**: Each node generates a full group of completions for a single prompt rather than contributing a subset of a group (horizontal).
    - **Mechanism**: Vertical collaboration ensures group advantage is computed using the local model's own $G$ rewards, preventing normalization bias from stronger or weaker models in the swarm.
    - **Design Motivation**: Section 4.7 demonstrates that horizontal F-TIS leads to significant degradation in 3B models, confirming vertical collaboration as the preferred paradigm for heterogeneous RL.

### Loss & Training
The final F-TIS loss is defined as:
$$\mathcal{L}_{F\text{-}TIS} = \frac{1}{G}\sum_i \frac{1}{|a_i|}\sum_t \min\big(\pi_\theta/\pi_{\theta_{gen}},\;C\big)\cdot \min\big(\mathcal{R}_{i,\theta}\hat{A}_{t,i},\;\text{clip}(\mathcal{R}_{i,\theta}\hat{A}_{t,i},\;1-\epsilon,\;1+\epsilon)\big)$$
Following DR-GRPO, the **KL term is omitted** as it provides no stability gains while consuming memory. Hyperparameters: LR $1\times 10^{-6}$, group size 12, batch size 16/24, $\epsilon=0.2$, $C=2$, $g=50$, binary reward (correct format + correct answer = 1). Dataset: GSM8K, 50 iterations, validated with pass@1 greedy decoding.

## Key Experimental Results

### Main Results
All experiments were conducted using vertical decentralized RL. Two models were trained jointly on GSM8K and evaluated on MATH-500 for OOD performance.

| Setting | Model | Alone (MATH-500) | F-TIS Collaborative (MATH-500) | Change |
| :--- | :--- | :--- | :--- | :--- |
| Size: 1.5B + 3B Base | 1.5B Base | 0.406 | 0.470 | **+6.4%** |
| | 3B Base | 0.575 | 0.540 | -3.5% |
| Size: 1.5B + 3B Coder | 1.5B Coder | 0.410 | 0.470 | **+6.0%** |
| | 3B Coder | 0.478 | 0.590 | **+11.2%** |
| Expertise: 1.5B Base + 1.5B Coder | 1.5B Base | 0.406 | 0.403 | -0.3% |
| | 1.5B Coder | 0.410 | 0.410 | 0 |
| Expertise: 3B Base + 3B Coder | 3B Base | 0.575 | 0.520 | -5.5% |
| | 3B Coder | 0.478 | 0.530 | **+5.2%** |
| Trainable: 1.5B + 1.5B PEFT | 1.5B Base | 0.406 | 0.430 | +2.4% |
| | 1.5B PEFT | 0.412 | 0.430 | +1.8% |
| Trainable: 3B + 3B PEFT | 3B Base | 0.575 | 0.513 | -6.2% |
| | 3B PEFT | 0.500 | 0.560 | **+6.0%** |

> On in-distribution GSM8K, F-TIS validation curves eventually align with the individual baselines (Figures 4–9), though initial convergence is slower due to the filtering of high-variance samples.

### Ablation Study

| Configuration | Key Observation | Description |
| :--- | :--- | :--- |
| NoIS (No IS) | Both 1.5B + 3B collapse | Heterogeneous off-policy noise breaks GRPO |
| VIS (Token-level IS) | 1.5B stable, 3B worse than TIS | High variance in token-level IS for large models |
| TIS (No filtering) | More stable than NoIS/VIS, but < baseline | Clamping helps but is insufficient for OOD samples |
| F-NoIS (Filtering only) | Close to baseline | Filtering is the primary driver of stability |
| **F-TIS (Full)** | Matches baseline, better on OOD | TIS and filtering are complementary |
| F-VIS (Filtering + VIS) | Fast start, severe late-stage drop | Confirms TIS as a better foundation than VIS |
| Horizontal F-TIS | 3B significantly degrades | Swarm-averaged advantage introduces bias |
| $g\in\{5,10,50,100\}$ | 1.5B prefers small $g$, 3B prefers large $g$ | Capacity determines tolerance for off-policy exploration |

### Key Findings
- F-TIS unexpectedly **improves OOD math capability** in several pairings (e.g., 3B Coder + 1.5B Coder improved 3B Coder on MATH-500 by +11.2%). This is attributed to the fact that training on coder models alone can lead to overfitting to coding styles; external samples preserve reasoning diversity.
- **Filtering ($g$)** is the primary factor for stability, while the IS form (VIS vs TIS) determines the performance ceiling. Both are required.
- Vertical collaboration + local group advantage is a hidden prerequisite for heterogeneous GRPO; horizontal collaboration fails due to mixed-model normalization.
- Low communication cost (8 bytes/token) makes "decentralized RL across WAN" technically feasible.

## Highlights & Insights
- **F-TIS as a Minimum Viable Patch**: The method does not change generation, rewards, or architecture. It simply adds a clamp and a mask to the loss, enabling GRPO to handle large off-policy gaps.
- **$g$ as a Capacity-Related Hyperparameter**: Small models need strict filtering (small $g$) early on to avoid confusion, while large models benefit from relaxed filtering later to explore off-policy signals. This suggests $g$ could be dynamically scheduled.
- **Passive Regularization in OOD**: Heterogeneous collaboration is a "win" for weaker models and acts as "passive regularization" for stronger models—the companion's off-policy samples serve as anti-overfitting perturbations. This is particularly effective in PEFT; using full-parameter off-policy samples to train LoRA increased 3B PEFT performance by 6%.
- **Vertical vs. Horizontal Bias**: Advantage normalization across models (horizontal) confuses "is this a good sample for this model" with "is this a good sample for the swarm," introducing systemic reward shaping that causes collapse in heterogeneous settings.

## Limitations & Future Work
- Experiments were conducted only on Qwen2.5 (1.5B/3B/Coder/PEFT) across two datasets (GSM8K/MATH-500). Generalization to other model families or tasks (Agent, dialogue) remains unverified.
- The tradeoff where stronger models occasionally lose performance while weaker ones gain lacks a theoretical framework. Without incentive alignment, nodes with stronger models might lack motivation to join a decentralized swarm.
- The threshold $g=50$ is a fixed constant, though ablations suggest it should vary by size/stage. An automatic scheduling policy is missing.
- KL estimation $\mathcal{D}_{KL}(\pi_\theta\Vert\pi_{\theta_{gen}})$ uses per-sequence values which might be sensitive to generation length.
- Lack of wall-clock speed benchmarks: The true benefit of decentralized compute is wall-clock time reduction via heterogeneous resources, but throughput comparisons were not provided.

## Related Work & Insights
- **vs. `yao2025efficient_rl_offpolicy` (TIS)**: Original TIS targeted small off-policy drift between generator and trainer; this work extends it to large gaps between different models and identifies the necessity of filtering.
- **vs. DeepSeek-v3 / HTTT Filtering**: While they used filtering for single-model stability, this work uses it as a stabilizer for heterogeneous collaboration.
- **vs. Distributed GRPO (LlamaRL, etc.)**: These frameworks assume homogeneity. This is the first work to systematically address "distributional heterogeneity" in model size, expertise, and parameters.
- **vs. PEFT Tuning**: Traditional LoRA+GRPO is typically isolated. This work shows that training LoRA with full-parameter off-policy samples boosts the final capacity of the PEFT model.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combination of TIS, filtering, and vertical decentralization to solve heterogeneous collapse).
- Experimental Thoroughness: ⭐⭐⭐ (Covers three types of heterogeneity but limited to Qwen2.5/Math).
- Writing Quality: ⭐⭐⭐⭐ (Clear chain of logic in model development).
- Value: ⭐⭐⭐⭐ (A practical, minimum viable recipe for decentralized LLM post-training).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](../../ACL2026/llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ACL 2026\] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting](../../ACL2026/llm_alignment/taming_extreme_tokens_covariance-aware_grpo_with_gaussian-kernel_advantage_rewei.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] UDM-GRPO: 统一离散扩散模型的稳定高效 GRPO](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](../../ACL2026/llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ACL 2026\] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting](../../ACL2026/llm_alignment/taming_extreme_tokens_covariance-aware_grpo_with_gaussian-kernel_advantage_rewei.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)

</div>

<!-- RELATED:END -->
