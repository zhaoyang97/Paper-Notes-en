---
title: >-
  [Paper Note] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay
description: >-
  [NeurIPS 2025][LLM Alignment][LLM reinforcement fine-tuning] Two complementary techniques are proposed to improve the data efficiency of LLM reinforcement fine-tuning (GRPO): (1) DOTS — an attention-based mechanism for p…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "LLM reinforcement fine-tuning"
  - "data efficiency"
  - "adaptive difficulty"
  - "online data selection"
  - "experience replay"
  - "GRPO"
date: 2026-05-08
content_hash: a8b9d840133aa48b
---

# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay

**Conference**: NeurIPS 2025
**arXiv**: [2506.05316](https://arxiv.org/abs/2506.05316)  
**Code**: [GitHub](https://github.com/ASTRAL-Group/data-efficient-llm-rl/)  
**Area**: LLM Alignment
**Keywords**: LLM reinforcement fine-tuning, data efficiency, adaptive difficulty, online data selection, experience replay, GRPO

## TL;DR

Two complementary techniques are proposed to improve the data efficiency of LLM reinforcement fine-tuning (GRPO): (1) DOTS — an attention-based mechanism for predicting adaptive difficulty that prioritizes moderate-difficulty questions to maximize gradient signal; and (2) Rollout Replay — reusing recent rollouts to reduce per-step computational overhead. Together, these techniques reduce training time by an average of 40.7% across 6 model–dataset combinations.

## Background & Motivation

RL fine-tuning of LLMs (e.g., via GRPO) has become the dominant approach for improving reasoning capabilities, yet its computational cost remains prohibitively high:

**Staggering cost**: Luo et al. report that training a 1.5B model on 40K samples requires 3,800 A100 GPU hours (~$4,500).

**Data efficiency neglected**: Existing research focuses on algorithmic improvements (e.g., GRPO, L1) while rarely examining how to select more informative training data.

**Two sources of waste**: (a) questions that are too easy or too hard produce zero gradient signal (all rollout rewards are either 0 or 1); (b) all rollouts must be generated from scratch at every step, even when the policy changes only marginally.

Core observation: In GRPO, when all rollout rewards for a given question are identical, the normalized advantage is 0 and no gradient update occurs — these computations are entirely wasted.

## Method

### Overall Architecture

A two-stage optimization pipeline: DOTS reduces the number of training steps required to reach target performance, while Rollout Replay reduces the computation time per step.

### Key Design 1: Adaptive Difficulty Definition

Given policy $\pi_t$ and question $q$, $G$ responses are sampled and their rewards $\{r_i^{(t)}\}_{i=1}^G$ are obtained. Adaptive difficulty is defined as the mean failure rate:

$$d_q^{(t)} = \frac{1}{G} \sum_{i=1}^{G} (1 - r_i^{(t)})$$

$d_q = 0$ indicates all responses are correct (too easy); $d_q = 1$ indicates all responses are incorrect (too hard). A key theorem establishes that targeting $d_q = 0.5$ is optimal:

**Theorem 1 (Gradient signal is maximized at 50% success rate)**: For a Bernoulli(p) reward distribution, the expected squared norm of the unclipped policy gradient satisfies:

$$\mathbb{E}[\|g\|^2] \propto p(1-p) \cdot (1 - 1/G)$$

This is maximized at $p = 0.5$, i.e., gradient signal is strongest when adaptive difficulty equals 0.5.

### Key Design 2: Attention-based Difficulty Prediction

Computing adaptive difficulty for all questions directly would require generating rollouts for each, which is computationally prohibitive. An efficient prediction framework is proposed:

1. **Reference set sampling**: At each step, $K$ (e.g., 256) questions are randomly sampled as a reference set $\mathcal{D}_{\text{ref}}$; rollouts are executed only for these questions to obtain ground-truth difficulties.
2. **Embedding similarity prediction**: An embedding model $E_\theta$ encodes all questions; the difficulty of unlabeled questions is estimated via attention-weighted aggregation over the reference set:

$$a_i = \frac{\exp(z_q^\top z_i / \sqrt{h})}{\sum_{j=1}^K \exp(z_q^\top z_j / \sqrt{h})}, \quad \hat{d}_q^{(t)} = \sum_{i=1}^K a_i d_i^{(t)}$$

3. **Platt scaling**: A lightweight MLP learns scale and bias parameters conditioned on the mean and standard deviation of reference set difficulties, improving prediction accuracy:

$$\hat{d}_{q,\text{cal}}^{(t)} = \sigma\left(w^{(t)} \cdot \left(\log \hat{d}_q^{(t)} - \log(1 - \hat{d}_q^{(t)})\right) + b^{(t)}\right)$$

The embedding model consists of a frozen Qwen2.5-Math-1.5B-Instruct backbone with a 3-layer MLP adapter.

### Key Design 3: Difficulty-targeted Online Data Selection

Questions are sampled according to predicted difficulty, with higher probability assigned to those closer to 0.5:

$$P(q) = \frac{\exp(-|\hat{d}_q - \alpha| / \tau)}{\sum_{q' \in \mathcal{D}} \exp(-|\hat{d}_{q'} - \alpha| / \tau)}$$

where $\alpha = 0.5$ is the target difficulty and $\tau$ is a temperature parameter.

**Implicit diversity guarantee**: Questions repeatedly selected at moderate difficulty shift away from 0.5 as the model trains on them, naturally causing them to exit the selection pool and allowing other questions to be sampled.

### Key Design 4: Rollout Replay

At each step, only $\delta B$ (e.g., 50%) new rollouts are generated; the remainder are reused from a FIFO buffer. To address off-policy bias, a corrected GRPO loss is applied using importance sampling ratios relative to the behavior policy (i.e., the policy that originally produced each rollout) rather than the old policy:

$$\tilde{r}_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{behavior}}}(o_{i,t}|q, o_{i,<t})}$$

Only rollouts that produce non-zero gradient signal (i.e., those whose group-average reward is neither 0 nor 1) are stored in the buffer.

### Loss & Training

A modified GRPO loss is employed, removing standard-deviation normalization (to avoid bias) and incorporating importance sampling with respect to the behavior policy:

$$\mathcal{J}_{\text{GRPO-RR}}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left(\min(\tilde{r}_{i,t}\hat{A}_i, \text{clip}(\tilde{r}_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{\text{KL}}\right)\right]$$

## Key Experimental Results

### Main Results: Training Time Savings

| Model | Dataset | Step Reduction | Per-step Time Reduction | **Total Time Saving** |
|-------|---------|---------------|------------------------|----------------------|
| Qwen2.5-Math-1.5B | MATH | 16.67% | 11.71% | 26.25% |
| Qwen2.5-Math-1.5B | DeepScaleR | 43.33% | 11.69% | 49.85% |
| Qwen2.5-Math-1.5B | ORZ | 13.33% | 11.66% | 23.30% |
| Qwen2.5-3B | DeepScaleR | 26.67% | 11.52% | 35.10% |
| Qwen2.5-3B | DeepMath | 56.67% | 11.35% | **61.65%** |
| Qwen2.5-Math-7B | DeepScaleR | 40.00% | 13.39% | 48.03% |
| **Average** | | | | **40.7%** |

The best case (Qwen2.5-3B + DeepMath) achieves 61.65% total training time reduction with a 56.67% reduction in the number of steps.

### Difficulty Prediction Quality

| Model | Dataset | Pearson Correlation ρ |
|-------|---------|----------------------|
| Qwen2.5-Math-1.5B | MATH | 0.784 ± 0.024 |
| Qwen2.5-Math-1.5B | DeepScaleR | 0.724 ± 0.032 |
| Qwen2.5-3B | DeepScaleR | 0.779 ± 0.019 |
| Qwen2.5-3B | DeepMath | 0.703 ± 0.008 |
| Qwen2.5-Math-7B | DeepScaleR | 0.708 ± 0.020 |

All settings yield ρ > 0.7, demonstrating that the attention-based prediction framework effectively tracks difficulty dynamics as the policy evolves.

### Ablation Study

1. **DOTS alone**: Steeper learning curves and faster convergence (13–57% step reduction).
2. **Rollout Replay alone**: Approximately 20% per-step time reduction (rollout generation accounts for 46–54% of total step time).
3. **Proportion of effective questions**: DOTS selects on average 25.4% more "effective questions" (those with difficulty strictly between 0 and 1) than vanilla GRPO.
4. **Superiority over external difficulty labels**: DOTS consistently outperforms difficulty-curriculum methods based on GPT-4o-mini annotations.

### Key Findings

1. **DOTS and RR are complementary**: DOTS accelerates convergence (fewer steps); RR reduces per-step overhead (fewer rollouts generated).
2. **Generalization beyond mathematics**: Both techniques are effective on scientific QA (SCP-25K dataset; MMLU physics/chemistry/biology subsets).
3. **Negligible prediction overhead**: With cached embeddings, difficulty prediction for 10K samples requires only 1.71 seconds.
4. **Buffer size sensitivity**: $C \in \{256, 512\}$; too small leads to a high proportion of stale rollouts, while too large increases memory overhead.

## Highlights & Insights

1. **Rigorous theoretical grounding**: Theorem 1 formally justifies targeting a 50% success rate by maximizing expected gradient norm.
2. **Elegant attention-based prediction**: Rolling out only 256 reference questions suffices to predict difficulty across the entire dataset, with minimal computational overhead.
3. **Implicit curriculum learning**: DOTS naturally induces a dynamic curriculum — questions mastered by the model drift away from the 0.5 difficulty target and exit the selection pool, giving way to new questions.
4. **Plug-and-play practicality**: Both techniques are drop-in additions that do not alter the core GRPO algorithm.
5. **Advantage over external annotations**: Adaptive difficulty outperforms static GPT-4o-mini difficulty labels because it tracks the evolving policy throughout training.

## Limitations & Future Work

1. **Validated only on GRPO**: Although the theoretical framework is general, experiments are limited to GRPO; applicability to PPO and other RL algorithms remains unverified.
2. **Binary reward assumption**: Theorem 1 assumes $r_i \in \{0, 1\}$; extension to continuous or partial reward signals is not discussed.
3. **Embedding model dependency**: Qwen2.5-Math-1.5B-Instruct is used as the encoder; adapting to other domains may require retraining the MLP adapter.
4. **Staleness risk in Rollout Replay**: When the policy changes rapidly, stale rollouts in the buffer may introduce bias; the clipping mechanism provides only an approximate remedy.
5. **No final performance comparison**: The paper focuses on "time to reach equivalent performance" but does not explore whether extended training yields superior results.

## Related Work & Insights

- The success of DeepSeek-R1 (Guo et al., 2025) established GRPO as the dominant LLM RL paradigm; the data efficiency improvements proposed here directly lower the barrier to its deployment.
- Experience replay has a long history in classical RL (e.g., DQN); the proposed Rollout Replay mechanism is a natural adaptation to the LLM RL setting.
- The principle of adaptive curriculum learning (zone of proximal development) has well-established analogues in both educational psychology and RL research.
- The attention-based difficulty prediction framework may also be applicable to preference data selection in RLHF settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The attention-based difficulty prediction is a standout contribution; other components represent reasonable combinations of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Six model–dataset combinations, multi-dimensional ablations, and generalization to non-mathematical domains.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play integration with an average 40% training time reduction makes this highly valuable for resource-constrained teams.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, strong alignment between theory and experiments, and intuitive figures.
- **Overall**: ⭐⭐⭐⭐ — A practically impactful contribution that makes a meaningful advance in the underexplored direction of LLM RL training efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[ICML 2026\] GIST: Targeted Data Selection for Instruction Tuning via Gradient Subspace Projection](../../ICML2026/llm_alignment/gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
