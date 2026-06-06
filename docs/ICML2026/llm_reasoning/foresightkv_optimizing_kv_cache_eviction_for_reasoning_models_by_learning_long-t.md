---
title: >-
  [Paper Note] ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution
description: >-
  [ICML 2026][LLM Reasoning][KV cache eviction] ForesightKV trains a lightweight scoring model to dynamically evict KV pairs based on "future attention contribution." It first distills an optimal eviction sequence from ful…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "KV cache eviction"
  - "reasoning models"
  - "Golden Eviction"
  - "GRPO"
  - "long-term contribution prediction"
date: 2026-05-08
content_hash: b526cefccf78ca68
---

# ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution

**Conference**: ICML 2026  
**arXiv**: [2602.03203](https://arxiv.org/abs/2602.03203)  
**Code**: https://github.com/RUCAIBox/ForesightKV  
**Area**: LLM Efficiency / KV Cache Compression / Long-context Reasoning  
**Keywords**: KV cache eviction, reasoning models, Golden Eviction, GRPO, long-term contribution prediction

## TL;DR
ForesightKV trains a lightweight scoring model to dynamically evict KV pairs based on "future attention contribution." It first distills an optimal eviction sequence from full traces using the Golden Eviction algorithm to serve as supervision signals. It then fine-tunes the strategy via GRPO reinforcement learning using the "sum of squared loss increments for low-entropy tokens" as a reward. On AIME2024/2025, ForesightKV outperforms SnapKV/H2O/R-KV with only half the KV budget, and a 4K budget retains 99% of the original model performance.

## Background & Motivation
**Background**: Reasoning LLMs (e.g., DeepSeek-R1, Qwen3 series) achieve breakthroughs in mathematics and coding tasks by generating 8K–32K token Chain-of-Thought (CoT). However, KV cache grows linearly with every token generated—a Qwen3-4B model consumes 4.5 GB of BFloat16 VRAM for a single sample at 32K length, significantly limiting concurrent batch sizes. Since decoding is memory-bound, transferring massive KV caches also slows down throughput. Mainstream solutions involve KV cache eviction: permanently discarding a portion of KV pairs every few steps based on an "importance score" to compress the cache back to budget $B$. Representative methods include SnapKV (using recent window attention), H2O (using accumulated attention), and R-KV (designed for reasoning models).

**Limitations of Prior Work**: Training-free rule-based methods rely on heuristics (recent window attention, accumulated attention, position, etc.) to estimate KV importance, failing to capture the complex attention patterns in reasoning data. The authors observe three types of heads in Qwen3-4B: global (vertical stripes), position-dependent (local), and semantic-dependent (blocky and dynamically switched). SnapKV, which uses the recent tokens' attention as an observation window, discards KV pairs that are "semantically unrelated to the current window but important for the future," causing significant performance drops. Another line of research, such as DMC, evaluates importance only once at the sequence start, failing to capture dynamic shifts in importance during generation.

**Key Challenge**: KV importance is a **function of future attention scores**, whereas all existing methods only perceive **historical** information, essentially "using the past to predict the future." Furthermore, eviction harms **low-entropy tokens** (the top-80% most certain detail tokens) far more than high-entropy decision tokens. Table 1 shows that under the same budget, the loss of low-entropy tokens increases by 147% while high-entropy loss only grows by 52%. Errors in numbers, symbols, and entities—which often appear asFacts in the preceding text—tend to derail subsequent reasoning processes if lost.

**Goal**: (1) Identify a "golden standard" eviction strategy that utilizes future information to provide training supervision; (2) Train a lightweight scoring model to distill "future-awareness" into a strategy that only considers the current state; (3) Align the training objective with the loss of low-entropy tokens that truly impact reasoning quality.

**Key Insight**: Since offline traces contain complete future attention, an oracle can be used to construct a "Golden Eviction" sequence that minimizes future damage. This sequence provides supervised pairwise ranking for an MLP scorer. The entire decoding process is then modeled as an MDP and fine-tuned using GRPO on group-relative advantages, specifically targeting "low-entropy tokens with severe degradation."

**Core Idea**: By combining two-stage training—oracle future attention distillation and low-entropy token loss rewards—an MLP learns to "predict the long-term contribution of each KV pair," upgrading rule-based eviction to data-driven strategy learning.

## Method

### Overall Architecture
The ForesightKV pipeline:

**Inference**: For every $L$ new tokens generated ($L=256$), the KV cache grows to $B+L$. For each attention group (GQA shared KV) in every layer, the scoring model $\pi_\theta$ inputs features $\mathbf{x}_n = \text{Concat}(\mathbf{k}_n, \mathbf{v}_n, \mathbf{a}_n)$ (where $\mathbf{a}_n$ represents fixed-length statistical features of attention scores) and outputs an importance score $\phi_n$. The $L$ most recent pairs are retained as mandatory, and $L$ pairs are evicted from the remainder using a Top-$2L$ + multinomial sampling strategy to compress the cache back to budget $B$. The LLM parameters are frozen, and only the lightweight MLP scorers are trained.

**Training**: A two-stage process. The first stage calculates optimal eviction sequences offline via Golden Eviction for supervised pairwise ranking. The second stage treats eviction as an MDP and fine-tunes the model via GRPO using "severe degradation of low-entropy tokens" as a negative reward.

### Key Designs

1. **Golden Eviction (Oracle Distillation via Future Attention)**:
    - **Function**: Labels which $L$ KV pairs should be discarded at each eviction step within a full reasoning trace $\mathbf{w}=\{w_1,\dots,w_T\}$ to supervise the scoring model.
    - **Mechanism**: The full-length attention matrix $\mathbf{A}^h \in \mathbb{R}^{T\times T}$ is computed on the original model. Query dimensions are partitioned into blocks of step size $L$. Average pooling is applied to each block within each GQA group head to obtain a block score $\tilde{\mathbf{a}}_t^{h'}$. For an eviction at step $t$, each KV pair $i$ is assigned a "future score" $\alpha_{i,t}^{h'} = \max_{t\le j\le M}(\tilde{\mathbf{a}}_{i,j}^{h'})$ based on the **maximum score across all future blocks**. The $B-L$ pairs with the largest $\alpha$ are kept. Appendix A proves this strategy minimizes the impact on the output upper bound. Training utilizes a Pairwise Ranking Loss $\mathcal{L}_{\text{supervised}} = \sum_t \sum_{\alpha_i < \alpha_j} \max(0, m-(\phi_i - \phi_j))$ to align the scorer's predictions with the oracle's future score ranking.
    - **Design Motivation**: Conventional methods are inherently limited by "using history to estimate future importance," missing dynamic patterns. Golden Eviction uses future attention as ground truth. Table 2 shows its loss ratio is only 1.07 (compared to 1.41+ for R-KV/SnapKV), confirming that the supervision signal is an order of magnitude stronger than baselines.

2. **MDP + GRPO Reinforcement Learning for Distribution Alignment**:
    - **Function**: Bridges the gap between "oracle trajectories" used in supervision and the "drifted distributions" caused by the scorer's own selections during inference, while shifting the training objective to "improving actual reasoning quality."
    - **Mechanism**: KV eviction is modeled as an MDP—the state $s_t$ is the current KV cache, the action $a_t$ is selecting $B$ pairs to retain from $B+L$, and the policy is a per-group scorer $\pi_{\theta_{h,l}}$. The reward focuses on the observation in §2.2: tokens are filtered into a set $E=\{w_t \mid w_t \in \mathbf{w}_\text{low},\ \Delta\mathcal{L}(w_t)>\eta\}$ where the original entropy was in the bottom 80% (low entropy) and the loss increment after eviction exceeds threshold $\eta$. The reward is defined as the negative sum of squared loss increments (MSE) on this subset: $R_t = -\sum_{t\in E}[\Delta\mathcal{L}(w_t)]^2$. GRPO samples $G$ different trajectories on the same sequence to calculate group-relative advantages $\hat A_t = (R_t - \text{Mean})/\text{Std}$.
    - **Design Motivation**: (1) Scored trajectories encounter state distributions not seen during supervised training, requiring on-policy correction. (2) Reward ablations in Table 4 show that blindly reducing total loss degrades performance (50.6 vs base 51.7), and targeting high-entropy tokens is even worse (49.6). Only the MSE reward $\mathcal{L}_\text{ours}$ focusing on **low-entropy + severe degradation** successfully pushes AIME24 performance from 51.7 to 54.5, validating that low-entropy tokens are the true bottleneck.

3. **Top-K Multinomial Sampling for Action Parameterization**:
    - **Function**: Decides specifically which KV pairs to discard after the scorer outputs scores $\Phi$, ensuring stability while allowing for RL exploration.
    - **Mechanism**: $\mathcal{D}_t = \text{Multinomial}_L(\text{Softmax}(\text{Top}_{2L}(-\Phi)))$. The $2L$ pairs with the lowest scores are selected as high-confidence candidates, and $L$ pairs are then sampled within this pool based on the softmax of their negative scores.
    - **Design Motivation**: Pure top-$K$ is greedy and sensitive to local ranking errors, while pure multinomial is too noisy for irrevocable KV eviction. This hybrid approach prunes obviously bad KV pairs first while maintaining controlled randomness for GRPO trajectory diversity.

### Loss & Training
Supervision stage: Pairwise Ranking Loss (Eq. 6) with margin $m$. RL stage: GRPO objective:
$$\mathcal{J}(\theta) = \mathbb{E}_{o\sim\pi_{\theta_\text{old}}}\sum_t \min(r_t(\theta)\hat A_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t) - \beta\cdot \text{KL}[\pi_\theta\|\pi_\text{ref}]$$
where $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$. Each attention group has an independent scorer (MLP, hidden dimension 16), while the LLM remains frozen. The training budget is $B\le 2K$.

## Key Experimental Results

### Main Results
Models: DeepSeek-R1-Distill-Qwen-7B, Qwen3-4B, Qwen3-1.7B. Benchmarks: AIME2024 / AIME2025. Metric: Pass@1 (average over 32 trials).

| Setting (Qwen3-4B, AIME24) | Budget | Pass@1 | Comparison |
|------------------------|------|--------|------|
| Full KV | 32K | 55.6 | Baseline Upper Bound |
| R-KV | 2K | 44.8 | Prev. SOTA |
| **ForesightKV** | **1K** | **54.5** | Overtakes R-KV by 9.7 pts with half budget |
| **ForesightKV** | **4K** | ≈Full | Retains ~99% of performance |
| **ForesightKV** | 2K | — | Retains ~92% |

Efficiency (Qwen3-4B, A800, 32K generation):

| Method | Budget | Max Batch Size | Throughput | Gain |
|------|------|--------------|------|------|
| Full | — | 11 | 37.73 | 1.00× |
| ForesightKV | 4K | 48 | 193.95 | 5.14× |
| ForesightKV | 2K | 70 | 268.36 | 7.11× |
| ForesightKV | 1K | 96 | 369.43 | **9.79×** |

### Ablation Study

| Dimension | Setting | AIME24 | Notes |
|----------|------|--------|------|
| Reward Function | Base (SL only) | 51.7 | Pre-RL |
| | $-\mathcal{L}_\text{all}$ | 50.6 | Total loss fails |
| | $-\mathcal{L}_\text{low}$ | 53.5 | Low-entropy only |
| | $-\mathcal{L}_\text{high}$ | 49.6 | High-entropy backfires |
| | $-\mathcal{L}_\text{low,large}$ | 53.8 | Target degradation |
| | $-\mathcal{L}_\text{ours}$ (MSE) | **54.5** | MSE best at penalizing catastrophes |
| Features | Attn-only | ↓ | Inaccurate without KV |
| Sampling | Pure Top-K | ↓ | Exploration deficiency |
| | Pure Multinomial | ↓ | Excess noise |
| | Top-K+MN | **best** | Balanced stability and exploration |

Golden Eviction Comparison (Qwen3-4B, lower loss ratio is better):

| Method | (1024,256) | (2048,256) |
|------|-----------|-----------|
| **Golden** | **1.0711** | **1.0166** |
| R-KV | 1.4101 | 1.1606 |
| SnapKV | 1.4091 | 1.1281 |
| H2O | 1.2730 | 1.0948 |

### Key Findings
- **Reward design is the key to RL success**: Blindly optimizing for total loss leads to performance drops. It is necessary to target the MSE of "low-entropy and severely degraded" tokens. This validates the observation that low-entropy tokens dominate reasoning quality, contrary to the intuition that "high-entropy tokens are the decision points." Errors in decision points affect one branch, but errors in facts or numbers contaminate all subsequent reasoning.
- **Strong Generalization**: Scorer models trained at budget $B\le 2K$ still retain 99% performance under a 4K budget, indicating the model learns "intrinsic importance" rather than overfitting to a specific budget.
- **Smaller Budgets Yield Higher Speedups**: A 1K budget on 32K generation increases throughput to 9.79× because decoding is memory-bound; smaller KV caches allow for larger concurrent batches and less HBM transfer.

## Highlights & Insights
- **"Oracle Distillation + On-policy RL"** is an elegant application of the data-driven control paradigm to KV eviction: the supervision stage solves the "unknown future" via oracle future information, and the RL stage corrects distribution drift.
- **Transferability of Reward Engineering**: Decomposing loss into the "low vs. high entropy × degradation magnitude" quadrants reveals that only "low-entropy severe degradation" requires optimization. This philosophy is reusable in LLM alignment, cache strategies, and temperature scheduling.
- **Top-K + Multinomial "Prune then Sample"**: This is a practical discrete action parameterization for discrete control problems involving a large number of "junk" candidates and a few "boundary" candidates (e.g., pruning, routing, or frame selection).
- **Minimalist Scorer Design**: With an MLP hidden layer of only 16, the overhead is negligible. Since input features already encode semantic information, the model only needs to learn a relative ranking.

## Limitations & Future Work
- **Reliance on Full Traces for Golden Eviction**: Data collection is expensive as it requires running long-context inferences and storing $T\times T$ attention matrices.
- **Domain Narrowness**: Primarily validated on mathematical reasoning. The "low-entropy token dominance" hypothesis requires further testing on coding, agents, and long-document QA tasks.
- **Per-head Scorer Scalability**: Scorer parameters scale with layers and groups. Cross-layer sharing or LoRA-style parameterization should be explored.
- **Irreversibility of Eviction**: Discarded KV pairs cannot be recovered.
- **Manual Hyperparameters**: Budgets and eviction intervals $L$ remain manual; adaptive budgets based on sentence difficulty are a natural next step.

## Related Work & Insights
- **vs. SnapKV / H2O**: These are training-free rule-based methods. ForesightKV upgrades "estimating the future from history" to "learning the mapping to the future," showing its greatest advantage in semantic-dependent heads.
- **vs. R-KV**: Also designed for reasoning models, but ForesightKV outperforms R-KV by 9 points with half the budget, proving data-driven > rules.
- **vs. DMC / Lancucki et al.**: Those methods use a one-time assessment, missing dynamic evolution. ForesightKV's periodic re-scoring matches the dynamic switching patterns of reasoning heads.
- **vs. Token Merging / KV Quantization**: These reduce per-pair storage costs and are orthogonal/complementary to ForesightKV's reduction of pair quantity.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Golden Oracle distillation and low-entropy MSE reward is a first in KV eviction literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and ablations across multiple budgets and models, though lacks non-reasoning tasks.
- Writing Quality: ⭐⭐⭐⭐ Powerful motivation (KV patterns + low-entropy loss spikes) and clear figures.
- Value: ⭐⭐⭐⭐⭐ 9.79× throughput with half budget and no performance loss makes it a highly practical plug-and-play solution for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ICML 2026\] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)

</div>

<!-- RELATED:END -->
