---
title: >-
  [Paper Note] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation
description: >-
  [ICLR 2026][LLM Reasoning][GRPO] This paper identifies that the advantage function in GRPO (std normalization) causes update magnitudes to peak at medium-difficulty problems while implicitly suppressing updates on both hard and easy problems. To address this, the authors propose MathForge — combining DGPO (replacing std with MAD for difficulty-balanced normalization + softmax difficulty weighting) and MQR (question reformulation via three aspects: narrative context, abstract terminology, and nested sub-problems, increasing difficulty while preserving original answers). On Qwen2.5-Math-7B, MathForge outperforms GRPO by an average of +4.56% across six mathematical reasoning benchmarks.
tags:
  - ICLR 2026
  - LLM Reasoning
  - GRPO
  - difficulty-aware
  - mathematical reasoning
  - RLVR
  - data augmentation
date: 2026-05-08
content_hash: 211bbd9d37bf9969
---

# Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation

**Conference**: ICLR 2026
**arXiv**: [2601.20614](https://arxiv.org/abs/2601.20614)
**Code**: [GitHub](https://github.com/AMAP-ML/MathForge)
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: GRPO, difficulty-aware, mathematical reasoning, RLVR, data augmentation

## TL;DR

This paper identifies that the advantage function in GRPO (std normalization) causes update magnitudes to peak at medium-difficulty problems while implicitly suppressing updates on both hard and easy problems. To address this, the authors propose MathForge — combining DGPO (replacing std with MAD for difficulty-balanced normalization + softmax difficulty weighting) and MQR (question reformulation via three aspects: narrative context, abstract terminology, and nested sub-problems, increasing difficulty while preserving original answers). On Qwen2.5-Math-7B, MathForge outperforms GRPO by an average of +4.56% across six mathematical reasoning benchmarks.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become the dominant paradigm for enhancing LLM mathematical reasoning (e.g., DeepSeek-R1). GRPO is the most representative algorithm in this paradigm, replacing value networks with group-relative advantage estimation.

**Limitations of Prior Work**:

1. **Algorithmic level**: The GRPO advantage function $\hat{A}_{GR,i} = \frac{r_i - \text{mean}}{\text{std}}$ uses standard deviation normalization, resulting in per-question total update magnitude $\sum|A| = 2G\sqrt{p(1-p)}$ — maximized at $p=0.5$ and decaying as $p$ approaches 0 or 1. This means harder questions (small but nonzero $p$) receive smaller gradient updates than medium-difficulty ones.

2. **Data level**: Existing RLVR data augmentation methods (e.g., Liang et al. 2025) primarily paraphrase questions to improve diversity without systematically increasing difficulty. The lack of challenging training data limits the upper bound of model reasoning capability.

**Key Challenge**: Hard but solvable problems are the most valuable training material (exposing model weaknesses while providing correct answers to learn from), yet GRPO produces the smallest update magnitudes precisely on such problems.

**Key Insight**: The paper addresses the "neglect of hard problems" simultaneously at the algorithmic and data levels — DGPO corrects GRPO's inherent imbalance and up-weights hard problems, while MQR generates harder training questions.

## Method

### Overall Architecture

Original training data → MQR (three-aspect reformulation to increase difficulty while preserving original answers) → augmented dataset (original + reformulated) → DGPO training (MAD normalization + difficulty weighting + valid-token averaging) → enhanced policy model. MathForge forms a synergistic loop: MQR expands the difficulty frontier of the data, and DGPO efficiently learns from the augmented data.

### Key Designs

1. **DGPO: Difficulty-aware Group Policy Optimization**

    - **Difficulty-balanced Group Advantage Estimation (DGAE)**: Replaces GRPO's standard deviation normalization with mean absolute deviation (MAD) normalization: $\hat{A}_{DG,i} = \frac{r_i - \text{mean}(\{r_i\})}{\text{MAD}(\{r_i\})}$, where $\text{MAD} = \frac{1}{G}\sum|r_i - \text{mean}|$.
    - **Theorem 2**: Under DGAE, the total update magnitude per question $\sum|\hat{A}_{DG,i}| = G$ is constant and independent of difficulty — completely eliminating the bell-shaped bias $2G\sqrt{p(1-p)}$ in GRPO, without requiring a binary reward assumption.
    - **Difficulty-aware Question-level Weighting (DQW)**: On top of the balanced estimation, DQW further prioritizes hard questions via softmax weighting: $\lambda_s = B_v \cdot \frac{\exp(D_s/T)}{\sum\exp(D_s/T)}$, where $D_s = -\text{mean}(\{r_{si}\})$ is the difficulty measure and $T=2.0$ is the temperature.
    - Valid-token-level averaging: Loss is computed as token-level average only over valid queries (neither all-correct nor all-incorrect), preventing gradient instability.

2. **MQR: Multi-aspect Question Reformulation**

    - A large reasoning model (o3 by default) reformulates training questions along three dimensions:
        - **Adding narrative context**: Embeds narrative noise, challenging the model to extract key mathematical information from distractors.
        - **Introducing abstract terminology**: Abstracts concrete concepts, challenging the model to comprehend abstract mathematical notions.
        - **Nesting sub-problems**: Increases the number of reasoning steps and cross-domain knowledge requirements.
    - Key constraint: All reformulations must preserve the original gold answer, eliminating the need for answer regeneration.
    - Design Motivation: Mathematical reasoning requires diverse skills; systematically increasing question difficulty pushes the boundary of model performance.

### Loss & Training

DGPO objective:

$$\mathcal{J}_{DGPO}(\theta) = \frac{1}{\sum_{s=1}^{B_v}\sum_{i=1}^{G}|o_{si}|}\sum_{s=1}^{B_v}\lambda_s\sum_{i=1}^{G}\sum_{t=1}^{|o_{si}|}\min[I_{sit}\hat{A}_{DG,si}, \text{clip}(I_{sit}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{DG,si}]$$

- Pure accuracy reward ($r \in \{0,1\}$), no KL divergence.
- 8× NVIDIA H20 GPUs, built on the Open-R1 codebase.
- DQW temperature $T=2.0$ (ensuring the max/min weight ratio within a batch $\leq e^{0.5} \approx 1.65$).

## Key Experimental Results

### Main Results

Qwen2.5-Math-7B trained on the MATH dataset, averaged over 6 benchmarks:

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg. | $\Delta_{GRPO}$ |
|--------|--------|--------|-------|---------|---------|----------|------|----------------|
| Base | 12.19 | 4.79 | 35.23 | 48.60 | 15.07 | 16.33 | 22.04 | - |
| GRPO | 20.94 | 8.44 | 58.98 | 72.20 | 27.76 | 37.33 | 37.61 | - |
| Dr.GRPO | 21.04 | 8.23 | 58.59 | 72.05 | 28.58 | 35.89 | 37.40 | -0.21 |
| DAPO | 21.25 | 8.75 | 58.20 | 72.70 | 29.50 | 37.22 | 37.94 | +0.33 |
| GRPO-AD | 21.56 | 9.48 | 59.06 | 73.25 | 29.14 | 37.07 | 38.26 | +0.65 |
| DGPO | 23.85 | 10.21 | 61.02 | 74.25 | 31.07 | 38.33 | 39.79 | **+2.18** |
| MQR | 25.00 | 11.77 | 59.38 | 77.85 | 31.43 | 40.81 | 41.04 | +3.43 |
| **MathForge** | 24.58 | 12.60 | 59.84 | 79.95 | 33.36 | 42.67 | **42.17** | **+4.56** |

### Ablation Study

Component ablation of DGPO (Qwen2.5-Math-7B):

| Setting | Avg. | $\Delta_{GRPO}$ |
|---------|------|----------------|
| GRPO | 37.61 | - |
| +Valid-token averaging | 37.71 | +0.10 |
| +DGAE | 38.65 | +1.04 |
| +DGAE+DQW (full DGPO) | 39.79 | +2.18 |

DQW temperature sensitivity: $T=1.0$ → 39.03, $T=2.0$ → **39.79**, $T=5.0$ → 39.53, $T=10.0$ → 39.27

Cross-model generalization (all surpass GRPO): Qwen2.5-Math-1.5B +4.45, Qwen2.5-3B +3.54, DeepSeek-Math-7B +2.86.

### Key Findings

- DGAE and DQW contribute +0.94% and +1.14% respectively, and are complementary.
- MathForge consistently outperforms GRPO across all four tested model architectures, demonstrating model-agnostic generalizability.
- DGPO is composable with other methods: +GPG → +0.99, +DAPO → +1.97, +GSPO → +1.61.
- Models trained with DGPO produce more concise outputs (Fig. 1b), suggesting they learn more efficient reasoning paths.

## Highlights & Insights

- The theoretical contributions are rigorous: Theorems 1 and 2 formally prove the bell-shaped update bias of GRPO and the constant-magnitude balance of DGAE, with clear mathematical derivations.
- The two-step "balance then weight" design (DGAE → DQW) is more effective than directly applying difficulty weighting on top of GRPO (e.g., GRPO-AD).
- The "answer-preserving" constraint in MQR is a critical design choice: it increases difficulty while avoiding answer regeneration, substantially reducing data augmentation cost.
- The synergy between DGPO and MQR is super-additive (42.17 > 39.79 + 41.04 − 37.61), rather than a simple sum.

## Limitations & Future Work

- MQR relies on a large reasoning model (o3) as the reformulator, increasing data augmentation cost.
- Validation is limited to mathematical reasoning; the approach has not been tested on other reasoning tasks such as code generation or logical reasoning.
- The temperature hyperparameter in DQW requires tuning (though $T=2.0$ proves robust across all experiments).
- MAD normalization is equivalent to std normalization when reward distributions are symmetric; its theoretical advantages are more pronounced under non-binary rewards but are not fully validated in that setting.

## Related Work & Insights

- **vs. GRPO**: GRPO's std normalization induces a bell-shaped update bias; DGPO achieves constant update magnitude via MAD — a simple yet theoretically grounded correction.
- **vs. GRPO-AD (Zhang & Zuo 2025)**: GRPO-AD applies difficulty weighting on top of GRPO without correcting the underlying imbalance, yielding limited gains (+0.65 vs. DGPO's +2.18).
- **vs. DAPO/GPG**: These methods address orthogonal aspects such as sampling strategies and KL divergence, and are composable with DGPO.
- **Data augmentation insight**: The "answer-preserving constraint" in MQR is a practical design principle — ensuring mathematical equivalence of augmented data.

## Rating

- Novelty: ⭐⭐⭐⭐ Theoretically grounded insights (Theorems 1/2); the MAD-for-std substitution is simple but well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks × four models × multiple ablations + dynamic analysis + composability experiments.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are tightly integrated; ablations are comprehensive.
- Value: ⭐⭐⭐⭐ A general improvement for RLVR training; DGPO can be directly incorporated into existing pipelines.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](../../ACL2026/llm_reasoning/ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)

<!-- RELATED:END -->
