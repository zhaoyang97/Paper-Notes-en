---
title: >-
  [Paper Note] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
description: >-
  [ICLR 2026][Interpretability][attention sink] ZeroTuning is proposed to enhance LLM performance across 15 datasets without training by applying head-specific scaling to the attention scores of the initial token (e.g., `<BOS>`), requiring only 4 lines of code modifications.
tags:
  - ICLR 2026
  - Interpretability
  - attention sink
date: 2026-05-08
content_hash: f8bc45c2f030bc6d
---
# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2505.11739](https://arxiv.org/abs/2505.11739)
- **Code**: [https://anonymous.4open.science/r/ZeroTuning](https://anonymous.4open.science/r/ZeroTuning)
- **Area**: Interpretability
- **Keywords**: Attention tuning, initial token, attention sink, zero-training enhancement, head specificity

## TL;DR
ZeroTuning is proposed to enhance LLM performance across 15 datasets without training by applying head-specific scaling to the attention scores of the initial token (e.g., `<BOS>`), requiring only 4 lines of code modifications.

## Background & Motivation

### Core Problem
While token-level attention tuning (e.g., PASTA, ACT) is effective, it relies on **external heuristics** to identify task-specific "important" tokens, introducing bias and limiting applicability. Is there a **universal, task-agnostic control point**?

### Attention Sink Phenomenon
Initial tokens tend to serve as attention sinks (attention convergence points), yet their potential for performance enhancement remains untapped.

### Key Findings
- Adjusting the attention of the initial token consistently produces the largest and most stable gains.
- The direction of gain depends on the task: classification tasks require upward adjustment ($\gamma > 1$), while QA tasks require downward adjustment ($\gamma < 1$).
- Different attention heads respond heterogeneously to the scaling of the initial token.

## Method

### Overall Architecture

ZeroTuning treats the initial token (e.g., `<BOS>`) as a unified, task-agnostic attention lever. It first applies head-specific scaling to the attention scores of this token using a scaling factor $\gamma$, followed by softmax renormalization. This process does not modify any weights, only adjusts the attention distribution, and can be implemented with just 4 lines of code. The underlying logic is that the initial token acts as an attention sink; tuning its weight proportionally reshapes the relative attention differences between all other tokens, thereby correcting model biases without introducing any external "important token" heuristics.

### Key Designs

The paper summarizes the mothod into three steps: head behavior profiling → selective rescaling → renormalization. The calibration process includes both supervised and unsupervised variants. The following four design points detail the mechanism, the rationale for choosing the initial token, the location of application, and the determination of $\gamma$.

**1. Initial Token Scaling Operator: Leveraging the Entire Attention Distribution with a Single Factor**

Given the attention weights of one head in one layer, a scaling factor $\gamma > 0$ is introduced to act only on the initial token: $a_0' = \frac{\gamma a_0}{D}$, $a_i' = \frac{a_i}{D}$, where the normalization term $D = (\gamma-1)a_0 + 1$ ensures the distribution remains valid after scaling. The elegance of this operator lies in **modifying only the initial token while maintaining the relative proportions between all other tokens**. When $\gamma > 1$, the initial token is amplified, flattening (making more uniform) the remaining distribution; when $\gamma < 1$, the initial token is compressed, sharpening (making more focused) the remaining distribution. Thus, a single scalar can continuously adjust attention in both "flattening" and "sharpening" directions without per-token heuristics.

**2. Leverage Efficacy Analysis: Why Adjusting the Initial Token Yields Maximum Gain**

This addresses the fundamental question of why the initial token is chosen as the lever. The change in attention difference between any two tokens caused by scaling can be written as $E_{\text{diff},i,j} = |a_i - a_j| \cdot \frac{|\gamma-1| a_0}{(\gamma-1) a_0 + 1}$. The derivative with respect to the original weight of the initial token $a_0$ is $\frac{\partial E_{\text{diff},i,j}}{\partial a_0} = |a_i - a_j| |\gamma-1| \cdot \frac{1}{((\gamma-1)a_0+1)^2} \geq 0$, which is always non-negative. This implies that the larger the original attention of the initial token, the stronger its regulatory efficacy as a lever. Since attention sinks naturally cause the initial token to occupy extremely high weights, this provides a theoretical explanation for why tuning the initial token alone consistently yields the maximum gain.

**3. Selective Application across Layers and Heads: Acting Only Where Necessary**

Scaling is not applied uniformly across the entire model but is selectively applied to specific layers and heads, which constitutes the "head behavior profiling" step. In terms of layers, adjustments in shallow (1–10) and middle layers (11–21) are generally more effective than in deep layers (22–31), as the former handle representation learning and knowledge integration with larger regulatory space, while the latter focus on task-specific reasoning. Regarding heads, different attention heads respond heterogeneously: some heads improve performance when the initial token is amplified (up-effective), while others improve when it is reduced (down-effective). This divergence stems from the functional specialization of heads during pre-training. ZeroTuning performs head behavior profiling to evaluate the sensitivity of each head and applies $\gamma$ only to dominant head types (ablation shows tuning a moderate subset of approximately 40%–70% of heads is optimal).

**4. Two Calibration Modes: Determining $\gamma$ with or without Labels**

There are two paths to determine the scaling factor $\gamma$. The supervised mode directly searches for the $\gamma$ that maximizes accuracy on a labeled validation set. The unsupervised mode instead **minimizes output entropy**. The paper observes a strong correlation between the $\gamma$ that minimizes entropy and the $\gamma$ that maximizes accuracy, allowing near-optimal scaling intensity selection even without labels. Unlabeled input can be drawn from in-domain offline corpora or directly from test batches for test-time search. Once $\gamma$ is determined, the scaled scores are renormalized using softmax. In high-performance kernels where attention scores cannot be modified directly, ZeroTuning equivalently scales the query/key states to maintain compatibility with SDPA and FlashAttention.

## Main Results

### Classification Tasks

| Model | Vanilla | ACT | Auto-PASTA | **ZeroTuning** |
|------|---------|-----|------------|---------------|
| Llama-3.1-8B Avg | 59.59 | 60.11 | 63.73 | **71.44** |
| Qwen-2-7B Avg | 55.10 | - | 65.57 | **68.19** |
| Deepseek-R1-14B Avg | 67.67 | - | 69.04 | **71.87** |

Maximum single-dataset improvement: 73.20 → 91.60 (+18.4%) on SST-2, 44.60 → 66.60 (+22.0%) on SUBJ.

### Multiple-Choice QA Tasks

| Model | Vanilla | Auto-PASTA | **ZeroTuning** |
|------|---------|------------|---------------|
| Llama-3.1-8B Avg | 58.84 | 60.18 | **61.48** |
| Qwen-2-7B Avg | 63.10 | 64.01 | **64.84** |
| Deepseek-R1-14B Avg | 60.05 | 60.31 | **62.20** |

Deepseek-R1-14B on LogiQA: 27.80 → 35.60 (+7.80%).

### MT-Bench Dialogue

| Model | Vanilla | ZeroTuning |
|------|---------|-----------|
| Llama-3.1-8B | 7.804 | **7.966** |
| Llama-2-13B | 6.650 | **6.916** |

### Key Findings

1. **Adjusting only a single token consistently outperforms multi-token tuning methods.**
2. **Strong inverse correlation between accuracy and output entropy**: Validates the feasibility of the unsupervised mode.
3. **Head-specific tuning is significantly superior to uniform tuning.**
4. **Robust performance maintained across quantized reasoning, long context, and few-shot settings.**
5. **Modification requires only 4 lines of code.**

## Highlights & Insights

1. **Minimalist Design**: One token, one scaling factor, 4 lines of code.
2. **In-depth Theoretical Analysis**: Complete derivation from attention reshaping to bias correction.
3. **Unsupervised Mode**: Based on entropy minimization, requiring no labeled data.
4. **Kernel Agnostic**: Compatible with SDPA and FlashAttention.
5. **Consistency**: Effective across various models and tasks.

## Limitations & Future Work

1. The optimal scaling direction depends on the task (Classification vs. QA), requiring preliminary experiments or heuristic judgment.
2. Head behavior profiling still incurs some computational cost.
3. Improvement margins are relatively limited on already powerful models (e.g., Deepseek-R1-14B).
4. Mixed Up+Down strategies have not yet surpassed single-direction strategies; joint optimization remains to be explored.
5. Limited improvement observed in generative tasks (open-ended dialogue).

## Related Work & Insights
- **Attention Tuning**: PASTA, Auto-PASTA, ACT — rely on identifying important tokens.
- **Attention Sink Research**: StreamingLLM, Barbero et al. — explains the phenomenon but does not exploit it.
- **Inference-time Optimization**: Self-consistency, CoT — focus on prompt engineering.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Minimalist yet effective idea, transforming attention sinks from passive observations into active levers.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 15 datasets, 4 models, multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logical progression with a complete narrative from theory to method to experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Deployable with 4 lines of code, no training, and no extra memory overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](../../ICML2025/interpretability/deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICLR 2026\] On the Predictive Power of Representation Dispersion in Language Models](on_the_predictive_power_of_representation_dispersion_in_language_models.md)

</div>

<!-- RELATED:END -->
