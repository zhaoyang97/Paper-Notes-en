---
title: >-
  [Paper Note] Efficient Reasoning with Balanced Thinking
description: >-
  [ICLR 2026][Model Compression][Large language model reasoning] This paper proposes ReBalance, a training-free framework that simultaneously mitigates overthinking and underthinking in large reasoning models (LRMs) via confidence-guided dynamic hidden-state steering vectors, achieving joint improvements in both reasoning efficiency and accuracy.
tags:
  - ICLR 2026
  - Model Compression
  - Large language model reasoning
  - overthinking
  - underthinking
  - hidden-state steering
  - training-free acceleration
date: 2026-05-08
content_hash: 89404c885fa4df26
---

# Efficient Reasoning with Balanced Thinking

**Conference**: ICLR 2026
**arXiv**: [2603.12372](https://arxiv.org/abs/2603.12372)
**Code**: [GitHub](https://github.com/yu-lin-li/ReBalance)
**Area**: Model Compression / Efficient Inference
**Keywords**: Large language model reasoning, overthinking, underthinking, hidden-state steering, training-free acceleration

## TL;DR

This paper proposes ReBalance, a training-free framework that simultaneously mitigates overthinking and underthinking in large reasoning models (LRMs) via confidence-guided dynamic hidden-state steering vectors, achieving joint improvements in both reasoning efficiency and accuracy.

## Background & Motivation

- **Background**: Large reasoning models (e.g., DeepSeek-R1, QwQ) have acquired powerful reasoning capabilities through SFT and RL training, yet face significant computational efficiency challenges in practical deployment.
- **Limitations of Prior Work**: LRMs exhibit two opposing failure modes — **overthinking**: expending redundant reasoning steps on simple problems; and **underthinking**: prematurely converging on complex problems without sufficiently exploring the reasoning path.
- **Key Challenge**: Existing methods for mitigating overthinking (e.g., suppressing reflective keywords, adjusting reasoning length) tend to induce underthinking, forming an inherent trade-off between the two. As shown in Fig. 2(a), prior methods reduce the reasoning length of both correctly and incorrectly solved samples simultaneously, indicating the introduction of underthinking.
- **Goal**: To alleviate overthinking without inducing underthinking, thereby achieving balanced reasoning.
- **Key Insight**: The paper observes that stepwise confidence and confidence variance can serve as continuous indicators of reasoning state — high variance reflects hesitation and path-switching (overthinking), while persistently high confidence reflects premature commitment (underthinking).
- **Core Idea**: Confidence signals are used to identify reasoning states; a hidden-state steering vector is constructed from overthinking to underthinking representations; a dynamic control function then modulates the steering direction and magnitude based on real-time confidence.

## Method

### Overall Architecture

ReBalance operates in two phases — offline and online:
1. **Offline Phase**: A single forward pass is performed on a small-scale dataset to identify overthinking/underthinking steps, extract hidden-state prototypes, compute the steering vector, and fit the dynamic control function.
2. **Online Phase**: During inference, the dynamic control function computes a steering weight based on real-time confidence, which is used to inject the steering vector into the hidden states.

### Key Designs

#### 1. Explicit Modeling of Overthinking and Underthinking

- **Function**: Classifies reasoning steps into an overthinking set $\mathcal{O}$ and an underthinking set $\mathcal{U}$ based on confidence-derived metrics.
- **Mechanism**: The step-level confidence is defined as $c_s = \exp\left(\frac{1}{|\mathcal{T}_s|}\sum_{t \in \mathcal{T}_s} \ln p_t^{\max}\right)$, and the confidence variance within a sliding window is $\operatorname{Var}(c_s; \mathcal{W}_s)$. Empirical quantile thresholds $\tau_c^L, \tau_c^H, \tau_v^L, \tau_v^H$ are used to categorize steps as:
    - Overthinking $\mathcal{O} = \{s: c_s \leq \tau_c^L \wedge v_s \geq \tau_v^H\}$ (low confidence, high variance)
    - Underthinking $\mathcal{U} = \{s: c_s \geq \tau_c^H \wedge v_s \leq \tau_v^L\}$ (high confidence, low variance)
- **Design Motivation**: The correspondence between confidence patterns and reasoning states is empirically validated in Fig. 2(b).

#### 2. Confidence-Based Steering Vector Extraction

- **Function**: Extracts prototype representations of overthinking and underthinking from deep hidden states to construct the steering vector.
- **Mechanism**: Prototypes $\bm{\mu}^O$ and $\bm{\mu}^U$ are obtained by averaging the first-token hidden states of steps in $\mathcal{O}$ and $\mathcal{U}$, respectively. The steering vector is defined as $\mathbf{v} = \frac{\bm{\mu}^O - \bm{\mu}^U}{\|\bm{\mu}^O - \bm{\mu}^U\|_2}$. Hidden-state adjustment follows $\tilde{\mathbf{h}}_{t_s^{(1)}} = \mathbf{h}_{t_s^{(1)}} + \alpha_s \mathbf{v}$, where $\alpha_s = \lambda_s \delta_s$, with $\delta_s = +1$ to mitigate underthinking and $\delta_s = -1$ to mitigate overthinking.
- **Design Motivation**: Deep hidden states exhibit stronger discriminative power for reasoning patterns (see appendix), and the first token conditions subsequent generation via causal attention.

#### 3. Model-Behavior-Driven Dynamic Control Function

- **Function**: Adaptively adjusts the steering direction and magnitude based on real-time confidence.
- **Mechanism**: $g(c_s, v_s) = \text{sign}(c_s - \tau_c^H) \cdot B(c_s, v_s) \cdot \tanh(|c_s - \tau_c^H|)$
    - Direction $\delta_s$: negative when $c_s < \tau_c^H$ (mitigating overthinking); positive when $c_s > \tau_c^H$ (mitigating underthinking).
    - Magnitude $\lambda_s$: smoothly saturated via $\tanh$; $B(c_s, v_s)$ is a variance-aware amplitude function that adaptively switches among $B_m$, $B_o$, and $B_u$ according to the current reasoning state.
- **Design Motivation**: Avoids hard switching and ensures numerical stability; parameters $B_m$, $B_o$, etc. are derived adaptively from model behavior, requiring no manual tuning.

### Loss & Training

ReBalance is a training-free method and involves no loss function design. All components are obtained from a single offline forward pass.

## Key Experimental Results

### Main Results

| Model / Dataset | MATH-500 Acc↑ | MATH-500 Tokens↓ | AIME24 Acc↑ | GSM8K Acc↑ |
|---|---|---|---|---|
| R1-Distill-1.5B Baseline | 79.6 | 4516 | 23.3 | 76.0 |
| R1-Distill-1.5B ReBalance | **83.0** | 3474 (−23%) | **33.3** | **78.3** |
| R1-Distill-7B Baseline | 89.8 | 3699 | 46.7 | 89.2 |
| R1-Distill-7B ReBalance | **92.6** | 2903 (−22%) | **53.3** | **91.6** |
| Qwen3-14B Baseline | 93.8 | 4470 | 66.7 | 95.1 |
| Qwen3-14B ReBalance | **94.0** | 3641 (−19%) | **73.3** | **96.3** |
| QwQ-32B Baseline | 94.8 | 4535 | 66.7 | 96.3 |
| QwQ-32B ReBalance | **95.4** | 3551 (−22%) | **73.3** | **96.7** |

### Ablation Study

- Using the steering vector without the dynamic control function yields limited performance gains and even degradation on some datasets.
- Removing the variance-aware amplitude $B(c_s, v_s)$ prevents the model from distinguishing normal from abnormal reasoning states.
- Shallow vs. deep hidden states: deep layers (e.g., the second-to-last and third-to-last) perform best; shallow layers lack sufficient discriminative power.

### Key Findings

1. ReBalance consistently outperforms all baselines across 4 models (0.5B–32B) and 9 benchmarks.
2. It **simultaneously** reduces reasoning length (15–30%) **and** improves accuracy (typically +2–10%), a combination rarely achieved by prior methods.
3. Steering vectors extracted from small-scale seen datasets generalize robustly to unseen datasets.
4. Unlike token-level suppression methods such as SEAL and DEER, ReBalance does not sacrifice valuable intermediate reasoning steps.

## Highlights & Insights

1. **Core Insight**: High confidence variance corresponds to overthinking (hesitation); persistently high confidence corresponds to underthinking (premature commitment). This observation is both intuitive and empirically validated.
2. **Methodological Value**: Overthinking and underthinking are unified and addressed within a single framework rather than handled separately.
3. **Strong Practicality**: Training-free and plug-and-play, requiring only a one-time offline computation on a small dataset, making deployment cost extremely low.
4. **Counter-Intuitive Finding**: Outputs with shorter reasoning chains achieve higher accuracy, demonstrating that redundant reasoning genuinely introduces hallucinations.

## Limitations & Future Work

1. The steering vector is extracted from a fixed dataset and may not generalize to all task distributions; online update mechanisms warrant further exploration.
2. Confidence computation relies on token probabilities; robustness to different sampling strategies (e.g., top-k, nucleus sampling) has not been fully validated.
3. While the quantile thresholds $q_L, q_H$ are adaptive, optimal values may vary across models and tasks.
4. Validation is currently limited to mathematical and code reasoning; effectiveness on natural language reasoning and multi-step planning remains to be verified.

## Related Work & Insights

- **Overthinking Mitigation**: SEAL (Chen et al., 2025b) reduces reasoning length by suppressing reflective keywords but may induce underthinking; NoThinking (Ma et al., 2025b) takes a more aggressive approach by skipping the thinking phase entirely.
- **Reasoning Efficiency**: Unlike RL-based reasoning length control methods, ReBalance is entirely training-free.
- **Hidden-State Manipulation**: The approach draws on ideas from representation engineering, but targets reasoning patterns rather than behavioral control.
- **Insight**: Using confidence as a probe for reasoning quality can be generalized to extract more fine-grained reasoning dynamics from LRMs.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ Unifying overthinking and underthinking under a single model and addressing both via hidden-state steering is a novel and elegant approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 4 models × 9 benchmarks; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐⭐ Training-free and plug-and-play; highly deployment-friendly.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning](swireasoning_switch-thinking_in_latent_and_explicit_for_pareto-superior_reasonin.md)
- [\[ICLR 2026\] A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)

<!-- RELATED:END -->
