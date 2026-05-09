---
title: >-
  [Paper Note] Steering When Necessary: Flexible Steering Large Language Models with Backtracking
description: >-
  [NeurIPS 2025][Video Understanding][activation steering] This paper proposes FASB (Flexible Activation Steering with Backtracking), a framework that dynamically determines the necessity and intensity of intervention by tracking the internal states of an LLM during generation, and introduces a backtracking mechanism to correct already-deviated tokens. FASB achieves a True\*Info score of 80.56% on TruthfulQA and an average accuracy of 78.8% across six multiple-choice tasks, significantly outperforming all baselines.
tags:
  - NeurIPS 2025
  - Video Understanding
  - activation steering
  - representation engineering
  - backtracking mechanism
  - truthfulness
  - dynamic intervention
date: 2026-05-08
content_hash: 585c39d09fc4b5ab
---

# Steering When Necessary: Flexible Steering Large Language Models with Backtracking

**Conference**: NeurIPS 2025
**arXiv**: [2508.17621](https://arxiv.org/abs/2508.17621)
**Code**: [https://github.com/gjw185/FASB](https://github.com/gjw185/FASB)
**Area**: Video Understanding / LLM Alignment
**Keywords**: activation steering, representation engineering, backtracking mechanism, truthfulness, dynamic intervention

## TL;DR

This paper proposes FASB (Flexible Activation Steering with Backtracking), a framework that dynamically determines the necessity and intensity of intervention by tracking the internal states of an LLM during generation, and introduces a backtracking mechanism to correct already-deviated tokens. FASB achieves a True\*Info score of 80.56% on TruthfulQA and an average accuracy of 78.8% across six multiple-choice tasks, significantly outperforming all baselines.

## Background & Motivation

Despite their remarkable success in text generation, LLMs remain prone to producing undesired outputs such as harmful content and hallucinations. Controlling LLM behavior without costly fine-tuning is a central challenge.

**Activation steering** has emerged as a lightweight approach that guides output by directly modifying internal activations at inference time, avoiding large-scale data collection and fine-tuning costs.

Two major limitations of existing activation steering methods:

**Indiscriminate intervention**: Methods such as CAA and ORTHO apply a fixed-strength steering vector to all generations, failing to distinguish between outputs that are already correct and those that require correction. Forcing intervention on correct generations can degrade output quality.

**Question-only judgment**: Methods such as CAST determine whether to intervene based solely on the representation of the input question. However, in complex settings such as truthfulness, it is difficult to predict from the question alone whether the response will deviate. Experiments confirm that on TruthfulQA, a question-only classifier produces prediction probabilities concentrated in the 0.3–0.5 range, making fine-grained decisions unreliable.

**Root Cause**: (1) Different generations require different degrees of intervention; (2) intervening only after deviation is detected is often too late—already-generated deviated tokens cannot be corrected.

**Core Idea**: Track LLM internal states in real time after each token generation, jointly considering the question and the generated content to dynamically determine intervention necessity and intensity. A backtracking mechanism is introduced: upon detecting deviation, the model rolls back several tokens and regenerates, placing intervention before the point of deviation.

## Method

### Overall Architecture

FASB operates in two steps. Step 1 uses a probe method to identify attention heads most relevant to the target behavior, yielding steering vectors and classifiers. Step 2 applies classifiers to track states in real time during generation; once deviation is detected, backtracking and adaptive activation steering are triggered.

### Key Designs

1. **Attention Head Localization and Steering Vector Extraction (Probe Method)**:

    - **Function**: Identify the attention heads in the LLM most relevant to the target behavior (e.g., truthfulness) and construct steering vectors.
    - **Mechanism**: Concatenate question and answer, extract activations at each attention head's last token position, and train a logistic regression probe $p_\theta(x) = \text{sigmoid}(\langle \theta^{\ell,h}, x^{\ell,h} \rangle)$. The top-$k$ heads with the highest validation accuracy are selected, and the probe parameters $\theta^{\ell,h}$ are directly used as steering vectors.
    - **Design Motivation**: Attention heads whose probes effectively distinguish positive from negative samples are the most appropriate targets for intervention. The probe parameters themselves constitute the optimal directional vector for separating positive and negative activations. Restricting intervention to a small number of critical heads minimizes disruption to the model's overall behavior.

2. **Real-Time State Tracking**:

    - **Function**: Evaluate after each token generation whether the current generation deviates from the desired behavior.
    - **Mechanism**: After generating the $j$-th token, each of the top-$k$ head classifiers predicts on the activation of the $j$-th token, and the average deviation probability is computed as $p(x_{i,j}) = \frac{1}{k} \sum_{(\ell,h) \in \text{top-k}} (1 - p_{\theta^{\ell,h}}(x_{i,j}^{\ell,h}))$.
    - **Design Motivation**: Reusing hidden states already available during generation incurs zero additional computation. Jointly considering the question and generated content yields more accurate judgments than question-only approaches.

3. **Backtracking Mechanism**:

    - **Function**: Upon detecting deviation, roll back $s$ tokens and regenerate under activation steering.
    - **Mechanism**: If the deviation probability after the $j$-th token exceeds threshold $\beta$, the first $j-s$ tokens are retained and subsequent tokens are regenerated with steering vectors applied. Intervention intensity is adaptive: $r = \mathbb{I}(p > \beta) \cdot p \cdot \alpha$.
    - **Design Motivation**: The key insight is that intervening only after deviation is detected can only influence subsequent content—already-generated deviated tokens are irrecoverable. The backtracking mechanism corrects deviated content at minimal cost (only $s$ additional tokens regenerated).

### Implementation of Activation Steering

Intervention is applied exclusively to the selected top-$k$ attention heads. The multi-head attention output becomes:
$$h_{i,j-s+1}^\ell = \text{concat}(x_{i,j-s+1}^{\ell,1} + r\theta^{\ell,1}c^{\ell,1}, \cdots, x_{i,j-s+1}^{\ell,H} + r\theta^{\ell,H}c^{\ell,H}) W^{\ell,O}$$

where $c^{\ell,H}$ is a binary scalar equal to 1 only for selected heads, ensuring precision of intervention.

## Key Experimental Results

### Main Results

TruthfulQA open-ended generation task (LLaMA2-7B-CHAT):

| Method | True (%) | Info (%) | True\*Info (%) | MC1 (%) | MC2 (%) |
|--------|---------|---------|----------------|---------|---------|
| Baseline | 66.83 | 99.51 | 66.50 | 33.41 | 51.07 |
| ITI | 94.49 | 80.55 | 76.11 | 38.31 | 57.15 |
| CAA | 71.60 | 83.84 | 60.03 | 34.03 | 52.76 |
| SADI-HEAD | 77.72 | 98.53 | 76.58 | 35.90 | 54.65 |
| **Probe (FASB)** | 93.88 | 85.81 | **80.56** | **48.71** | **66.58** |

Average accuracy across six multiple-choice tasks:

| Method | COPA | StoryCloze | NLI | MMLU | SST2 | Winogrande | Avg. |
|--------|------|-----------|-----|------|------|-----------|------|
| Baseline | 64.4 | 60.2 | 63.5 | 60.2 | 92.2 | 50.2 | 65.1 |
| ITI | 66.6 | 59.7 | 64.3 | 60.2 | 92.3 | 51.5 | 65.8 |
| **Probe (FASB)** | **90.0** | **93.5** | **80.0** | 62.4 | **92.8** | **54.1** | **78.8** |

### Ablation Study

| Configuration | True\*Info | MC1 | MC2 | Note |
|--------------|-----------|-----|-----|------|
| Probe (full) | **80.56** | **48.71** | **66.58** | All components |
| All fixed | 76.11 | 38.31 | 57.15 | Same intervention strength for all samples |
| w/o Adaptive | 82.08 | 42.96 | 62.06 | Binary intervene/not, no strength adaptation |
| w/o Backtracking | 62.11 | 35.01 | 53.55 | Intervene only after detection; too late |
| After Question | 72.55 | 41.86 | 59.88 | Use question hidden state only |

### Key Findings

- **Backtracking is critical**: Without backtracking, True\*Info drops to 62.11%—below the baseline—confirming that intervening after deviation detection is indeed too late.
- **Question-only judgment is insufficient**: After Question achieves only 72.55%, far below the 80.56% of real-time tracking, validating the need for online state monitoring.
- **Adaptive intensity is effective**: Fixed strength 76.11% → non-adaptive 82.08% → fully adaptive 80.56% (optimal on True\*Info).
- **Cross-model generalization**: Consistent effectiveness across six LLMs (LLaMA2, LLaMA3.1, Qwen2.5), with MC1 improving by 24.61 on Qwen2.5-7B.
- **Cross-domain generalization**: Classifiers trained on TruthfulQA transfer effectively to Natural Questions (MC2: 49.54→59.25) and TriviaQA (61.22→67.55).

## Highlights & Insights

- The backtracking mechanism is the paper's most central contribution—transforming "passive post-detection intervention" into "proactive rollback and correction," resolving an inherent limitation of activation steering at minimal additional cost (only $s$ tokens regenerated).
- State tracking reuses hidden states already available during generation, incurring zero additional computational overhead.
- The probe method elegantly unifies probe parameters as both steering vectors and classifier weights, yielding a concise and coherent design.

## Limitations & Future Work

- The backtracking step size $s$ and threshold $\beta$ require hyperparameter search, increasing tuning overhead.
- The approach is currently validated only in truthfulness settings; effectiveness in safety, fairness, and other domains remains to be explored.
- The backtracking mechanism may be triggered multiple times in long-sequence generation, and cumulative overhead warrants evaluation.
- Caching activations of top-$k$ heads at inference time introduces a modest memory overhead.

## Related Work & Insights

- ITI is a pioneering work but applies indiscriminate intervention; FASB's approach of directly reusing probe parameters as steering vectors is more elegant.
- The backtracking idea shares conceptual similarities with beam search backtracking, but operates at the level of internal representations rather than tokens.
- The empirical validation that LLM internal states can be tracked in real time by lightweight probes offers insights for interpretability research.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/video_understanding/aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)

<!-- RELATED:END -->
