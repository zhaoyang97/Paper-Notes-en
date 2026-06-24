---
title: >-
  [Paper Note] Advancing Sequential Numerical Prediction in Autoregressive Models
description: >-
  [ACL 2025][Numerical Prediction] Proposes Numerical Token Integrity Loss (NTIL)—a two-level numerical prediction loss function. At the token level, it replaces cross-entropy with exponentially position-weighted Earth Mover's Distance (EMD) to maintain numerical order. At the sequence level, it penalizes overall numerical deviation through differentiable numerical construction. This approach significantly improves the numerical prediction accuracy of autoregressive models acro…
tags:
  - "ACL 2025"
  - "Numerical Prediction"
  - "Earth Mover's Distance"
  - "Autoregressive Models"
  - "Loss Functions"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 42f0d65cffde8aa4
---

# Advancing Sequential Numerical Prediction in Autoregressive Models

**Conference**: ACL 2025  
**arXiv**: [2505.13077](https://arxiv.org/abs/2505.13077)  
**Code**: [GitHub](https://github.com/xfey/NTIL)  
**Area**: Other  
**Keywords**: Numerical Prediction, Earth Mover's Distance, Autoregressive Models, Loss Functions, Multimodal Large Language Models

## TL;DR

Proposes Numerical Token Integrity Loss (NTIL)—a two-level numerical prediction loss function. At the token level, it replaces cross-entropy with exponentially position-weighted Earth Mover's Distance (EMD) to maintain numerical order. At the sequence level, it penalizes overall numerical deviation through differentiable numerical construction. This approach significantly improves the numerical prediction accuracy of autoregressive models across tasks such as object detection, text detection, mathematical reasoning, and clock recognition.

## Background & Motivation

**Background**: Autoregressive models (LLM/MLLM) have become the mainstream choice for sequence generation tasks, widely applied in VQA, object detection, mathematical reasoning, and other tasks that require precise numerical outputs. Standard training methods use cross-entropy (CE) loss for token-by-token optimization.

**Limitations of Prior Work**:
- **Limitation 1 (Token-level)**: CE treats each digit token as an independent category, ignoring the ordinal relationship between numerical values (e.g., predicting "2" and predicting "9" yield the same CE loss relative to the ground truth "3", but "2" is clearly closer to the correct answer).
- **Limitation 2 (Sequence-level)**: CE calculates loss token-by-token, failing to capture the global numerical error composed of multiple tokens (e.g., predicting "1.01" compared to "1.98" for a target of "0.98"; the former is numerically closer but yields a higher CE loss).

**Key Challenge**: Autoregressive models generate numerical values token-by-token, but traditional CE loss completely overlooks the ordinal relationships of digit tokens and numerical integrity across tokens, which limits numerical prediction accuracy.

**Goal**: To design a training loss function that maintains numerical order at the token level and preserves numerical integrity at the sequence level, thereby enhancing the precision of autoregressive models in all tasks involving numerical outputs.

**Key Insight**: Introducing Earth Mover's Distance (EMD) for the first time into autoregressive model training, combined with differentiable numerical construction to achieve sequence-level numerical optimization across tokens.

**Core Idea**: Using EMD instead of CE to address the neglect of token-level ordinal relationships, and leveraging differentiable numerical reconstruction to resolve the sequence-level global numerical deviation, followed by joint optimization across both levels.

## Method

### Overall Architecture

NTIL = Exponentially Position-Weighted EMD (token-level) + Relative Deviation Measure + Magnitude Deviation Measure (sequence-level). The final loss is a weighted sum of these three components:

$$\mathcal{L} = \mathbf{W_{exp}} \cdot \text{EMD} + \alpha \cdot \mathcal{L}_{relative} + \beta \cdot \mathcal{L}_{magnitude}$$

### Key Designs

1. **Exponential Position-Based Weighting**:
    - Function: Replaces CE at the token level to model the ordinal relationships of digit tokens.
    - Mechanism: EMD measures the minimum cost of "moving" the predicted distribution to the target distribution, naturally encoding distance relationships between numbers. It further introduces an exponential position-based weighting $W_{exp} = [(1+\sigma)^{n-i-1}]$ to impose higher penalties on errors in high-order digits.
    - Design Motivation: In decimal positional notation, high-order digits (e.g., hundreds) have a much greater impact on the final numerical value than low-order digits (e.g., units). Exponential weighting naturally reflects this characteristic.

2. **Differentiable Numerical Value Construction**:
    - Function: Reconstructs continuous numerical values from discrete token prediction distributions, enabling backpropagation for sequence-level losses.
    - Mechanism: Approximates argmax using Gumbel-Softmax (ensuring consistency with a low temperature and low noise), then computes the weighted sum of predicted digit indices by their positional values to reconstruct the final numerical value.
    - Design Motivation: Directly backpropagating gradients through discrete argmax is infeasible, and Gumbel-Softmax provides a differentiable relaxation.

3. **Dual Sequence-Level Deviation Measures**:
    - Function: Penalizes overall numerical error from both relative and magnitude deviation perspectives.
    - Mechanism: Relative deviation $L_{relative} = |X-Y|/max(X,Y)+\epsilon$ provides a normalized ratio of error. Magnitude deviation $L_{magnitude} = \log(max(X,Y)/min(X,Y))$ penalizes orders-of-magnitude differences.
    - Design Motivation: Relative deviation handles ratio differences within the same order of magnitude (e.g., relative deviations for 1 vs 10 and 1 vs 100 are similar), while magnitude deviation complementarily distinguishes differences across different orders of magnitude (e.g., log(10)=2.3 vs log(100)=4.6).

### Loss & Training

- Total loss: $L = W_{exp}·EMD + \alpha·L_{relative} + \beta·L_{magnitude}$
- NTIL is applied only to numerical tokens, while non-numerical tokens still utilize standard CE.
- Hyperparameters $\alpha$ and $\beta$ are adjustable weights.
- Can be seamlessly integrated into the training pipelines of LLMs (Baichuan2, Qwen2.5, LLaMA3, Yi, MiniCPM3) and MLLMs (PaliGemma, LLaVA-1.5, Yi-VL, Qwen2-VL).

## Key Experimental Results

### Main Results - Image Localization (Acc@0.5)

| Model | CE | EMD | NTIL (Ours) |
|---|---|---|---|
| PaliGemma (3b) | 0.785 | 0.789 | **0.795** |
| LLaVA-1.5 (7b) | 0.818 | 0.820 | **0.822** |
| Yi-VL (6b) | 0.733 | 0.740 | **0.744** |
| Qwen2-VL (2b) | 0.863 | 0.859 | **0.866** |
| Qwen2-VL (7b) | 0.860 | 0.855 | **0.862** |

### Text Detection (Acc@0.5 avg)

| Model | CE | EMD | NTIL (Ours) |
|---|---|---|---|
| PaliGemma (3b) | 0.193 | 0.241 | **0.263** |
| Qwen2-VL (2b) | 0.720 | 0.718 | **0.732** |
| Qwen2-VL (7b) | 0.764 | 0.751 | **0.770** |
| LLaVA-1.5 (7b) | 0.675 | 0.690 | **0.698** |

### Clock Recognition

| Model | CE Acc(%) | Ours Acc(%) | CE Time Deviation (min) | Ours Time Deviation (min) |
|---|---|---|---|---|
| LLaVA-1.5 (7b) | 95.1 | **98.3** | 8.52 | **4.14** |
| Yi-VL (6b) | 76.2 | **87.4** | 56.58 | **26.58** |
| Qwen2-VL (2b) | 81.3 | **85.3** | 32.34 | **24.66** |

### Ablation Study

| Exp | Rel | Mag | PaliGemma-MathVista | LLaVA-MathVista | Yi-VL-Clock |
|---|---|---|---|---|---|
| ✗ | ✓ | ✓ | 0.137 | 0.166 | 0.834 |
| ✓ | ✗ | ✓ | 0.137 | 0.154 | 0.856 |
| ✓ | ✓ | ✗ | 0.142 | 0.143 | 0.876 |
| ✓ | ✓ | ✓ | **0.157** | **0.170** | **0.874** |

### Key Findings

- NTIL consistently outperforms CE and pure EMD across 4 categories of tasks on 5 MLLMs and 5 LLMs.
- The three loss components are complementary: removing any of them degrades performance on certain tasks.
- The improvement is more pronounced on smaller models (e.g., text detection on PaliGemma: 0.193 → 0.263, a 36% improvement).
- The performance gain in mathematical reasoning tasks is relatively moderate, likely because numerical prediction is only one step in the reasoning chain.
- In the clock recognition task, not only is accuracy improved, but time deviation is also significantly reduced.

## Highlights & Insights

- **Novelty**: Recommends EMD as an optimization objective for autoregressive models for the first time, proposing global numerical optimization across multiple time steps.
- **Clever Utilization of Positional Notation**: Exponential position-based weighting naturally encodes the importance differences in the decimal positional system.
- **High Generality**: The method is model-architecture-independent and can be plugged in-place into any LLM/MLLM.
- **Gumbel-Softmax Bridging Discrete and Continuous**: Elegantly solves the differentiability challenge from discrete token predictions to continuous numerical reconstructions.
- **Complementary Dual Metrics**: Relative deviation handles scale-invariant errors, while magnitude deviation handles order-of-magnitude errors.

## Limitations & Future Work

- It only handles decimal integers and simple floating-point numbers; applicability to more complex numerical formats (scientific notation, fractions, etc.) has not been verified.
- The sequence-level loss relies on the temperature parameter of Gumbel-Softmax, which may introduce additional tuning costs.
- Improvements on larger models and mathematical reasoning tasks are relatively small, possibly because larger models already possess stronger baseline numerical prediction capabilities.
- The interaction with other decoding strategies (such as beam search, sampling) has not been explored.
- The handling of negative numbers and special values (0, NaN, Inf) is not discussed.

## Related Work & Insights

- **vs Wasserstein GAN**: Both use EMD, but WGAN uses it for generator-discriminator training stability, whereas this work applies it for token-level optimization in supervised learning.
- **vs Standard CE**: CE is a classification loss that completely ignores distances between classes; NTIL models numerical distances through the dual application of EMD and sequence-level metrics.
- **vs Pure EMD**: Experiments show that pure EMD can even underperform CE on some tasks (e.g., Qwen2-VL 7b localization); NTIL resolves this issue through sequence-level constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing EMD into autoregressive numerical prediction for the first time + sequence-level differentiable numerical construction, presenting a novel entry point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 MLLMs, 5 LLMs, and 4 tasks with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Explains problem motivation clearly (the two limitations of CE are highly intuitive), with a well-structured description of the methodology.
- Value: ⭐⭐⭐⭐ A plug-and-play, general-purpose improvement scheme that holds value for all autoregressive tasks involving numerical outputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [\[ACL 2025\] On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)

</div>

<!-- RELATED:END -->
