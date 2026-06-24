---
title: >-
  [Paper Note] Internal Value Alignment in Large Language Models through Controlled Value Vector Activation
description: >-
  [ACL 2025][LLM Alignment][Value Alignment] This work proposes the ConVA (Controlled Value Vector Activation) framework, which accurately identifies value vectors in the LLM's latent space using context-controlled datasets, and activates target values at inference time using a gated minimal perturbation mechanism. It achieves an average 29.6% increase in control success rate across the 10 Schwartz basic values while maintaining 97%+ of text fluency and general capabilities.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Value Alignment"
  - "Activation Engineering"
  - "Value Vector"
  - "Schwartz Theory of Basic Human Values"
  - "Interpretability"
date: 2026-05-08
content_hash: 37e258252f9f65b7
---

# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation

**Conference**: ACL 2025  
**arXiv**: [2507.11316](https://arxiv.org/abs/2507.11316)  
**Code**: [https://github.com/hr-jin/ConVA](https://github.com/hr-jin/ConVA)  
**Area**: LLM Alignment/Safety  
**Keywords**: Value Alignment, Activation Engineering, Value Vector, Schwartz Theory of Basic Human Values, Interpretability

## TL;DR
This work proposes the ConVA (Controlled Value Vector Activation) framework, which accurately identifies value vectors in the LLM's latent space using context-controlled datasets, and activates target values at inference time using a gated minimal perturbation mechanism. It achieves an average 29.6% increase in control success rate across the 10 Schwartz basic values while maintaining 97%+ of text fluency and general capabilities.

ConVA locates and modifies the value encoding direction directly in the latent space of LLMs through context-controlled value vector identification and gated activation, achieving internal value alignment without fine-tuning.

## Background & Motivation

**Background**: Value alignment of LLMs has received increasing attention due to its ability to provide clarity and transparency in model behavior, and support adaptation to evolving social norms.

**Limitations of Prior Work**: Existing behavior-level alignment methods (SFT, RLHF, ICA) treat the LLM as a black box, lacking an interpretable understanding of internal value encoding within the model, and fail to guarantee that the model consistently follows alignment goals.

**Key Challenge**: Although activation engineering is more interpretable, it faces two key challenges: (a) a lack of high-quality data to explain the internal values of the model, as directly generated data exhibits severe context bias; (b) modifying activation values to ensure consistent values leads to a significant degradation in model performance.

**Goal**: How to accurately locate the value encoding direction in the latent space of LLMs and achieve consistent value control under minimal perturbation, while preserving the model's general capabilities and output fluency.

**Key Insight**: Based on the linear representation hypothesis, bias is eliminated by constructing context-controlled datasets, value vectors are HTML/text-identified using a linear classifier, and adaptive control is implemented using a gated mechanism.

**Core Idea**: Classifiers are trained using contextually consistent positive and negative sample pairs to locate the value direction, and a gated mechanism is used at inference time to apply minimal activation shifts, ensuring consistency in values.

## Method

### Overall Architecture

ConVA consists of two phases: (1) **Context-Controlled Value Vector Identification**—constructing positive and negative sample pairs to train linear classifiers and extracting the normal vectors of the classification planes as value vectors; (2) **Gated Value Vector Activation**—determining whether the input is related to the target value via a gate during inference, and if so, shifting the embeddings along the direction of the value vector with minimal perturbation.

### Key Designs

**Module 1: Context-Controlled Data Generation**

- **Function**: Construct high-quality positive and negative sample pairs for identifying value vectors.
- **Mechanism**: First, diverse positive samples (with different personas, sentence structures, and scenarios) are generated using GPT-4o, and then a corresponding negative sample is generated for each positive sample, ensuring they only contrast on the target value while keeping other contexts as consistent as possible.
- **Design Motivation**: Directly generated positive and negative samples suffer from severe context bias (e.g., "security" might be misinterpreted as "digital security"), which prevents the classifier from finding the accurate value encoding direction. Aligning context pair-by-pair eliminates interference from noisy words.

**Module 2: Linear Classifier Value Vector Extraction**

- **Function**: Train a linear classifier in the embedding space of specified layers of the LLM and extract the normal vector of the classification plane as the value vector.
- **Mechanism**: Using the Concept Activation Vector (CAV) approach, a binary classifier $P_V(\mathbf{e}) = \text{sigmoid}(\mathbf{w}^T\mathbf{e} + b)$ is trained, where the normalized weight vector $\mathbf{v} = \mathbf{w}/\|\mathbf{w}\|$ serves as the value vector.
- **Design Motivation**: The linear representation hypothesis posits that human-readable concepts are encoded linearly in the activation space of models. Consequently, the normal vector of the classification plane logically corresponds to the encoding direction of that concept.

**Module 3: Gated Value Vector Activation**

- **Function**: Shift embeddings layer-by-layer across multiple layers during inference to achieve value control, while employing a gated mechanism to preserve general capabilities.
- **Mechanism**: Alignment is formulated as a constrained optimization problem—minimizing the shift magnitude $|\epsilon|$ subject to the constraint that the modified embedding is classified as aligned with the target value ($P_V(\hat{\mathbf{e}}) \geq P_0$). An indicator function $\mathbb{I}(g(x) > g_0)$ serves as the gate to apply control exclusively to value-relevant inputs.
- **Design Motivation**: Fixed-magnitude perturbations struggle to balance control success rate and fluency; minimizing perturbation ensures text fluency, while the gating mechanism prevents unnecessary intervention on irrelevant inputs, safeguarding general capabilities like MMLU.

### Loss & Training

- Classifier training utilizes standard binary cross-entropy loss.
- The closed-form solution for value vector activation is $\epsilon = I \cdot \frac{\text{sigmoid}^{-1}(P_0) - \mathbf{w}^T\mathbf{e} - b}{\mathbf{w}^T\mathbf{v}}$.
- Control is applied only to layers with test accuracy > 0.9, excluding the final 5 layers.
- The gating unit uses a Deberta-based human value detector.

## Key Experimental Results

### Main Results

User study results based on the 10 Schwartz basic values and Llama-2-7b-chat:

| Method | CSR (Annotator 1) | FR (Annotator 1) | CSR (Annotator 2) | FR (Annotator 2) | CSR (Annotator 3) | FR (Annotator 3) |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| ICA | 0.30 | 1.00 | 0.40 | 1.00 | 0.43 | 0.99 |
| CAA | 0.49 | 0.86 | 0.52 | 0.86 | 0.47 | 0.85 |
| **ConVA** | **0.79** | **1.00** | **0.87** | **0.99** | **0.83** | **1.00** |

### Ablation Study

Protective effect of the gating mechanism on general capabilities (MMLU benchmark):

| Method | Average MMLU Score |
|------|--------------|
| Vanilla LLM | 0.476 |
| ConVA w.o. gate | 0.272 |
| ConVA (with gate) | 0.455 |

### Key Findings

- ConVA achieves an average CSR relative improvement of 29.6% (t-test p=6.29e-07) in automated evaluations across the 10 value dimensions, with a fluency rate consistently at or above 97%.
- Without context-controlled data (ConVA w.o. CCD), performance is across-the-board lower than full ConVA, proving the effectiveness of controlled data construction.
- Even when faced with malicious prompts guided by opposing values, ConVA successfully maintains value control.
- ConVA generalizes to multiple backbone models including Qwen2.5-{3, 7, 14, 32, 72}B, Llama-3-8B, Vicuna-13B, and Mistral-7B.
- Cosine similarity analysis of value vectors reveals that the inner value structure of LLMs aligns with the high-level groupings of Schwartz theory.

## Highlights & Insights

1. The **context-controlled data construction** is highly ingenious—using pair-wise generation ensures positive and negative samples differ only in the value dimension, drastically improving value vector identification accuracy.
2. The combined design of **gating + optimization-driven perturbation** balances control efficacy and general capability, which is more elegant than fixed-magnitude perturbation methods.
3. Only 100 pairs of training samples are needed to complete value vector identification, requiring minimal data.
4. Analysis of the internal value structures of LLMs uncovers alignments and conflicts with human value systems, offering significant interpretability value.

## Limitations & Future Work

1. Control effectiveness is uneven across different value dimensions (e.g., lower success rate in the "power" dimension), which is likely limited by the value knowledge present in the LLMs' training data.
2. Currently, only single-value alignment is supported, whereas real-world scenarios typically involve complex combinations of multi-dimensional values with varying weights.
3. Based on the linear representation hypothesis, but some studies indicate that certain features require at least a two-dimensional subspace to be accurately represented.
4. The performance of the gating unit limits the upper bound of the framework; in the future, better human value detectors can further improve performance.

## Related Work & Insights

- **Activation Engineering**: CAA (Rimsky et al., 2024) uses a fixed-magnitude mean difference as the control vector, whereas ConVA learns via a classifier and optimizes the perturbation magnitude for each token.
- **Behavior-level Alignment**: BaseAlign (Yao et al., 2024a) is based on RLHF, requiring substantial data and computational resources, whereas ConVA achieves lightweight alignment with only 100 pairs of samples.
- **Value Exploration**: UniVaR (Cahyawijaya et al., 2025) provides a cross-lingual value exploration and analysis tool, while ConVA additionally offers direct value control capabilities.
- Preliminary attempts at multi-concept control in controllable text generation (Zhang et al., 2025; Chakraborty et al., 2024) can serve as potential directions for multi-dimensional value alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of context-controlled data construction with gating optimization-driven perturbation brings outstanding novelty to the value alignment domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 10 value dimensions, 9 backbone models, including user studies and ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with well-supported motivations, and the methodology is presented mathematically and intuitively.
- **Value**: ⭐⭐⭐⭐ — Offers an interpretable, lightweight, and scalable new paradigm for internal value alignment in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models](../../ICML2026/llm_alignment/valueflow_toward_pluralistic_and_steerable_value-based_alignment_in_large_langua.md)
- [\[ICML 2026\] Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](../../ICML2026/llm_alignment/toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)
- [\[ACL 2025\] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](../../ICLR2026/llm_alignment/cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)
- [\[ICLR 2026\] Reward Models Inherit Value Biases from Pretraining](../../ICLR2026/llm_alignment/reward_models_inherit_value_biases_from_pretraining.md)

</div>

<!-- RELATED:END -->
