---
title: >-
  [Paper Note] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates
description: >-
  [ACL 2025][LLM (Other)][FFN Sub-Updates] LLMBraces dynamically adjusts the contribution weights of sub-updates by computing the relevance scores between each value vector in the FFN layers and the input. Using extremely few parameters (75% fewer than LoRA), it simultaneously improves model prediction accuracy and enables controllable text generation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "FFN Sub-Updates"
  - "Relevance Scores"
  - "Parameter-Efficient Fine-Tuning"
  - "Controllable Text Generation"
  - "Low-Rank Projection"
date: 2026-05-08
content_hash: f289b27b4ecb2eb4
---

# LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates

**Conference**: ACL 2025  
**arXiv**: [2503.16334](https://arxiv.org/abs/2503.16334)  
**Area**: LLM NLP / Model Interpretability and Parameter-Efficient Fine-Tuning  
**Keywords**: FFN Sub-Updates, Relevance Scores, Parameter-Efficient Fine-Tuning, Controllable Text Generation, Low-Rank Projection

## TL;DR

LLMBraces dynamically adjusts the contribution weights of sub-updates by computing the relevance scores between each value vector in the FFN layers and the input. Using extremely few parameters (75% fewer than LoRA), it simultaneously improves model prediction accuracy and enables controllable text generation.

## Background & Motivation

- **Knowledge Storage Role of FFN Layers**: Existing studies (Geva et al. 2021, 2022) show that the FFN layers of Transformers can be viewed as key-value memories, where each FFN update can be decomposed into multiple sub-updates $w_{i,j} \cdot v_j$, and the value vectors usually encode human-interpretable concepts.
- **Noise Issue in Sub-Updates**: Not all sub-updates are relevant to the current input, and irrelevant sub-updates can introduce noise or erroneous predictions.
- **Core Hypothesis**: Dynamically adjusting the contributions of sub-updates—amplifying those highly relevant to the input and suppressing irrelevant ones—can improve model accuracy.
- **Extension Potential**: The same mechanism can be applied to controllable generation, guiding the output by measuring the alignment of value vectors with target attributes (such as sentiment).

## Method

### Overall Architecture

Without modifying the original LLM parameters, LLMBraces introduces a lightweight **relevance module** $R(\cdot)$ for each FFN layer to calculate the alignment between each value vector $v_j$ and the input hidden state $h_i$. The resulting relevance scores are then used as additive adjustments to enhance the original sub-update weights.

Standard FFN: 
$$\text{FFN}(h_i) = \sum w_{i,j} \cdot v_j$$

Enhanced FFN_AUG: 
$$\text{FFN\_AUG}(h_i) = \sum (w_{i,j} + g \cdot r_{i,j}) \cdot v_j$$

### Key Designs

#### 1. Relevance Module

Core calculation: 
$$r_i = R(W_V, h_i) = \frac{(R \cdot W_V)^T (R \cdot h_i)}{\sqrt{d_r}}$$

- **$R \in \mathbb{R}^{d_r \times d}$** is a learnable low-rank projection matrix with orthonormal rows.
- High-dimensional value matrices and hidden states are projected into a shared low-dimensional subspace to calculate relevance.
- Low-rank projection combined with orthogonal constraints guarantees computational efficiency while faithfully preserving the geometric structure of the original high-dimensional representations.
- Each $r_{i,j}$ quantifies the alignment between the $j$-th value vector and the current token context.

#### 2. Gating Mechanism

- A learnable gating parameter $g^\ell = \sigma(g)$ is introduced, where $g$ is initialized to -5 (near-zero initialization).
- This ensures that the model behavior in the early stage of training is close to the original model, gradually introducing relevance-based enhancements as training progresses.
- An additive formulation is preferred over a multiplicative one because the original weights $w_{i,j}$ can be negative, which multiplicative or binary filtering cannot handle effectively.

#### 3. Task-Specific Relevance

For controllable text generation (e.g., sentiment steering or toxicity detoxification):

- Attribute-specific tokens (e.g., positive sentiment words like "happy", "joyful") are passed through the target LLM to extract hidden states.
- Pooling and projection via an MLP yield the attribute representation $h_c$.
- Compute the conditional relevance score: 
  $$r_c = R(W_V, h_c)$$
- Superimpose it with the original relevance score: 
  $$r_i \leftarrow r_i + s \cdot r_c$$
- Here, $s$ is a user-adjustable scalar controlling the direction and intensity of attribute steering.

#### 4. Loss & Training

- All parameters of the original LLM are frozen.
- Only the newly introduced parameters are trained: the low-rank projection matrix $R$ and the gating parameter $g$ for each layer.
- Training is performed using the standard language modeling objective.

## Key Experimental Results

### Main Results

**Commonsense Reasoning Fine-Tuning (Average Accuracy of 8 Tasks)**:

| Model | Method | # Params | Param % | AVG |
|------|------|--------|----------|-----|
| Qwen2.5-1.5B | LoRA r=16 | 2.2M | 0.14% | 79.83 |
| Qwen2.5-1.5B | **LLMBraces r=16** | **0.6M** | **0.04%** | **80.28** |
| Llama2-7B | LoRA r=16 | 8.4M | 0.12% | 81.58 |
| Llama2-7B | **LLMBraces r=16** | **2.1M** | **0.03%** | **81.50** |
| Llama3-8B | LoRA r=16 | 6.8M | 0.08% | 84.72 |
| Llama3-8B | **LLMBraces r=32** | **4.2M** | **0.05%** | **86.51** |

- LLMBraces achieves performance comparable to or surpassing LoRA while using only 25%-30% of LoRA's parameters.

**Zero-Shot Generalization (Average of 6 Tasks)**:

| Model | Method | AVG |
|------|------|-----|
| Qwen2.5-1.5B | LoRA r=16 | 23.76 |
| Qwen2.5-1.5B | **LLMBraces r=32** | **27.07** (+13.9%) |
| Llama2-7B | LoRA r=16 | 27.17 |
| Llama2-7B | **LLMBraces r=16** | **32.49** (+19.6%) |
| Llama3-8B | LoRA r=16 | — |
| Llama3-8B | **LLMBraces** | — (+29.7%) |

- The improvement is particularly significant in the zero-shot setting: Qwen2.5 +13.9%, Llama2 +19.6%, Llama3 +29.7%.

### Key Findings

1. **Extremely High Parameter Efficiency**: Requiring only 25% of LoRA's parameters to achieve better results, which is attributed to directly manipulating existing value vectors instead of adding extra parameters.
2. **Obvious Advantages in Zero-Shot Scenarios**: Significant improvements in tasks such as factual knowledge (PopQA, TriviaQA) and truthfulness (TruthfulQA), indicating that relevance enhancement helps retain knowledge.
3. **Controllable Generation Capability**: Excellent performance in both sentiment steering and toxicity detoxification tasks, where the target generation attributes can be flexibly controlled simply by adjusting the scalar $s$.
4. **Rationality of the Additive Formulation**: Able to handle cases where original weights are negative, while the gating mechanism ensures training stability.

## Highlights & Insights

- **Clear Theoretical Motivation**: Grounded in the interpretability research of FFN as key-value memory, naturally deriving the method of "adjusting weights of sub-updates."
- **Extreme Parameter Efficiency**: The low-rank orthogonal projection matrix is the only additional parameter, making the design exceptionally streamlined.
- **Unified Framework**: The same mechanism serves dual purposes: performance enhancement and controllable generation.
- **Near-Zero Initialization Strategy**: Gating parameters initialized to -5 ensure that the original model behavior is not disrupted at the start of training, representing a clever engineering detail.
- **Plug-and-Play**: Seamlessly integrates into any Transformer-based LLM without modifying the model architecture.

## Limitations & Future Work

- Requires training the relevance module separately for each target LLM.
- Controllable generation relies on a predefined list of attribute tokens, and its efficacy on complex attributes (e.g., writing style) remains to be validated.
- Currently only validated on small-to-medium scale models (1.5B–8B); the efficacy on larger-scale models remains unknown.
- Although parameters are few, it introduces an extra low-rank projection calculation during inference, incurring additional computational overhead.

## Related Work & Insights

- **LLM Internal Mechanism Analysis**: Geva et al. (2021, 2022) interpret the FFN as a key-value memory; nostalgebraist (2020) explores vocabulary projection of internal representations.
- **Model Editing**: ROME/MEMIT (directly modifying parameters to update facts)—LLMBraces does not target specific outputs but functions as a general enhancement.
- **Parameter-Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation)—LLMBraces replaces additional matrices with relevance scores, using fewer parameters.
- **Controllable Text Generation**: Activation steering, PEFT methods—LLMBraces achieves more flexible guidance using relevance scores.

## Rating

- **Novelty**: ★★★★☆ — Based on the interpretability of FFN sub-updates, a lightweight and effective enhancement method is proposed with a novel perspective.
- **Value**: ★★★★★ — Extremely low parameter budget, plug-and-play, supports both performance improvement and controllable generation, indicating high practical value.
- **Experimental Thoroughness**: ★★★★☆ — Covers three models and multi-task scenarios with thorough ablation studies, though lacking evaluation on large-scale models and inference speed comparison.
- **Writing Quality**: ★★★★☆ — Clearly written with coherent logic from motivation to method and experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)
- [\[ACL 2025\] SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)
- [\[ACL 2025\] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](../../ICML2026/llm_nlp/scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)

</div>

<!-- RELATED:END -->
