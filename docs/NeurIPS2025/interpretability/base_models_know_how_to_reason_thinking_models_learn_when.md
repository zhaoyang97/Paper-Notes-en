---
title: >-
  [Paper Note] Base Models Know How to Reason, Thinking Models Learn When
description: >-
  [NeurIPS 2025][Interpretability][Thinking Models] Through unsupervised SAE clustering, this work discovers a taxonomy of reasoning mechanisms in thinking models…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Thinking Models"
  - "Steering Vectors"
  - "SAE"
  - "Reasoning Mechanisms"
  - "Base Models"
date: 2026-05-08
content_hash: 6304f7af75f0f4a6
---

# Base Models Know How to Reason, Thinking Models Learn When

**Conference**: NeurIPS 2025
**arXiv**: [2510.07364](https://arxiv.org/abs/2510.07364)  
**Code**: [https://github.com/cvenhoff/thinking-llms-interp](https://github.com/cvenhoff/thinking-llms-interp)  
**Area**: LLM Reasoning / Mechanistic Interpretability / Representation Engineering
**Keywords**: Thinking Models, Steering Vectors, SAE, Reasoning Mechanisms, Base Models

## TL;DR
Through unsupervised SAE clustering, this work discovers a taxonomy of reasoning mechanisms in thinking models, then activates the corresponding latent capabilities in base models via steering vectors. The resulting hybrid model recovers up to 91% of the performance gap between thinking and base models—without any weight updates—demonstrating that base models already possess reasoning capabilities, and that thinking models merely learn *when* to deploy them.

## Background & Motivation
**Background**: Thinking models such as DeepSeek R1 and QwQ substantially outperform base models via long-chain reasoning, yet the source of this advantage remains unclear.

**Limitations of Prior Work**:
- Multiple competing hypotheses exist regarding why thinking models are effective: acquiring new capabilities vs. better structuring vs. reusing existing capabilities vs. simply leveraging more computation.
- Existing analyses rely on manual inspection of reasoning traces, which is subjective and may overlook systematic patterns.

**Key Challenge**: Do thinking models *learn new reasoning methods*, or do they *learn to invoke existing methods at the right moment*?

**Goal**: Provide causal evidence that base models already possess reasoning capabilities.

**Key Insight**: Use unsupervised SAE clustering to discover reasoning categories, then activate the corresponding capabilities in base models via steering vectors.

**Core Idea**: Pre-training teaches models *how* to reason; post-training (RLVR) teaches models *when* to reason.

## Method

### Overall Architecture
The pipeline consists of three stages: (1) **Unsupervised reasoning mechanism taxonomy**—sentence-level activations from thinking model reasoning traces are clustered via SAE to identify ~15 reasoning categories; (2) **Steering vector extraction**—for each reasoning category, the activation difference between base and thinking models is computed; (3) **Hybrid model**—during base model generation, a classifier detects which reasoning mechanism is currently applicable and injects the corresponding steering vector.

### Key Designs

1. **Unsupervised Reasoning Taxonomy (Top-K SAE Clustering)**:

    - **Function**: Automatically discovers interpretable reasoning mechanism categories from thinking model reasoning traces.
    - **Mechanism**: A restricted-decoder SAE (latent dimensionality 10–25, far smaller than the input dimensionality of 1536+) is trained on 430K reasoning trace sentences; each latent feature corresponds to one reasoning mechanism. The optimal configuration is selected using a three-dimensional scoring criterion of completeness, independence, and consistency.
    - **Design Motivation**: Avoids human annotation bias and ensures that the taxonomy is both complete and interpretable.

2. **Steering Vector Extraction and Application**:

    - **Function**: Encodes each reasoning mechanism as a directional vector; injecting it into a base model activates the corresponding capability.
    - **Mechanism**: For each SAE cluster, the mean activation difference between the base model and the thinking model on sentences belonging to that cluster is computed as the steering vector. During generation, SAE activations are computed at each token position to identify the most active reasoning category, and the corresponding vector is injected.
    - **Design Motivation**: If the base model already possesses latent reasoning capabilities, a simple directional offset should suffice to activate them.

3. **Hybrid Model Inference**:

    - **Function**: Base model + selective steering = recovery of thinking model performance.
    - **Mechanism**: The base model handles primary token generation; at each position, a classifier based on the thinking model's SAE activations detects the current reasoning category and applies the corresponding steering vector. The optimal steering coefficient and window size are selected via perplexity. Only 12% of tokens are steered.
    - **Design Motivation**: Achieving significant improvement with only 15 steering vectors rules out the alternative explanation that steering merely biases the model toward specific tokens.

### Loss & Training
- **SAE**: Top-K sparse autoencoder, where $K$ constrains the number of simultaneously active reasoning categories.
- **Hybrid model**: No training required; zero weight updates; steering vectors are injected only at inference time.
- **Selection criterion**: Steering intensity is chosen to minimize thinking model perplexity.

## Key Experimental Results

### Main Results
Hybrid model performance on GSM8K and MATH500:

| Base Model | Thinking Model | Base Acc | Hybrid Acc | Thinking Acc | Gap Recovery |
|---|---|---|---|---|---|
| Llama-3.1-8B | R1-Distill-8B | 37.8% | 63.4% | 83.4% | **56.1%** |
| Qwen2.5-14B | R1-Distill-14B | 90.8% | 93.0% | 94.2% | **64.7%** |
| Qwen2.5-32B | R1-Distill-32B | 92.6% | 94.4% | 94.8% | **81.8%** |
| Qwen2.5-32B | QwQ-32B | 92.6% | 94.8% | 96.4% | 57.9% |

### Ablation Study

| Configuration | Description |
|---|---|
| Random steering vectors | Performance degrades, ruling out coincidence |
| Wrong-category steering | Performance degrades, demonstrating the importance of category matching |
| Only 12% of tokens steered | The majority of tokens are generated autonomously by the base model |

### Key Findings
- Only 15 steering vectors suffice to recover up to 91% of the performance gap (Qwen2.5-Math-1.5B on MATH500).
- Llama-8B achieves the largest absolute gain (+25.6%), confirming that base models do possess latent reasoning capabilities.
- The approach generalizes to both distillation-trained models (R1-Distill) and RLVR-trained models (QwQ).
- The optimal reasoning mechanism taxonomy contains 15–25 categories, consistent with the number of "basic reasoning operations" identified in cognitive science.

## Highlights & Insights
- **Redefining the role of post-training**: RLVR does not teach models *how* to reason, but rather *when* to reason—a finding that fundamentally reshapes the understanding of thinking model training.
- **Methodological contribution of unsupervised reasoning taxonomy**: Using a restricted SAE for clustering constitutes a novel tool for uncovering the cognitive structure of AI systems.
- **Minimalist causal validation**: Steering only 12% of tokens with 15 vectors yields substantial improvements, constituting strong causal evidence.
- **Cross-architecture and cross-training generalization**: The approach is effective across Qwen, Llama, distillation-trained, and RLVR-trained models.

## Limitations & Future Work
- The hybrid model relies on the thinking model's SAE classifier, meaning the thinking model is still required at deployment time.
- Validation is limited to mathematical reasoning tasks (GSM8K/MATH500); code and logical reasoning tasks remain untested.
- The gap recovery for Qwen2.5-Math-1.5B is 0% (as the base model is already very strong), indicating that steering yields no additional benefit when the base model is sufficiently capable.
- Sentence-level clustering granularity may be insufficiently fine-grained.

## Related Work & Insights
- **vs. Gandhi et al.**: Their work suggests thinking models acquire new capabilities; the evidence presented here points instead to the reuse of existing ones.
- **vs. Ward et al.**: The most closely related work, which also finds that RL reuses pre-trained representations, but the present paper provides a more complete causal validation.
- **vs. Steering vector work [Turner et al.]**: Previously applied to controlling style and sentiment; this paper is the first to apply steering vectors to activating reasoning mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The insight that "base models already know how to reason; thinking models only learn when to" is highly influential.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Cross-validation across 3 base models and 4 thinking models is comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous logic with well-designed figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Reshapes the understanding of the thinking model training paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distributional Autoencoders Know the Score](distributional_autoencoders_know_the_score.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](../../ICLR2026/interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)

</div>

<!-- RELATED:END -->
