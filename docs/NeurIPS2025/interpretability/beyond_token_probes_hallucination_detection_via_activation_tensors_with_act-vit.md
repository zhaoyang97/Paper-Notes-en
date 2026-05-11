---
title: >-
  [Paper Note] Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT
description: >-
  [NeurIPS 2025][Interpretability][Hallucination Detection] This paper organizes all hidden-layer activations of an LLM into an "activation tensor" (layers × tokens × hidden dimension), treats it analogously to an image…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Hallucination Detection"
  - "Activation Tensors"
  - "Vision Transformer"
  - "Cross-Model Generalization"
  - "Probing"
date: 2026-05-08
content_hash: 1d805e99a76a2e2a
---

# Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT

**Conference**: NeurIPS 2025
**arXiv**: [2510.00296](https://arxiv.org/abs/2510.00296)
**Code**: [https://github.com/BarSGuy/ACT-ViT](https://github.com/BarSGuy/ACT-ViT)
**Area**: Interpretability
**Keywords**: Hallucination Detection, Activation Tensors, Vision Transformer, Cross-Model Generalization, Probing

## TL;DR
This paper organizes all hidden-layer activations of an LLM into an "activation tensor" (layers × tokens × hidden dimension), treats it analogously to an image, and processes it with a ViT-based architecture (ACT-ViT) that supports joint training across multiple LLMs. The method consistently outperforms conventional probing approaches across 15 LLM–dataset combinations and demonstrates strong zero-shot/few-shot transfer to unseen datasets and unseen LLMs.

## Background & Motivation
**Background**: Among methods for detecting LLM hallucinations, probing classifiers—linear classifiers trained on hidden representations—represent an efficient white-box approach. However, conventional probing operates on isolated single-layer, single-token positions, requiring prior selection of the optimal layer and token position.

**Limitations of Prior Work**:
- **Non-fixed signal location**: The optimal probing position varies substantially across samples, datasets, and LLMs—Mistral's optimal position is (layer 14, token 0), whereas Qwen's optimal position lies at the last few tokens of the final layers.
- **LLM specificity**: A separate probe must be trained for each LLM, precluding cross-model dataset sharing or transfer learning.
- **Incomplete utilization**: Using activations from only one layer–token position wastes a large amount of available information.

**Core Insight**: The activation tensor $\mathbf{A} \in \mathbb{R}^{L \times N \times D}$ (layers × tokens × hidden dimension) is structurally analogous to an image (height × width × channels), making it amenable to treatment with vision model techniques.

**Core Idea**: Treat all hidden-layer activations of an LLM as an "image" and apply a ViT to adaptively attend to the most informative layer–token combinations, enabling efficient hallucination detection that generalizes across LLMs.

## Method

### Overall Architecture
Extract the LLM activation tensor → Apply pooling to compress spatial dimensions (layer and token axes) → Map each LLM's activations to a shared feature space via a dedicated Linear Adapter → Process with a shared ViT backbone → Binary classification (hallucination / correct).

### Key Designs

1. **Activation Tensor**:

   - Definition: $\mathbf{A} \in \mathbb{R}^{L_M \times N \times D_M}$, containing the hidden states of all LLM layers across all output tokens.
   - Analogy to images: layers → vertical spatial dimension; tokens → horizontal spatial dimension; hidden dimension → channels.
   - Encodes the complete internal state, avoiding information loss caused by selecting specific layers or tokens.

2. **Pooling + Linear Adapter**:

   - Pooling: Max-pooling is applied over the "spatial" dimensions (layers and tokens) to produce a fixed size of $(L_p, N_p) = (8, 100)$, accommodating varying numbers of layers across LLMs and varying input token lengths.
   - Linear Adapter: Each LLM $M$ has a dedicated linear transformation $\mathbf{W}_M \in \mathbb{R}^{D_M \times D'}$ that maps its hidden dimension to a shared dimension $D'$.
   - Design Motivation: Motivated by the hypothesis that different LLMs learn approximately linearly transformable representations of real-world knowledge. A single linear layer is sufficient to align feature spaces across LLMs.

3. **ViT-Based Backbone**:

   - The pooled and adapted tensor is divided into non-overlapping patches; intra-patch positional encodings and global positional encodings are added.
   - Flattened patches are processed by a standard Transformer encoder.
   - The self-attention mechanism allows the model to adaptively attend to the layer–token positions most informative for hallucination detection, without requiring manual specification.

### Loss & Training
- Joint training: All available LLMs and datasets are trained simultaneously, sharing a single ViT backbone while maintaining independent Linear Adapters per LLM.
- Transfer to new LLMs: The backbone is frozen and only the new LLM's Linear Adapter is trained (lightweight adaptation).
- All 15 combinations are trained within 3 hours on a single GPU; inference speed is approximately $10^{-5}$ seconds per sample.

## Key Experimental Results

### Main Results (AUC, 15 LLM–Dataset Combinations)

| Method | Mis-7B Movies | LlaMa-8B TriviaQA | Qwen-7B HQA | Avg. Gain |
|---|---|---|---|---|
| Logits-mean | 63.0 | 66.0 | 66.2 | - |
| Probe[*] (best layer–token) | ~80–85 | ~75–82 | ~72–80 | - |
| ACT-ViT(s) (single combination) | ~85–88 | ~80–84 | ~78–83 | +3–5 vs Probe |
| **ACT-ViT** (multi-LLM joint) | **~88–92** | **~84–88** | **~82–87** | **+5–10 vs Probe** |

### Ablation Study

| Setting | Result |
|---|---|
| Zero-shot to new datasets (seen LLMs) | Strong generalization; often surpasses probes trained on the target dataset |
| Fine-tuning LA to new LLM with 5% data | Outperforms single-model probes trained on 100% data in most cases |
| Multi-LLM joint vs. single-LLM | Joint training consistently superior; cross-LLM signals are complementary |

### Key Findings
- ACT-ViT consistently outperforms conventional probing across all 15 combinations, with significant average gains.
- Multi-LLM joint training markedly surpasses single-model training, confirming that hallucination signals from different LLMs are complementary.
- Adapting to a new LLM requires training only the Linear Adapter (very few parameters) with as little as 5% of data—highly practical for deployment.
- Strong zero-shot generalization to new datasets suggests that hallucination detection signals share cross-task commonalities.
- ViT self-attention is more effective than MLP: ACT-MLP (flattening followed by MLP) performs substantially worse.

## Highlights & Insights
- The **"activation tensor = image" analogy** is particularly elegant: it reframes an NLP problem as a vision problem, leveraging ViT self-attention to adaptively identify the most informative layer–token positions, entirely avoiding the need for manual position selection in conventional probing.
- The success of **cross-LLM joint training** validates an important hypothesis: different LLMs share common mechanisms for encoding hallucinations, which can be aligned via linear transformation.
- **Exceptional efficiency**: inference at $10^{-5}$ seconds per sample (five orders of magnitude faster than LLM-based detection methods), with training completed in 3 hours across all 15 combinations.

## Limitations & Future Work
- Requires white-box access to hidden states of all LLM layers—not applicable to API-only models.
- Storage overhead for activation tensors is substantial (~0.2 GB per sample per LLM); optimized storage strategies are needed for large-scale deployment.
- Evaluation is limited to 7–8B scale models; performance on larger (70B+) or smaller (1B) models remains unknown.
- The linear adaptation assumption may not hold for LLMs with substantially different architectures.
- The method focuses solely on factual QA-type hallucinations; its effectiveness on more complex error types such as reasoning errors and subjective biases is unknown.

## Related Work & Insights
- **vs. Orgad et al. (2024)**: They identify the importance of "exact token" probing but still require an external algorithm for localization. ACT-ViT resolves the localization problem automatically by processing the full activation tensor.
- **vs. logits/probability-based methods**: These require no training but are limited in information (only the output layer is used). ACT-ViT exploits information from all layers.
- **Implications for interpretability**: The ViT attention maps can reveal which layer–token combinations carry the primary hallucination signal, offering a new perspective for LLM interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the activation-tensor-as-image perspective and the cross-LLM joint training paradigm are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 combinations, multiple settings (single-model / multi-model / zero-shot / few-shot / transfer), and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The analogy is intuitive, Figure 1 is well designed, and the experimental analysis is systematic.
- Value: ⭐⭐⭐⭐⭐ Introduces an efficient and general new paradigm for hallucination detection; cross-LLM transfer capability represents an important breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)

</div>

<!-- RELATED:END -->
