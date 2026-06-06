---
title: >-
  [Paper Note] A Layer-wise Analysis of Supervised Fine-Tuning
description: >-
  [ACL 2026][Model Compression][Supervised Fine-Tuning] A layer-wise analysis of SFT in 1B-32B models is conducted through information-theoretic, geometric…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Supervised Fine-Tuning"
  - "Layer-wise Analysis"
  - "Parameter-Efficient Fine-Tuning"
  - "Catastrophic Forgetting"
  - "LoRA"
date: 2026-05-08
content_hash: 20aa36bf5942bcee
---

# A Layer-wise Analysis of Supervised Fine-Tuning

**Conference**: ACL 2026  
**arXiv**: [2604.11838](https://arxiv.org/abs/2604.11838)  
**Code**: [GitHub](https://github.com/lshowway/base)  
**Area**: Model Compression  
**Keywords**: Supervised Fine-Tuning, Layer-wise Analysis, Parameter-Efficient Fine-Tuning, Catastrophic Forgetting, LoRA

## TL;DR
A layer-wise analysis of SFT in 1B-32B models is conducted through information-theoretic, geometric, and optimization perspectives. The study finds that instruction-following capabilities are concentrated in middle layers (20%-80%) rather than uniformly distributed. Based on this, a Mid-Block Efficient Tuning strategy is proposed, selectively updating middle layers to achieve up to a 10.2% improvement over standard LoRA on GSM8K.

## Background & Motivation

**Background**: Supervised Fine-Tuning (SFT) is a fundamental method for aligning LLMs with human intent. Research indicates that only approximately 1,000 curated samples are needed to transform a base model into an instruction-following agent. Existing studies have revealed that SFT primarily recalibrates attention patterns and adjusts stylized token distributions, essentially acting as a "surface-level" adaptation.

**Limitations of Prior Work**: Current parameter-efficient fine-tuning methods (e.g., LoRA) apply updates uniformly across all layers, implicitly assuming that all layers contribute equally to alignment. However, this assumption is sub-optimal—different layers may have entirely different functional roles. Crucially, uniform updates may waste parameter budgets on insensitive layers while leading to insufficient updates in sensitive ones.

**Key Challenge**: While it is known "what changes" during SFT (attention patterns, token distributions), it remains unclear "where it changes"—specifically, how these changes are distributed across model depth and which layers are critical for instruction-following capabilities.

**Goal**: (1) Systematically reveal the layer-wise change patterns induced by SFT; (2) Identify the layer intervals most critical for task adaptation; (3) Propose more efficient fine-tuning strategies based on analytical insights.

**Key Insight**: A systematic layer-level dissection is performed across 1B-32B model scales using integrated metrics: information-theoretic (entropy, effective rank), geometric (CKA, cosine similarity), and optimization-based (weight change magnitude).

**Core Idea**: Effective alignment in SFT is "architecturally localized" rather than uniform. Middle layers (20%-80%) serve as a stable foundation for knowledge integration, while top layers are the primary source of catastrophic forgetting. Therefore, updates should be concentrated on the middle layers.

## Method

### Overall Architecture
A pipeline for layer-wise representation analysis of Base and SFT models is constructed. Given Base and SFT checkpoints of the same architecture, hidden state matrices for each layer are extracted from the same dataset. Layer-wise differences are then quantified from three perspectives: optimization dynamics, information dynamics, and geometric reconstruction.

### Key Designs

1. **Optimization Dynamics (Weight Change)**:

    - **Function**: Quantifies the actual magnitude of parameter changes in each layer after SFT.
    - **Mechanism**: Defines $\Delta \mathcal{W}^{(l)}$ as the Frobenius distance between the Base and SFT models for all projection matrices (Q/K/V/O) in the attention module of the $l$-th layer. A high $\Delta \mathcal{W}^{(l)}$ indicates that the layer has undergone aggressive parameter modification.
    - **Design Motivation**: Directly observes the distribution of "acting forces" from SFT in the parameter space to verify if non-uniform updates occur due to gradient decay.

2. **Information Dynamics (Entropy & Effective Rank)**:

    - **Function**: Monitors the impact of SFT on the information capacity of the representation space.
    - **Mechanism**: Utilizes matrix-based $\alpha$-order entropy and effective rank to analyze changes in information density per layer before and after SFT. Prompt entropy quantifies token-level density within sequences, while Dataset entropy quantifies diversity across samples. Effective rank measures the true dimensionality of the representation space.
    - **Design Motivation**: Validates the information bottleneck hypothesis—whether SFT forces the model to compress general pre-training features to fit task constraints.

3. **Geometric Reconstruction (CKA, Cosine Similarity, Mean Shift)**:

    - **Function**: Determines whether SFT merely rotates the representation space or fundamentally reconstructs it.
    - **Mechanism**: CKA measures the global structural similarity between Base and SFT models at each layer; cosine similarity measures directional re-orientation; mean shift measures whether representations are physically transported to new regions of the vector space.
    - **Design Motivation**: Correlates parameter space changes (optimization dynamics) with representation space changes (geometric reconstruction) to establish a causal chain.

### Verification Experimental Design

The paper establishes causal relationships through three complementary verification experiments: (1) **Layer-wise Probing**: Predicting the next token directly from the output of each intermediate layer to observe the "dormancy $\rightarrow$ emergence" pattern of task capability; (2) **Layer-wise Weight Change**: Tracking the L2 update magnitude of each layer after LoRA fine-tuning; (3) **Layer-wise Swapping**: Replacing specific layer blocks of the Base model with corresponding SFT layers (and vice versa) to observe performance changes.

## Key Experimental Results

### Main Results (Mid-Block Efficient Tuning vs Standard LoRA, GSM8K Accuracy)

| Model | Standard LoRA | Mid-Block (Best) | Gain |
|------|--------------|-----------------|------|
| OLMo2-1B | 0.19 | 0.21 (01100) | +10.5% |
| OLMo2-7B | 0.28 | 0.375 (01000) | +33.9% |
| OLMo2-13B | 0.27 | 0.30 (01110) | +11.1% |
| OLMo2-32B | 0.29 | 0.32 (01100) | +10.3% |

### Ablation Study (Block Selection, OLMo2-7B, GSM8K)

| Layer Configuration | Accuracy | Description |
|---------|----------|------|
| 10000 (Bottom 20%) | ~0.22 | Worst, significantly below baseline |
| 01000 (Upper-Middle) | 0.375 | **Best**, exceeds baseline by 10pp |
| 00010 (Lower-Middle) | ~0.27 | Near baseline |
| 00001 (Top 20%) | ~0.135 | Extremely poor, mapping layers cannot work independently |
| 11111 (All Layers) | 0.28 | Standard LoRA baseline |

### Key Findings
- **Depth-dependent patterns are consistent across all model scales (1B-32B)**: CKA remains stable in shallow layers ($>0.98$) and drops sharply in the last ~20% of layers.
- **Layer-wise probing exhibits a "dormant-to-emergent" pattern**: In OLMo2-32B, accuracy is near zero for the first 50 layers and rises sharply to 0.60 in the final 14 layers.
- **Weight changes follow a J-shaped trajectory**: Changes in early layers are minimal (~0.05), increasing significantly ($>0.10$) as the output layer is approached.
- **The performance gap between optimal middle layers and worst edge layers often exceeds 20%**, confirming the criticality of layer selection.
- **Layer swapping experiments show an inverted U-shape**: Replacing edge layers leads to performance degradation, while replacing middle layers can yield slight improvements.

## Highlights & Insights
- **The complementarity of the three perspectives** is a methodological highlight: Information theory examines "how much information changed," geometry examines "how much the spatial structure changed," and optimization examines "how much the parameters changed." These mutually validate a complete chain of evidence.
- The discovery that **"middle layers are stable bases for knowledge integration, while top layers are the main source of catastrophic forgetting"** has broad practical implications—it can guide layer selection strategies for LoRA, freezing strategies, and layer allocation in multi-task fine-tuning.
- The **Mid-Block strategy achieves better performance with fewer parameters**, suggesting that "precision targeting" is more effective than "broad updates," which is insightful for the field of parameter-efficient fine-tuning.

## Limitations & Future Work
- Validated only on standard dense decoder-only architectures; not extended to MoE or encoder-decoder architectures.
- Focuses only on the SFT stage; layer-wise dynamics after RLHF/DPO were not investigated.
- The 20%-80% range for Mid-Block was chosen empirically; an adaptive method for determining layer boundaries is lacking.
- Evaluation tasks are primary mathematical reasoning (GSM8K); generalization to other task types requires further validation.
- Future work could explore combining adaptive methods like AdaLoRA to allow the model to automatically learn the optimal rank allocation for each layer.

## Related Work & Insights
- **vs Standard LoRA**: LoRA applies low-rank updates uniformly across all layers, wasting parameter budget. This paper proves that concentrating on middle layers is more effective.
- **vs Layer-wise Pruning Literature**: Pruning research focuses on "which layers can be removed," while this paper focuses on "which layers should be updated," making them complementary.
- **vs Surface Alignment Hypothesis**: This paper provides a layer-wise refinement of this hypothesis—surface alignment does not occur uniformly across all layers but is concentrated at specific depths.

## Rating
- Novelty: ⭐⭐⭐⭐ Analytical perspectives are comprehensive, though the core finding (large changes in top layers) is not entirely counter-intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models from 1B-32B, though downstream evaluation tasks are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich visualizations, though formulas are somewhat dense.
- Value: ⭐⭐⭐⭐ Highly relevant for PEFT practitioners; the Mid-Block strategy is simple and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ACL 2026\] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)
- [\[ACL 2026\] LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference](leap_layer-wise_exit-aware_pretraining_for_efficient_transformer_inference.md)
- [\[ACL 2026\] When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](when_reviews_disagree_fine-grained_contradiction_analysis_in_scientific_peer_rev.md)

</div>

<!-- RELATED:END -->
