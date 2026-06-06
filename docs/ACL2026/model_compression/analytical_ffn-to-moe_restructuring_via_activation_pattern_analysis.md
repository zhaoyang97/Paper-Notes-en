---
title: >-
  [Paper Note] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis
description: >-
  [ACL 2026][Model Compression][FFN-to-MoE] Ours proposes an analytical post-training framework that rapidly restructures dense FFNs into sparse MoEs via neuron activation pattern analysis. By distinguishing high-frequency…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "FFN-to-MoE"
  - "Activation Pattern Analysis"
  - "Shared Experts"
  - "Analytical Routing"
  - "Post-training Compression"
date: 2026-05-08
content_hash: 6765acb925e90f91
---

# Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis

**Conference**: ACL 2026  
**arXiv**: [2502.04416](https://arxiv.org/abs/2502.04416)  
**Code**: [GitHub](https://github.com/JarvisPei/CMoE)  
**Area**: Model Compression / MoE  
**Keywords**: FFN-to-MoE, Activation Pattern Analysis, Shared Experts, Analytical Routing, Post-training Compression  

## TL;DR

Ours proposes an analytical post-training framework that rapidly restructures dense FFNs into sparse MoEs via neuron activation pattern analysis. By distinguishing high-frequency shared experts from low-frequency routed experts and constructing routers from activation statistics, the method achieves a 1.17× speedup with only 2k samples for fine-tuning.

## Background & Motivation

**Background**: Mixture-of-Experts (MoE) architectures decouple parameter scale from computational cost through sparse activation. However, traditional methods require training MoE models from scratch, which is prohibitively expensive.

**Limitations of Prior Work**: (1) Existing dense-to-MoE methods (e.g., MoEfication) rely on weight clustering, ignoring differences in activation frequencies across neurons; (2) Methods like LLaMA-MoE require 200B tokens of continual training to recover performance; (3) A critical observation is overlooked—neuron activation frequencies follow a bimodal distribution, where few neurons are always active and most are only conditionally active.

**Key Challenge**: Treating high-frequency (always active) neurons and low-frequency (conditionally active) neurons uniformly forces the router to activate most experts for almost every input, thereby destroying MoE sparsity.

**Goal**: To design an analytical (training-efficient) FFN-to-MoE method by exploiting the bimodal structure of activation patterns.

**Key Insight**: FFN hidden layer activations are highly sparse and bimodal. High-frequency neurons can be placed in a shared expert, while low-frequency neurons are clustered into routed experts based on co-activation. The router can then be directly constructed from statistical data.

**Core Idea**: The structured partitioning into shared and routed experts leverages the natural activation patterns, allowing the router to select only among experts that are truly input-dependent.

## Method

### Overall Architecture

The process consists of three stages: (A) Activation Pattern Analysis—computing the activation rate $\mu_i$ for each neuron using a small amount of calibration data; (B) Expert Construction—assigning high-frequency neurons to a shared expert and clustering low-frequency neurons into routed experts via a balanced assignment algorithm; (C) Analytical Router—directly constructing the routing function from activation statistics without training.

### Key Designs

1. **Shared/Routed Expert Splitting based on Activation Rate**:
    - **Function**: Constructing natural expert partitions using the bimodal activation structure.
    - **Mechanism**: Calculates the activation rate $\mu_i$ for each neuron (the proportion of times it appears in top-$K_a$). High-frequency neurons enter the shared expert $E^s$ (always active), while others are clustered into routed experts $E_i^r$ based on activation pattern similarity.
    - **Design Motivation**: High-frequency neurons are important for almost all inputs. Distributing them across different routed experts would force most experts to be active simultaneously, undermining sparsity.

2. **Analytical Router Construction**:
    - **Function**: Determining which routed experts to activate for each input without training.
    - **Mechanism**: The task of minimizing reconstruction error $\|F_{MoE}(\mathbf{x}) - F(\mathbf{x})\|^2$ is reduced to minimizing the output contribution of inactive experts. Using the $L_1$ norm of each expert's hidden state as a contribution proxy, the router selects the top-$N_k$ experts with the largest contributions.
    - **Design Motivation**: This bypasses expensive router training by deriving routing signals directly from the activation statistics of the original FFN.

3. **Hierarchical Sparsity (Recursive Application to Existing MoE)**:
    - **Function**: Applying the framework inside each expert of an existing MoE model to achieve fine-grained sparsity.
    - **Mechanism**: The same shared/routed splitting is applied recursively to the FFN of each expert in an MoE model.
    - **Design Motivation**: While Dense-to-MoE applies to dense models, recursive application extends the framework to further accelerate MoE models.

### Loss & Training

Analytical restructuring is entirely training-free (the training-free baseline can be deployed directly). Optional fine-tuning with 2k samples using standard language modeling loss further enhances quality.

## Key Experimental Results

### Main Results

| Configuration | Speedup | Processing Time | Quality |
|---------------|---------|-----------------|---------|
| Training-free | 1.17×   | Minutes         | Usable  |
| +2k Fine-tuning | 1.17× | Minutes + FT    | Surpasses methods requiring orders of magnitude more resources |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Unified vs. Split Experts | Split is significantly better | Validates the value of bimodal splitting |
| Analytical vs. Learned Routing | Analytical is comparable | Eliminates the need for router training |
| Recursive Hierarchical Sparsity | Effective | Further accelerates MoE models |

### Key Findings
- Bimodal activation patterns are prevalent across multiple LLM architectures (LLaMA-2, Mistral, etc.).
- Minutes of processing plus fine-tuning on 2k samples can outperform methods requiring 200B tokens of training.
- The quality of the analytical router is close to that of a learned router, significantly reducing costs.

## Highlights & Insights
- **Observation-driven design**: Starting from the bimodal distribution of activation patterns, the method design is natural and elegant.
- The efficiency comparison of "minutes of processing vs. 200B token training" is highly compelling.
- The recursive application idea makes the method applicable to both dense and MoE models.

## Limitations & Future Work
- The 1.17× speedup is relatively modest and may be insufficient for extreme low-latency scenarios.
- The choice of the shared expert ratio requires adjustment based on the specific model.
- Not yet tested on vision or multimodal models.
- Future work could integrate orthogonal technologies like quantization for further acceleration.

## Related Work & Insights
- **vs. MoEfication**: Ours distinguishes between shared and routed neurons instead of uniform clustering, fundamentally leveraging the activation structure.
- **vs. LLaMA-MoE**: Eliminates the need for large-scale continual training, reducing costs by several orders of magnitude.
- **vs. Activation Sparsity (e.g., DejaVu)**: Operates at a different granularity and can be used in combination.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Integration of bimodal activation observations with an analytical router.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models, multiple tasks, and comparison against strong baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent logical flow from observation to motivation, method, and validation.
- **Value**: ⭐⭐⭐⭐ Direct practical value for efficient LLM inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](impact_importance-aware_activation_space_reconstruction.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)
- [\[ACL 2026\] When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](when_reviews_disagree_fine-grained_contradiction_analysis_in_scientific_peer_rev.md)

</div>

<!-- RELATED:END -->
