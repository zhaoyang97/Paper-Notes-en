---
title: >-
  [Paper Note] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis
description: >-
  [ACL 2026][Model Compression][FFN-to-MoE] This paper proposes an analytical post-training framework that rapidly restructures dense FFN layers into sparse MoE by analyzing neuron activation patterns — distinguishing high…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "FFN-to-MoE"
  - "activation pattern analysis"
  - "shared experts"
  - "analytical routing"
  - "post-training compression"
date: 2026-05-08
content_hash: d9e1b4794c2e635e
---

# Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis

**Conference**: ACL 2026
**arXiv**: [2502.04416](https://arxiv.org/abs/2502.04416)
**Code**: [GitHub](https://github.com/JarvisPei/CMoE)
**Area**: Model Compression / MoE
**Keywords**: FFN-to-MoE, activation pattern analysis, shared experts, analytical routing, post-training compression

## TL;DR

This paper proposes an analytical post-training framework that rapidly restructures dense FFN layers into sparse MoE by analyzing neuron activation patterns — distinguishing high-frequency shared experts from low-frequency routed experts and constructing routers directly from activation statistics — achieving 1.17× speedup with only 2k-sample fine-tuning.

## Background & Motivation

**Background**: MoE architectures decouple parameter scale from computational cost via sparse activation, yet conventional approaches require training MoE models from scratch at prohibitive cost.

**Limitations of Prior Work**: (1) Existing dense-to-MoE methods (e.g., MoEfication) rely on weight clustering and ignore differences in neuron activation frequency; (2) methods such as LLaMA-MoE require continued training on 200B tokens to recover quality; (3) a critical observation has been overlooked — neuron activation frequencies follow a bimodal distribution, with a small subset always active and the majority conditionally active.

**Key Challenge**: Treating always-active high-frequency neurons and conditionally active low-frequency neurons uniformly forces the router to activate most experts for nearly every input, undermining the sparsity of MoE.

**Goal**: Exploit the bimodal structure of activation patterns to design an analytical (large-scale training-free) FFN-to-MoE conversion method.

**Key Insight**: FFN hidden-layer activations are highly sparse and bimodally distributed — high-frequency neurons are assigned to a shared expert, while low-frequency neurons are clustered into routed experts by co-activation similarity, and the router is constructed directly from statistics.

**Core Idea**: The structured partition into shared and routed experts leverages the natural structure of activations, enabling the router to select only among genuinely input-dependent experts.

## Method

### Overall Architecture

The framework consists of three stages: (A) Activation Pattern Analysis — computing the activation rate $\mu_i$ of each neuron using a small calibration dataset; (B) Expert Construction — high-frequency neurons are assigned to the shared expert, while low-frequency neurons are clustered into routed experts via a balanced assignment algorithm; (C) Analytical Router — the routing function is constructed directly from activation statistics without any training.

### Key Designs

1. **Shared/Routed Expert Partition Based on Activation Rate**:

    - Function: Constructs a natural expert partition by exploiting the bimodal activation structure.
    - Mechanism: The activation rate $\mu_i$ (proportion of occurrences in the top-$K_a$ activations) is computed for each neuron. High-frequency neurons enter the shared expert $E^s$ (always active); the remainder are clustered into routed experts $E_i^r$ by activation pattern similarity.
    - Design Motivation: High-frequency neurons are important for nearly all inputs; distributing them across routed experts would force most experts to be activated at all times, destroying sparsity.

2. **Analytical Router Construction**:

    - Function: Determines which routed experts to activate for each input without any training.
    - Mechanism: The reconstruction error minimization $\|F_{MoE}(\mathbf{x}) - F(\mathbf{x})\|^2$ is reduced to minimizing the output contribution of inactive experts. The $L_1$ norm of each expert's hidden state is used as a proxy for contribution, and the router selects the top-$N_k$ experts with the largest contributions.
    - Design Motivation: This bypasses costly router training by deriving routing signals directly from the activation statistics of the original FFN.

3. **Hierarchical Sparsity (Recursive Application to Existing MoE)**:

    - Function: Achieves finer-grained sparsity by recursively applying the framework within each expert of an existing MoE model.
    - Mechanism: The same shared/routed partition is applied recursively to the FFN of each expert in an MoE model.
    - Design Motivation: While the dense→MoE pipeline targets dense models, recursive application extends the method to further accelerate existing MoE models.

### Loss & Training

The analytical restructuring is entirely training-free (the training-free baseline can be deployed directly). An optional fine-tuning step using 2k samples with a standard language modeling loss further improves quality.

## Key Experimental Results

### Main Results

| Configuration | Speedup | Processing Time | Quality |
|---|---|---|---|
| Training-free | 1.17× | Minutes | Usable |
| +2k fine-tuning | 1.17× | Minutes + fine-tuning | Surpasses methods requiring orders of magnitude more resources |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Unified vs. partitioned experts | Partition significantly better | Validates the value of bimodal splitting |
| Analytical vs. learned router | Analytical is comparable | No router training required |
| Recursive hierarchical sparsity | Effective | Further accelerates MoE models |

### Key Findings
- The bimodal activation pattern is pervasive across multiple LLM architectures (LLaMA-2, Mistral, etc.).
- Minutes of processing plus 2k-sample fine-tuning suffices to surpass methods requiring 200B tokens of training.
- The analytical router achieves quality comparable to learned routers at drastically reduced cost.

## Highlights & Insights
- The method is observation-driven — starting from the bimodal distribution of activation patterns, the design is natural and elegant.
- The efficiency contrast of "minutes of processing vs. 200B-token training" is highly compelling.
- The recursive application extends the method to both dense and MoE models.

## Limitations & Future Work
- The 1.17× speedup is relatively modest and may be insufficient for extreme low-latency scenarios.
- The proportion of shared experts requires model-specific tuning.
- The method has not been evaluated on vision or multimodal models.
- Future work could combine the approach with orthogonal techniques such as quantization for further acceleration.

## Related Work & Insights
- **vs. MoEfication**: Distinguishes shared and routed neurons rather than applying uniform clustering, fundamentally exploiting activation structure.
- **vs. LLaMA-MoE**: Eliminates the need for large-scale continued training, reducing cost by orders of magnitude.
- **vs. Activation Sparsity Methods (e.g., DejaVu)**: Operates at a different granularity and can be used in combination.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the bimodal activation observation and the analytical router is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and tasks against strong baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The observation→motivation→method→validation logical chain is exemplary.
- Value: ⭐⭐⭐⭐ Directly applicable to efficient LLM inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](impact_importance-aware_activation_space_reconstruction.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)
- [\[ACL 2026\] A Computational Method for Measuring "Open Codes" in Qualitative Analysis](a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)

</div>

<!-- RELATED:END -->
