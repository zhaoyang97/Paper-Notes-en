---
title: >-
  [Paper Note] Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning
description: >-
  [CVPR 2025][LLM Safety][Multilinear Operator Networks] To address the catastrophic forgetting problem of multilinear operator networks in Leveled Fully Homomorphic Encryption (Leveled FHE) scenarios, an incremental learning method combining Low-Rank Adaptation (LoRA) and Gradient Projection Memory (GPM) mechanisms is proposed to achieve continual learning while preserving data security.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Multilinear Operator Networks"
  - "Fully Homomorphic Encryption"
  - "Low-Rank Adaptation"
  - "Catastrophic Forgetting"
  - "Gradient Projection Memory"
date: 2026-05-08
content_hash: d06799f96afaeb6e
---

# Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Secure Incremental Learning  
**Keywords**: Multilinear Operator Networks, Fully Homomorphic Encryption, Low-Rank Adaptation, Catastrophic Forgetting, Gradient Projection Memory

## TL;DR

To address the catastrophic forgetting problem of multilinear operator networks in Leveled Fully Homomorphic Encryption (Leveled FHE) scenarios, an incremental learning method combining Low-Rank Adaptation (LoRA) and Gradient Projection Memory (GPM) mechanisms is proposed to achieve continual learning while preserving data security.

## Background & Motivation

### Background

**Background**: Background**: In security-sensitive domains (such as healthcare, finance, defense), data needs to be encrypted to prevent unauthorized access. Fully Homomorphic Encryption (FHE) allows direct computation on encrypted data, yielding correct results without decryption.

**Limitations of Prior Work**:

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional networks are incompatible with FHE**: Non-linear operations in ViTs and CNNs (such as ReLU and Softmax) are incomputable in the encrypted domain. Processing encrypted data requires prior decryption, exposing sensitive information and posing security risks.

### Key Challenge

**Key Challenge**: Polynomial networks are compatible but prone to forgetting**: Multilinear Operator Networks only use multilinear operations and return identical results for both raw and encrypted data, making them the current SOTA FHE-compatible architecture. However, such networks suffer from severe catastrophic forgetting in incremental learning.

### Proposed Solution

**Proposed Solution**: Existing incremental learning methods are incompatible**: Methods like EWC and PackNet rely on non-linear operations and cannot be used under the FHE framework.

**Key Challenge**: Security requirements mandate the use of multilinear networks, but multilinear networks lack effective incremental learning mechanisms. Existing incremental learning methods are also incompatible with the multilinear constraints of FHE.

**Goal**: To design an incremental learning method that is both compatible with fully homomorphic encryption and capable of effectively mitigating catastrophic forgetting.

**Key Insight**: Adapting two technologies, LoRA and GPM, to multilinear operator networks, enabling them to have incremental learning capabilities while maintaining FHE compatibility.

**Core Idea**: Efficiently adapting to new tasks through low-rank matrix decomposition while preventing forgetting by projecting gradients onto the orthogonal complement of the subspaces of past tasks, with the entire process maintaining compatibility with FHE.

## Method

### Overall Architecture

Based on the Multilinear Operator Network, a low-rank adaptation module is introduced for each new task. When training a new task, the gradient projection memory mechanism ensures that gradient updates do not destroy the knowledge acquired from past tasks. All operations retain their multilinear properties, ensuring FHE compatibility.

### Key Designs

1. **Multilinear Operator Network Architecture**:
    - Function: Provides a neural network foundation compatible with Leveled FHE
    - Mechanism: All operations in the network are multilinear (including convolutions, additions, etc.) and contain no non-linear activation functions. The same computation is performed on both raw data $x$ and encrypted data $\text{Enc}(x)$ to obtain consistent results
    - Design Motivation: Leveled FHE only supports addition and multiplication operations, which multilinear networks naturally satisfy

2. **Low-Rank Adaptation Module (LoRA)**:
    - Function: Provides a parameter-efficient adaptation mechanism for new tasks
    - Mechanism: A low-rank decomposition $\Delta W = A \cdot B$ is added alongside the weight matrix of the multilinear operator, where $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$, and $r \ll d$. The new task only requires training the low-rank matrices, significantly reducing the number of trainable parameters
    - Design Motivation: Full parameter fine-tuning leads to severe forgetting. Low-rank adaptation restricts the degrees of freedom for parameter updates, naturally providing anti-forgetting effects while maintaining multilinear computation characteristics

3. **Gradient Projection Memory (GPM)**:
    - Function: Protects past task knowledge when training new tasks
    - Mechanism: The important feature subspaces of each past task (obtained via SVD of activation values) are recorded. During new task training, gradients are projected onto the orthogonal complement of these subspaces, ensuring that updates do not affect critical parameter directions of past tasks
    - Design Motivation: Simply using low-rank adaptation may still destroy past knowledge in certain directions; GPM provides stronger protection

### Loss & Training

- **Classification Loss**: Standard multi-class classification loss (compatible with multilinear operations)
- **Gradient Projection**: After backpropagation, gradients are projected onto the orthogonal complement of past task subspaces before updating parameters
- **Incremental Training**: When a new task arrives, older low-rank modules are frozen, and a new module is added and trained under the constraint of GPM

## Key Experimental Results

### Main Results

The paper evaluates the method on standard incremental learning benchmarks (pp. 24341-24350, 10 pages in total):
- Compared to baselines without incremental learning strategies, accuracy in multi-task scenarios is significantly improved.
- While preserving FHE compatibility, the performance approaches or even surpasses some incremental learning methods that require non-linear operations.
- The effectiveness of the method is validated under incremental learning settings on datasets such as CIFAR-100.

### Ablation Study

- **LoRA vs. Full Parameter Fine-Tuning**: Low-rank adaptation is significantly superior to full parameter fine-tuning in preventing forgetting.
- **With vs. Without GPM**: GPM is crucial for maintaining the accuracy of past tasks.
- **Selection of Rank $r$**: An optimal rank value exists; a value too large approaches full parameter fine-tuning, while a value too small lacks sufficient representation capability.

### Key Findings

- Multilinear operator networks are more sensitive to catastrophic forgetting than traditional networks (due to the lack of implicit regularization from non-linear layers).
- The combination of LoRA and GPM is particularly effective for multilinear networks.
- Inference results on encrypted data are entirely consistent with those on plaintext data, validating FHE compatibility.

## Highlights & Insights

1. **Novel Problem Formulation**: This work represents the first systematic study of incremental learning in encrypted scenarios, filling the gap at the intersection of security and plasticity.
2. **Ingenious Technical Adaptation**: LoRA (originating from the LLM domain) is successfully adapted to the unique architecture of multilinear operator networks.
3. **Guaranteed Security**: The entire training and inference process does not require data decryption, achieving true security preservation.
4. **High Practical Value**: The method has direct application value for security-sensitive scenarios such as medical image analysis and financial risk control.

## Limitations & Future Work

1. **Limited Representation Capability of Multilinear Networks**: Lacking non-linear activation functions, the performance on complex tasks may be insufficient.
2. **High FHE Computational Overhead**: The computational overhead of FHE itself remains a bottleneck for practical deployment.
3. **Scalability with the Number of Tasks**: As the number of tasks increases, low-rank modules and GPM subspace information continuously accumulate.
4. **Limited to Classification Tasks**: Adapting the method to dense prediction tasks such as detection and segmentation remains to be explored.

## Related Work & Insights

- **Multilinear Operator Networks**: The foundational architecture compatible with FHE.
- **LoRA**: A parameter-efficient fine-tuning method originating from the LLM domain.
- **GPM (Gradient Projection Memory)**: A continual learning method based on protecting gradient directions.
- **Leveled FHE**: An encryption scheme allowing finite-depth operations on encrypted data.
- **Insights for Future Work**: The intersection of secure computation and continual learning warrants further attention.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[CVPR 2025\] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](../../NeurIPS2025/llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
