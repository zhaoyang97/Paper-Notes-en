---
title: >-
  [Paper Note] DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models
description: >-
  [CVPR 2025][Model Compression][Mixture-of-Experts] This paper proposes the DeRS (Decompose-Replace-Synthesis) paradigm. By leveraging the extremely high similarity (cosine similarity >0.999) among experts in upcycled MoEs, DeRS decomposes $N$ experts into 1 shared base weight + $N$ lightweight delta weights. By compressing the delta weights via sparsification, quantization, or low-rank representations, it reduces MoE layer parameters by 65% with zero performance degradation…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Mixture-of-Experts"
  - "Parameter-Efficient"
  - "Upcycling"
  - "Sparse Matrix"
date: 2026-05-08
content_hash: 86fd6eecdf387e26
---

# DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models

**Conference**: CVPR 2025  
**arXiv**: [2503.01359](https://arxiv.org/abs/2503.01359)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Mixture-of-Experts, Model Compression, Parameter-Efficient, Upcycling, Sparse Matrix

## TL;DR
This paper proposes the DeRS (Decompose-Replace-Synthesis) paradigm. By leveraging the extremely high similarity (cosine similarity >0.999) among experts in upcycled MoEs, DeRS decomposes $N$ experts into 1 shared base weight + $N$ lightweight delta weights. By compressing the delta weights via sparsification, quantization, or low-rank representations, it reduces MoE layer parameters by 65% with zero performance degradation, or reduces additional training parameters by 2270-fold.

## Background & Motivation

**Background**: Upcycled MoEs copy the FNN layer of a pre-trained dense model $N$ times to initialize $N$ experts, which then differentiate through subsequent fine-tuning. This approach is more resource-efficient than training MoEs from scratch and has been widely adopted in NLP, vision, and multimodal tasks.

**Limitations of Prior Work**: The $N$ experts introduce a massive number of parameters (e.g., 3.4B out of the 5B total parameters in MoE-LLaVA-Phi are occupied by experts). However, since they are initialized from the same FFN, their trained expert weights exhibit an extremely high cosine similarity (>0.999), indicating significant redundancy.

**Key Challenge**: Although the experts in upcycled MoEs are functionally differentiated, their weight differences in the weight space are extremely minuscule (the delta weights are negligible compared to the base weight). Currently, no existing methods exploit this characteristic to compress parameters.

**Goal**: How to leverage the high similarity among upcycled MoE experts to achieve extreme parameter compression in both training and inference stages.

**Key Insight**: Decompose each expert $W_i$ into a shared base $W_{base}$ plus an expert-specific micro-adjustment $\Delta_i$, and then represent $\Delta_i$ in a lightweight form.

**Core Idea**: The variation among upcycled MoE experts lies only in the minuscule delta weights; compressing these deltas using sparsification, quantization, or low-rank representations can substantially reduce parameters with almost no performance loss.

## Method

### Overall Architecture
The DeRS paradigm comprises three steps: Decompose (splitting $N$ expert weights into $W_{base} + \Delta_i$) $\rightarrow$ Replace (substituting the original $\Delta_i$ with a lightweight representation $\mathcal{F}(\Delta_i)$) $\rightarrow$ Synthesis (online synthesizing expert weights as $W_{base} + \mathcal{F}(\Delta_i)$ during inference). Based on this framework, two applications are proposed: DeRS Compression for the inference stage and DeRS Upcycling for the training stage.

### Key Designs

1. **DeRS Compression (Inference-stage Compression)**:

    - **Function**: Compressing pre-trained vanilla upcycled MoE models.
    - **Mechanism**: Two methods: (a) Sparsification: randomly dropping a ratio $p$ of elements in $\Delta_i$ (e.g., $p=0.9$) and storing them in compact vectors, reducing the MoE layer parameters from $N \cdot d \cdot d_h$ to $(1 + N(1-p)) \cdot d \cdot d_h$; (b) Quantization: quantizing $\Delta_i$ from 16-bit to $k$ bit (e.g., 2-bit), reducing the storage cost from $N \times 16$ to $16 + N \times k$. Experiments show that dropping 90% of elements or quantizing to 2-bit has almost no impact on performance.
    - **Design Motivation**: Delta weights are extremely small and redundant; the fact that random sparsification works indicates their low information density.

2. **DeRS Upcycling - Sparse Matrix (Training Stage)**:

    - **Function**: Building MoE experts in a parameter-efficient manner during training.
    - **Mechanism**: Instead of duplicating the FFN $N$ times, this method maintains one shared $W_{shared}$ (initialized from the original FFN) $+ N$ sparse increments $\mathcal{F}(\Delta_i)$. Each increment is represented by an index vector $I_i$ and a value vector $V_i$, where $I_i$ is randomly generated and fixed, and $V_i$ is trainable and zero-initialized. At a sparsity rate of $p=0.9999$, the additional parameters are only 0.26M (vs. 1.24B in the vanilla model, a 4770x reduction).
    - **Design Motivation**: Since the delta is inherently small after training, it is better to directly restrict its parameter size during training, encouraging the model to learn differentiation within a constrained space.

3. **DeRS Upcycling - Low-Rank Matrix (Training Stage)**:

    - **Function**: An alternative parameter-efficient representation for expert differentiation.
    - **Mechanism**: Similar to LoRA, using $\Delta_i = A_i \cdot B_i$ (where $A_i \in \mathbb{R}^{d \times r}$ and $B_i \in \mathbb{R}^{r \times d_h}$). $A_i$ is randomly initialized, and $B_i$ is zero-initialized. Parameter size is reduced from $N \cdot d \cdot d_h$ to $d \cdot d_h + N \cdot r \cdot (d + d_h)$.
    - **Design Motivation**: Low-rank decomposition is another mature parameter-efficient approach, complementing the sparse matrix method.

### Loss & Training
The training strategy aligns with the original MoE-LLaVA, fine-tuning on the LLaVA-mix-665k dataset, where every other FFN layer is replaced with an MoE layer (4 experts, top-2 activation). In DeRS Upcycling, $W_{shared}$ and $V_i$ (or $A_i, B_i$) are trained jointly.

## Key Experimental Results

### Main Results

| Model | Method | Extra Params ↓ | Overall Performance |
|------|------|----------|---------|
| MoE-LLaVA-StableLM | Vanilla | 1.24B | 57.4 |
| MoE-LLaVA-StableLM | DeRS-SM | 0.26M (↓4770x) | 57.7 (+0.3) |
| MoE-LLaVA-StableLM | DeRS-LM | 1.20M (↓1033x) | 57.5 (+0.1) |
| MoE-LLaVA-Phi | Vanilla | 2.52B | 60.8 |
| MoE-LLaVA-Phi | DeRS-SM | 1.11M (↓2270x) | 61.1 (+0.3) |
| MoE-LLaVA-Phi | DeRS-LM | 2.42M (↓1041x) | 61.0 (+0.2) |

### Ablation Study

| DeRS Compression Config | MoE Layer Parameter Changes | Performance Impact |
|---------------------|-------------|---------|
| Sparsification drop=0.9 | Equivalent to 4 -> 1.4 experts | Lossless |
| Sparsification drop=0.99 | Equivalent to 4 -> 1.04 experts | Negligible drop |
| Quantization 2-bit delta | Storage reduced by $\frac{16+4 \times 2}{4 \times 16}$=37.5% | Lossless |
| Quantization 1-bit delta | Extreme compression | Slight drop |

### Key Findings
- Performance does not degrade but slightly improves after dropping 90% of elements in the delta weights, demonstrating the massive redundancy of deltas.
- DeRS-SM (Sparse Matrix) remains functional at an extremely high sparsity rate (99.99%), indicating that expert differentiation requires very few parameters.
- DeRS Upcycling not only compresses parameters but also marginally improves performance (+0.3), potentially due to regularization effects.
- Consistency is observed across three tasks (general multimodal, medical multimodal, and code generation) and six MoE architectures.
- It is equally effective on Med-MoE; the redundancy patterns of experts in medical tasks align with those in general tasks.

## Highlights & Insights
- **Exploitation of structural property unique to upcycling**: Distinct from general MoE compression methods (e.g., expert pruning, expert merging), DeRS precisely leverages the shared initialization attribute of experts in upcycled MoE, rendering it an elegant, problem-specific solution.
- **Functioning with a 2270x parameter reduction**: This striking result reveals that the information variance required for expert differentiation in upcycled MoEs is far less than commonly assumed.
- **Applicability across training and inference stages**: DeRS Compression (post-training compression) and DeRS Upcycling (parameter-efficient training) cover the entire lifecycle of MoE models.

## Limitations & Future Work
- Applicable only to upcycled MoEs; it is not suitable for MoEs trained from scratch (such as Switch Transformer) as they lack shared expert initialization.
- The indices $I_i$ of the sparse matrix are randomly fixed; exploring learnable index positions might further enhance performance.
- Online synthesis $W_{base} + \mathcal{F}(\Delta_i)$ in DeRS Compression introduces extra computation, and its impact on inference latency is not analyzed.
- Experiments are mainly conducted on 3B-scale models; effectiveness on larger models (e.g., 70B-grade MoEs) remains unverified.

## Related Work & Insights
- **vs. LoRA**: Structurally, DeRS-LM resembles LoRA. However, the core innovation of DeRS lies in first extracting the shared base and then performing low-rank adaptation exclusively on the delta, rather than direct low-rank adaptation on the entire weight.
- **vs. Expert Pruning (e.g., MC-SMoE)**: These methods compress models by merging or pruning whole experts, which can discard expert-specific knowledge. DeRS retains all experts but compresses their differential representations.
- **vs. General MoE**: Experiments show that the cosine similarity of experts in upcycled MoEs exceeds 0.999, which is much higher than that in MoEs trained from scratch. DeRS specifically exploits this gap.

## Rating
- Novelty: ⭐⭐⭐⭐ Sharp observation (cosine similarity >0.999), elegant and natural paradigm design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 tasks and 6 architectures, covering both training and compression scenarios, with highly detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, with an intuitive "Decompose-Replace-Synthesis" taxonomy.
- Value: ⭐⭐⭐⭐ Direct practical value for upcycled MoE deployment; the 2270x compression factor is highly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[ICLR 2026\] Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees](../../ICLR2026/model_compression/efficient_quantization_of_mixture-of-experts_with_theoretical_generalization_gua.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](../../NeurIPS2025/model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[ICLR 2026\] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](../../ICLR2026/model_compression/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)

</div>

<!-- RELATED:END -->
