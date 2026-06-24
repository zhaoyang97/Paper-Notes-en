---
title: >-
  [Paper Note] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation
description: >-
  [CVPR 2025][Model Compression][Mamba Architecture] This paper reveals that the projector, rather than the SSM, is the critical component for transfer learning in the Mamba architecture. Based on this finding, the authors propose ProDiaL, a method that indirectly fine-tunes frozen projector weights through a diagonal-centric linear transformation matrix. By training less than 1% of the parameters, ProDiaL outperforms LoRA/DoRA on downstream tasks across both vision and languag…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Mamba Architecture"
  - "Parameter-Efficient Fine-Tuning"
  - "Projector"
  - "Diagonal Transformation"
  - "State Space Models"
date: 2026-05-08
content_hash: c7ae169eb4f118be
---

# Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation

**Conference**: CVPR 2025  
**arXiv**: [2411.15224](https://arxiv.org/abs/2411.15224)  
**Code**: None  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning  
**Keywords**: Mamba Architecture, Parameter-Efficient Fine-Tuning, Projector, Diagonal Transformation, State Space Models

## TL;DR

This paper reveals that the projector, rather than the SSM, is the critical component for transfer learning in the Mamba architecture. Based on this finding, the authors propose ProDiaL, a method that indirectly fine-tunes frozen projector weights through a diagonal-centric linear transformation matrix. By training less than 1% of the parameters, ProDiaL outperforms LoRA/DoRA on downstream tasks across both vision and language Mamba models.

## Background & Motivation

**Background**: With its selective SSM mechanism and hardware-aware operations, the Mamba architecture achieves context-aware reasoning while maintaining linear computational complexity, making it widely adopted in LLMs and vision models. As Mamba models scale up, Parameter-Efficient Fine-Tuning (PEFT) becomes increasingly important.

**Limitations of Prior Work**: Existing PEFT methods (such as LoRA, Adapter, and Prompt Tuning) are almost entirely designed for transformer attention modules. When directly applying these methods to Mamba, the SSM is naturally treated as the core counterpart to attention and targeted for tuning. The only prior work exploring Mamba PEFT (Halloran et al.) applies LoRA to SSM parameters $W_x$ (which controls $B$, $C$, $\Delta$), projectors, and embeddings, but lacks a detailed analysis of each component's actual contribution.

**Key Challenge**: Although the SSM is the theoretical core of Mamba, its role in transfer learning may be overestimated. The projector, which accounts for approximately 65% of the model parameters and directly controls input/output information mapping, might be the true key to downstream task adaptation. However, fully fine-tuning the projector introduces too many parameters, necessitating a targeted PEFT solution.

**Goal**: (1) Systematically analyze the contribution of each Mamba component to downstream tasks; (2) Design a highly efficient fine-tuning method specifically tailored for the projector.

**Key Insight**: By analyzing the relationship between pre-trained and fine-tuned projector weights from a linear transformation perspective, the authors discover that the transformation matrix $T = W^{-1}W'$ approximates an identity matrix—diagonal elements are close to 1, off-diagonal elements are close to 0, and training gradients are heavily concentrated on the diagonal.

**Core Idea**: Freeze the pre-trained projector $W$ and indirectly update the weights by training a transformation dominated by a block-diagonal matrix $D_b$ and assisted by a low-rank matrix $\epsilon$: $W' = sWD_b + \epsilon$, requiring less than 1% of the parameters.

## Method

### Overall Architecture

ProDiaL targets the Input-Projector and/or Output-Projector within the Mamba block for parameter-efficient fine-tuning. The pre-trained weights $W$ are frozen, and a learnable block-diagonal matrix $D_b$, a scaling parameter $s$, and a low-rank matrix $\epsilon = B_\epsilon A_\epsilon$ are attached. During the forward pass, $W' = sWD_b + \epsilon$ is computed. After training, the transformed weights are directly folded into the original weights, introducing zero extra inference overhead.

### Key Designs

1. **Projector Dominance Discovery**:

    - **Function**: Identify the core component in the Mamba architecture that should be targeted by PEFT.
    - **Mechanism**: Systematically perform selective fine-tuning experiments on various components of the Mamba block (SSMs' $W_x$, Input-Projectors, Output-Projectors, and Embeddings). On Vision Mamba, fine-tuning only the Out-Proj (1.789M parameters) achieves a 72.77% accuracy, while tuning only the SSM (1.441M) yields only 70.04%. On Mamba LLM, tuning only the Out-Proj (28.3M) reaches 40.89%, outperforming SSM tuning (5.38M) at 37.52%. Crucially, Both-Proj tuning via LoRA (2.36M) achieves 38.33%, which still significantly outperforms SSM tuning (37.52%), ruling out the influence of parameter size differences.
    - **Design Motivation**: Overturn the intuitive assumption that "SSM is the core of Mamba, so PEFT must focus on SSM". The projector controls the input/output transformations of information, which is more directly relevant to feature adaptation for downstream tasks.

2. **Diagonal-Centric Linear Transformation**:

    - **Function**: Indirectly update projector weights using minimal parameters.
    - **Mechanism**: Model the relationship between the fine-tuned and pre-trained weights as $W' = WT$, and determine the transformation matrix $T_{det} = W^{-1}W'$ via pseudo-inverse. Visualizations reveal that $T_{det}$ resembles an identity matrix, where diagonal values are near 1 and off-diagonal values are near 0. Analyzing the $L_1$ norm during training confirms that gradients are concentrated on the diagonal. Thus, $T$ is decomposed into a diagonal matrix $D$ and an off-diagonal matrix $b$: $W' = WD + Wb = WD + \epsilon$. A block-diagonal matrix is used to replace the pure diagonal matrix (to increase expressivity and allow tiny rotations); LoRA low-rank decomposition is applied to the off-diagonal part: $\epsilon = B_\epsilon A_\epsilon$.
    - **Design Motivation**: While LoRA models weight updates via addition ($W' = W + \Delta W$), ProDiaL operates from a multiplicative perspective ($W' = WT$), which aligns better with the empirical behavior of projector fine-tuning. The diagonal-centric prior significantly constrains the search space of learnable parameters.

3. **Full Formulation of ProDiaL**:

    - **Function**: Formulate the analytical findings into a trainable PEFT module.
    - **Mechanism**: $W' = sWD_b + \epsilon$, where $D_b = [\mathbb{I} - \text{relu}(\mathbb{I} * D_a)] + (1 - \mathbb{I}) * D_a$, $D_a = \text{diag}(x_1, \dots, x_n)$ ($x_i \in \mathbb{R}^{(d_{in}/r_b) \times (d_{in}/r_b)}$ are small matrices), and $\epsilon = B_\epsilon A_\epsilon$. The block size $r_b$ and the rank $r_\epsilon$ flexibly control the parameter count. $s \in \mathbb{R}^{d_{out}}$ is a learnable scaling parameter per output dimension. $D_b$ stabilizes the close-to-identity initialization by learning deviations from the identity matrix.
    - **Design Motivation**: The block-diagonal structure introduces extra degrees of freedom (enabling local rotation/mixing) compared to a pure diagonal matrix, while maintaining a parameter size far smaller than a full matrix. The relu constraint ensures non-negative diagonal elements, and $\mathbb{I} - \text{relu}(\mathbb{I} * D_a)$ allows the initial values to decay starting from 1.

### Loss & Training

Standard data cross-entropy loss is used for downstream classification tasks. Vision Mamba models pre-trained on ImageNet are transferred to datasets like StanfordCars, Caltech, and Flowers. Mamba LLM (130M) pre-trained on PILE is transferred to HellaSwag, Winogrande, ARC-E, and ARC-C reasoning tasks. After training, $D_b$ and $\epsilon$ are merged back into $W$, adding no deployment overhead.

## Key Experimental Results

### Main Results

Mamba LLM (130M) downstream reasoning task accuracy (%):

| Method | Params | HellaSwag | Winogrande | ARC-E | ARC-C | Avg. |
|------|--------|-----------|------------|-------|-------|------|
| Full-FT | 130M | 38.23 | 53.12 | 53.54 | 28.84 | 43.43 |
| Strong (SSM+Proj+Emb) | 3.80M | 38.66 | 53.04 | 54.17 | 28.67 | 43.64 |
| Both-Proj LoRA | 2.36M | 38.33 | 53.12 | 53.87 | 29.52 | 43.71 |
| **Both-Proj ProDiaL** | **2.42M** | **38.92** | **53.28** | **55.18** | **28.84** | **44.06** |

Vision Mamba (Vim-tiny) downstream classification accuracy (%):

| Method | Params | StanfordCars | Caltech | Flowers | Avg. |
|------|--------|-------------|---------|---------|------|
| Full-FT | 7.00M | 90.06 | 92.86 | 92.05 | 91.66 |
| Both-Proj LoRA | 0.63M | 85.06 | 96.01 | 87.32 | 89.46 |
| Both-Proj DoRA | 0.69M | 85.18 | 96.09 | 86.60 | 89.29 |
| **Both-Proj ProDiaL** | **0.67M** | **85.38** | **96.24** | **88.00** | **89.87** |

### Ablation Study

| Configuration | Vision Avg. | LLM Avg. | Description |
|------|-----------|---------|------|
| SSM Fine-tuning | 70.04 | 37.52 | SSM is not the key to PEFT |
| Both-Proj Fine-tuning | 92.22 (5.35M) | 44.16 (84.9M) | Projectors are the core components |
| In-Proj only | 91.96 | 44.44 | A single projector is also effective |
| Out-Proj only | 91.98 | 44.51 | Out-Proj is slightly superior |
| ProDiaL w/o $\epsilon$ | Decrease | Decrease | Off-diagonal terms contribute |
| ProDiaL w/o $D_b$ | Decrease | Decrease | Diagonal terms are the core |

### Key Findings

- Projectors are the core of Mamba PEFT: Tuning only the projectors via LoRA (2.36M) outperforms the "Strong" baseline (3.80M) that tunes SSM + Proj + Emb.
- Embedding parameters can be detrimental in transfer learning (performance drops when included), which differs from their role in Transformers.
- The diagonal prior is rigorously validated: Visualizations of the transformation matrix $T_{det}$ and gradient $L_1$ analyses consistently show that diagonal elements dominate.
- ProDiaL scales consistently across different model sizes (Mamba-130M/370M/1.4B, Vim-tiny/small).
- The trained parameters can be seamlessly folded back into the weight matrices, without increasing inference latency.

## Highlights & Insights

- The discovery that "the projector, rather than the SSM, is the core of Mamba PEFT" is both surprising and highly convincing—rigorously validated through controlled experiments that isolate the parameter-size factor (LoRA 2.36M vs SSM 5.38M, where Proj still wins).
- Analyzing weight modification from a multiplicative perspective (linear transformation $W'=WT$) instead of the traditional additive perspective ($W'=W+\Delta W$) offers a novel paradigm for PEFT design. The empirical finding that $T$ approximates an identity matrix could potentially generalize to other architectures.
- The parameter-merging property of ProDiaL maintains the same deployment advantages as LoRA, while delivering better inductive bias through the diagonal prior.

## Limitations & Future Work

- Experiments are only conducted on Mamba-1/2 architectures, leaving other hybrid variants (e.g., Jamba, Zamba) unexplored.
- The scales of Vision Mamba (Vim-tiny) and Mamba LLM (130M) are relatively small; verification on billion-parameter models is still required.
- The block size $r_b$ and low-rank constraint $r_\epsilon$ require tuning for different downstream tasks.
- Future directions: (1) Generalize ProDiaL to hybrid Mamba-Transformer architectures; (2) Explore theoretical explanations for why projectors dominate transfer learning; (3) Verify its efficacy in generative tasks such as diffusion models.

## Related Work & Insights

- **vs LoRA**: LoRA models weight changes additively ($W + BA$), whereas ProDiaL uses a multiplicative approach ($WD_b + \epsilon$). The diagonal-centric prior of ProDiaL fits the adaptation behavior of Mamba projectors better, consistently outperforming LoRA at the same parameter budget.
- **vs DoRA**: DoRA decomposes weight updates into magnitude and direction. ProDiaL decomposes the linear transformation into diagonal (scaling) and off-diagonal (rotation/mixing) components, which is a more natural fit for Mamba projectors.
- **vs Strong (Halloran et al.)**: While Strong jointly fine-tunes SSM $W_x$, projectors, and embeddings, this paper demonstrates that only fine-tuning projectors yields superior results, and excluding embeddings further improves accuracy.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to systematically analyze the PEFT contributions of different Mamba components, discovering projector dominance and designing a diagonal-prior-driven method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across both vision and language modalities with multiple model scales and ablation dimensions, though the evaluated models are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical flow from the discovery to the proposed method, with intuitive and convincing visual analyses.
- **Value**: ⭐⭐⭐⭐ — Provides practical guidance and methodology for Mamba PEFT, with the discovery of projector dominance holding independent scientific value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](../../ICML2025/model_compression/moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)

</div>

<!-- RELATED:END -->
