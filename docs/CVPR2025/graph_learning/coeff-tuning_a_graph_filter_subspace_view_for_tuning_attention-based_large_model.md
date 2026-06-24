---
title: >-
  [Paper Note] Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models
description: >-
  [CVPR 2025][Graph Learning][PEFT] This paper reinterprets multi-head attention as a graph convolutional filter subspace, and linearly combines pre-trained attention maps by learning an extremely small set of subspace combination coefficients ($H \times H$ matrices). This breaks the convex hull constraint caused by the softmax function to expand the feature space, improving the performance of various PEFT methods in a plug-and-play manner at near-zero parameter cost.
tags:
  - "CVPR 2025"
  - "Graph Learning"
  - "PEFT"
  - "Graph Convolutional Filters"
  - "Attention Subspace"
  - "Subspace Coefficients"
  - "Plug-and-play"
date: 2026-05-08
content_hash: 0b30e772e23a78f4
---

# Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models

**Conference**: CVPR 2025  
**arXiv**: [2503.18337](https://arxiv.org/abs/2503.18337)  
**Code**: [https://github.com/ZichenMiao/Coeff_Tuning](https://github.com/ZichenMiao/Coeff_Tuning)  
**Area**: Parameter-Efficient Fine-Tuning / Graph Learning  
**Keywords**: PEFT, Graph Convolutional Filters, Attention Subspace, Subspace Coefficients, Plug-and-play

## TL;DR
This paper reinterprets multi-head attention as a graph convolutional filter subspace, and linearly combines pre-trained attention maps by learning an extremely small set of subspace combination coefficients ($H \times H$ matrices). This breaks the convex hull constraint caused by the softmax function to expand the feature space, improving the performance of various PEFT methods in a plug-and-play manner at near-zero parameter cost.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) is a core technology for adapting large pre-trained models to downstream tasks. Mainstream methods such as LoRA, Adapter, and VPT primarily focus on tensor decomposition, efficiently adjusting the weights of linear transformation layers using low-rank matrices or learnable prompts.

**Limitations of Prior Work**: Existing PEFT methods focus on optimizing the weight matrices of linear transformations, lacking an overall perspective on the core operation of Transformers—the multi-head attention layer. Softmax attention limits the attention score of each row to the range (0,1) with a row sum of 1, meaning that each output token embedding is restricted to the convex hull of the value embeddings, which constrains the expressiveness of the attention layer.

**Key Challenge**: Fine-tuning requires enhancing the expressiveness of the attention layer to adapt to downstream tasks, but directly updating the parameters for generating attention maps ($W_q, W_k$) incurs a heavy parameter load, and softmax normalization consistently restricts outputs to the convex hull.

**Goal**: To break through the convex hull constraint of softmax attention and enhance the expressiveness of multi-head attention with minimal parameters.

**Key Insight**: Multi-head attention is viewed as a graph convolution operation, where the attention map of each head serves as a graph convolutional filter, and $H$ attention heads form a filter subspace. By learning unconstrained linear combination coefficients, new filters with output values outside the (0,1) range can be constructed using existing filters.

**Core Idea**: Linearly combine $H$ attention maps using $H^2$ learnable coefficients to break the convex hull constraint of softmax, thereby expanding the attention feature space.

## Method

### Overall Architecture
For each attention layer, the original Q/K/V projections are kept unchanged. After calculating the $H$ attention maps $\{F^1, ..., F^H\}$, they are linearly combined using a learnable $H \times H$ coefficient matrix $\alpha$ to obtain $H$ new attention maps $\hat{F}^h = \sum_{i=1}^H \alpha[h,i] F^i$. Since $\alpha$ is unconstrained, the elements of the combined attention maps can exceed the (0,1) range or even be negative, thus breaking the convex hull constraint.

### Key Designs

1. **Subspace Coefficient ($\alpha$)**:

    - **Function**: Construct a more expressive filter subspace by linearly combining existing attention maps.
    - **Mechanism**: A learnable $H \times H$ matrix is added to each attention layer, where $\hat{F}^h = \sum_i \alpha[h,i] F^i$. Since $\alpha$ is unconstrained, the newly combined attention maps can contain negative values, allowing output tokens to lie outside the convex hull of the value embeddings. For ViT-B/16 (12 layers, 12 heads), only $12 \times 12 \times 12 = 1728$ parameters are needed in total.
    - **Design Motivation**: Toy experiments strictly verify that when only tuning $W_q$ and $W_k$, the output is restricted within the input range, whereas introducing $\alpha$ successfully enables mapping to positions outside the target range.

2. **Residual Parameterization**:

    - **Function**: Stabilize the fine-tuning process and preserve pre-trained knowledge.
    - **Mechanism**: Parameterize the coefficient matrix as $\alpha' = \alpha + I$, where $\alpha$ is initialized to zero. This makes the initial state equivalent to the original pre-trained model, and during fine-tuning, $\alpha$ learns the residual correction relative to the identity mapping. Ablation studies show that random initialization of $\alpha$ yields only 44.28%, identity initialization (non-residual) yields 61.57%, while residual parameterization achieves 73.49%.
    - **Design Motivation**: Prevent destructive changes to pre-trained features in the early stages of fine-tuning.

3. **Coefficient Dropout Regularization**:

    - **Function**: Prevent overfitting and enhance generalization.
    - **Mechanism**: Directly apply dropout to $\alpha$ during training, randomly setting elements to zero with a probability of $p$. The optimal dropout rate is found to be $p=0.2$, which achieves a score of 77.7 on VL-BART (compared to 75.1 with $p=0$ and 72.3 with $p=0.4$).
    - **Design Motivation**: Although $\alpha$ contains few parameters, it has a significant impact on attention maps and can easily cause overfitting in few-shot scenarios.

### Loss & Training
The training configuration is kept consistent with the base PEFT methods. As a plug-and-play module, Coeff-Tuning does not alter the original training pipeline when combined with LoRA, DoRA, SSF, etc., only introducing the coefficient combination step during attention computation.

## Key Experimental Results

### Main Results

**VTAB-1k Few-Shot Classification (ViT-B/16)**:

| Method | Extra Params (M) | Average Accuracy |
|------|-------------|-----------|
| Linear Probing | 0 | 52.94% |
| Adapter | 0.234 | 55.82% |
| VPT | 0.06 | 64.85% |
| LoRA | 0.305 | 72.91% |
| **Coeff α only** | **0.002** | **69.78%** |
| FACT-TT≤16 | 0.040 | 73.04% |
| **FACT + Coeff-Tuning** | **0.042** | **73.64%** |
| SNELL-8 | 0.268 | 74.17% |
| **SNELL-8 + Coeff-Tuning** | **0.270** | **74.70%** |

**VL-BART Multimodal Understanding**:

| Method | Params % | Average Score |
|------|--------|--------|
| Full Model | 100% | 77.3 |
| LoRA | 5.94% | 76.5 |
| **Coeff + LoRA** | **5.94%** | **77.1** |
| DoRA | 5.96% | 77.4 |
| **Coeff + DoRA** | **5.96%** | **77.7** |

### Ablation Study

| Configuration | Average Accuracy |
|------|-----------|
| Random initialization of $\alpha$ | 44.28% |
| Identity matrix initialization of $\alpha$ | 61.57% |
| $I + \alpha$, zero initialization (Residual parameterization) | **73.49%** |

| Dropout rate $p$ | VL-BART Avg |
|-----------------|-------------|
| 0.0 | 75.1 |
| 0.1 | 77.3 |
| 0.2 | **77.7** |
| 0.4 | 72.3 |

### Key Findings
- With only 1728 parameters ($\alpha$ only), performance improves from 52.94% to 69.78%, approaching LoRA's level (72.91%).
- Combining it with any PEFT method yields consistent performance gains (+0.5% to 1.0%) with negligible extra parameters.
- Residual parameterization is crucial: random initialization causes performance collapse (44.28%), whereas residual parameterization ensures smooth progression from the pre-trained state.
- It is also effective in SDXL concept customization tasks, improving concept alignment from 0.73/0.74 to 0.75.

## Highlights & Insights
- **Reinterpreting Attention from a Graph Convolutional Perspective**: Understanding multi-head attention as a graph convolutional filter subspace is a highly insightful theoretical contribution. This perspective uncovers the convex hull constraint of softmax attention, providing a theoretical foundation to transcend it.
- **Extreme Parameter Efficiency**: Marked improvements are achieved with only 1728 parameters, making this one of the most parameter-efficient PEFT methods. The core lies in manipulating the combination of attention maps rather than the weight matrices themselves.
- **Plug-and-Play Versatility**: It can be seamlessly integrated with any PEFT methods like LoRA, DoRA, and SSF, offering an orthogonal dimension of optimization.

## Limitations & Future Work
- The core assumption relies on the linear combination of attention heads, which may lead to limited combinatorial diversity when the number of heads is small (e.g., $H=4$).
- The theoretical analysis is based on a single attention layer, lacking theoretical guarantees for the effects of multi-layer stacking.
- Evaluations are only conducted on ViT, VL-BART, and SDXL, without testing on larger-scale LLMs (e.g., LLaMA).
- The dropout rate requires tuning, as the optimal value varies across target tasks.

## Related Work & Insights
- **vs. LoRA**: LoRA modifies the low-rank delta of the weight matrix, while Coeff-Tuning alters the combination coefficients of attention maps. The two methods are orthogonal and complementary.
- **vs. OFT**: OFT maintains the hyperspherical energy of neuron relations, whereas Coeff-Tuning expands the attention feature space. Their underlying ideas are entirely different.
- **vs. DeepViT**: DeepViT regenerates diverse attention maps to prevent training collapse, while Coeff-Tuning extends expressiveness during fine-tuning via new combinations of pre-trained attention maps.
- This work serves as a valuable reference for edge-device fine-tuning scenarios requiring extreme parameter efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The graph convolutional filter subspace perspective is highly novel with strong theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers vision, multimodal, and generative tasks, but lacks evaluation on large LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivations are clear, and the toy example is intuitive.
- Value: ⭐⭐⭐⭐ The extreme parameter efficiency and plug-and-play versatility hold significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](../../ICML2026/graph_learning/message_tuning_outshines_graph_prompt_tuning_a_prismatic_space_perspective.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[ICML 2025\] Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via Contextual Stochastic Block Models](../../ICML2025/graph_learning/graph_attention_is_not_always_beneficial_a_theoretical_analysis_of_graph_attenti.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](../../ICML2026/graph_learning/gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)

</div>

<!-- RELATED:END -->
