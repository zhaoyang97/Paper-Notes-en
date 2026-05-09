---
title: >-
  [Paper Note] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling
description: >-
  [Multimodal VLM] This paper proposes MaTVLM, which replaces a portion of Transformer layers in a pretrained VLM with Mamba-2 layers and trains the resulting model via single-stage knowledge distillation, achieving 3.6× inference speedup and 27.5% memory reduction while maintaining competitive performance.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: edf6090a406cd111
---

# MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.13440](https://arxiv.org/abs/2503.13440)
- **Code**: [https://github.com/hustvl/MaTVLM](https://github.com/hustvl/MaTVLM)
- **Area**: Multimodal VLM
- **Keywords**: Mamba-Transformer hybrid architecture, knowledge distillation, efficient VLM, linear complexity, inference acceleration

## TL;DR

This paper proposes MaTVLM, which replaces a portion of Transformer layers in a pretrained VLM with Mamba-2 layers and trains the resulting model via single-stage knowledge distillation, achieving 3.6× inference speedup and 27.5% memory reduction while maintaining competitive performance.

## Background & Motivation

Current vision-language models (VLMs) are predominantly built on the Transformer architecture, whose quadratic self-attention complexity poses severe computational and memory bottlenecks when processing long sequences. Although RNN-based models such as Mamba have demonstrated potential as linear-complexity alternatives, existing Mamba-based VLMs suffer from three key issues:

**Insufficient global context capture**: Mamba's sequential processing limits its ability to capture long-range dependencies, leading to degraded performance on complex reasoning tasks.

**Convergence difficulties**: Sequential processing results in inefficient gradient propagation, making from-scratch VLM training slow to converge and computationally expensive.

**Complex training pipelines**: Existing Mamba VLMs require multi-stage training to achieve optimal performance, which is difficult to scale.

The authors observe an intrinsic mathematical connection between attention and Mamba-2—attention without softmax is essentially equivalent to a linear RNN—providing a theoretical basis for weight initialization between the two. Based on this insight, the paper proposes organically combining Mamba with Transformers to strike a balance between efficiency and performance.

## Method

### Overall Architecture

MaTVLM is built upon a pretrained VLM (TinyLLaVA) and comprises three components: a visual encoder, a connector, and a language model. The core modification targets the language model: a subset of Transformer decoder layers is replaced by Mamba-2 decoder layers (only the attention module is replaced; all other components remain unchanged), distributed uniformly at specified ratios (e.g., 12.5%, 25%, 50%).

### Mathematical Connection Between Attention and Mamba-2

The standard attention output is:

$$y_n = \sum_{t=1}^{n} \text{softmax}\left(\frac{Q_n K_t^\top}{\sqrt{d}}\right) V_t$$

Removing the softmax allows this to be reformulated as a linear RNN:

$$h_n = h_{n-1} + K_n^\top W_V x_n, \quad y_n = \frac{Q_n}{\sqrt{d}} h_n$$

Comparing this with the Mamba-2 SSM formulation $h_t = A_t h_{t-1} + B_t x_t$, $y_t = C_t^\top h_t$ yields the following correspondence:

$$x_t \leftrightarrow W_V x_t, \quad B_t \leftrightarrow W_K x_t, \quad C_t \leftrightarrow W_Q x_t$$

Consequently, the linear projection weights for $x$, $B$, and $C$ in Mamba-2 can be initialized from the $V$, $K$, and $Q$ weights of the attention layer, accelerating convergence.

### Knowledge Distillation Strategy

Single-stage distillation is employed, training only the Mamba-2 layers and the connector while freezing the Transformer layers. The loss function consists of three components:

**1. Probability distribution distillation** (KL divergence):

$$\mathcal{L}_{\text{prob}} = T^2 \cdot \text{KL}(P_t \| P_s) = T^2 \cdot \sum_i P_t(i) \log\frac{P_t(i)}{P_s(i)}$$

where $P_t$ and $P_s$ are the temperature-scaled output distributions of the teacher and student models, respectively.

**2. Layer-wise distillation** (L2 alignment):

$$\mathcal{L}_{\text{layer}} = \sum_{i=1}^{m} \|T_{l_i}(x) - S_{l_i}(x)\|_2$$

This ensures that the outputs of the student's Mamba layers are aligned with those of the corresponding Transformer layers in the teacher.

**3. Total loss**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{layer}} + \beta \cdot \mathcal{L}_{\text{prob}} + \gamma \cdot \mathcal{L}_{\text{ce}}$$

In experiments, $\alpha = \beta = 1.0$ and $\gamma = 0$ (the sequence prediction loss is omitted).

## Experiments

### Main Results

| Method | LLM | MME-P | MMB | TextVQA | GQA | MM-Vet | SQA-I | POPE | MMMU |
|------|-----|-------|-----|---------|-----|--------|-------|------|------|
| TinyLLaVA (Teacher) | Phi2-2.7B | 1466.4 | 66.1 | 60.3 | 62.1 | 37.5 | 73.0 | 87.2 | 38.4 |
| **MaTVLM (Ours)** | Hybrid | **1484.0** | 61.2 | 57.7 | 60.4 | 35.3 | 68.1 | **87.4** | **40.0** |
| Cobra (Mamba) | Mamba 2.8B | - | - | 57.9 | - | - | - | 88.2 | - |
| LLaVA-Phi | Phi2-2.7B | 1335.1 | 59.8 | 48.6 | - | 28.9 | 68.4 | 85.0 | - |
| MoE-LLaVA-2.7Bx4 | Phi2-2.7B | 1396.4 | 65.5 | 50.2 | 61.1 | 31.1 | 68.7 | 85.0 | - |

MaTVLM surpasses the teacher model by 17.6 points on MME, and by 0.2 and 1.6 points on POPE and MMMU, respectively. Compared to models of similar scale, it achieves gains of 87.7 points on MME and 7.0 points on TextVQA.

### Efficiency Comparison

| Metric | TinyLLaVA | MaTVLM (25% Mamba) | MaTVLM (50% Mamba) |
|------|-----------|--------------------|--------------------|
| Inference speedup | 1.0× | ~2.0× | **3.6×** |
| Memory reduction | 0% | ~15% | **27.5%** |

As the number of generated tokens increases, the inference time gap between MaTVLM and TinyLLaVA continues to widen.

### Ablation Study

| Distillation Strategy | MME | GQA | TextVQA | POPE |
|---------|-----|-----|---------|------|
| No distillation | 1320.5 | 56.8 | 52.1 | 83.4 |
| $\mathcal{L}_{\text{prob}}$ only | 1445.2 | 59.6 | 56.4 | 86.8 |
| $\mathcal{L}_{\text{layer}}$ only | 1401.3 | 58.7 | 55.2 | 85.9 |
| $\mathcal{L}_{\text{prob}} + \mathcal{L}_{\text{layer}}$ | **1484.0** | **60.4** | **57.7** | **87.4** |

Combining both distillation losses yields the best results. Among the two losses used individually, probability distribution distillation outperforms layer-wise distillation.

### Key Findings

1. The hybrid architecture achieves performance competitive with the pure Transformer teacher on most benchmarks.
2. A 25% Mamba-2 replacement ratio offers the optimal performance–efficiency trade-off.
3. Initializing Mamba-2 from attention weights is critical for accelerating convergence.
4. Single-stage distillation eliminates the complexity of multi-stage training pipelines.

## Highlights & Insights

1. **Theoretical contribution**: The paper establishes the mathematical correspondence Q↔C, K↔B, V↔x between attention and Mamba-2, providing a principled basis for weight transfer.
2. **Practical value**: The 3.6× speedup and 27.5% memory reduction make VLM deployment in resource-constrained environments feasible.
3. **Training efficiency**: Single-stage distillation substantially simplifies the training pipeline and lowers the technical barrier.
4. **Scalability**: The proposed method is generalizable to other VLM architectures and larger-scale models.

## Limitations & Future Work

1. Validation is limited to TinyLLaVA (3.1B parameters); scaling to larger models has not been explored.
2. The selection of replacement ratios (12.5%/25%/50%) lacks an adaptive strategy.
3. Mamba-2 still exhibits a performance gap on tasks requiring strong global reasoning (e.g., a 4.9-point drop on ScienceQA).
4. Distillation relies on a high-quality pretrained teacher model.

## Related Work & Insights

- **Efficient VLMs**: TinyLLaVA, MobileVLM, Qwen2.5-VL, etc.
- **SSM models**: Mamba, Mamba-2, and their applications to VLMs (Cobra, ML-Mamba).
- **Hybrid architectures**: MambaInLlama, MOHAWK, MambaVision.
- **Knowledge distillation**: DistillVLM, MAD, LLAVADI.

## Rating

- **Novelty**: ★★★★☆ — The attention-to-Mamba weight mapping and single-stage distillation strategy are novel.
- **Practicality**: ★★★★★ — The substantial inference speedup and memory savings offer strong practical value.
- **Experimental Thoroughness**: ★★★★☆ — Multi-benchmark validation, efficiency analysis, and ablation studies are included.
- **Writing Quality**: ★★★★☆ — The paper is logically structured with rigorous mathematical derivations.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)

<!-- RELATED:END -->
