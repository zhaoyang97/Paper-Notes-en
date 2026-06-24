---
title: >-
  [Paper Note] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models
description: >-
  [Multimodal VLM] This paper proposes the ReCAD framework, which rewrites CAD scripts as parametric code for SFT, then applies GRPO-based reinforcement learning with a hierarchical primitive curriculum learning strategy, enabling VLMs to generate high-precision, editable parametric CAD models from text or image inputs. The approach substantially outperforms existing methods in both in-distribution and out-of-distribution settings.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: d8c0747501f2ab8e
---

# ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models

| Info | Content |
|------|------|
| **Conference** | AAAI 2026 |
| **arXiv** | [2512.06328](https://arxiv.org/abs/2512.06328) |
| **Code** | Not released |
| **Area** | multimodal_vlm |
| **Keywords** | CAD generation, reinforcement learning, vision-language models, parametric code, curriculum learning |

## TL;DR

This paper proposes the ReCAD framework, which rewrites CAD scripts as parametric code for SFT, then applies GRPO-based reinforcement learning with a hierarchical primitive curriculum learning strategy, enabling VLMs to generate high-precision, editable parametric CAD models from text or image inputs. The approach substantially outperforms existing methods in both in-distribution and out-of-distribution settings.

## Background & Motivation

- **Practical demand for CAD modeling**: Industrial CAD modeling is time-consuming and requires high precision; generative CAD modeling has attracted broad attention from both academia and industry.
- **Limitations of prior work**:
    - Traditional encoder-decoder approaches (DeepCAD, Text2CAD, etc.) exhibit limited generalization and struggle to generate precise CAD models.
    - PLM-based methods (CAD-LLaMA, CAD-Coder) rely primarily on SFT to inject knowledge, treating PLMs merely as "semantic interpreters" without fully exploiting their generative priors.
    - Directly generating low-level parameter sequences (e.g., coordinates) lacks understanding of design intent, and parameter adjustments can easily produce invalid geometry (e.g., non-closed loops).
- **Core insight**: Parametric CAD modeling inherently requires precise mathematical reasoning, symbolic manipulation, and logical constraint satisfaction—exactly the capabilities that RLVR (reinforcement learning with verifiable rewards) has proven effective at in mathematics and code generation.
- **Key innovation**: Even when only simple function interfaces are provided (exposing only curve coordinate values), RL training can induce the emergence of complex CAD operations (e.g., circular patterns, mirror operations) that have not appeared in prior methods.

## Method

### 1. Problem Definition and CAD Hierarchy

Based on the Sketch-Extrude (SE) paradigm, five levels of hierarchical primitives are defined:

$$\mathcal{P} = \{\text{L (Loop)}, \text{F (Face)}, \text{S (Sketch)}, \text{SE (Sketch-Extrude)}, \text{MSE (Multi-SE)}\}$$

Components are progressively encapsulated from curves to loops, faces, sketches, and extrusions. A set of lightweight function interfaces is designed: the bottom layer provides only coordinate value access, while higher-level components are organized through structured encapsulation, ultimately forming a complete CAD model.

### 2. Parametric Code Generation and SFT Stage

**Conversion from hard-coded to parametric code**: Directly converting CAD sequences into code produces inflexible, overfitting-prone "hard-coded" representations. ReCAD uses a VLM (GPT-4o) to rewrite hard-coded code $C = f(P)$ into parametric code:

$$\{\hat{C}^i\}_{i=1}^N = \text{VLM}(I, C)$$

Quality filtering is then performed by computing cosine similarity of rendered images using a DINOv2 encoder:

$$\mathcal{C} = \left\{\hat{C}^i \mid \cos(E(\hat{I}_i), E(I)) > \tau_s \right\}$$

where $\tau_s = 0.95$ is the similarity threshold.

**Text description generation**: Leveraging the semantic information (scale, quantity) naturally embedded in parametric code, a VLM is used to generate both abstract descriptions $T^A$ and precise descriptions $T^D$, avoiding the verbose or imprecise annotations common in prior methods.

**SFT training**: Standard causal language modeling objectives are applied to fine-tune Qwen2.5-VL-7B-Instruct on both text-to-CAD and image-to-CAD tasks. UltraChat (23%) and OpenCodeReasoning (5%) data are mixed in to preserve general capabilities, yielding ReCAD-Base.

### 3. Reinforcement Learning Stage (Learn Under Guidance)

GRPO (Group Relative Policy Optimization) is adopted for reinforcement learning. The core innovation is a **guided learning strategy**.

**Hard problem identification**: Before RL training, $N$ solutions are sampled for each query $q_i$ and maximum reward is computed. If $\max\{R(q_i)\} < \tau_h$ (where $\tau_h = 0.8$), the problem is marked as hard.

**Guided objective**: For hard problems, parametric code $\mathcal{C}$ is incorporated as an off-policy guidance signal during rollout, providing complementary knowledge and leveraging the model's in-context learning ability to enhance reasoning:

$$\hat{\mathcal{J}}(\theta; \mathcal{C}) = \frac{1}{N-|\mathcal{C}|}\sum_{i=1}^{N-|\mathcal{C}|}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}\text{CLIP}(r_{i,t}, A_i, \epsilon) + \frac{1}{|\mathcal{C}|}\sum_{j=1}^{|\mathcal{C}|}\frac{1}{|\tau_j|}\sum_{t=1}^{|\tau_j|}\text{CLIP}(\hat{r}_{j,t}, A_j, \epsilon) - \beta\mathbb{D}_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}]$$

The final training objective adaptively switches based on difficulty:

$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}\left[\mathbf{1}_{\text{hard}}(q_i) \cdot \hat{\mathcal{J}}(\theta; \mathcal{C}_i) + (1 - \mathbf{1}_{\text{hard}}(q_i)) \cdot \mathcal{J}(\theta)\right]$$

### 4. Hierarchical Primitive Learning (HPL)

A curriculum learning strategy is designed to progressively learn from simple to complex, following the CAD hierarchy:

- **Learning order**: L → F → S → SE → MSE, with increasing complexity at each stage.
- **Intra-stage ordering**: Sorted by number of curves involved, from fewer to more.
- This mimics the human learning process, mastering foundational skills before tackling composite designs.

### 5. Reward Function Design

A unified reward combining geometric accuracy and semantic fidelity:

$$R(y_\pi, \Omega) = \lambda_1 \cdot \min\{\text{IOU}_{best}(\hat{\Omega}, \Omega),\ \phi(\text{sim}(\hat{I}, I), \tau)\} + \lambda_2 \cdot R_f(y_\pi)$$

- $\text{IOU}_{best}$: IoU under optimal alignment (geometric consistency)
- $\text{sim}(\hat{I}, I)$: DINOv2 feature cosine similarity (visual fidelity)
- $\phi(s, \tau)$: threshold-based linear scaling function, $\tau = 0.55$
- $R_f$: format reward (whether a valid `<think>` block is present)
- $\lambda_1 = 0.1$, $\lambda_2 = 0.9$

For the image-to-CAD task, since absolute scale information is unavailable from the input, geometry is normalized via the inertia matrix before reward computation.

## Key Experimental Results

### Text-to-CAD Generation

| Method | P-F1↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-------|-----------|---------|-----|
| GPT-4o | 50.55 | 107.55 | 165.67 | 15.14 |
| CAD-LLaMA | 60.02 | 41.77 | 98.12 | **0.39** |
| **ReCAD-VL** | **61.48** | **34.31** | **72.47** | 0.81 |

OOD (Fusion 360) setting:

| Method | P-F1↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-------|-----------|---------|-----|
| CAD-LLaMA | 50.47 | 60.36 | 142.48 | 1.29 |
| **ReCAD-VL** | **55.25** | **34.67** | **84.89** | **0.93** |

ReCAD-VL achieves substantial improvements in both in-distribution and OOD settings; Mean CD is reduced by 40% under the OOD setting, demonstrating strong generalization.

### Image-to-CAD Generation

| Method | IOU_best↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-----------|-----------|---------|-----|
| CAD-Coder | 61.23 | 8.09 | 73.47 | **1.05** |
| **ReCAD-VL** | **63.14** | **7.45** | **29.61** | 1.12 |

OOD (Fusion 360) setting:

| Method | IOU_best↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-----------|-----------|---------|-----|
| CAD-Coder | 45.32 | 84.02 | 272.06 | 2.23 |
| **ReCAD-VL** | **54.93** | **17.01** | **80.23** | **0.91** |

Mean CD drops from 73.47 to 29.61 (in-distribution) and from 272.06 to 80.23 (OOD), representing highly significant improvements.

### Ablation Study

| Configuration | P-F1 | Median CD↓ | Mean CD↓ | IR↓ |
|------|------|-----------|---------|-----|
| SFT only | 53.53 | 84.78 | 155.67 | 3.21 |
| RL only | 55.61 | 107.32 | 179.50 | 4.77 |
| w/o HPL | 59.63 | 44.64 | 90.83 | 2.42 |
| w/o Guidance | 60.03 | 42.85 | 87.34 | 0.93 |
| **Full model** | **61.48** | **34.31** | **72.47** | **0.81** |

- SFT and RL are both indispensable; neither alone achieves satisfactory results.
- Removing HPL increases reconstruction error and failure rate, confirming the effectiveness of hierarchical curriculum learning.
- The guided strategy provides complementary knowledge, further improving generation quality.

## Key Findings

1. **Emergent capabilities**: Through simple coordinate interfaces alone, RL training induces the emergence of complex CAD operations such as circular patterns and mirror operations, which do not explicitly appear in the training data.
2. **Zero-shot generalization**: Although trained exclusively on CAD generation tasks, ReCAD-VL demonstrates strong zero-shot capability across related tasks including CAD understanding, editing, and debugging.
3. **OOD robustness**: On Fusion 360 OOD data, ReCAD exhibits minimal performance degradation, whereas prior methods (e.g., CAD-LLaMA) suffer significant drops, indicating that parametric code combined with RL effectively mitigates overfitting.
4. **Complementarity of SFT and RL**: SFT injects external knowledge while RL reinforces generalization through self-exploration; their combination far surpasses either component in isolation.

## Highlights & Insights

- **Parametric code representation**: Rewriting hard-coded CAD code as parametric code preserves semantic information while enhancing flexibility, serving as a bridge between PLM code capabilities and the CAD domain.
- **Guided RL**: For hard problems, parametric code is incorporated as an off-policy guidance signal, cleverly leveraging LLMs' in-context learning ability to overcome the limitations of on-policy RL.
- **Hierarchical primitive curriculum learning**: The natural hierarchy of CAD structures is combined with curriculum learning, progressively building capability from curves to complete models.
- **Unified reward function**: Both geometric accuracy (IoU) and visual fidelity (DINOv2 feature similarity) are jointly considered to ensure generation quality.

## Limitations & Future Work

- **Dependence on GPT-4o**: Both parametric code rewriting and text description generation rely on GPT-4o, resulting in relatively high data preparation costs.
- **Restricted to the Sketch-Extrude paradigm**: Only sketch-extrude-based CAD modeling is supported; more complex operations (e.g., revolve, sweep, loft) are not handled.
- **Missing scale information**: Absolute scale cannot be recovered in the image-to-CAD task, necessitating geometric normalization.
- **Failure modes**: Two failure categories are observed: mismatch with the input description, and spatial parameterization errors.
- **Computational requirements**: Training requires 8×A800 80GB GPUs, posing a relatively high resource barrier.
- **7B model only**: The potential benefits of larger-scale models have not been explored.

## Related Work & Insights

- **CAD sequence modeling**: DeepCAD, CAD-Translator, Text2CAD, and others employ encoder-decoder or Transformer architectures.
- **PLM for CAD**: CAD-LLaMA and CAD-Coder adapt PLMs via SFT, but are constrained by the scope of injected knowledge.
- **Constraint-based sketch generation**: Vitruvion and SketchDNN focus on 2D sketches and rely on external constraint solvers.
- **RLVR**: DeepSeek-R1's GRPO has been validated for mathematical and code reasoning; LUFFY introduces off-policy reasoning signals—ReCAD extends this paradigm to the CAD domain.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](../../ICLR2026/multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)

</div>

<!-- RELATED:END -->
