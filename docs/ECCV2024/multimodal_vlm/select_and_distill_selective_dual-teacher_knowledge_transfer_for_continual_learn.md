---
title: >-
  [Paper Note] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][Continual Learning] This paper proposes a Selective Dual-Teacher Knowledge Transfer (SND) framework. By measuring the representation discrepancy between the pre-trained VLM and the recently fine-tuned VLM, it adaptively selects the appropriate teacher for knowledge distillation on an unlabeled reference dataset, mitigating catastrophic forgetting while maintaining zero-shot classification capabilities.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Continual Learning"
  - "Knowledge Distillation"
  - "Vision-Language Models"
  - "Catastrophic Forgetting"
  - "Zero-Shot Transfer"
date: 2026-05-08
content_hash: e81f33528e307d1a
---

# Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2403.09296](https://arxiv.org/abs/2403.09296)  
**Code**: [Project Page](https://chuyu.org/research/snd)  
**Area**: Multimodal VLM  
**Keywords**: Continual Learning, Knowledge Distillation, Vision-Language Models, Catastrophic Forgetting, Zero-Shot Transfer

## TL;DR
This paper proposes a Selective Dual-Teacher Knowledge Transfer (SND) framework. By measuring the representation discrepancy between the pre-trained VLM and the recently fine-tuned VLM, it adaptively selects the appropriate teacher for knowledge distillation on an unlabeled reference dataset, mitigating catastrophic forgetting while maintaining zero-shot classification capabilities.

## Background & Motivation
Large-scale vision-language models (such as CLIP) exhibit powerful zero-shot generalization capabilities, but face two core challenges when adapting to sequential downstream tasks:

**Catastrophic Forgetting**: After fine-tuning on a new task, the model forgets the knowledge of previous tasks.

**Zero-Shot Degradation**: The fine-tuning process damages the inherent zero-shot transfer capabilities of pre-trained VLMs.

Existing methods have their limitations:
- **Rehearsal-based methods** (e.g., iCaRL): Require storing previous data, posing privacy risks and storage burdens.
- **Data-free methods** (e.g., DF-CL): Mainly target closed-set classification, unable to handle the open-vocabulary nature of VLMs.
- **ZSCL**: Only distills knowledge from the pre-trained model to maintain zero-shot capabilities, but fails to effectively retain previous task knowledge, because the pre-trained model has never been fine-tuned on previous tasks.

The core insight of this paper: **It is necessary to leverage two teachers simultaneously** — the recently fine-tuned VLM to retain previous task knowledge, and the original pre-trained VLM to maintain zero-shot capabilities. The key question is: for each image in the reference dataset, how to determine which teacher to distill knowledge from?

## Method

### Overall Architecture
Given the current task $\mathcal{T}^k$, the recently fine-tuned model $g_{k-1}$, the pre-trained model $g_0$, and an unlabeled reference dataset $\mathcal{X}^{ref}$ (e.g., ImageNet), the framework executes the following pipeline:
1. For each image in reference dataset, calculate the representation discrepancy between the dual teachers.
2. Select the appropriate teacher for knowledge distillation based on the discrepancy magnitude.
3. Combine with the cross-entropy loss of the current task for joint training.

### Key Designs

1. **Dual-Teacher Discrepancy**:

    - Function: Measures the representation discrepancy between $g_{k-1}$ and $g_0$ on the same reference image to guide teacher selection.
    - Mechanism: If the reference image is similar to the distribution of previously fine-tuned data, the representations of $g_{k-1}$ will significantly deviate from those of $g_0$ (as fine-tuning alters representation), resulting in a large discrepancy $d$. If the image does not belong to the previous data distribution, the representations from both teachers should be similar (small discrepancy), because the representation of $g_{k-1}$ for such images remains unchanged by fine-tuning.
    - Formal expression: $\mathbb{E}_{\mathbf{x} \in \mathcal{X}^{1:k-1}}[d(g_{k-1}(\mathbf{x}), g_0(\mathbf{x}))] \geq \mathbb{E}_{\mathbf{x}' \notin \mathcal{X}^{1:k-1}}[d(g_{k-1}(\mathbf{x}'), g_0(\mathbf{x}'))]$
    - Euclidean distance is used for the discrepancy measurement.
    - Design Motivation: Without accessing prior task data, the reference image's relevance to the prior training data can be inferred solely based on the representation discrepancy between the dual teachers.

2. **Selection Scoring Function**:

    - Function: Maps the dual-teacher discrepancy to a selection score in $[0, 1]$.
    - Core formula: $\eta(\mathbf{x}) = \sigma\left(\frac{d(g_{k-1}(\mathbf{x}), g_0(\mathbf{x})) - \delta}{\gamma}\right)$
    - where $\sigma$ is the sigmoid function, and $\delta$ and $\gamma$ are normalization hyperparameters.
    - $\eta > 0.5$: Favors selecting $g_{k-1}$ as the teacher (mitigating catastrophic forgetting).
    - $\eta < 0.5$: Favors selecting $g_0$ as the teacher (maintaining zero-shot capabilities).
    - Design Motivation: Uses a sigmoid function to achieve smooth teacher selection rather than hard switching.

3. **Selective Dual-Teacher KD**:

    - Function: Weighted distillation of knowledge from both teachers based on the selection score.
    - Distillation loss: $\mathcal{L}_{KD}^{dual} = \sum_{\mathbf{x} \sim \mathcal{X}^{ref}} \eta(\mathbf{x}) \cdot \mathcal{L}_{KD}^{k-1} + (1 - \eta(\mathbf{x})) \cdot \mathcal{L}_{KD}^{0}$
    - where $\mathcal{L}_{KD}^{k-1} = d(g_{k-1}(\mathbf{x}), g_k(\mathbf{x}))$ retains knowledge of previous tasks.
    - $\mathcal{L}_{KD}^{0} = d(g_0(\mathbf{x}), g_k(\mathbf{x}))$ maintains zero-shot capabilities.
    - Design Motivation: Instead of simply selecting a single teacher, soft-weighting via the $\eta$ score is applied to smoothly transition the contributions of both teachers.

### Loss & Training
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{KD}^{dual}$
- During training, only the image encoder is updated, while the text encoder remains frozen.
- Uses the AdamW optimizer with a cosine learning rate scheduler, starting with an initial learning rate of $1 \times 10^{-5}$.
- Weight decay is set to $5 \times 10^{-4}$.
- The text prompt is fixed as "a photo of a \<CLASS\>".

### Evaluation Protocol Design
This paper proposes a more rigorous evaluation protocol:
- **Multiple Training Sequences**: Construct $K$ cyclically left-shifted training sequences to ensure each dataset has the opportunity to serve as the first and last task.
- **Three Metrics**: Average accuracy, catastrophic forgetting (average maximum performance drop), and zero-shot degradation (average maximum performance drop on unseen tasks).

## Key Experimental Results

### Main Results: MTIL Benchmark (Average of 8 Sequences)

| Method | Catastrophic Forgetting (↓) | Zero-Shot Degradation (↓) | Average Accuracy (↑) |
|------|--------------|--------------|--------------|
| Continual FT | 12.04 | 21.15 | 75.17 |
| LwF | 8.56 | 10.03 | 78.81 |
| iCaRL | 8.04 | 12.92 | 78.55 |
| ZSCL | 3.64 | 3.00 | 82.92 |
| MoE-Adapters | 2.71 | 2.17 | 82.29 |
| **Ours (SND)** | **1.20** | **1.96** | **84.92** |

### Ablation Study: Teacher Selection Strategy (Sequence S1)

| Configuration | Forgetting (↓) | Degradation (↓) | Average Accuracy (↑) |
|------|---------|---------|--------------|
| Distillation from $g_0$ only | 5.26 | 2.51 | 81.35 |
| Distillation from $g_{k-1}$ only | 2.63 | 3.36 | 83.61 |
| **Dual-Teacher Selective Distillation** | **1.70** | **1.55** | **84.48** |

### Key Findings
- **Substantial Reduction in Forgetting**: The catastrophic forgetting of SND (1.20) is only 1/3 of ZSCL (3.64), and 10 times lower than Continual FT (12.04).
- **Excellent Zero-Shot Preservation**: Zero-shot degradation (1.96) is better than ZSCL (3.00), close to MoE-Adapters (2.17), but with a 2.63% higher overall accuracy.
- **MCIL is More Challenging but Still Leading**: In the class-incremental learning setup, SND achieves the best results in both forgetting (1.35) and degradation (1.65).
- **Empirical Validation of the Dual-Teacher Discrepancy Hypothesis**: The discrepancy between $g_1$ and $g_0$ on the Aircraft dataset is 1.059, significantly higher than on other non-fine-tuned datasets (0.067-0.170), validating the effectiveness of the selection mechanism.
- **Visualization Validation**: Reference images with high selection scores $\eta$ are indeed visually similar to the previously fine-tuned tasks.

## Highlights & Insights
- **Clear Problem Definition**: Deconstructs VLM continual learning into two quantifiable sub-goals (anti-forgetting + zero-shot preservation), which is more comprehensive than prior work.
- **Intuitive and Simple Selection Mechanism**: Utilizes the representation discrepancy between the dual teachers to automatically identify the attribution of reference images, without requiring any label information.
- **Rigorously Evaluated**: The evaluation protocol with $K$ cyclic sequences is more reliable than previous setups that only select 1-2 sequences.
- **No Extra Parameters**: Unlike MoE-Adapters, which require adding new adapter modules, SND achieves this solely through weighting the distillation loss, adding no model parameters.

## Limitations & Future Work
- **Dependence on Reference Dataset**: If the domain gap between the reference dataset and the previously fine-tuned tasks is too large (e.g., medical images), $g_{k-1}$ will rarely be selected as the teacher, and catastrophic forgetting may not be effectively mitigated.
- **Hyperparameters $\delta$ and $\gamma$**: The normalization parameters of the selection scoring function need to be tuned for specific scenarios.
- **Only Working on the Image Encoder**: Continual learning on the text encoder side has not been explored.
- **Scalability**: As the number of tasks increases, $g_{k-1}$ may deviate too far from $g_0$, which could degrade the reliability of the discrepancy measurement.
- Using larger-scale reference datasets (e.g., CC12M, subset of LAION 5B) could be considered.

## Related Work & Insights
- **ZSCL**: The most direct predecessor, which only distills knowledge from the pre-trained model to maintain zero-shot capabilities. This work adds the recently fine-tuned model as a second teacher.
- **MoE-Adapters**: Introduces incremental adapters as a mixture of experts, which is another anti-forgetting strategy but requires extra modules and test-time selectors.
- **iCaRL/LwF**: Classic continual learning methods that show limited effectiveness in VLM scenarios.
- Insight: In data-free scenarios, the representation discrepancy of the model itself inherently contains rich "data attribution" information, which can serve as a selection criterion for various distillation strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of dual-teacher discrepancy-driven selective distillation is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 training sequences, both MTIL and MCIL benchmarks, and detailed analysis/visualization.
- Writing Quality: ⭐⭐⭐⭐ Well-structured paper with smooth logical transitions for motivation and methodology derivations.
- Value: ⭐⭐⭐⭐ VLM continual learning is an important direction; the proposed method is simple, effective, and practically valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICLR 2026\] Fed-Duet: Dual Expert-Orchestrated Framework for Continual Federated Vision-Language Learning](../../ICLR2026/multimodal_vlm/fed-duet_dual_expert-orchestrated_framework_for_continual_federated_vision-langu.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](../../CVPR2025/multimodal_vlm/synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2025/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)

</div>

<!-- RELATED:END -->
