---
title: >-
  [Paper Note] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages
description: >-
  [CVPR 2025][Multimodal VLM][Few-Shot Adaptation] By analyzing the learning dynamics of PEFT in few-shot adaptation, this work discovers that the training process naturally divides into two stages: "task-level feature extraction" and "available class specialization". Accordingly, the authors propose 2SFS: first tuning LayerNorm to learn general/task-level features, and then training a linear classifier to enhance known-class discrimination. 2SFS matches or exceeds SOTA perform…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Few-Shot Adaptation"
  - "CLIP"
  - "Parameter-Efficient Fine-Tuning"
  - "Two-Stage Learning"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 84fea95b4037ca8b
---

# Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages

**Conference**: CVPR 2025  
**arXiv**: [2503.11609](https://arxiv.org/abs/2503.11609)  
**Code**: [https://github.com/FarinaMatteo/rethinking_fewshot_vlms](https://github.com/FarinaMatteo/rethinking_fewshot_vlms)  
**Area**: Multimodal VLMs  
**Keywords**: Few-Shot Adaptation, CLIP, Parameter-Efficient Fine-Tuning, Two-Stage Learning, Vision-Language Models

## TL;DR

By analyzing the learning dynamics of PEFT in few-shot adaptation, this work discovers that the training process naturally divides into two stages: "task-level feature extraction" and "available class specialization". Accordingly, the authors propose 2SFS: first tuning LayerNorm to learn general/task-level features, and then training a linear classifier to enhance known-class discrimination. 2SFS matches or exceeds SOTA performance under both base-to-novel and all-to-all settings.

## Background & Motivation

The **Key Challenge** of Few-Shot Adaptation (FSA) lies in the massive parameter scale of VLMs (hundreds of millions) versus the extremely limited samples per class (e.g., 16-shot), making full-parameter fine-tuning highly prone to overfitting. Existing methods primarily fall into two categories:

1. **Prompt Tuning** (CoOp, CoCoOp, MaPLe, etc.): Inserting learnable context vectors into the text/vision encoders.
2. **Adapter-based** (CLIP-Adapter, TaskRes, CLIP-LoRA, etc.): Appending parameterized adapter modules to the frozen model.

These methods typically perform well in specific settings (either base-to-novel or all-to-all) but exhibit **poor generalization across settings**. For example, MMA achieves SOTA in base-to-novel but degrades severely in all-to-all, whereas CLIP-LoRA behaves conversely.

More critically, there is a lack of deep understanding in the community regarding what actually happens during FSA training. Through empirical studies, the authors unveil a pivotal phenomenon for the first time: **there exists a natural "breakpoint" in the learning dynamics of PEFT**, before which the performance on both base and novel classes increases concurrently, and after which base class performance continues to rise while novel class performance begins to deteriorate.

## Method

### Overall Architecture

2SFS (Two-Stage Few-Shot Adaptation) splits a fixed computational budget of $m$ iterations into two distinct stages:
- **First Stage** ($\alpha \times m$ steps): Fine-tuning LayerNorm parameters to learn task-level features.
- **Second Stage** ($(1-\alpha) \times m$ steps): Freezing the feature extractor and training a linear classifier.

During inference, base classes are retrieved directly via lookup ($O(1)$), while novel classes are computed dynamically through the text encoder.

### Key Designs

1. **Learning Dynamics Analysis and Breakpoint Discovery**:
    - **Function**: Reveal the intrinsic patterns of PEFT training to provide a theoretical foundation for the two-stage design.
    - **Mechanism**: Train three PEFT methods (LayerNorm tuning, LoRA, BitFit) on CLIP ViT-B/16 for 8000 steps under a 16-shot setting, continuously monitoring base/novel accuracy on held-out datasets. It is observed that all PEFT methods follow a consistent pattern—there exists a "breakpoint" where, prior to it, base and novel class accuracy **simultaneously improve** (learning generalizable task-level features); beyond it, base accuracy keeps rising while novel accuracy begins to degrade (specializing to the available data).
    - **Design Motivation**: This is not overfitting in the conventional sense (as it is evaluated on held-out test data), but rather a process where PEFT overrides task-general knowledge when specializing to existing classes. LayerNorm tuning demonstrates the highest robustness after the breakpoint (slowest degradation), making it the chosen PEFT strategy for the first stage.

2. **Stage One: LayerNorm Fine-Tuning**:
    - **Function**: Learn transferable task-level feature representations.
    - **Mechanism**: Fine-tune the scale $\gamma$ and shift $\beta$ parameters of all LayerNorm instances in both vision and text encoders. For a $d$-dimensional activation vector $\mathbf{a}$, $\text{LayerNorm}(\mathbf{a}) = \gamma \odot \frac{\mathbf{a} - \mu(\mathbf{a})}{\sigma(\mathbf{a})} + \beta$. Standard softmax cross-entropy loss is optimized on base classes for $\alpha \times m$ steps.
    - **Design Motivation**: LayerNorm possesses very few parameters but globally aligns feature distributions. Experiments show it degrades the slowest post-breakpoint, identifying it as the best candidate for capturing task-level features. The key is to halt fine-tuning exactly at $\alpha \times m$ steps to avoid entering the specialization phase.

3. **Stage Two: Linear Classifier + Selective Inference**:
    - **Function**: Enhance the discriminability of base classes while preserving the generalization of novel classes.
    - **Mechanism**: Freeze the LayerNorm parameters learned in Stage One, initialize the classifier weight matrix $\Phi_{\mathcal{B}}$ with the text encoder's base class embeddings from Stage One $\phi_b = f^t_{\omega^{*t}_{LN}}(b)$, and then optimize $\Phi_{\mathcal{B}}$ over the remaining $(1-\alpha) \times m$ steps. During inference, base classes directly leverage row vectors of the classifier $\phi_b^*$ ($O(1)$ lookup), whereas novel classes are computed dynamically through the text encoder $f^t_{\omega^{*t}_{LN}}(c)$.
    - **Design Motivation**: Switching to a disjoint set of parameters (from LayerNorm to classifier weights) prevents further fine-tuning from distorting the task-level features learned in the first stage. While the linear classifier is simple, initializing and optimizing it with high-quality Stage One features yields surprisingly strong performance. Selective inference is a unique by-product of this two-stage design.

### Loss & Training

- Standard softmax cross-entropy loss is employed in both stages for optimizing either LayerNorm parameters or classifier weights.
- Hyperparameters are kept fixed: $\alpha = 0.5$ (equal split of training budget), $m = 8000$ steps, SGD optimizer.
- **Exactly the same hyperparameters** are shared across all 11 datasets, 2 settings, and 3 backbones.

## Key Experimental Results

### Main Results (Base-to-Novel, ViT-B/16, 16-shot Average)

| Method | Base | Novel | HM |
|------|------|-------|-----|
| CLIP zero-shot | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| CLIP-LoRA | 85.32 | 70.63 | 77.28 |
| MMA | 83.20 | 76.80 | 79.87 |
| **2SFS** (Ours) | **85.55** | **75.48** | **80.20** |

Single-dataset highlights:

| Dataset | Metric | 2SFS | MMA | Description |
|--------|------|------|-----|------|
| Stanford Cars | HM | 78.46 | 75.70 | Gain of 2.76% |
| Oxford Flowers | HM | 85.83 | 85.48 | Slightly better |
| Caltech101 | HM | 96.52 | 96.15 | Slightly better |
| ImageNet | HM | 74.20 | 74.02 | Slightly better |

### Ablation Study

| Configuration | Description |
|------|------|
| Replacing LN with LoRA in Stage One | HM drops by ~1-2%, verifying LN robustness |
| $\alpha = 0$ (Classifier only) | Degrades to linear probing, leading to a significant HM drop |
| $\alpha = 1$ (LN tuning only) | Lacks the classifier specialization phase, leading to weaker base class performance |
| Different backbones (ViT-B/32, ViT-L/14) | Shows consistent trends, where 2SFS achieves optimal or near-optimal results |

### Key Findings

- **2SFS is the only method that maintains competitive performance in both base-to-novel and all-to-all settings**: MMA performs well in base-to-novel but struggles in all-to-all, whereas CLIP-LoRA behaves conversely.
- The breakpoint phenomenon **consistently occurs** across all datasets and PEFT methods, indicating it is an inherent property of PEFT learning dynamics.
- LayerNorm tuning leverages remarkably few parameters yet yields surprisingly strong results, demonstrating that the scale/shift of LayerNorm regulates critical aspects of feature distribution.
- Selective inference provides practical benefits in execution efficiency: base-class inference comes at zero computed text-embedding cost.

## Highlights & Insights

- **Simple yet profound insight**: The core contribution is not a complicated architecture, but an in-depth analysis of PEFT learning dynamics. The return to the old-school recipe of "learning global features first, then learning a classifier" becomes convincing because of the anchoring theoretical/empirical analysis.
- **Fixed hyperparameters across settings/backbones/datasets**: Extremely rare in the FSA field, underscoring the strong robustness of the method.
- **The breakpoint phenomenon itself is a valuable contribution**: It can guide early-stopping strategies for other PEFT methods.
- Selective inference offers actual runtime efficiency gains.

## Limitations & Future Work

- The position of the breakpoint varies depending on the dataset and PEFT method; while $\alpha = 0.5$ is chosen empirically, adaptive scheduling for the breakpoint could yield better results.
- Only classification tasks have been validated; expansion to other downstream tasks like object detection or segmentation is yet to be explored.
- Stage Two utilizes a simple linear classifier; incorporating a more sophisticated classification head (e.g., non-linear adapters) might provide further gains.
- Evaluated only on the CLIP family of models; whether the findings generalize to other VLMs like SigLIP or EVA-CLIP remains to be verified.

## Related Work & Insights

- Orthogonal to prompt tuning techniques like CoOp/CoCoOp. 2SFS can be generalized as a broader framework addressing "when to halt PEFT and switch to a simple classifier".
- The breakpoint phenomenon shares similarities with catastrophic forgetting in continual learning, yet differs fundamentally (as base class performance does not drop; rather, novel class performance degrades).
- Insight: Do similar two-stage dynamics exist in other PEFT contexts such as instruction tuning?

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core insight is novel; the method is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 datasets × 2 settings × 3 backbones with fixed hyperparameters, offering a comprehensive comparison.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical flow, seamlessly transitioning from observation to methodology to experiments.
- **Value**: ⭐⭐⭐⭐ The analysis of PEFT learning dynamics holds broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)

</div>

<!-- RELATED:END -->
