---
title: >-
  [Paper Note] OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad
description: >-
  [CVPR 2025][Self-Supervised Learning][SAM] OCRT proposes a plug-and-play three-stage pipeline—Object (Slot Attention decoupling), Concept (importance filtering), and Relation (concept graph reasoning)—which significantly improves the accuracy of SAM on weakly-supervised medical/camouflaged segmentation and the robustness of CLIP under adversarial attacks, without altering the FM backbone.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "SAM"
  - "CLIP"
  - "Slot Attention"
  - "Concept Graph"
  - "Adversarial Fine-tuning"
date: 2026-05-08
content_hash: 963e1e1764e06a67
---

# OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad

**Conference**: CVPR 2025  
**arXiv**: [2503.18695](https://arxiv.org/abs/2503.18695)  
**Code**: See paper (Github)  
**Area**: Foundation Models / Generalization & Robustness / Concept Learning  
**Keywords**: SAM, CLIP, Slot Attention, Concept Graph, Adversarial Fine-tuning

## TL;DR
OCRT proposes a plug-and-play three-stage pipeline—Object (Slot Attention decoupling), Concept (importance filtering), and Relation (concept graph reasoning)—which significantly improves the accuracy of SAM on weakly-supervised medical/camouflaged segmentation and the robustness of CLIP under adversarial attacks, without altering the FM backbone.

## Background & Motivation
1. **Background**: Foundation models such as SAM and CLIP experience severe performance degradation under Out-of-Distribution (OOD) scenarios (distribution shifts, weak supervision, and adversarial attacks).
2. **Limitations of Prior Work**: Existing generalization methods (e.g., LoRA, Adapters, adversarial training, feature decoupling) are mostly task-specific or model-specific (e.g., SAM using mask alignment, CLIP using semantic consistency), making them hard to generalize.
3. **Key Challenge**: FMs lack human cognitive-like abstraction capabilities to map "dense pixels $\rightarrow$ discrete objects $\rightarrow$ high-level concepts $\rightarrow$ relations", rendering them highly sensitive to low-level factors such as background and illumination.
4. **Goal**: Design a fine-tuning framework that is universally applicable to both VFMs and MMFMs, does not alter the backbone, and can inject the three-level prior of object-concept-relation.
5. **Key Insight**: Humans first decouple objects from dense visual signals, abstract concepts, and then perform relational reasoning. This inductive bias is crucial for OOD generalization.
6. **Core Idea**: Use Slot Attention for decoupling $\rightarrow$ apply concept importance weighting and filtering $\rightarrow$ construct a concept graph for relational reasoning $\rightarrow$ inject the relations back to fine-tune the FM.

## Method

### Overall Architecture
- The input image $\mathbf{x}$ is processed through the FM encoder to obtain $\mathbf{z}$.
- (1) Object Phase: Slot Attention + GRU iteration + spatial broadcast decoder decouples $\mathbf{z}$ into $K$ object-centric slots $\mathbf{o} \in \mathbb{R}^{K \times D_o}$ and their corresponding spatial masks $\mathbf{m}_k$.
- (2) Concept Phase: Concept $\mathbf{c}_k = \mathbf{z}_k \odot \mathbf{m}_k$ is extracted. The importance of each concept is estimated via the mean cosine similarity $\omega_k$, and Top-$\tilde K$ concepts are selected to suppress irrelevant ones.
- (3) Relation Phase: The retained concepts are formed into a flexible-degree concept graph to perform high-order factor extraction and relational reasoning $\rightarrow$ outputting relation tokens to be injected into the FM.
- Joint training is conducted with the FM base loss (e.g., teacher-student dice/focal loss for SAM, adversarial fine-tuning $L_2$ loss for CLIP).

### Key Designs

1. **Object: Slot Attention + Iterative Decoupling**

    - **Function**: Decouples the dense patch features output by the FM into $K$ object slots.
    - **Mechanism**: Learnable query slots interact with FM features via cross-attention, normalized across the slot dimension with softmax to ensure competition, and iteratively refined using a GRU. A spatial broadcast decoder outputs the mask for each slot and reconstructed features $\hat{\mathbf{z}} = \sum_k \hat{\mathbf{z}}_k \odot \mathbf{m}_k$, supervised by a reconstruction loss $\|\hat{\mathbf{z}} - \mathbf{z}\|^2$.
    - **Design Motivation**: Unsupervisedly decompose objects to prevent the FM from introducing background biases when directly operating on pixels.

2. **Concept: Importance Weighting + Top-K Suppression**

    - **Function**: Retain informative concepts from the $K$ object slots for the target task.
    - **Mechanism**: $\omega_k = \frac{1}{K} \sum_j \frac{\mathbf{c}_k \cdot \mathbf{c}_j}{\|\mathbf{c}_k\|\|\mathbf{c}_j\|}$ measures the average similarity of the $k$-th concept with other concepts. Redundant/common concepts receive higher scores (task-related), while isolated background concepts receive lower scores. An indicator function is used to select the Top-$\tilde K$ concepts: $\mathbf{z}_{cpt} = \mathbf{z}_{obj} \odot \sum \mathbb{I}_{[k\in \text{Top}^{\tilde K}]} \mathbf{m}_k$.
    - **Design Motivation**: Not all slots are useful in the open world; explicitly suppressing irrelevant concepts significantly reduces OOD noise.

3. **Relation: Concept Graph + High-Order Reasoning**

    - **Function**: Form a graph with the retained concepts as nodes and edge weights determined by concept similarities, followed by high-order factor aggregation.
    - **Mechanism**: Use GAT/GCN-style message passing to perform multiple rounds of relational reasoning on the concept graph, producing relation tokens concatenated with queries/prompts of the FM decoder (such as SAM's mask decoder or the rear stages of CLIP's image encoder).
    - **Design Motivation**: Human generalization relies on "relations" rather than isolated objects. Injecting graph priors into the FM explicitly models "which relations between objects remain invariant" under OOD settings.

### Loss & Training
- Total loss = FM base loss ($\mathcal{L}^{\text{base}}_{\text{SAM/CLIP}}$) + Slot reconstruction loss $\mathcal{L}_{\text{REC}}$ + relation reasoning task loss.
- Only the OCRT module and the decoder head are trained, while the FM encoder is frozen or fine-tuned using LoRA.

## Key Experimental Results

### Main Results
**SAM Weakly-Supervised Segmentation** (using box / point / poly weak labels):

| Dataset | Metric | SAM | OCRT |
|--------|------|-----|------|
| COCO 2017 | box mIoU | 74.29 | Significant Improvement |
| Pascal VOC | point mIoU | 69.21 | Significant Improvement |
| kvasir-SEG (Medical) | poly mIoU | 54.03 | Substantial Improvement |
| ISIC (Dermatology) | point mIoU | 53.42 | Substantial Improvement |

**CLIP Adversarial Robustness (LLaVA backbone)**: Under various attacks (PGD/FGSM), OCRT reduces hallucination and preserves zero-shot performance.

### Ablation Study

| Configuration | Performance Drop |
|------|---------|
| Remove Object stage | Significant drop (no spatial structure) |
| Remove Concept importance suppression | Moderate drop (introduces noisy slots) |
| Remove Relation graph | Moderate drop (only concepts, no relations) |
| Top-$\tilde K$ selected too large | Performance drop (noise) |
| Top-$\tilde K$ selected too small | Performance drop (insufficient information) |

### Key Findings
- All three stages are indispensable. The Object stage is the foundation, and the Relation stage yields the largest improvement margins.
- OCRT achieves the largest gains on extreme OOD scenarios such as medical and camouflaged segmentation, demonstrating its capability to suppress object-concept distribution shifts.
- The number of slots $K$ does not need to be precisely aligned with the number of objects in the scene, showing the model's robustness to $K$.

## Highlights & Insights
- **Cognitive Science-Driven FM Fine-Tuning Framework**: Injecting "object-centric $\rightarrow$ concept $\rightarrow$ relation" as an explicit structure serves as a beneficial complement to current fine-tuning methods dominated by LoRA/Adapters.
- **Unsupervised Concept Importance Estimation**: Employing "mean similarity with other slots" as saliency requires no annotations, offering a simple and highly transferable trick.
- **Unification of VFM and MMFM**: The same framework can be integrated with both SAM and CLIP, demonstrating the universality of the triad structure.
- **Concept Graphs Naturally Suit Relational Reasoning**: The design can be readily extended to tasks such as VQA and scene graph generation.

## Limitations & Future Work
- Slot Attention exhibits slow training speeds and is sensitive to hyperparameters ($K$, iterations, temperature).
- The concept importance formula assumes "important = similar to others," which might mistakenly suppress unique yet critical concepts (e.g., rare diseases).
- Relational graph reasoning introduces additional computation and memory overhead.
- Broadening the framework to LLMs (beyond CLIP) has not yet been explored.
- **Directions for Improvement**: Utilizing sparse OT (Optimal Transport) or matching to replace Top-K suppression to preserve key rare concepts.

## Related Work & Insights
- **vs. LoRA / Adapter**: LoRA lacks structural priors, whereas this work explicitly injects the object-concept-relation structure.
- **vs. Original Slot Attention**: The original work only performs unsupervised object decoupling; this work introduces concept filtering and graph reasoning to establish a complete pipeline.
- **vs. SAM-LoRA Fine-Tuning / CLIP Adversarial Fine-Tuning**: Outperforms both types of methods across multiple OOD datasets.
- **Insight**: Any training setup involving "foundation models + weakly-supervised tasks" can benefit from this universal patch: "first decouple objects, then filter concepts, and finally perform graph reasoning".

## Rating
- Novelty: ⭐⭐⭐⭐ The alignment of the triad architecture with cognitive science is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task + multi-attack + cross-FM validation
- Writing Quality: ⭐⭐⭐ Formulas are abundant but the logic is clear, notations are slightly complex
- Value: ⭐⭐⭐⭐ Plug-and-play and brings significant improvements to OOD settings, serving as a valuable reference for any FM fine-tuning work

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Plug-and-Play Compositionality for Boosting Continual Learning with Foundation Models](../../ICLR2026/self_supervised/plug-and-play_compositionality_for_boosting_continual_learning_with_foundation_m.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](../../ICML2025/self_supervised/what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](mos_modeling_object-scene_associations_in_generalized_category_discovery.md)
- [\[CVPR 2026\] SECOS: Semantic Capture for Rigorous Classification in Open-World Semi-Supervised Learning](../../CVPR2026/self_supervised/secos_semantic_capture_for_rigorous_classification_in_open-world_semi-supervised.md)
- [\[ICLR 2026\] Boosting Open Set Recognition Performance through Modulated Representation Learning](../../ICLR2026/self_supervised/boosting_open_set_recognition_performance_through_modulated_representation_learn.md)

</div>

<!-- RELATED:END -->
