---
title: >-
  [Paper Note] Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition
description: >-
  [NeurIPS 2025 (Spotlight)][Video Understanding][Explainable video action recognition] This paper proposes DANCE, a framework that achieves structured and motion-aware explainable video action recognition by disentangling action explanations into three concept types: motion dynamics, objects, and scenes.
tags:
  - NeurIPS 2025 (Spotlight)
  - Video Understanding
  - Explainable video action recognition
  - concept bottleneck model
  - motion disentanglement
  - pose sequences
  - concept discovery
date: 2026-05-08
content_hash: 0879e2f710ad7c59
---

# Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition

**Conference**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2511.03725](https://arxiv.org/abs/2511.03725)  
**Code**: [Available](https://jong980812.github.io/DANCE/)  
**Area**: Video Understanding / Explainable AI  
**Keywords**: Explainable video action recognition, concept bottleneck model, motion disentanglement, pose sequences, concept discovery

## TL;DR

This paper proposes DANCE, a framework that achieves structured and motion-aware explainable video action recognition by disentangling action explanations into three concept types: motion dynamics, objects, and scenes.

## Background & Motivation

Video action recognition models have achieved remarkable performance, yet their decision-making processes remain opaque. Existing explainability methods exhibit notable limitations:

**Saliency methods** (Saliency Tubes, GradCAM, etc.): produce entangled explanations that cannot distinguish whether a model relies on motion or spatial context.

**Language-based methods** (LLM-generated concept descriptions): can describe objects and scenes but struggle to express motion dynamics — motion constitutes **tacit knowledge**, i.e., knowledge that is intuitively understood but difficult to verbalize.

From a cognitive science perspective, humans perceive actions by separately analyzing two factors:
- **Temporal dynamics**: how motion unfolds over time
- **Spatial context**: surrounding objects and scenes

Consequently, ideal video XAI should explicitly disentangle temporal dynamics from spatial context — a requirement unmet by existing approaches.

## Method

### Overall Architecture

DANCE adopts an ante-hoc concept bottleneck design, inserting a concept layer between a pretrained video backbone encoder and the final classifier. The prediction pipeline is:

**Input video → Video features → Three-type concept activations (motion dynamics, objects, scenes) → Action prediction**

Each concept type has its own concept layer parameters $W_C = [W_C^m; W_C^o; W_C^s]$, ensuring explicit disentanglement across concept types.

### Key Designs

**1. Motion Dynamics Concepts**

Core innovation: motion concepts are defined via **human pose sequences** rather than textual descriptions.

- **Key segment selection**: informative short clips are extracted from videos via keyframe detection
- **Pose sequence extraction**: a 2D pose estimator extracts per-frame poses $P_i^s \in \mathbb{R}^{L \times J \times 2}$ for each key segment
- **Concept discovery via clustering**: pose sequences from all training videos are aggregated and clustered using the FINCH algorithm to discover representative motion patterns
- **Concept annotation**: binary labels $c_k^m = I(\sum_s a_{i,s,k})$ are automatically generated from cluster assignments

Advantage: pose sequences provide appearance-agnostic motion representations that allow users to intuitively understand how an action unfolds over time.

**2. Object & Scene Concepts**

- GPT-4o is queried to retrieve objects and scenes associated with each action class
- Pseudo-labels are automatically generated via a vision-language dual encoder (InternVid)
- Object pseudo-labels: $\tilde{c}_i^o = E_T(\mathcal{O}) E_V(V_i)$

**3. Concept Bottleneck Architecture**

- The pretrained video backbone (VideoMAE) is frozen; only the concept layer and classifier are trained
- The concept layer projects video features into concept space to obtain activations $z = [z_m; z_o; z_s]$
- The classifier predicts actions based on concept activations

### Loss & Training

Training proceeds in two stages:

**Stage 1: Concept layer training**
- Motion dynamics concepts: binary cross-entropy loss (motion labels are multi-label)
$$\mathcal{L}_m = -\frac{1}{M_m}\sum_{k=1}^{M_m}[c_k^m\log\sigma_k(z_m) + (1-c_k^m)\log(1-\sigma_k(z_m))]$$
- Object/scene concepts: cosine cubed loss, emphasizing directional alignment

**Stage 2: Classifier training**
- The concept layer is frozen; the final linear classifier is trained with cross-entropy loss and sparsity regularization
$$\mathcal{L}_{cls} = -\frac{1}{K}\sum_k y_k\log\hat{y}_k + \lambda[(1-\alpha)\frac{1}{2}\|W_A\|_F + \alpha\|W_A\|_{1,1}]$$
- L1 regularization promotes weight sparsity, enhancing interpretability

## Key Experimental Results

### Main Results

**Table 1: Video Action Recognition Performance (Top-1 Accuracy %)**

| Method | KTH | Penn Action | HAA-100 | UCF-101 |
|:---:|:---:|:---:|:---:|:---:|
| Non-explainable baseline | 89.7 | 97.8 | 73.5 | 88.4 |
| CBM + UCF-101 attributes | - | - | - | 86.8 |
| LF-CBM + entangled language concepts | 87.4 | 96.3 | 66.5 | 85.5 |
| LF-CBM + disentangled language concepts | 89.9 | 97.7 | 65.3 | 83.7 |
| **DANCE** | **91.1** | **98.1** | **70.7** | **87.5** |

Key findings:
- DANCE **surpasses** the non-explainable baseline on KTH and Penn Action (+1.4 and +0.3)
- Only marginal drops on HAA-100 and UCF-101 (−2.8 and −0.9)
- DANCE consistently outperforms language-concept-based CBM variants across all datasets

**User Study Results (Figure 6)**

| Compared Method | DANCE Better | Comparable | Other Better |
|:---:|:---:|:---:|:---:|
| vs GPT-4o concept CBM | >70% | ~20% | <10% |
| vs VTCD (saliency method) | >70% | ~20% | <10% |
| vs Expert-defined concepts | >70% | ~15% | <15% |

Motion dynamics concept interpretability score: Ours **4.3/5**, language-based method 2.3/5, expert-defined concepts 3.4/5.

### Ablation Study

**Cross-domain model editing experiment (Figure 10)**

Under severe domain shift (UCF-101 → UCF-101-SCUBA):
- Adjusting concept weights for 3 classes improves accuracy from 77.7% to **82.0%** (+4.3%)
- No retraining required

**Sample-level intervention (Figure 9)**

- Deactivating irrelevant scene concepts (e.g., "Table Tennis Club") corrects erroneous predictions
- Demonstrates DANCE's support for fine-grained, transparent prediction control

**Temporal direction sensitivity check (Figure 7)**

- A forward video is predicted as "Bowing FullBody"; the reversed video is predicted as "Burpee"
- Validates that the model genuinely relies on motion dynamics concepts rather than spatial context alone

### Key Findings

1. **Cleaner concepts yield better performance**: replacing language descriptions of motion with pose sequences consistently outperforms language-concept methods across all datasets
2. **Interpretability and accuracy need not conflict**: DANCE even improves performance on KTH and Penn Action
3. **Motion dynamics concepts are the most intuitive**: 89.7% of user study participants rated them 4 or 5 out of 5
4. **Supports training-free model debugging**: concept weight editing recovers performance under domain shift without retraining

## Highlights & Insights

1. **Representing motion concepts via pose sequences is the key innovation**: this circumvents the tacit knowledge problem of motion — rather than describing motion verbally, the approach directly visualizes pose sequences
2. **Fully automated concept discovery**: motion concepts are discovered through clustering; object/scene concepts are extracted via LLM, requiring no manual annotation
3. **Ante-hoc explainability design**: explanations are not post-hoc rationalizations — the model itself makes predictions through concepts, guaranteeing faithfulness
4. **Practical model debugging capability**: the editability of concept weights makes model debugging and domain adaptation straightforward

## Limitations & Future Work

1. Relies on the quality of the 2D pose estimator; inaccurate estimates degrade motion concept quality
2. Applicable only to human-centric action recognition; not suitable for non-human actions (e.g., natural phenomena)
3. The linear concept layer may limit the modeling of complex inter-concept interactions
4. The number of concepts (especially motion concepts) depends on clustering hyperparameter selection
5. A ~0.9% performance gap remains on UCF-101; scalability to large-scale datasets requires further investigation

## Related Work & Insights

- **Concept Bottleneck Models (CBM)** [Koh et al., 2020]: foundation for the concept layer design in this work
- **Label-Free CBM** [Oikarinen et al., 2023]: automated concept discovery via LLM; DANCE's object/scene concept discovery builds upon this
- **VTCD** [Kowal et al., 2024]: optimization-based video concept discovery, requiring additional computational overhead
- **Saliency Tubes** [Stergiou et al., 2019]: 3D saliency method, but produces entangled explanations

## Rating

- **Novelty**: ★★★★★ — The definition and discovery of motion dynamics concepts is a pioneering contribution to video XAI
- **Technical Depth**: ★★★★☆ — The concept bottleneck framework is well-established; innovation lies primarily in concept definition and the discovery pipeline
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation across 4 datasets, user studies, ablations, and model editing experiments
- **Writing Quality**: ★★★★★ — Polished figures, clear narrative structure; a well-deserved Spotlight paper
- **Value**: ★★★★☆ — Model debugging and editing capabilities have direct practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[NeurIPS 2025\] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[NeurIPS 2025\] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

</div>

<!-- RELATED:END -->
