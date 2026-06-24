---
title: >-
  [Paper Note] Nonverbal Interaction Detection
description: >-
  [ECCV 2024][Object Detection][Nonverbal Interaction] This work presents the first systematic study of human nonverbal interaction (gestures, expressions, gaze, postures, touch), introducing a large-scale dataset NVI, a new task NVI-DET, and a dual multi-scale hypergraph-based detection model NVI-DEHR, which achieves state-of-the-art performance on both nonverbal interaction detection and HOI detection tasks.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Nonverbal Interaction"
  - "Hypergraph Learning"
  - "Social Intelligence"
  - "Human Detection"
  - "High-Order Relation Modeling"
date: 2026-05-08
content_hash: 99d2eeb85c358fb3
---

# Nonverbal Interaction Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.08133](https://arxiv.org/abs/2407.08133)  
**Code**: [Available](https://github.com/weijianan1/NVI)  
**Area**: Object Detection / Social Signal Processing  
**Keywords**: Nonverbal Interaction, Hypergraph Learning, Social Intelligence, Human Detection, High-Order Relation Modeling

## TL;DR

This work presents the first systematic study of human nonverbal interaction (gestures, expressions, gaze, postures, touch), introducing a large-scale dataset NVI, a new task NVI-DET, and a dual multi-scale hypergraph-based detection model NVI-DEHR, which achieves state-of-the-art performance on both nonverbal interaction detection and HOI detection tasks.

## Background & Motivation

Nonverbal behaviors (facial expression, gaze, gesture, posture, touch) account for nearly two-thirds of human social interaction, yet they receive far less attention in the AI field than linguistic analysis. Existing methods face the following core challenges:

**Isolated Studies**: Existing datasets and methods focus individually on a single type of nonverbal signal (e.g., RAF-DB only focuses on expressions, VACATION only on gaze, EgoGesture only on gestures), failing to capture the concurrence and correlation of multiple signals. For instance, gaze avoidance is often accompanied by angry/sad expressions and crossed arms.

**Data Limitations**: Some datasets are collected under controlled conditions (e.g., MSR-Action3D, UTKinect-Action), showing a significant gap from real-world social scenarios.

**Lack of a Unified Task**: Different types of interactions are described and studied at varying granularities (individual expressions, pairwise HOI, group gaze), lacking a unified framework.

**Insufficient Relation Modeling**: HOI detection mainly handles pairwise relations, whereas nonverbal interactions often involve multi-person high-order relations, requiring stronger relation reasoning capabilities.

The core insight is: all nonverbal signals—regardless of target type or number of participants—can be summarized as a combination of "individual behaviors" (individual actions influenced by others) and "collective behaviors" (joint group action). Thus, they can be formalized into a unified triplet $\langle \text{individual}, \text{group}, \text{interaction} \rangle$.

## Method

### Overall Architecture

NVI-DEHR is extended based on the DETR architecture and consists of four core modules: (1) a shared visual encoder to extract image features; (2) an instance decoder to detect individuals and social groups; (3) dual multi-scale hypergraphs to model high-order individual-to-individual and group-to-group relations; (4) an interaction decoder to predict nonverbal interaction categories.

### Key Designs

1. **NVI Dataset and Taxonomy**: A large-scale dataset containing 13,711 images, 49K+ human annotations, and 72K social interactions was constructed. Utilizing a hierarchical taxonomy, 22 atomic behaviors are defined under 5 major categories (gaze, touch, facial expression, gesture, posture) (e.g., gaze-following, mutual-gaze, handshake, smile, etc.). The dataset annotations include individual bounding boxes, social group bounding boxes, and interaction categories.

2. **Dual Multi-Scale Hypergraph**: This is the core innovation of the model. Conventional graphs can only model pairwise relations, whereas hyperedges in hypergraphs can connect an arbitrary number of vertices, making them naturally suited for high-order relation modeling.

    - Two hypergraphs are constructed: $\mathcal{G}_h$ (using individuals as vertices to model the relationships among individuals within the same group) and $\mathcal{G}_g$ (using groups as vertices to model relationships among related groups).
    - Multi-scale design: Each hypergraph comprises multiple scales from $s=1$ to $s=S$. When $s=1$, vertices are independent; when $s>1$, the $s$ most similar vertices are selected via an affinity matrix to form a hyperedge.
    - Affinity calculation: $A_{ij} = \mathbf{v}_i^\top \mathbf{v}_j / \|\mathbf{v}_i\| \|\mathbf{v}_j\|$
    - Hyperedge formation: $e_i^s = \arg\max_{\mathcal{O} \subseteq \mathcal{V}} \|A_{\mathcal{O},\mathcal{O}}\|_{1,1}$, s.t. $|\mathcal{O}|=s$ and $v_i \in \mathcal{O}$, solved via a greedy algorithm.
    - Design Motivation: The number of participants in social interactions is not fixed. The multi-scale design can capture various group scales ranging from 2 to 5 people. Research shows that in discussions with more than 10 people, 80% of the speech contributions come from 4-5 individuals.

3. **Hypergraph Convolutional Learning**: Information is propagated among vertices through $L$ hyperedge convolutional layers:

    $\mathbf{V}_h^{s,(l)} = (\mathbf{D}_{h,v}^s)^{-\frac{1}{2}} \mathbf{H}_h^s (\mathbf{D}_{h,e}^s)^{-1} \mathbf{H}_h^{s\top} (\mathbf{D}_{h,v}^s)^{-\frac{1}{2}} \mathbf{V}_h^{s,(l-1)} \theta_h^{s,(l)}$

   Finally, features of different scales are aggregated via an MLP: $\mathbf{F}_h = \text{MLP}([\mathbf{V}_h^{1,(L)}, \mathbf{V}_h^{2,(L)}, \ldots, \mathbf{V}_h^{S,(L)}])$

4. **Interaction Decoder**: Interaction queries $\mathbf{Q}_n = (\mathbf{F}_h + \mathbf{F}_g)/2$ are dynamically generated based on the high-order features after hypergraph learning. A Transformer decoder is then utilized to predict interaction categories. Dynamic query initialization is more effective than random initialization as it incorporates rich relational contextual information.

### Loss & Training

- End-to-end training is performed using bipartite matching via the Hungarian algorithm.
- Loss function: $\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{GIoU} + \lambda_3 \mathcal{L}_c$
    - $\mathcal{L}_1$ and $\mathcal{L}_{GIoU}$: Localization loss
    - $\mathcal{L}_c$: Focal loss (classification)
    - Coefficient settings: $\lambda_1=2.5, \lambda_2=1, \lambda_3=2$
- Utilizing a ResNet-50 backbone with default settings of $N=64$ queries, $C=256$ channels, $S=5$ scales, and $L=2$ convolutional layers.
- Trained for 90 epochs, with a learning rate of 1e-4 for the first 60 epochs, reduced to 1e-5 for the remaining 30 epochs.
- Evaluation metric: mR@K (mean Recall@K), averaged over IoU thresholds of {0.25, 0.5, 0.75}.

## Key Experimental Results

### Main Results — NVI-DET

| Method | mR@25 | mR@50 | mR@100 | AR (Average) |
|------|-------|-------|--------|----------|
| m-QPIC | 59.44 | 71.46 | 80.07 | 70.32 |
| m-CDN | 59.01 | 72.94 | 82.61 | 71.52 |
| m-GEN-VLKT | 56.68 | 74.32 | 84.18 | 71.72 |
| **NVI-DEHR (Ours)** | **59.46** | **76.01** | **88.52** | **74.67** |

NVI-DEHR achieves an AR of 74.67 on the test set, bringing a 2.95 improvement over the best baseline m-GEN-VLKT.

### Ablation Study — Number of Hypergraph Scales S

| S | mR@25 | mR@50 | mR@100 | AR |
|---|-------|-------|--------|----|
| 1 | 53.39 | 69.81 | 81.90 | 68.37 |
| 3 | 53.52 | 70.45 | 83.92 | 69.30 |
| 5 | **54.85** | **73.42** | **85.33** | **71.20** |
| 6 | 54.59 | 73.11 | 85.24 | 70.98 |

### Ablation Study — Number of Hyperedge Convolutional Layers L

| L | mR@25 | mR@50 | mR@100 | AR |
|---|-------|-------|--------|----|
| 0 (No Hypergraph) | 53.50 | 69.44 | 81.71 | 68.22 |
| 1 | 53.76 | 71.74 | 83.61 | 69.70 |
| 2 | **54.85** | **73.42** | **85.33** | **71.20** |
| 3 | 54.36 | 72.47 | 84.85 | 70.56 |

### HOI-DET Generalization Ability

| Method | HICO-DET Full | Rare | Non-Rare | V-COCO S1 | V-COCO S2 |
|------|-------------|------|----------|-----------|-----------|
| HOICLIP (Prev. SOTA) | 34.54 | 30.71 | 35.70 | 63.5 | 64.8 |
| **NVI-DEHR** | **35.30** | **31.43** | **36.64** | **64.1** | **65.3** |

### Key Findings

- CLIP-based method m-GEN-VLKT performs poorly on NVI-DET, indicating that transferring knowledge from vision-language models to nonverbal interaction detection is more challenging than to HOI.
- The multi-scale hypergraph brings continuous improvements from $S=1$ to $S=5$ (68.37 $\rightarrow$ 71.20 AR), yet slightly drops at $S=6$—consistent with findings in social psychology.
- $L=2$ layers of hypergraph convolution achieve the best balance, whereas excessive layers lead to noise propagation.
- The model also achieves SOTA on standard HOI-DET datasets, demonstrating the superior generalization ability of the hypergraph structure.

## Highlights & Insights

1. **Novelty of Problem Definition**: The first to study multiple nonverbal signals under a unified framework, proposing the triplet formulation of $\langle \text{individual}, \text{group}, \text{interaction} \rangle$, which is more expressive than the binary formulation of HOI.
2. **Natural Match of Hypergraphs**: Nonverbal interactions naturally involve an arbitrary number of participants. Hyperedges in hypergraphs can connect any number of vertices, making them more suitable than ordinary graphs.
3. **Value of Dataset**: NVI is the first comprehensive nonverbal interaction dataset covering 22 atomic behaviors across 5 major categories.
4. **Dual-Task Validation**: NVI-DEHR achieves SOTA on two different but related tasks, demonstrating the generalizability of the design.

## Limitations & Future Work

1. The **ambiguity and subtlety** of nonverbal signals remain major challenges (e.g., distinguishing mutual gaze vs. gaze aversion).
2. Individuals with severe occlusions are difficult to detect.
3. Long-tail distribution issue: rare behaviors such as beckon have extremely small sample sizes.
4. **Temporal information in videos** is not considered, leaving the understanding of dynamic nonverbal interactions to be explored.
5. The dataset is extended based on PIC 2.0, which may limit scene diversity.

## Related Work & Insights

- Key difference from HOI-DET: Nonverbal signals are more subtle, ambiguous, and involve multiple people, rather than just pairwise interactions.
- Hypergraph learning has been applied in human parsing and HOI detection; this work is the first to apply it to nonverbal interaction.
- Future direction: It can be combined with large language models to enable natural language description and reasoning of nonverbal signals.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of nonverbal interaction, with three major contributions: dataset, task, and model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual-task validation on NVI-DET and HOI-DET with thorough ablation studies, though comparisons with more visual relation detection methods are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-explained motivation, with social psychology background adding persuasiveness.
- **Value**: ⭐⭐⭐⭐ — Holds significant potential to advance fields such as social robotics, human-computer interaction, and affective computing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SL-HOI: Streamlined Open-Vocabulary Human-Object Interaction Detection](../../CVPR2026/object_detection/streamlined_open-vocabulary_human-object_interaction_detection.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[ICML 2025\] UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction](../../ICML2025/object_detection/ui-vision_a_desktop-centric_gui_benchmark_for_visual_perception_and_interaction.md)
- [\[ICML 2026\] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](../../ICML2026/object_detection/earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)
- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)

</div>

<!-- RELATED:END -->
