---
title: >-
  [Paper Note] Confidence Self-Calibration for Multi-Label Class-Incremental Learning
description: >-
  [ECCV2024][Graph Learning][multi-label class-incremental learning] To address the overconfident predictions and false-positive errors caused by partial labels in Multi-Label Class-Incremental Learning (MLCIL), a Confidence Self-Calibration (CSC) framework is proposed. It calibrates label relationships using a Class-Incremental Graph Convolutional Network (CI-GCN) and calibrates confidence via max-entropy regularization, significantly outperforming SOTA methods on MS-COCO and…
tags:
  - "ECCV2024"
  - "Graph Learning"
  - "multi-label class-incremental learning"
  - "confidence calibration"
  - "graph convolutional network"
  - "max-entropy regularization"
  - "partial label"
date: 2026-05-08
content_hash: e157ab8d70c28951
---

# Confidence Self-Calibration for Multi-Label Class-Incremental Learning

**Conference**: ECCV2024  
**arXiv**: [2403.12559](https://arxiv.org/abs/2403.12559)  
**Authors**: Kaile Du, Yifan Zhou, Fan Lyu, Yuyang Li, Chen Lu, Guangcan Liu (Southeast University, Institute of Automation, Chinese Academy of Sciences)  
**Area**: Graph Learning  
**Keywords**: multi-label class-incremental learning, confidence calibration, graph convolutional network, max-entropy regularization, partial label

## TL;DR

To address the overconfident predictions and false-positive errors caused by partial labels in Multi-Label Class-Incremental Learning (MLCIL), a Confidence Self-Calibration (CSC) framework is proposed. It calibrates label relationships using a Class-Incremental Graph Convolutional Network (CI-GCN) and calibrates confidence via max-entropy regularization, significantly outperforming SOTA methods on MS-COCO and VOC.

## Background & Motivation

The core challenge of Multi-Label Class-Incremental Learning (MLCIL) is the **task-level partial label problem**: in each incremental task, only new classes of the current task are annotated, while past and future class labels are missing. This inherently results in disjoint new and old label spaces.

Existing methods (such as AGCN, KRT) overlook a key phenomenon: **the model outputs overconfident prediction distributions under the partial-label setting**, producing many false-positive errors. For example, even if a test image only contains "person", the model might output high confidence for the old class "dog". As the label space continuously expands, this overconfidence issue exacerbates catastrophic forgetting.

The authors' motivation is straightforward: since the issue lies in confidence calibration, it should be addressed simultaneously at both levels of **label relationship calibration** and **confidence calibration**.

## Core Problem

1. **Disrupted Label Relationships**: Under the partial label setup, complete label co-occurrence statistics are unattainable, making cross-task label relationships difficult to construct.
2. **Overconfident Confidence**: In the absence of labels, the model easily confuses new and old class features. The output distribution presents a multi-modal overconfident state, where precision is far lower than recall, leading to a persistently high false-positive rate.
3. **Exacerbated Catastrophic Forgetting**: The superposition of the above two factors leads to a severe degradation of performance on old classes.

## Method

### Overall Architecture CSC

CSC comprises two major components:

**(1) Class-Incremental Graph Convolutional Network (CI-GCN)** — Label Relationship Calibration

CI-GCN is a two-layer stacked GCN structure that does not rely on prior statistical information:

1. **General GCN**: Uses a learnable general correlation matrix (CM) $A_g$ to automatically learn cross-task label relationships through gradient updates. $A_g$ is divided into the old-task part $A_g^{1:t-1}$ (inherited from previous tasks to preserve old label relationships) and the new-task part $A_g^t$ (establishing relationships for the new label space). The key innovation is that CM updates via gradient descent utilizing both ground-truth and pseudo-labels jointly, preventing the error accumulation of fixed statistical matrices.

2. **Specific GCN**: Adaptively generates an image-specific CM $A_s$ unique to each image from the output $V_1$ of the General GCN. Specifically, it applies global pooling and convolution on $V_1$ to obtain global feature $v$, and computes $A_s = \sigma(V_1' W)$ through a convolutional layer after concatenation. This provides sample-level fine-grained label relationships.

The graph nodes $V_0$ are decoupled from the feature map $F$ extracted by the CNN backbone and the class activation map $M$: $V_0 = M^\top \otimes F$. The two-layer GCN computes:

$$V_1 = \text{LReLU}(A_g V_0 W_g), \quad V_2 = \text{LReLU}(A_s V_1 W_s)$$

**(2) Max-Entropy Regularization** — Confidence Calibration

Observing that the model's output distribution is overconfident (low entropy), the authors quantify the uncertainty of old-class predictions using Shannon entropy:

$$H = -\mathbb{E}_{c \in \mathcal{C}^{1:t-1}} [\hat{y}_c^t \log(\hat{y}_c^t)]$$

During training, taking the negative sign realizes max-entropy regularization, punishing overconfident output distributions:

$$L = L_{\text{cls}} - \beta H$$

where $L_{\text{cls}}$ integrates cross-entropy (for new classes) and knowledge distillation (for old classes), and $\beta$ controls the regularization intensity. The final prediction fuses the classifier output and the graph representation: $\hat{y}^t = \hat{y}_{\text{cls}}^t + \hat{y}_{\text{gcn}}^t$.

### Key Designs

- Learnable General CM rather than statistically fixed, avoiding accumulated errors from pseudo-labels.
- Specific CM is adaptively generated for each image, introducing more flexibility in handling rare label combinations.
- Both types of CM scale automatically as the number of classes increases, requiring no manual adjustments.
- Max-entropy regularization acts only on old classes, target-reducing false positives.

## Key Experimental Results

### MS-COCO 2014

| Setting | Method | Buffer | Last mAP | CF1 | OF1 |
|------|------|--------|----------|-----|-----|
| B0-C10 | KRT (SOTA) | 0 | 65.9 | 55.6 | 56.5 |
| B0-C10 | **CSC** | 0 | **72.8** | **64.9** | **66.8** |
| B0-C10 | KRT-R | 5/class | 68.3 | 60.0 | 61.0 |
| B0-C10 | **CSC-R** | 5/class | **73.7** | **67.3** | **68.1** |
| B0-C10 | **CSC-R** | 20/class | **74.8** | **67.8** | **68.6** |

CSC without a buffer (72.8%) even surpasses the performance of KRT-R using a 20/class buffer (70.2%).

### PASCAL VOC 2007

| Setting | Method | Buffer | Last mAP | Avg. mAP |
|------|------|--------|----------|----------|
| B0-C4 | KRT-R | 2/class | 83.4 | 90.7 |
| B0-C4 | **CSC-R** | 2/class | **87.9** | **92.4** |
| B4-C2 | AGCN-R | 2/class | 59.3 | 74.3 |
| B4-C2 | **CSC-R** | 2/class | **86.6** | **90.4** |

CSC-R outperforms AGCN-R by 27.3% in the most challenging B4-C2 scenario, demonstrating exceptional robustness.

### Ablation Study

| Component | mAP (B0-C10) | CF1 | OF1 |
|------|-------------|-----|-----|
| Baseline (KD only) | 42.4 | 45.3 | 43.7 |
| + Max-Entropy | 47.6 | 50.3 | 49.5 |
| + CI-GCN | 69.3 | 59.0 | 59.5 |
| + Combination of both (CSC) | **72.8** | **64.9** | **66.8** |

CI-GCN contributes the most (mAP +26.9%), and Max-Entropy further improves performance by 3.5% on top of it. Max-Entropy reduces the false-positive rate from 35% to 19%.

### CM Structure Ablation

The G → S (Softmax) combination is optimal (mAP 72.8%), outperforming the fixed statistical CM Z → Z (64.1%), validating the advantage of the learnable CM.

## Highlights & Insights

- **Insightful Problem Analysis**: It is the first to explicitly point out the connection between overconfident output distributions and false-positive errors in MLCIL, and proposes a solution from the perspective of calibration.
- **Exquisite CI-GCN Design**: The dual-layer General + Specific structure calibrates label relationships from macro to fine-grain, and the CM can be learned and expanded to avoid building up statistical errors.
- **Simple and Effective Max-Entropy Regularization**: Merely one additional regularization term significantly reduces the false-positive rate (35% → 19%), demonstrating orthogonal improvements across different methods.
- **Overwhelming Experimental Results**: CSC without a buffer even outperforms SOTA methods with a buffer of 20/class.
- **Strong Robustness**: Extremely small performance fluctuations across different scenarios (incremental step sizes).

## Limitations & Future Work

- Only validated on MS-COCO (80 classes) and VOC (20 classes); larger-scale datasets (such as Open Images) have not been tested.
- The backbone network is based on CNN (TResNetM); the adaptation of CI-GCN under Vision Transformer architectures has not been explored.
- Max-entropy regularization only operates on old classes, and confidence calibration for new classes remains uninvestigated.
- Random initialization of the General CM might affect the performance of the first task, suggesting the need for exploring better initialization strategies.
- The class incremental order is fixed lexicographically, and the sensitivity to random or hard-first ordering is not analyzed.

## Related Work & Insights

| Method | Label Relationship Modeling | Confidence Calibration | Mechanism |
|------|------------|-----------|---------|
| AGCN | Fixed statistical CM + pseudo-labels | None | Constructing a fixed graph with pseudo-labels |
| KRT (Prev. SOTA) | Knowledge recovery and transfer tokens | None | old/new knowledge token framework |
| **CSC (Ours)** | Learnable dual-layer CM (CI-GCN) | Max-Entropy regularization | Label relationship calibration + confidence calibration |

Compared with KRT, CSC offers advantages in: (1) CM is learned through gradients instead of being statically fixed, yielding stronger adaptability; (2) it explicitly addresses the overconfident confidence issue overlooked by KRT. Compared with L2P (ViT-B/16, 86M parameters), CSC achieves better results with fewer parameters (TResNetM, 29.4M).

## Related Work & Insights

- **Universality of Calibration Perspective**: The overconfidence problem is prevalent in incremental learning. The Max-Entropy concept can be transferred to tasks such as single-label CIL and incremental semantic segmentation.
- **Learnable Graph Structures**: The design concept in CI-GCN where the CM is learned via gradients can be extended to other scenarios that require dynamic relational graph construction.
- **Complementarity with Focal Loss**: Focal Loss focuses on hard samples, while Max-Entropy regularization focuses on calibration. The two might be complementary.
- **Comparison with Label Smoothing**: Both adjust the "sharpness" of the output distribution, but Max-Entropy directly optimizes information entropy, which is theoretically more direct.

## Rating

- Novelty: ⭐⭐⭐⭐ (The calibration perspective is novel for MLCIL, and the CI-GCN design is reasonable)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, multiple scenarios, detailed ablation studies, and comprehensive visualizations)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem formulation and abundant charts)
- Value: ⭐⭐⭐⭐ (An important advancement in the MLCIL direction; the calibration perspective is highly inspiring)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[ECCV 2024\] SENC: Handling Self-collision in Neural Cloth Simulation](senc_handling_self-collision_in_neural_cloth_simulation.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](../../AAAI2026/graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[ICLR 2026\] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs](../../ICLR2026/graph_learning/self-consistency_improves_the_trustworthiness_of_self-interpretable_gnns.md)

</div>

<!-- RELATED:END -->
