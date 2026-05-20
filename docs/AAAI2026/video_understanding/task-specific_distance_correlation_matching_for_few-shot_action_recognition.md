---
title: >-
  [Paper Note] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition
description: >-
  [AAAI 2026][Video Understanding][Few-Shot Action Recognition] This paper proposes TS-FSAR, a framework that employs α-distance correlation to capture nonlinear inter-frame dependencies and combines task-specific matching…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Few-Shot Action Recognition"
  - "Distance Correlation"
  - "CLIP Fine-tuning"
  - "Task-Specific Matching"
  - "Side-Tuning"
date: 2026-05-08
content_hash: 59df263621c18222
---

# Task-Specific Distance Correlation Matching for Few-Shot Action Recognition

**Conference**: AAAI 2026
**arXiv**: [2512.11340](https://arxiv.org/abs/2512.11340)  
**Code**: None  
**Area**: Video Understanding / Few-Shot Learning
**Keywords**: Few-Shot Action Recognition, Distance Correlation, CLIP Fine-tuning, Task-Specific Matching, Side-Tuning

## TL;DR

This paper proposes TS-FSAR, a framework that employs α-distance correlation to capture nonlinear inter-frame dependencies and combines task-specific matching matrices for query-support matching. An adapted frozen CLIP guides the training of a ladder side network, achieving substantial improvements over prior methods on temporally sensitive datasets such as SSv2-Full.

## Background & Motivation

**Background**: Few-shot action recognition (FSAR) aims to recognize novel action categories from only a few labeled samples. Mainstream approaches follow two lines: designing better metrics (e.g., set-based matching) and efficiently adapting large-scale pretrained models (e.g., CLIP). Set-matching methods such as HyRSM (Hausdorff distance) and TSAM (optimal transport) have demonstrated competitive performance.

**Limitations of Prior Work**: (1) **Metric limitation**: Existing set-matching methods universally rely on cosine similarity to construct inter-frame relation matrices; however, cosine similarity is approximately equivalent to the Pearson correlation coefficient and can only capture linear dependencies, failing to model more complex nonlinear relationships. (2) **Matching paradigm**: Existing methods perform matching using only instance-level information, ignoring task-specific contextual cues. (3) **CLIP adaptation**: Parameter-efficient side-network tuning (e.g., skip-fusion layers in EMP-Net) reduces memory consumption, but newly introduced layers are difficult to optimize under limited data.

**Key Challenge**: Videos contain complex nonlinear temporal dynamics between frames (especially in temporally sensitive datasets such as SSv2), which cosine similarity cannot capture; side-network tuning is memory-efficient but training-unstable, particularly on datasets dominated by static appearance where pretrained weights are more critical.

**Goal**: (1) Design a matching metric that captures both linear and nonlinear inter-frame relationships; (2) incorporate task-specific information for more accurate matching; (3) improve the training effectiveness of side networks under limited data.

**Key Insight**: Replace cosine similarity with α-distance correlation to measure inter-frame dependencies (capturing statistical dependencies of arbitrary order); generate a matching matrix from task prototypes to weight the importance of inter-frame relationships; use the output distribution of an adapted frozen CLIP to guide side-network learning.

**Core Idea**: Model nonlinear inter-frame dependencies via α-distance correlation, perform task-specific matching driven by task prototypes, and guide side-network training through CLIP knowledge distillation.

## Method

### Overall Architecture

TS-FSAR comprises three components. (1) **Ladder Side Network (LSN)**: A lightweight side network that receives intermediate features from a frozen CLIP and outputs frame-level tokens for metric computation. (2) **Task-Specific Distance Correlation Matching (TS-DCM)**: Composed of inter-frame α-distance correlation (IF-DαC) and task-specific matching (TSM), computing query-support similarity. (3) **Guiding LSN with Adapted CLIP (GLAC)**: Aligns the LSN output distribution with that of an adapted frozen CLIP to stabilize training.

### Key Designs

1. **Inter-Frame α-Distance Correlation (IF-DαC)**:

    - **Function**: Computes a comprehensive dependency matrix between query and support video frames.
    - **Mechanism**: Given the $i$-th support frame feature $\mathbf{V}_\mathcal{S}^i$ and $j$-th query frame feature $\mathbf{V}_\mathcal{Q}^j$ output by LSN, each column is treated as an observation of a random variable. The α-power Euclidean distance matrix $\hat{a}_{kl} = \|\mathbf{x}_k - \mathbf{x}_l\|^\alpha$ is computed and doubly centered to obtain α-D matrices $\mathbf{A}^i, \mathbf{B}^j$. The α-distance correlation is then $m_{ij} = \text{tr}(\mathbf{A}^i \mathbf{B}^j) / \sqrt{\text{tr}(\mathbf{A}^i \mathbf{A}^i) \text{tr}(\mathbf{B}^j \mathbf{B}^j)}$. The parameter $\alpha \in (0,2)$ controls sensitivity to dependencies at different scales; experiments set $\alpha = 0.8$.
    - **Design Motivation**: A fundamental property of distance correlation is that it equals zero if and only if two random variables are independent, thereby capturing arbitrary (including nonlinear) dependencies. By contrast, cosine similarity (≈ Pearson correlation) is sensitive only to linear relationships. Distinguishing actions in temporally sensitive datasets such as SSv2 depends on fine-grained nonlinear temporal patterns.

2. **Task-Specific Matching (TSM)**:

    - **Function**: Generates a matching matrix encoding the relative importance of inter-frame relationships, enabling task-aware matching.
    - **Mechanism**: A query-specific task prototype is first constructed as $\mathbf{p}^\mathcal{T} = \tilde{\mathbf{v}}^\mathcal{Q} + \frac{1}{N_\mathcal{S}} \sum \tilde{\mathbf{v}}_i^\mathcal{S}$ (average fusion of query and support class tokens). The prototype is then fed into a learnable linear generator $\mathcal{G}(\cdot)$ to produce a $T \times T$ matching matrix $\mathbf{M}^{task}$. The final similarity score is $\langle \mathbf{M}^{task}, \mathbf{M}^{IF\text{-}D^\alpha C} \rangle$ (inner product).
    - **Design Motivation**: The IF-DαC matrix provides dependency strengths for all frame pairs, but the importance of different pairs should vary by task. Weighting these relationships via a task-prototype-driven matching matrix directs attention toward inter-frame patterns most relevant to the current task. Experiments show that simple average fusion outperforms concatenation and cross-attention.

3. **GLAC Module (Guiding LSN with Adapted CLIP)**:

    - **Function**: Uses the output distribution of an adapted frozen CLIP to guide LSN training.
    - **Mechanism**: On the LSN side, the mean of the video-level α-D matrix serves as the representation $\widetilde{\mathbf{A}}_{\alpha\text{-}D}$; a softmax over its inner product with learnable class α-D prototype weights yields the prediction distribution $\mathbf{p}$. On the CLIP side, frame-level CLS tokens from the frozen CLIP are processed by an MHSA adapter to model inter-frame dependencies, averaged, and compared with text embeddings via cosine similarity to obtain the guiding distribution $\mathbf{q}$. Training minimizes the KL divergence $\text{KL}(\mathbf{p} \| \mathbf{q})$ along with cross-entropy losses for both branches.
    - **Design Motivation**: LSN is a newly introduced lightweight network with randomly initialized parameters that is difficult to train under few-shot conditions. Using pretrained CLIP knowledge—adapted to the video domain via adapters—as a "teacher" to guide LSN learning makes the α-D features more reliable, thereby improving downstream distance correlation estimation.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{LSN} + \lambda_1 \mathcal{L}_{TS\text{-}DCM} + \lambda_2 \mathcal{L}_{GLAC}$, where $\mathcal{L}_{LSN}$ is the visual-language alignment cross-entropy between LSN outputs and text embeddings, $\mathcal{L}_{TS\text{-}DCM}$ is the episodic matching cross-entropy, and $\mathcal{L}_{GLAC} = \text{KL}(\mathbf{p} \| \mathbf{q}) + \text{CE}(\mathbf{p}, y) + \text{CE}(\mathbf{q}, y)$. AdamW optimizer with cosine learning rate scheduling is adopted.

## Key Experimental Results

### Main Results

| Dataset | Setting | Ours | Prev. SOTA | Gain |
|--------|------|---------|---------|------|
| SSv2-Full | 1-shot | **75.1** | 66.7 (D2ST) | **+8.4** |
| SSv2-Full | 5-shot | **83.5** | 81.9 (D2ST) | +1.6 |
| SSv2-Small | 1-shot | 60.5 | 60.5 (TSAM) | — |
| SSv2-Small | 5-shot | **70.3** | 69.3 (D2ST) | +1.0 |
| HMDB51 | 1-shot | **85.0** | 84.5 (TSAM) | +0.5 |
| HMDB51 | 5-shot | **88.9** | 88.9 (TSAM) | — |
| UCF101 | 1-shot | **98.7** | 98.3 (TSAM) | +0.4 |
| Kinetics | 1-shot | **96.3** | 96.2 (TSAM) | +0.1 |

### Ablation Study

| Configuration | SSv2-Full 1-shot | HMDB51 1-shot | Note |
|------|-----------------|---------------|------|
| Zero-shot CLIP | 37.0 | 75.9 | Baseline |
| + LSN | 67.1 | 77.7 | Gain from side-network tuning |
| + LSN + IF-DαC | 71.4 | 82.1 | α-distance correlation +4.3% |
| + LSN + IF-DαC + TSM | 73.8 | 83.4 | Task-specific matching +2.4% |
| + Full TS-FSAR | **75.1** | **85.0** | GLAC guidance +1.3–1.6% |

### Key Findings

- The large gain on SSv2-Full (+8.4%) results from two compounding factors: (1) the dataset contains fine-grained temporal variations where α-DC captures nonlinear patterns; (2) its training set is approximately 10× larger than other datasets, providing more supervision for LSN training.
- Nonlinear metrics (α-DC, DC, HSIC) consistently outperform cosine similarity, with α-DC achieving the best results.
- GLAC yields larger improvements on static datasets (HMDB51), confirming that insufficient LSN training primarily affects static tasks.
- Efficiency: 14M parameters, ~9 GB GPU memory, 0.42 s/episode—significantly more efficient than CLIP-FSAR (89M, ~20 GB).

## Highlights & Insights

- Introducing distance correlation into the metric design of few-shot action recognition represents an important upgrade over cosine similarity.
- The task-specific matching approach—generating inter-frame importance weights from task prototypes—is simple yet effective and generalizable to other set-matching scenarios.
- The GLAC module addresses the common "difficulty in training newly introduced layers" problem in side-network tuning and constitutes a general-purpose technique.
- The 8.4% 1-shot improvement on SSv2-Full is substantial, demonstrating the considerable potential of nonlinear metrics for temporally sensitive tasks.

## Limitations & Future Work

- Improvements on static datasets (UCF101, Kinetics) are limited, as these datasets rely more on CLIP pretraining knowledge than on temporal modeling.
- Computing IF-DαC involves constructing and doubly centering $(P+1) \times (P+1)$ distance matrices, incurring non-trivial computational overhead.
- The LSN feature dimension is fixed at 256; more flexible architectural designs remain unexplored.
- The simple averaging strategy for task prototype fusion may be insufficiently fine-grained in multi-shot scenarios.

## Related Work & Insights

- **vs. TSAM**: TSAM employs optimal transport for matching but still constructs cost matrices based on cosine similarity; TS-FSAR replaces cosine similarity with α-DC and adds task-specific matching.
- **vs. EMP-Net**: Both use side-network tuning for CLIP, but EMP-Net lacks a training guidance mechanism; the GLAC module in TS-FSAR yields an additional gain of 1.1%–8.2%.
- **vs. DeepBDC**: DeepBDC applies distance correlation in few-shot image classification; TS-FSAR extends this to video understanding by introducing inter-frame α-DC and task-specific matching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The introduction of α-distance correlation is novel; the task-specific matching design is well-motivated; the GLAC module addresses a practical pain point.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on five datasets with detailed ablations covering metric comparison, prototype fusion strategies, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear, mathematical formulations are precise, and ablation analysis is logically structured.
- **Value**: ⭐⭐⭐⭐ Achieves important breakthroughs in temporally sensitive few-shot action recognition; α-DC as a metric is generalizable to broader matching tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](../../ICCV2025/video_understanding/trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)

</div>

<!-- RELATED:END -->
