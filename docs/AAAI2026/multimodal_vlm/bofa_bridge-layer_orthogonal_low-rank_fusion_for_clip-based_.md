---
title: >-
  [Paper Note] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning
description: >-
  [AAAI 2026][Multimodal VLM][Class-incremental learning] This paper proposes BOFA, a framework that exclusively fine-tunes the existing cross-modal projection layer (bridge-layer) in CLIP. By constraining parameter updates within a low-rank "safe subspace" orthogonal to old-task features via Orthogonal Low-Rank Fusion, and combining this with cross-modal hybrid prototypes, BOFA achieves state-of-the-art exemplar-free class-incremental learning without introducing any additional parameters or inference overhead.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Class-incremental learning
  - CLIP
  - orthogonal low-rank fusion
  - cross-modal prototypes
  - catastrophic forgetting
date: 2026-05-08
content_hash: 7848c1cf6769c629
---

# BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning

**Conference**: AAAI 2026
**arXiv**: [2511.11421](https://arxiv.org/abs/2511.11421)
**Code**: Not released
**Area**: Multimodal VLM
**Keywords**: Class-incremental learning, CLIP, orthogonal low-rank fusion, cross-modal prototypes, catastrophic forgetting

## TL;DR

This paper proposes BOFA, a framework that exclusively fine-tunes the existing cross-modal projection layer (bridge-layer) in CLIP. By constraining parameter updates within a low-rank "safe subspace" orthogonal to old-task features via Orthogonal Low-Rank Fusion, and combining this with cross-modal hybrid prototypes, BOFA achieves state-of-the-art exemplar-free class-incremental learning without introducing any additional parameters or inference overhead.

## Background & Motivation

Class-incremental learning (CIL) requires models to continuously learn new categories without forgetting prior knowledge. Vision-language models such as CLIP provide a strong foundation for CIL through cross-modal representations. The prevailing paradigm freezes the CLIP backbone and introduces lightweight trainable modules (adapters/prompts). However, this strategy suffers from three limitations:

1. **Forgetting transferred, not eliminated**: Adapters themselves still overwrite old knowledge during sequential multi-task training; catastrophic forgetting is merely shifted from the backbone to the adapter.
2. **Extra parameters and inference overhead**: Although lightweight, these additional modules still increase model parameter count and inference latency.
3. **Limitations of text prototypes**: Text prototypes derived from hand-crafted prompts (e.g., "a photo of a [CLASS]") lack fine-grained discriminability, constraining classification performance.

## Core Problem

**How can CLIP be incrementally adapted without introducing any additional parameters while effectively preventing catastrophic forgetting?** Furthermore, how can visual and textual modality information be better fused to improve classification accuracy?

## Method

### Overall Architecture

BOFA comprises three synergistic components:

1. **Fine-tuning only the cross-modal bridge-layer**: The CLIP image encoder is structured as $g_i = g_2 \circ g_1$, where $g_1$ is the ViT backbone extracting $d_o$-dimensional visual features $\mathbf{x}_o$, and $g_2$ is a linear projection layer (weight $\mathbf{W} \in \mathbb{R}^{d_o \times d}$) mapping features into the shared embedding space. BOFA freezes $g_1$ and the text encoder $g_t$, fine-tuning only this existing projection layer $g_2$—**without introducing any external modules**—thereby preserving the original CLIP architecture and inference cost.

2. **Orthogonal Low-Rank Fusion**: Constrains parameter updates within the approximate null space of old-task features to prevent forgetting.

3. **Cross-Modal Hybrid Prototypes**: Fuses text prototypes with visual prototypes to enhance classification capability.

### Key Designs

#### Orthogonal Safe Subspace (OSS)

**Problem formulation**: After learning a parameter update $\Delta\mathbf{W}_{new}$ for a new task, the embedding of old-task features shifts from $\mathbf{X}_{old}\mathbf{W}_{old}$ to $\mathbf{X}_{old}(\mathbf{W}_{old} + \Delta\mathbf{W}_{new})$. The interference term $\mathbf{X}_{old}\Delta\mathbf{W}_{new}$ is the primary source of forgetting. Ideally, one requires $\mathbf{X}_{old}\Delta\mathbf{W}_{new} \approx \mathbf{0}$.

**Approximate null space construction**: Since high-dimensional feature matrices are typically full-rank, an exact null space does not exist. The authors therefore define an approximate null space as the subspace that minimizes the projected magnitude of old features, by minimizing the interference metric:

$$\mathbf{P}^* = \arg\min_{\mathbf{P}^\top\mathbf{P}=\mathbf{I}_k} \|\mathbf{X}_{old}\mathbf{P}\|_F^2$$

**Proposition 1**: The optimal solution $\mathbf{P}^*$ is spanned by the eigenvectors corresponding to the $k$ smallest eigenvalues of the cumulative scatter matrix $\mathbf{S}_{old} = \mathbf{X}_{old}^\top\mathbf{X}_{old}$ of old-task features.

**Incremental update**: The scatter matrix can be accumulated incrementally as $\mathbf{S}_{new} = \mathbf{S}_{old} + \mathbf{X}_{new}^\top\mathbf{X}_{new}$, requiring no storage of historical data.

#### LoRA within the Orthogonal Safe Subspace

The parameter update is decomposed as $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}$ (LoRA formulation), but $\mathbf{A} = \mathbf{P}^*$ (the OSS basis) is **fixed**, with only $\mathbf{B} \in \mathbb{R}^{k \times d}$ being trained. This guarantees that the row space of the update strictly lies within the safe subspace.

**Data-driven initialization**: Since $\mathbf{A}$ is frozen, zero-initializing $\mathbf{B}$ leads to optimization difficulties. The authors therefore first perform full fine-tuning of the bridge-layer to obtain an "oracle" update $\Delta\tilde{\mathbf{W}}_{new}$, then initialize via the closed-form solution: $\mathbf{B}_0 = \mathbf{P}^{*\top}\Delta\tilde{\mathbf{W}}_{new}$, serving as a safe and task-adaptive starting point.

#### Cross-Modal Hybrid Prototypes

**Static fusion**: For each class $c$, the text prototype $\mathbf{z}_t^c$ and visual prototype $\mathbf{z}_i^c$ are linearly interpolated: $\mathbf{p}_c = (1-\lambda)\mathbf{z}_t^c + \lambda\mathbf{z}_i^c$, where $\lambda$ is determined by grid search on the first task and then fixed.

**Dynamic refinement**: During training, visual prototypes for all seen classes are continuously updated via EMA to accommodate drift in the bridge-layer feature space. After the entire incremental sequence, all visual prototypes are recomputed using the final fused weight $\mathbf{W}_{fused}$.

**Hierarchical inference**: A two-stage classification procedure — a lightweight task-level auxiliary classifier (trained on high-dimensional features $\mathbf{x}_o$) first narrows the candidate class set, after which the hybrid prototypes perform fine-grained classification over the reduced candidates.

### Loss & Training

- Standard cross-entropy loss is applied based on CLIP cosine-similarity classification probabilities (Eq. 1).
- Learning rate starts at 0.05 with cosine annealing decay.
- Upon arrival of each new task: (1) construct the OSS from the smallest eigenvectors of the old-task scatter matrix; (2) perform full fine-tuning of the bridge-layer to obtain the oracle update for initialization; (3) freeze $\mathbf{A}=\mathbf{P}^*$ and train only $\mathbf{B}$; (4) update the scatter matrix and prototypes after fusion.

## Key Experimental Results

Evaluated on 9 benchmark datasets using the B-m Inc-n protocol, with CLIP ViT-B/16 (LAION-400M) as the backbone on an RTX 4090 GPU.

**Table 1 Main Results** ($\bar{\mathcal{A}}$ / $\mathcal{A}_B$, average accuracy / final accuracy):

| Dataset | BOFA | RAPF | SimpleCIL | PROOF (exemplar) |
|---|---|---|---|---|
| Aircraft B0 Inc10 | **69.94 / 59.67** | 50.38 / 23.61 | 59.24 / 48.09 | - |
| CIFAR100 B0 Inc10 | **86.50 / 79.34** | 86.14 / 78.04 | 84.15 / 76.63 | - |
| Cars B0 Inc10 | **93.77 / 89.23** | 82.89 / 62.85 | 92.04 / 86.85 | - |
| ImageNet-R B0 Inc20 | **85.42 / 79.62** | 81.26 / 70.48 | 81.06 / 74.48 | - |
| CUB200 B0 Inc20 | **86.09 / 79.10** | 79.09 / 62.77 | 83.81 / 77.52 | - |
| UCF101 B0 Inc10 | **93.22 / 88.08** | 92.28 / 80.33 | 90.44 / 85.68 | - |
| SUN397 B0 Inc30 | **85.62 / 78.87** | 82.13 / 72.47 | 82.13 / 75.58 | - |
| Food101 B0 Inc10 | **89.01 / 82.74** | 88.57 / 81.15 | 87.89 / 81.65 | - |
| ObjectNet B0 Inc20 | **58.04 / 45.14** | 48.67 / 27.43 | 52.06 / 40.13 | - |

BOFA achieves the best performance on all 9 datasets, surpassing RAPF by **19.56%** in $\bar{\mathcal{A}}$ on Aircraft, **9.37%** on ObjectNet, and **10.88%** on Cars.

**Table 2 vs. Methods with Exemplar Storage** (10 exemplars per class):

| Method | Avg. $\bar{\mathcal{A}}$ over 7 datasets |
|---|---|
| BOFA (no exemplars) | **78.87** |
| PROOF (with exemplars) | 74.66 |
| MEMO (with exemplars) | 66.00 |
| iCaRL (with exemplars) | 66.17 |

Without using any historical exemplars, BOFA still outperforms PROOF (which stores 10 exemplars per class) by **4.21%**.

### Ablation Study

- **Orthogonal low-rank fusion ablation** (Figure 3): Fine-tuning the bridge-layer alone (FT) outperforms standard LoRA, indicating that naive low-rank constraints are neither sufficiently adaptive nor anti-forgetting; RAPF adapted to the bridge-layer also underperforms BOFA; BOFA simultaneously achieves the adaptability of FT and the stability of orthogonal constraints.
- **t-SNE visualization** (Figure 4, CIFAR100 B0 Inc5): Without fusion, old-class features are highly entangled; after applying orthogonal low-rank fusion, new and old class features separate more clearly and align better with the prototypes.

## Highlights & Insights

1. **Zero additional parameters**: The method leverages CLIP's existing projection layer rather than introducing adapters, achieving truly zero parameter growth and zero inference overhead increase.
2. **Theoretically grounded anti-forgetting mechanism**: The orthogonal safe subspace is rigorously defined (Proposition 1), constraining updates to directions corresponding to the smallest eigenvalues of the old-task scatter matrix — both intuitive and mathematically elegant.
3. **Incremental scatter matrix maintenance**: $\mathbf{S}_{new} = \mathbf{S}_{old} + \mathbf{X}_{new}^\top\mathbf{X}_{new}$ requires no storage of historical data, fully consistent with the exemplar-free setting.
4. **Data-driven LoRA initialization**: The two-step procedure of full fine-tuning followed by projection onto the safe subspace balances plasticity and stability.
5. **Comprehensive state-of-the-art across all 9 datasets**, even surpassing methods that rely on exemplar storage.

## Limitations & Future Work

1. **Scatter matrix storage**: Maintaining a $d_o \times d_o$ scatter matrix (approximately 2.4 MB for ViT-B/16 where $d_o=768$), per-class mean features, and auxiliary classifiers introduces cumulative storage overhead as the number of tasks grows, though the authors claim greater efficiency than RAPF's per-class covariance matrices.
2. **Hierarchical inference complexity**: Two-stage classification requires additional task-level auxiliary classifiers, whose count grows linearly with the number of tasks.
3. **Approximate nature of the safe subspace**: The orthogonal subspace only approximately eliminates interference; when the rank $k$ is poorly chosen or the number of tasks is large, available "safe directions" may be exhausted.
4. **CLIP-specific applicability**: The method relies on CLIP's unique bridge-layer structure; generalization to other VLMs or architectures is not discussed.
5. **Hyperparameter sensitivity**: Both $k$ (low-rank dimension) and $\lambda$ (modality mixing coefficient) require tuning; $\lambda$ is fixed via grid search on the first task but may need different settings across datasets.
6. **Lack of large-scale experiments**: The largest dataset evaluated contains only 300 classes (SUN397); performance at larger scales (e.g., ImageNet-1K with 1,000 classes) remains unverified.

## Related Work & Insights

| Method | Extra Params | Anti-Forgetting Strategy | Exemplar Storage | Modality Usage |
|---|---|---|---|---|
| L2P/DualPrompt | Yes (prompts) | Prompt pool selection | No | Vision only |
| CODA-Prompt | Yes (prompts) | Attention composition | No | Vision only |
| CoOp | Yes (prompts) | No special handling | No | Text prompts |
| MOE-Adapter | Yes (adapter+MoE) | Expert selection | No | Cross-modal |
| PROOF | Yes (projection heads) | Task-specific heads | Optional | Cross-modal |
| RAPF | Yes (adapter) | Adaptive fusion | No | Cross-modal |
| **BOFA** | **None** | Orthogonal safe subspace | No | Cross-modal hybrid prototypes |

The core innovation of BOFA lies in **restricting adaptation to existing layers rather than introducing new modules**, a design philosophy that is relatively novel in the CIL literature. The most direct comparison is with RAPF — while RAPF also performs parameter fusion, it requires additional adapters and maintains per-class covariance matrices ($|\mathcal{Y}| \cdot d^2$), whereas BOFA requires only a single global scatter matrix ($d_o^2$), yielding greater efficiency.

**Broader implications:**

1. The **orthogonal subspace constraint** can be generalized to other fine-tuning scenarios requiring anti-forgetting, such as continual instruction tuning of large language models.
2. The **bridge-layer insight** merits attention — CLIP's linear projection layer, though simple, possesses sufficient plasticity to accommodate downstream task knowledge.
3. **Eigenvalue analysis of the scatter matrix** provides a quantitative tool for understanding inter-task interference and can be used to analyze task similarity.
4. The cross-modal hybrid prototype strategy is applicable to other open-world recognition tasks requiring fusion of textual and visual information.

## Rating

- **Novelty**: 8/10 — The combination of orthogonal safe subspace and bridge-layer fine-tuning is creative, though orthogonal-constraint-based anti-forgetting has precedents (e.g., OWM).
- **Technical Depth**: 8/10 — Mathematical derivations are complete (Proposition 1 is proven) and the LoRA initialization strategy is clever, though the core intuition (minimal eigenvalue directions cause less interference) is relatively straightforward.
- **Experimental Thoroughness**: 8/10 — Covers 9 datasets with multiple baselines, ablations, and visualizations, but lacks detailed computational cost comparisons and larger-scale experiments.
- **Writing Quality**: 9/10 — Clear structure with a coherent logical flow from problem analysis to method design.
- **Value**: 7/10 — The zero-additional-parameter design has practical appeal, but the method is limited to CLIP architectures and hierarchical inference introduces additional overhead.
- **Overall**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)

</div>

<!-- RELATED:END -->
