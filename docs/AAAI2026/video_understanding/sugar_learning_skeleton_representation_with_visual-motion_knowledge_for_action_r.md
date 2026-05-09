---
title: >-
  [Paper Note] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition
description: >-
  [AAAI 2026][Video Understanding][Skeleton-based action recognition] This paper proposes the SUGAR paradigm, which leverages GPT-generated **motion descriptions** and **visual descriptions** as prior knowledge to supervise skeleton encoders via contrastive learning, producing more discriminative representations. These representations are then fed into an LLM (LLaMA2-7B) with untouched pretrained weights as the classifier, complemented by a newly designed Temporal Query Projection (TQP) module for efficient skeleton-based action classification and zero-shot inference.
tags:
  - AAAI 2026
  - Video Understanding
  - Skeleton-based action recognition
  - large language models
  - visual-motion knowledge
  - contrastive learning
  - zero-shot recognition
date: 2026-05-08
content_hash: f3130a252b6ec20c
---

# SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition

**Conference**: AAAI 2026
**arXiv**: [2511.10091](https://arxiv.org/abs/2511.10091)
**Code**: N/A
**Area**: Video Understanding / Action Recognition
**Keywords**: Skeleton-based action recognition, large language models, visual-motion knowledge, contrastive learning, zero-shot recognition

## TL;DR

This paper proposes the SUGAR paradigm, which leverages GPT-generated **motion descriptions** and **visual descriptions** as prior knowledge to supervise skeleton encoders via contrastive learning, producing more discriminative representations. These representations are then fed into an LLM (LLaMA2-7B) with untouched pretrained weights as the classifier, complemented by a newly designed Temporal Query Projection (TQP) module for efficient skeleton-based action classification and zero-shot inference.

## Background & Motivation

**State of the Field**: Skeleton-based action recognition aims to model spatiotemporal graph structures for classifying human actions. Its lightweight data format makes it suitable for human-computer interaction and intelligent surveillance. Methods such as ST-GCN, 2s-AGCN, and CTR-GCN have continuously advanced graph convolutional networks, yet distinguishing fine-grained similar actions (e.g., drinking water vs. eating) remains challenging. Recent LLMs (e.g., Vicuna, LLaMA) have demonstrated strong cross-modal capabilities, and some works have begun employing LLMs as action recognizers.

**Limitations of Prior Work**:

**How can LLMs understand skeleton data?** LLMs are pretrained primarily on human language data, creating a substantial gap between skeleton data (coordinate sequences) and text. ActionLLM employed VQ-VAE to learn discrete tokens, but ensuring alignment between skeleton tokens and text tokens remains difficult.

**How can LLMs distinguish similar actions?** Daily activities contain many similar actions (e.g., drinking from a bottle vs. a can). Without appearance information, skeleton motion trajectories alone are insufficient for disambiguation. Existing methods lack high-level semantic information to enhance the discriminability of skeleton representations.

**Limitations of conventional linear classifiers**: The logistic distribution of linear layers cannot directly accommodate action categories from other datasets, limiting zero-shot generalization.

**Root Cause**: Skeleton data contains motion information but lacks visual context, while LLMs possess rich knowledge of human activities but cannot directly process skeleton inputs. The key challenge is **enabling LLMs to learn more discrete (discriminable) skeleton representations**.

**Starting Point**: Rather than directly training LLMs on skeletons, the paper first uses rich linguistic knowledge (motion descriptions + visual descriptions) to **supervise the skeleton encoder** in learning text-aligned discrete representations, which are then fed into an LLM for classification.

## Method

### Overall Architecture

The SUGAR training pipeline consists of three steps:
- **Step 1: Text Construction** — GPT and VLMs generate motion and visual descriptions.
- **Step 2: Skeleton Representation Learning** — Contrastive learning aligns skeleton representations with text descriptions.
- **Step 3: Action Recognition** — TQP projects skeleton representations into the LLM space; LoRA fine-tuning adapts the LLM for classification.

At inference, only skeleton sequences are required for action recognition.

### Key Designs

#### 1. **Text Construction (Step 1)**

**Motion Knowledge Generation**: A predefined action list is defined, and each action is decomposed into movements of 6 body parts (head, hands, arms, hips, legs, feet). GPT-3.5-turbo is used to automatically generate fine-grained motion descriptions $\mathcal{T}_m$. For instance, "drinking water" is described as "the head tilts slightly back, the hand grips the cup, the arm raises toward the mouth..."

**Visual Knowledge Generation**: GPT-4V extracts action-relevant visual information $\mathcal{T}_v$ from video frames. Three rules are imposed to prevent VLMs from generating irrelevant content (e.g., appearance, environment). Frame similarity is computed via the CLIP visual encoder, and the most dissimilar frame set is selected to filter redundant frames.

**Design Motivation**: Motion descriptions can differentiate body-part trajectories across actions (e.g., differences in arm angles), but when trajectories are highly similar (e.g., arm movements in eating and drinking are alike), visual information (e.g., whether the hand holds a cup or food) is needed for disambiguation.

#### 2. **Skeleton–Text Contrastive Learning (Step 2)**

**Skeleton Encoder**: CTR-GCN serves as the backbone, stacking multiple GCN blocks for spatial aggregation:
$$\text{H}^{l+1} = \sigma(\text{D}^{-1/2}\text{A}\text{D}^{-1/2}\text{H}^l\text{W}^l)$$

The multi-scale temporal modeling module of CTR-GCN is also employed, but **temporal-dimension pooling is discarded** to preserve complete temporal information.

**Text Encoder**: The CLIP text encoder encodes motion descriptions $m = E_t(\mathcal{T}_m)$ and visual descriptions $v_i = E_t(\mathcal{T}_{v_i})$ separately, then randomly combines them as $\mathbf{t} = \{m, v_i | i \in I_v\}$.

**Multi-Instance Contrastive Learning Loss**: Unlike one-to-one contrastive learning, a skeleton representation may positively match **multiple text descriptions**. The MIL-NCE loss is adopted:
$$\mathcal{L}_{MIL} = -\frac{1}{|B|} \sum_i \log \frac{\sum_j \sum_n \exp(\mathbf{s}_i^\top \mathbf{t}_{j,n}/\tau)}{\sum_k \sum_n \exp(\mathbf{s}_i^\top \mathbf{t}_{k,n}/\tau)}$$

#### 3. **Temporal Query Projection (TQP) Module (Step 3)**

**Problem**: The skeleton encoder outputs long sequences (e.g., 1000 frames), which are computationally prohibitive for LLMs and poorly suited to LLM modeling of non-linguistic long sequences. However, simple pooling/compression disrupts continuous temporal topology.

**Design**: The skeleton representation $\mathbf{s} \in \mathbb{R}^{L_s \times d}$ is divided into segments by a hyperparameter $k$. Learnable query vectors $\mathbf{q} \in \mathbb{R}^{k \times d}$ are defined, and multiple weight-shared Q-Formers perform **chained queries** over the skeleton representation:

$$\hat{\mathbf{s}}_t = f_Q^t(\mathbf{s}_{t-1}, \mathbf{s}_t)$$

**Core Novelty**: Unlike independent queries in a single Q-Former, TQP feeds the query output of each segment as the query input for the next segment, enabling **continuous temporal information modeling**. The sequence length is ultimately compressed to $L=128$ tokens.

**Why 128?** Ablation studies show that LLMs struggle to model excessively long non-linguistic sequences (performance degrades at 1000 tokens), while compressing to 1 token causes signal homogenization that prevents action discrimination; 128 is the optimal trade-off.

#### 4. **LLM Fine-Tuning (Step 3)**

The skeleton encoder parameters are frozen, and **LLaMA2-7B** is fine-tuned for 1 epoch using **LoRA** (r=64, alpha=16). A fixed instruction template is defined; action tokens are fed as input and the LLM predicts the action category. The loss is:
$$\mathcal{L}_{LoRA} = \text{CrossEntropy}(f_{LLM}(\hat{\mathbf{s}}), y)$$

where $y$ contains the ground-truth category and action description. The LLM outputs both the category and an action description.

### Loss & Training

Two-stage training:
- **Stage 1**: Train the skeleton encoder with $\mathcal{L}_{MIL}$; SGD optimizer, lr=0.01, 200 epochs, batch=200.
- **Stage 2**: Freeze the encoder; fine-tune the LLM with $\mathcal{L}_{LoRA}$; lr=2e-5, 1 epoch, batch=128.
- Hardware: 2× NVIDIA A6000.

## Key Experimental Results

### Main Results

| Method | Classifier | Input | Toyota SH X-sub | SH X-view1 | SH X-view2 | NTU60 X-sub | NTU60 X-view | NTU120 X-sub | NTU120 X-view |
|--------|-----------|-------|-----------------|------------|------------|-------------|-------------|-------------|-------------|
| 2s-AGCN | FC | Joint+Bone | 55.7 | 21.6 | 53.3 | 84.2 | 93.0 | 78.2 | 82.9 |
| ST-GCN | FC | Joint+Bone | 62.9 | 40.6 | 51.4 | 81.5 | 88.3 | 82.1 | 84.5 |
| UNIK | FC | Joint+Bone | 62.1 | 33.4 | 63.6 | 86.8 | 94.4 | 80.8 | 86.5 |
| LLM-AR | LLM | Joint | 67.0 | 36.1 | 66.6 | 95.0 | 98.4 | 88.7 | 91.5 |
| **SUGAR** | LLM | Joint | **70.2** | **50.9** | **67.1** | **95.2** | 97.8 | **90.1** | 89.7 |

**Zero-Shot Results**:

| Method | Protocol 1 (NTU60→NTU120) | | Protocol 2 (NTU60→PKU) | |
|--------|---------------------------|---|-------------------------|---|
| | Top-1 | Top-5 | Top-1 | Top-5 |
| ST-GCN (FC) | 30.1 | 45.2 | 36.9 | 55.2 |
| LLM-AR (LLM) | 59.7 | 84.1 | 49.4 | 74.2 |
| **SUGAR (LLM)** | **65.3** | **89.8** | **53.4** | **77.6** |

### Ablation Study

**Impact of Visual-Motion Knowledge** (Toyota Smarthome):

| Configuration | Accuracy |
|---------------|----------|
| No visual, no motion knowledge | 69.2% |
| Visual knowledge only | 69.4% |
| Motion knowledge only | 72.1% |
| **Visual + Motion knowledge** | **73.4%** |

**Comparison of Bridging Modules** (Toyota Smarthome):

| Method | Accuracy |
|--------|----------|
| Cross-Attention | 52.1% |
| One Q-Former | 70.7% |
| One linear layer | 70.4% |
| **Temporal Query Projection** | **73.4%** |

**Impact of Action Token Length** (NTU60):
- Full length (1000): moderate performance
- 128: optimal
- 1: worst (excessive compression leads to signal homogenization)
- Conclusion: LLMs are ill-suited for excessively long non-linguistic sequences, yet over-compression loses discriminability.

### Key Findings

1. **Visual-motion knowledge yields a 4.2% gain**: from 69.2% to 73.4%; motion knowledge contributes more (+2.9%), visual knowledge contributes less in isolation (+0.2%), but their combination is optimal.
2. **Largest advantage on Toyota Smarthome**: X-view1 improves from 36.1% to 50.9% (+14.8%), as this dataset contains many compound daily activities that benefit from visual context.
3. **Zero-shot inference substantially outperforms linear methods**: Linear classifiers' logistic distributions cannot accommodate new dataset categories, whereas LLM-based reasoning over natural-language action lists inherently supports open-set recognition. Protocol 1 Top-1 improves from 30.1% to 65.3%.
4. **TQP outperforms a single Q-Former by 2.7%**: The chained query design preserves temporal continuity.
5. **t-SNE visualization**: After training, skeleton representations of similar actions (e.g., "drinking from a bottle" vs. "drinking from a can") become clearly separated, confirming that visual-motion knowledge supervision induces more discriminative representations.

## Highlights & Insights

1. **"Improving the data, not the model" paradigm**: SUGAR does not design a stronger skeleton encoder; instead, it uses rich linguistic knowledge to **improve the quality of skeleton representations**, offering a lightweight yet effective methodology.
2. **Potential of LLMs as general-purpose recognizers**: Using only Joint input (without Bone or multi-stream fusion), SUGAR surpasses traditional methods that use Joint+Bone, demonstrating that LLM pretrained knowledge can compensate for limited input signals.
3. **Body-part decomposition of motion descriptions**: Decomposing actions into movements of 6 body parts (following the HAKE framework) provides structured prior knowledge.
4. **TQP's chained query design**: Compresses sequence length while preserving temporal continuity, outperforming both independent Q-Formers and linear projection.
5. **Zero-shot capability**: Demonstrates the inherent advantage of LLM-based methods over linear classifiers in open-set scenarios.

## Limitations & Future Work

1. **Joint-only input**: Bone and Motion streams are not utilized; multi-stream fusion could yield further gains.
2. **Unverified quality of GPT-generated descriptions**: The approach relies on GPT-3.5/4V generation quality, which may contain noise or erroneous descriptions.
3. **Fixed LLM scale at 7B**: The impact of larger or smaller models is not explored.
4. **Predefined action list required**: Although zero-shot recognition is supported, a predefined set of candidate category names is still necessary, precluding truly open-vocabulary recognition.
5. **Limited gains on NTU datasets**: NTU60/120 are captured in controlled single-scene settings with high skeleton quality and large inter-class variation; SUGAR's advantages are more pronounced in complex scenarios such as Toyota Smarthome.

## Related Work & Insights

- Extending CLIP-based contrastive learning from image-text to skeleton-text is a natural progression; the key contribution here lies in **the choice of text content**—using fine-grained motion and visual descriptions rather than simple class-name templates (e.g., "A action of {}").
- ActionLLM pioneered the use of LLMs for skeleton-based recognition; SUGAR builds upon it by introducing external knowledge to supervise representation learning, achieving 3–15% improvements.
- The chained Q-Former design in TQP is generalizable to other long-sequence token compression scenarios (e.g., video tokens, audio tokens).

## Rating

- Novelty: ⭐⭐⭐⭐ (Visual-motion knowledge-supervised skeleton learning + TQP design; paradigm-level innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets, zero-shot experiments, extensive ablations, visualization analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear pipeline, coherent logic, well-coordinated figures and text)
- Value: ⭐⭐⭐⭐ (Provides a new paradigm for LLM-based skeleton recognition; strong practical utility for zero-shot recognition)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](../../ICCV2025/video_understanding/adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)

<!-- RELATED:END -->
