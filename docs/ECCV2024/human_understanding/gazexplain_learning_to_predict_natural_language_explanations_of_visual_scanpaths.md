---
title: >-
  [Paper Note] GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths
description: >-
  [ECCV 2024][Human Understanding][Visual scanpaths] This work proposes GazeXplain, which is the first to combine visual scanpath prediction with natural language explanations. Through an attention-language decoder, a semantic alignment mechanism, and cross-dataset co-training, it achieves explainable prediction of human gaze behavior.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Visual scanpaths"
  - "natural language explanations"
  - "gaze prediction"
  - "semantic alignment"
  - "cross-dataset co-training"
date: 2026-05-08
content_hash: 2430f86722629356
---

# GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths

**Conference**: ECCV 2024  
**arXiv**: https://arxiv.org/abs/2408.02788  
**Code**: None  
**Area**: Visual Attention / Explainable AI  
**Keywords**: Visual scanpaths, natural language explanations, gaze prediction, semantic alignment, cross-dataset co-training

## TL;DR

This work proposes GazeXplain, which is the first to combine visual scanpath prediction with natural language explanations. Through an attention-language decoder, a semantic alignment mechanism, and cross-dataset co-training, it achieves explainable prediction of human gaze behavior.

## Background & Motivation

1. **Background**: When humans explore visual scenes, the scanpaths formed by eye movements—the spatial-temporal sequence of fixation points—reflect underlying attention processes. Understanding visual scanpaths is crucial for applications such as human-computer interaction, autonomous driving, and user experience design. Existing scanpath prediction models (e.g., ChenLSTM, Gazeformer) perform well in predicting the "when" and "where" of gaze, but fail to explain the "what" and "why."

2. **Limitations of Prior Work**: 
    - **Lack of explainability**: Traditional scanpath models only output sequences of fixation locations and durations, offering no explanations for why these fixations occur, leading to a comprehension gap.
    - **Lack of annotated explanation data**: Existing eye-tracking datasets only annotate fixation coordinates and timestamps, lacking semantic explanation labels for each fixation.
    - **Task specificity**: Existing models are typically trained on a single dataset for a single task (such as free-viewing, object search, or VQA), resulting in poor generalization.
    - **Disconnection between vision and language**: Gaze behavior contains rich semantic information, but existing methods fail to connect visual attention with natural language understanding.

3. **Key Challenge**: Scanpath prediction requires deep semantic understanding, but existing models only perform shallow spatial predictions, failing to clarify the cognitive processes and semantic reasons behind fixations.

4. **Goal**: 
    - Construct scanpath annotation data with natural language explanations.
    - Design a unified model capable of simultaneously predicting scanpaths and generating natural language explanations.
    - Achieve generalization capability across datasets and tasks.

5. **Key Insight**: Utilize a large vision-language model (LLaVA) for semi-automated gaze explanation annotation, then design a unified architecture that integrates attention decoding and language generation, and enhance quality and generalization through semantic alignment and cross-dataset co-training.

6. **Core Idea**: Enable the scanpath prediction model to not only predict where humans look but also explain why they look there using natural language, thereby achieving explainable modeling of human visual attention.

## Method

### Overall Architecture

GazeXplain is built on a general vision-language encoder, with the core innovation lying in the attention-language decoder:

1. **Vision-Language Encoder**:
    - Visual encoding: ResNet-50 extracts local image features $V_R \in \mathbb{R}^{C \times hw}$, and a Transformer encoder obtains global contextual features $V_T \in \mathbb{R}^{d \times hw}$.
    - Language encoding: RoBERTa processes task instructions to yield semantic embeddings $t_I \in \mathbb{R}^{d_{text}}$.
    - Multimodal fusion: Visual and language features are concatenated to obtain $V_I \in \mathbb{R}^{d \times hw}$.

2. **Attention-Language Decoder**:
    - The attention decoder predicts the sequence of fixation locations and durations.
    - The language decoder generates a natural language explanation for each fixation point.

3. **Semantic Alignment Mechanism**: Ensures semantic consistency between fixations and explanations.

4. **Cross-Dataset Co-Training**: Co-trains on multiple eye-tracking datasets.

### Key Designs

1. **Attention-Language Decoder**: 
    - **Function**: Jointly predict scanpaths and generate natural language explanations for each fixation point.
    - **Mechanism**:
     - **Attention Decoder**: Uses a Transformer model to generate saliency feature vectors $\{s_k\}_{k=1}^K$, predicting the spatial-temporal distribution of fixations $\{m_k\}$ through cosine similarity with the joint embedding $V_I$, while simultaneously predicting the log-normal distribution parameters $\{\mu_k, \sigma_k^2\}$ of fixation durations and sequence end tokens $\{e_k\}$.
     - **Language Decoder**: (1) Extracts local features $g_k$ from visual features $V_T$ based on the fixation location $y_k$; (2) Projects visual features $g_k$ and the semantic embedding $t_I$ into the same dimension using learnable parameters and positional encodings; (3) Feeds the fused features into a pre-trained language model (BLIP) to generate explanation texts $\{w_\ell^k\}_{\ell=1}^L$.
    - **Design Motivation**: By providing natural language explanations for each fixation point, the model is forced to understand the semantic content of the fixated region, which in turn improves the accuracy of scanpath predictions.

2. **Semantic Alignment Mechanism**: 
    - **Function**: Ensure that predicted fixations, generated explanations, and visual features remain consistent in the semantic space.
    - **Mechanism**: Compute four types of pairwise similarities:
     - Visual similarity $s_{i,j}^r$: Cosine similarity of visual features of fixated regions extracted by a pre-trained ResNet (used as pseudo-labels).
     - Explanation similarity $s_{i,j}^e$: Cosine similarity of the linguistic features of explanations for different fixations.
     - Fixation similarity $s_{i,j}^f$: Cosine similarity of the visual features of fixation points.
     - Multimodal similarity $s_{i,j}^m$: Cross-modal cosine similarity between explanation linguistic features and fixation visual features.
    - Alignment Loss: $\mathcal{L}_{aln} = \frac{1}{K'^2} \sum_{i,j} [(s_{i,j}^e - s_{i,j}^r)^2 + (s_{i,j}^f - s_{i,j}^r)^2 + (s_{i,j}^m - s_{i,j}^r)^2]$
    - **Design Motivation**: If two fixations target similar visual content, their explanations and fixation features should also be similar. This consistency constraint facilitates the coordination of multimodal representations.

3. **Cross-Dataset Co-Training**: 
    - **Function**: Enable the model to learn simultaneously from multiple eye-tracking datasets belonging to different tasks, thereby increasing generalization capability.
    - **Mechanism**: Unify instructions of different tasks into a VQA format—converting free-viewing into "What do you see in the image?" and object searching into "Is there a [target] in the image?". Images and scanpaths are uniformly scaled to a resolution of 384x512. Optionally, the observer's answer can be included to capture individual differences.
    - **Design Motivation**: Training on a single dataset tends to overfit to specific tasks. Co-training allows the model to learn common attention patterns across different tasks.

### Loss & Training

The final training objective is the sum of three losses:

$$\mathcal{L} = \mathcal{L}_{fix} + \mathcal{L}_{exp} + \mathcal{L}_{aln}$$

- **Scanpath prediction loss** $\mathcal{L}_{fix}$: Conditional log-probability of fixation locations + log-normal distribution loss of durations.
- **Language generation loss** $\mathcal{L}_{exp}$: Standard cross-entropy loss for autoregressive language modeling.
- **Semantic alignment loss** $\mathcal{L}_{aln}$: The multi-view consistency loss described above.
- **Training strategy**: Supervised learning is performed first for 8 epochs (lr=4×10⁻⁴, batch=16), followed by 2 epochs of Self-Critical Sequence Training (SCST, lr linearly decaying from 10⁻⁵, batch=8).

## Key Experimental Results

### Main Results

Scanpath prediction results on 4 eye-tracking datasets/subsets:

| Dataset | Metric | GazeXplain | Gazeformer | ChenLSTM | Gain |
|--------|------|-----------|-----------|---------|------|
| AiR-D (VQA) | SM↑ | **0.386** | 0.357 | 0.350 | +8.1% |
| AiR-D | CC↑ | **0.662** | 0.550 | 0.629 | +5.2% |
| AiR-D | NSS↑ | **1.851** | 1.512 | 1.727 | +7.2% |
| OSIE (Free-view) | SM↑ | **0.380** | 0.372 | 0.377 | +0.8% |
| OSIE | CC↑ | **0.748** | 0.685 | 0.722 | +3.6% |
| COCO-Search18 TP | SM↑ | **0.480** | 0.433 | 0.448 | +7.1% |
| COCO-Search18 TP | SS↑ | **0.541** | 0.470 | 0.475 | +13.9% |
| COCO-Search18 TA | SM↑ | **0.373** | 0.354 | 0.366 | +1.9% |

### Ablation Study

Analysis of component contributions on the AiR-D dataset:

| Configuration (EXP/ALN/CT) | SM↑ | CC↑ | NSS↑ | CIDEr-R↑ | Description |
|-------------------|-----|-----|------|----------|------|
| ✗/✗/✗ | 0.337 | 0.582 | 1.582 | 61.9 | Baseline |
| ✓/✗/✗ | 0.339 | 0.614 | 1.674 | 91.9 | Language decoder alone is effective |
| ✓/✓/✗ | 0.346 | 0.631 | 1.733 | 115.1 | Semantic alignment further improves |
| ✗/✗/✓ | 0.356 | 0.582 | 1.597 | 66.7 | Co-training alone is effective |
| ✓/✗/✓ | 0.378 | 0.647 | 1.797 | 97.3 | Integration of explanation and co-training |
| ✓/✓/✓ | **0.386** | **0.662** | **1.851** | **123.1** | Optimal with all components |

### Key Findings

- Adding language explanations to fixation points does not impair scanpath prediction performance but significantly improves prediction accuracy instead (SM increases from 0.337 to 0.386).
- The semantic alignment mechanism improves CIDEr-R from 97.3 to 123.1, while also enhancing scanpath prediction metrics.
- Cross-dataset co-training yields the largest SM improvement on AiR-D (from 0.346 to 0.386), and CT is more effective on exploratory tasks such as OSIE and COCO-Search18 TA.
- Competing models like ChenLSTM and Gazeformer experience a performance decline when trained on multiple datasets, proving that the design of GazeXplain is crucial for exploiting multi-source data.
- The explanations generated by GazeXplain outperform direct BLIP generation in terms of faithfulness, diversity, and lexical richness.

## Highlights & Insights

- **Pioneering new task**: This work is the first to propose the explainable scanpath prediction task, reasoning "where to look" and "why look" under a unified model.
- **Innovative data annotation**: Utilizing LLaVA for semi-automated annotation combined with manual quality control, natural language explanations were annotated for 86,407 fixation points across 4 datasets.
- **Explanation aids prediction**: An unexpected yet profound finding—forcing the model to explain gaze behavior actually enhances accuracy of gaze prediction, indicating that semantic understanding is essential for attention modeling.
- **Elegant semantic alignment**: Utilizing visual similarity as a self-supervised signal constraints multimodal consistency between explanations and fixations.
- **Strong generalization**: Demonstrates SOTA performance on two additional datasets, COCO-FreeView and WebSaliency.

## Limitations & Future Work

- Explanations generated by LLaVA may contain noise (e.g., text recognition errors, inaccurate descriptions of small objects), and despite manual quality control, about 0.58% abnormal annotations remain.
- The granularity of explanations is fixed at the fixation-point level, and higher-level, scanpath-level comprehensive explanations remain unexplored.
- Currently, only BLIP is used as the language decoder; more powerful LLMs could be explored to improve explanation quality.
- Fixation duration information is not utilized to adjust the level of detail in explanations (e.g., longer fixations might require more detailed explanations).
- The data blending ratio for cross-dataset co-training may require more meticulous tuning.

## Related Work & Insights

- **ChenLSTM / Gazeformer**: Existing SOTA scanpath prediction methods, upon which GazeXplain builds by adding explanation capabilities.
- **BLIP / LLaVA**: Vision-language models used for language decoding and data annotation, respectively.
- **AiR-D / OSIE / COCO-Search18**: Core eye-tracking datasets.
- **Image Captioning / Visual Explanation**: GazeXplain introduces image captioning techniques into gaze explanation.
- **Insight**: Explainability is not just an add-on output of a model; it can conversely boost the core performance of the model—a concept highly applicable to other tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneers a new research direction in explainable scanpath prediction, with innovations in task definition, data annotation, and model design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 4+2 datasets with comprehensive ablation studies and multi-dimensional evaluation (scanpath + saliency + explanation quality + diversity + faithfulness).
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure with detailed and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Opens up a completely new direction for understanding human visual attention, possessing broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Forecasting 3D Scanpaths in Egocentric Video](../../CVPR2026/human_understanding/forecasting_3d_scanpaths_in_egocentric_video.md)
- [\[ECCV 2024\] A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](a_simple_baseline_for_spoken_language_to_sign_language_trans.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](../../AAAI2026/human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[CVPR 2026\] Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes](../../CVPR2026/human_understanding/beyond_scanpaths_graph-based_gaze_simulation_in_dynamic_scenes.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)

</div>

<!-- RELATED:END -->
