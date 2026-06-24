---
title: >-
  [Paper Note] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection
description: >-
  [ECCV 2024][Object Detection][Zero-shot Keypoint Detection] This paper proposes the OpenKD model, which opens up prompt diversity across three dimensions: modality (vision + text), semantics (seen vs. unseen), and language (diverse text). By employing a multimodal prototype set, auxiliary keypoint-text interpolation, and LLM-based text parsing, OpenKD achieves generalized zero- and few-shot keypoint detection, obtaining SOTA performance on Animal Pose, AwA, CUB, and NABirds.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Zero-shot Keypoint Detection"
  - "few-shot learning"
  - "CLIP"
  - "Multimodal Prompting"
  - "LLM Parsing"
date: 2026-05-08
content_hash: 3fd8309a2f21b1b4
---

# OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection

**Conference**: ECCV 2024  
**arXiv**: [2409.19899](https://arxiv.org/abs/2409.19899)  
**Code**: [https://github.com/AlanLuSun/OpenKD](https://github.com/AlanLuSun/OpenKD)  
**Area**: Keypoint Detection / Object Detection  
**Keywords**: Zero-shot Keypoint Detection, few-shot learning, CLIP, Multimodal Prompting, LLM Parsing

## TL;DR

This paper proposes the OpenKD model, which opens up prompt diversity across three dimensions: modality (vision + text), semantics (seen vs. unseen), and language (diverse text). By employing a multimodal prototype set, auxiliary keypoint-text interpolation, and LLM-based text parsing, OpenKD achieves generalized zero- and few-shot keypoint detection, obtaining SOTA performance on Animal Pose, AwA, CUB, and NABirds.

## Background & Motivation

Keypoint detection is a foundational task in computer vision (used in pose estimation, action recognition, fine-grained classification, etc.). However, existing fully supervised methods can only predict a fixed set of keypoints for a fixed species, and generalizing to new species/keypoints requires annotating large amounts of new data. Few-shot keypoint detection (FSKD) achieves few-shot detection using visual prompts (annotated support images), while zero-shot keypoint detection (ZSKD) leverages VLMs like CLIP to achieve zero-shot detection via text prompts.

**Limitations of Prior Work**:

**Single Modality**: Most methods support only visual prompts or text prompts, failing to exploit the complementary advantages of both modalities simultaneously.

**Closed Semantics**: Models cannot handle unseen text prompts (e.g., seeing "eye" during training but encountering "knee" during testing), resulting in extremely poor novel keypoint detection performance.

**Rigid Language**: Existing ZSKD only supports templated simple text (e.g., "the nose of a cat") and cannot handle query questions in diverse natural language styles (e.g., "Can you detect the nose and ears of a cat?").

**Core Idea**: OpenKD "opens up prompt diversity" from three dimensions: employing a multimodal prototype set to handle modality diversity, utilizing LLM-based reasoning for auxiliary keypoint-text interpolation to handle semantic diversity, and incorporating LLM-based text parsing to support language diversity.

## Method

### Overall Architecture

OpenKD is built on episodic training, where each episode contains a support set (visual/text prompt) and a query set (images to be detected). The inference pipeline: 1) The CLIP encoder extracts image and text features $\to$ 2) A residual adaptation network fine-tunes the features $\to$ 3) A multimodal keypoint prototype set is constructed $\to$ 4) Correlation is performed between prototypes and query features $\to$ 5) A heatmap is generated via decoding $\to$ 6) Multimodal heatmaps are fused to obtain the final prediction.

### Key Designs

1. **基于 CLIP 的多模态特征提取与投影 (CLIP-based Multimodal Feature Extraction and Projection)**:

    - **Function**: Uses CLIP RN50 as a shared backbone to extract support/query image features and text features.
    - **Core Idea**: The original CLIP image encoder retains only the classification token via attention pooling, discarding spatial information. The authors reuse the V/O projection matrices of CLIP's attention pooling via $\mathbf{X}' = \mathbf{X}\mathbf{W}_v\mathbf{W}_o$ to obtain projected image tokens, thereby preserving spatial location information.
    - **Design Motivation**: Keypoint detection requires precise spatial localization rather than just global features; meanwhile, reusing CLIP's projection matrices can narrow the modal gap between image tokens and text features.

2. **残差特征适配（Residual Feature Adaptation）**:

    - **Function**: Uses two lightweight adaptation networks $\mathcal{A}_v$ and $\mathcal{A}_t$ to fine-tune image and text features, respectively, in a residual manner.
    - **Core Idea**: $\mathbf{X}^s := \mathbf{X}^s + \mathcal{A}_v(\mathbf{X}^s)$, $\mathbf{t}_n := \mathbf{t}_n + \mathcal{A}_t(\mathbf{t}_n)$.
    - **Design Motivation**: CLIP's pre-trained features are aligned at the image level rather than the keypoint level, necessitating adaptation to the fine-grained task space of keypoint detection.

3. **多模态关键点 Prototype Set**:

    - **Function**: Unifiedly converts visual and text prompts into keypoint prototypes to construct the prototype set $\mathcal{T} = \mathcal{T}^v \cup \mathcal{T}^t$.
    - Visual Prototype (VKP): Performs Gaussian-weighted accumulation centered on keypoint locations on the support image feature map $\mathbf{X}^s$ to yield $\mathbf{\Phi}_n \in \mathbb{R}^d$. For K-shot, it takes the average of the same category of keypoints: $\mathbf{\Psi}_n^v = \frac{1}{K}\sum_k \mathbf{\Phi}_{k,n}$.
    - Text Prototype (TKP): Directly encodes the keypoint text using the CLIP text encoder to obtain $\mathbf{\Psi}_n^t$.
    - **Design Motivation**: Unifies prompts of different modalities into a shared $d$-dimensional feature space, enabling the model to flexibly handle visual prompts, text prompts, or a combination of both.

4. **辅助关键点与文本插值（Auxiliary Keypoints & Texts Interpolation）**:

    - **Function**: Generates auxiliary training samples in both visual and textual domains, drastically improving novel keypoint detection capability.
    - **Visual Interpolation**: Generates the auxiliary keypoint position $\hat{\mathbf{p}}$ via linear interpolation with $z=0.5$ between two known keypoints, and filters out points outside the foreground using saliency detection.
    - **Text Interpolation**: Leverages an LLM (GPT-3.5) to infer potential body parts that lie between two known keypoints. A Chain of Thought (CoT) prompt is employed to enhance reasoning quality. The process is repeated $R$ times, returning 3 answers each time to construct a candidate text pool $\{{\hat{t}_i}\}_{i=1}^{3R}$.
    - **False Text Control (FTC) Selection Strategy**: Samples top-$\eta$ results from the candidate text pool, but rejects a candidate if the cosine similarity between the visual feature of the auxiliary keypoint $\hat{\mathbf{\Phi}}$ and the candidate text feature $\hat{\mathbf{t}}_i$ is below a threshold $\alpha$.
    - **Design Motivation**: The category of keypoints seen during training is limited (e.g., only "eye", "nose"), preventing generalization to novel keypoints. Generating intermediate keypoints and their corresponding texts via interpolation expands the spatial reasoning capability of the model.

5. **模态内/模态间对比学习 (Intra- and Inter-modality Contrastive Learning)**:

    - **Function**: Introduces two contrastive losses to enhance the discriminative ability of the prototypes.
    - $\mathcal{L}_{tt}$ (Intra-text Contrastive): Randomly samples the TKP sets of two species, constructs a similarity matrix $\mathbf{J}$, and optimizes both the cross-species invariance for identical keypoint types and the distinctiveness for different keypoint types.
    - $\mathcal{L}_{vt}$ (Visual-Text Contrastive): Aligns VKPs closer to TKPs, applying a stop gradient to TKPs to prevent the superior text representations from being degraded by inferior visual feature adaptations.
    - **Design Motivation**: Experiments reveal that text prototypes inherently possess better clustering properties and lower variance, so visual prototypes are aligned toward text prototypes rather than vice versa.

6. **LLM 作为语言解析器 (LLM as Language Parser)**:

    - **Function**: Uses an LLM to parse diverse natural language text prompts, extracting keypoint and object keywords to synthesize a standard prompt format.
    - For example, given the input "Can you localize the left eye and nose of cat?", the LLM extracts "left eye", "nose", and "cat".
    - The parsing accuracy of GPT-3.5 reaches 96%+, while Vicuna reaches 93%+.

### Loss & Training

Total loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{kp} + \lambda_2 \mathcal{L}_{tt} + \lambda_3 \mathcal{L}_{vt}$

- $\mathcal{L}_{kp}$: Multi-group heatmap regression loss (MSE) that separately supervises the visual and textual group heatmaps.
- Default: $\lambda_1=1$, $\lambda_2=\lambda_3=0.002$.
- The CLIP text encoder is frozen, while the last two layers of the image encoder are fine-tuned, with temperature $\tau=0.05$.

## Key Experimental Results

### Main Results

**1-shot Keypoint Detection (Animal Pose Dataset, average PCK@0.1 over 5 sub-problems):**

| Method | Novel | Base |
|------|-------|------|
| ProtoNet | 15.47 | 37.73 |
| FSKD-D | 44.75 | 49.93 |
| **OpenKD** | **50.32** | **54.39** |
| **OpenKD+Text** | **63.19** | **64.93** |

**0-shot Keypoint Detection (Animal Pose Dataset, PCK@0.1):**

| Method | Novel | Base |
|------|-------|------|
| CLAMP | 21.92 | 59.47 |
| CLAMP† (with auxiliary text) | 59.84 | 59.51 |
| **OpenKD** | **63.37** | **65.59** |

**Cross-Dataset 1-shot Results (Novel keypoints):**

| Method | Animal Pose | AwA | CUB | NABird |
|------|-------------|-----|-----|--------|
| FSKD-D | 44.75 | 64.76 | 77.89 | 56.04 |
| OpenKD | 50.32 | 66.71 | 78.39 | 53.35 |
| OpenKD+T | **63.19** | **79.02** | 73.29 | 53.40 |

### Ablation Study

**Effects of Auxiliary Keypoints and Texts (Animal Pose):**

| Training Config | 1-shot Novel | 0-shot Novel | Description |
|---------|-------------|-------------|------|
| Primary keypoints only | 21.36 | 1.26 | Fails to detect novel keys |
| + Auxiliary keypoints | 47.54 | 2.00 | Visual interpolation dramatically improves 1-shot |
| Primary text only | 16.18 | 25.60 | Baseline 0-shot capability |
| + Auxiliary text | 15.87 | **63.14** | Text interpolation dramatically improves 0-shot |
| All (Ours) | **50.32** | **63.37** | Optimal combination |

**Ablation of Contrastive Learning (AwA):**

| Config | 1-shot Novel | 0-shot Base | Description |
|------|-------------|-------------|------|
| W/o CL | 65.56 | 81.67 | baseline |
| +$\mathcal{L}_{tt}$ | 66.05 | 84.00 | Improves text discriminativeness |
| +$\mathcal{L}_{tt}$+$\mathcal{L}_{vt}$(stop grad) | **66.71** | **84.32** | Best |

### Key Findings

- **Auxiliary text interpolation contributes the most**: 0-shot Novel leaps from 25.60 to 63.14 (+37.54%), proving that auxiliary text inferred by the LLM is crucial for generalization.
- **Text prompts outperform visual prompts (on base keypoints)**: 0-shot base > 1-shot base because text features yield better clustering and lower variance.
- **Multimodal combination is complementary**: 1-shot+text evaluation (63.19%) significantly outperforms either single modality, compensating for each modality's weaknesses.
- **CoT reasoning brings massive improvements**: Text interpolation utilizing CoT prompts outperforms non-CoT counterparts by 4.5% (78.30 vs. 73.80) on AwA Novel.
- **Robust LLM parsing**: GPT-3.5 delivers a parsing accuracy of > 96% on diverse texts, with only a tiny performance drop of 1.29% on Novel keypoints.

## Highlights & Insights

- **The formulation of "opening prompt diversity along three dimensions" is valuable**: It clearly identifies deficiencies across the modality, semantic, and language levels, offering targeted solutions for each.
- **Dual roles of the LLM**: It acts both as a reasoner (text interpolation) and a parser (diverse text parsing), elegantly utilizing distinct capabilities of LLMs.
- **Stop gradient strategy**: Noticing that text representation quality outperforms visual representation, the authors apply unidirectional alignment (VKPs $\to$ TKPs). This avoids the "weaker modality dragging down the stronger one," offering generalizable insights for other multimodal alignment tasks.
- **Auxiliary text + FTC selection strategy**: Filtering low-quality candidates via visual-text similarity successfully balances recall and precision.

## Limitations & Future Work

- **Limited improvement on CUB and NABirds**: Auxiliary text reasoning for avian keypoints is more difficult, and the LLM's reasoning quality on fine-grained body parts degrades.
- **Dependence on LLM APIs**: Both text interpolation and parsing require calling LLMs (e.g., GPT-3.5), raising latency and overhead.
- **Exclusive usage of CLIP RN50**: The performance of stronger ViT-based CLIP or more advanced VLMs (e.g., SigLIP) remains unexplored.
- **Imprecise spatial locations for text-based auxiliary keypoints**: Text descriptions generated by LLM reasoning may not perfectly match the actual interpolated geometric positions, an issue FTC merely alleviates rather than fully solves.

## Related Work & Insights

- **vs CLAMP**: CLAMP also employs CLIP for animal pose estimation but only supports textual prompts and cannot handle novel text. OpenKD's auxiliary text interpolation scheme (tested as CLAMP† when integrated into CLAMP) drastically improves CLAMP's original performance as well (Novel from 21.92 to 59.84), proving the universality of this approach.
- **vs FSKD-D**: A purely vision-based FSKD method. OpenKD substantially outperforms it on 1-shot (50.32 vs. 44.75) by introducing the text modality, reaching 63.19 when augmented with text prompts during testing.
- **vs ProtoNet / RelationNet**: Classic few-shot learning methods exhibit sub-optimal performance on keypoint detection, indicating the critical need for task-specific architectures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Represents the first systematic exploration of opening prompt diversity along three dimensions. The combination of auxiliary text interpolation and LLM parsing is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively covers four datasets, FSKD/ZSKD/hybrid evaluations, rigorous ablations, and multiple baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with a well-defined problem formulation, though the mathematical notations are dense and require careful reading.
- Value: ⭐⭐⭐⭐ Advances the boundaries of multimodal and zero-shot keypoint detection, offering highly generalizable ideas for LLM-assisted pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/object_detection/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/object_detection/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](../../ICCV2025/object_detection/upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)

</div>

<!-- RELATED:END -->
