---
title: >-
  [Paper Note] PVChat: Personalized Video Chat with One-Shot Learning
description: >-
  [ICCV 2025][Medical Imaging][personalized video LLM] This paper proposes PVChat, the first video large language model supporting personalized subject learning from a single reference video. Through a ReLU-routed Mixture-…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "personalized video LLM"
  - "one-shot learning"
  - "mixture-of-heads"
  - "video question answering"
  - "identity-aware"
date: 2026-05-08
content_hash: 33d898047103adc4
---

# PVChat: Personalized Video Chat with One-Shot Learning

**Conference**: ICCV 2025
**arXiv**: [2503.17069](https://arxiv.org/abs/2503.17069)  
**Code**: [https://github.com/PVChat](https://github.com/PVChat)  
**Area**: Medical Imaging
**Keywords**: personalized video LLM, one-shot learning, mixture-of-heads, video question answering, identity-aware

## TL;DR

This paper proposes PVChat, the first video large language model supporting personalized subject learning from a single reference video. Through a ReLU-routed Mixture-of-Heads (ReMoH) attention mechanism, a systematic data augmentation pipeline, and a progressive image-to-video training strategy, PVChat achieves identity-aware video question answering and surpasses existing state-of-the-art ViLLMs across diverse scenarios including medical, TV drama, and anime settings.

## Background & Motivation

Existing video large language models (ViLLMs) excel at general video understanding tasks such as recognizing activities like "talking" or "eating," but fall significantly short in identity-aware understanding—they cannot interpret personalized scenarios such as "Wilson is undergoing chemotherapy" or "Tom is discussing with Sarah." This limitation restricts their utility in practical applications such as intelligent healthcare and smart home systems.

Existing personalized models support only image understanding and cannot model the dynamic temporal cues present in video—such as motion patterns, interaction dynamics, and contextual dependencies. PVChat aims to fill this gap in personalized video understanding, requiring only a single reference video to learn individual-specific characteristics and enable identity-aware question answering.

## Method

### Overall Architecture

PVChat is built upon Mistral-7B-Instruct-v0.3 and consists of a visual encoder, a ReMoH-enhanced LLM, and a two-stage training strategy. The pipeline proceeds as follows:
1. Personalized subject information is extracted from a reference video and encoded as tokens.
2. Query video frames are processed by the visual encoder to extract frame-level features.
3. The ReMoH attention mechanism enhances subject-specific feature learning.
4. The LLM generates identity-aware question answering responses.

### Key Designs

1. **Systematic Data Augmentation Pipeline**: Addresses the scarcity of personalized video data.

    - **Positive sample generation**: DeepFaceLab face extraction → FaceNet + DBSCAN multi-person disambiguation → face quality assessment (EAR, orientation, sharpness) → InternVideo2 gender/age classification → ConsisID for generating scene-diverse videos (rich but with weaker ID consistency) + PhotoMaker → LivePortrait animation (strong ID consistency but limited content variety). The two approaches are complementary.
    - **Negative sample retrieval**: CLIP-based top-k retrieval of visually similar faces from Laion-Face-5B → LivePortrait animation as hard negative samples, supplemented by 30 randomly sampled videos from CelebV-HQ for content-rich negatives.
    - **QA generation**: InternVideo2 generates four categories of question-answer pairs (existence, appearance, action, location) → ChatGPT-4o refinement to replace generic pronouns with subject names. Each input video is expanded into 81 videos and 1,455 QA pairs.

2. **ReLU-Routed Mixture-of-Heads (ReMoH)**: Divides attention heads into shared heads (always activated) and routed heads (activated on demand). Unlike MoH's Top-k selection (non-fully differentiable and inflexible), ReMoH employs ReLU routing for fully differentiable dynamic selection:
    $$s_i = \begin{cases} \alpha_1, & 1 \leq i \leq n \\ \alpha_2 \text{ReLU}(\mathbf{W}_r \mathbf{x}_t)_i, & n < i \leq n+m \end{cases}$$
   where $[\alpha_1, \alpha_2] = \text{Softmax}(\mathbf{W}_h \mathbf{x}_t)$ balances contributions from shared and routed heads. The natural sparsity of ReLU outputs enables selective activation with only 2 additional MLP parameters. Visualization demonstrates that ReMoH effectively allocates specific heads to learn target subject features, with head activation patterns differing substantially depending on whether the target person is present.

3. **Sparsity Control Strategies**:

    - **Smooth Proximity Regularization (SPR)**: $\mathcal{L}_{SPR} = \beta_p \cdot \|\frac{1}{n}(\mathbf{W}_r \mathbf{x}_t)\|$, where $\beta_{p+1} = \beta_p \cdot e^{k \cdot (T_s - R_s)}$ provides step-wise adaptive weight adjustment, scaling exponentially by distance to achieve smooth training.
    - **Head Activation Enhancement (HAE)**: $\mathcal{L}_{HAE} = e^{2 \cdot (R_s - T_s)} - 1$ (when $R_s > T_s$), preventing all expert heads from collapsing to zero activation.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{LM} + \mathcal{L}_{SPR} + \mathcal{L}_{HAE}$

Two-stage training:
- **Stage 1 — Image Understanding**: The visual encoder is frozen; only ReMoH components are trained along with LoRA fine-tuning of the LLM. The first frame with existence/attribute description QA pairs is used to learn static identity features.
- **Stage 2 — Video Reasoning**: The last few layers of the visual encoder are unfrozen to enhance cross-frame feature integration. QA tasks are extended to action recognition and location recognition, with both positive and negative samples used for training.
- Training configuration: 1× NVIDIA L20 GPU; Stage 1 for 1 epoch, Stage 2 for 7 epochs; batch size 2; total training time approximately 3 hours.
- 8 frames are uniformly sampled per video at a resolution of 1080×1920.

## Key Experimental Results

### Main Results

| Model | Acc↑ | BLEU↑ | BertScore↑ | ES↑ | DC↑ |
|------|------|-------|-----------|-----|-----|
| InternVideo2 | 0.342 | 0.046 | 0.875 | 3.041 | 1.812 |
| VideoLLaMA2 | 0.470 | 0.082 | 0.890 | 3.012 | 3.301 |
| **PVChat (Ours)** | **0.901** | **0.562** | **0.952** | **4.940** | **4.201** |

Accuracy improves from 0.470 to 0.901 (+91.7%), BLEU from 0.082 to 0.562 (+585%), and entity specificity from 3.012 to 4.940.

### Ablation Study

**ReMoH Attention Mechanism**:

| Method | Acc↑ | BLEU↑ | BS↑ | ES↑ | DC↑ |
|------|------|-------|-----|-----|-----|
| Baseline (Q-former) | 0.733 | 0.550 | 0.904 | 4.735 | 4.142 |
| Baseline + MoH | 0.813 | 0.558 | 0.926 | 4.939 | 4.191 |
| **Baseline + ReMoH** | **0.901** | **0.562** | **0.952** | **4.940** | **4.201** |

**SPR and HAE Losses**:

| Method | Activation Rate | Loss | Acc↑ |
|------|--------|------|------|
| w/o SPR and HAE | – | nan | – |
| w/o HAE | 0.217 | 0.085 | 0.746 |
| **PVChat (Full)** | **0.552** | **0.028** | **0.901** |

**Data Type Contribution**:

| Data Type | Acc↑ | BLEU↑ | BS↑ |
|----------|------|-------|-----|
| Original positives only | 0.464 | 0.417 | 0.905 |
| + Negative samples | 0.584 | 0.418 | 0.931 |
| + ConsisID positives | 0.781 | 0.532 | 0.927 |
| **+ LivePortrait positives** | **0.901** | **0.562** | **0.952** |

### Key Findings

- Training with only positive samples causes the model to predict "present" for all queries; negative samples are critical for identity discrimination.
- Expert heads in ReMoH exhibit markedly different activation patterns depending on whether the target subject is present, validating domain-specific learning.
- Removing both SPR and HAE leads to training divergence (loss = nan); the SPR+HAE combination is essential for training stability.
- Head activation rate increases from 0.217 (SPR only) to 0.552 (full model), confirming that HAE effectively prevents expert heads from becoming dormant.
- 16 tokens per subject is optimal; additional tokens degrade performance.

## Highlights & Insights

- PVChat is the first personalized large language model supporting video input, filling an important research gap.
- The systematic data augmentation pipeline is elegantly designed; the combination of complementary positive samples (ConsisID + LivePortrait) and hard negative samples offers strong reference value.
- The advantage of ReLU routing over Top-k selection lies in its full differentiability and flexible adaptability, with a minimal design overhead of only 2 additional MLP parameters.
- Training completes in approximately 3 hours on a single L20 GPU, demonstrating high efficiency.
- A dataset comprising 66 individual scenarios, 304 original videos, 2,304 augmented videos, and 30,000+ QA pairs will be publicly released.

## Limitations & Future Work

- Each new subject requires individual one-shot learning (approximately 3 hours of fine-tuning), precluding zero-shot personalization.
- Only 8 frames are sampled per video, which may be insufficient for long-video understanding.
- The data augmentation pipeline relies on multiple external tools (ConsisID, LivePortrait, DeepFaceLab, etc.), resulting in considerable pipeline complexity.
- Evaluation covers a limited number of scenarios (6 scenes, 25 subjects); generalizability requires validation at larger scale.

## Related Work & Insights

- The ReLU routing combined with SPR/HAE sparsity control strategy is generalizable to other MoE/MoH architectures.
- The positive and negative sample generation pipeline can inspire data construction for other personalized models.
- The two-stage paradigm of image pre-training followed by video fine-tuning represents a general and effective transfer learning strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First personalized video LLM; novel ReMoH design; systematic data augmentation pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scenario, multi-metric evaluation; complete ablation of ReMoH/SPR/HAE/data types; head activation visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured; detailed pipeline descriptions; rich illustrations.
- **Value**: ⭐⭐⭐⭐ Opens a new direction in personalized video understanding with strong application potential in healthcare and smart home domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)
- [\[NeurIPS 2025\] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](../../NeurIPS2025/medical_imaging/a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](../../NeurIPS2025/medical_imaging/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)

</div>

<!-- RELATED:END -->
