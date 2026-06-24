---
title: >-
  [Paper Note] Text-Guided Video Masked Autoencoder
description: >-
  [ECCV 2024][Video Understanding][video MAE] A text-guided masking (TGM) strategy is proposed to mask salient video regions by utilizing natural language descriptions instead of motion priors, unifying MAE with video-text contrastive learning to achieve state-of-the-art relative performance on five action recognition datasets and one egocentric dataset.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "video MAE"
  - "text-guided masking"
  - "contrastive learning"
  - "self-supervised pretraining"
  - "action recognition"
date: 2026-05-08
content_hash: 0278a622465500f0
---

# Text-Guided Video Masked Autoencoder

**Conference**: ECCV 2024  
**arXiv**: [2408.00759](https://arxiv.org/abs/2408.00759)  
**Code**: Not mentioned  
**Area**: Video Understanding / Self-Supervised Learning  
**Keywords**: video MAE, text-guided masking, contrastive learning, self-supervised pretraining, action recognition

## TL;DR
A text-guided masking (TGM) strategy is proposed to mask salient video regions by utilizing natural language descriptions instead of motion priors, unifying MAE with video-text contrastive learning to achieve state-of-the-art relative performance on five action recognition datasets and one egocentric dataset.

## Background & Motivation
**Background**: Video masked autoencoders (Video MAE) have shown great potential in video understanding. VideoMAE and ST-MAE use random masking, while subsequent works (e.g., MGM, MGMAE) explore motion-based masking strategies.

**Limitations of Prior Work**: Masking strategies based on visual priors (such as motion vectors or optical flow) rely on the assumption that input videos satisfy specific conditions (e.g., foreground motion must be greater than background), which limits their robustness. Not all videos conform to these assumptions.

**Key Challenge**: Masking salient regions helps learn better representations, but defining "saliency" relies on specific visual assumptions, which compromises generalization capability.

**Goal**: (1) Can natural language replace visual priors to define salient regions in videos? (2) Can the generative pretraining of MAE be unified with the discriminative pretraining of contrastive learning?

**Key Insight**: Natural language descriptions serve as information-dense representations of videos, implicitly capturing saliency without mode-specific assumptions. The text-video correspondences calculated in the aligned CLIP space are utilized for masking.

**Core Idea**: Replace motion-guided masking with text-guided masking, and jointly pretrain using the MAE reconstruction loss and video-text contrastive loss.

## Method

### Overall Architecture
For each video, BLIP-2 is first used offline to generate text descriptions for 3 frames. During pretraining, the similarity between each video patch and the text is computed in the aligned CLIP space, and the patches with the highest similarity (i.e., the most salient regions) are masked. The MAE encoder only processes visible patches, while the decoder reconstructs the masked regions. Optionally, a video-text contrastive loss is added to the encoder output.

### Key Designs

1. **Text-Guided Masking (TGM)**

    - Function: Determines mask positions based on the semantic correspondence of text descriptions.
    - Mechanism: For each frame $f_t$, features $V_t \in \mathbb{R}^{\frac{H}{h} \times \frac{W}{w} \times D}$ are extracted patch-by-patch using CLIP ViT-B/32. The cosine similarity with the text embedding $w$ is computed, and the top-k patches are selected as masks: $k = \frac{H}{h} \cdot \frac{W}{w} \cdot \gamma$.
    - The optimal masking ratio is 0.6, which is significantly lower than that of VideoMAE (0.9) and MGM (0.75), indicating that the regions masked by TGM have higher information density.
    - Design Motivation: Natural language captures both nouns (objects) and verbs (actions) simultaneously, eliminating the need for visual prior assumptions.

2. **Caption Generation**

    - Function: Generates video descriptions for unannotated K400 and SSv2 datasets.
    - Mechanism: Three keyframes are uniformly sampled from each video, and BLIP-2 is used for offline inference to generate three descriptions. During training, one description is randomly selected.
    - Design Motivation: Since K400 and SSv2 lack human-annotated captions, they must be automatically generated. Although frame-level descriptions contain noise, they are sufficient to support the masking strategy.

3. **Video-Text Alignment**

    - Function: Adds an optional video-text contrastive loss to the MAE framework.
    - Mechanism: Mean pooling is applied to the visible patches output by the MAE encoder to obtain a global video embedding $v_i$. The InfoNCE loss is computed with the text embedding $t_i$:
    $$\mathcal{L}^{\text{NCE}}(q, k^+, \mathcal{N}^-) = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_{k \in \{k^+\} \cup \mathcal{N}^-} \exp(\text{sim}(q, k)/\tau)}$$
    The final loss is $\mathcal{L}_{\text{MSE}} + \mathcal{L}^{\text{NCE}}$.
    - Design Motivation: MAE learns local reconstruction capability, while contrastive learning provides global semantic alignment. The two are complementary.

### Loss & Training
- **Pure MAE**: MSE reconstruction loss.
- **Unified Framework**: $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mathcal{L}^{\text{NCE}}$ (the contrastive loss incurs no additional encoder computational overhead).
- Training ViT-B from scratch with input patch size of $2 \times 16 \times 16$.
- 16-frame input at 224x224 resolution.
- AdamW optimizer with lr=1.5e-4 and cosine decay.
- Both BLIP and CLIP are frozen and do not receive gradients.

## Key Experimental Results

### Main Results — Pure MAE Comparison (200 epochs, ViT-B)

| Masking Strategy | SSv2 FT | SSv2 LP | K400 FT | K400 LP |
|----------|---------|---------|---------|---------|
| Tube (Random) | 66.6 | 25.7 | 78.4 | 38.1 |
| MGM (Motion) | 67.3 | 33.0 | 79.9 | 32.1 |
| TGM (Text) | 67.1 | 26.2 | **79.9** | 33.8 |

### Main Results — Unified Framework (MAE + Contrastive Learning, SSv2)

| Masking Strategy | MAE-only FT | +Contrastive FT | MAE-only LP | +Contrastive LP | LP Gain |
|----------|----------|----------|----------|----------|--------|
| Tube | 64.9 | 65.5 | 20.8 | 33.3 | +12.5 |
| MGM | 67.3 | 67.0 | 33.0 | **37.1** | +4.1 |
| TGM | 67.1 | **67.5** | 26.2 | 33.4 | +7.2 |

### Transfer Learning — Small Datasets & Egocentric (K400 pretrained 200 epochs)

| Dataset | TGM LP | TGM+Contrastive LP | TGM+Contrastive R@1 |
|--------|--------|-------------|-------------|
| UCF101 | 67.7 | **87.1** | **97.6** |
| HMDB51 | 41.6 | **64.3** | **99.1** |
| Diving48 | 11.3 | **19.9** | — |
| Epic-Kitchens | 14.4 | **20.1** | — |

### Ablation Study

| Configuration | SSv2 FT | Notes |
|------|---------|------|
| Masking ratio 0.55 | 67.1 | Slightly lower |
| Masking ratio 0.60 | **67.5** | Optimal |
| Masking ratio 0.75 | 66.4 | Over-masking |
| Bottom-K (Masking least relevant) | 67.2 | Still better than random |
| Top-K (Masking most relevant) | **67.5** | Optimal |
| 1-frame description | 66.5 | Slightly lower |
| 3-frame description | **67.5** | More diverse descriptions are better |

### Key Findings
- TGM competes with motion-guided masking without any explicit visual cues, proving that natural language can effectively capture video saliency.
- Contrastive learning provides the largest boost to linear probing (up to +12.5%), showing that more separable semantic representations are learned.
- The optimal masking ratio of 0.6 is significantly lower than the 0.75-0.9 of other MAE methods, as TGM masks the most information-dense areas.
- Even using "vision-free" text descriptions from GPT3.5 yields decent linear probing performance (54.0), demonstrating the robustness of text guidance.

## Highlights & Insights
- **Unification of MAE and Contrastive Learning**: While FLIP previously reported these two to be adversarial, this paper finds them to be synergistic in the video domain. Even with pure MAE training, the contrastive loss naturally decreases, indicating that the MAE encoder implicitly learns text-aligned semantics.
- **Masking Ratio as a Signal**: The optimal masking ratio of 0.6 is an interesting finding on its own — each masked patch in TGM carries greater information content, meaning not as many patches need to be masked to construct a sufficiently challenging pretraining task.

## Limitations & Future Work
- Relies on frame-level image descriptions from BLIP-2, which fails to capture temporal details within the video.
- Relies on the quality of the aligned CLIP space to generate masks.
- Trained only on ~200K videos, which is much smaller in scale compared to ViCLIP (200M).
- Video-level description models or multi-frame joint descriptions were not explored.

## Related Work & Insights
- **vs MGM/MGMAE**: Motion-guided masking relies on motion vectors/optical flow, whereas TGM achieves comparable or superior performance without any explicit motion information, opening up a new direction for language-guided MAE.
- **vs CoCa/FLIP**: CoCa combines captioning and contrastive learning, while FLIP introduces image masking to accelerate CLIP. This work unifies MAE + masked contrastive learning in the video domain, and finds them to be synergistic (whereas FLIP reported them as adversarial).
- **vs InternVideo**: InternVideo alternates between MAE and contrastive learning training using different backbones, while this work shares the backbone and optimizes jointly.

## Rating
- Novelty: ⭐⭐⭐⭐ First exploration of text-guided masked video MAE, showing a clear and clever perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, systematic ablation of masking ratios/text sources/masking orientations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, deep insights, and thorough analysis in discussions.
- Value: ⭐⭐⭐⭐ Opens a new direction for language-guided video MAE, with a highly generalizable unified framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [\[ECCV 2024\] Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data](rethinking_video-text_understanding_retrieval_from_counterfactually_augmented_da.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[ECCV 2024\] UniINR: Event-guided Unified Rolling Shutter Correction, Deblurring, and Interpolation](uniinr_event-guided_unified_rolling_shutter_correction_deblurring_and_interpolat.md)

</div>

<!-- RELATED:END -->
