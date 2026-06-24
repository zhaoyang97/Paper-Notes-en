---
title: >-
  [Paper Note] A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars
description: >-
  [ECCV 2024][Human Understanding][Spoken2Sign] This paper proposes the first baseline system for Spoken2Sign translation with 3D Avatar output. The system translates spoken text into 3D sign language animations through a three-step pipeline (dictionary construction $\to$ SMPLSign-X 3D pose estimation $\to$ retrieval-connection-rendering translation). It achieves a back-translation BLEU-4 of 25.46 on Phoenix-2014T, while its 3D sign language byproducts (keypoint enhancement and…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Spoken2Sign"
  - "Sign Language Production"
  - "SMPLSign-X"
  - "3D Avatar"
  - "co-articulation"
date: 2026-05-08
content_hash: a396d24888d11c91
---

# A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars

**Conference**: ECCV 2024  
**arXiv**: [2401.04730](https://arxiv.org/abs/2401.04730)  
**Code**: [https://github.com/FangyunWei/SLRT](https://github.com/FangyunWei/SLRT)  
**Area**: 3D Vision / Sign Language Translation / Human Pose Estimation  
**Keywords**: Spoken2Sign, Sign Language Production, SMPLSign-X, 3D Avatar, co-articulation  

## TL;DR
This paper proposes the first baseline system for Spoken2Sign translation with 3D Avatar output. The system translates spoken text into 3D sign language animations through a three-step pipeline (dictionary construction $\to$ SMPLSign-X 3D pose estimation $\to$ retrieval-connection-rendering translation). It achieves a back-translation BLEU-4 of 25.46 on Phoenix-2014T, while its 3D sign language byproducts (keypoint enhancement and multi-view understanding) significantly improve the performance of sign language understanding tasks.

## Background & Motivation
The field of sign language translation has long focused on the Sign2Spoken (sign language to spoken language) direction, whereas the reverse Spoken2Sign (spoken language to sign language) remains severely understudied. Existing Spoken2Sign research mainly outputs 2D keypoint sequences or synthesizes 2D videos using generative models. However, 2D keypoint sequences are difficult for deaf individuals to understand, while 2D videos suffer from blurriness and visual distortions. With the maturity of parametric 3D human models like SMPL-X, deploying 3D Avatars to display sign language has become feasible—this not only avoids the distortions of 2D representations but also allows viewing sign language from arbitrary angles, bringing it closer to real-life communication scenarios.

## Core Problem
How to construct an end-to-end Spoken2Sign system that translates input text into high-quality 3D sign language animations? This involves three sub-problems: (1) Since existing sign language datasets lack ready-made dictionaries, how can they be built automatically? (2) How to accurately estimate temporally consistent 3D sign language representations from monocular sign language videos? (3) How to smoothly concatenate retrieved isolated 3D signs to simulate natural co-articulation transitions?

## Method

### Overall Architecture
The system operates in three stages: **Dictionary Construction** $\to$ **3D Sign Estimation** $\to$ **Spoken2Sign Translation**. Given an input spoken text, it is first translated into a gloss sequence (sign language annotation sequence) using a Text2Gloss translator (mBART). Then, for each gloss, the corresponding 3D sign is retrieved from a pre-built gloss-to-3D sign dictionary. Next, a sign connector predicts the duration of co-articulation between adjacent signs to interpolate transition frames in 3D space. Finally, Blender is used to render the 3D Avatar animation.

### Key Designs
1. **Dictionary Construction (CTC Forced Alignment Segmentation)**: A trained continuous sign language recognition model, TwoStream-SLR, is used to segment continuous sign language videos into isolated sign snippets via a CTC forced alignment algorithm to automatically build a gloss-video dictionary. This is superior to using external dictionaries, as the segmented isolated signs do not contain meaningless movements (e.g., raising or lowering hands) at the beginning or end, making them more suitable for subsequent sign concatenation.

2. **SMPLSign-X (Sign-Specific 3D Estimator)**: Built upon SMPLify-X, three improvements are introduced specifically for sign language characteristics—(a) **Invisible Joint Regularization $\mathcal{L}_{unseen}$**: Pushes joints with low HRNet detection confidence (<0.65) towards the rest pose to prevent erroneous estimation of unseen lower body parts or occluded hands; (b) **Upright Upper Body Constraint $\mathcal{L}_{upright}$**: Enforces consistent depth for joints like the neck and pelvis to ensure the signer's upper body remains upright and does not tilt; (c) **Temporal Smoothing Loss $\mathcal{L}_{smooth}$**: Constrains pose parameter discrepancies between adjacent frames to resolve temporal jittering caused by frame-by-frame independent fitting.

3. **Sign Connector**: A 4-layer MLP that takes the 3D keypoints of the final frame of the preceding sign, the initial frame of the succeeding sign, and their Euclidean distance difference as inputs, and outputs the number of frames for the co-articulation. The training objective employs the L1 loss. During inference, transitional animations are generated by uniformly interpolating frames in the 3D joint space according to the predicted frame count. Compared with fixed-length interpolation, dynamic duration prediction matches the variability of actual co-articulation much better.

4. **Sign Retrieval**: Since each gloss can map to multiple video instances, an isolated sign language recognition model (NLA-SLR) is trained. During retrieval, the instance with the highest confidence for the target gloss is chosen to ensure high-quality 3D sign selection.

### Loss & Training
- Total loss of SMPLSign-X: $\mathcal{L} = \mathcal{L}_{joint} + \mathcal{L}_{prior} + \mathcal{L}_{penetration} + \lambda_1\mathcal{L}_{unseen} + \lambda_2\mathcal{L}_{upright} + \lambda_3\mathcal{L}_{smooth}$, by default $\lambda_1=3e5, \lambda_2=7e5, \lambda_3=1e3$
- Multi-stage optimization using L-BFGS optimizer, fitting SMPL-X parameters with 300 epochs per frame
- Text2Gloss uses mBART, trained for 80 epochs with lr=1e-5, dropout=0.3, label smoothing=0.2
- Sign Connector uses Adam optimizer with lr=1e-5, filtering out extremely long co-articulations

## Key Experimental Results

**Spoken2Sign Back-Translation (P-2014T)**:

| Method | Metric | Dev BLEU-4 | Dev ROUGE | Test BLEU-4 | Test ROUGE |
|------|------|------------|-----------|-------------|------------|
| Progressive Transformer | 2D Keypoints | 11.82 | 33.18 | 10.51 | 32.46 |
| FS-Net | 2D Keypoints + GAN | 16.92 | 35.74 | 21.10 | 42.57 |
| SignDiff | 2D Keypoints + Diffusion | 18.26 | 39.62 | 22.15 | 46.82 |
| **Ours** | **3D Avatar** | **24.16** | **49.12** | **25.46** | **49.68** |

**Comparison of 3D Sign Estimators (P-2014T Back-Translation)**:

| Method | Dev BLEU-4 | Dev ROUGE | 2D KL↓ | TC↑ |
|------|------------|-----------|--------|-----|
| HRNet (2D pseudo GT) | 22.94 | 48.81 | 0.00 | 0.961 |
| SMPLify-X | 19.21 | 44.28 | 31.56 | 0.945 |
| OSX | 22.31 | 47.71 | 26.87 | 0.969 |
| **SMPLSign-X** | **24.16** | **49.12** | **22.09** | **0.982** |

**Deaf User Ratings (1-5 scale)**:

| Dataset | Method | Naturalness | Smoothness | Similarity |
|--------|------|--------|--------|--------|
| P-2014T | SMPLify-X | 1.52 | 1.98 | 2.41 |
| P-2014T | **Ours** | **3.58** | **4.04** | **3.94** |
| CSL | SMPLify-X | 1.27 | 1.75 | 1.69 |
| CSL | **Ours** | **3.78** | **4.14** | **3.78** |

### Ablation Study
- **Contributions of the three loss functions**: Removing $\mathcal{L}_{unseen}$ causes the largest performance drop (Dev BLEU-4 decreases from 24.16 to 22.57), while $\mathcal{L}_{upright}$ and $\mathcal{L}_{smooth}$ each contribute around 1 BLEU point.
- **Importance of Sign Connector**: Direct concatenation without the connector drops Dev BLEU-4 from 24.16 to 20.69 (a decrease of 3.5 points).
- **Importance of Sign Retrieval**: Replacing optimal retrieval with random selection drops Dev BLEU-4 from 24.16 to 22.25.
- **Co-articulation modeling in 3D is superior to 2D**: The default 3D connector has an L1 prediction error of 1.04, compared to 1.83 for fixed length, 1.22 for hand-only, and 1.34 for without coordinate distance.
- **3D Keypoint Enhancement**: Improves top-1 accuracy by around 1.2% - 1.5% on WLASL and MSASL.
- **Multi-View Understanding**: Front + side dual-stream input improves Dev BLEU-4 by about 1 point on P-2014T compared to front-only input.

## Highlights & Insights
- **Simplicity of the Retrieval-Connection Paradigm**: No end-to-end generation of sign language sequences is required. Instead, the task is decomposed into "translation $\to$ retrieval $\to$ concatenation $\to$ rendering", where each step can be independently optimized and replaced, making it highly engineering-friendly.
- **Sign-language Specific Priors of SMPLSign-X**: Three simple regularization losses (unseen/upright/smooth) allow the model to significantly outperform general 3D pose estimators, demonstrating the importance of domain priors.
- **Dynamic Duration Prediction by Sign Connector**: Predicting the co-articulation frame count with a 4-layer MLP yields much better results than fixed interpolation, with virtually no extra overhead.
- **Inspiring Byproduct Utility**: 3D representations naturally support rotation augmentation and multi-view inputs, easily providing free data augmentation for sign language understanding tasks.

## Limitations & Future Work
- **Data Scarcity**: Insufficient text-gloss pairs for training can limit the quality of Text2Gloss translation, with the construction of large-scale sign language datasets being a major bottleneck.
- **Quality of 2D Keypoint Pseudo-labels**: 3D estimation relies on 2D keypoints from HRNet; inaccurate 2D detections degrade the 3D results. Sign-specific 2D detectors might yield better performance.
- **Unresolved Depth Ambiguity**: Monocular-to-3D depth estimation remains an open problem, which the current method primarily mitigates through prior constraints.
- **Coarse Co-articulation Modeling**: Simple linear interpolation cannot fully capture the non-linear dynamics of real co-articulations. Learning-based transition generation could be explored.
- **Lack of Facial Expressions and Non-Manual Signals**: Facial expressions carry crucial semantic meaning in sign language, but current system modeling for them remains limited.
- **Support Only Seen Glosses**: The retrieval-based framework cannot handle out-of-vocabulary glosses, limiting its generalizability.

## Related Work & Insights
- **vs. FS-Net**: FS-Net uses an external dictionary, fixed-length 2D keypoint interpolation, and GANs to generate 2D videos. This work builds a cleaner dictionary via CTC segmentation, applies dynamic 3D space interpolation, and renders using an Avatar. This avoids the distortions of generative models, heavily outperforming FS-Net in back-translation (24.16 vs 16.92 Dev BLEU-4).
- **vs. SignDiff**: SignDiff introduces diffusion models to generate 2D sign videos, outperforming FS-Net but still limited to 2D. Under the same back-translator, the proposed method exceeds SignDiff by over 4 BLEU points while producing a more flexible 3D output.
- **vs. SMPLify-X / SMPLer-X / OSX**: General 3D human pose estimators fail to account for sign-language properties (upright posture, static lower body, temporal coherence). By adding three targeted constraints, this method significantly outperforms them in both back-translation and visual quality.

## Insights & Connections
- The modular retrieval-connection paradigm is applicable to other tasks requiring long-sequence action generation (e.g., dance generation, gesture synthesis), where each atomic action can be independently estimated and concatenated.
- Domain-specific priors can be naturally integrated into general parametric models (e.g., SMPL-X) through regularization losses without changing the architecture.
- The rotation augmentation idea of 3D representations can be transferred to other 2D keypoint-based human understanding tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce 3D Avatars to Spoken2Sign, though the overall pipeline is a combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering back-translation, 3D estimation comparison, ablation studies, user studies, and byproduct validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely well-structured, with clear concepts, intuitive flowcharts, and solid background introduction.
- Value: ⭐⭐⭐⭐ Establishes a 3D baseline for Spoken2Sign and releases the dictionary dataset, facilitating subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](../../CVPR2025/human_understanding/lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)
- [\[CVPR 2026\] Learning Effective Sign Features without Text for Gloss-free Sign Language Translation](../../CVPR2026/human_understanding/learning_effective_sign_features_without_text_for_gloss-free_sign_language_trans.md)
- [\[CVPR 2026\] BoostSLT: Boosting Sign Language Translation via a Plug-and-Play Diffusion-Based Semantic Enhancer](../../CVPR2026/human_understanding/boostslt_boosting_sign_language_translation_via_a_plug-and-play_diffusion-based_.md)
- [\[ECCV 2024\] GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths](gazexplain_learning_to_predict_natural_language_explanations_of_visual_scanpaths.md)
- [\[ECCV 2024\] SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](scape_a_simple_and_strong_category-agnostic_pose_estimator.md)

</div>

<!-- RELATED:END -->
