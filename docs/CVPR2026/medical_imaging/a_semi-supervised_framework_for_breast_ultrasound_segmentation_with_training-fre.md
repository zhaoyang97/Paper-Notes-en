---
title: >-
  [Paper Note] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement
description: >-
  [CVPR 2026][Medical Imaging][Semi-supervised segmentation] This paper proposes a semi-supervised framework for breast ultrasound (BUS) image segmentation. It employs GPT-5-generated appearance descriptions combined with…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Semi-supervised segmentation"
  - "breast ultrasound"
  - "pseudo-labels"
  - "dual-teacher framework"
  - "contrastive learning"
  - "SAM"
  - "Grounding DINO"
date: 2026-05-08
content_hash: 82fc073ad1cbffbb
---

# A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement

**Conference**: CVPR 2026
**arXiv**: [2603.06167](https://arxiv.org/abs/2603.06167)  
**Code**: To be confirmed  
**Area**: Medical Imaging
**Keywords**: Semi-supervised segmentation, breast ultrasound, pseudo-labels, dual-teacher framework, contrastive learning, SAM, Grounding DINO

## TL;DR

This paper proposes a semi-supervised framework for breast ultrasound (BUS) image segmentation. It employs GPT-5-generated appearance descriptions combined with Grounding DINO and SAM for training-free pseudo-label generation (APPG), and refines labels via a dual-teacher framework (static + dynamic) using Uncertainty-Entropy Weighted Fusion (UEWF) and Adaptive Uncertainty-guided Reverse Contrastive Learning (AURCL). The method approaches fully supervised performance using only 2.5% labeled data.

## Background & Motivation

### 1. State of the Field
Breast ultrasound (BUS) is a critical imaging modality for breast cancer screening, and accurate tumor segmentation is fundamental to computer-aided diagnosis. Deep learning methods rely on large-scale pixel-level annotations, yet annotation in medical imaging is extremely costly—requiring expert radiologists to label images pixel by pixel in a time-consuming and expensive process. Semi-supervised learning (SSL) alleviates this by leveraging abundant unlabeled data alongside a small amount of labeled data, but faces particular challenges in the BUS setting.

### 2. Limitations of Prior Work
BUS images present unique difficulties: (1) low contrast between tumors and surrounding tissue with blurry boundaries; (2) high morphological variability across tumors (oval, round, lobulated); and (3) inherent speckle noise and artifacts in ultrasound. These factors severely undermine the core assumption of SSL methods—that a model can learn reliable pseudo-labels from limited annotations—especially in the extremely low-label regime (e.g., 2.5%), where pseudo-label quality is poor and the model falls into a confirmation bias cycle.

### 3. Root Cause
Conventional SSL methods (e.g., Mean Teacher) rely on the model itself to generate pseudo-labels, but the model is inherently unreliable under extremely sparse annotation, producing noisy pseudo-labels that further mislead training. This is a chicken-and-egg dilemma: good pseudo-labels are needed to train a good model, yet good pseudo-labels presuppose a good model.

### 4. Paper Goals
(1) Obtain high-quality initial pseudo-labels under extreme label scarcity to break the cold-start dilemma; (2) continuously refine pseudo-labels during training to avoid single-teacher confirmation bias; (3) enhance the model's discriminative capability in boundary-uncertain regions.

### 5. Starting Point
The paper exploits vision-language foundation models (GPT-5 + Grounding DINO + SAM) as a training-free pseudo-label generator to bypass the cold-start phase, followed by a dual-teacher framework with uncertainty-aware fusion for continuous refinement.

### 6. Core Idea
The problem is addressed in three steps: (1) APPG leverages general appearance priors of breast tumors, translating them into natural language prompts that drive foundation models to generate training-free pseudo-labels; (2) a static teacher (frozen after pseudo-label warmup) and a dynamic teacher (updated via EMA) provide complementary perspectives; (3) UEWF adaptively fuses the two teachers' outputs weighted by uncertainty, while AURCL employs reverse contrastive learning to specifically enhance boundary discrimination.

## Method

### Overall Architecture

The framework comprises three stages: (1) **APPG pseudo-label generation**: GPT-5 descriptions → Grounding DINO localization → SAM segmentation, producing initial pseudo-labels for all unlabeled data; (2) **Warmup training**: the model is trained to convergence on labeled data plus pseudo-labeled data, then frozen as the static teacher $T^A$; (3) **Dual-teacher semi-supervised training**: $T^A$ (frozen) and $T^B$ (EMA) jointly generate pseudo-labels for the student network; these are fused via UEWF to supervise student training, with AURCL further enhancing learning in uncertain regions.

### Key Designs

#### 1. APPG (Appearance-Prompted Pseudo-Label Generation)

**Function**: Leverages general appearance knowledge of breast tumors to generate segmentation pseudo-labels for unlabeled BUS images without any training.

**Mechanism**: Breast tumors in ultrasound exhibit predictable appearance characteristics—hypoechoic (dark) regions typically shaped as oval, round, or lobulated structures. GPT-5 converts this medical knowledge into natural language descriptions (e.g., "dark oval region," "dark round mass," "dark lobulated area"), which serve as text prompts for Grounding DINO to perform open-vocabulary object detection and produce bounding boxes. These bounding boxes are then passed as spatial prompts to SAM (Segment Anything Model), which outputs pixel-level segmentation masks used as pseudo-labels.

**Design Motivation**: (1) Entirely training-free, requiring no annotated data, by exploiting the zero-shot capability of VLMs; (2) the appearance characteristics of breast tumors are universal (hypoechoic dark regions across all BUS images), making them well-suited to generic textual descriptions; (3) Grounding DINO excels at open-vocabulary detection while SAM excels at prompt-based precise segmentation, making their combination naturally complementary.

#### 2. Dual-Teacher Framework

**Function**: Provides two complementary pseudo-label sources to avoid single-teacher confirmation bias.

**Mechanism**:
- **Static teacher $T^A$**: After warmup training on APPG pseudo-labels plus limited ground-truth annotations, its weights are fully frozen. It encodes initial segmentation knowledge derived from foundation models, unaffected by subsequent training noise, providing a stable pseudo-label baseline.
- **Dynamic teacher $T^B$**: Initialized identically, it continuously tracks student model updates via exponential moving average (EMA). It captures knowledge acquired during training and adapts to distribution shifts, but may accumulate errors.
- Each teacher independently generates pseudo-labels $\hat{y}^A$ and $\hat{y}^B$, which are fused into the final pseudo-label $\hat{y}^F$ via UEWF.

**Design Motivation**: A single EMA teacher (e.g., Mean Teacher) under extreme label scarcity is prone to a degenerative cycle—noisy pseudo-labels → biased student → biased EMA teacher → worse pseudo-labels. The static teacher serves as an anchor independent of the training process, breaking this cycle.

#### 3. UEWF (Uncertainty-Entropy Weighted Fusion)

**Function**: Adaptively fuses pseudo-labels from the two teachers according to their respective prediction confidence.

**Mechanism**: The information entropy $H^A$ and $H^B$ of each teacher's prediction is computed per pixel. Lower entropy indicates greater certainty. Inverse entropy is used as the weighting scheme:

$$\hat{y}^F = w^A \cdot \hat{y}^A + w^B \cdot \hat{y}^B, \quad w^A = \frac{H^B}{H^A + H^B}, \quad w^B = \frac{H^A}{H^A + H^B}$$

For each pixel, the teacher with higher uncertainty (higher entropy) receives lower weight, while the more confident teacher (lower entropy) receives higher weight, enabling per-pixel adaptive selection of the more reliable prediction.

**Design Motivation**: Different teachers exhibit different reliability in different regions—$T^A$ may be more stable in capturing overall shape, while $T^B$ may be more precise in fine-grained regions. Inverse-entropy weighting requires no additional parameters, is computationally simple, and naturally adapts per pixel.

#### 4. AURCL (Adaptive Uncertainty-Guided Reverse Contrastive Learning)

**Function**: Specifically targets uncertain regions such as blurry boundaries to enhance the model's discriminative capability.

**Mechanism**: Four steps: (1) **Uncertainty map computation**: multiple Monte Carlo Dropout passes through the student model are used to compute per-pixel variance as an uncertainty measure—higher variance indicates greater model uncertainty; (2) **Low-confidence region extraction**: pixels with uncertainty exceeding threshold $\tau$ are selected as the boundary regions where the model is undecided; (3) **Probability reversal (Reverse)**: predicted probabilities in low-confidence regions are inverted ($1-p$)—the intuition being that if the model uncertainly predicts a pixel as foreground, the inverted representation is more likely to correspond to background; (4) **Patch-level contrastive learning**: inverted low-confidence region patch features are contrasted against high-confidence foreground/background patch features—pulling them closer to features of the corresponding class and pushing them away from the other class.

**Design Motivation**: Standard SSL exploits high-confidence regions well but struggles with ambiguous boundary areas. AURCL explicitly targets these hard regions, cleverly converting uncertainty into a learnable signal via probability reversal. Contrastive learning constrains boundary representations in feature space, complementing pixel-level supervision losses.

### Loss & Training

- Labeled data: standard supervised segmentation loss (CE + Dice)
- Unlabeled data: segmentation loss on UEWF-fused pseudo-labels + AURCL contrastive loss
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_1 \mathcal{L}_{\text{unsup}} + \lambda_2 \mathcal{L}_{\text{AURCL}}$
- The student model is a U-Net variant; $T^B$ uses EMA decay rate $\alpha = 0.999$
- In APPG, three appearance descriptions are used per image to generate candidate boxes; NMS removes redundancy and the highest-confidence box is selected

## Key Experimental Results

### Main Results

Evaluated on four public BUS datasets (BUSI, UDIAT, BUS-BRA, TN3K) with labeling ratios of 2.5%, 5%, and 10%.

**Key results (Dice %)**:

| Method | BUSI 2.5% | BUSI 5% | BUSI 10% | UDIAT 2.5% | UDIAT 5% | UDIAT 10% |
|--------|-----------|---------|----------|------------|---------|-----------|
| Supervised-only | 51.2 | 60.8 | 69.4 | 53.7 | 63.2 | 71.5 |
| Mean Teacher | 58.6 | 66.3 | 73.8 | 60.4 | 68.1 | 75.2 |
| CPS | 59.1 | 67.0 | 74.1 | 61.2 | 69.3 | 75.8 |
| UniMatch | 62.4 | 69.5 | 76.2 | 64.0 | 71.8 | 78.1 |
| **Proposed** | **71.8** | **75.3** | **79.6** | **72.5** | **76.9** | **81.4** |
| Full Supervision | 80.2 | 80.2 | 80.2 | 82.1 | 82.1 | 82.1 |

Key findings: (1) Under 2.5% annotation, the proposed method (71.8% Dice on BUSI) substantially outperforms the strongest baseline UniMatch (62.4%), a gain of +9.4%; (2) with 2.5% labels, the method achieves 89.5% of full supervision performance (71.8/80.2), rising to 93.9% at 5%; (3) consistent improvements across all four datasets and all labeling ratios.

### Ablation Study

**Component ablation (BUSI 2.5% Dice)**:

| Configuration | Dice (%) |
|---------------|----------|
| Baseline (single-teacher Mean Teacher) | 58.6 |
| + APPG pseudo-label initialization | 65.2 |
| + Dual-teacher (simple average) | 67.8 |
| + UEWF (replacing simple average) | 69.5 |
| + AURCL (full method) | 71.8 |

Each component contributes clearly: APPG (+6.6%) > dual-teacher (+2.6%) > UEWF (+1.7%) > AURCL (+2.3%). Training-free pseudo-labels from APPG constitute the largest single contribution.

### APPG Pseudo-Label Quality

The average Dice between APPG-generated pseudo-labels and ground truth: BUSI 66.3%, UDIAT 68.7%. While imperfect, this substantially surpasses model predictions at random initialization (~35–40%), providing a strong starting point for subsequent training.

## Highlights & Insights

1. **VLMs as a free lunch**: The combination of GPT-5 + Grounding DINO + SAM translates domain knowledge (breast tumor appearance) into initial pseudo-labels at zero annotation cost, elegantly bypassing the cold-start dilemma.
2. **Static + dynamic teacher design is simple yet effective**: The static teacher acts as an anchor against drift, the dynamic teacher tracks learning progress, and UEWF performs per-pixel adaptive fusion—without requiring complex architectural modifications.
3. **The reversal operation in AURCL is insightful**: Converting "uncertain" predictions into "reversely certain" signals is a clever idea; contrastive learning constructs boundary-aware representations in feature space.
4. **Pronounced advantage under extreme label scarcity**: Achieving near-fully-supervised performance with only 2.5% labeled data (just a few annotated images) has significant practical value for medical settings where annotation resources are severely constrained.

## Limitations & Future Work

1. **APPG relies on the generality of appearance priors**: The approach is effective for breast tumors, but not all lesions exhibit uniformly hypoechoic appearance; extension to other organs or lesion types requires redesigning the prompts.
2. **Deployment cost of GPT-5 + Grounding DINO + SAM**: Although training-free, inference costs are non-trivial, particularly the API costs associated with GPT-5.
3. **Upper bound on pseudo-label quality**: APPG's ~67% Dice still leaves considerable room for improvement; SAM's segmentation accuracy on low-contrast ultrasound images is inherently limited.
4. **Stronger segmentation backbones not explored**: The method is primarily evaluated on U-Net; stronger architectures such as Swin-UNet and TransUNet have not been tested.
5. **Sensitivity of contrastive learning hyperparameters**: The effect of the uncertainty threshold $\tau$ and contrastive temperature in AURCL on performance is not sufficiently analyzed.

## Related Work & Insights

- **Evolution of SSL segmentation methods**: Mean Teacher → CPS (mutual learning) → UniMatch (multi-view consistency) → the proposed method (VLM pseudo-labels + dual-teacher). The trend is toward incorporating stronger priors to compensate for annotation scarcity.
- **VLMs in medical image analysis**: Methods such as MedSAM and SAMed fine-tune SAM for medical segmentation but require target-domain annotations; APPG is entirely training-free, better suited to the low-annotation regime.
- **The role of uncertainty in SSL**: Prior work primarily uses uncertainty for pseudo-label filtering (discarding uncertain samples); the reverse approach in AURCL transforms uncertain regions into exploitable positive signals, a strategy generalizable to other SSL frameworks.

## Rating

⭐⭐⭐⭐ The framework is complete with strongly complementary components. The APPG approach of leveraging VLMs for training-free pseudo-label generation is highly practical for low-annotation medical scenarios, and validation is comprehensive across four datasets. Limitations include the restricted generality of APPG's appearance priors and the absence of discussion on VLM inference costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](../../AAAI2026/medical_imaging/propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](../../NeurIPS2025/medical_imaging/self_iterative_label_refinement_via_robust_unlabeled_learning.md)

</div>

<!-- RELATED:END -->
