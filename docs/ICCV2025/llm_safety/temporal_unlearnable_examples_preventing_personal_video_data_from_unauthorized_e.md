---
title: >-
  [Paper Note] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation
description: >-
  [ICCV 2025][LLM Safety][video data privacy] This paper presents the first study on preventing video data from being exploited by deep trackers without authorization. It proposes a DiT-based generative framework for producing Temporal Unlearnable Examples (TUE), employing a temporal contrastive loss to induce trackers to rely on perturbation noise for temporal matching rather than learning genuine data structure. The method achieves strong transferability across models, datasets, and tasks.
tags:
  - ICCV 2025
  - LLM Safety
  - video data privacy
  - unlearnable examples
  - visual object tracking
  - generative perturbation
  - contrastive learning
date: 2026-05-08
content_hash: 70b801fc072c26a3
---

# Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation

**Conference**: ICCV 2025
**arXiv**: [2507.07483](https://arxiv.org/abs/2507.07483)
**Code**: None
**Area**: LLM Safety
**Keywords**: video data privacy, unlearnable examples, visual object tracking, generative perturbation, contrastive learning

## TL;DR
This paper presents the first study on preventing video data from being exploited by deep trackers without authorization. It proposes a DiT-based generative framework for producing Temporal Unlearnable Examples (TUE), employing a temporal contrastive loss to induce trackers to rely on perturbation noise for temporal matching rather than learning genuine data structure. The method achieves strong transferability across models, datasets, and tasks.

## Background & Motivation
With the proliferation of social media, large volumes of user-uploaded videos are collected to train commercial Visual Object Tracking (VOT) models. Large-scale VOT datasets such as TrackingNet, LaSOT, and GOT-10k are primarily composed of such user videos, raising serious **privacy and copyright concerns** — personal videos are used for tracker training without consent, and tracking data often involves sensitive trajectories (e.g., individual movements, vehicle routes, military targets).

**Limitations of Prior Work**: Unlearnable Examples (UE) have been extensively studied in image classification — imperceptible perturbations are added to training data to render it "unlearnable." However, directly extending image UE to video introduces three challenges:

**Efficiency**: Video data is high-resolution and multi-frame; conventional iterative optimization (EM-based) UE generation is computationally intensive. On GOT-10k, the EM method requires 33 hours and 3.4 GB of parameter storage.

**Effectiveness**: VOT relies on cross-frame temporal matching (rather than semantic recognition as in image classification), and target objects undergo large scale variations across frames, making it harder to design UE that supports scale-invariant matching.

**Transferability**: Existing UE methods are optimized for specific classification models and generalize poorly across different tracker architectures, datasets, and tasks.

**Core Idea: Train a lightweight generator (rather than optimizing per sample) to produce target-aware temporal perturbations; combined with temporal contrastive learning, trackers trained on the protected data learn "shortcut matching" driven by perturbation noise rather than genuine visual structure.**

## Method

### Overall Architecture
The pipeline consists of four stages: (1) training a TUE generator using the proxy tracker SiamFC and a tracking dataset; (2) applying the trained generator to produce protective perturbations for user videos; (3) the unauthorized party training a tracker on TUE-augmented data; (4) evaluating on clean test data — data protected by TUE should cause severe degradation in tracker performance.

### Key Designs
1. **DiT-based TUE Generator**:

    - Function: Generates target-aware imperceptible perturbations added to video frames to corrupt tracker training.
    - Mechanism: Employs a DiT-S/8 architecture (12 layers, 6-head attention, hidden dimension 384), taking the target crop and normalized bounding box state $\tilde{b}_i$ as input, generating perturbations in a single forward pass:
    $\hat{z} = z + G_w(z, \tilde{b}_i), \quad \hat{x} = \Phi(x, G_w(c(x, b_j), \tilde{b}_j), b_j)$
      The optimization objective minimizes the tracking loss (inducing the tracker to easily learn perturbation shortcuts):
    $\min_w \mathcal{L}(f_\theta(\hat{z}) * f_\theta(\hat{x}), y)$
    - Design Motivation: Compared with EM methods that iteratively optimize perturbations per video (requiring $T$-step PGD), the generator approach follows a **train-once, generate-everywhere** paradigm. Parameter count is reduced from 3.4 GB to 124 MB (28× compression) and training time from 33 hours to 7 hours (4× speedup). More critically, the generator supports **zero-shot transfer** to unseen datasets.

2. **Target-State Conditioning**:

    - Function: Injects bounding box information of the target as a condition into the generator, enabling perturbations to adapt to targets of varying scales.
    - Mechanism: A fully connected layer maps the normalized target state $\tilde{b} \in \mathbb{R}^4$ (comprising normalized top-left coordinates, width, and height) to the hidden space as the conditional input to DiT.
    - Design Motivation: Targets undergo large scale variations across frames in VOT. Ablation experiments show that removing target conditioning ("- Condi.") leads to a notable drop in protection performance (AO: 22.5 vs. 16.1), confirming that dynamic adaptation to target state is critical for learning effective TUE.

3. **Temporal Contrastive Loss (TCL)**:

    - Function: Further reinforces tracker dependency on TUE noise, enlarging the distributional gap between clean and TUE samples.
    - Mechanism: The template TUE $\hat{z}$ serves as the anchor, the TUE search region $\hat{e}$ as the positive sample, and clean templates/targets from the same or other videos as negative samples:
    $\min_w [\mathcal{L}(f_\theta(\hat{z}) * f_\theta(\hat{x}), y) + \lambda \mathcal{L}_{cl}(f_\theta(\hat{z}), f_\theta(\hat{e}))]$
      where $\lambda = 0.05$.
    - Design Motivation: Error minimization alone may be insufficient. TCL creates a larger distributional margin by pulling TUE features closer together and pushing clean features away. Trackers trained on TUE data can only learn noise-driven matching patterns and naturally fail on clean test data.

4. **Context-Aware Perturbation**:

    - Function: Applies perturbations simultaneously to both the target region and the surrounding context region.
    - Mechanism: Modern trackers leverage not only the central target but also the surrounding context for template matching. Perturbing only the target region is insufficiently effective (AUC: 39.6 vs. 29.5); the contextual information must also be corrupted.
    - Design Motivation: Trackers can learn adequate temporal matching cues from the context region. Ablation results confirm that adding context noise further reduces OTB AUC from 39.6 to 29.5.

### Loss & Training
- Alternating optimization: each iteration first updates the generator $G_w$ (inner loop), then updates the proxy tracker $f_\theta$ (outer loop).
- Generator is trained for 50 epochs, learning rate $5 \times 10^{-6}$, batch size 16.
- Proxy tracker: SiamFC (chosen for training efficiency and good generator generalization).
- Perturbation constraint: $L_\infty$ norm $\leq 8/255$, imperceptible to the human eye.
- Training data: GOT-10k; completed in 7 hours on a single RTX 4090.

## Key Experimental Results

### Main Results
Tracker performance after training on protected data (lower values indicate stronger protection):

| Tracker | Method | GOT-10k AO↓ | GOT-10k SR0.5↓ | OTB AUC↓ | LaSOT AUC↓ |
|--------|------|-------------|----------------|----------|-----------|
| SiamFC | Clean | 35.5 | 39.0 | 58.6 | 34.0 |
| SiamFC | EM | 21.4 | 18.2 | 29.5 | 17.6 |
| SiamFC | **TUE+TCL** | **12.1** | **9.0** | **11.4** | **9.5** |
| OSTrack-256 | Clean | 71.0 | 80.4 | 67.4 | 62.3 |
| OSTrack-256 | EM | 26.3 | 24.0 | 48.8 | 29.9 |
| OSTrack-256 | **TUE+TCL** | **18.0** | **15.1** | **30.5** | **22.0** |
| SeqTrack-256 | Clean | 74.7 | 84.7 | 68.1 | 63.6 |
| SeqTrack-256 | **TUE** | **2.3** | **0.7** | - | - |
| DropTrack-384 | Clean | 75.9 | 86.8 | 69.4 | 66.5 |
| DropTrack-384 | **TUE** | **17.1** | **12.9** | **36.7** | **25.2** |

### Ablation Study

| Configuration | GOT-10k AO↓ | OTB AUC↓ | Notes |
|------|-------------|----------|------|
| EM baseline (target only) | 27.0 | 39.6 | Conventional method |
| EM + Context | 21.4 | 29.5 | With context noise |
| TUE Generator | 16.1 | 17.6 | Generative approach |
| TUE - Condition | 22.5 | 19.7 | Without target conditioning |
| **TUE + TCL** | **12.1** | **11.4** | Full method |

Model complexity comparison:

| Method | Training Time | Learnable Parameters |
|------|---------|------------|
| EM + Context | 33 hours | 3.4 GB |
| **TUE Generator** | **7 hours** | **124 MB** |

### Key Findings
- **Generative approach substantially outperforms iterative optimization**: The TUE Generator comprehensively surpasses the EM baseline in both efficiency (4× speedup, 28× parameter compression) and protection effectiveness.
- **TCL is a critical enhancement**: On SiamFC, adding TCL further reduces GOT-10k AO from 16.1 to 12.1.
- **Strong cross-model transferability**: A generator trained on the simple SiamFC effectively degrades performance of complex trackers (OSTrack, SeqTrack, DropTrack).
- **Zero-shot cross-dataset transfer**: A generator trained on GOT-10k can be directly applied to generate TUE for other datasets without retraining.
- **Cross-task transfer**: TUE not only disrupts VOT but also transfers to Video Object Segmentation (VOS) and long-term point tracking tasks.
- An effective TUE generator can be trained using only 25% of GOT-10k videos (approximately 2,300 sequences).

## Highlights & Insights
- **Socially significant problem formulation**: This is the first study to address video data privacy in the VOT domain, with practical urgency in the era of large-scale foundation models.
- **Elegant substitution of iterative optimization with generative modeling**: A single DiT forward pass replaces multi-step PGD iteration, achieving both higher efficiency and stronger protection.
- **Clear intuition behind temporal contrastive learning**: By pulling TUE pairs closer and pushing clean samples away, the method maximizes the distributional gap between "shortcut matching" and "genuine matching."
- **Lightweight proxy, strong transfer**: Training on the simplest SiamFC yields protection against exploitation by state-of-the-art trackers, suggesting that TUE captures universal vulnerabilities inherent to the tracking task.

## Limitations & Future Work
- Perturbations are applied only to the target/context region; new designs are needed for full-frame perturbation scenarios (e.g., action recognition protection).
- The $L_\infty \leq 8/255$ constraint may be destroyed by high-compression video encoding (e.g., after uploading to YouTube).
- Robustness against defense strategies based on adversarial training or perturbation removal (e.g., adversarial training, image purification) is not sufficiently discussed.
- Protection effectiveness exhibits slight fluctuations for large models such as SeqTrack when trained for more epochs; long-training stability warrants further attention.
- In practical deployment, users must be capable of generating and applying TUE, and usability requires improvement.

## Related Work & Insights
- **Extending unlearnable examples from images to video** opens a new research direction; protecting temporal matching is considerably more complex than protecting classification.
- **Using DiT as a lightweight noise generator** is noteworthy: no multi-step diffusion is required, and a single forward pass suffices to produce effective perturbations.
- **The intersection of contrastive learning and data privacy** merits further exploration — contrastive signals can effectively amplify "shortcut" effects.
- The proposed method may also serve as a reference for other temporal matching tasks such as video retrieval and Re-ID.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First study on video privacy protection in the VOT domain; both the generative TUE framework and TCL design are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive transferability validation across models, datasets, and tasks.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, method description is detailed, and pipeline diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ High problem importance, strong practical utility, and meaningful implications for AI security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](../../ICLR2026/llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[ICCV 2025\] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[NeurIPS 2025\] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](../../NeurIPS2025/llm_safety/enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)
- [\[NeurIPS 2025\] When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](../../NeurIPS2025/llm_safety/when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)

</div>

<!-- RELATED:END -->
