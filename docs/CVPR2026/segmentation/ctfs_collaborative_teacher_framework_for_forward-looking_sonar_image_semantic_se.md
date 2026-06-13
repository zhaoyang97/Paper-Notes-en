---
title: >-
  [Paper Note] CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels
description: >-
  [CVPR 2026][Segmentation][Sonar image segmentation] This paper proposes CTFS, the first semi-supervised semantic segmentation framework specifically designed for forward-looking sonar (FLS) images. It introduces a multi-…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Sonar image segmentation"
  - "semi-supervised learning"
  - "multi-teacher collaboration"
  - "pseudo-label reliability"
  - "extremely limited annotations"
date: 2026-05-08
content_hash: b1e16ad4b10f3743
---

# CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels

**Conference**: CVPR 2026
**arXiv**: [2603.21071](https://arxiv.org/abs/2603.21071)  
**Code**: Available  
**Area**: Segmentation / Underwater Imaging
**Keywords**: Sonar image segmentation, semi-supervised learning, multi-teacher collaboration, pseudo-label reliability, extremely limited annotations

## TL;DR
This paper proposes CTFS, the first semi-supervised semantic segmentation framework specifically designed for forward-looking sonar (FLS) images. It introduces a multi-teacher collaboration mechanism (one general teacher + two sonar-specific teachers simulating acoustic shadow and energy attenuation, respectively), combined with multi-view pseudo-label reliability assessment (intra-teacher stability × inter-teacher consistency). With only 2% labeled data, CTFS achieves 62.32% mIoU, surpassing the state of the art by 5.08 percentage points.

## Background & Motivation

**Background**: Forward-looking sonar is a key technology for underwater perception (rescue, detection, biological survey). Sonar images present unique challenges—severe speckle noise, low texture contrast, acoustic shadows, and geometric distortion. Sonar data is highly specialized and expensive to annotate.

**Limitations of Prior Work**: (1) Conventional teacher-student semi-supervised methods (Mean Teacher/FixMatch) do not account for sonar characteristics, causing teachers to generate large numbers of low-quality pseudo-labels on noisy sonar images. (2) Some methods under 10% labeled data even underperform fully supervised baselines—semi-supervised frameworks can be counterproductive in the sonar domain. (3) Teacher model designs tailored to the physical characteristics of sonar imaging are absent.

**Key Challenge**: The weak/strong augmentation strategies of standard semi-supervised methods are designed for natural images (e.g., color jitter) and are ineffective or even harmful for sonar images—sonar "noise" originates from the physical imaging process rather than image processing artifacts.

**Key Insight**: Augmentation strategies for the teacher must be grounded in the physics of sonar imaging; acoustic shadow and energy attenuation are the two dominant physical characteristics of sonar images.

**Core Idea**: (1) Three-teacher collaboration—a general teacher, an acoustic shadow teacher, and an energy attenuation teacher, rotating in a fixed order to guide the student; (2) Multi-view pseudo-label reliability assessment—intra-teacher stability × inter-teacher consistency → reliable pseudo-labels.

## Method

### Overall Architecture
Labeled data → supervised student training ($\mathcal{L}_{sup}$); unlabeled data → three teachers activated in rotation $\phi(e)$: after epoch $E$, cycling as $T_{general} \to T_{sonar\_a} \to T_{sonar\_b}$ every 3 epochs → weakly augmented inputs fed to the active teacher to generate pseudo-labels → strongly augmented inputs fed to the student → multi-view reliability assessment → reliability-weighted unsupervised loss ($\mathcal{L}_{unsup}$) → total loss $\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_u \cdot \mathcal{L}_{unsup}$.

### Key Designs

1. **Collaborative Teacher Strategy (CBTS)**:

    - **General teacher $T_{general}$**: Standard geometric transformations and color perturbations as weak augmentation—learns general semantic representations.
    - **Acoustic shadow teacher $T_{sonar\_a}$**: Simulates shadows produced when sonar is occluded by obstacles. Within a fan-shaped region, intensity decays with distance: $I_o(x,y) = I_i(x,y) \times [1-\alpha(1-d/R)]$, where $d$ is the distance to the shadow origin and $R=0.2\times\min(H,W)$.
    - **Energy attenuation teacher $T_{sonar\_b}$**: Simulates the directional energy attenuation of acoustic waves propagating through seawater: $I_o(x,y) = I_i(x,y) \times (1-\gamma \times y/H)$, applying linear attenuation along the vertical direction.
    - **Rotation strategy**: The three teachers cycle in the fixed order general → acoustic shadow → energy attenuation; each teacher is updated via EMA.
    - **Design Motivation**: (1) The general teacher provides a foundation for generic semantic learning; (2) sonar-specific teachers train the student to acquire sonar-invariant representations—correctly segmenting even under acoustic shadow or energy attenuation; (3) mapping each teacher to a specific physical perturbation preserves feature-level semantic consistency.

2. **Multi-View Reliability Assessment (MVRA)**:

    - **Function**: Dynamically quantifies pseudo-label reliability, replacing simple single-threshold confidence filtering.
    - **Grid partitioning**: Images are divided into $m \times m$ grid cells; prediction probabilities within each cell are averaged to reduce computational cost.
    - **Intra-teacher stability**: Cosine similarity between a teacher's predictions on the original image and multiple augmented views: $r_{ij}^t = \frac{1}{N_{A_w^t}} \sum_k \cos(f_{ij}^{ot}, f_{ij}^{kt})$.
    - **Inter-teacher consistency**: Pairwise cosine similarity among the three teachers' predictions on the same image: $C_{ij} = \frac{1}{N_\mathcal{D}} \sum_{(p,q)} \cos(f_{ij}^{op}, f_{ij}^{oq})$.
    - **Composite reliability**: $R_{ij} = \Pi(C_{ij}) \times \frac{1}{N_T}\sum_t r_{ij}^t$, where the penalty term $\Pi(C_{ij}) = \delta + (1-\delta)C_{ij}$ penalizes low inter-teacher consistency.
    - Grid-level scores are then broadcast to the pixel level.
    - **Design Motivation**: The characteristics of sonar images render single-teacher confidence unreliable; only multi-dimensional assessment (self-stability + multi-teacher consensus) can effectively filter noisy pseudo-labels.

3. **Reliability-Guided Adaptive Constraint**:

    - The unsupervised loss employs reliability in a **dual role**: (1) hard-threshold filtering $\Delta = \mathbb{1}[R_b^n > \psi]$; (2) soft reliability weighting $\times R_b^n$.
    - **Design Motivation**: Combines filtering with fine-grained weighting—high-reliability regions receive strong supervision; low-reliability regions receive weak supervision or are ignored.

4. **FSSG New Dataset**:

    - Collected in Bohai Bay using a multi-beam forward-looking sonar (Oculus M750d, 750 kHz / 1.2 MHz) mounted on an ROV.
    - 3,761 images, 11 object categories (including a diver class), with a long-tailed distribution.
    - Captured at distances of 2–15 m from various angles to increase diversity.

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_u \cdot \mathcal{L}_{unsup}$$
where
$$\mathcal{L}_{unsup} = \frac{1}{N_u}\sum_n \frac{\sum_b \text{CE}(p_n^s[b], p_n^t[b]) \times R_b^n \times \Delta}{N_p}.$$

## Key Experimental Results

### Main Results (mIoU%)

| Method | Backbone | FLSMD 2% | FLSMD 5% | FLSMD 10% | FSSG 2% | FSSG 5% |
|--------|----------|:---:|:---:|:---:|:---:|:---:|
| Labeled Only | DINOv2-S | 51.08 | 61.02 | 68.48 | 35.61 | 41.67 |
| AEL | RN-101 | 52.70 | 64.89 | 70.73 | 53.51 | 57.84 |
| UniMatch V2 | DINOv2-S | 57.24 | 66.49 | 69.81 | 58.78 | 61.21 |
| SemiVL | RN-101 | 53.38 | 65.16 | 70.02 | 58.85 | 62.19 |
| **CTFS (Ours)** | DINOv2-S | **62.32** | **68.08** | **72.27** | **59.53** | **65.12** |

Under 2% labeled data, CTFS outperforms UniMatch V2 by **5.08%** mIoU.

### Ablation Study

| Configuration | FLSMD 2% mIoU |
|---------------|:---:|
| General teacher only (baseline) | ~57 |
| + Acoustic shadow teacher | ~59 |
| + Energy attenuation teacher | ~60 |
| + MVRA reliability assessment | ~62 |
| **Full CTFS** | **62.32** |

### Tail-Class Performance
The most significant improvements are observed on few-shot categories such as hook, shampoo bottle, tire, and ROV, indicating that multi-teacher collaboration enhances representation learning for rare classes.

### Key Findings
- **Each sonar-specific teacher contributes independently**: the acoustic shadow teacher yields ~+2% and the energy attenuation teacher ~+1%—physical augmentations genuinely help the student understand sonar imaging.
- **MVRA is more reliable than single-threshold confidence filtering**—multi-view assessment effectively filters spuriously high-confidence pseudo-labels induced by sonar noise.
- **Tail-class improvements are most pronounced**—the multi-angle supervision signals provided by multiple teachers are especially beneficial for few-shot categories.
- **CTFS consistently achieves state-of-the-art performance across all annotation ratios**—leading at 2%/5%/10%, demonstrating the robustness of the framework.

## Highlights & Insights
- **Acoustic physics simulation as a data augmentation paradigm**: Explicitly modeling acoustic shadow and energy attenuation—the two core physical processes of sonar imaging—as weak augmentations for the teachers is an excellent example of injecting domain physical knowledge into semi-supervised learning. This paradigm is transferable to other imaging modalities such as SAR (scattering), infrared (thermal conduction), and ultrasound (acoustic impedance).
- **First semi-supervised segmentation framework for sonar**: Fills the gap in semi-supervised learning for underwater sonar perception. Sonar annotation is extremely costly (requiring expert knowledge), making semi-supervised methods highly valuable for this domain.
- **"General-to-specific" curriculum via teacher rotation**: The general → sonar_a → sonar_b cycle enables the student to progressively develop from generic competence to domain-specific understanding—analogous to a coarse-to-fine learning process.
- **FSSG dataset contribution**: 3,761 images with 11 categories and semantic segmentation annotations, alleviating the scarcity of sonar datasets.

## Limitations & Future Work
- The fixed rotation period (every 3 epochs) and fixed order of the three teachers may be suboptimal—adaptive scheduling (based on student performance) or parallel multi-teacher inference may be more effective.
- The acoustic physics simulations are simplified approximations (fan-shaped shadow, linear attenuation)—more accurate acoustic simulation (accounting for frequency, reflector material, etc.) could further improve performance.
- Validation is limited to indoor tank and shallow-water scenarios—generalization to complex deep-sea environments (current interference, multipath reflection, biofouling) remains to be evaluated.
- The FSSG dataset, though valuable, is still limited in scale (3,761 images)—acoustic simulation could be used to generate additional synthetic data.
- Deeper integration of multi-teacher collaboration with the intrinsic properties of foundation models (e.g., DINOv2) is worth exploring.

## Related Work & Insights
- **vs. Mean Teacher / UniMatch V2**: General-purpose semi-supervised methods disregard sonar-specific characteristics and therefore yield limited performance in the sonar domain. The sonar-specific teachers in CTFS directly improve domain adaptation.
- **vs. Dual Teacher**: Employs two structurally different teachers but does not design augmentations based on sonar physical characteristics.
- **vs. FLSMD dataset**: CTFS substantially outperforms all existing methods on this benchmark.
- **vs. Beyond-Pixels / SemiVL**: These methods underperform the supervised baseline on sonar data, confirming the particular difficulty of semi-supervised learning in the sonar domain.
- **Takeaway**: Any domain with distinctive imaging physics (SAR / infrared / ultrasound) can adopt the "physics-augmented teacher" paradigm to design domain-specific semi-supervised frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐ The sonar physics teachers combined with multi-view reliability assessment are innovative; this is the first semi-supervised segmentation framework for sonar.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, multiple annotation ratios, tail-class analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The sonar physics background and augmentation design are described clearly.
- Value: ⭐⭐⭐⭐ Practically valuable for underwater perception; the physics-augmented teacher idea has broad generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [\[NeurIPS 2025\] COS3D: Collaborative Open-Vocabulary 3D Segmentation](../../NeurIPS2025/segmentation/cos3d_collaborative_open-vocabulary_3d_segmentation.md)

</div>

<!-- RELATED:END -->
