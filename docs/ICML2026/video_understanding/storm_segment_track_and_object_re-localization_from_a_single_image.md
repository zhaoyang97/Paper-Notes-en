---
title: >-
  [Paper Note] STORM: Segment, Track, and Object Re-Localization from a Single Image
description: >-
  [ICML 2026][Video Understanding][Reference-conditioned 6D tracking] STORM proposes a "single reference image" 6D pose tracking framework: hierarchical spatial fusion attention (HSFA) aligns reference-query features (prod…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Reference-conditioned 6D tracking"
  - "HSFA"
  - "Tracking verifier"
  - "Energy-like score"
  - "Zero-shot registration"
date: 2026-05-08
content_hash: 5bddabdc7a4ca944
---

# STORM: Segment, Track, and Object Re-Localization from a Single Image

**Conference**: ICML 2026  
**arXiv**: [2511.09771](https://arxiv.org/abs/2511.09771)  
**Code**: https://github.com/YuDeng321/STORM  
**Area**: Video Understanding / 6D Pose Tracking / Reference Segmentation / Embodied Perception  
**Keywords**: Reference-conditioned 6D tracking, HSFA, Tracking verifier, Energy-like score, Zero-shot registration

## TL;DR
STORM proposes a "single reference image" 6D pose tracking framework: hierarchical spatial fusion attention (HSFA) aligns reference-query features (producing segmentation masks + SAM3D mesh), then a BCE-trained Tracking Verifier outputs a logit whose negative is used as an energy score $E=-g_\theta$. If the score exceeds a threshold for $L=3$ consecutive frames, automatic re-localization is triggered. This pushes annotation-free 6D tracking accuracy on LM-O / YCB-V close to the ground-truth mask upper bound.

## Background & Motivation

**Background**: Current SOTA 6D pose estimation and tracking methods (FoundationPose, SAM-6D, Pos3R, etc.) mostly rely on CAD models, manual masks, or per-object fine-tuning, requiring cumbersome object-specific preparation for deployment. General foundation models (SAM3, DINOv3) provide strong semantics but lack reference-conditioned mechanisms, making it impossible to specify which instance to track with "just one image".

**Limitations of Prior Work**: (1) Reference-query template matching often uses shallow cosine similarity, which collapses under occlusion, motion blur, or drastic viewpoint changes due to nonlinear manifold distortions; (2) Existing trackers "blindly follow"—once the target drifts out of the local neighborhood, there is no intrinsic signal to determine "I have lost the target", leading to silent drift; (3) Even with recovery heuristics (particle filtering, histogram matching), false positives are common, preventing a closed loop.

**Key Challenge**: There is a dual gap of **distribution shift** and **occlusion uncertainty** between reference and query images. Pure geometric matching cannot solve the former, and pure semantic matching is insufficient for the latter. Tracking is a **self-feedback system**; without a "self-assessment signal", closed-loop recovery is impossible.

**Goal**: (i) Achieve single-reference-image 6D tracking without CAD or per-object training; (ii) Make "tracking failure detection" a learnable module; (iii) Enable automatic recovery under severe occlusion and rapid viewpoint changes.

**Key Insight**: Reconstruct segmentation and tracking from "independent engineering modules" into "coupled learning modules"—the former compresses the reference view into an object-centric representation via hierarchical attention, while the latter formalizes "whether tracking is still compatible with initial memory" as a binary verification problem, borrowing energy scoring from OOD detection (Liu 2020) for smooth thresholding.

**Core Idea**: Use a BCE-trained compatibility verifier to simultaneously serve as "instance matching loss supervision" and "tracking validity energy scoring", unifying invariance, robustness, and closed-loop recovery in a single logit scalar.

## Method

### Overall Architecture
STORM consists of two coupled modules. **SOM (Segmenting Object Module)**: Takes one or more reference images $I_{ref}$ + current query image $I_q$ (optionally with VLM semantic prompts), outputs the target mask on the query via HSFA, then uses SAM3D to generate a canonical 3D mesh $\mathcal{P}_{ref}$ from the reference, which, together with the mask, is fed into the frozen FoundationPose to obtain the 6D pose. **TOM (Tracking Object Module)**: Maintains a FIFO memory pool $\mathcal{M}$ of size $K=16$ for successful tracking crops. For each frame, DINOv3 features $\phi(x_t)$ are paired with $\mathcal{M}$ to compute logit $g_\theta(x_t,\mathcal{M})$, defining energy $E(x_t,\mathcal{M})\triangleq -g_\theta(x_t,\mathcal{M})$. After EMA smoothing, if $L=3$ consecutive frames have $\tilde E_{t-k}>\tau$, re-localization is triggered ($\tau$ is set at the 95th percentile on the validation set). Frozen components: DINOv3, CLIP/VLM, SAM3D, FoundationPose; trainable components: SOM (HSFA + segmentation head) + TOM (lightweight attention verifier).

### Key Designs

1. **HSFA Hierarchical Spatial Fusion Attention**:

    - **Function**: Aligns any number (single or multiple) of reference patches with query pixel features at multiple scales, supports optional VLM semantic conditioning, and outputs an implicit token-to-token alignment matrix corresponding to the mask.
    - **Mechanism**: (a) Self-attention aggregates the reference view into an object-centric latent representation $\mathcal{Z}_{ref}$; (b) Query features $\mathcal{Z}_{query}$ retrieve $\mathcal{Z}_{ref}$ via cross-attention—shallow layers perform global semantic anchoring to original reference features, deep layers perform local geometric alignment to refined spatial features; (c) The fusion block iterates $n$ times for progressive refinement; (d) When VLM provides a text description $T$, a zero-initialized AdaLN/FiLM injects its CLIP embedding $e_t$ as a condition, modifying visual token statistics: $\hat F_{i,c}=(1+s_c(e_t))(F_{i,c}-\mu_i)/(\sigma_i+\epsilon)+b_c(e_t)$, and uses sigmoid gating in cross-attention to suppress irrelevant reference channels; finally, the cross-attention softmax weights serve as the alignment matrix $W$ to project reference objectness onto the query to obtain the mask.
    - **Design Motivation**: Traditional cosine template matching is sensitive to nonlinear perturbations, and fixed reference concatenation cannot handle "variable number of reference views at inference". HSFA makes alignment **learned + hierarchical + conditional**, with only mask loss supervision (no explicit correspondence loss), avoiding fragile keypoint alignment.

2. **Energy-like Tracking Verifier (TOM)**:

    - **Function**: For each frame, outputs a 0–1 probability and a scalar energy score to determine "whether the current observation still belongs to the initial tracked object", and makes closed-loop re-localization decisions accordingly.
    - **Mechanism**: During training, $(x_t,\mathcal{M},y)$ triplets are used for BCE: $\mathcal{L}_{TOM}=-\mathbb{E}[y\log\sigma(g_\theta)+(1-y)\log(1-\sigma(g_\theta))]$, with positive samples from true compatible observation-memory pairs, and negatives synthesized via identity confusion (different objects in the same scene) + drift-like random cropping; at inference, energy is defined as $E=-g_\theta$, EMA is applied to get $\tilde E_t$, and only if $L=3$ consecutive frames have $\tilde E_{t-k}>\tau$ is tracking loss declared, avoiding single-frame jitter false positives; $\tau$ is set at the 95th percentile of the compatible-pair distribution on a held-out set.
    - **Design Motivation**: Existing trackers assume "the target is always in the local neighborhood" and lack a failure signal. Treating verification as an OOD-style continuous energy threshold problem leverages BCE's stable training and allows flexible energy smoothing/temperature control at inference, with mathematical equivalence between energy and logit thresholds ($E>\tau\Leftrightarrow g_\theta<-\tau$), facilitating engineering tuning.

3. **SAM3D Geometric Anchor + Train/Frozen Boundary**:

    - **Function**: Lifts 2D masks to metric 3D coordinates, enabling the frozen FoundationPose to perform precise registration; also clarifies which parameters are trained and which are frozen.
    - **Mechanism**: SAM3D generates a canonical mesh $\mathcal{P}_{ref}$ from the reference image as a "rigid geometric reference"—instead of hard texture/geometry matching, the mesh serves as a soft latent geometric constraint; at runtime, SAM3D / DINOv3 / FoundationPose / CLIP are all frozen, only SOM (HSFA + segmentation head) and TOM (lightweight attention verifier) are trained, greatly reducing training cost while retaining foundation model priors.
    - **Design Motivation**: Single-view mesh prediction (e.g., Direct3D-S2) is unstable, but as long as it serves as a "structural scaffold" rather than precise geometry, downstream pose registration can tolerate noise; freezing foundation models ensures zero-shot generalization is not contaminated by limited training data.

### Loss & Training
SOM uses standard segmentation loss (supervising the mask, with correspondence emerging implicitly, no explicit correspondence loss); TOM uses BCE (see formula 3); inference: DINOv3 feature → TOM logit → EMA → thresholding → closed loop. Memory pool FIFO size is 16, cleared after re-localization, and only high-confidence frames are added.

## Key Experimental Results

### Main Results
Annotation-free 6D tracking accuracy ($\mathrm{ADD}_\mathrm{AUC}$ / $\mathrm{ADD\text{-}S}_\mathrm{AUC}$ / AR) on LM-O / YCB-V:

| Dataset | Method | $\mathrm{ADD}_\mathrm{AUC}$ | $\mathrm{ADD\text{-}S}_\mathrm{AUC}$ | AR |
|---|---|---|---|---|
| LM-O | FP + CNOS | 57.0 | 68.0 | 41.0 |
| LM-O | **STORM** | **74.0 ± 1.28** | **89.0 ± 1.25** | **53.0 ± 2.02** |
| LM-O | FP + Ground Truth | 78.0 | 93.0 | 56.0 |
| YCB-V | FP + CNOS | 73.0 | 92.0 | 69.0 |
| YCB-V | **STORM** | **77.0 ± 1.25** | **98.0 ± 1.20** | **73.0 ± 1.23** |
| YCB-V | FP + Ground Truth | 78.0 | 99.0 | 74.0 |

BOP instance segmentation (mean AP over 5 datasets, annotation-free section):

| Method | LM-O | T-LESS | TUD-L | HB | YCB-V | Mean ↑ | Time (s) |
|---|---|---|---|---|---|---|---|
| **STORM (SOM)** | **57.8** | **53.0** | **73.3** | **74.1** | **80.3** | **67.7** | **0.046** |
| NOCTIS | 48.9 | 47.9 | 58.3 | 60.7 | 68.4 | 56.8 | 0.990 |
| SAM6D | 46.0 | 45.1 | 56.9 | 59.3 | 60.5 | 53.6 | 2.795 |
| CNOS (FastSAM) | 39.7 | 37.4 | 48.0 | 51.1 | 59.9 | 47.2 | 0.221 |

### Ablation Study

| Configuration | Key Change | Conclusion |
|---|---|---|
| Full STORM | mean AP 67.7 | Complete framework |
| w/o HSFA deep iteration | Significant degradation | Multi-scale cross-attention is key to segmentation robustness |
| w/o VLM semantic injection | Increased multi-instance confusion | Text conditioning mainly helps in ambiguous scenarios |
| TOM with fixed cosine metric | Tracking-loss detection AUC ↓ | Learned logit distinguishes true drift better than fixed metric |
| Disable EMA smoothing + consecutive $L$ check | False trigger rate rises significantly | 3-frame gating clearly suppresses false positives |

### Key Findings
- STORM raises annotation-free pipeline on LM-O from 57.0 to 74.0, only 4 points from the ground-truth mask upper bound (78.0)—indicating mask quality is the current bottleneck, and TOM nearly exhausts the pose head's capacity.
- SOM runs inference in just 0.046s per sample on H100, 20–60× faster than NOCTIS / SAM6D, thanks to frozen DINOv3 + lightweight HSFA design.
- The TOM verifier is more stable than fixed-metric baselines on the Tracking Failure Benchmark, and consecutive frame gating makes re-localization decisions immune to single-frame noise.

## Highlights & Insights
- **Both "how to segment" and "how to verify" are implemented as learned alignment**, avoiding the industry-standard but fragile cosine template engineering.
- **Energy score = negative logit** mathematical equivalence combines BCE training stability with flexible energy thresholding at inference, and can be directly transferred to any "learnable binary matching + temporal closed loop" task (e.g., ReID, semi-supervised object tracking).
- **Freezing foundation models + training two small modules** minimizes training footprint, allowing STORM to leverage DINOv3 / FoundationPose zero-shot generalization while enabling low-cost fine-tuning for new tasks—high engineering friendliness.
- **VLM uses zero-initialized AdaLN for conditional injection**: treats semantics as "identity-preserving feature statistic correction" rather than hard concatenation, avoiding early-stage text channel interference with visual learning—an elegant transfer of Cond-DM ideas to visual alignment.

## Limitations & Future Work
- The authors acknowledge that zero-shot here only means "no test-time mask/box/fine-tuning"; BOP train/test object identities may overlap, so this is not true category-disjoint generalization to new objects.
- SAM3D single-image reconstruction quality determines the pose upper bound; performance may degrade on reflective, transparent, or textureless objects. Future work could explore multi-view adaptive mesh refinement.
- TOM's $\tau$ 95th percentile calibration is based on synthetic drift negatives and may not be robust to real-world long-tail occlusion distributions; adding online adaptive thresholds or Bayesian uncertainty estimation is a natural extension.
- A single reference image only covers one viewpoint; severe occlusion still requires manual multi-view references. When to actively request new reference images via active learning remains an open question.

## Related Work & Insights
- **vs FoundationPose (Wen 2024)**: This work directly reuses its pose head, but fills the gaps of "self-assessment of tracking validity" and "how to obtain masks without CAD".
- **vs CNOS / PerSAM**: They use shallow cosine template matching, while STORM uses hierarchical attention for learned alignment, showing clear robustness under occlusion.
- **vs SAM-6D / Pos3R**: They perform frame-level processing + explicit 2D-3D keypoint matching, while STORM introduces temporal closed loop via the verifier.
- **vs Energy score in OOD detection (Liu 2020)**: This is the first systematic transfer of energy thresholding from OOD classification to 6D tracking failure detection.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of HSFA + Energy-like verifier is a new attempt in 6D tracking, with prior art for each module individually
- Experimental Thoroughness: ⭐⭐⭐⭐ LM-O / YCB-V + 5 BOP datasets + 5 RQs + 5 seed error bars, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Module boundaries and frozen/trainable boundaries are clearly described, energy score derivation is concise
- Value: ⭐⭐⭐⭐ Highly practical for robotics / embodied perception, open-source code + accuracy close to GT upper bound

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](../../CVPR2026/video_understanding/storm_referring_multi_object_tracking.md)
- [\[CVPR 2026\] Temporally Consistent Long-Term Memory for 3D Single Object Tracking](../../CVPR2026/video_understanding/chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](../../CVPR2026/video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](../../CVPR2026/video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](../../CVPR2026/video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)

</div>

<!-- RELATED:END -->
