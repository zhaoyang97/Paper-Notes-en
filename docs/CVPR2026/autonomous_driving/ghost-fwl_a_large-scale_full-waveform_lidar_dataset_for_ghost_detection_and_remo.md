---
title: >-
  [Paper Note] Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal
description: >-
  [CVPR 2026][Autonomous Driving][Full-Waveform LiDAR] Ghost-FWL introduces the first large-scale mobile full-waveform LiDAR dataset (24K frames, 7.5 billion peak-level annotations) and proposes FWL-MAE, a self-supervised pretraining framework for ghost detection and removal, reducing SLAM trajectory error by over 66% and cutting 3D detection false positive rates by 50×.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Full-Waveform LiDAR
  - Ghost Detection
  - Dataset
  - Self-Supervised Learning
  - Masked Autoencoder
date: 2026-05-08
content_hash: 8dda7fa2aaa6a9b2
---

# Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal

**Conference**: CVPR 2026
**arXiv**: [2603.28224](https://arxiv.org/abs/2603.28224)
**Code**: [https://keio-csg.github.io/Ghost-FWL/](https://keio-csg.github.io/Ghost-FWL/)
**Area**: Autonomous Driving / 3D Vision
**Keywords**: Full-Waveform LiDAR, Ghost Detection, Dataset, Self-Supervised Learning, Masked Autoencoder

## TL;DR
Ghost-FWL introduces the first large-scale mobile full-waveform LiDAR dataset (24K frames, 7.5 billion peak-level annotations) and proposes FWL-MAE, a self-supervised pretraining framework for ghost detection and removal, reducing SLAM trajectory error by over 66% and cutting 3D detection false positive rates by 50×.

## Background & Motivation

**Background**: LiDAR is a core sensor for autonomous driving, robotics, and large-scale terrain mapping. It reconstructs 3D geometry by measuring the time-of-flight of laser pulses. Conventional LiDAR outputs only processed point clouds (range + intensity), discarding the rich physical information embedded in raw waveforms.

**Limitations of Prior Work**: LiDAR systems widely suffer from "ghost points" — laser pulses undergo multipath reflections off glass and specular surfaces, generating spurious 3D points that do not correspond to real objects. As sensor sensitivity increases, ghost artifacts become more severe. These phantom points cause: (1) false positives in 3D object detection (e.g., detecting "ghost pedestrians" behind glass); (2) localization drift and map errors in SLAM.

**Key Challenge**: Existing ghost removal methods rely on geometric consistency of point clouds (e.g., spatial symmetry), which requires dense, static scanning environments. In mobile LiDAR scenarios (autonomous driving, robotics), point clouds are sparse and dynamic, leaving geometric cues insufficient to distinguish ghosts from real reflections. Full-waveform LiDAR (FWL) records the complete temporal intensity profile of each pulse, containing temporal and amplitude cues that can distinguish ghosts — yet no FWL ghost detection dataset exists.

**Goal**: (1) Construct the first FWL ghost detection dataset targeting mobile scenarios; (2) propose a baseline framework for ghost detection and removal from FWL data; (3) design a self-supervised pretraining method suited to FWL data to address the high cost of annotation.

**Key Insight**: FWL data naturally encodes multipath reflection information — ghost reflections exhibit distinguishable patterns in peak temporal position, amplitude, and width compared to true object reflections. The proposed approach learns these physical signatures to detect ghosts.

**Core Idea**: Leverage the temporal-intensity information of full-waveform LiDAR — rather than point cloud geometry alone — to detect and remove ghost reflections.

## Method

### Overall Architecture
The system consists of three main components: (1) **Ghost-FWL Dataset** — collected using a custom FWL acquisition system across 10 indoor and outdoor scenes (24K frames), with a semi-automatic annotation pipeline that assigns a physical source label (Object / Glass / Ghost / Noise) to each reflection peak; (2) **FWL-MAE Self-Supervised Pretraining** — pretrains a Transformer encoder on unannotated mobile trajectory data to learn physically meaningful representations of FWL signals; (3) **Ghost Detection and Removal** — applies a lightweight classification head on top of the pretrained encoder to predict the category of each FWL data point, then removes 3D points predicted as Ghost.

### Key Designs

1. **Ghost-FWL Dataset Construction**:

    - **Function**: Provides the first large-scale, mobile-scenario, peak-level annotated FWL ghost detection dataset.
    - **Mechanism**: A custom FWL acquisition system directly accesses the FPGA module of the LiDAR hardware to extract raw waveform data. The sensor produces $512 \times 400$ pixel histograms, recording up to 700 time bins per direction (~1 ns resolution, maximum range 105 m). Data collection covers 4 indoor scenes (offices, lounges, gymnasium) and 6 outdoor scenes (building entrances, glass facades, pedestrian areas) across different time periods (morning / afternoon / evening). Two acquisition strategies are employed: multi-viewpoint static capture (37–55 viewpoints per scene, totaling 24,412 annotated frames) and mobile trajectory capture (8,933 unannotated frames for self-supervised pretraining).
    - **Design Motivation**: Existing datasets are either static high-precision scans (unsuitable for mobile scenarios), lack peak-level annotations (PixSet), or are not publicly available (Scheuble et al.). Synthetic multipath simulation is computationally expensive and physically inaccurate. Ghost-FWL is 100× larger than the previously largest annotated FWL dataset.

2. **Semi-Automatic Annotation Pipeline**:

    - **Function**: Efficiently provides peak-level physical source annotations for large-scale FWL data.
    - **Mechanism**: (a) A commercial 360° LiDAR (Livox Mid-360) combined with fastlio2 SLAM constructs a high-precision 3D ground-truth map ($\mathcal{M}$) for each scene; (b) glass regions $\mathcal{G}$ and reflection regions $\mathcal{R}$ are manually labeled in the map; (c) multi-frame FWL data are accumulated to obtain high-SNR waveforms, from which peaks are extracted and converted to point clouds; (d) the FWL point cloud is registered to the GT map; (e) each point is automatically classified as Object (close to GT map surface), Glass (within glass regions), Ghost (passing through / reflected from glass without correspondence in the GT map), or Noise, based on the nearest-neighbor distance $d(\mathbf{x}) = \min_{\mathbf{y} \in \mathcal{M}} \|\mathbf{x} - \mathbf{y}\|$ and region definitions. Results are reviewed by domain experts.
    - **Design Motivation**: Ghost reflections are virtual and have no direct ground truth. The pipeline indirectly identifies ghosts by comparing spatial deviations between the high-precision GT map and FWL data.

3. **FWL-MAE (Full-Waveform LiDAR Masked Autoencoder)**:

    - **Function**: Learns physically meaningful FWL representations under limited annotated data.
    - **Mechanism**: Given an FWL data volume $\mathbf{V} \in \mathbb{R}^{H \times W \times T}$, spatial $(x, y)$ patches are randomly sampled and fully masked along the time axis $T$. A 6-layer, 6-head Transformer encoder produces latent representations. Unlike MARMOT, FWL-MAE additionally employs a linear head to estimate each histogram peak's position $p$, amplitude $a$, and width $w$. The loss function is:

      $$\mathcal{L}_{\text{FWL-MAE}} = \mathcal{L}_{\text{MSE}} + \lambda_p \mathcal{L}_1^{\text{peak-}p} + \lambda_a \mathcal{L}_1^{\text{peak-}a} + \lambda_w \mathcal{L}_1^{\text{peak-}w}$$

      where voxel reconstruction uses MSE loss and peak attribute prediction uses L1 loss.
    - **Design Motivation**: Annotation of FWL data at the peak level is extremely costly. Self-supervised pretraining exploits large quantities of unannotated mobile trajectory data to learn general representations. Existing methods such as MARMOT perform only voxel-level reconstruction without modeling physical peak attributes, whereas peak position, amplitude, and width are precisely the cues that distinguish ghost reflections.

### Loss & Training
Ghost detection employs Focal Loss to handle severe class imbalance (Noise constitutes the vast majority of samples). The encoder pretrained by FWL-MAE is frozen; only the classification head (2 linear layers) is trained. Raw FWL data is preprocessed to $(128, 128, 256)$, with leading and trailing bins corresponding to ceiling/floor reflections and internal sensor noise removed. The dataset is split into 13,853 training / 2,994 validation / 1,427 test frames.

## Key Experimental Results

### Main Results — Ghost Detection

| Method | Recall↑ | Ghost Removal Rate↑ |
|--------|---------|---------------------|
| MARMOT | 0.746 | 0.910 |
| Ours w/o FWL-MAE | 0.704 | 0.900 |
| **Ours (with FWL-MAE)** | **0.751** | **0.918** |

### Downstream Task — SLAM

| Method | ATE (m)↓ | RTE (m)↓ |
|--------|----------|----------|
| Dual-Peak | 0.715±0.433 | 0.741±0.406 |
| Multi-Peak | 1.547±1.394 | 1.602±1.381 |
| **Ours** | **0.245±0.138** | **0.245±0.131** |

Ghost removal reduces ATE by 66–84% and RTE by 67–85%.

### Downstream Task — 3D Object Detection

| Method | Ghost FP Rate↓ |
|--------|----------------|
| Dual-Peak | 75.8% |
| Multi-Peak | 67.9% |
| **Ours** | **1.34%** |

The ghost-induced false positive rate drops from 67.9% to 1.34%, a reduction of approximately 50×.

### Key Findings
- FWL-MAE pretraining provides a clear benefit for ghost detection (Recall improves from 0.704 to 0.751), validating the effectiveness of self-supervised pretraining for learning FWL physical features.
- FWL-MAE outperforms generic MARMOT pretraining, demonstrating that explicitly modeling peak attributes — rather than performing voxel reconstruction alone — is critical.
- The impact of ghost removal on downstream tasks is dramatic: in SLAM, the Multi-Peak baseline suffers severe trajectory drift due to ghosts (ATE 1.547 m), which drops to 0.245 m after removal; in 3D detection, the false positive rate falls from 67.9% to 1.34%.
- Improvements are most pronounced near glass surfaces, precisely where ghost artifacts are most severe.

## Highlights & Insights
- The problem formulation is highly precise: the paper identifies full-waveform data as an overlooked source of rich information. Conventional practice crops waveforms into peak point clouds, inadvertently discarding the key evidence needed to distinguish ghosts. Retaining complete waveforms and learning physical signatures represents a conceptually clean and well-motivated advance.
- The dataset scale and annotation quality are impressive: 24K frames with 7.5 billion peak-level annotations. The strategy of using high-precision 3D maps as indirect ground truth is particularly elegant.
- The addition of peak attribute prediction heads (position / amplitude / width) in FWL-MAE is simple yet targeted, capturing the essential difference between FWL data and ordinary images or generic histograms.
- The dramatic downstream improvements (50× false positive reduction) directly demonstrate the practical value of the research.

## Limitations & Future Work
- Current annotations cover only static multi-viewpoint captures; continuous mobile sequences are left unannotated due to labeling cost. Extending annotation to mobile sequences would enable temporal models.
- The dataset addresses only glass-induced ghosts; other reflective materials (water surfaces, polished metal) and adverse weather conditions (rain, fog) are not covered.
- The FWL-MAE Transformer encoder has only 6 layers, which may be insufficient for complex multipath reflection patterns.
- Ghost detection Recall is 0.751, leaving approximately 25% of ghosts undetected — potentially inadequate for safety-critical autonomous driving applications.
- The current method performs per-frame detection independently, without exploiting temporal information. Cross-frame consistency could provide additional cues for ghost detection.

## Related Work & Insights
- **vs. UNIST / Lee et al. (static methods)**: These approaches rely on geometric consistency from static high-precision scanners and are unsuitable for mobile scenarios. Ghost-FWL leverages FWL temporal-intensity information, enabling detection from sparse single frames.
- **vs. Scheuble et al. (FWL methods)**: Their end-to-end FWL approach targets ranging accuracy rather than ghost detection, and their dataset (only 240 frames, not publicly released) is 100× smaller than Ghost-FWL.
- **vs. PixSet**: The only publicly available FWL dataset, but it lacks peak-level annotations and cannot support ghost detection training.
- **vs. MARMOT**: A general transient image MAE that performs voxel reconstruction only. FWL-MAE's peak attribute prediction heads are better suited to LiDAR data.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to formulate FWL ghost detection as a task; the dataset fills a critical gap in the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Complete evaluation chain from ghost detection to SLAM and 3D detection is highly convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear; dataset construction is described in thorough detail.
- **Value**: ⭐⭐⭐⭐⭐ — Dataset and code are publicly released; direct practical implications for autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](../../ICLR2026/autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](neural_distribution_prior_for_lidar_ood_detection.md)

</div>

<!-- RELATED:END -->
