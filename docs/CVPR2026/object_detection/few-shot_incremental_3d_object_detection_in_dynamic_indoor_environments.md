---
title: >-
  [Paper Note] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments
description: >-
  [CVPR 2026][Object Detection][Few-shot incremental learning] This paper proposes FI3Det, the first few-shot incremental 3D object detection framework. During the base training stage, a VLM-guided unknown object learning module enables early awareness of potential novel categories. During the incremental stage, a gated multimodal prototype imprinting module fuses 2D semantic and 3D geometric features for novel class detection. FI3Det achieves an average improvement of 17.37% in novel class mAP on ScanNet V2 and SUN RGB-D.
tags:
  - CVPR 2026
  - Object Detection
  - Few-shot incremental learning
  - 3D object detection
  - vision-language models
  - multimodal prototypes
  - indoor scene understanding
date: 2026-05-08
content_hash: 95b8dd4385cfbf02
---

# Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments

**Conference**: CVPR 2026
**arXiv**: [2604.07997](https://arxiv.org/abs/2604.07997)
**Code**: [https://github.com/zyrant/FI3Det](https://github.com/zyrant/FI3Det)
**Area**: 3D Vision
**Keywords**: Few-shot incremental learning, 3D object detection, vision-language models, multimodal prototypes, indoor scene understanding

## TL;DR

This paper proposes FI3Det, the first few-shot incremental 3D object detection framework. During the base training stage, a VLM-guided unknown object learning module enables early awareness of potential novel categories. During the incremental stage, a gated multimodal prototype imprinting module fuses 2D semantic and 3D geometric features for novel class detection. FI3Det achieves an average improvement of 17.37% in novel class mAP on ScanNet V2 and SUN RGB-D.

## Background & Motivation

1. **Background**: 3D object detection methods (e.g., VoteNet, TR3D, FCAF3D) have achieved strong performance on fixed category sets, but operate under a static paradigm—assuming all class annotations are available during a single training session. Incremental 3D detection methods (SDCoT, AIC3DOD) can progressively recognize new classes but still require abundant annotations for novel categories.
2. **Limitations of Prior Work**: (a) Existing incremental 3D detection methods rely on rich novel-class annotations, which is unrealistic in dynamic indoor embodied environments where new objects appear without immediate large-scale labeling; (b) the 2D domain already has few-shot incremental detection methods (ONCE, Sylph, IL-DETR), yet the 3D domain remains entirely unexplored; (c) data-efficient 3D detection approaches (GFS-VL, MixSup) focus primarily on pseudo-label generation while neglecting feature-level learning.
3. **Key Challenge**: Under extremely limited novel-class samples, how can a model learn new categories without forgetting previously learned ones? Complex layouts and diverse object configurations in indoor 3D scenes increase inter-class variation, exacerbating this tension.
4. **Goal**: (a) Define and address the new task of few-shot incremental 3D object detection; (b) establish early awareness of novel categories during the base stage; (c) efficiently adapt to new categories during the incremental stage while preserving performance on old classes.
5. **Key Insight**: The authors observe that novel-class objects often already appear in training scenes without annotation (as illustrated in Fig. 2, unlabeled novel objects frequently co-occur alongside base-class objects). Leveraging the zero-shot recognition capability of VLMs enables mining of these unknown objects during base training to establish early knowledge of novel categories.
6. **Core Idea**: During the base stage, VLMs are used to mine unlabeled unknown objects for feature- and box-level learning; during the incremental stage, multimodal prototypes fusing 2D semantics and 3D geometry enable few-shot novel class detection.

## Method

### Overall Architecture

FI3Det consists of two stages. In the **base training stage**, a VLM-guided unknown object learning module is added on top of the TR3D detector, comprising unknown object mining (generating pseudo 3D boxes and 2D semantic features) and unknown object weighting (suppressing noise). In the **incremental learning stage**, the detector parameters are frozen, and a gated multimodal prototype imprinting module constructs 2D semantic and 3D geometric prototypes with adaptive gating for novel class detection. The inputs are 3D point cloud scenes with corresponding RGB images; the output is 3D bounding box predictions over base and novel classes.

### Key Designs

1. **VLM-guided Unknown Object Learning**

   - **Function**: During the base training stage, VLMs are used to mine unlabeled unknown objects in the scene, providing auxiliary supervision signals that endow the detector with early awareness of novel categories.
   - **Mechanism**: Proceeds in two steps—(a) **Unknown Object Mining**: GroundingDINO generates 2D bounding boxes; a class-agnostic segmentation model extracts 2D masks $\mathbf{M}^{2D}$, which are projected into 3D space to obtain $\mathbf{M}^{3D}$. For each instance, the average VLM feature $\mathbf{f}_j^{2D}$ and fitted 3D box $\mathbf{b}_j^{3D}$ are computed. An objectness head (foreground awareness) and a feature head (2D–3D feature alignment) are additionally introduced. (b) **Unknown Object Weighting**: **Point-level weighting** applies a Gaussian function $w_{e,j}^{point} = \exp(-\|\mathbf{p}_e - \mathbf{c}_j\|_2^2 / 2\sigma^2)$ to assign higher weights to points near the box center; **box-level weighting** $w_j^{box} = \|\frac{1}{|\mathcal{B}_j|}\sum \text{norm}(\hat{\mathbf{f}}_e^{2D})\|_2$ measures feature consistency within the box, with more consistent boxes deemed more reliable.
   - **Design Motivation**: Pseudo-labels generated by VLMs are noisy and would introduce erroneous supervision if used directly. The Gaussian point-level weighting is motivated by the intuition that segmentation errors are larger near boundaries; the box-level weighting is grounded in the prior that features within the same object should be semantically consistent. The two-level weighting scheme jointly suppresses noise effectively.

2. **Gated Multimodal Prototype Imprinting**

   - **Function**: During the incremental stage, efficiently constructs classification prototypes from a small number of novel-class samples without retraining the detector, thereby avoiding catastrophic forgetting.
   - **Mechanism**: Modality-specific prototypes $\mathbf{T}^{2D}$ and $\mathbf{T}^{3D}$ are built from aligned 2D features $\hat{\mathbf{F}}^{2D}$ and 3D geometric features $\mathbf{F}^{3D}$, respectively. A momentum update strategy $\mathbf{T}_c^{3D} \leftarrow \mu \mathbf{T}_c^{3D} + (1-\mu)\bar{\mathbf{F}}_c^{3D}$ ($\mu=0.999$) stabilizes prototype estimation. Cosine similarity classification scores $\mathbf{S}^{3D}$ and $\mathbf{S}^{2D}$ are then computed per modality. **Multimodal gated fusion** employs two sets of learnable gating functions: $[\alpha^{3D}, \alpha^{2D}] = \text{Softmax}(\text{MLP}([\mathbf{F}^{3D}; \hat{\mathbf{F}}^{2D}]))$ controls modality weights, while $\gamma = \sigma(\text{MLP}([\mathbf{F}^{3D}; \hat{\mathbf{F}}^{2D}]))$ rebalances class-level contributions. The final fused score is $\mathbf{S}^{fuse} = \gamma \odot (\alpha^{3D} \odot \mathbf{S}^{3D} + \alpha^{2D} \odot \mathbf{S}^{2D})$.
   - **Design Motivation**: Single-modality prototypes fail to exploit the complementary advantages of 2D semantics and 3D geometry. Simple summation ignores the distinct characteristics of each modality. Adaptive gating enables the model to dynamically adjust modality contributions based on the specific scene and object features, while $\gamma$ additionally prevents overconfident predictions for certain categories.

3. **Auxiliary Loss Design**

   - **Function**: Provides three forms of supervision for unknown object learning.
   - **Mechanism**: (a) Foreground supervision $\mathcal{L}_{obj}$: BCE + Dice loss trains the objectness head using weighted continuous foreground scores rather than hard labels; (b) Feature supervision $\mathcal{L}_{feat}$: cosine similarity loss aligns 3D features with VLM 2D features; (c) Regression supervision $\mathcal{L}_{reg}^{unk}$: weighted DIOU loss learns geometric localization of unknown objects. All three losses apply joint point-level and box-level weights.
   - **Design Motivation**: Class-agnostic foreground detection capability, semantic feature alignment, and spatial localization ability are each indispensable; together they ensure that prototypes in the incremental stage can correctly match novel class proposals.

### Loss & Training

Base training: $\mathcal{L} = \mathcal{L}_{det} + \mathcal{L}_{aux}$, where $\mathcal{L}_{aux} = \mathcal{L}_{aux-obj} + \mathcal{L}_{aux-feat} + \mathcal{L}_{aux-box}$. Incremental stage: detector parameters are frozen; only prototypes and gating functions are updated using $\mathcal{L}_{inc}$ on novel classes.

## Key Experimental Results

### Main Results

ScanNet V2, batch incremental setting (1-way 5-shot):

| Method | Base mAP | Novel mAP | All mAP |
|--------|----------|-----------|---------|
| Imprinting | 71.47 | 0.23 | 67.72 |
| IL-DETR | 65.63 | 0.35 | 62.00 |
| SDCOT++ | 62.12 | 0.09 | 58.68 |
| AIC3DOD | 70.54 | 4.59 | 66.88 |
| VLM-vanilla | 71.81 | 14.09 | 68.60 |
| **FI3Det** | **72.84** | **38.48** | **70.94** |

SUN RGB-D, batch incremental setting (1-way 5-shot):

| Method | Base mAP | Novel mAP | All mAP |
|--------|----------|-----------|---------|
| AIC3DOD | 58.83 | 0.02 | 52.95 |
| VLM-vanilla | 62.12 | 11.93 | 57.10 |
| **FI3Det** | **63.05** | **73.17** | **64.07** |

### Ablation Study

| Configuration | Base | Novel | All | Note |
|---------------|------|-------|-----|------|
| VLM-vanilla (baseline) | 71.81 | 14.09 | 68.60 | No proposed modules |
| + UOM | 72.73 | 25.43 | 70.10 | +Unknown object mining, Novel +11.34 |
| + UOM + UOW | 72.83 | 32.46 | 70.61 | +Weighting, Novel +7.03 |
| + UOM + GPI | 72.73 | 28.94 | 70.30 | +Gated prototype imprinting |
| + UOM + UOW + GPI (Full) | 72.84 | 38.48 | 70.94 | Full model, best Novel mAP |

Gating component ablation:

| Configuration | Novel mAP | Note |
|---------------|-----------|------|
| No gating | 32.46 | Direct summation |
| $\alpha^*$ only | 36.58 | +Modality weighting, +4.12 |
| $\gamma$ only | 34.68 | +Class rebalancing |
| $\alpha^*$ + $\gamma$ | 38.48 | Optimal combination |

### Key Findings

- **UOM contributes the most**: Unknown object mining improves Novel mAP from 14.09% to 25.43% (+80%), confirming that establishing early novel-class awareness during the base stage is critical.
- Base-class performance remains stable across all variants (~72.8%), demonstrating that the prototype imprinting strategy effectively prevents catastrophic forgetting.
- On SUN RGB-D 1-way 5-shot, FI3Det's Novel mAP (73.17%) even surpasses its Base mAP (63.05%), demonstrating exceptional novel-class adaptation capability.
- Hyperparameters $\sigma=0.5$ and $\mu=0.999$ constitute the optimal configuration; the monotonic improvement with larger $\mu$ indicates that momentum stabilization is critical for few-shot prototype estimation.

## Highlights & Insights

- **Unknown object learning during the base stage** is a particularly elegant idea: novel-class objects frequently appear in training scenes without annotation, and leveraging VLMs to mine these "dark matter" objects endows the detector with early novel-class awareness. This observation and its exploitation are transferable to any incremental learning or open-world detection task.
- **Two-level weighting (point-level + box-level)** offers a practical approach to handling noisy pseudo-labels: Gaussian spatial weighting and feature consistency weighting filter noise from spatial and semantic perspectives respectively, forming a reusable technique.
- **Multimodal gated fusion** is more flexible than simple weighting or concatenation; the $\gamma$ gate can suppress overconfident predictions from one modality on certain categories, improving overall robustness.

## Limitations & Future Work

- The detection capability of the current VLM (GroundingDINO) constrains the quality of unknown object mining; the performance ceiling may be further raised as more powerful VLMs emerge.
- Experiments are limited to indoor scenes (ScanNet V2, SUN RGB-D); large-scale outdoor settings such as autonomous driving have yet to be validated.
- Freezing the detector parameters during the incremental stage means that feature representations are not further optimized for novel classes, which may be limiting when novel and base class distributions differ substantially.
- The prototype imprinting approach maintains a single prototype per class; it remains an open question whether multiple prototypes—as in FedMEPD—could better capture intra-class variation.

## Related Work & Insights

- **vs. SDCoT++**: SDCoT++ pioneered incremental 3D detection but requires abundant novel-class annotations, leading to severe performance degradation in the few-shot setting (Novel mAP 0.09%). FI3Det avoids large-scale retraining through prototype imprinting.
- **vs. AIC3DOD**: AIC3DOD performs reasonably well in the full incremental setting but falls far short in the few-shot regime (Novel 4.59% vs. FI3Det's 38.48%), due to the absence of VLM-guided pretraining and multimodal fusion.
- **vs. VLM-vanilla**: Directly using VLM pseudo-boxes without weighting or multimodal fusion yields a Novel mAP of 14.09%; FI3Det's weighting and gated fusion raise this to 38.48%, underscoring the importance of noise handling and multimodal integration.

## Rating

- Novelty: ⭐⭐⭐⭐ First to define and address few-shot incremental 3D detection; the VLM-guided unknown object learning approach during the base stage is original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, both batch and sequential incremental settings, multiple ablations, and complete hyperparameter analysis
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is detailed, and figures are informative
- Value: ⭐⭐⭐⭐ Opens a new research direction for dynamic environment perception in embodied intelligence

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_fewshot_pill_recognition_under_visual_d.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)

</div>

<!-- RELATED:END -->
