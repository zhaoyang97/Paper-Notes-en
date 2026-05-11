---
title: >-
  [Paper Note] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data
description: >-
  [NeurIPS 2025][Image Generation][Single-image 3D human reconstruction] GeneMAN proposes a generalizable single-image 3D human reconstruction framework that requires **no parametric body model (e.g., SMPL)**. By training human-specific 2D/3D diffusion prior models on large-scale multi-source human data, and combining a geometry initialization-sculpting pipeline with multi-space texture refinement, GeneMAN achieves high-fidelity 3D human reconstruction from in-the-wild images, handling diverse body proportions, complex poses, and personal belongings.
tags:
  - NeurIPS 2025
  - Image Generation
  - Single-image 3D human reconstruction
  - diffusion model priors
  - template-free reconstruction
  - multi-source data
  - Score Distillation Sampling
  - texture refinement
date: 2026-05-08
content_hash: d8ea142a51a1d762
---

# GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data

**Conference**: NeurIPS 2025
**arXiv**: [2411.18624](https://arxiv.org/abs/2411.18624)
**Code**: [Project Page](https://roooooz.github.io/GeneMAN/)
**Area**: Image Generation
**Keywords**: Single-image 3D human reconstruction, diffusion model priors, template-free reconstruction, multi-source data, Score Distillation Sampling, texture refinement

## TL;DR

GeneMAN proposes a generalizable single-image 3D human reconstruction framework that requires **no parametric body model (e.g., SMPL)**. By training human-specific 2D/3D diffusion prior models on large-scale multi-source human data, and combining a geometry initialization-sculpting pipeline with multi-space texture refinement, GeneMAN achieves high-fidelity 3D human reconstruction from in-the-wild images, handling diverse body proportions, complex poses, and personal belongings.

## Background & Motivation

### Problem Definition

Reconstructing high-fidelity 3D human models from a single in-the-wild photograph is a core requirement for applications such as VR/AR, telepresence, digital humans, film, and gaming. However, due to severe 3D information loss, it remains a highly ill-posed problem.

### Limitations of Prior Work

**Template-based methods** (PaMIR, ICON, ECON, SiTH, SIFU, TeCH, etc.) rely on parametric body models such as SMPL/SMPL-X as geometric priors, but these models cannot represent 3D details of loose clothing and produce severe artifacts (e.g., bent legs) when pose/shape estimation is inaccurate.

**Template-free methods** (PIFu, PIFuHD, PHORHUM, etc.) avoid parametric constraints but still suffer from poor texture consistency and insufficient geometric detail due to the lack of adequate human-specific priors.

**General image-to-3D methods** (Zero-1-to-3, Magic123, etc.) perform well on general objects but lack human-specific priors, resulting in inaccurate body geometry and loss of facial and clothing details.

### Three Core Challenges

| Challenge | Description |
|-----------|-------------|
| **Variable body proportions** | In-the-wild photos include full-body, half-body, and head-shoulder crops; existing methods primarily target full-body reconstruction |
| **Personal belongings** | People in everyday photos frequently hold objects, stand on items, or wear accessories, significantly complicating reconstruction |
| **Natural pose and texture consistency** | The absence of general human-specific geometry/texture models leads to implausible geometry and cross-view texture inconsistency |

Furthermore, the scarcity of high-quality 3D human data exacerbates these challenges.

## Method

### Overall Architecture

The GeneMAN pipeline consists of four core components:

```
Multi-source data curation → Prior model training → Geometry initialization & sculpting → Multi-space texture refinement
```

#### 1. Multi-Source Human Dataset Construction

To improve generalizability, the authors collect over **50K multi-view instances** from four data source categories:

| Source | Specific Datasets |
|--------|------------------|
| 3D Scans | RenderPeople, CustomHumans, HuMMan, THuman2.0/3.0, X-Humans, Objaverse human subset |
| Multi-view Videos | DNA-Rendering, ZJU-Mocap, AIST++, Neural Actor, Actors-HQ |
| 2D Images | DeepFashion, LAION-5B |
| Synthetic Augmentation | ControlNet-generated multi-view humans with diverse clothing + image cropping augmentation (covering half-body/close-up proportions) |

#### 2. GeneMAN Prior Models

**2D Prior**: Stable Diffusion V1.5 is fine-tuned on all multi-source human data, with an equal amount of LAION-5B images mixed in to preserve general capability. Trained with AdamW (lr=1e-5) on 4×A100 for 5 days. Provides rich human geometry and texture detail.

**3D Prior**: Zero-1-to-3 is fine-tuned on 3D scans, multi-view videos, synthetic data, and DeepFashion images, with 20% Objaverse data added to preserve general object reconstruction capability. Trained with AdamW (lr=1e-4) on 8×A100 for 1 week. Ensures multi-view consistency.

### Key Designs

#### Geometry Initialization & Sculpting

**Stage 1: NeRF Initialization**

- Instant-NGP is used as the NeRF backbone, with resolution progressively increased from 256 to 384 over 5000 training steps
- Reference-view supervision loss $\mathcal{L}_{\text{ref}}$: RGB MSE + mask MSE
- Depth/normal priors: inferred by the human foundation model **Sapiens**, with depth loss (normalized negative Pearson correlation) and normal loss (MSE) applied respectively
- Novel-view guidance: hybrid SDS loss

$$\mathcal{L}_{\text{guid}} = \mathcal{L}_{\text{2D-SDS}}(\phi_{2d}, g(\theta)) + \mathcal{L}_{\text{3D-SDS}}(\phi_{3d}, g(\theta))$$

**Stage 2: DMTet Sculpting**

- The NeRF is converted to an explicit mesh and used to initialize DMTet (a hybrid SDF-mesh representation)
- Optimized for 3000 steps at resolution 512, with MSE + perceptual loss supervising normals; novel-view guidance is provided by HumanNorm pretrained normal/depth-adaptive diffusion models
- SDF regularization loss $\mathcal{L}_{\text{sdf}}$ prevents excessive geometric deviation

#### Multi-Space Texture Refinement

**Latent-Space Optimization (Coarse Texture)**:

- Hybrid SDS loss (2D + 3D priors) is used to optimize texture representation for 10000 steps
- Reference-view MSE loss ensures consistency with the input image
- **Training-free multi-view strategy**: the same Gaussian noise is added to renderings from different viewpoints, which are then concatenated into a single image for inference, achieving cross-view texture consistency without retraining the diffusion model

$$\mathcal{L}_{\text{coarse}} = \lambda_{\text{ref}}^c (\|{\hat{I}} - g_c(\theta;\hat{c})\|_2 + \|\hat{m} - g_c(\theta;\hat{c})\|_2) + \lambda_{\text{guid}}^c \mathcal{L}_{\text{guid}}^c$$

**Pixel-Space Optimization (Fine Texture)**:

- Uses the SDEdit framework: render coarse texture image → add noise → multi-step denoising with GeneMAN 2D prior + ControlNet to obtain refined image
- UV texture map optimized for 1000 steps with MSE + LPIPS loss:

$$\mathcal{L}_{\text{fine}} = \|I_{\text{fine}} - I_{\text{coarse}}\|_2 + \lambda_{LP} \cdot \text{LPIPS}(I_{\text{fine}}, I_{\text{coarse}})$$

### Loss & Training

Summary of loss functions involved across all training stages:

| Stage | Loss Function | Role |
|-------|--------------|------|
| NeRF Init | $\mathcal{L}_{\text{ref}}$ (RGB+mask) | Reference-view reconstruction |
| NeRF Init | $\mathcal{L}_{\text{depth}}$ (Pearson) | Depth consistency |
| NeRF Init | $\mathcal{L}_{\text{normal}}$ (MSE) | Normal consistency |
| NeRF/DMTet | $\mathcal{L}_{\text{guid}}$ (2D+3D SDS) | Novel-view human prior guidance |
| DMTet Sculpting | $\mathcal{L}_{\text{sdf}}$ | Geometry regularization |
| Coarse Texture | $\mathcal{L}_{\text{coarse}}$ (ref+SDS) | Consistent texture learning |
| Fine Texture | $\mathcal{L}_{\text{fine}}$ (MSE+LPIPS) | High-fidelity texture refinement |

## Key Experimental Results

### Main Results: Quantitative Comparison

Test set: 50 samples from in-the-wild internet images and the CAPE dataset; 120 viewpoints rendered per method over a 360° rotation.

| Method | in-the-wild PSNR↑ | LPIPS↓ | CLIP-Sim↑ | CAPE PSNR↑ | LPIPS↓ | CLIP-Sim↑ |
|--------|:-----------------:|:------:|:---------:|:----------:|:------:|:---------:|
| PIFu | 26.97 | 0.035 | 0.594 | 26.91 | 0.028 | 0.764 |
| GTA | 25.06 | 0.064 | 0.568 | 30.38 | 0.019 | 0.785 |
| TeCH | 25.74 | 0.053 | 0.713 | 27.60 | 0.025 | 0.826 |
| SiTH | 20.41 | 0.129 | 0.608 | 21.99 | 0.048 | 0.815 |
| **GeneMAN** | **32.24** | **0.013** | **0.730** | 28.49 | **0.015** | **0.838** |

- On in-the-wild images, GeneMAN's PSNR exceeds the second-best method by approximately **5.3 dB**, with LPIPS reduced by **63%**
- Highest CLIP-Sim score, indicating superior multi-view consistency
- On CAPE, GeneMAN achieves the best LPIPS and CLIP-Sim, with competitive PSNR

### User Study

40 participants, 30 test cases, 1200 pairwise comparisons. **73.08%** of participants preferred GeneMAN's reconstruction results (combined geometry and texture evaluation), far surpassing all baseline methods.

### Ablation Study

| Ablation | Key Finding |
|----------|-------------|
| Geometry initialization vs. sculpting | DMTet sculpting smooths overly noisy surfaces while recovering high-frequency details such as clothing wrinkles and facial features |
| Latent-space texture vs. pixel-space refinement | Latent-space texture is broadly reasonable but suffers from back-view inconsistency and slight blurriness; pixel-space optimization significantly improves detail |
| GeneMAN 2D prior vs. original DeepFloyd-IF | The original 2D prior causes front-back inconsistency in shirt hems; GeneMAN 2D prior ensures multi-view consistency |
| GeneMAN 3D prior vs. original Zero-1-to-3 | The original 3D prior produces unnatural poses (forward head lean); GeneMAN 3D prior captures more natural body posture |

### Key Findings

- The template-free design enables GeneMAN to effectively handle loose clothing (skirts, dresses) and personal belongings (e.g., basketballs), avoiding cascading errors from SMPL estimation failures
- Collecting large-scale multi-source data and fine-tuning diffusion models as human priors is the key source of generalization capability
- The training-free multi-view consistency strategy (shared noise + concatenated inference) effectively improves cross-view consistency without additional training overhead

## Highlights & Insights

1. **Data-driven prior learning**: Rather than relying on hand-crafted parametric models, GeneMAN learns 2D/3D human priors from 50K+ multi-source data — a more scalable paradigm where more data yields stronger priors.
2. **Complementary hybrid prior design**: The 2D prior supplies detail (texture, fine-grained geometry) while the 3D prior enforces consistency (multi-view, natural pose); the two are organically combined via SDS loss, each fulfilling a distinct role.
3. **Coarse-to-fine hierarchical strategy** is applied throughout: NeRF → DMTet for geometry from coarse to fine, and latent space → pixel space for texture from coarse to fine, with the most appropriate representation and supervision at each stage.
4. **Template-free design** eliminates dependence on accurate SMPL estimation, making the framework naturally suited for children, non-standard body shapes, and object occlusion scenarios that existing methods struggle with.
5. **Training-free multi-view consistency strategy** is an elegant engineering design — adding the same noise to batch-rendered views and concatenating them for inference improves cross-view texture consistency without retraining the model.

## Limitations & Future Work

1. **Slow inference**: The full pipeline takes approximately 1.4 hours on a single A100 80G GPU, including NeRF optimization, DMTet sculpting, and two-stage texture optimization — far from meeting the demands of interactive applications.
2. **High training cost**: The 3D prior requires 8×A100 for one week, and the 2D prior requires 4×A100 for five days, posing a high barrier to entry.
3. **Reliance on the SDS optimization paradigm**: SDS inherently suffers from well-known issues such as over-saturation and mode collapse; while partially mitigated by the hybrid prior design, these issues are not fundamentally resolved.
4. **Lack of hand/face-specific quantitative evaluation**: The reported metrics (PSNR/LPIPS/CLIP-Sim) are global measures and do not specifically assess the reconstruction accuracy of facial expressions or hand poses.
5. **Dataset bias risk**: Synthetic data generated by ControlNet may introduce specific distributional biases; the paper provides no in-depth analysis of how domain gaps across multi-source data are balanced.

## Related Work & Insights

- **LRM (Hong et al., 2023)**: Demonstrates that transformer-based 3D reconstructors trained on multi-source data exhibit strong generalization, inspiring GeneMAN's data strategy
- **DreamCraft3D (Sun et al., 2023)**: Combines 2D+3D priors in a coarse-to-fine optimization pipeline; GeneMAN extends and deepens this paradigm
- **HumanNorm (Huang et al., 2024)**: Pretrained normal/depth-adaptive diffusion models are directly adopted by GeneMAN for DMTet sculpting guidance
- **Sapiens (Khirodkar et al., 2024)**: A human foundation model that provides normal and depth priors for the reference view, serving as a geometric signal source in GeneMAN
- The overall technical approach can be viewed as a **"human-specialized DreamCraft3D"**, with the core innovation lying in replacing general-domain data with multi-source human data to train stronger domain-specific priors

## Rating

| Dimension | Score (1–5) | Comments |
|-----------|:-----------:|---------|
| Novelty | 3.5 | Individual components (SDS, NeRF→DMTet, SDEdit texture refinement) are human-specialized adaptations of existing methods; core innovation lies in multi-source data-driven prior learning and the template-free design |
| Technical Depth | 4.0 | System design is solid; module selection and loss design at each stage of the multi-stage pipeline are well-motivated |
| Experimental Thoroughness | 4.0 | Comprehensive quantitative, qualitative, user study, and ablation experiments; in-the-wild image tests cover diverse challenging scenarios |
| Writing Quality | 4.0 | Clear structure, rich illustrations, and detailed method descriptions |
| Value | 3.5 | Results are impressive, but the long inference time must be reduced to an interactive level before practical deployment |
| **Overall** | **3.8** | A solid systems paper that advances single-image human reconstruction through multi-source data, dual priors, and multi-stage optimization |

## Related Work & Insights (Comparison Table)

| Method | Type | Human Prior | Texture Optimization | Strengths | Limitations |
|--------|------|-------------|---------------------|-----------|-------------|
| PIFu / PIFuHD | Template-free | Pixel-aligned implicit field | None | End-to-end, no SMPL dependency | Poor side-view geometry, unrealistic texture |
| PaMIR / ICON / ECON | Template-based | SMPL features | None / Limited | Exploits body structure information | Clothing detail limited by SMPL topology |
| SiTH | Template-based | Fine-tuned diffusion for back-view hallucination | SDEdit | Relatively fast | Depends on HPS accuracy, poor generalization on wild images |
| TeCH | Template-based | SDS optimization | Diffusion refinement | Rich detail | Overly noisy surface, inconsistent texture |
| GTA | Template-based | Transformer triplane | Feed-forward | Fast inference | Depends on SMPL, fails on loose clothing |
| HumanLRM | Template-free | Diffusion-guided feed-forward | Implicit field | No SMPL, feed-forward | Texture inconsistency, insufficient geometric detail |
| **GeneMAN** | **Template-free** | **2D+3D diffusion priors fine-tuned on multi-source data** | **Latent + pixel space** | **Strong generalization, robust to body proportions and personal belongings** | **Slow inference (~1.4h)** |

**Core distinction**: GeneMAN simultaneously achieves template-free reconstruction (avoiding cascading SMPL estimation errors) and rich prior learning (diffusion priors trained on 50K+ multi-source data), which is the fundamental reason for its substantial advantage over template-based methods on in-the-wild images.

**Additional Insights**:

1. **Data > Architecture**: GeneMAN's individual modules (NeRF, DMTet, SDS, SDEdit) are all standard components; its competitive edge stems from domain-specific priors trained on 50K+ multi-source human data. This suggests that in vertical-domain 3D reconstruction, high-quality domain data collection and curation may be more impactful than architectural innovation.
2. **Generalizability of the hybrid prior paradigm**: The dual-prior design — 2D prior for detail + 3D prior for consistency — is transferable to single-image 3D reconstruction in other specific domains (animals, vehicles, architecture, etc.).
3. **Complementarity with feed-forward methods**: GeneMAN follows an optimization-based approach (~1.4h), yielding high quality but slow speed; feed-forward methods such as LRM/Instant3D are fast but quality-limited. A promising future direction is to initialize feed-forward networks with GeneMAN-level priors to combine quality and speed.
4. **Sapiens as a human foundation model**: The paper uses Sapiens to provide depth/normal priors for the reference view, suggesting that foundation models can serve as plug-and-play geometric signal sources for downstream 3D tasks.
5. **Training-free multi-view consistency trick**: Adding the same noise to multi-view renderings and concatenating them for inference is a simple yet effective technique that may generalize to other SDS-based 3D generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](../../ICLR2026/image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](../../CVPR2026/image_generation/interedit_navigating_textguided_multihuman_3d_moti.md)

</div>

<!-- RELATED:END -->
