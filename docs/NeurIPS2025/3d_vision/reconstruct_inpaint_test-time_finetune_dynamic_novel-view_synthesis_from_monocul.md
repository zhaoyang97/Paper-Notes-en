---
title: >-
  [Paper Note] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos
description: >-
  [NeurIPS 2025][3D Vision][dynamic novel-view synthesis] This paper proposes CogNVS, which decomposes dynamic scene novel-view synthesis into a three-stage pipeline — 3D reconstruction (recovering visible pixels) → video…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "dynamic novel-view synthesis"
  - "monocular video"
  - "video inpainting"
  - "test-time finetuning"
  - "diffusion models"
date: 2026-05-08
content_hash: b00d3b471aec0d42
---

# Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos

**Conference**: NeurIPS 2025
**arXiv**: [2507.12646](https://arxiv.org/abs/2507.12646)
**Code**: [https://cog-nvs.github.io/](https://cog-nvs.github.io/)
**Area**: 3D Vision
**Keywords**: dynamic novel-view synthesis, monocular video, video inpainting, test-time finetuning, diffusion models

## TL;DR
This paper proposes CogNVS, which decomposes dynamic scene novel-view synthesis into a three-stage pipeline — 3D reconstruction (recovering visible pixels) → video diffusion inpainting (generating occluded regions) → test-time finetuning (adapting to the target video distribution) — training the inpainting model with purely 2D video self-supervision to achieve zero-shot generalization to new test videos.

## Background & Motivation

**Background**: Dynamic novel-view synthesis from monocular video is an extremely challenging problem. Two main directions exist: (i) test-time optimization of 4D representations (e.g., 4DGS, Shape-of-Motion) — geometrically accurate but requiring hours of computation and failing when novel viewpoints deviate significantly from training views; (ii) large-scale feed-forward video models (GCD, TrajectoryCrafter) — fast but lacking 3D consistency.

**Limitations of Prior Work**:
   - 4D optimization methods cannot handle occluded regions — they can only synthesize "co-visible" pixels
   - Data-driven methods suffer from over-hallucination — objects appear or disappear abruptly
   - **Multi-view** training data for dynamic scenes is scarce — the best available data source is monocular 2D video

**Key Challenge**: 3D reconstruction yields geometrically precise but incomplete renderings; diffusion-based generation yields complete but geometrically inconsistent results.

**Key Insight**: Disentanglement — co-visible pixels are rendered via 3D reconstruction (accurate), while occluded pixels are generated via video diffusion inpainting (creative).

**Core Idea**: CogNVS is a conditional video inpainting diffusion model that generates only occluded regions while preserving known regions. During training, 2D video self-supervision is used (randomly sampled camera trajectories construct co-visibility masks as training pairs); at test time, test-time finetuning (TTF) adapts the model to the target video.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Reconstruct**: MegaSAM is used to obtain a dynamic 3D reconstruction $\mathcal{G}_{src}$ and camera odometry $\mathbf{c}_{src}$ from the monocular video. The scene is rendered to novel viewpoints to produce partially visible pixels $\mathbf{V}_{nvs}^{cov}$. (2) **Inpaint**: CogNVS (built on CogVideoX-5B) conditions on the partially visible rendering to generate the complete novel-view video $\mathbf{V}_{nvs}$ — inpainting occluded regions while also allowing updates to visible-region appearance (e.g., view-dependent lighting). (3) **Test-Time Finetune**: CogNVS is finetuned for 200–400 steps using AdamW on self-supervised pairs constructed from the target video, reducing the train-test domain gap.

### Key Designs

1. **Self-Supervised Data Construction**:

    - Function: Constructs inpainting training pairs from **arbitrary** 2D monocular videos without requiring multi-view ground truth.
    - Mechanism: Given a source video → reconstruct → randomly sample $N$ novel camera trajectories → find co-visible 3D points $\mathcal{G}_{src,n}^{cov}$ between source and novel viewpoints → render to the source viewpoint to obtain a "partially visible source video" $\mathbf{V}_{src,n}^{cov}$. Training pair = (partially visible source video, complete source video).
    - Design Motivation: **3D-aware masks** (as opposed to random 2D masks) better simulate real 3D visibility, as occlusion patterns are directly tied to viewpoint changes.

2. **CogNVS Architecture**:

    - Built on CogVideoX-5B (Transformer-based video diffusion model) with self-attention and 3D-RoPE.
    - Originally an image-to-video model, adapted to video-to-video — the conditioning video and target video are shape-aligned, eliminating the need for padding.
    - Conditioning input: VAE-encoded partially visible novel-view rendering $\mathbf{z}_{cond}$.
    - Target output: complete novel-view video.
    - Training objective: standard score matching $\|\epsilon_\theta(\mathbf{z}_k, k, \mathbf{z}_{cond}) - \epsilon\|_2^2$.

3. **Test-Time Finetuning (TTF)**:

    - Function: Reduces the domain gap between pretraining data and the target test video.
    - Mechanism: Self-supervised pairs are constructed from the target test video itself (using the same reconstruction + random trajectory approach), and CogNVS is finetuned for 200–400 steps with AdamW.
    - Design Motivation: Lighting, appearance, and motion patterns vary substantially across videos — the general priors of the pretrained model are insufficient. TTF allows the model to internalize the characteristics of the target video while retaining general inpainting capability.
    - Key Findings: **TTF is the critical component separating "competitive" from "state-of-the-art" performance** — without TTF, performance is on par with competing methods; with TTF, it surpasses all prior approaches.

### Loss & Training
- Pretraining: 10K 2D videos (SA-V, TAO, YouTube-VOS, DAVIS); full fine-tuning of all 42 Transformer layers; 12K steps; 3 days × 8 A6000 GPUs.
- TTF: 200–400 AdamW steps per test video, learning rate 2e-5.

## Key Experimental Results

### Main Results — Kubric-4D (Synthetic Dynamic Scenes, Zero-Shot)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Training Data |
|--------|-------|-------|--------|------|---------------|
| GCD | 20.1 | 0.734 | 0.186 | 85.3 | Kubric (in-domain) |
| TrajectoryCrafter | 18.5 | 0.701 | 0.228 | 102.4 | Large-scale |
| Gen3C | 19.3 | 0.715 | 0.210 | 93.5 | Large-scale |
| **CogNVS (zero-shot+TTF)** | **20.8** | **0.745** | **0.172** | **78.6** | 2D video (OOD) |

### Main Results — DyCheck (Real Dynamic Scenes)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Type |
|--------|-------|-------|--------|------|
| Shape-of-Motion | 21.2 | 0.665 | 0.341 | Test-time optimization (hours) |
| MoSca | 20.8 | 0.651 | 0.356 | Test-time optimization |
| **CogNVS (TTF)** | **21.9** | **0.683** | **0.312** | Feed-forward + TTF (minutes) |

### Ablation Study — Impact of TTF

| Configuration | Kubric PSNR↑ | DyCheck PSNR↑ | Notes |
|---------------|-------------|--------------|-------|
| CogNVS (w/o TTF) | 19.2 | 20.4 | Pretrained model only |
| CogNVS (**w/ TTF**) | **20.8** | **21.9** | **+1.6/+1.5 PSNR** |
| Random 2D mask training (w/o TTF) | 18.1 | 19.2 | 3D mask > 2D mask |

### Key Findings
- **TTF is the single most critical component** — it contributes 1.5+ PSNR improvement, lifting performance from "competitive" to "surpassing all prior methods."
- **Zero-shot performance exceeds in-domain-trained GCD** — CogNVS has never seen Kubric data yet outperforms GCD, demonstrating that the 2D video self-supervision + TTF paradigm generalizes more effectively than dataset-specific training.
- 3D-aware masks outperform random 2D masks by 1+ PSNR — occlusion patterns must be aligned with 3D visibility.
- CogNVS produces **sharp dynamic objects** (whereas other methods are blurry in dynamic regions) — because the inpainting model specializes in occluded regions and does not need to fit dynamics from limited viewpoints as 4D representations do.

## Highlights & Insights
- The **"reconstruct + inpaint + finetune" three-stage disentanglement** is conceptually clean and each stage can be improved independently — best SLAM for reconstruction, best diffusion model for inpainting, standard TTF for adaptation.
- **Self-supervised training data construction is the core contribution** — it transforms "NVS training requiring multi-view GT" into "training with arbitrary 2D video," unlocking vast quantities of internet video data.
- TTF represents the best of both worlds between test-time optimization and feed-forward inference — retaining the robustness of data-driven methods (from large-scale pretraining) while achieving optimization-level accuracy (from test-time adaptation).
- The method is the **first** to achieve "feed-forward speed + optimization-level accuracy" for dynamic scene NVS.

## Limitations & Future Work
- Inference still requires ~5 minutes per video due to the size of CogVideoX — smaller, faster diffusion models could accelerate this.
- The method depends on MegaSAM reconstruction quality — failures in reconstruction propagate incorrect visibility masks.
- Real-scene evaluation (DyCheck) covers only 5 videos — validation on larger real-world benchmarks remains to be conducted.
- When the novel viewpoint diverges substantially from the source (e.g., 180° rotation), the reconstructed co-visible region becomes too sparse, effectively degrading to near-pure inpainting.

## Related Work & Insights
- **vs. GCD (ECCV'24)**: GCD is trained on Kubric and is domain-specific; CogNVS trains on 2D videos and generalizes in a zero-shot manner.
- **vs. TrajectoryCrafter (concurrent)**: TrajectoryCrafter over-hallucinates occluded regions (without preserving geometry); CogNVS's reconstruction prior ensures accuracy in co-visible regions.
- **vs. CAT4D (concurrent)**: CAT4D employs score distillation and has limited generalization; CogNVS achieves stronger generalization via TTF.
- The "inpaint rather than reconstruct" paradigm can be extended to other 3D tasks such as 3D completion and scene editing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes dynamic NVS as a video inpainting problem; the combination of self-supervised training and TTF is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Zero-shot evaluation on 3 datasets with detailed ablations (TTF / mask type / training data / step count).
- Writing Quality: ⭐⭐⭐⭐⭐ Stage-by-stage presentation is clear; the explanation of self-supervised data construction is intuitive.
- Value: ⭐⭐⭐⭐⭐ Advances the state of the art in monocular dynamic NVS, with methodological insights applicable to broader 3D vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)

</div>

<!-- RELATED:END -->
