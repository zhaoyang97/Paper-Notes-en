---
title: >-
  [Paper Note] No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves
description: >-
  [ICLR2026][Self-Supervised Learning][Diffusion Transformer] This paper proposes Self-Representation Alignment (SRA), which identifies that internal representations of diffusion Transformers exhibit a quality gradient alo…
tags:
  - "ICLR2026"
  - "Self-Supervised Learning"
  - "Diffusion Transformer"
  - "Self-Representation Alignment"
  - "Self-Distillation"
  - "DiT/SiT Accelerated Training"
  - "Internal Representation Guidance"
date: 2026-05-08
content_hash: b1084c1f99391611
---

# No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves

**Conference**: ICLR2026  
**arXiv**: [2505.02831](https://arxiv.org/abs/2505.02831)  
**Code**: [https://github.com/vvvvvjdy/SRA](https://github.com/vvvvvjdy/SRA)  
**Area**: Self-Supervised Learning / Image Generation  
**Keywords**: Diffusion Transformer, Self-Representation Alignment, Self-Distillation, DiT/SiT Accelerated Training, Internal Representation Guidance

## TL;DR

This paper proposes Self-Representation Alignment (SRA), which identifies that internal representations of diffusion Transformers exhibit a quality gradient along two dimensions—increasing layer depth and decreasing noise level. Based on this observation, SRA aligns early-layer, high-noise representations of a student network to late-layer, low-noise representations of an EMA teacher, **requiring no external representation components (DINOv2/CLIP/MAE)**, and substantially accelerates convergence while improving generation quality on DiT and SiT (SiT-XL/2 achieves FID 1.58 at 800 epochs, comparable to REPA which relies on DINOv2).

## Background & Motivation

**Background**: Diffusion Transformers (DiT, SiT) have become the dominant architecture for image generation, achieving strong performance on ImageNet through the scalability of Transformers. Recent works (MaskDiT, REPA, TREAD) have demonstrated that incorporating **representation learning guidance** into diffusion training can simultaneously accelerate convergence and improve final generation quality.

**Limitations of Prior Work**: Existing representation guidance methods all require **external components**—MaskDiT/SD-DiT attach discriminative losses from MAE/IBOT during DiT training (requiring additional reconstruction branches and masking strategies); REPA directly relies on frozen large-scale pretrained visual encoders (DINOv2) to provide patch-level alignment targets. These approaches either increase architectural complexity or become difficult to apply in domains lacking high-quality pretrained encoders (e.g., video generation).

**Key Challenge**: Representation guidance is critical for accelerating diffusion model training, yet dependence on external components bottlenecks extension to new domains and architectures. A key question emerges: **Can guidance signals be extracted entirely from the diffusion model itself, eliminating all external dependencies?**

**Goal**: (1) Verify whether diffusion Transformers naturally exhibit exploitable representation quality gradients internally; (2) Design a zero-external-dependency self-representation alignment method to accelerate training.

**Key Insight**: Unlike representation models that extract features from clean images, diffusion models operate on noisy inputs and progressively denoise—meaning their internal process is inherently a "coarse-to-fine" discrimination process. The authors systematically analyze SiT-XL/2 and DiT-XL/2 via PCA visualization and linear probing, finding that internal representations indeed exhibit a monotonically increasing quality trend along two dimensions—**increasing layer depth** and **decreasing noise level**: features at shallow layers under high noise are coarse and blurry, while features at deep layers under low noise are semantically clear. This intrinsic "coarse-to-fine" process can naturally serve as a self-supervised alignment signal.

**Core Idea**: Align latent representations from the student network's early layers at high-noise timesteps to those from the EMA teacher's late layers at low-noise timesteps, leveraging the model's **own representation quality gradient** for self-supervised guidance.

## Method

### Overall Architecture

SRA augments the standard diffusion generation training (noise prediction loss for DiT or velocity field loss for SiT) with a **self-representation alignment loss** $\mathcal{L}_{sa}$. The framework comprises two networks: a trainable **student model** $f$ and a **teacher model** $f_*$ updated via EMA. When the student processes input at timestep $t$ (higher noise), the output features at layer $m$ are transformed through a projection head and aligned to the output features of the teacher at layer $n$ ($n \geq m$) when processing input at timestep $t-k$ (lower noise). The projection head can be discarded after training, **leaving the inference architecture of the original diffusion Transformer unchanged**.

### Key Designs

1. **"Coarse-to-Fine" Empirical Finding and Internal Representation Quality Gradient**:

    - Function: Provides the methodological foundation for SRA by demonstrating that diffusion Transformers naturally contain exploitable representation guidance signals.
    - Mechanism: Patch features are extracted from pretrained SiT-XL/2 and DiT-XL/2 at different layers and noise levels; semantic structure is observed via PCA visualization and discriminative ability is quantified via ImageNet linear probing. Results show: (a) Layer 3 representations are near-random under high noise, but by layer 20 under low noise they clearly delineate semantic regions; (b) linear probing accuracy peaks at intermediate-to-deep layers (around layer 20) before declining as the model shifts toward high-frequency generation details. Both trends are highly consistent across DiT and SiT.
    - Design Motivation: Since diffusion models already contain weak-to-strong discriminative signals internally, no external introduction is necessary. This finding constitutes the core insight of the entire method.

2. **Self-Representation Alignment Loss (SRA Loss)**:

    - Function: Constructs a self-supervised alignment objective by exploiting representation quality differences across layers and timesteps.
    - Mechanism: The student's layer-$m$ output $\mathbf{y} = f^m(\mathbf{x}_t, t, c)$ at timestep $t$ is transformed by a lightweight projection head $j_\psi$ and aligned to the teacher's layer-$n$ output $\mathbf{y}_* = f_*^n(\mathbf{x}_{t-k}, t-k, c)$ at timestep $t-k$ via patch-wise distance minimization: $\mathcal{L}_{sa} = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\text{dist}(\mathbf{y}_*^{[i]}, j_\psi(\mathbf{y}^{[i]}))\right]$, where dist is the smooth-$\ell_1$ distance. Key constraints: $m \leq n$ (layer-wise: shallow to deep) and $k \geq 0$ (timestep-wise: high to low noise), simultaneously exploiting quality differences along both dimensions.
    - Design Motivation: Layer-wise alignment alone (same timestep) or timestep-wise alignment alone (same layer) provides limited benefit; combining both dimensions yields sufficiently strong guidance. The projection head prevents direct alignment from disrupting the generative role of each layer-timestep pairing.

3. **EMA Teacher Network and Stability-Trick-Free Design**:

    - Function: Provides a stable and continuously improving alignment target.
    - Mechanism: Teacher parameters are updated via $\zeta_t = \alpha \zeta_t + (1-\alpha)\zeta_s$. Unlike self-supervised learning (BYOL/DINO), which requires cosine-scheduled $\alpha$, centering, and batch normalization for stability, SRA uses a fixed $\alpha = 0.9999$ throughout training without any collapse-prevention mechanisms. The authors hypothesize this is because the diffusion generation loss itself provides strong gradient signals that naturally stabilize training.
    - Design Motivation: Minimalist design—no clustering constraints, batch normalization, or centering are introduced, reducing hyperparameter tuning burden and implementation complexity.

### Loss & Training

The total loss is a weighted sum of the generation loss and the self-alignment loss: $\mathcal{L} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{sa}$, with $\lambda = 0.2$. The generation loss is noise prediction MSE for DiT and velocity field prediction MSE for SiT. Default alignment layer configurations: B model $3 \to 8$, L model $6 \to 16$, XL model $8 \to 20$ (SiT) / $8 \to 16$ (DiT). The timestep offset $k$ is sampled uniformly from $[0, 0.2)$ (SiT) or $\lfloor[0, 200)\rfloor$ (DiT). Training follows the original DiT/SiT setup throughout: AdamW, lr=1e-4, batch size 256, latents extracted via SD-VAE.

## Key Experimental Results

### Main Results: ImageNet 256×256 (CFG)

| Method | External Dependency | Epochs | FID↓ | sFID↓ | IS↑ | Pre.↑ | Rec.↑ |
|--------|--------------------|----|------|-------|-----|-------|-------|
| DiT-XL/2 | None | 1400 | 2.27 | 4.60 | 278.2 | 0.83 | 0.57 |
| SiT-XL/2 | None | 1400 | 2.06 | 4.50 | 270.3 | 0.82 | 0.59 |
| MaskDiT | MAE loss | 1600 | 2.28 | 5.67 | 276.6 | 0.80 | 0.61 |
| DiT + TREAD | MAE loss | 740 | 1.69 | 4.73 | 292.7 | 0.81 | 0.63 |
| SiT + REPA | DINOv2 | 800 | **1.42** | 4.70 | 305.7 | 0.80 | 0.65 |
| **SiT + SRA** | **None** | **400** | 1.85 | **4.50** | 297.2 | **0.82** | 0.61 |
| **SiT + SRA** | **None** | **800** | 1.58 | 4.65 | **311.4** | 0.80 | 0.63 |

SRA at 400 epochs surpasses the original SiT-XL at 1400 epochs; at 800 epochs it achieves FID 1.58 and IS 311.4, substantially outperforming MaskDiT and approaching REPA's 1.42, all **without relying on any external model**.

### ImageNet 512×512 and Text-to-Image

| Setting | Method | FID↓ | IS↑ | PickScore↑ |
|---------|--------|------|-----|-----------|
| 512×512, 200 ep | SiT + REPA | **2.08** | 274.6 | - |
| 512×512, 200 ep | **SiT + SRA** | 2.17 | **279.3** | - |
| COCO T2I, 150K | MMDiT | 5.86 | - | 20.05 |
| COCO T2I, 150K | MMDiT + REPA | **4.60** | - | 20.88 |
| COCO T2I, 150K | **MMDiT + SRA** | 4.85 | - | **21.14** |

SRA proves effective at higher resolutions and in text-to-image settings; at 512×512 it surpasses REPA on IS and sFID, and achieves the highest PickScore on COCO T2I.

### Ablation Study (SiT-B/2, 400K iter, no CFG)

| Configuration | FID↓ | IS↑ | Note |
|---------------|------|-----|------|
| Baseline SiT-B/2 | 33.02 | 43.71 | No SRA |
| $3 \to 3$, same-layer alignment | 37.08 | 41.54 | No layer-wise quality signal; worse than baseline |
| $3 \to 8$, $k=0$ (fixed same timestep) | 31.07 | 47.32 | Layer-wise gradient only |
| $3 \to 8$, $k \in [0,0.2)$ | **29.10** | **50.20** | Layer-wise + timestep-wise dual guidance; optimal |
| $3 \to 8$, no projection head | 34.23 | 41.07 | Direct alignment disrupts feature space |
| $3 \to 8$, with projection head | **29.10** | **50.20** | Projection head preserves generative roles |
| $\lambda=0.1$ | 30.65 | 48.31 | Alignment signal too weak |
| $\lambda=0.2$ | **29.10** | **50.20** | Optimal balance |
| $\lambda=0.4$ | 29.75 | 49.30 | Excessive weight slightly impairs generation |

### Key Findings

- **Both layer-wise and timestep-wise alignment are indispensable**: Layer-wise alignment alone ($k=0$) yields FID 31.07; adding the timestep offset improves it to 29.10; same-layer same-timestep alignment ($3 \to 3$) even underperforms the baseline.
- **The projection head is critical**: Removing it degrades FID from 29.10 to 34.23 (worse than baseline), as direct alignment disrupts the generative roles assigned to each layer-timestep pairing.
- **Teacher representation quality strongly correlates with generation quality**: Linear probing accuracy exhibits a near-linear negative correlation with FID, validating the core hypothesis that better representation guidance leads to better generation.
- **SRA's gains do not saturate**: REPA saturates around 200 epochs (as the external encoder's information is exhausted), whereas SRA's EMA teacher continuously improves (linear probing accuracy rises from 38.1% at 200K iterations to 54.2% at 800K iterations), enabling progressively stronger guidance.
- **Larger models benefit more from SRA**: The relative FID improvement scales from B to L to XL, consistent with scaling laws observed in self-supervised learning.

## Highlights & Insights

- **The "coarse-to-fine" dual-dimension insight is remarkably elegant**: Diffusion models inherently transition from noise to clarity, and layer-wise representations are progressively refined from shallow to deep. Jointly exploiting these two quality gradients constructs a self-supervised objective that requires no external signal—an insight with strong generality and scalability.
- **Minimalist design philosophy**: The entire method adds only a projection head and an EMA copy, requiring no masking, no additional encoders, and none of the SSL stabilization tricks (centering, BN, dynamic momentum scheduling). The projection head is discarded after training without affecting inference. The method is engineering-friendly.
- **Complementarity with REPA**: REPA converges rapidly in early training but saturates later (external encoder information exhausted); SRA converges more gradually but continues improving over time (teacher keeps improving). The paper hints that the two can be combined, with REPA providing a fast warm-start followed by sustained SRA optimization.

## Limitations & Future Work

- **Alignment layers and timestep offsets require manual selection**: Different model sizes (B/L/XL) require individually tuned layer indices (e.g., $3 \to 8$ vs. $8 \to 20$); the paper provides heuristic principles but no automated search mechanism.
- **Experiments are primarily limited to ImageNet and COCO**: Validation at high resolution (1024+) or on large-scale text-image data is absent. The text-to-image experiment uses only a small-scale setup (150K iterations of MMDiT on COCO2014).
- **Video generation is not validated**: The authors explicitly note that compute constraints precluded text-to-video experiments, though they argue for SRA's conceptual feasibility in video—a domain that lacks strong pretrained encoders and where SRA's zero-external-dependency advantage is most pronounced.
- **The EMA teacher incurs additional memory and compute overhead**: A full model copy must be maintained with an additional forward pass; the appendix provides concrete training speed and memory data, but optimization strategies are not discussed in depth.
- **Theoretical understanding remains limited**: Why does representation guidance accelerate generative training? Why does combining layer-wise and timestep-wise dimensions so substantially outperform either alone? The paper acknowledges these questions are experiment-driven and lacks theoretical analysis.

## Related Work & Insights

- **vs. REPA**: REPA uses frozen DINOv2 for patch-wise alignment targets, achieving extremely fast early convergence (<200 epochs) but experiencing guidance saturation or even degradation in later stages (as demonstrated by concurrent work). SRA requires no external model; the EMA teacher continuously improves, enabling sustained FID reduction beyond 200 epochs. The two approaches have clear complementary potential.
- **vs. MaskDiT/SD-DiT/TREAD**: These methods introduce MAE/IBOT discriminative losses or token routing strategies, requiring additional masking mechanisms and auxiliary branches with strong architectural invasiveness. SRA does not alter the original architecture and is more plug-and-play.
- **vs. BYOL/DINO self-distillation**: EMA teachers in SSL require careful stability design (dynamic momentum scheduling, centering, BN) to prevent mode collapse. SRA achieves stable training with a fixed $\alpha=0.9999$, possibly because the generation loss itself provides sufficiently strong gradient signals to naturally prevent collapse—a difference worth further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ The "coarse-to-fine" dual-dimension insight is novel, though the EMA teacher self-distillation framework is widely used in SSL; SRA represents a clever adaptation of this paradigm to diffusion models rather than an entirely new framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across B/L/XL model sizes and 256/512 resolutions on DiT/SiT with detailed ablations; large-scale T2I/T2V experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is developed naturally and fluidly; the logical chain from empirical analysis to method design to experimental validation is clear and complete, with high-quality figures and tables.
- Value: ⭐⭐⭐⭐ Provides a simple, zero-external-dependency acceleration scheme for diffusion model training, with particularly strong practical value for domains lacking strong pretrained encoders (video, 3D).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](../../CVPR2026/self_supervised/diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](../../ICML2026/self_supervised/flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](../../AAAI2026/self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2026/self_supervised/representation_learning_for_spatiotemporal_physica.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)

</div>

<!-- RELATED:END -->
