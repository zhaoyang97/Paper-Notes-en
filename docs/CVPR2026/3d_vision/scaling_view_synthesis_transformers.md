---
title: >-
  [Paper Note] Scaling View Synthesis Transformers (SVSM)
description: >-
  [CVPR 2026][3D Vision][Novel View Synthesis] This work establishes, for the first time, scaling laws for geometry-free NVS Transformers. It proposes the effective batch size hypothesis ($B_\text{eff} = B \cdot V_T$) to r…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Scaling Laws"
  - "Transformer"
  - "Encoder-Decoder"
  - "Computational Efficiency"
  - "PRoPE"
date: 2026-05-08
content_hash: e56b3628f6b3f076
---

# Scaling View Synthesis Transformers (SVSM)

**Conference**: CVPR 2026
**arXiv**: [2602.21341](https://arxiv.org/abs/2602.21341)  
**Code**: [https://www.evn.kim/research/svsm](https://www.evn.kim/research/svsm)  
**Area**: 3D Vision / Novel View Synthesis / Scaling Laws
**Keywords**: Novel View Synthesis, Scaling Laws, Transformer, Encoder-Decoder, Computational Efficiency, PRoPE

## TL;DR

This work establishes, for the first time, scaling laws for geometry-free NVS Transformers. It proposes the effective batch size hypothesis ($B_\text{eff} = B \cdot V_T$) to reveal the root cause of the underestimation of encoder-decoder architectures, designs a unidirectional encoder-decoder architecture called SVSM, and achieves a new state of the art on RealEstate10K (30.01 PSNR) with less than half the training FLOPs. The Pareto frontier shifts 3× to the left relative to LVSM decoder-only.

## Background & Motivation

**Lack of scaling analysis in NVS**: Systematic scaling laws have been established in NLP (Chinchilla, Kaplan) and 2D vision (DiT), yet the 3D vision/NVS domain remains entirely unexplored—there are no principled, compute-optimal guidelines for model design or training configuration.

**Severe redundancy in decoder-only architectures**: LVSM decoder-only re-processes all context tokens for every target view. Its MLP cost scales as $\propto V_T \times (V_C + 1)$ and attention cost as $\propto V_T \times (V_C + 1)^2$, both growing linearly with the number of target views.

**Encoder-decoder unfairly dismissed**: The encoder-decoder variant in the original LVSM paper performed substantially worse than decoder-only, but this work identifies two root causes: (a) a fixed-size scene latent representation introduces an information bottleneck, and (b) the comparison was conducted under unequal compute budgets—neither reflects an inherent architectural disadvantage.

**Unknown interaction between target views and batch size**: Standard NVS training practice reconstructs multiple target views per scene, yet the effect of increasing $V_T$ versus increasing $B$ on training dynamics has never been formally analyzed.

**Open question on multi-view ($V_C > 2$) scaling**: Whether extending an encoder-decoder to more context views causes scaling degradation due to the scene representation bottleneck remains an open problem.

## Method

### Overall Architecture

Context images $\mathcal{C} = \{(I_i, g_i, K_i)\}$ → **Transformer Encoder** (bidirectional self-attention) → scene representation $z = E[\mathcal{C}]$ (all patch tokens, no fixed bottleneck) → **Cross-Attention Decoder** (unidirectional) → parallel rendering of $V_T$ target views $\tilde{I} = D[z, g_T, K_T]$. The core design principle is: encode once, decode many times; target views do not interact with each other but can be decoded in parallel.

### 1. SVSM Architecture (Section 3)

- **Encoder**: A standard ViT that applies bidirectional self-attention across all context images and outputs a set of patch tokens as the scene representation. The key distinction from LVSM enc-dec is that patch tokens are retained in full rather than compressed into a fixed number of learnable tokens, thereby avoiding the information bottleneck.
- **Decoder**: Extracts information from the scene representation $z$ via cross-attention and renders target views autoregressively. Each target view is decoded independently but shares $z$, enabling parallel execution.
- **Computational complexity**: $\chi_\text{MLP}(\text{SVSM}) \propto V_T + V_C$; $\chi_\text{Attn}(\text{SVSM}) \propto V_C \times (V_T + V_C)$. When $V_T \gg V_C$, this reduces to $O(V_T)$, compared to LVSM's $O(V_T \cdot V_C + V_T)$.
- **Trade-off**: The encoder cannot actively discard information irrelevant to the target. At equal parameter count and training steps, SVSM underperforms LVSM; however, by reinvesting the compute savings from amortized rendering into larger models and longer training, **SVSM substantially outperforms LVSM under equal compute budgets**.

### 2. Effective Batch Size Hypothesis (Section 4)

- **Definition**: $B_\text{eff} \equiv B \cdot V_T$, where $B$ is the number of scenes and $V_T$ is the number of target views per scene.
- **Empirical validation**: Ablations on DL3DV ($V_C = 8$) and RE10K ($V_C = 2$) with fixed $B_\text{eff}$ across varying $(B, V_T)$ combinations show that final PSNR differences are within $\pm 0.1$–$0.2$, and training loss curves are nearly identical.
- **Implication for LVSM**: $\chi(\text{LVSM}) \propto B \cdot V_T \cdot (V_C + 1) = B_\text{eff} \cdot (V_C + 1)$, which is independent of the $(B, V_T)$ split—adjusting $V_T$ yields no compute savings.
- **Implication for SVSM**: $\chi(\text{SVSM}) \propto B \cdot (V_C + V_T) = B_\text{eff} + B \cdot V_C$. Reducing $B$ while increasing $V_T$ preserves $B_\text{eff}$ (and thus performance) while reducing total FLOPs—this is the source of the encoder-decoder efficiency advantage.
- **Insight**: The poor enc-dec performance reported in the original LVSM paper stems from comparing under equal iteration counts rather than equal FLOPs, which obscures the computational efficiency of the encoder-decoder design.

### 3. Stereo Scaling Laws (Section 5, $V_C = 2$)

- **Experimental setup**: RE10K, $V_T = 6$, batch size 256, patch size 16; model sizes sweep from 7M to 300M parameters across 3–4 training token counts, with total compute spanning $10^3\times$ (100 petaflops to 100 exaflops).
- **Scaling results**: Both model families exhibit identical Pareto frontier slopes on log-log plots, but SVSM's frontier is shifted 3× to the left—achieving equal performance at one-third the FLOPs.
- **Chinchilla analysis**: For each compute budget $\chi$, optimal $(N_\text{opt}, D_\text{opt})$ is identified and fitted as $N_\text{opt} \propto \chi^a$, $D_\text{opt} \propto \chi^b$. SVSM: $a = 0.52$, $b = 0.47$ ($a \approx b$, consistent with Chinchilla—doubling the budget should be split equally between model size and data); LVSM: $a = 0.65$, $b = 0.33$ (more model-biased).
- **Training stability**: $1/\sqrt{L}$ residual scaling (depth-$\mu$P) is applied to ensure fair comparison across models of different depths.
- **Final models**: SVSM-416M (Pareto-optimal) and SVSM-740M (iteration-matched) both surpass LVSM-171M at approximately 0.77 zflops—roughly half the FLOPs of LVSM.

### 4. Multi-View Scaling Laws (Section 6, $V_C > 2$)

- **Problem**: Directly extending SVSM to $V_C = 4$ causes the Pareto frontier to saturate rapidly, and scaling behavior disappears.
- **Root cause**: The scene representation in the encoder-decoder acts as an information bottleneck, causing pose information to be lost in deeper layers.
- **Solution — PRoPE**: Projected Rotary Position Encoding projects queries, keys, and values into a common reference coordinate frame via camera pose transforms before each attention layer, and then applies the inverse transform. Pose information is thereby embedded at every layer rather than only at the initial embedding.
- **Effect**: With PRoPE, SVSM recovers ideal scaling behavior, and its Pareto frontier remains superior to LVSM+PRoPE.

### 5. Fixed Latent Representation Scaling Experiments (Section 7)

- **Setup**: Objaverse dataset, $V_C = 8$; comparison between SVSM-fixed (fixed latent + unidirectional decoder) and LVSM enc-dec (fixed latent + bidirectional decoder).
- **Findings**: Both exhibit similar scaling behavior; SVSM-fixed retains a 5× compute advantage (Pareto frontier shifted 5× to the left). However, **both are substantially worse than the bottleneck-free design**—the fixed latent representation is the primary limiting factor for scaling.

## Key Experimental Results

### Table 1: Stereo NVS ($V_C = 2$) — Largest Models

| Model | Params | Training FLOPs | PSNR↑ | SSIM↑ | LPIPS↓ | FPS ($V_C=4$) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM Enc-Dec | 173M | 2.53 zflops | 28.58 | 0.893 | 0.114 | 52.9 |
| LVSM Dec-Only | 171M | 1.60 zflops | 29.67 | 0.906 | 0.098 | 19.5 |
| SVSM (Iter-matched) | 740M | 0.74 zflops | 29.80 | 0.907 | 0.098 | 42.7 |
| **SVSM (Pareto)** | **416M** | **0.77 zflops** | **30.01** | **0.910** | **0.096** | **61.8** |

### Table 2: Comparison with Explicit Geometry Methods (RealEstate10K)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| pixelNeRF | 20.43 | 0.589 | 0.550 |
| pixelSplat | 26.09 | 0.863 | 0.136 |
| MVSplat | 26.39 | 0.869 | 0.128 |
| GS-LRM | 28.10 | 0.892 | 0.114 |
| **SVSM** | **30.01** | **0.910** | **0.096** |

### Table 3: Multi-View NVS ($V_C > 2$)

| Model | Params | Training FLOPs | PSNR↑ | LPIPS↓ | FPS ($V_C=4$) | FPS ($V_C=16$) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LVSM+PRoPE | 171M | 43 eflops | 26.19 | 0.145 | 104.7 | 23.8 |
| SVSM (Iter) | 711M | 32 eflops | 26.29 | 0.141 | 280.4 | 230.4 |
| **SVSM (Pareto)** | **400M** | **44 eflops** | **26.87** | **0.129** | **411.1** | **333** |

## Key Findings

1. **3× compute efficiency**: SVSM's Pareto frontier has the same slope as LVSM's but is shifted 3× to the left—equal performance requires only one-third the training compute.
2. **Chinchilla law reproduced across modalities**: SVSM's $a \approx 0.52$, $b \approx 0.47$ ($a \approx b$) is consistent with NLP findings—doubling the compute budget should be split equally between model size and data.
3. **$B_\text{eff}$ governs performance**: The effective batch size $B \cdot V_T$ is the sole determinant of final performance; different $(B, V_T)$ splits with equal $B_\text{eff}$ differ by at most 0.2 PSNR.
4. **PRoPE unlocks multi-view scaling**: Without PRoPE, SVSM saturates rapidly at $V_C > 2$; with PRoPE, scaling resumes and the frontier remains superior to LVSM.
5. **Fixed latent representations are the scaling bottleneck**: Regardless of decoder directionality, a fixed-size scene representation severely limits scaling capacity.
6. **Inference speed**: SVSM renders at 4× the speed of LVSM at $V_C = 4$, extrapolating to 14× at $V_C = 16$.

## Highlights & Insights

**Highlights**:
- The effective batch size hypothesis is conceptually concise yet deeply insightful: it simultaneously explains the root cause of the encoder-decoder's underestimation and provides a principled approach to exploit its efficiency advantage.
- This work establishes the first Chinchilla-style compute-optimal training recipe in 3D vision.
- The experimental design is exceptionally rigorous, covering a $10^3\times$ FLOPs sweep, three datasets, and multiple $V_C$ settings.

**Limitations**:
- Training data is limited to small posed datasets such as RE10K and DL3DV with repeated sampling, diverging from the standard $<1$ epoch scaling practice.
- At large $V_C$, the quadratic encoder complexity reduces rendering speed below that of LVSM enc-dec (e.g., at $V_C = 8$).
- Coverage is restricted to sparse-to-moderate view settings; linear-attention models may hold advantages at $V_C \gg 16$.
- The study is limited to deterministic rendering; the applicability of these scaling laws to diffusion-based NVS models is not investigated.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The effective batch size hypothesis and NVS scaling laws fill a fundamental gap in 3D vision research.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic $10^3\times$ FLOPs analysis across stereo, multi-view, and fixed-latent settings.
- Writing Quality: ⭐⭐⭐⭐⭐ — Chinchilla-style rigorous presentation with professional, clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Compute-optimal training recipes and architectural design principles are directly transferable to other 3D vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation](freescale_scaling_3d_scenes.md)
- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
