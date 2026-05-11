---
title: >-
  [Paper Note] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity
description: >-
  [ICCV 2025][Neural surface reconstruction] This paper proposes HiNeuS, a unified neural surface reconstruction framework that simultaneously addresses three core challenges—reflective ambiguity, low-texture degradation…
tags:
  - "ICCV 2025"
  - "Neural surface reconstruction"
  - "SDF"
  - "reflection handling"
  - "low-texture regions"
  - "Eikonal constraint"
date: 2026-05-08
content_hash: 74630da5023f1e5d
---

# HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity

**Conference**: ICCV 2025
**arXiv**: [2506.23854](https://arxiv.org/abs/2506.23854)
**Code**: Coming soon
**Area**: Other
**Keywords**: Neural surface reconstruction, SDF, reflection handling, low-texture regions, Eikonal constraint

## TL;DR

This paper proposes HiNeuS, a unified neural surface reconstruction framework that simultaneously addresses three core challenges—reflective ambiguity, low-texture degradation, and detail preservation—through three innovations: SDF-guided visibility verification, planar conformal regularization, and rendering-prioritized Eikonal relaxation.

## Background & Motivation

Neural surface reconstruction is a fundamental technique in 3D computer vision. SDF-based methods (e.g., NeuS, VolSDF) surpass purely density-field approaches (e.g., NeRF) in surface recovery accuracy by combining volumetric rendering with geometric priors. However, existing methods still perform poorly in three critical scenarios, and these problems are typically addressed **in isolation**:

**Multi-view inconsistency (reflective ambiguity)**: Strong reflections and indirect illumination violate the multi-view photometric consistency assumption. Specular reflections cause significant color variation for the same surface point across viewpoints, leading to geometric artifacts. Existing methods such as Ref-NeuS handle specularities but introduce noise in low-texture regions.

**Low-texture surface degradation**: Regions lacking texture (e.g., white walls, road surfaces) provide sparse visual cues, leading to over-regularization that erodes valid geometric structures. Surface regularization methods improve planar regions but over-smooth fine details.

**Detail–geometry conflict**: The conventional Eikonal constraint $\|\nabla f(\mathbf{x})\|_2 = 1$ enforces uniform smoothness, which conflicts with the preservation of high-frequency geometric details (thin structures, sharp edges). Methods such as Neuralangelo recover fine structures but are not robust to view-dependent effects.

The root cause of these three problems lies in the need for a **dynamic balance between appearance constraints (photometric loss) and geometric constraints (Eikonal)**. Existing methods either address only one aspect or sidestep the conflict through staged optimization. The central goal of HiNeuS is to realize a unified framework in which appearance and geometry constraints **co-evolve** throughout training.

## Method

### Overall Architecture

HiNeuS builds upon the SDF volumetric rendering framework (analogous to NeuS/VolSDF), employing an SDF network, a color network, and a proposal network. On top of the standard rendering loss $\mathcal{L}_{rgb}$, three innovation modules are introduced to form a unified loss: $\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{planar} + \mathcal{L}_{eikonal}$. The three modules are bidirectionally coupled through shared SDF values, rendering errors, and appearance features.

### Key Designs

1. **SDF-guided multi-view consistency verification**: A visibility factor $\mathbb{V}_j$ is computed via continuous SDF evaluation to resolve reflective ambiguity. For a surface point $\mathbf{x}_0^i$ observed from viewpoint $j$, $K$ points are sampled along the ray and occlusion is determined by SDF values:

$$\mathbb{V}_j = \prod_{k=1}^{K} \sigma(\beta f(\mathbf{x}_k^{(j)}))$$

When $f(\mathbf{x}) > 0$ (free space), $\sigma(\beta f) \approx 1$; when $f(\mathbf{x}) < 0$ (occupied), $\sigma(\beta f) \approx 0$. The product over the entire ray approaches 1 only when all points lie in free space. An **ambiguity factor** $\lambda_{ambiguity}$ is then defined as the weighted average of Mahalanobis color distances among visible viewpoints, used to downweight the loss on highly ambiguous rays. A **self-reflection compensation** term is also introduced: for rays with indirect reflections, the color is modeled as $\mathbf{C}' = (1-\mathbb{S})\mathbf{C} + \mathbb{S}g(\mathbf{x}_r)$, where $\mathbb{S}$ is the reflection probability and $g$ is a reflection MLP. **Design motivation**: The continuity of the SDF avoids mesh discretization artifacts and enables detection of occlusions behind thin structures.

2. **Local geometry constraint regularization (planar conformal)**: Local planarity is enforced in low-texture regions. Neighboring points $\mathbf{x}_k$ are sampled along the ray near surface point $\mathbf{x}_0$, and their SDF values are constrained to satisfy a local linear relationship:

$$\mathcal{L}_{planar} = \frac{1}{K}\sum_{k=1}^{K} \lambda_{pla}^k \left|\frac{f(\mathbf{x}_k)}{\|\mathbf{x}_k - \mathbf{x}_0\|} - \mathbf{n}_0^\top \frac{\mathbf{x}_k - \mathbf{x}_0}{\|\mathbf{x}_k - \mathbf{x}_0\|}\right|$$

The key lies in the adaptive weight $\lambda_{pla}^k = \frac{\epsilon}{\|\mathbf{c}_{feat}(\mathbf{x}_k) - \mathbf{c}_{feat}(\mathbf{x}_0)\|_2 + \epsilon}$: **regions with large texture variation receive low weights** (preserving sharp edges), while **texturally uniform regions receive high weights** (enforcing planarity). This automatically distinguishes locations requiring smoothing from those requiring detail preservation through appearance features.

3. **Rendering-prioritized Eikonal relaxation**: The strength of the Eikonal constraint is dynamically adjusted according to the principle that "geometric constraints should be relaxed where rendering error is large":

$$\omega(\mathbf{x}) = \lambda_{pla}(\mathbf{x}) \cdot \exp(-\gamma\|\mathbf{C}(\mathbf{x}) - \hat{\mathbf{C}}(\mathbf{x})\|_2)$$

$$\mathcal{L}_{eikonal} = \frac{1}{|\mathcal{S}|}\sum_{\mathbf{x}\in\mathcal{S}} \omega(\mathbf{x})(\|\nabla f(\mathbf{x})\|_2 - 1)^2$$

Three core properties: (i) large rendering error → small $\omega$ → relaxed Eikonal constraint → geometric adjustment permitted to improve rendering; (ii) high planarity → large $\lambda_{pla}$ → regularization is maintained in low-texture regions; (iii) exponential decay provides gradual convergence. By using rendering error as "learned attention," this design automatically resolves the conflict between geometric accuracy and detail preservation.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{planar} + \mathcal{L}_{eikonal}$
- SDF network: 8-layer MLP (256 channels) with geometric initialization; color network: 4-layer MLP (128 channels)
- Progressive hash encoding: resolution from $32^3$ to $2048^3$
- Adam optimizer, learning rate decayed from $10^{-2}$ to $10^{-5}$, 500k iterations total
- $\beta = 100$, $\gamma = 5.0$, rendering error clipped to $\leq 0.2$, $\lambda_{pla}$ annealed from 0.1 to 1.0 over 100k steps

## Key Experimental Results

### Main Results

**NeRF-Synthetic (PSNR dB ↑)**

| Method | Chair | Lego | Mic | Ship | Avg. |
|--------|-------|------|-----|------|------|
| NeuS | 27.95 | 29.85 | 29.89 | 25.46 | 27.44 |
| 3DGS | 35.83 | 35.78 | 35.36 | 30.80 | 32.68 |
| **HiNeuS** | **37.29** | **36.11** | **38.17** | **34.39** | **35.00** |

Gain of 2.32 dB over 3DGS and 7.56 dB over NeuS.

**GlossySynthetic (Chamfer Distance mm ↓)**

| Method | Angel | Horse | Avg. |
|--------|-------|-------|------|
| Ref-NeuS | 0.0041 | 0.0062 | 0.0048 |
| NeRO | 0.0034 | 0.0049 | 0.0042 |
| **HiNeuS** | **0.0032** | **0.0045** | **0.0038** |

21.4% reduction compared to Ref-NeuS.

### Ablation Study

**Mip-NeRF 360 dataset (contribution of each module)**

| Configuration | Avg PSNR ↑ | Impact |
|---------------|-----------|--------|
| Full method | 29.50 | Baseline |
| w/o ambiguity factor $\lambda_{ambiguity}$ | 28.08 | −1.42 dB, severe degradation in reflective scenes |
| Mesh visibility (replacing SDF) | 28.72 | −0.78 dB, occlusion artifacts on thin structures |
| w/o planar constraint $\mathcal{L}_{planar}$ | 29.05 | −0.45 dB, instability in low-texture regions |
| Eikonal w/o adaptive weight $\omega$ | 28.20 | −1.30 dB, excessive smoothing of details |

### Key Findings

- A **nonlinear synergistic effect** exists among the three modules: removing the ambiguity factor alone (−1.42 dB) has a substantially larger impact than removing the planar constraint (−0.45 dB)
- SDF-based continuous visibility verification outperforms mesh discretization on thin structures (TBell: 0.0033 vs. 0.0036 mm)
- On UrbanScene3D real-world urban scenes, HiNeuS correctly ignores dynamic objects such as moving vehicles while recovering fine structures such as street lamp poles
- The reflection compensation term $\mathbb{S}g(\mathbf{x}_r)$ contributes 37% of the total improvement on GlossySynthetic
- The method generalizes to inverse rendering tasks (material decomposition, view-consistent relighting)

## Highlights & Insights

- The design philosophy of **"letting rendering error guide geometric constraints"** is particularly insightful—using an observable quantity (rendering quality) to automatically regulate the strength of an unobservable one (geometric regularization)
- The continuity of the SDF is fully exploited: visibility, reflection probability, and adaptive weights are all computed from SDF values, forming a unified mathematical framework
- The adaptive weights of the planar constraint cleverly leverage appearance feature differences to distinguish texture boundaries from low-texture regions
- Although the three modules appear independent, they form a co-optimized system through bidirectional coupling via the SDF and rendering error

## Limitations & Future Work

- The 500k-iteration training time is substantial, and efficiency lags behind recent 3D Gaussian-based methods
- The capacity of the reflection MLP is limited and may be insufficient for complex multi-bounce illumination
- A detailed efficiency comparison with high-performance NeuS variants such as NeuS2 and Instant-NeuS is absent
- The $t_{max} = 0.1$ setting for self-reflection compensation is heuristic and may require scene-adaptive tuning
- In extreme low-texture scenarios (e.g., entirely white walls), the planarity assumption may be insufficient, necessitating the incorporation of depth priors

## Related Work & Insights

- **NeuS/VolSDF** provide the foundation—the SDF-to-density mapping and volumetric rendering framework
- **Ref-NeuS** is the primary comparison for reflection handling—parametric BRDF approaches offer limited flexibility
- **Neuralangelo** represents the state of the art in high-fidelity reconstruction—but does not address reflections
- Insight: The adaptive relaxation strategy can be generalized to other 3D reconstruction tasks involving conflicting constraints

## Rating

- **Novelty**: ⭐⭐⭐⭐ Each of the three modules contributes original ideas; the synergistic design of the unified framework is particularly notable
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers synthetic and real data, indoor and outdoor scenes, multiple benchmarks, and detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and physical intuitions are clearly explained
- **Value**: ⭐⭐⭐⭐ Provides a systematic solution to practical pain points in neural surface reconstruction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](../../AAAI2026/others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[ICCV 2025\] FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases](fixtalk_taming_identity_leakage_for_high-quality_talking_head_generation_in_extr.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](loss_functions_for_predictor-based_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
