---
title: >-
  [Paper Note] Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere
description: >-
  [CVPR 2026][3D Vision][Novel View Synthesis] Addressing the limitation of Spherical Harmonics (SH) in representing high-frequency specular reflections for "view-dependent appearance" in radiance fields, this paper proposes **Spherical Voronoi (SV)**. By using a set of learnable sites to softly partition the sphere into regions, SV serves as an explicit spherical function representation that is easier to optimize than SH or Spherical Gaussians (SG) while sharply modeling glint…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Novel View Synthesis"
  - "3D Gaussian Splatting"
  - "Spherical Representation"
  - "Specular Reflection"
  - "Differentiable Voronoi"
date: 2026-05-08
content_hash: d779e228b262f18f
---

# Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Di_Sario_Spherical_Voronoi_Directional_Appearance_as_a_Differentiable_Partition_of_the_CVPR_2026_paper.html)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, 3D Gaussian Splatting, Spherical Representation, Specular Reflection, Differentiable Voronoi

## TL;DR
Addressing the limitation of Spherical Harmonics (SH) in representing high-frequency specular reflections for "view-dependent appearance" in radiance fields, this paper proposes **Spherical Voronoi (SV)**. By using a set of learnable sites to softly partition the sphere into regions, SV serves as an explicit spherical function representation that is easier to optimize than SH or Spherical Gaussians (SG) while sharply modeling glint-level highlights. It is further extended as "learnable lighting probes" for spatially-varying reflections, achieving SOTA results on reflection benchmarks like Ref-NeRF and GlossySynthetic (Ref-NeRF PSNR 36.09).

## Background & Motivation
**Background**: Radiance field methods, represented by 3D Gaussian Splatting (3DGS), currently dominate novel view synthesis. Each primitive uses low-order SH (typically degree-3) to store view-dependent colors. SH is popular due to its orthogonality, global support, smooth loss landscape, and ease of optimization.

**Limitations of Prior Work**: SH is a band-limited representation. To accurately capture sharp, high-frequency signals (such as a glint on a smooth surface), the required coefficients grow quadratically with frequency, leading to parameter explosion. Furthermore, Gibbs ringing artifacts occur at function discontinuities (Figure 2 in the paper shows that reconstructing a highlight requires ~30 coefficients yet still exhibits ringing). Consequently, 3DGS struggles to reproduce complex view-dependent appearances of glossy surfaces—a primary reason current benchmark PSNRs have plateaued.

**Key Challenge**: Existing explicit alternatives face a trade-off between "expressiveness" and "optimizability." Spherical Gaussians (SG) and Spherical Beta (SB) can express sharp signals with local support. However, because their kernels are compactly supported, they suffer from weak gradients when lobes are misaligned, extreme sensitivity to initialization, and ill-conditioned gradients at high concentrations, making them prone to local minima. In short: SH is easy to optimize but lacks expressiveness; SG/SB are expressive but difficult to optimize.

**Goal**: To identify an explicit spherical representation that is **easy to optimize, capable of expressing high frequencies, and extensible to spatially-varying reflections**, unifying the modeling of diffuse and specular appearances.

**Key Insight**: The authors draw inspiration from differentiable Voronoi diagrams. Rather than using a set of "competing" and overlapping kernels to fit a spherical function, it is more effective to **partition** the sphere into non-overlapping regions. Partitioning naturally ensures coverage of the entire domain, where every direction is explicitly assigned to a (soft) region, avoiding weight competition or coverage gaps common in SG/SB.

**Core Idea**: Replace spherical harmonic bases with a "differentiable soft Spherical Voronoi partition," dividing the directional domain into learnable regions with smooth boundaries. This SV representation is queried along the reflection direction and implemented as spatially distributed light probes, incorporating reflections into a unified explicit differentiable framework.

## Method

### Overall Architecture
The paper defines SV as a general explicit spherical function representation, applied to radiance fields in two ways: (i) directly replacing view-direction parametrization (diffuse/general view-dependent appearance), and (ii) serving as light probes for reflection modeling.

The core of SV consists of a set of **directional sites** $s_1,\dots,s_K \in \mathbb{S}^2$ on the sphere and their corresponding **function values** $c_1,\dots,c_K$. For a query direction $\omega$, the function value is a weighted combination of site values: $f_{SV}(\omega) = \sum_{k=1}^{K} w_k(\omega;\lambda_k)\, c_k$, where weights are determined by a softmax. When integrated into 3DGS, each Gaussian carries additional parameters $\{\lambda_k\},\{s_k\},\{c_k\}$ which are optimized jointly to output RGB view-dependent colors.

Reflection modeling follows a deferred rendering pipeline: mapping the 2DGS scene into geometry buffers (position/normal/roughness/diffuse color), then synthesizing specular components using "far-field cubemaps + near-field Voronoi light probes" in a lighting pass, finally adding the diffuse component. The reflection pipeline is as follows:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["视角方向 ω"] --> B["球面软Voronoi划分<br/>softmax 加权站点"]
    B -->|直接查询 f(ω)| C["视角相关 diffuse 外观"]
    B -->|沿反射方向 f(ωr)| D["可学习光照探针<br/>kNN 探针 + 逆距离插值"]
    E["2DGS 几何pass<br/>位置/法线/粗糙度/diffuse"] --> D
    E --> F["远场 cubemap"]
    D --> G["近场镜面 Cn"]
    F --> H["远场镜面 Cf"]
    G --> I["β 混合近/远场<br/>粗糙度调温度 λ"]
    H --> I
    C --> J["最终颜色 C = D + Cspec"]
    I --> J
```

### Key Designs

**1. Soft Spherical Voronoi Partition: Slicing the Sphere into Non-overlapping Learnable Regions via Softmax**

This step addresses the conflict between SH’s limited expressiveness and the optimization difficulties of SG/SB. SV defines the function value at direction $\omega$ as a weighted sum:

$$f_{SV}(\omega;\lambda,s,c) = \sum_{k=1}^{K} w_k(\omega;\lambda_k)\, c_k, \qquad w_k(\omega;\lambda) = \frac{\exp(\lambda_k\, s_k\cdot\omega)}{\sum_{k'=1}^{K}\exp(\lambda_{k'}\, s_{k'}\cdot\omega)}.$$

The weights are calculated using the softmax of the dot product between the site direction and query direction $s_k\cdot\omega$. The temperature $\lambda_k>0$ controls the sharpness: low $\lambda$ leads to smooth boundaries and soft transitions, while high $\lambda$ approaches a hard Voronoi tessellation capable of expressing sharp discontinuities (Figure 6 illustrates transitions for $\lambda=\{1,5,25\}$). Standard soft SV occurs when all sites share $\lambda$; assigning unique $\lambda_k$ to each site yields a "weighted" variant with locally adaptive angular sharpness.

The effectiveness lies in two properties of softmax: **all sites have well-defined gradients** (unlike compactly supported kernels which may have zero-gradient dead zones), and softmax normalization naturally creates a **clean, non-overlapping spherical decomposition**. The weights sum to 1 for every direction, preventing kernels from "fighting" over weights or leaving gaps. 2D fitting experiments (Figure 4, 100 random initializations) demonstrate that SV consistently converges to better reconstructions than SH/SG/SB with lower variance.

**2. View-direction Parametrization: Implicit Temperature Encoding via Site Norms, 8 Sites to Match Degree-3 SH Degrees of Freedom**

In standard radiance fields (without explicit reflection modeling), the view-direction representation is replaced by SV, where $f_{SV}(\omega):\mathbb{S}^2\to\mathbb{R}^3$ outputs RGB. For a fair comparison with degree-3 SH using **equal parameter counts**, the authors utilize **8 sites per Gaussian**: each site stores a 3D position and 3D radiance, totaling 48 learnable parameters (exactly matching degree-3 SH).

A clever design is that temperature is not stored separately: the direction of the site vector $s_k$ is represented by the unit vector $\hat s_k = s_k/\|s_k\|$, while its **norm $\|s_k\|$ is used as the temperature** $\lambda_k = \|s_k\|$. Thus, temperature is optimized along with the site vector without extra parameters. The authors also note that while the original Beta-Splatting backbone selected the best checkpoint every 500 iterations (effectively using the test set for early stopping), this work adopts a cleaner protocol of training for a fixed number of iterations without test set supervision.

**3. Learnable Lighting Probes: Querying SV Along Reflection Directions to Model Spatially-Varying Reflections**

Querying only along the reflection direction $f(\omega_r)$ (where $\omega_r = 2(\omega\cdot n)n - \omega$, following Ref-NeRF's approach) implicitly assumes far-field lighting. When glossy objects are near other objects or light sources, appearance depends on both direction and position. Far-field assumptions fail here, resulting in blurred reconstructions (Figure 3). Neural fields can concatenate reflection direction and position into an MLP, but this is difficult for explicit Gaussian representations.

The solution is **spatially distributed learnable lighting probes**: a set of probes is placed in the scene, each parameterized by a position $p_i$, a blend weight $\beta_i\in[0,1]$, and an SV function $(\lambda_i,s_i,c_i)$. For a surface point $P$, the system identifies its $k$-nearest probes $\mathcal{N}=\text{kNN}(P)$ and interpolates them using normalized inverse distance weights $\tilde w_i = \|P-p_i\|^{-1}/\sum_{j\in\mathcal{N}}\|P-p_j\|^{-1}$ to obtain near-field color and the blending factor:

$$C_n = \sum_{i\in\mathcal{N}} \tilde w_i\, f^{i}_{SV}(\omega_r), \qquad \beta = \sum_{i\in\mathcal{N}} \tilde w_i\, \beta_i.$$

Each probe explicitly encodes a local reflection field as an SV function queried along the reflection direction, making it naturally suited for specular appearance modeling in 3DGS—a key distinction from prior implicit or non-reflection-directional probe work.

**4. Deferred Rendering and Roughness-Modulated Temperature: Controlling SV Sharpness via Roughness, Unifying Near/Far Field Specular**

The reflection pipeline is built on a 2DGS backbone (offering more accurate normal estimation). Each primitive learns two material parameters: roughness $r\in[0,1]$ and diffuse color $d\in\mathbb{R}^3$. The geometry pass rasterizes all 2D Gaussians to obtain per-pixel position $P$, normal $N$, roughness $R$, and diffuse color $D$. The final color is $C = D + C_{spec}$, where the specular term is a spatially varying blend of near and far fields:

$$C_{spec} = \beta\, C_n + (1-\beta)\, C_f,$$

The far-field $C_f$ is evaluated at $\omega_r$ using a learnable cubemap. A critical coupling is the **modulation of temperature by roughness**:

$$\lambda = (1-R)\,\lambda_{max} + R\,\lambda_{min},$$

Low roughness corresponds to high $\lambda$ (sharper reflection lobes), while high roughness broadens the lobe. Notably, $\lambda$ is not directly learned here as it is in view-direction parametrization; instead, it is physically derived from the surface roughness $R$. This allows SV's angular sharpness to automatically adapt to material changes, creating a unified, differentiable, and fully explicit diffuse+specular appearance model.

## Key Experimental Results

### Main Results: View-direction Parametrization (diffuse / general view-dependency)
Based on the Beta-Splatting backbone with 8 sites per Gaussian (48 parameters, aligned with degree-3 SH). SV consistently outperforms SH/SG/SB color parametrizations across all datasets and even exceeds the strong neural baseline Zip-NeRF (PSNR, higher is better):

| Dataset | SH | SG | SB | SV (Ours) | Zip-NeRF |
|--------|-----|-----|-----|----------|----------|
| Mip-NeRF360 | 28.09 | 28.18 | 28.12 | **28.57** | 28.55 |
| NeRF-Synthetic | 34.15 | 34.26 | 34.10 | **34.53** | 33.67 |
| DeepBlending | 29.80 | 29.67 | 29.56 | **30.48** | - |
| Tanks&Temples | 24.50 | 24.71 | 24.54 | **24.75** | 23.64 |

### Main Results: Reflection Parametrization (Voronoi Light Probes)
Based on the 2DGS backbone, using 128 probes for synthetic scenes and 1024 probes + 2048 sites for real scenes; cubemap resolution is $256\times256\times6\times3$. SV achieves SOTA on Ref-NeRF and GlossySynthetic, and is competitive on Ref-Real (PSNR):

| Dataset | 3DGS | 3DGS-DR | Ref-GS | Ours |
|--------|------|---------|--------|------|
| Ref-NeRF | 30.37 | 34.13 | 35.57 | **36.09** |
| GlossySynthetic | 26.50 | 30.36 | 31.27 | **31.30** |
| Ref-Real | 23.85 | 23.83 | 23.81 | **23.91** |

Notably, Ours slightly outperforms Ref-GS—which uses the same 2DGS backbone but relies on an MLP decoder for near-field reflections—while remaining **fully explicit**.

### Ablation Study: Probe Parametrization Method (Ref-NeRF, fixed parameter count)
Replacing the spherical representation in light probes while keeping parameters constant shows SV leading by a significant margin:

| Probe Representation | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|--------|--------|---------|
| SG | 34.19 | 0.967 | 0.060 |
| SB | 34.07 | 0.966 | 0.061 |
| Cubemap | 34.48 | 0.970 | 0.056 |
| **SV** | **36.09** | **0.976** | **0.050** |

### Key Findings
- **Voronoi partitioning is most expressive for equivalent parameter counts**: Under the same capacity, SV outperforms SG/SB/Cubemap (by over +1.6 PSNR in probe ablations), suggesting that "spatial partitioning" is a more compact and efficient local appearance basis than "stacking local kernels."
- **Optimization robustness is a structural advantage**: 100 random initialization repetitions (Figure 4) show that SV's convergence distribution is concentrated and yields higher final values, whereas SG/SB easily fall into local minima due to compactly supported kernels.
- **Explicit models can outperform MLPs**: On reflection benchmarks, the fully explicit SV slightly beats Ref-GS (which uses an MLP decoder) and surpasses Zip-NeRF in view parametrization, indicating that the right representation can break through current PSNR bottlenecks.

## Highlights & Insights
- **Conceptual shift from "Stacking" to "Partitioning"**: Reconceptualizing spherical functions as "soft partitions of non-overlapping regions" rather than a "sum of competing kernels" is a significant shift. Softmax normalization ensures global coverage and stable gradients for all sites, bypassing the initialization sensitivity and dead gradients of SG/SB.
- **Multi-purpose Temperature Parameter**: Temperature is implicitly given by the site norm $\|s_k\|$ in view parametrization (zero extra parameters) and physically derived from roughness $R$ in reflections ($\lambda=(1-R)\lambda_{max}+R\lambda_{min}$). A single sharpness knob is "learned" in one scenario and "physically constrained" in the other, resulting in a unified design.
- **Transferable logic**: Differentiable soft Voronoi partitioning with softmax temperature is essentially an "adaptive, mutually exclusive, full-coverage" directional domain decomposition. This logic can be transferred to any task requiring fitting high-frequency, discontinuous functions in the spherical/directional domain (e.g., BRDF fitting, environment lighting encoding, point cloud directional features).

## Limitations & Future Work
- The authors acknowledge that the far-field is still approximated by a cubemap, and near-field probes rely on kNN inverse distance interpolation. Probe density (128 synthetic / 1024 real) and placement strategies may affect the accuracy of spatially-varying reflections; adaptive probe placement was not explored.
- ⚠️ The reflection pipeline depends on the accuracy of 2DGS normal estimation. Inaccurate normals will bias the reflection direction $\omega_r$, subsequently affecting SV queries; this coupled error was not isolated in the ablation studies.
- Performance on Ref-Real (real reflections) is only comparable to or slightly better than the baseline, indicating that near-field/inter-reflections in real-world scenes remain an open challenge. The memory/speed overhead from the number of sites (up to 2048 for reflections) as scene complexity grows is also a consideration.

## Related Work & Insights
- **vs Spherical Harmonics (SH)**: SH is globally supported and easy to optimize but band-limited, requiring many coefficients for high-frequency/discontinuous data and causing Gibbs ringing. SV uses locally adaptive soft partitions for sharp high-frequency expression with higher quality at equal parameter counts.
- **vs Spherical Gaussians/Beta (SG/SB)**: While SG/SB express sharp local signals, their compact support leads to initial sensitivity and ill-conditioned gradients. SV utilizes softmax for non-overlapping partitions, providing stable gradients for all sites and more robust optimization.
- **vs Ref-NeRF**: Both query radiance along the reflection direction. Ref-NeRF uses an MLP conditioned on direction and position for spatially-varying reflections, whereas this work replaces the MLP with explicit learnable SV light probes and kNN interpolation to maintain a fully explicit differentiable framework.
- **vs Ref-GS**: Also built on the 2DGS backbone, but Ref-GS models near-field reflections via an MLP decoder. This work is fully explicit (Voronoi light probes), outperforming Ref-GS on reflection benchmarks while being more interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces differentiable Voronoi partitioning as a fundamental replacement for the SH/SG/SB paradigm in spherical appearance representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 datasets and includes 2D fitting robustness and probe ablations, although systematic ablations on normal error and probe density are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation of background motivation; formulas and visualizations (Figure 4/6) strongly support the claims.
- Value: ⭐⭐⭐⭐⭐ Achieves SOTA on reflection benchmarks while remaining fully explicit, providing a reusable new basis for high-frequency appearance modeling in 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] MeshSplatting: Differentiable Rendering with Opaque Meshes](meshsplatting_differentiable_rendering_with_opaque_meshes.md)
- [\[CVPR 2026\] Learning Differentiable Hierarchies in 3D Gaussian Splatting](learning_differentiable_hierarchies_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting](intrinsic_geometry-appearance_consistency_optimization_for_sparse-view_gaussian_.md)
- [\[CVPR 2026\] D-Prism: Differentiable Primitives for Structured Dynamic Modeling](d-prism_differentiable_primitives_for_structured_dynamic_modeling.md)

</div>

<!-- RELATED:END -->
