---
title: >-
  [Paper Note] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper identifies co-adaptation among Gaussians as the root cause of appearance artifacts in sparse-view 3D Gaussian Splatting, proposes the Co-Adaptation Score (CA) metric to quantify this entanglement, and introduces two plug-and-play regularization strategies—Gaussian Dropout and multiplicative opacity noise injection—that consistently reduce co-adaptation and improve novel view rendering quality across five baseline met…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "sparse-view"
  - "co-adaptation"
  - "Dropout"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 290ce5a16532e755
---

# Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2508.12720](https://arxiv.org/abs/2508.12720)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, sparse-view, co-adaptation, Dropout, novel view synthesis

## TL;DR

This paper identifies co-adaptation among Gaussians as the root cause of appearance artifacts in sparse-view 3D Gaussian Splatting, proposes the Co-Adaptation Score (CA) metric to quantify this entanglement, and introduces two plug-and-play regularization strategies—Gaussian Dropout and multiplicative opacity noise injection—that consistently reduce co-adaptation and improve novel view rendering quality across five baseline methods and three datasets.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis capabilities in dense-view settings. Its core idea is to represent scenes with a set of 3D Gaussian ellipsoids, rendered via differentiable alpha compositing. Each pixel's color is computed by a weighted blend of multiple Gaussians projected onto that location—this multi-Gaussian compositing mechanism underlies 3DGS's ability to efficiently fit scene appearance. However, when training views are reduced from dense to sparse (e.g., only 3 views), 3DGS frequently suffers from severe quality degradation in novel view rendering, including not only geometric distortions but also prominent "appearance artifacts"—anomalous color blotches in rendered images that do not correspond to any real scene content.

**Limitations of Prior Work**: The dominant line of work on improving sparse-view 3DGS focuses almost exclusively on geometric regularization. Methods such as DNGaussian, FSGS, and CoherentGS introduce monocular depth priors to constrain multi-view depth consistency; Binocular3DGS leverages pretrained keypoint matchers to produce denser initialization point clouds; other methods exploit diffusion model priors to synthesize unseen views. However, these works primarily target geometric accuracy and rarely analyze the true origin of appearance artifacts in sparse-view settings. The anomalous colors that appear in novel views—purple blotches in flower scenes, pink noise in dinosaur scenes—are widespread yet consistently overlooked.

**Key Challenge**: The problem stems from a fundamental tension between the 3DGS optimization objective and its scene representation. The training objective minimizes the reconstruction loss $\mathcal{L}(R(\mathcal{G},v), I_v)$ between rendered images and ground-truth training views, which supervises only the final rendered output and imposes no explicit constraints on the internal parameters of individual Gaussians (position, shape, color, opacity). Under dense supervision, diverse multi-directional signals compel each Gaussian to encode faithful scene appearance. Under sparse supervision, however, multiple Gaussians with arbitrary colors can freely "cooperate" to fit a given pixel—as long as their weighted combination matches the training-view ground truth, their individual colors can deviate arbitrarily from the true scene appearance. This mirrors the co-adaptation problem in neural networks: multiple neurons become overly dependent on one another, producing individually meaningless outputs.

**Goal**: The paper decomposes the problem into three levels: (1) how to characterize and understand co-adaptation among Gaussians in sparse-view 3DGS; (2) how to quantitatively measure the severity of this co-adaptation; and (3) how to design lightweight strategies that effectively alleviate co-adaptation, thereby eliminating appearance artifacts and improving novel view rendering quality.

**Key Insight**: The authors draw an analogy from the classical concept of co-adaptation in neural networks. Hinton et al. identified as early as 2012 that neurons in neural networks form excessive interdependencies that lead to overfitting, and proposed Dropout to break such dependencies. The authors recognize that Gaussians in 3DGS share a structural parallel with neurons: just as a network output is jointly produced by multiple neurons, a pixel's color is jointly produced by multiple Gaussians. Taken to its logical conclusion, the Dropout idea can equally be applied to break excessive entanglement among Gaussians.

**Core Idea**: Transfer theoretical insights about co-adaptation from neural networks to 3DGS scene representations. By randomly dropping Gaussians (Dropout) or injecting opacity noise, each Gaussian is forced to independently encode correct scene appearance, thereby eliminating appearance artifacts under sparse-view settings.

## Method

### Overall Architecture

The proposed framework operates at two levels: **diagnosis** and **treatment**. At the diagnosis level, the Co-Adaptation Score (CA) is introduced—a metric that quantifies Gaussian entanglement by performing multiple random subset renderings of the same view and computing pixel-level variance. At the treatment level, two plug-and-play regularization strategies are proposed—Random Gaussian Dropout and Multiplicative Opacity Noise—which act directly on the Gaussian set or opacity parameters during training, requiring no modifications to the baseline method's network architecture or loss functions. The method takes a standard 3DGS training pipeline (Gaussian set + training views) as input and outputs a regularized scene representation for high-quality novel view rendering.

### Key Designs

1. **Co-Adaptation Score (CA)**

    - **Function**: Quantitatively measures the severity of co-adaptation in a set of optimized Gaussians at a given viewpoint, producing a scalar value where higher CA indicates stronger Gaussian entanglement.
    - **Mechanism**: The core intuition is that if Gaussians are overly interdependent, randomly removing a subset should cause large fluctuations in the rendered output; if they are independent, rendering should remain relatively stable. Concretely, $K$ random 50% Dropout renderings are performed on the full Gaussian set, producing $K$ rendered images $\{I^{(1)}, \ldots, I^{(K)}\}$ of the same target view. The visible region is defined as $\Omega_v = \bigcap_{k=1}^K \{u \mid \alpha_u^{(k)} > 0.8\}$ (pixels where accumulated alpha exceeds 0.8 across all renderings), and the CA score is the mean pixel-level variance over this region: $\mathrm{CA}(v) = \frac{1}{|\Omega_v|}\sum_{u \in \Omega_v}\mathrm{Var}(I_u^{(1)}, \ldots, I_u^{(K)})$. High variance implies that different subsets produce drastically different renderings, indicating that Gaussians engage in strong mutual dependence. The paper also provides a theoretical derivation in the appendix showing that CA directly reflects the coupling between Gaussian color and opacity attributes.
    - **Design Motivation**: Prior to proposing mitigation strategies, an objective quantification tool is needed to (1) validate the hypothesis that co-adaptation causes appearance artifacts, (2) evaluate the effectiveness of different mitigation strategies, and (3) understand how co-adaptation evolves with the number of training views and training iterations. Conventional rendering quality metrics (PSNR, SSIM, LPIPS) only indirectly reflect the problem and cannot directly measure the internal entanglement state of Gaussians. The CA metric fills this gap, transforming "co-adaptation" from a qualitative concept into a quantitatively trackable phenomenon.

2. **Random Gaussian Dropout**

    - **Function**: At each training iteration, randomly drops a fraction $p$ of Gaussians and renders only with the remaining subset; at test time, all Gaussians are used with a compensatory opacity rescaling.
    - **Mechanism**: Each Gaussian independently samples a Bernoulli variable $z_g \sim \text{Bernoulli}(1-p)$; only Gaussians with $z_g=1$ form the training subset $\mathcal{G}_\text{train} = \{g \in \mathcal{G} \mid z_g = 1\}$, which is used to render the image and compute the loss against the ground truth. At test time, no Dropout is applied, but all Gaussian opacities are scaled by $(1-p)$ to compensate for the training-time random dropping: $\alpha_g^{\text{test}} = (1-p) \cdot \alpha_g^{\text{train}}$. This directly mirrors the inference-time rescaling in standard neural network Dropout. The optimal Dropout probability is $p=0.2$, which effectively breaks entanglement without excessive information loss.
    - **Design Motivation**: Dropout forces each rendering ray to produce the correct color even when some Gaussians are absent. This means the model cannot rely on any fixed combination of Gaussians to fit a given pixel—neighboring Gaussians along the same ray must learn similar color and opacity characteristics so that they can substitute for one another. Additionally, since some Gaussians are randomly removed during training, the remaining ones tend to increase their volume to maintain consistent surface coverage, which helps reduce geometric inconsistencies and surface holes—particularly beneficial in sparse-view settings. Dropout thus not only breaks appearance entanglement but also indirectly improves geometric structure.

3. **Multiplicative Opacity Noise**

    - **Function**: Injects multiplicative Gaussian noise into each Gaussian's opacity parameter during training, slightly perturbing each Gaussian's contribution weight to the final pixel color.
    - **Mechanism**: At each training iteration, opacity is perturbed as $\text{opacity} \leftarrow \text{opacity} \cdot (1 + \epsilon)$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$. The optimal noise scale is $\sigma = 0.8$. This multiplicative noise causes each Gaussian's rendering contribution to vary slightly across iterations—sometimes weighted higher, sometimes lower—forcing the overall representation to be robust to small opacity perturbations of individual Gaussians. The authors systematically compare noise injection on different parameters: noise on 3D positions causes training instability and blurring; noise on SH coefficients has negligible effect since SH only affects color but not visibility or per-pixel Gaussian participation; noise on scales also reduces co-adaptation but introduces noticeable blurring. Only opacity noise achieves a "soft enough" perturbation that neither disrupts convergence nor introduces blur, while effectively destabilizing the dependency structure among co-adapted Gaussians.
    - **Design Motivation**: Opacity plays a critical role in the 3DGS rendering formula—it directly determines each Gaussian's blending weight in alpha compositing. Co-adapted Gaussians can produce the correct training pixel color precisely because their opacities and colors form a finely tuned combination. By injecting opacity noise, this precise coordination is disrupted at every iteration, forcing the model to learn a representation that is robust to small opacity variations. Compared to Dropout (which completely removes Gaussians), opacity noise provides a softer and more continuous regularization signal—effectively a "soft" version of Dropout.

### Loss & Training

Neither regularization strategy introduces additional loss terms; both operate on the rendering process itself rather than on the loss function.

- **Dropout** modifies the set of Gaussians participating in rendering prior to the forward pass; the existing reconstruction loss of the baseline method (typically L1 + SSIM) is used unchanged for backpropagation. At test time, opacity scaling by $(1-p)$ provides compensation. This zero-extra-cost property makes integration straightforward—no modification to the loss computation is required.
- **Opacity noise** perturbs opacity parameters during each forward rendering pass, also without modifying the loss function. The perturbation is applied only in the forward pass; gradients flow normally back through the perturbed opacities to the original parameters.
- The two strategies can be applied independently or jointly, but experiments show that combining them yields no additional benefit over using either alone—indicating that they address the same underlying problem (co-adaptation) rather than orthogonal issues. This "non-additive" result serves as important evidence supporting the core hypothesis.
- For test-time rendering, the paper compares three strategies: (A) single Dropout rendering at inference, (B) averaging 5 random Dropout renderings, and (C) rendering all Gaussians with opacity scaled by $(1-p)$. Strategy C achieves the best quality-efficiency trade-off—matching Strategy B in quality while being 5× faster, and clearly outperforming Strategy A.

## Key Experimental Results

### Main Results

The two strategies are validated on LLFF (3 views), DTU (3 views), and Blender (8 views) datasets across five baseline methods: 3DGS, DNGaussian, FSGS, CoR-GS, and Binocular3DGS.

**LLFF Dataset (3 training views)**

| Method | Setting | PSNR↑ | SSIM↑ | LPIPS↓ | Train CA↓ | Test CA↓ |
|------|------|-------|-------|--------|-----------|----------|
| 3DGS | baseline | 19.36 | 0.651 | 0.232 | 0.00754 | 0.00821 |
| 3DGS | w/ dropout | 20.20 | 0.691 | 0.211 | 0.00175 | 0.00234 |
| 3DGS | w/ opacity noise | 19.91 | 0.676 | 0.223 | 0.00153 | 0.00230 |
| DNGaussian | baseline | 18.93 | 0.599 | 0.295 | 0.00723 | 0.00765 |
| DNGaussian | w/ dropout | 19.43 | 0.623 | 0.302 | 0.00324 | 0.00382 |
| FSGS | baseline | 20.43 | 0.682 | 0.248 | 0.00458 | 0.00476 |
| FSGS | w/ dropout | 20.82 | 0.716 | 0.200 | 0.00193 | 0.00221 |
| CoR-GS | baseline | 20.17 | 0.703 | 0.202 | 0.00503 | 0.00516 |
| CoR-GS | w/ dropout | 20.64 | 0.712 | 0.217 | 0.00144 | 0.00162 |
| Binocular3DGS | baseline | 21.44 | 0.751 | 0.168 | 0.00185 | 0.00195 |
| Binocular3DGS | w/ dropout | 22.12 | 0.777 | 0.154 | 0.00088 | 0.00098 |
| Binocular3DGS | w/ opacity noise | 22.12 | 0.780 | 0.155 | 0.00066 | 0.00076 |

**DTU Dataset (3 training views)**

| Method | Setting | PSNR↑ | SSIM↑ | LPIPS↓ | Train CA↓ | Test CA↓ |
|------|------|-------|-------|--------|-----------|----------|
| 3DGS | baseline | 17.30 | 0.824 | 0.152 | 0.00210 | 0.00287 |
| 3DGS | w/ dropout | 17.75 | 0.850 | 0.135 | 0.00076 | 0.00226 |
| DNGaussian | baseline | 18.91 | 0.790 | 0.176 | 0.00511 | 0.00574 |
| DNGaussian | w/ dropout | 19.86 | 0.828 | 0.149 | 0.00120 | 0.00192 |
| Binocular3DGS | baseline | 20.71 | 0.862 | 0.111 | 0.00140 | 0.00158 |
| Binocular3DGS | w/ dropout | 21.03 | 0.875 | 0.108 | 0.00075 | 0.00115 |

### Ablation Study

**Effect of Dropout probability $p$ (Binocular3DGS, LLFF)**

| Dropout $p$ | PSNR↑ | SSIM↑ | LPIPS↓ | Train CA↓ | Test CA↓ |
|-------------|-------|-------|--------|-----------|----------|
| 0.0 (Baseline) | 21.440 | 0.751 | 0.168 | 0.001845 | 0.001951 |
| 0.1 | 21.901 | 0.768 | 0.157 | 0.000995 | 0.001066 |
| 0.2 | 22.123 | 0.777 | 0.154 | 0.000875 | 0.000978 |
| 0.3 | 22.037 | 0.777 | 0.156 | 0.000848 | 0.000951 |
| 0.4 | 22.025 | 0.775 | 0.158 | 0.000849 | 0.000926 |
| 0.5 | 21.927 | 0.773 | 0.163 | 0.000871 | 0.000982 |
| 0.6 | 21.793 | 0.768 | 0.170 | 0.000848 | 0.000978 |

**Effect of opacity noise scale $\sigma$ (Binocular3DGS, LLFF)**

| Noise scale $\sigma$ | PSNR↑ | SSIM↑ | LPIPS↓ | Train CA↓ | Test CA↓ |
|---------------------|-------|-------|--------|-----------|----------|
| 0.0 (Baseline) | 21.440 | 0.751 | 0.168 | 0.001845 | 0.001951 |
| 0.2 | 21.864 | 0.764 | 0.161 | 0.001126 | 0.001239 |
| 0.4 | 22.065 | 0.774 | 0.155 | 0.000859 | 0.000964 |
| 0.6 | 21.999 | 0.777 | 0.155 | 0.000794 | 0.000895 |
| 0.8 | 22.119 | 0.780 | 0.155 | 0.000660 | 0.000762 |
| 1.0 | 22.053 | 0.779 | 0.159 | 0.000560 | 0.000640 |

**Test-time rendering strategy comparison (Binocular3DGS w/ dropout $p=0.2$, LLFF)**

| Strategy | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| A: Single Dropout rendering at inference | 21.977 | 0.769 | 0.162 |
| B: Average of 5 Dropout renderings | 22.124 | 0.776 | 0.157 |
| C: All Gaussians + scaled opacity | 22.123 | 0.777 | 0.154 |

### Key Findings

- **Dropout generally outperforms opacity noise**: Across nearly all baseline methods and datasets, Dropout yields larger improvements in rendering quality metrics (PSNR, LPIPS) than opacity noise. For example, 3DGS on LLFF improves from baseline PSNR 19.36 to 20.20 (+0.84 dB) with Dropout, versus 19.91 (+0.55 dB) with noise. On stronger baselines such as Binocular3DGS, however, the two strategies perform comparably or opacity noise marginally outperforms on certain metrics.
- **The two strategies are non-additive**: Applying both Dropout and opacity noise simultaneously to Binocular3DGS yields PSNR of 22.11, essentially identical to using either Dropout (22.12) or noise (22.12) alone. This is because both strategies address the same underlying co-adaptation problem, and their combination provides no orthogonal benefit.
- **Non-monotonic relationship between CA and rendering quality**: Lower CA does not necessarily imply better rendering quality. In the Dropout ablation, $p=0.2$ achieves the best PSNR (22.123) with Test CA of 0.000978, while $p=0.6$ yields lower Test CA (0.000978) but worse PSNR (21.793). Similarly, opacity noise at $\sigma=1.0$ achieves the lowest CA (0.000640) but lower PSNR (22.053) than $\sigma=0.8$ (22.119). This indicates that over-suppressing co-adaptation can harm the model's representational capacity.
- **Three empirical findings are consistent**: (1) Increasing the number of training views naturally reduces CA—the most central observation, confirming that co-adaptation is intrinsically a product of sparse supervision; (2) CA decreases rapidly in the early training phase before stabilizing; (3) CA on training views is consistently lower than on novel views, as direct supervision makes co-adaptation easier to suppress on training views.
- **Binocular3DGS exhibits an anomalous CA spike after 20K iterations**: This occurs because the method introduces a warp-based photometric loss after 20K steps; this view-warping supervision may introduce geometric mismatches that inadvertently reinforce undesirable dependencies among Gaussians. Both Dropout and opacity noise effectively suppress this spike.

## Highlights & Insights

- **Transferring neural network theory to 3D scene representations**: The most elegant contribution of this work is recognizing the structural analogy between Gaussians in 3DGS and neurons in neural networks—a pixel's color is jointly produced by multiple Gaussians, just as a network output is jointly computed by multiple neurons. This analogy is not merely a qualitative metaphor; it leads directly to the quantifiable CA metric and to actionable Dropout and noise strategies, forming a highly coherent logical chain.
- **Elegant and intuitive metric design**: The CA score—"randomly drop half the Gaussians and measure how unstable the rendering becomes"—is highly intuitive, simple to implement, and physically interpretable. It is further supported by theoretical analysis (the appendix proves that CA directly reflects color–opacity coupling), making it more than an empirical heuristic. This metric design paradigm can be transferred to any representation system based on multi-element compositing—e.g., sample points in NeRF, surface patches in mesh rendering, and so on.
- **Theoretical value of the non-additivity finding**: The observation that combining the two strategies yields no extra benefit in turn confirms that they both address the same problem (co-adaptation) rather than coincidentally achieving positive effects along different dimensions. This "negative result" actually strengthens the paper's central argument.
- **Systematic comparison of noise on different parameters**: The authors do not directly select opacity noise but first systematically compare the effects of injecting noise into positions, SH coefficients, scales, and opacities, ruling out alternatives before selecting opacity. This rigorous ablation provides valuable insights into the functional roles of different 3DGS parameters.

## Limitations & Future Work

- **Lack of adaptive mechanisms**: The Dropout probability $p$ and noise scale $\sigma$ are fixed hyperparameters requiring manual tuning. Ideally, an adaptive mechanism should dynamically adjust regularization strength based on the current CA score—applying stronger regularization early in training when co-adaptation is severe, and relaxing it as training converges.
- **Computational cost of the CA metric**: Although CA is not used during training, its computation requires multiple random Dropout renderings, which can be expensive for large-scale scenes. If CA is to serve as a signal for adaptive regularization, more efficient approximation methods will be necessary.
- **Insufficient analysis of performance variation across scene types**: The Blender dataset exhibits the anomalous phenomenon where training-view CA exceeds novel-view CA, which the authors attribute only briefly to "circular object structure and 8-view coverage" without deeper analysis. The manifestation patterns of co-adaptation likely vary considerably across scene types (indoor/outdoor, textured/untextured, object-level/scene-level).
- **Scale inflation from Dropout is a side effect**: The authors note that Dropout encourages remaining Gaussians to expand in size to maintain surface coverage, which can reduce geometric holes but may also cause over-smoothing. The paper does not quantitatively analyze the degree of this scale inflation or its potential negative impact on rendering detail.
- **No comparison with feed-forward methods**: The paper validates its strategies only on per-scene optimization methods, without exploring whether the co-adaptation concept also applies to feed-forward 3DGS methods (e.g., pixelSplat, MVSplat, DepthSplat), whose co-adaptation patterns may differ fundamentally as they predict Gaussian parameters directly via pretrained networks. The paper also does not extend its analysis to geometric artifacts (e.g., floaters), focusing exclusively on appearance artifacts.
- **Limited theoretical depth**: Although the appendix provides a theoretical derivation relating CA to color–opacity coupling, formal theoretical guarantees explaining why Dropout and noise effectively reduce co-adaptation remain absent. Why are $p=0.2$ or $\sigma=0.8$ optimal? Do theoretically optimal values depend on scene complexity or Gaussian count? These questions warrant further investigation.

## Related Work & Insights

- **vs. DNGaussian / FSGS / CoR-GS (geometric regularization approaches)**: These methods primarily improve geometric accuracy through monocular depth priors and photometric consistency constraints, addressing appearance artifacts only indirectly. This paper directly intervenes at the root cause of appearance artifacts (co-adaptation); the two paradigms are orthogonal and complementary—the proposed Dropout/noise strategies can be stacked on top of geometric regularization methods for further gains. Experiments show that even on the already strong Binocular3DGS baseline, Dropout still improves PSNR from 21.44 to 22.12.
- **vs. concurrent Dropout-based 3DGS works**: Recent concurrent works also apply Dropout in sparse-view 3DGS, but attribute the improvements to "reducing the number of active Gaussians to mitigate overfitting" or "enhancing gradient flow to distant Gaussians." This paper's contribution lies in identifying and formalizing "co-adaptation suppression" as the core mechanism, providing a more principled explanation.
- **vs. classical Dropout / neural network regularization**: This work essentially transfers the Dropout idea proposed by Hinton et al. in 2012 from connection weight space to 3D spatial element space. Analogous ideas may apply to other differentiable rendering-based 3D representations—for instance, Dropout on sample points in NeRF, or random dropping of points in point-based rendering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core concept of co-adaptation in 3DGS is a long-overlooked but genuinely important perspective; the CA metric is novel and theoretically grounded; however, the specific strategies (Dropout, noise injection) are not technically original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across five baseline methods × three datasets, detailed ablations over Dropout probability, noise scale, and inference strategies, training dynamics analysis, and visual comparisons are all highly thorough.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain—from phenomenon observation to metric definition to strategy design to experimental validation—is exceptionally clear; Figure 2's co-adaptation visualization is intuitive and accessible.
- **Value**: ⭐⭐⭐⭐ The co-adaptation perspective and CA metric make a conceptual contribution to understanding and improving sparse-view 3DGS; the two plug-and-play strategies are highly practical with minimal integration overhead; long-term impact depends on whether the community develops more advanced mitigation mechanisms around this concept.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](../../ECCV2024/3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](../../CVPR2025/3d_vision/dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
