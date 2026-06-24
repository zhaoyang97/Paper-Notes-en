---
title: >-
  [Paper Note] TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens
description: >-
  [CVPR 2026][3D Vision][Feed-forward 3DGS reconstruction] TokenGS transforms feed-forward 3D Gaussian Splatting reconstruction from "regressing depth along a ray for each pixel" to "directly regressing 3D coordinates using a set of learnable Gaussian tokens." This completely decouples the number of Gaussians from input resolution and the number of views, achieving SOTA performance on both static and dynamic scenes with cleaner geometry, robustness to pose noise…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Feed-forward 3DGS reconstruction"
  - "learnable Gaussian tokens"
  - "encoder-decoder"
  - "direct coordinate regression"
  - "test-time optimization"
date: 2026-05-08
content_hash: 1deab8bf88568e77
---

# TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Ren_TokenGS_Decoupling_3D_Gaussian_Prediction_from_Pixels_with_Learnable_Tokens_CVPR_2026_paper.html)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Feed-forward 3DGS reconstruction, learnable Gaussian tokens, encoder-decoder, direct coordinate regression, test-time optimization

## TL;DR
TokenGS transforms feed-forward 3D Gaussian Splatting reconstruction from "regressing depth along a ray for each pixel" to "directly regressing 3D coordinates using a set of learnable Gaussian tokens." This completely decouples the number of Gaussians from input resolution and the number of views, achieving SOTA performance on both static and dynamic scenes with cleaner geometry, robustness to pose noise, and support for low-cost test-time token fine-tuning.

## Background & Motivation
**Background**: Feed-forward neural reconstruction has progressed rapidly. The dominant paradigm uses a large encoder-only Transformer (e.g., GS-LRM, MVSplat, DepthSplat) to directly predict "pixel-aligned" 3DGS primitives from multiple posed images—meaning a depth is predicted for each pixel (or patch) along the camera ray, which is then back-projected into a Gaussian center $\mu$.

**Limitations of Prior Work**: This pixel-aligned paradigm has three fundamental flaws. First, by treating Gaussian means as depths along rays, the model struggles to internally correct noisy poses and multi-view inconsistencies; it is particularly awkward for dynamic scenes where points/pixels need to be warped over time. Second, the number of Gaussians is strictly tied to resolution $\times$ view count—32 images at 512×512 results in over 8 million Gaussians. Repeating the same input $N$ times yields $N$ times redundant Gaussians, meaning representation scale follows the number of views rather than scene complexity. Third, while feed-forward reconstruction could benefit from self-supervised test-time refinement, directly optimizing Gaussian parameters destroys the learned priors of the network, especially in sparse-view settings.

**Key Challenge**: The root cause is the hard coupling between "Gaussian primitives $\leftrightarrow$ input pixels"—positions are tied to camera rays and counts are tied to pixel grids. This coupling limits geometric expressiveness (unable to fill occluded areas) and creates redundancy and brittleness.

**Goal**: Decouple this binding—ensuring Gaussian positions are no longer restricted to rays and Gaussian counts are no longer equal to the number of pixels, while maintaining stable training and supporting dynamic scenes and test-time optimization.

**Key Insight**: The authors draw inspiration from the point-map approach in feed-forward SfM models (like VGGT)—directly regressing 3D coordinates instead of depth. The key difference is the **absence of any explicit point-map or ground-truth depth supervision**, relying solely on self-supervised rendering loss. Once the model predicts free 3D coordinates instead of ray depths, the encoder-only architecture can be replaced with a DETR-style encoder-decoder, allowing a set of learnable tokens to cross-attend to image features.

**Core Idea**: Replace "per-pixel ray depth regression" with "a set of learnable Gaussian tokens directly regressing 3D coordinates," making the number of predicted Gaussians a hyperparameter independent of the input.

## Method

### Overall Architecture
The input consists of $N$ posed images with extrinsic parameters $\{T_i \in SE(3)\}$ and intrinsic parameters. The output is a set of $M$ 3DGS primitives $G \in \mathbb{R}^{M\times 14}$ (each containing mean $\mu\in\mathbb{R}^3$, color $c$, scale $s$, opacity $\sigma$, and quaternion rotation $q$), which can be synthesized into new views via volume rendering. The pipeline follows three steps: the encoder encodes multi-view images and camera parameters into image tokens; the decoder allows a set of learnable 3DGS tokens to "extract" 14-dimensional attributes (including directly regressed 3D coordinates) from image tokens via cross-attention; after training, token embeddings can optionally be fine-tuned at test time to further improve quality. Direct coordinate regression introduces a "zero-gradient" problem, which the authors solve using a visibility loss.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Multi-view posed images<br/>+ Intrinsics/Extrinsics"] --> B["ViT Encoder<br/>Image patch ⊕ Plücker encoding<br/>→ Image tokens"]
    B --> C["Direct 3D coordinate regression<br/>+ Visibility loss<br/>Solves zero-gradient problem"]
    C --> D["Learnable Gaussian token decoder<br/>Tokens cross-attend to image tokens<br/>→ 64 Gaussians per token"]
    D -->|Dynamic Scenes| E["Static/Dynamic tokens<br/>Bullet-time + Causal attention mask"]
    D --> F["Volume rendering supervised training"]
    E --> F
    F -->|Test-time (Optional)| G["Test-time scaling<br/>Context extension + Token tuning"]
    G --> H["Output: 3DGS Representation<br/>Novel view synthesis / Scene flow"]
```

### Key Designs

**1. Direct Coordinate Regression + Visibility Loss: Decoupling Gaussian positions from camera rays**

Pixel-aligned methods predict Gaussian centers as depth along rays, meaning each Gaussian must fall on a specific camera ray. This fails to reconstruct occlusions or geometry outside the frustum, and noisy poses lead to distorted predictions. TokenGS **directly regresses 3D coordinates for $\mu$** in a global coordinate system defined by camera extrinsics and shared across all views, trained only with self-supervised rendering loss (without ground-truth point-maps, which are hard to obtain and would re-introduce the ray-coupling the authors aim to avoid). XYZ coordinates are mapped to the real domain via the activation $f(x)=\mathrm{sign}(x)\cdot(\exp(x)-1)$. This change provides three benefits: extrapolation/completion of geometry beyond input views, robustness to pose noise, and elimination of "spiky" artifacts common in depth prediction networks.

However, direct coordinate regression introduces a non-trivial training challenge—the **zero-gradient problem**: Gaussians falling outside all camera frustums do not participate in rendering and thus receive no gradient from the rendering loss. These "dead" Gaussians degrade training stability, waste model capacity, and manifest as floaters. Instead of using explicit 3D supervision like VGGT, the authors introduce a **visibility loss**. Each Gaussian center $\mu_m=(x_m,y_m,z_m)$ is projected onto the image planes of all supervised views, obtaining pixel coordinates normalized by width and height:

$$\tilde{u}_m^i = 2(u_m^i/W)-1, \qquad \tilde{v}_m^i = 2(v_m^i/H)-1,$$

Points inside the image satisfy $\tilde{u},\tilde{v}\in[-1,1]$. The loss measures the distance of each Gaussian to the "nearest visible boundary":

$$\mathcal{L}_{\text{vis}} = \sum_m \min_{\mathbf{I}_i\in\mathcal{I}_{sup}}\left[\mathrm{ReLU}(|\tilde{u}_m^i|-1)+\mathrm{ReLU}(|\tilde{v}_m^i|-1)\right].$$

If a Gaussian projects into at least one view, this term is 0; otherwise, it provides a gradient to pull it back into the field of view (truncated at a constant 1.0 in practice to avoid spurious gradients). The total loss is $\mathcal{L}=\mathcal{L}_{\text{MSE}}+\lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}}+\lambda_{\text{vis}}\mathcal{L}_{\text{vis}}$. Ablations show this improves PSNR by 0.4 and removes floating artifacts.

**2. Learnable Gaussian Token Encoder-Decoder: Decoupling Gaussian count from image resolution**

Simply changing coordinate parameterization is insufficient if the model remains encoder-only and predicts Gaussians per-pixel. The authors adopt a DETR-style encoder-decoder. The encoder is a ViT: each view is patchified and linearly projected to $C$ dimensions, while Plücker coordinates for each view are also patchified and projected. These are summed $a_{vk}=x_{vk}+s_{vk}$ and flattened into a token sequence for standard ViT layers (allowing cross-view attention) to produce image tokens $B$. The decoder initializes $N_t$ learnable token embeddings (3DGS tokens). Each decoder block performs three steps: (i) cross-attention to image tokens $B$, (ii) self-attention between GS tokens, and (iii) per-token MLP. Each output embedding regresses $N_G=64$ Gaussians via linear layers (14 attributes: color/opacity via scaled tanh, scale via truncated exp, rotation via unit normalization).

Crucially, **the decoding of each Gaussian token is independent of the number of input pixels**, making the total number of predicted Gaussians $N_t\times N_G$ a hyperparameter independent of resolution or view count. This allows the model to freely allocate Gaussians to areas of high scene complexity. Experiments show tokens spontaneously allocate more Gaussians to high-frequency detail regions, and Gaussians produced by a single token tend to cluster in the scene or fall in similar 3D positions across different scenes, exhibiting slot-like specialization. Since image tokens usually far outnumber GS tokens ($N_I\gg N_t$), the authors **share key-value projections for image tokens** across all decoder cross-attention layers (computed once at the start), reducing memory overhead from $O(N_I D_{dec})$ to $O(N_I)$, allowing for deeper decoders (24 layers). Additionally, LayerScale and QK-normalization are used for stability, and LayerNorm before the regression head is intentionally omitted as it was found to decrease performance.

**3. Static/Dynamic Tokens + Bullet-time + Causal Attention Mask: Decomposing dynamic scenes into structure and motion**

Pixel-aligned representations cannot move Gaussian centers over time (e.g., BTimer can only simulate dynamics by changing opacity to make objects "flicker"). In contrast, TokenGS predicts continuous 3D coordinates, naturally expressing motion. The authors split the token set into static $T^S$ (time-invariant) and dynamic $T^D$ groups. Each dynamic token is augmented with a bullet-time temporal embedding:

$$\tilde{T}^D_j(t) = T^D_j + \mathrm{Linear}( \tau(t) ),$$

where $\tau(t)$ is a sine-coded temporal value. A **structured causal mask** is applied in the decoder self-attention: dynamic tokens can attend to static tokens (dynamic $\rightarrow$ static), static tokens attend to each other, and dynamic tokens attend to each other, but static tokens do not attend to dynamic ones. This inductive bias represents the idea that "the dynamic parts of a scene depend on its static structure," allowing the model to automatically decompose the scene into time-invariant structure and time-varying motion while maintaining consistent correspondences for dynamic tokens across frames—enabling not just intermediate frame reconstruction but also emergent trackable scene flow.

**4. Dual-axis Test-time Scaling: Context extension + Token tuning without retraining**

The encoder-decoder design naturally supports two complementary forms of test-time scaling without retraining. **Context Extension (CE)**: Feeding more image tokens at inference time than during training (e.g., training on 2 views, testing on 4)—in encoder-only paradigms, this would increase the number of predicted Gaussians, but here the count remains fixed while the context grows, improving quality (experiments show benefits up to 4x input). **Token Tuning (TT)**: A lightweight test-time training where **only** the Gaussian token embeddings are fine-tuned. Network parameters and image features remain frozen. Using the self-supervised rendering loss on input views for several dozen steps allows attention patterns to adapt to the specific scene, improving Gaussian parameters while preserving the global priors learned by the network. The authors compared this with a baseline of "direct Gaussian parameter optimization": the latter yields higher PSNR and faster convergence in well-posed multi-view scenarios but destroys geometry (failing on novel views), whereas token tuning improves geometry—reflecting the stronger priors encoded within the tokens. Both scaling methods can be combined.

## Key Experimental Results

### Main Results

On RealEstate10K (2-view, 256×256), TokenGS matches or exceeds pixel-aligned SOTA with fewer Gaussians:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | #GS |
|------|-------|-------|--------|-----|
| MVSplat | 26.39 | 0.869 | 0.128 | 131K |
| DepthSplat | 27.47 | 0.889 | 0.114 | 131K |
| GS-LRM | 28.10 | 0.892 | 0.114 | 131K |
| Ours (1024 tok) | 28.02 | 0.896 | 0.147 | **66K** |
| Ours (4096 tok) | 28.41 | 0.903 | 0.135 | 262K |
| Ours (4096 tok, +TT) | **28.82** | **0.910** | 0.130 | 262K |

The base model with 50% Gaussians reaches the same PSNR tier as GS-LRM; the tuned version with 2x Gaussians clearly wins; TT further improves quality and avoids overfitting even with only 2 training views.

Cross-context length generalization on DL3DV (model trained on 4 views, testing 2/6 views with context extension, denoted as Ours*):

| Method | #Views | PSNR↑ | LPIPS↓ | Time(s) | #GS |
|------|--------|-------|--------|---------|-----|
| DepthSplat | 6 | 24.19 | 0.147 | 0.132 | 688K |
| Ours* | 6 | 23.69 | 0.311 | 0.085 | 262K |
| Ours* (+TT) | 6 | **24.51** | 0.295 | 5.61 | **262K** |

With 6 views, Ours+TT outperforms the baseline with 72% fewer Gaussians and faster feed-forward inference (0.085s vs 0.132s). On view extrapolation, TokenGS outperforms baselines without any point-map/depth labels and matches versions fine-tuned with PM-Loss supervision. On the dynamic Kubric 4D dataset (4 views), it achieves 24.84 PSNR, surpassing BTimer's 24.45.

### Ablation Study

| Configuration / Analysis | Key Metric | Description |
|------|---------|------|
| w/o Visibility Loss | PSNR 28.4 | Floating artifacts and geometric degradation appear |
| w/ Visibility Loss | PSNR 28.8 | +0.4 PSNR, removes floaters |
| Token Tuning (TT) | Geometry improvement | Preserves priors, sharper novel views |
| Direct Gaussian Param Opt | Higher near-view PSNR but broken geometry | Over-optimization destroys geometry |
| Token Count 256→4096 | PSNR increases monotonically | Two-dimensional adjustability decoupled from views |

### Key Findings
- **Visibility loss is critical for direct coordinate regression**: It specifically addresses the zero-gradient problem that leads to floaters. Removing it drops PSNR by 0.4 and induces degraded geometry.
- **Token Tuning vs. Gaussian Parameter Optimization reveals the value of "priors"**: While direct parameter optimization yields better quantitative metrics (near-view PSNR), it destroys geometry. Tuning only tokens improves geometry, proving that token space encodes stronger global priors.
- **Gaussian count and view count are independently adjustable**: Increasing token count from 256 to 4096 monotonically improves quality regardless of input view count, allowing a trade-off between compression efficiency and fidelity.
- **Spontaneous token specialization**: Individual tokens cluster Gaussians within a scene and consistently assign them to similar 3D locations across different scenes; high-frequency regions naturally receive more tokens/Gaussians.

## Highlights & Insights
- **Applying DETR's "object query" concept to 3DGS**: Using learnable tokens as "Gaussian queries" successfully decouples Gaussian quantity from resolution/views—a clean, transferable structural insight that shifts from "dense per-pixel prediction" to "fixed-capacity set prediction."
- **Visibility loss as the "glue" for self-supervised coordinate regression**: Without relying on 3D ground truth, a simple soft constraint that pulls Gaussians into the field of view bypasses the ray-coupling inherent in point-map supervision.
- **"Prior-preserving optimization in token space" is transferable**: The paradigm of freezing the backbone and tuning a small set of token embeddings for test-time adaptation can be applied to other feed-forward prediction tasks.
- **Elegant dynamic modeling via causal masking**: Using a unidirectional dynamic $\rightarrow$ static attention writes the "motion depends on structure" bias into the architecture, naturally resulting in scene flow and static-dynamic decomposition.

## Limitations & Future Work
- **Acknowledged Limitations**: Inherits common feed-forward weaknesses such as difficulty with large-scale environments and fine geometric details. Test-time token tuning is relatively expensive, requiring dozens of optimization steps (TT increases inference from ~0.08s to 4–5.6s).
- **Metric Duality**: TokenGS's LPIPS is generally worse than baselines (e.g., 0.314–0.326 on DL3DV vs DepthSplat's 0.178). High PSNR/SSIM coupled with weaker perceptual metrics suggests a gap in rendering sharpness that the paper does not fully explore.
- **Direct Parameter Optimization is quantitatively superior**: The authors note that in well-posed multi-view scenes, direct Gaussian tuning is faster and higher in PSNR. TT's value is "preserving geometry" rather than "highest score." Hybrid strategies are left for future work.
- **Future Directions**: Exploring adaptive token allocation (adjusting token count based on scene complexity), reducing TT iterations, and joint token/Gaussian optimization to balance quantitative metrics with geometric quality.

## Related Work & Insights
- **vs. GS-LRM / Pixel-aligned Methods**: These methods regress depth per-pixel along rays, tying Gaussian count to pixels. TokenGS regresses 3D coordinates directly and uses tokens to decouple quantity, resulting in cleaner geometry (no spikes), robustness to pose noise, and better extrapolation, albeit with weaker LPIPS and slower TT.
- **vs. VGGT / Feed-forward SfM**: Both regress 3D coordinates/point-maps, but VGGT uses explicit 3D/depth supervision. TokenGS uses only self-supervised rendering loss + visibility loss, avoiding the difficulty of obtaining ground truth and the ray-coupling issues.
- **vs. BTimer (Dynamic)**: Pixel-aligned BTimer cannot move Gaussian centers, relying on opacity changes that cause flickering. TokenGS uses continuous 3D coordinates + bullet-time tokens for continuous, trackable motion and higher PSNR.
- **vs. PM-Loss Fine-tuning**: In view extrapolation, TokenGS matches or beats baselines fine-tuned with point-map supervision without requiring any point-map data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decoupling both Gaussian positions and counts from pixels using DETR-style tokens + direct coordinate regression + visibility loss is a coherent and rare combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers static/dynamic/extrapolation/pose noise/test-time scaling with solid ablations, though the LPIPS disadvantage could be more deeply discussed.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-to-design mapping with lucid diagrams.
- Value: ⭐⭐⭐⭐⭐ Provides a new "pixel-decoupled" paradigm for feed-forward 3DGS. The two-dimensional adjustability and prior-preserving optimization are highly attractive for deployment.

## Related Papers

- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners](featurising_pixels_from_dynamic_3d_scenes_with_linear_in-context_learners.md)
- [\[CVPR 2026\] Registration-Free Learnable Multi-View Capture of Faces in Dense Semantic Correspondence](registration-free_learnable_multi-view_capture_of_faces_in_dense_semantic_corres.md)
- [\[CVPR 2026\] ORD: Object-Relation Decoupling for Generalized 3D Visual Grounding](ord_object-relation_decoupling_for_generalized_3d_visual_grounding.md)
- [\[CVPR 2026\] PaNDaS: Learnable Shape Interpolation Modeling with Localized Control](pandas_learnable_shape_interpolation_modeling_with_localized_control.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners](featurising_pixels_from_dynamic_3d_scenes_with_linear_in-context_learners.md)
- [\[CVPR 2026\] Registration-Free Learnable Multi-View Capture of Faces in Dense Semantic Correspondence](registration-free_learnable_multi-view_capture_of_faces_in_dense_semantic_corres.md)
- [\[CVPR 2026\] ORD: Object-Relation Decoupling for Generalized 3D Visual Grounding](ord_object-relation_decoupling_for_generalized_3d_visual_grounding.md)
- [\[CVPR 2026\] SparseSplat: Towards Applicable Feed-Forward 3D Gaussian Splatting with Pixel-Unaligned Prediction](sparsesplat_towards_applicable_feed-forward_3d_gaussian_splatting_with_pixel-una.md)

</div>

<!-- RELATED:END -->
