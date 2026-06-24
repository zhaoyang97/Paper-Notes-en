---
title: >-
  [Paper Note] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation
description: >-
  [CVPR 2025][3D Vision][3D Scene Generation] MIDI extends the pre-trained image-to-3D single-object generation model into a multi-instance diffusion model. Through a novel multi-instance attention mechanism, it directly captures spatial interaction relationships between objects during the 3D generation process. It simultaneously generates multiple 3D instances with correct spatial layouts from a single image, significantly outperforming existing methods on both synthetic and r…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Scene Generation"
  - "Multi-Instance Diffusion"
  - "Single Image 3D Reconstruction"
  - "Spatial Relations"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 9a591f0fb53ece56
---

# MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.03558](https://arxiv.org/abs/2412.03558)  
**Code**: [https://github.com/huanngzh/MIDI-Page](https://github.com/huanngzh/MIDI-Page) (Project Page)  
**Area**: 3D Vision / 3D Scene Generation  
**Keywords**: 3D Scene Generation, Multi-Instance Diffusion, Single Image 3D Reconstruction, Spatial Relations, Attention Mechanism

## TL;DR
MIDI extends the pre-trained image-to-3D single-object generation model into a multi-instance diffusion model. Through a novel multi-instance attention mechanism, it directly captures spatial interaction relationships between objects during the 3D generation process. It simultaneously generates multiple 3D instances with correct spatial layouts from a single image, significantly outperforming existing methods on both synthetic and real datasets.

## Background & Motivation

1. **Background**: Generating 3D scenes from a single image is a core challenge in computer vision. Existing methods fall into three categories: (a) feed-forward reconstruction methods (e.g., Total3D, InstPIFu) that directly regress 3D geometry using neural networks; (b) retrieval-based methods (e.g., DiffCAD) that retrieve and align 3D models from a database; (c) compositional generation methods (e.g., REPARO, Gen3DSR) that segment the scene $\rightarrow$ inpaint object images $\rightarrow$ generate 3D objects individually $\rightarrow$ optimize spatial relationships.

2. **Limitations of Prior Work**: Feed-forward methods are limited by the scarcity of scene-level training data, leading to poor generalization. Retrieval methods depend heavily on database coverage and cannot achieve precise matches. Compositional generation pipelines are lengthy (segmentation $\rightarrow$ inpainting $\rightarrow$ generation $\rightarrow$ optimization), causing error accumulation, and the lack of global scene context during individual object generation leads to discordant spatial relationships.

3. **Key Challenge**: Pre-trained image-to-3D single-object generation models offer high quality, but generating objects independently cannot model spatial relationships among them. Subsequent spatial relationship optimization cannot retroactively improve the already generated 3D objects.

4. **Goal**: How to model spatial relationships among multiple objects in an end-to-end manner while generating high-quality 3D objects.

5. **Key Insight**: Instead of using the 3D generation model as a "tool" in a multi-stage pipeline, the multi-instance interaction mechanism is directly introduced inside the generation model. This allows multiple objects to be generated simultaneously during the denoising process while coordinating spatial relationships.

6. **Core Idea**: Extend the self-attention in pre-trained 3D object diffusion models to multi-instance attention. This enables latent tokens of multiple instances to query each other during denoising, thereby capturing chess-board/spatial relationships directly during the generation process.

## Method

### Overall Architecture
MIDI is built upon a pre-trained 3D object generation model (VAE + DiT). Given a single scene image, it segments the images and masks of $N$ objects. Denoising is performed simultaneously on the 3D latent representations of the $N$ objects—sharing DiT weights and noise levels, but with independent noise for each. The key extension is the integration of multi-instance attention layers into the DiT to model cross-instance interactions, along with injecting local object images and global scene context via cross-attention.

### Key Designs

1. **Multi-Instance Attention Mechanism (Multi-Instance Attention)**:

    - **Function**: Directly model spatial relationships among multiple instances during the denoising process of 3D generation.
    - **Mechanism**: Extend a subset of object self-attention layers ($K=5$ layers) in the DiT to multi-instance attention. Previously, tokens of each instance could only query themselves; now, tokens of each instance can query those of all instances in the scene: $f^i_{out} = \text{Attention}(f^i, \{f^j\}_{j=1}^N)$. This allows the attention mechanism to consider the joint set of tokens from all instances, capturing spatial dependencies among objects.
    - **Design Motivation**: When $K=0$ (no multi-instance attention), the model fails to capture correct spatial relationships. When $K=21$ (all layers converted), the model overfits and disrupts the pre-trained prior, leading to distorted object geometry. $K=5$ achieves the best balance between modeling interactions and retaining the pre-trained prior.

2. **Global-Local Image Conditional Encoding**:

    - **Function**: Provide local object information and global scene context for each instance.
    - **Mechanism**: For each instance $z^i$, its RGB image $c_l^i$, mask $m_l^i$, and global scene image $c_g$ are concatenated along the channel dimension to form a 7-channel (3+1+3) composite representation $y \in \mathbb{R}^{h \times w \times 7}$. This is encoded using a DINO ViT with an extended input channel size and injected into the DiT via cross-attention.
    - **Design Motivation**: Local object images provide appearance info, while the global scene image provides spatial layout and context. Removing the global scene image leads to incorrect object placement and missing spatial relations (validated by ablation studies).

3. **Mixed Training Strategy (Mixed Training)**:

    - **Function**: Fine-tune on limited scene data while maintaining the generalization ability of the pre-trained model.
    - **Mechanism**: During training, with a 30% probability, the multi-instance diffusion model degenerates into a standard single-object generation model by disabling multi-instance attention and training on Objaverse single-object data. This acts as a regularization, preventing the model from overfitting to the smaller scene dataset. Parameter-efficient fine-tuning is performed using LoRA.
    - **Design Motivation**: The 3D-Front scene dataset contains only about 15,000 scenes, which is far smaller than the large-scale single-object data used for pre-training. Without regularization, the object geometric quality degrades.

### Loss & Training
- Employs a rectified flow architecture, where all instances share the noise level $t$ and are perturbed along linear trajectories.
- Loss function: $\mathbb{E}\left[\sum_{i=1}^N \|z_0^i - \epsilon^i - \epsilon_\theta(z_t^i, t, \tau_\theta(y))\|_2^2\right]$
- Uses LoRA for efficient fine-tuning of the pre-trained model; initializes the image encoder with DINO and extends the input channels to 7.

## Key Experimental Results

### Main Results

**3D-Front (Synthetic Data)**:

| Method | CD-S↓ | F-Score-S↑ | CD-O↓ | F-Score-O↑ | IoU-B↑ | Runtime |
|------|-------|------------|-------|------------|--------|---------|
| InstPIFu | 0.138 | 39.99 | 0.165 | 38.11 | 0.299 | 32s |
| DiffCAD | 0.117 | 43.58 | 0.190 | 37.45 | 0.392 | 64s |
| REPARO | 0.129 | 41.68 | 0.160 | 40.85 | 0.339 | 4min |
| **MIDI** | **0.080** | **50.19** | **0.103** | **53.58** | **0.518** | 40s |

**BlendSwap (Synthetic Data)**:

| Method | CD-O↓ | F-Score-O↑ | IoU-B↑ |
|------|-------|------------|--------|
| REPARO | 0.151 | 42.84 | 0.410 |
| Gen3DSR | 0.148 | 40.76 | 0.449 |
| **MIDI** | **0.090** | **62.94** | **0.663** |

Object-level F-Score improved by 12+ points, spatial layout IoU improved by 15+ points.

### Ablation Study

| #K | Scene | Objaverse | CD-S↓ | F-Score-S↑ | CD-O↓ | F-Score-O↑ | IoU-B↑ |
|----|-------|-----------|-------|------------|-------|------------|--------|
| 0 | ✓ | ✓ | 0.145 | 40.94 | 0.096 | 54.16 | 0.327 |
| **5** | **✓** | **✓** | **0.080** | **50.19** | **0.103** | **53.58** | **0.518** |
| 21 | ✓ | ✓ | 0.127 | 44.88 | 0.141 | 48.55 | 0.423 |
| 5 | ✗ | ✓ | 0.134 | 41.49 | 0.102 | 52.91 | 0.459 |
| 5 | ✓ | ✗ | 0.137 | 42.00 | 0.126 | 51.62 | 0.502 |

### Key Findings
- **Multi-instance attention is key**: Without multi-instance attention ($K=0$), IoU-B is only 0.327, which jumps to 0.518 when $K=5$, demonstrating the critical role of spatial relationship modeling.
- **Number of attention layers requires balance**: $K=5$ is optimal, while $K=21$ (all layers converted) actually degrades performance. Too many attention layers lead to overfitting on the small-scale scene data, disrupting the pre-trained 3D prior.
- **Global scene image is indispensable**: Without the global scene image, CD-S increases from 0.080 to 0.134, and the spatial layout severely degrades.
- **Mixed training acts as regularization**: Training without mixing Objaverse data degrades object geometric quality (CD-O increases from 0.103 to 0.126).
- **Significant running efficiency**: MIDI takes 40s vs. REPARO's 4min vs. Gen3DSR's 9min, which is 6-13 times faster.

## Highlights & Insights
- **Paradigm shift from "tool use" to "capability internalization"**: Previous compositional generation methods treat the 3D generation model as a tool in a multi-stage pipeline, whereas MIDI models spatial relationships directly inside the model. This mindshift—internalizing processes typically reserved for post-processing/optimization into the generation process itself—presents an important design paradigm.
- **Partial layer conversion instead of full conversion**: Converting only 5 out of 21 layers of self-attention to multi-instance attention models interaction while successfully preserving the pre-trained prior. This "minimally invasive" fine-tuning strategy is worth noting—fewer modifications to pre-trained models are preferred.
- **Mixed training prevents forgetting**: Falling back to single-object mode training with a 30% probability functions similarly to "replay" strategies in multi-task learning, effectively preventing the degradation of object generation capacity caused by scene-level fine-tuning.

## Limitations & Future Work
- **Does not handle backgrounds**: Planar backgrounds like floors and walls are not within the generation scope and require auxiliary methods.
- **Limited number of objects**: Current experiments primarily handle a small number of objects (2-5). The scalability to scenes with a massive number of objects remains unverified.
- **Poor performance on low-resolution inputs**: The authors acknowledge suboptimal performance when processing small-sized object image inputs.
- **Complex interactions are not modeled**: Interactions such as human-object interactions (e.g., "panda playing guitar") require dedicated datasets and designs.
- **Dependency on scene segmentation**: The input image requires segmented object masks, and the segmentation quality directly impacts the generation results.
- **Future directions**: Extending to open-world scene generation; introducing explicit 3D geometric prior to enhance attention efficiency; exploring the implicit 3D perception capabilities of the model.

## Related Work & Insights
- **vs REPARO [Lu et al.]**: REPARO adopts a typical multi-stage compositional generation pipeline (segmentation $\rightarrow$ inpainting $\rightarrow$ individual object generation $\rightarrow$ optimization). MIDI bypasses the object image inpainting and post-optimization stages, achieving multi-instance coordination directly within the diffusion model to avoid error accumulation. MIDI outperforms REPARO by a wide margin across all metrics (IoU-B: 0.518 vs. 0.339).
- **vs Gen3DSR [Li et al.]**: Similar to REPARO, Gen3DSR utilizes a multi-stage pipeline and takes 9 minutes to process a single scene. MIDI requires only 40 seconds and achieves superior results.
- **vs DiffCAD [Choi et al.]**: DiffCAD is a retrieval-based method utilizing diffusion models to assist CAD retrieval and alignment. It is limited by database coverage and cannot generate object shapes that are absent from the database. MIDI possesses generative capabilities, freeing it from database constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm of multi-instance diffusion is indeed novel, and internalizing spatial relationships between objects into the generation process serves as an elegant solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three types of data (synthetic, real, and stylized), with a comprehensive and persuasive ablation study.
- Writing Quality: ⭐⭐⭐⭐ The logic is precise, and the comparison diagram with existing methods (Fig. 2) is highly intuitive.
- Value: ⭐⭐⭐⭐⭐ Explores a new paradigm for 3D scene generation, carrying strong guidance for subsequent works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)

</div>

<!-- RELATED:END -->
