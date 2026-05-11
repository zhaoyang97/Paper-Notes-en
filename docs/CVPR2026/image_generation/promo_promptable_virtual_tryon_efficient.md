---
title: >-
  [Paper Note] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On
description: >-
  [CVPR 2026][Image Generation][Virtual try-on] PROMO is built on a FLUX.1-dev Flow Matching DiT backbone and achieves high-fidelity, efficient multi-garment virtual try-on without a traditional reference network…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Virtual try-on"
  - "Flow Matching DiT"
  - "multi-condition generation"
  - "temporal self-reference"
  - "outfit style control"
date: 2026-05-08
content_hash: 7c68d21f5990a4f5
---

# PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On

**Conference**: CVPR 2026
**arXiv**: [2603.11675](https://arxiv.org/abs/2603.11675)
**Code**: Unavailable (Xiaohongshu team)
**Area**: Image Generation / Virtual Try-On
**Keywords**: Virtual try-on, Flow Matching DiT, multi-condition generation, temporal self-reference, outfit style control

## TL;DR
PROMO is built on a FLUX.1-dev Flow Matching DiT backbone and achieves high-fidelity, efficient multi-garment virtual try-on without a traditional reference network, by combining latent-space multimodal condition concatenation, temporal self-reference KV caching, 3D-RoPE grouped conditioning, and a fine-tuned VLM style-prompt system. Inference is 2.4× faster than the non-accelerated baseline, and the method surpasses existing VTON and general image-editing approaches on VITON-HD and DressCode.

## Background & Motivation
Virtual try-on (VTON) is a core capability for e-commerce, enabling consumers to obtain reliable outfit references online and reduce returns. Existing mainstream methods suffer from three categories of problems: (1) early warping-based methods (TPS, appearance flow) perform poorly under occlusion and large deformations; (2) GAN-based methods struggle to preserve fine garment details and natural body geometry; (3) diffusion-model methods substantially improve photorealism but universally rely on a Reference Net to encode garment features—IDM-VTON, OOTDiffusion, and FitDiT each employ an additional full network, doubling parameter count, complicating initialization and interaction logic, and slowing inference. Furthermore, most existing methods neglect outfit style control (e.g., whether a shirt is tucked in or left out), or depend on closed-source VLMs (e.g., PromptDresser uses GPT-4o) to generate style descriptions.

## Core Problem
How can high-fidelity multi-garment virtual try-on be achieved without a reference network? Specific sub-problems: (1) How can multiple heterogeneous conditions (person image, multiple garments, pose, mask) be injected efficiently without inflating computation? (2) How can the structure of a Flow Matching DiT be exploited to accelerate inference? (3) How can controllable outfit styling (e.g., "front tuck," "slim fit") be realized?

## Method

### Overall Architecture
PROMO is built on FLUX.1-dev (Flow Matching DiT) and fine-tuned with LoRA (rank = 128, 580M trainable parameters). The overall pipeline: given a person image $I_P$, garment images $\{I_{G_i}\}$, and an optional style text $T_{style}$, the model generates a new image $I_{new}$ of the person wearing the target garments. Condition injection adopts **latent-space multimodal concatenation**: the masked person image, each garment image, and a merged pose+mask condition are each encoded to the latent space via a shared VAE, then concatenated into a condition token sequence and fed into the DiT together with the denoising latent $z_t$ and text embeddings. Different conditions use different resolutions according to their information density (garments and person at original resolution; pose+mask downsampled to 25%), avoiding the constraint imposed by methods such as IC-LoRA that require all concatenated images to share a uniform resolution.

### Key Designs

1. **Temporal Self-Reference and 3D-RoPE Condition Encoding**
    - Function: Enables efficient inference acceleration and supports flexible outfit composition with single-garment training and multi-garment inference.
    - Mechanism: During inference, the Key-Value pairs of condition tokens (garment images $C_i$) are computed and cached at the first timestep; subsequent steps reuse the cached KV by computing only the Query for $z_t$ and $T_{style}$, yielding a 2.4× speedup (22.2 s → 9.2 s). In terms of attention visibility, $z_t$ and $T_{style}$ attend globally, while each $C_i$ attends only to itself. The RoPE time dimension is repurposed as a condition group identifier—$z_t$ receives time encoding 0, spatial conditions receive $i$, and garment conditions receive $(i, x, y+\Delta)$—allowing the model to distinguish "which garment goes where" with zero additional parameters, completing multi-garment try-on in a single forward pass and avoiding iterative error accumulation.
    - Design Motivation: Condition tokens carry semantically invariant information throughout denoising, making per-step recomputation unnecessary. RoPE grouped encoding leverages the existing positional encoding mechanism to achieve zero-parameter generalization across multiple garments.

2. **Latent-Space Multimodal Concatenation and Spatial Condition Compression**
    - Function: Unified encoding of multimodal condition inputs with a substantial reduction in token count.
    - Mechanism: The masked person image and each garment image are encoded to the latent space via a shared VAE and concatenated into a condition token sequence, with different resolutions assigned according to information density (garments and person at original resolution). The pose condition is pasted directly onto the agnostic mask image and then downsampled by 2×, compressing the original $2N$ tokens to $N/4$ (an 87.5% token reduction). Body parsing masks are also used for region-aware loss weighting, with body-region weights $1+\lambda$ and background weights $1-\lambda$ ($\lambda = 0.5$), focusing the model on garment details.
    - Design Motivation: Eliminates the uniform-resolution constraint of methods such as IC-LoRA; merging spatial conditions substantially reduces attention computation overhead with negligible information loss.

3. **Style Prompt System and TryOff Module**
    - Function: Extends input flexibility—supporting text-based style control and scenarios without standalone flat-lay garment images.
    - Mechanism: Qwen2.5-VL-72B is used to annotate a small dataset; after strict filtering, Qwen2.5-VL-7B is fine-tuned to generate structured outfit descriptions guided by a Pydantic OpenAPI JSON schema. Because all fine-tuning data is fully compliant, the 7B model is more accurate than the 72B model. The TryOff module extracts garment regions from model photos, enabling training and inference on unpaired data.
    - Design Motivation: Addresses PromptDresser's limitations of supporting only a single garment and relying on a closed-source model. The TryOff module extends applicability to real-world scenarios where standalone flat-lay garment images are unavailable.

### Loss & Training
- Flow Matching objective with region-aware weighting: $\mathcal{L} = \mathbb{E}_{t,z_0,\epsilon}[\mathbf{W} \odot \|\mathbf{v} - \mathbf{v}_\theta(z_t, t, \mathbf{c})\|^2]$
- Weighted loss design for downsampled parsing masks: because 16× downsampling loses fine details, a weighting scheme is applied to parsed regions to preserve discriminability.
- Optimizer: Prodigy (adaptive learning rate, default lr = 1); 16× H800 GPUs; effective batch size 16; 90K training steps.
- Training data: VITON-HD + DressCode training sets; resolution 1024×768.

## Key Experimental Results

| Dataset | Metric | PROMO | Any2AnyTryon | OOTDiffusion | CatVTON | Gain |
|--------|------|-------|--------------|--------------|---------|------|
| VITON-HD (paired) | SSIM↑ | 0.8913 | 0.9107 | 0.8883 | 0.8944 | 2nd |
| VITON-HD (paired) | LPIPS↓ | 0.0887 | 0.1208 | 0.0800 | 0.1600 | 2nd |
| VITON-HD (paired) | FID↓ | 3.3103 | 3.0828 | 3.6623 | 6.5372 | 2nd |
| VITON-HD (paired) | KID↓ | 0.4902 | 1.0565 | 0.8550 | 3.9591 | **Best** |
| VITON-HD (unpaired) | FID↓ | 4.7393 | 5.5404 | 7.0463 | 8.4567 | **Best** |
| VITON-HD (unpaired) | KID↓ | 0.4992 | 1.5258 | 2.7910 | 4.4897 | **Best** |
| DressCode (paired) | LPIPS↓ | 0.1111 | 0.1569 | 0.1905 | 0.1882 | **Best** |

**vs. General Image Editing Models**: PROMO comprehensively outperforms Seedream 4.0, Qwen-Image-Edit, and Nanobanana (Gemini 2.5-Flash-Image) on both VITON-HD and DressCode; general editing models exhibit noticeable color inconsistency and artifacts on VTON tasks.

**User Study (In-The-Wild)**: 13 persons × 40 garments = 520 groups, evaluated by 9 annotators:

| Method | Texture Consistency | Body Consistency | Style Consistency | Color Consistency | Overall Excellence |
|------|----------|----------|----------|----------|------------|
| PROMO | 93.65% | **94.62%** | **96.92%** | 97.88% | **84.42%** |
| Huiwa | **94.42%** | 88.85% | 94.80% | **99.04%** | 78.85% |
| Kling | 87.12% | 93.46% | 79.87% | 96.53% | 60.19% |
| Douyin | 96.73% | 79.04% | 85.19% | 95.77% | 61.54% |

### Ablation Study
- **3D-RoPE**: Removal causes a large drop across all metrics (FID 3.31→6.73, KID 0.49→1.72); the model can no longer correctly distinguish different condition groups, producing visible wrong-garment assignment and artifacts. This is the most critical component.
- **Style Prompts**: Removal raises FID from 3.31 to 3.72 and KID from 0.49 to 0.89, confirming that text guidance has a positive effect on quality while also providing style controllability.
- **Region-Aware Loss**: Removal raises unpaired KID from 0.50 to 0.95, with the effect being particularly pronounced in complex background scenarios.
- **Temporal Self-Reference**: Reduces inference time from 22.2 s to 9.2 s (2.4× speedup) with negligible change in SSIM/LPIPS/FID, confirming that condition KV caching is nearly lossless.
- **Spatial Condition Merging**: Reduces inference time from 11.1 s to 9.2 s (1.2× speedup) with no significant change in quality metrics, validating the rationale of exploiting spatial redundancy to reduce token count.

## Highlights & Insights
- **"Subtractive" engineering philosophy**: No reference network, no explicit warping, no closed-source VLM—every design choice simplifies the system while improving performance. Replacing the reference network with KV caching is a particularly elegant idea.
- **Creative use of 3D-RoPE**: Redefining the RoPE time axis as a condition group ID achieves zero-parameter multi-condition grouping and enables generalization from single-garment training to multi-garment inference.
- **Practical VLM distillation paradigm**: Large model annotation → strict filtering → small model fine-tuning yields a 7B model that is more accurate than the 72B model precisely because it has only seen compliant data. This pipeline is broadly reusable.
- **Comprehensive commercial-grade evaluation**: Assessment is not limited to academic benchmarks; user studies are conducted against commercial products including Huiwa, Kling, and Douyin, with an overall excellence rate of 84.42%—leading the comparison.

## Limitations & Future Work
- **Paired SSIM/LPIPS not state-of-the-art**: In the paired setting, SSIM is slightly below Any2AnyTryon, indicating room for improvement in pixel-level reconstruction accuracy.
- **Dependency on human parsing and DensePose**: The preprocessing pipeline remains heavy, requiring segmentation and pose estimation models; end-to-end simplification is a natural future direction.
- **Only benchmark evaluation is publicly available**: The paper mentions a self-collected in-the-wild dataset but does not release it.
- **Quality assurance for multi-garment inference**: While 3D-RoPE enables single-garment training to generalize to multi-garment inference, interactions between garments (e.g., coordinating tops and bottoms) are not directly optimized during training.
- **LoRA fine-tuning constraints**: Using only LoRA may limit the model's ability to adapt to the VTON-specific distribution; full-parameter fine-tuning could yield further gains.

## Related Work & Insights
- **vs. FitDiT**: Both are DiT-based VTON methods, but FitDiT employs a dual-network architecture (main DiT + reference DiT), whereas PROMO avoids a reference network via temporal self-reference, resulting in fewer parameters and faster inference.
- **vs. IDM-VTON / OOTDiffusion**: These use a UNet + reference network architecture; PROMO achieves a substantial LPIPS improvement on DressCode (0.111 vs. 0.190), demonstrating the advantage of a DiT backbone.
- **vs. CatVTON**: Both adopt concatenation-based condition injection, but CatVTON concatenates in image space and requires uniform resolution, whereas PROMO concatenates in latent space and supports different resolutions for different conditions.
- **vs. PromptDresser**: Both support style control, but PromptDresser relies on GPT-4o (closed-source, token-costly, single-garment only), while PROMO's self-trained 7B VLM is more efficient and accurate.
- **vs. General Editing Models (Seedream / Qwen / Gemini)**: General models exhibit severe color inconsistency and detail loss on VTON tasks, confirming that dedicated VTON models retain a clear advantage.

## Inspirations & Connections
- The temporal self-reference (KV cache reuse) approach is transferable to other multi-condition generation tasks, such as image editing and multi-subject customization.
- The 3D-RoPE grouped conditioning design may be applicable to any scenario requiring the model to distinguish among multiple reference images.
- The large-to-small VLM distillation pipeline is a valuable reference for the industry in building low-cost annotation pipelines.
- Framing VTON as structured image editing suggests that VTON training data could in turn be used to train general image editing models.

## Rating
- Novelty: ⭐⭐⭐⭐ — The application of temporal self-reference to DiT, 3D-RoPE grouped conditioning, and the VLM distillation style system are each novel; overall the work is an elegant combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluation across three datasets (VITON-HD, DressCode, in-the-wild), comprehensive comparison with both VTON methods and general editing models, ablation studies covering all key design components, and a user study against commercial products.
- Writing Quality: ⭐⭐⭐⭐ — System design is explained clearly with rich figures and tables; some formula notation definitions could be made more concise.
- Value: ⭐⭐⭐⭐ — An industrially oriented practical framework; multiple technical designs are transferable to other conditional generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2026\] FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation](fontcrafter_high-fidelity_element-driven_artistic_font_creation_with_visual_in-c.md)
- [\[CVPR 2026\] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)

</div>

<!-- RELATED:END -->
