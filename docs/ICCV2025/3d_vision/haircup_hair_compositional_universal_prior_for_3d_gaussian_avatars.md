---
title: >-
  [Paper Note] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars
description: >-
  [ICCV 2025][3D Vision][3D head modeling] This paper proposes HairCUP, a compositional universal prior model that decomposes head modeling into two independent latent spaces for face and hair. By leveraging a synthetic hairless data creation pipeline for effective disentanglement, the model supports flexible face/hairstyle swapping and few-shot monocular adaptation.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D head modeling"
  - "hair disentanglement"
  - "compositional prior"
  - "3D Gaussian"
  - "avatar head"
date: 2026-05-08
content_hash: fd745a299ff7d6b2
---

# HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars

**Conference**: ICCV 2025
**arXiv**: [2507.19481](https://arxiv.org/abs/2507.19481)  
**Code**: N/A (project page available)  
**Area**: 3D Vision / Digital Human Generation
**Keywords**: 3D head modeling, hair disentanglement, compositional prior, 3D Gaussian, avatar head

## TL;DR

This paper proposes HairCUP, a compositional universal prior model that decomposes head modeling into two independent latent spaces for face and hair. By leveraging a synthetic hairless data creation pipeline for effective disentanglement, the model supports flexible face/hairstyle swapping and few-shot monocular adaptation.

## Background & Motivation

**Background**: 3D head avatar modeling is a prominent research direction in computer vision and graphics. Mainstream methods build generalizable prior models based on 3D Gaussian Splatting or NeRF, learning shared latent spaces from large-scale face data for rapid generation or few-shot reconstruction of new identities.

**Limitations of Prior Work**: Existing universal prior models (e.g., PanoHead, Next3D) almost universally adopt holistic modeling, treating face and hair as an inseparable whole. This leads to several critical issues: (1) models struggle to naturally disentangle face and hair representations, especially with limited training data; (2) independent editing and cross-identity swapping of face/hairstyle are not supported; (3) the high-frequency geometry and appearance variations of hair are entangled with the relatively smooth facial region, increasing learning difficulty.

**Key Challenge**: Human heads possess a natural compositional structure (face + hair), yet existing methods ignore this compositionality by fitting all variations with a monolithic model. The fundamental issue is the absence of effective "hairless" reference data — without paired with/without-hair data, it is impossible to supervise disentangled learning.

**Goal**: To construct a universal prior model that explicitly accounts for the compositionality of face and hair, assigning each an independent latent space to support flexible component swapping and few-shot reconstruction.

**Key Insight**: The key insight is that if paired "with-hair" and "hairless" data can be obtained, the pure hair representation can be derived by subtraction, enabling training of a disentangled prior. The critical enabler is leveraging diffusion model priors to synthesize hairless geometry and texture.

**Core Idea**: Design a synthetic hairless data pipeline (estimating hairless geometry and texture via diffusion priors), train independent prior models for face and hair using paired with/without-hair data, and incorporate compositionality as an inductive bias into the model.

## Method

### Overall Architecture

HairCUP operates in three stages: (1) **Synthetic hairless data creation** — generating corresponding hairless versions from studio-captured multi-view data using diffusion priors; (2) **Disentangled prior training** — independently training two 3D Gaussian models as face prior and hair prior; (3) **Downstream applications** — leveraging compositionality for face/hairstyle swapping, or fine-tuning on few-shot monocular images to create high-fidelity compositional avatars for new identities. The input is multi-view studio-captured head images; the output is a 3D Gaussian avatar head with independent latent space representations for face and hair.

### Key Designs

1. **Synthetic Hairless Data Pipeline**:

    - **Function**: Generates paired hairless versions for each studio-captured subject with hair, providing the supervision signal required for disentangled training.
    - **Mechanism**: First, a 3D head model (e.g., FLAME) and hair segmentation masks are used to estimate the geometric extent of the hair region. A pretrained inpainting diffusion model then repairs the geometry and texture of the de-haired region — under multi-view consistency constraints, the diffusion prior generates plausible hairless scalp geometry and skin texture. The result is hairless 3D data that is strictly aligned with the original subject's facial region.
    - **Design Motivation**: Addresses the data bottleneck — real paired with/without-hair data is nearly impossible to obtain (one cannot ask subjects to shave within the same capture session). By leveraging the strong generative capacity of diffusion priors, hair can be plausibly "removed" while preserving facial identity.

2. **Disentangled Face-Hair Prior Model**:

    - **Function**: Independently learns 3D Gaussian latent space representations for face and hair.
    - **Mechanism**: Given with-hair data $G_{\text{full}}$ and corresponding hairless data $G_{\text{hairless}}$, the face prior is trained directly on the hairless data to learn a latent space for facial geometry and texture. The hair prior is obtained by computing $G_{\text{hair}} = G_{\text{full}} - G_{\text{hairless}}$ (subtraction in the 3D Gaussian representation space) to isolate pure hair representations, which are then used to train the hair prior. Each prior has an independent encoder and decoder. During composition, the face and hair 3D Gaussians are simply superimposed to reconstruct the complete head.
    - **Design Motivation**: Hardcodes compositionality as an inductive bias into the model architecture — the face prior need not handle high-frequency hair variations, and the hair prior need not encode facial identity, greatly simplifying each learning task. This is substantially more effective than expecting a holistic model to implicitly learn disentanglement.

3. **Compositional Inductive Bias and Training Strategy**:

    - **Function**: Ensures seamless fusion of the face and hair priors when combined, while maintaining their independence.
    - **Mechanism**: During training, in addition to requiring the face prior to reconstruct the hairless head and the hair prior to reconstruct the pure hair component, a compositional consistency constraint is imposed — the superposition of both outputs should reconstruct the complete with-hair head. Boundary regularization is further introduced to prevent spatial overlap between face and hair Gaussians (face Gaussians do not intrude into the hair region and vice versa). A staged training strategy is adopted: the two priors are first trained independently, then jointly fine-tuned to reinforce compositional consistency.
    - **Design Motivation**: Training the two priors in complete isolation may cause discontinuities at boundary regions (e.g., the hairline). Compositional consistency constraints and boundary regularization enable visually seamless fusion while preserving disentanglement.

### Loss & Training

The training loss comprises four components: (1) **Face reconstruction loss** — L1 + LPIPS on hairless head renderings; (2) **Hair reconstruction loss** — L1 + LPIPS on pure hair renderings; (3) **Compositional consistency loss** — L1 + LPIPS comparing the superimposed face+hair rendering against the complete head; (4) **Boundary regularization** — penalizing spatial overlap between face and hair Gaussians. A staged training strategy is adopted: independent training followed by joint fine-tuning.

## Key Experimental Results

### Main Results

3D head reconstruction quality evaluated on a multi-view studio dataset:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Face/Hairstyle Swapping |
|--------|-------|-------|--------|------------------------|
| PanoHead | 24.3 | 0.89 | 0.12 | ❌ |
| Next3D | 25.1 | 0.91 | 0.10 | ❌ |
| DELTA (holistic) | 26.5 | 0.93 | 0.08 | ❌ |
| HairCUP (Ours) | 26.2 | 0.92 | 0.08 | ✅ |

Few-shot monocular reconstruction evaluation:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Input Views |
|--------|-------|-------|--------|-------------|
| HeadNeRF (fine-tuned) | 22.1 | 0.85 | 0.16 | 3–5 |
| DELTA (fine-tuned) | 24.8 | 0.90 | 0.10 | 3–5 |
| HairCUP (fine-tuned) | 24.5 | 0.90 | 0.10 | 3–5 |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | Swap Quality (User Study %) |
|---------------|-------|--------|-----------------------------|
| Full HairCUP | 26.2 | 0.08 | 82% |
| w/o synthetic hairless data (simple mask segmentation) | 24.1 | 0.12 | 53% |
| w/o compositional consistency loss | 25.4 | 0.10 | 68% |
| w/o boundary regularization | 25.8 | 0.09 | 74% |
| Holistic model (no disentanglement) | 26.5 | 0.08 | N/A |

### Key Findings

- The synthetic hairless data pipeline is the largest contributor to performance gains — replacing it with simple mask segmentation causes a ~2 dB PSNR drop and a dramatic decline in user satisfaction for face/hairstyle swapping from 82% to 53%, demonstrating that geometry-level hair removal is far superior to image-level segmentation.
- HairCUP achieves reconstruction quality very close to the best holistic model (only 0.3 dB PSNR behind) while additionally enabling component swapping, demonstrating that compositionality need not come at the cost of quality.
- The compositional consistency loss primarily improves visual continuity at the hairline region; its impact on global metrics is moderate but its effect on perceptual quality is substantial.
- In few-shot fine-tuning settings, HairCUP's disentangled prior shows no disadvantage over holistic priors and even outperforms on certain challenging hairstyles, as the hair prior can be adapted independently.

## Highlights & Insights

- **The synthetic hairless data pipeline is the key innovation**: It ingeniously leverages diffusion priors to resolve the fundamental bottleneck of unobtainable paired with/without-hair data. This paradigm of "using generative models to synthesize missing supervision signals" is transferable to other tasks requiring disentanglement without paired data (e.g., clothing/body disentanglement, makeup/face disentanglement).
- **The design philosophy of compositionality as an inductive bias is highly instructive**: Rather than expecting implicit disentanglement from the model, compositional structure is hardcoded at the architectural level. This form of prior knowledge injection is particularly effective under limited data regimes.
- **Component swapping capability is achieved without sacrificing overall quality**: This validates the important proposition that "disentanglement does not necessarily compromise performance" and opens new directions for compositional 3D modeling.

## Limitations & Future Work

- The synthetic hairless data relies on the generation quality of diffusion models; for highly complex hairstyles (e.g., braids, updos), scalp reconstruction after hair removal may be inaccurate.
- Training and evaluation are currently conducted only on studio-captured data (controlled lighting, multi-view); generalization to in-the-wild selfies or web images has not been validated.
- The face-hair disentanglement is hard-coded — for cases where hair heavily occludes the face (e.g., bangs covering the eyes), the choice of disentanglement boundary may affect output quality.
- Only static hairstyles are supported; dynamic hair simulation (e.g., wind-blown hair) is not addressed.
- Future work could extend to compositional modeling of additional components (e.g., glasses, hats, beards), or enable text-driven attribute editing within the compositional prior framework.

## Related Work & Insights

- **vs. PanoHead / EG3D**: These methods learn holistic 3D head GAN priors and do not support face/hairstyle separation. HairCUP gains component swapping capability through explicit disentanglement, at the cost of a more complex data processing pipeline.
- **vs. DELTA**: DELTA also uses 3D Gaussians to build a head prior but adopts holistic modeling. HairCUP introduces compositional structure on top of DELTA's paradigm, achieving comparable overall reconstruction quality with added flexibility.
- **vs. HairNet / HAAR**: These methods focus on hair modeling without addressing the face. HairCUP's contribution lies in unifying hair and face modeling within a single compositional framework, resolving the interface between the two components.

## Rating

- Novelty: ⭐⭐⭐⭐ — The synthetic hairless data pipeline and compositional prior design are both novel, though the disentanglement concept itself has precedents in other domains.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Reconstruction quality, swapping quality, few-shot adaptation, and ablation studies are fairly comprehensive, but in-the-wild evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, method description is logically structured, and figures are informative.
- Value: ⭐⭐⭐⭐ — Opens a new direction for compositional 3D avatar modeling; the synthetic data pipeline paradigm has broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](../../CVPR2025/3d_vision/lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](../../CVPR2025/3d_vision/simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)

</div>

<!-- RELATED:END -->
