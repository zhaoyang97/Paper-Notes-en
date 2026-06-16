---
title: >-
  [Paper Note] Bridging Privacy and Provenance: Traceable Virtual Identity Generation
description: >-
  [CVPR 2026][AI Safety][Diffusion Model] This paper proposes a diffusion-based framework to generate "stable, reproducible, yet unrecognizable" virtual faces for users. By embedding a 128-bit invisible watermark as an identity fingerprint during generation, users can verify "this virtual face belongs to me" with a secret key without exposing their real faces,
tags:
  - CVPR 2026
  - AI Safety
  - Diffusion Model
date: 2026-05-08
content_hash: 84a458c499435ae4
---
# Bridging Privacy and Provenance: Traceable Virtual Identity Generation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Bridging_Privacy_and_Provenance_Traceable_Virtual_Identity_Generation_CVPR_2026_paper.html)  
**Code**: None  
**Area**: AI Safety / Face Privacy / Diffusion Models  
**Keywords**: Virtual Identity, Face Anonymization, Digital Watermarking, Traceability, Diffusion Models  

## TL;DR
This paper proposes a diffusion-based framework to generate "stable, reproducible, yet unrecognizable" virtual faces for users. By embedding a 128-bit invisible watermark as an identity fingerprint during generation, users can verify "this virtual face belongs to me" with a secret key without exposing their real faces, simultaneously achieving anonymity and traceability.

## Background & Motivation
**Background**: Face anonymization aims to remove "real facial features" from images while maintaining utility. Early methods relied on blurring, masking, or pixelation, which degraded image quality. The generative era shifted towards attribute editing or identity-conditioned synthesis, which can replace identity while preserving background and pose, significantly improving visual quality.

**Limitations of Prior Work**: The authors point out that existing methods are stuck between two mutually exclusive requirements. First is **identity inconsistency**—anonymizing different photos of the same person might result in completely different "strangers," making them unusable for authentication, login, or payment scenarios requiring identity stability. Second is **non-traceability**—once the original identity is mapped and discarded, the link between the virtual face and the real person is severed, leaving no mechanism to prove ownership in cases of misuse.

**Key Challenge**: Current approaches fail to address these issues simultaneously. Reversible identity transformations (e.g., FIT, RiDDLE, G2Face) rely on user keys to reconstruct the original face, creating a natural conflict between "anonymity" and "recoverable biometrics"—recoverability implies privacy is not truly protected. One-way "real-to-virtual" mappings (e.g., IVFG) ensure consistency for authentication but lose the link to the origin after discarding the original identity. **It is difficult to achieve anonymity, consistency, and traceability within a single framework.**

**Goal**: To create a virtual face that satisfies: (i) unrecognizability (anonymity), (ii) stable and consistent virtual identity across multiple photos of the same person, (iii) ownership verification (traceability), and (iv) fidelity of identity-irrelevant attributes like pose and expression.

**Key Insight**: Instead of choosing between "reversible transformation" and "explicit mapping," the authors propose a different path—**not to preserve or recover the original face, but to embed an invisible, verifiable identity fingerprint directly into the generated face**. This fingerprint, derived from a privacy-preserving transformation of the user's biometrics (e.g., hashed fingerprint descriptors), is invisible to the human eye but can be decoded for verification.

**Core Idea**: The framework integrates "virtual identity sampling + 3D geometry/expression conditioning + latent space watermarking" into a unified diffusion generation pipeline. A sampler creates stable, anonymous identity embeddings; 3DMM extracts pose and expression; and watermarking embeds the identity fingerprint. This results in a face that is "anonymous, consistent, controllable, and traceable."

## Method

### Overall Architecture
The framework (Fig. 2) decomposes virtual face generation into three serial modules, synthesized via a diffusion model. The **input** is the user's original face, and the **output** is an anonymous yet traceable watermarked virtual face.

The process: First, the original identity embedding $e_{ori}$ is extracted using a pre-trained face recognition model $E_{id}$ (ArcFace). It is processed by the **VID Sampler** to produce a virtual identity embedding $e_{vir}$ that is sufficiently distant from the original yet stable across images for the same user. Simultaneously, **head pose and expression** are estimated from the original face using a 3DMM pipeline (EMOCA for normal maps, SMIRK for fine-grained expression codes). Finally, the **Watermarked Virtual Face Generation** module feeds the identity embedding (injected via CLIP text space with the pseudo-prompt "photo of a `<id>` face"), normal maps (via ControlNet), and expression codes (via an IP-Adapter style adapter) into Stable Diffusion. During the VAE decoding stage, a user-specific 128-bit identity watermark is embedded. For verification, the authority decodes the bitstream from the watermark and compares it with the user-provided identity token to confirm ownership.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Original Face"] --> B["VID Sampling<br/>e_ori→e_vir (Anon/Consistent/Diverse)"]
    A --> C["Pose & Expression Preservation<br/>EMOCA Normal Map + SMIRK Expression Code"]
    B -->|Identity Condition (Pseudo-prompt)| D["Watermarked Face Generation<br/>SD Synthesis + 128-bit Latent Watermark"]
    C -->|ControlNet + Adapter Injection| D
    D --> E["Anonymous + Traceable Virtual Face"]
    E -->|Decode Bitstream vs Identity Token| F["Ownership Verification / Traceability"]
```

### Key Designs

**1. VID Sampler: Generating identity embeddings that are "unrecognizable" yet "consistent" on a hypersphere.**

This step addresses identity inconsistency and anonymity. Working in the $\ell_2$-normalized identity embedding space of ArcFace (unit hypersphere $S^{d-1}$), given $e_{ori}$ and a user-specific condition $k$, the sampler defines a rule $e_{vir} \sim p_\phi(e \mid e_{ori}, k)$ with three properties: **Anonymity**—ensuring the angular distance between virtual and original identities is large ($\cos(e_{vir}, e_{ori}) < \tau_a$); **Consistency**—ensuring virtual identities from different photos of the same user under a fixed $k$ are close; **Diversity**—ensuring different $k$ values produce different virtual identities for the same user.

Three samplers are proposed: ① **vMF Distribution Sampling**—uses a von Mises–Fisher distribution centered oppositely to $e_{ori}$ on $S^{d-1}$, applying rejection sampling to satisfy $\cos(e_{vir}, e_{ori}) < \tau_a$; training-free but harder to control. ② **Generative HS-AE**—trains a hyperspherical autoencoder $D_\psi$ to decode a latent code $z \in S^{L-1}$ back to the identity sphere: $e_{vir} = D_\psi(z)$; manifold-aware and highly diverse, but requires distance checks for anonymity. ③ **ID-Mixer**—a lightweight MLP $G_\phi$ that maps the original identity and a latent code: $e_{vir} = G_\phi(e_{ori}, z)$; explicitly conditioned on the original identity for user-specific generation, requiring a training balance between anonymity, consistency, and diversity.

**2. Pose and Expression Preservation: Extracting identity-irrelevant geometry via 3DMM.**

To prevent loss of expression or pose during identity swapping, the authors use a 3DMM pipeline based on the FLAME model. **3DMM naturally decomposes faces into identity, expression, and shape, allowing the transfer of "identity-irrelevant factors" without exposing the original identity**. Two modules are used: **EMOCA** provides stable, identity-irrelevant head pose and coarse geometry, rendered as a **normal map** for ControlNet to constrain global orientation. **SMIRK** captures fine-grained, asymmetric, or compound expressions, projecting high-dimensional expression codes via an IP-Adapter style adapter into the CLIP latent space for cross-attention modulation in the Diffusion UNet.

**3. Watermarked Virtual Face Generation: Embedding a 128-bit fingerprint during decoding.**

This addresses the "non-traceability" issue. The backbone is Stable Diffusion (SD 1.5), where the denoising UNet is controlled by three conditions: identity embedding $e_{vid}$ (identity), normal maps (geometry), and expression codes (emotion). The core traceability mechanism is at the **decoding end**: inspired by GenPTW, the authors implement a **128-bit encoder-decoder** for latent space watermarking. It is streamlined for "identity binding," removing tamper-localization components while retaining robustness against JPEG compression and photometric distortions. The denoised latent $\hat z_0$ passes through the watermark-aware decoder, implicitly embedding the bitstream. This bitstream, derived from the user's biometrics, serves as a cryptographically bound tag for future verification without exposing the real face.

### Loss & Training
Only HS-AE and ID-Mixer require training.

**HS-AE** is trained with a **KL-free** reconstruction objective on the hypersphere. The encoder predicts a unit latent direction $\mu \in S^{L-1}$. During training, Gaussian noise is added in the tangent plane and projected back: $z = \mathrm{norm}\big((1-\eta)\mu + \eta\,\mathrm{norm}(\mu + \sigma\xi_\perp)\big)$, where $\xi \sim \mathcal{N}(0,I)$ and $\xi_\perp = \xi - \langle\xi,\mu\rangle\mu$. The decoder uses cosine similarity loss $L = 1 - \cos(\hat e, e)$.

**ID-Mixer** uses a multi-task objective:

$$L = \lambda_{ano}L_{ano} + \lambda_{div}L_{div} + \lambda_{intra}L_{intra} + \lambda_{inter}L_{inter} + \lambda_{reg}L_{reg}$$

Where $L_{ano}$ pushes the virtual identity away from the original (Anonymity), $L_{div}$ separates virtual identities of the same source with different keys (Diversity), and $L_{intra}$ pulls virtual identities of the same source and key together (Consistency).

## Key Experimental Results

Models were trained on CelebA-HQ (identity-disjoint split) and tested on LFW and FFHQ. SD 1.5 was used with a fine-tuned Arc2Face text encoder.

### Main Results: Virtual Identity Consistency (CelebA-HQ)
EER↓ and AUC↑ measure whether virtual faces of the same person are consistently recognized.

| Method | ArcFace EER↓ | ArcFace AUC↑ | FaceNet EER↓ | AdaFace EER↓ |
|------|------|------|------|------|
| Original (Real) | 0.050 | 0.976 | 0.065 | 0.054 |
| IVFG | 0.161 | 0.916 | 0.164 | 0.160 |
| RiDDLE | 0.046 | 0.757 | 0.308 | 0.316 |
| G2Face | 0.241 | 0.835 | 0.237 | 0.258 |
| **Ours (ID-Mixer)** | 0.021 | 0.998 | 0.026 | 0.023 |
| **Ours (vMF)** | **0.002** | **1.000** | 0.006 | 0.001 |
| **Ours (HS-AE)** | **0.002** | **1.000** | 0.005 | 0.001 |

The vMF / HS-AE samplers reduce EER to 0.002, outperforming the **original real face dataset** in consistency across three recognition backbones.

### Anonymity and Diversity (CelebA-HQ)
Anonymity measures: IAR↑ (Inconsistency Rate) and Sim↓ (Cosine Similarity); Diversity: Div↑ and a new metric, Orthogonal Diversity (ODV↑).

| Method | ArcFace IAR↑ | ArcFace Sim↓ | ArcFace Div↑ | ArcFace ODV↑ |
|------|------|------|------|------|
| IVFG | 1.000 | 0.034 | 0.913 | 79.20 |
| RiDDLE | 0.998 | 0.061 | 0.765 | 75.13 |
| **Ours (ID-Mixer)** | 1.000 | 0.034 | 0.923 | 79.54 |
| **Ours (vMF)** | 1.000 | 0.037 | **0.997** | **84.06** |
| **Ours (HS-AE)** | 1.000 | 0.038 | 0.999 | 83.66 |

Ours matches the best baseline (IVFG) in anonymity while leading in diversity metrics.

### Watermark Effectiveness (Table 6)
| Dataset | Bit Acc↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|------|------|
| CelebA-HQ | 0.998 | 39.74 | 0.974 | 0.019 |
| FFHQ | 0.998 | 39.63 | 0.974 | 0.019 |

Decoding accuracy reaches 0.998 bits with minimal image degradation (PSNR > 39.6 dB).

### Ablation Study
| Configuration | Observation |
|------|------|
| Full impl. | Complete framework: anon + consistent + pose/expression fidelity + watermark. |
| w/o VID sampler | Uses $\mathcal{N}(0,1)$ random sampling; virtual faces deviate from the real face manifold. |
| w/o EMOCA | Loss of basic head pose and coarse geometric constraints. |
| w/o SMIRK | Degradation in fine-grained expression reconstruction. |
| w/o watermark | No significant change in visual quality (proves watermark is invisible). |

### Key Findings
- **Sampler is most critical**: Removing the VID sampler causes virtual faces to fall off the manifold.
- **Division of labor**: EMOCA handles global pose, while SMIRK handles high-frequency expression.
- **Utility trade-off**: Generating an entire virtual face leads to higher distribution shifts (higher FID) compared to local editing, a necessary cost for identity stability.

## Highlights & Insights
- **Decoupling Anonymity and Traceability**: Identity is handled via hypersphere sampling for anonymity, while traceability is handled via a latent watermark. This is cleaner than reversible transforms where a single key manages both conflictual goals.
- **Lightweight Watermarking**: By tailoring watermarks for identity tokens rather than forensic localization, the authors achieved near-zero overhead with 0.998 bit accuracy.
- **Hypersphere + KL-free Tangent Noise**: This training strategy avoids posterior collapse in normalized embedding spaces, offering insights for other retrieval or contrastive learning tasks.

## Limitations & Future Work
- **Image Quality Cost**: Full face regeneration leads to higher FID than local editing methods.
- **Watermark Robustness**: Robustness against adversarial attacks or removal is primarily discussed in supplementary materials; primary evidence for "watermark removal" resistance is limited.
- **Centralized Dependency**: Traceability relies on a mapping database; leakage of the database or tokens remains a risk.
- **Pre-trained Component Dependency**: Failure in any component (e.g., 3DMM for extreme poses) propagates to the final output.

## Related Work & Insights
- **vs. Reversible Transforms (FIT/RiDDLE/G2Face)**: These risk exposing biometrics if the transform is reversed; Ours is non-reversible and relies on invisible tags.
- **vs. One-way Mapping (IVFG)**: IVFG ensures consistency but lacks traceability; Ours provides superior consistency and adds a traceability layer.
- **vs. Generative Watermarking**: While others focus on forgery detection, Ours repurposes multi-bit watermarking for "identity tokenization."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Successfully integrates anonymity and traceability via decoupled channels).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive across metrics, though robustness details are largely supplementary).
- Writing Quality: ⭐⭐⭐⭐ (Clear logic and structure).
- Value: ⭐⭐⭐⭐⭐ (Directly addresses real-world virtual identity authentication needs).

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] Reinforcement-Guided Synthetic Data Generation for Privacy-Sensitive Identity Recognition](reinforcement-guided_synthetic_data_generation_for_privacy-sensitive_identity_re.md)
- [\[CVPR 2026\] No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation](no_way_to_steal_my_face_proactive_defense_against_identity-preserving_personaliz.md)
- [\[CVPR 2026\] GROW: Watermark Generation with Progressive Guidance for Diffusion Models](grow_watermark_generation_with_progressive_guidance_for_diffusion_models.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[CVPR 2026\] DeepProtect: Proactive Face-Swapping Defense using Identity Blending and Attribute Distortion](deepprotect_proactive_face-swapping_defense_using_identity_blending_and_attribut.md)

</div>

<!-- RELATED:END -->
