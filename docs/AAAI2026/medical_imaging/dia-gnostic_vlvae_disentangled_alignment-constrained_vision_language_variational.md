---
title: >-
  [Paper Note] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities
description: >-
  [AAAI 2026][Medical Imaging][Radiology Report Generation] This paper proposes DiA-gnostic VLVAE, a vision-language mixture-of-experts VAE that learns a three-factor latent space ($Z_v$ visual-specific / $Z_l$ language-sp…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Radiology Report Generation"
  - "Missing Modalities"
  - "Disentangled Representation"
  - "VAE"
  - "MoE"
date: 2026-05-08
content_hash: 149bd6724ddd3c9d
---

# DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities

**Conference**: AAAI 2026
**arXiv**: [2511.05968](https://arxiv.org/abs/2511.05968)  
**Code**: N/A  
**Area**: Medical Imaging
**Keywords**: Radiology Report Generation, Missing Modalities, Disentangled Representation, VAE, MoE

## TL;DR
This paper proposes DiA-gnostic VLVAE, a vision-language mixture-of-experts VAE that learns a three-factor latent space ($Z_v$ visual-specific / $Z_l$ language-specific / $Z_s$ shared), with dual constraints of orthogonality and contrastive alignment for disentanglement. The model generates reliable radiology reports even when clinical context is absent, achieving competitive BLEU@4 on IU X-Ray and MIMIC-CXR.

## Background & Motivation

**Background**: Radiology Report Generation (RRG) is a critical task of automatically converting medical images into textual reports. The field has evolved from purely image-based models (R2Gen) to knowledge graph-enhanced approaches (MKSG) and context-aware models (PromptMRG), progressively incorporating richer clinical information.

**Limitations of Prior Work**:
   - **Missing modalities**: In clinical practice, contextual information (medical history, symptoms, demographics) is frequently incomplete.
   - **Feature entanglement**: The intermingling of modality-specific and shared information leads to suboptimal fusion and clinically inaccurate hallucinated findings.
   - LLM-based methods incur heavy computational costs; knowledge graph-based methods exhibit poor adaptability.
   - Retrieval-augmented methods degrade to deterministic rules when context is missing.

**Key Challenge**: How can cross-modal alignment remain stable under missing modality conditions?

**Goal**: Achieve robust radiology report generation against missing modalities through disentangled representation learning.

**Key Insight**: Decompose the latent space into three factors ($Z_v$ visual-specific, $Z_l$ language-specific, $Z_s$ shared), and adopt an MoE strategy to infer the shared latent variable such that missing modalities are automatically down-weighted.

**Core Idea**: Three-factor latent space disentanglement (orthogonality constraints for separation + contrastive alignment for semantic association) combined with an MoE shared encoder that gracefully degrades under missing modalities.

## Method

### Overall Architecture
1. **Feature Extraction**: EfficientNetB0 + GCA for visual features; Transformer encoder for language features.
2. **Modality Abstractor**: Bidirectional cross-attention to fuse visual and language features.
3. **VL-MoE-VAE**: Learns the three-factor latent space $(Z_v, Z_l, Z_s)$.
4. **LLaMA-X Decoder**: Generates reports from disentangled representations.

### Key Designs

1. **Three-Factor Latent Space Decomposition**:

    - Visual-specific $Z_v$: inferred by VGG16 encoder, $q_{\phi_v}(Z_v|V)$
    - Language-specific $Z_l$: inferred by Transformer encoder, $q_{\phi_l}(Z_l|L)$
    - Shared $Z_s$: MoE strategy, $q_{\phi_s}(Z_s|V,L) = \sum_{M} \pi_M q_{\phi_s}(Z_s|M)$
    - **Design Motivation**: Each latent variable is constrained to encode only its designated information — modality-specific latent variables must reconstruct their corresponding modality, while the shared latent variable encodes cross-modal semantics.

2. **Disentangled Alignment Constraints**:

    - **Orthogonality Constraint**: $\mathcal{L}_{orth} = \|\tilde{Z}_s^\top \tilde{Z}_v\|_F^2 + \|\tilde{Z}_s^\top \tilde{Z}_l\|_F^2 + \|\tilde{Z}_v^\top \tilde{Z}_l\|_F^2$, enforcing statistical independence among the three latent subspaces.
    - **Contrastive Alignment**: InfoNCE loss maximizes $I(Z_s; Z_v)$ and $I(Z_s; Z_l)$, ensuring the shared space encodes semantics from both modalities.
    - **Design Motivation**: The ELBO alone cannot guarantee meaningful latent factors. Orthogonality ensures separation while contrastive alignment ensures semantic relevance — the two constraints are complementary.

3. **Inference under Missing Modalities**:

    - When the language modality is absent, a "null" token is passed in; the MoE router automatically assigns $\pi_L \approx 0$ and $\pi_V \approx 1$.
    - Theoretical proof: the degraded objective remains a valid marginal ELBO lower bound.
    - Contrastive alignment training ensures $Z_s$ retains cross-modal semantics even when inferred from a single modality.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{ELBO} + \lambda_1 \mathcal{L}_{orth} + \lambda_2 \mathcal{L}_{align}$. The LLaMA-X decoder incorporates RoPE, grouped-query attention, and SwiGLU FFN for efficiency.

## Key Experimental Results

### Main Results

| Method | IU X-Ray B@4 | MIMIC-CXR B@4 |
|------|------------|--------------|
| R2Gen | 0.165 | 0.103 |
| XProNet | 0.199 | 0.105 |
| EKAGen | 0.203 | - |
| SEI | 0.263 | 0.131 |
| **DiA (Ours)** | **0.266** | **0.134** |

### Ablation Study
- Removing orthogonality constraint: disentanglement quality degrades and feature interference increases.
- Removing contrastive alignment: significant performance drop under missing modality conditions.
- Replacing MoE with PoE: catastrophic performance degradation when modalities are missing.

### Key Findings
- MoE is more suitable than PoE for handling missing modalities — PoE produces overconfident posteriors when a modality is absent.
- The dual orthogonality + contrastive constraint significantly outperforms either constraint alone.
- The LLaMA-X decoder is more efficient than large LLMs and avoids the limitations of template-driven generation.
- Performance degrades gracefully rather than catastrophically under missing modality conditions.

## Highlights & Insights
- **The theoretical analysis of MoE vs. PoE** offers important reference value for multimodal VAE research — the overconfidence problem of PoE under missing modalities is a critical but often overlooked issue.
- **The "graceful degradation" design philosophy** is well-suited for clinical deployment — rather than failing when input is incomplete, the model automatically adjusts based on available information.
- **The dual orthogonality + contrastive constraint** represents strong practice in disentangled representation learning — the former enforces separation while the latter preserves semantics.

## Limitations & Future Work
- Validation is limited to chest X-ray report generation; other imaging modalities (CT, MRI) remain untested.
- The specific scale and training details of LLaMA-X are insufficiently described.
- Sensitivity of model performance to the temperature parameter $\tau$ in contrastive alignment is not thoroughly analyzed.
- NLG metrics (BLEU) may not fully reflect clinical accuracy.

## Related Work & Insights
- **vs. R2Gen/CvT2Dis**: Image-only methods lacking contextual information; DiA improves by incorporating clinical context.
- **vs. SEI**: SEI employs retrieval augmentation without disentanglement, leaving feature interference unresolved; DiA's three-factor decomposition is more principled.
- **vs. PromptMRG**: Prompt- and LLM-based approaches are computationally expensive and template-dependent; DiA is more efficient and flexible.
- **vs. DrFuse**: DrFuse also performs disentanglement but uses an adversarial objective; DiA's contrastive + orthogonality approach is more stable.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of three-factor VAE + MoE + dual constraints demonstrates theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two standard benchmarks with ablations and missing-modality experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear architectural diagrams, and coherent logical flow.
- Value: ⭐⭐⭐⭐ Practically meaningful for multimodal medical report generation under missing modality conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[CVPR 2026\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](../../CVPR2026/medical_imaging/prototypebased_knowledge_guidance_for_finegrained.md)
- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](../../NeurIPS2025/medical_imaging/variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)

</div>

<!-- RELATED:END -->
