---
title: >-
  [Paper Note] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval
description: >-
  [CVPR 2026][Self-Supervised Learning][UCDIR] This paper proposes TPSNet, which leverages CLIP-learned domain prompts as text priors to provide fine-grained semantic supervision, while introducing phase spectrum features as phase priors to bridge domain distribution gaps and preserve semantic integrity. Significant improvements in unsupervised cross-domain image retrieval (UCDIR) are achieved through the synergistic combination of text-phase dual priors.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - UCDIR
  - domain prompt
  - phase spectrum
  - text-phase dual priors
  - cross-domain alignment
date: 2026-05-08
content_hash: d705ffcaf1b8bb87
---

# Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.12711](https://arxiv.org/abs/2603.12711)
**Code**: N/A
**Area**: Cross-Domain Retrieval / Self-Supervised Learning
**Keywords**: UCDIR, domain prompt, phase spectrum, text-phase dual priors, cross-domain alignment

## TL;DR

This paper proposes TPSNet, which leverages CLIP-learned domain prompts as text priors to provide fine-grained semantic supervision, while introducing phase spectrum features as phase priors to bridge domain distribution gaps and preserve semantic integrity. Significant improvements in unsupervised cross-domain image retrieval (UCDIR) are achieved through the synergistic combination of text-phase dual priors.

## Background & Motivation

### Limitations of Prior Work

**State of the Field**: **Unsupervised Cross-Domain Image Retrieval (UCDIR)** aims to retrieve semantically equivalent images across heterogeneous image domains (e.g., real photos and sketches) without annotated data. The core challenge lies in the compounded difficulty of lacking labels and substantial domain distribution gaps.

**Two key limitations of existing methods**: (1) **Pseudo-label noise** — prior methods generate pseudo-labels via K-means clustering as supervision signals, but such discrete labels are frequently inaccurate, introducing noise into both intra-domain representation learning and cross-domain alignment, while also yielding unreliable class prototypes; (2) **Semantic degradation from cross-domain alignment** — strategies such as adversarial training and statistical distribution alignment inevitably compromise semantic information when suppressing domain discrepancies, as domain-specific and semantic features are entangled.

**Two solution paths in TPSNet**: (1) CLIP-learned domain prompts serve as text priors, offering richer and more precise semantic supervision than discrete pseudo-labels; (2) Phase spectra extracted via Fourier transform serve as phase priors — the phase spectrum encodes structural and semantic information while being robust to domain shift — bridging domain gaps while preserving semantic integrity. The two paths operate synergistically.

## Method

### Overall Architecture

TPSNet consists of two modules: **Domain Prompt Generation (DPG)** uses CLIP contrastive learning to optimize $C$ class-specific prompts per domain; **Text-Phase Dual Priors Network (TPDP)** uses the learned domain prompts as text priors to guide semantic feature extraction, while phase spectrum features serve as phase priors to bridge domain gaps. A cross-attention mechanism fuses the dual priors to yield the final domain-invariant semantic representation.

### Key Designs

1. **Domain Prompt Generation**:

   - **Function**: Learns $C$ class-specific learnable text prompts per domain, serving as semantic supervision signals for subsequent stages.
   - **Mechanism**: After generating pseudo-labels via K-means clustering, $C$ learnable prompt templates ("An image of a [X]¹...[X]^M") are initialized. A frozen CLIP model is used for image-text contrastive learning: $\mathcal{L}_{prompt} = \mathcal{L}_{i2t} + \mathcal{L}_{t2i}$, optimizing only the [X] tokens. Image-text pairs are re-matched based on cosine similarity during contrastive learning, partially correcting inaccurate pseudo-labels.
   - **Design Motivation**: CLIP's textual representations provide richer semantic priors than discrete pseudo-labels; after contrastive optimization, the domain prompts encode precise class-level semantic information.

2. **Phase-Prior Domain-Invariant Feature Extraction**:

   - **Function**: Exploits the domain-invariant property of the phase spectrum to bridge domain gaps.
   - **Mechanism**: FFT is applied to the grayscale image to obtain $F(u,v)=|A(u,v)|e^{j\phi(u,v)}$. The phase is retained while the amplitude is replaced by a constant $R$: $F'(u,v) = Re^{j\phi(u,v)}$. The phase image is reconstructed via IFFT. A lightweight CNN extracts phase features $I^{phase}$, which are fused with RGB features via LayerNorm + Self-Attention to produce $I^f$.
   - **Design Motivation**: The phase spectrum encodes structural and edge information and is more robust to domain shift than the amplitude spectrum. Discarding the amplitude naturally eliminates certain domain-specific factors (e.g., style, color distribution).

3. **Text-Phase Dual-Prior Synergistic Fusion**:

   - **Function**: Employs cross-attention to enable text and phase priors to jointly guide domain-invariant representation learning.
   - **Mechanism**: Domain prompt text features $T'$ serve as Query; fused visual features $I^f$ serve as Key/Value: $I' = \text{CrossAttention}(T'; I^f)$. Joint training is performed using prototype cross-entropy loss $\mathcal{L}_{pce}$ and image-text contrastive loss $\mathcal{L}_{i2tce}$ (with label smoothing). Prototypes are updated via momentum: $\mathcal{P} \leftarrow m\mathcal{P} + (1-m)I'$.
   - **Design Motivation**: The text prior provides semantic guidance, the phase prior eliminates domain shift, and cross-attention enables their complementary enhancement in feature space.

### Loss & Training

$\mathcal{L} = \alpha \mathcal{L}_{pce} + \beta \mathcal{L}_{i2tce}$, where $\mathcal{L}_{i2tce}$ applies label smoothing $\sigma_j = (1-\epsilon)y_i + \epsilon/C$ to mitigate pseudo-label noise. Stage 1 optimizes only the prompt tokens; Stage 2 trains all components of TPDP.

## Key Experimental Results

### Main Results

**Office-Home (65 classes, 4 domains, 12 cross-domain scenarios) and DomainNet (7 classes, 6 domains)**:

| Method | Office-Home Avg. P@1 | Office-Home Avg. P@15 |
|--------|---------------------|-----------------------|
| DD | ~45 | ~35 |
| ProtoOT | ~50 | ~47 |
| ShieldIR | ~53 | ~50 |
| **TPSNet** | **Significant SOTA** | **Significant SOTA** |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| Pseudo-labels only (no domain prompt) | Baseline | High noise |
| + Text prior (domain prompt) | Significant ↑ | More precise semantic supervision |
| + Phase prior | Further ↑ | Domain-invariant features are beneficial |
| **Dual-prior synergy** | **Best** | Complementary enhancement is optimal |

### Key Findings

- The text prior alone yields significant gains — indicating that CLIP's semantic signals are substantially richer than clustering-based pseudo-labels.
- The phase prior yields larger improvements in scenarios with larger domain gaps (e.g., Art↔Clipart) — validating the domain-invariance hypothesis of the phase spectrum.
- Label smoothing is effective in alleviating pseudo-label noise.

## Highlights & Insights

- The dual-path design of text priors and phase priors is highly instructive — the former provides complementary domain-invariant signals from semantic space, the latter from frequency space. This "multi-view domain invariance" is more robust than single-alignment strategies.
- The operation of reconstructing images using a constant amplitude with the original phase is simple yet effective — confirming that the phase spectrum does encode structurally consistent semantic information across domains.

## Limitations & Future Work

- The method relies on K-means clustering to initialize domain prompts; clustering quality has a substantial impact on all subsequent steps.
- Phase spectra are extracted only from grayscale images, discarding potentially domain-invariant components contained in color information.
- The domain gaps in the evaluated datasets are relatively moderate; effectiveness under more extreme domain shifts remains to be verified.

## Related Work & Insights

- **vs. DD/CODA**: These methods directly use pseudo-labels for intra-domain contrastive learning and cross-domain alignment; TPSNet replaces pseudo-labels with domain prompts for superior semantic supervision.
- **vs. FDA/FUDA**: These methods perform domain adaptation by replacing low-frequency components in the frequency domain; TPSNet goes further by separating amplitude and phase, exploiting the natural domain-invariance of the phase.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-path combination of text priors and phase priors is a novel contribution to UCDIR.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two benchmarks, 12 cross-domain scenarios, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐ The structure is clear, though some figures and tables are complex.
- **Value**: ⭐⭐⭐ UCDIR is a meaningful problem, and the reported improvements are substantial.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale](craterbench-r_instance-level_crater_retrieval_for_planetary_scale.md)
- [\[NeurIPS 2025\] Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization](../../NeurIPS2025/self_supervised/minimal_semantic_sufficiency_meets_unsupervised_domain_generalization.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)

<!-- RELATED:END -->
