---
title: >-
  [Paper Note] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images
description: >-
  [CVPR 2026][Image Generation][Spatial Transcriptomics] HINGE is a framework that, for the first time, repurposes a pre-trained expression-space single-cell foundation model (sc-FM…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Spatial Transcriptomics"
  - "Single-Cell Foundation Model"
  - "Masked Diffusion"
  - "Histology-Conditioned Generation"
  - "SoftAdaLN"
date: 2026-05-08
content_hash: 4749b07341993051
---

# HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images

**Conference**: CVPR 2026
**arXiv**: [2603.19766](https://arxiv.org/abs/2603.19766)  
**Code**: [https://github.com/donghaifang/HINGE](https://github.com/donghaifang/HINGE)  
**Area**: Biomedical Imaging / Generative Models
**Keywords**: Spatial Transcriptomics, Single-Cell Foundation Model, Masked Diffusion, Histology-Conditioned Generation, SoftAdaLN

## TL;DR
HINGE is a framework that, for the first time, repurposes a pre-trained expression-space single-cell foundation model (sc-FM, CellFM) as a histology-image-conditioned spatial gene expression generator. It achieves state-of-the-art performance on three ST datasets while maintaining superior gene co-expression consistency, through three core mechanisms: identity-initialized SoftAdaLN for lightweight visual context injection, an expression-space masked diffusion process that aligns with the pre-training objective, and a warm-start curriculum to stabilize training.

## Background & Motivation

**Background**: Spatial transcriptomics (ST) enables in situ measurement of gene expression but suffers from high cost and low throughput. Directly predicting spatial gene expression from routinely acquired H&E histology sections offers a practical alternative.

**Two Classes of Existing Methods**: (1) Deterministic regression (ST-Net / HisToGene / TRIPLEX) — maps histology patches to expression vectors but ignores intrinsic biological stochasticity; (2) Conditional generation (Stem / STFlow) — models conditional distributions more flexibly, yet fails to capture gene–gene dependencies that are difficult to infer from histology alone.

**Potential of Single-Cell Foundation Models (sc-FMs)**: Models such as scGPT and CellFM, pre-trained on large-scale scRNA-seq data, encode rich gene–gene regulatory and co-expression relationships. However, they operate purely in expression space and lack any visual pathway.

**Four Adaptation Challenges**: (a) **Modality gap** — sc-FMs have no visual pathway; (b) **Objective mismatch** — sc-FMs are pre-trained with masked auto-encoding, whereas diffusion models perturb all inputs with Gaussian noise; (c) **Composition shift** — scRNA-seq captures single cells, while ST measures mixed cell populations; (d) **Limited supervision** — ST datasets are small and noisy, making full fine-tuning prone to catastrophic forgetting.

**Core Idea**: Freeze the sc-FM backbone + inject histology and timestep conditioning via identity-initialized SoftAdaLN + align the diffusion process with masked auto-encoding pre-training via masked diffusion + stabilize early training with a warm-start curriculum.

## Method

### Overall Architecture
Frozen CellFM Transformer backbone → identity-initialized SoftAdaLN inserted before each MHA and SGLU sub-layer → frozen histology encoder $\phi$ extracts visual embeddings → masked diffusion forward process (progressively increasing masking ratio) and reverse process (progressively unmasking genes) → warm-start curriculum biasing sampling toward low-masking timesteps → output conditional gene expression vectors.

### Key Designs

1. **SoftAdaLN Conditional Injection (addresses modality gap and prevents forgetting)**

    - **Function**: Inserts lightweight conditional modulation before each Transformer sub-layer of CellFM.
    - **Mechanism**: Histology embedding $\mathbf{v}=\phi(\mathbf{c})$ and timestep embedding $\mathbf{e}_t$ are concatenated and passed through a shared transform $\mathbf{c}_t = \varphi_{cond}([\mathbf{v}; \mathbf{e}_t])$; each sub-layer applies SoftAdaLN:
    $\text{SoftAdaLN}(\mathbf{h}|\mathbf{c}_t) = \text{SoftNorm}(\mathbf{h}) \odot (1+\mathbf{s}(\mathbf{c}_t)) + \boldsymbol{\kappa}(\mathbf{c}_t)$
    where SoftNorm is a softened variant of standard layer normalization: $\text{SoftNorm}(\mathbf{h}) = (1-\eta)\mathbf{h} + \eta \cdot \frac{\mathbf{h}-\mu}{\sigma+\varepsilon}$
    - **Identity Initialization**: $\eta=0$ (SoftNorm degenerates to identity), $\mathbf{s}=\mathbf{0}$, $\boldsymbol{\kappa}=\mathbf{0}$, gate $\boldsymbol{\tau}\approx\mathbf{1}$ — the original CellFM behavior is exactly recovered at initialization.
    - Only the modulation parameters $\{\eta, \theta_\varphi, \theta_s, \theta_\kappa, \theta_\tau\}$ are trained; CellFM and the image encoder remain fully frozen.
    - **Design Motivation**: Identity initialization ensures that pre-trained gene relationships are fully preserved at the start of training; histology information is progressively injected as training proceeds; parameter efficiency (modulation layers only) avoids forgetting on small datasets.

2. **Expression-Space Masked Diffusion Process (addresses objective mismatch)**

    - **Function**: Designs a diffusion process aligned with the masked auto-encoding pre-training of sc-FMs.
    - **Forward Process**: Bernoulli masking (not Gaussian noise) is applied independently to each component of the gene expression vector; the masking rate increases according to a power schedule $\bar{\alpha}_t = (1-t/T)^\zeta$. At $t=0$ all genes are visible; at $t=T$ all genes are masked.
    - **Reverse Process**: Starting from a fully masked, all-zero state, each step predicts the masked components; already-unmasked components are kept unchanged, progressively revealing the complete gene expression.
    - **Training Objective**: $\mathcal{L}(\theta) = \mathbb{E}[w_t \|(1-\mathbf{m}_t) \odot (f_\theta(\mathbf{x}_t, t, \phi(\mathbf{c})) - \mathbf{x}_0)\|_2^2]$, computed only at masked positions.
    - **Alignment Key**: Both the input form (partially masked observations) and the supervision mode (loss only at masked positions) are consistent with CellFM's masked auto-encoding pre-training, enabling effective knowledge transfer.
    - **Design Motivation**: Standard Gaussian diffusion perturbs every component with noise, producing input distributions entirely different from masked auto-encoding and thus impeding knowledge transfer. Masked diffusion bridges this gap.

3. **Warm-Start Curriculum (stabilizes training)**

    - **Function**: Prioritizes low-masking timesteps during the initial phase of training.
    - **Mechanism**: During the first few fine-tuning epochs, the timestep sampler is biased toward $t \approx 0$ (few genes masked) and gradually transitions to uniform sampling (high masking).
    - **Design Motivation**: Low masking means most genes are visible, closely resembling the inputs seen by CellFM during pre-training, which stabilizes early gradient updates and prevents forgetting caused by early-stage instability.

### Inference
Given a histology patch $\mathbf{c}$: initialize $\mathbf{x}_T=\mathbf{0}, \mathbf{m}_T=\mathbf{0}$ → at each step sample unmasking probability $\pi_t$ to reveal new genes → predict masked genes → fill in predictions and retain already-revealed genes → after $T$ steps, obtain the complete gene expression. Re-sampling masking trajectories yields diverse yet histology-consistent samples.

## Key Experimental Results

### Main Results (Three ST Datasets)

| Method | Type | cSCC PCC-50↑ | Her2ST PCC-50↑ | Kidney PCC-50↑ |
|------|------|:---:|:---:|:---:|
| ST-Net | Regression | 0.548 | 0.439 | 0.327 |
| BLEEP | Regression | 0.643 | 0.520 | 0.404 |
| TRIPLEX | Regression | 0.683 | 0.536 | 0.410 |
| MERGE | Regression | 0.609 | 0.483 | 0.242 |
| Stem | Generation | 0.676 | 0.559 | 0.388 |
| STFlow | Generation | 0.678 | 0.543 | 0.391 |
| **HINGE** | **Generation** | **0.710** | **0.571** | **0.424** |

HINGE consistently outperforms all regression and generative baselines across all three datasets.

### Co-expression Consistency Analysis
Gene expressions generated by HINGE exhibit substantially higher agreement with ground-truth ST data on the pairwise Pearson correlation matrix, demonstrating that gene relationship knowledge from sc-FM is successfully preserved and transferred.

### Spatial Marker Gene Expression Patterns
HINGE produces spatial expression distributions for marker genes that more closely match ground-truth patterns, with superior spatial consistency over all baselines.

### Ablation Study

| Configuration | cSCC PCC-50 | Note |
|------|:---:|------|
| w/o sc-FM (random backbone init.) | Significant drop | Value of gene relationship knowledge |
| Gaussian diffusion (non-masked) | ~0.68 | Objective misalignment impedes transfer |
| w/o SoftAdaLN (direct concatenation) | Drop | Naive injection disrupts pre-trained features |
| w/o warm-start | Training instability | Large gradients at high masking in early steps |
| Full fine-tuning of CellFM (unfrozen) | Drop | Catastrophic forgetting on small datasets |
| **Full HINGE** | **0.710** | All components are complementary |

### Key Findings
- **Value of sc-FM pre-training is quantified**: Random initialization vs. CellFM yields a clear PCC-50 gap, which is even more pronounced in co-expression consistency, confirming that sc-FM gene relationship knowledge plays a central role in conditional generation.
- **Necessity of objective alignment**: Masked diffusion vs. Gaussian diffusion yields ~3% PCC gap; though modest in absolute terms, the gap is substantially larger in gene co-expression analysis, underscoring the importance of presenting the model with inputs similar to those seen during pre-training.
- **Freezing outperforms fine-tuning**: Full fine-tuning of CellFM on limited ST data leads to worse performance; freezing + SoftAdaLN is the superior strategy.
- **Criticality of identity initialization**: Non-identity-initialized conditional injection disrupts pre-trained behavior, making progressive adaptation essential.

## Highlights & Insights
- **A generalizable paradigm for cross-modal foundation model repurposing**: HINGE demonstrates a clear recipe — freeze backbone + identity-initialized modulation + pre-training objective alignment — that can repurpose any unimodal pre-trained model (text-only, expression-only) as a conditional generator. This has direct implications for other cross-modal adaptation scenarios (e.g., protein → structure, audio → vision).
- **Biological intuition behind masked diffusion**: The generative process for gene expression is more naturally framed as "progressively revealing the value of each gene" rather than "denoising all genes from Gaussian noise," which aligns naturally with sc-FM's masked auto-encoding paradigm.
- **Preserving pre-trained knowledge > injecting new information**: Under limited ST supervision, retaining the gene relationships learned by sc-FM is more important than aggressively injecting histology information, challenging the intuition that "more conditioning = better."
- **Advantages of generative over regression models**: HINGE not only surpasses regression baselines on PCC but shows even greater advantages in spatial consistency and co-expression patterns, indicating that generative approaches yield biologically more meaningful predictions.

## Limitations & Future Work
- The current instantiation uses only CellFM as the sc-FM backbone; adaptation with other models such as scGPT and scFoundation remains to be validated.
- The resolution of H&E histology limits the capture of fine-grained cell subtype variation; higher-resolution imaging modalities (e.g., immunofluorescence) may unlock additional information.
- The three ST datasets (cSCC / Her2ST / Kidney) are relatively small; larger-scale data may release more of the sc-FM's potential.
- Inference requires multiple sampling passes followed by averaging, increasing computational cost.
- Combining HINGE with spatially-aware sc-FMs (e.g., scGPT-spatial) is a promising future direction.

## Related Work & Insights
- **vs. Stem / STFlow**: These conditional generative methods do not leverage gene relationship knowledge from sc-FMs and learn solely from histology, resulting in inferior co-expression consistency.
- **vs. TRIPLEX**: This multi-scale regression method is competitive on PCC but ignores biological stochasticity.
- **vs. scGPT-spatial**: Performs spatially-aware continued pre-training in expression space without conditioning on histology — complementary to HINGE.
- **vs. AdaLN (e.g., DiT)**: DiT's AdaLN is trained from scratch, whereas HINGE's SoftAdaLN uses identity initialization to preserve existing pre-trained behavior.
- **Inspiration**: The "freeze + identity modulation" repurposing paradigm is generalizable to any scenario requiring the addition of a new modality condition to a pre-trained model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First adaptation of sc-FM for histology-conditioned gene expression generation; the masked diffusion design aligned with pre-training is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + six baselines (regression and generative) + co-expression analysis + spatial marker patterns + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The mapping from four challenges to corresponding solutions is clear; mathematical derivations are complete.
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both computational biology (spatial transcriptomics prediction) and AI methodology (cross-modal foundation model adaptation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)
- [\[ICLR 2026\] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](../../ICLR2026/image_generation/scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)
- [\[CVPR 2026\] NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](nanosd_edge_efficient_foundation_model_for_real_time_image_restoration.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)

</div>

<!-- RELATED:END -->
