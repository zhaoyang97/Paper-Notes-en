---
title: >-
  [Paper Note] DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization
description: >-
  [ICCV 2025][Model Compression][LoRA merging] DuoLoRA introduces rank-dimension mask learning (ZipRank) for LoRA merging, combined with SDXL layer priors and a cycle-consistent merging loss (Constyle loss), enabling efficient content-style LoRA composition that surpasses ZipLoRA and other state-of-the-art methods across multiple benchmarks while reducing trainable parameters by 19×.
tags:
  - ICCV 2025
  - Model Compression
  - LoRA merging
  - content-style personalization
  - cycle consistency
  - rank disentanglement
  - diffusion models
date: 2026-05-08
content_hash: 286495f921b268b6
---

# DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization

**Conference**: ICCV 2025  
**arXiv**: [2504.13206](https://arxiv.org/abs/2504.13206)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: LoRA merging, content-style personalization, cycle consistency, rank disentanglement, diffusion models

## TL;DR
DuoLoRA introduces rank-dimension mask learning (ZipRank) for LoRA merging, combined with SDXL layer priors and a cycle-consistent merging loss (Constyle loss), enabling efficient content-style LoRA composition that surpasses ZipLoRA and other state-of-the-art methods across multiple benchmarks while reducing trainable parameters by 19×.

## Background & Motivation
Personalized generation with text-to-image diffusion models has attracted considerable attention. Users wish to simultaneously specify the content (e.g., a specific dog) and style (e.g., impressionism) of generated images, yet existing methods struggle to faithfully preserve both within a single model.

LoRA (Low-Rank Adapter) has become the dominant paradigm for personalization owing to its parameter efficiency: separate content and style LoRAs are trained and subsequently merged. ZipLoRA is a representative approach that merges content and style by learning masks over the output dimension of LoRA. However, ZipLoRA suffers from three critical limitations:

**Unreasonable independence assumption**: ZipLoRA treats content and style as independent entities, whereas they are inherently entangled.

**Poor parameter efficiency**: Learning masks over the output dimension requires a large number of trainable parameters (1.33M).

**Lack of adaptive rank flexibility**: The same rank is applied to both content and style LoRAs, ignoring the fact that different layers have different representational demands for content and style.

The core problem addressed in this paper is how to achieve adaptive rank flexibility while reducing fine-tuning cost and enhancing the separation of content and style distributions. The approach exploits the differences in content and style encoding preferences across layers of different resolutions in the SDXL UNet architecture.

## Method

### Overall Architecture
DuoLoRA comprises three core components: (1) ZipRank — mask learning in the rank dimension; (2) layer prior-guided merging strategy — exploiting content/style encoding preferences in the SDXL architecture; and (3) Constyle Loss — a cycle-consistent merging loss.

### Key Designs
1. **ZipRank: Rank-Dimension Mask Learning**

    - Function: Learns merging masks along the rank dimension of LoRA rather than the output dimension.
    - Mechanism: A diagonal mask matrix $M_r \in \mathbb{R}^{r \times r}$ is defined such that the rank-masked approximation is $\Delta W_{rank} = A M_r B = U_r M_r \Sigma_r V_r^\top$. Compared to ZipLoRA's mask over $d_{out}$ dimensions, the rank-dimension mask requires only $r$ parameters ($r \ll d_{out}$), reducing trainable parameters from 1.33M to 0.07M.
    - Design Motivation: The paper provides a theoretical proof that, under the same parameter budget, the approximation error of the rank-dimension mask satisfies $E_{rank} \leq E_{out}$ (the error of the output-dimension mask), establishing the theoretical superiority of the rank-dimension approach.
    - Key Advantage: Provides adaptive rank flexibility, allowing different layers to automatically learn different effective ranks.

2. **Layer Prior-Guided Merging Strategy**

    - Function: Leverages the content/style encoding preferences of layers at different resolutions in the SDXL UNet to guide merging.
    - Mechanism: Empirical analysis reveals that **low-resolution layers** (up_block.2, down_block.2, mid_block, resolution < 32) dominate content generation, while **high-resolution layers** (up_block.1, down_block.1, resolution ≥ 32) dominate style generation. Based on this finding, two strategies are proposed:
        - **Prior initialization**: In content-dominant layers, more entries of the content merger are initialized to 1, and vice versa.
        - **Explicit rank constraint**: In content-dominant layers, $Rank(m_c) > Rank(m_s)$ is enforced, formulated as a nuclear norm minimization problem: $\mathcal{L}_{layer\_prior} = \|m_c\|_1 + \lambda \max(0, \|m_s\|_* - \|m_c\|_*)$
    - Design Motivation: The hypothesis is validated through layer-freezing simulation experiments (selective weight scaling) — scaling down the weights of low-resolution layers causes the generated images to lose content while retaining style.

3. **Constyle Loss: Cycle-Consistent Merging**

    - Function: Leverages cycle consistency between content and style to optimize merging quality.
    - Mechanism: Inspired by CycleGAN, content and style are treated as two domains. Two cycles are defined:
        - **Style cycle**: content image → add style → remove style → reconstruct content image, minimizing $\mathcal{L}_{cycle\_style} = MSE(I_{cc}, I_{csc})$
        - **Content cycle**: style image → add content → remove content → reconstruct style image, minimizing $\mathcal{L}_{cycle\_content} = MSE(I_{ss}, I_{scs})$
    - Design Motivation: ZipLoRA processes content and style independently, neglecting their mutual dependence. Cycle consistency naturally models this interdependency.

### Loss & Training
The total loss is:
$$\mathcal{L} = \lambda_{layer\_prior} \mathcal{L}_{layer\_prior} + \lambda_{cycle} \mathcal{L}_{constyle}$$

where $\lambda_{cycle} = 0.01$ and $\lambda_{layer\_prior} = 0.1$. The merging stage is trained for 100 steps using the Adam optimizer with a learning rate of 0.01.

## Key Experimental Results

### Main Results
Comparison with state-of-the-art methods on 4 datasets (Dreambooth + StyleDrop dataset):

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ | CSD-s↑ | Params (M) |
|------|-------|---------|---------|--------|----------|
| Naïve Merging | 0.47 | 0.64 | 0.266 | 0.44 | - |
| B-LoRA | 0.45 | 0.57 | 0.281 | 0.28 | - |
| ZipLoRA | 0.53 | 0.65 | 0.285 | 0.41 | 1.33 |
| **DuoLoRA** | **0.56** | **0.69** | **0.314** | **0.48** | **0.07** |

Comparison with Paircustomization:

| Method | DINO↑ | CLIP-I↑ | CSD-s↑ |
|------|-------|---------|--------|
| Paircustomization | 0.56 | 0.65 | 0.47 |
| DuoLoRA | **0.62** | **0.69** | **0.50** |

### Ablation Study

| Configuration | DINO↑ | CLIP-I↑ | CSD-s↑ | Note |
|------|-------|---------|--------|------|
| ZipRank | 0.53 | 0.64 | 0.42 | Rank-dimension mask only |
| ZipRank + Layer-Priors | 0.54 | 0.67 | 0.45 | + Layer prior constraints |
| DuoLoRA (Full) | 0.56 | 0.69 | 0.48 | + Constyle Loss |

### Key Findings
- ZipRank achieves performance comparable to ZipLoRA with 19× fewer trainable parameters (0.07M vs. 1.33M).
- Layer prior initialization and rank constraints provide consistent gains (DINO +0.01, CSD-s +0.03).
- Constyle Loss further improves content-style disentanglement (CSD-s from 0.45 to 0.48).
- DuoLoRA consistently outperforms baselines in multi-concept stylization (2/3/4 concepts).
- 50% of users prefer DuoLoRA-generated results in the user study.

## Highlights & Insights
- **Theoretical superiority of rank-dimension masking**: Theorem 1 provides a rigorous mathematical proof that the approximation error upper bound of rank-dimension masks is lower than that of output-dimension masks.
- **Systematic validation of layer priors**: Beyond proposing the hypothesis, the paper validates it through weight-scaling experiments and converts the observation into a differentiable nuclear norm constraint.
- **Elegant introduction of cycle consistency**: The inter-domain cycle consistency idea from CycleGAN is transferred to the LoRA merging problem, representing an elegant cross-domain methodological transfer.
- **Remarkable parameter efficiency**: 0.07M parameters vs. ZipLoRA's 1.33M — a 19× reduction.

## Limitations & Future Work
- The method's design relies heavily on layer prior knowledge specific to the SDXL architecture; transferring to other diffusion models may require re-analysis.
- The Constyle loss increases training time (6.38 min vs. ZipLoRA's 5.48 min).
- Validation is currently limited to text-to-image diffusion models and has not been extended to video or 3D generation.
- Multi-concept scenarios still employ simple uniform-weight merging, which may be suboptimal.

## Related Work & Insights
- Unlike B-LoRA, which identifies specific layers (W4/W5) as responsible for content/style, DuoLoRA adopts a more general layer-resolution prior.
- The cycle-consistent loss idea can be generalized to other concept merging scenarios requiring disentanglement (e.g., multi-subject merging).
- The rank-dimension masking concept may also inspire general model merging approaches such as multi-task LoRA merging.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of rank-dimension masking and cycle-consistent loss is novel, though each individual component is not entirely new in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, user studies, and ablation experiments are all included, but large-scale quantitative evaluation is lacking.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, theoretical proofs are complete, and figures are informative.
- Value: ⭐⭐⭐⭐ The work offers practical value for the content-style personalization field, with a significant improvement in parameter efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[CVPR 2026\] OPAD: Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/opad_adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[ACL 2026\] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](../../ACL2026/model_compression/scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)

</div>

<!-- RELATED:END -->
