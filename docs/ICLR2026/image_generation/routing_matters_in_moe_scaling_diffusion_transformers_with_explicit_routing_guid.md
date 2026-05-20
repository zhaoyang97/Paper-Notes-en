---
title: >-
  [Paper Note] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance
description: >-
  [ICLR 2026][Image Generation][Mixture-of-Experts] This paper proposes ProMoE, an MoE framework for Diffusion Transformers that introduces a two-stage router (conditional routing + prototype routing) and a routing contras…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Mixture-of-Experts"
  - "DiT"
  - "Explicit Routing Guidance"
  - "Prototype Routing"
  - "Routing Contrastive Loss"
date: 2026-05-08
content_hash: 5617eb04f4236c61
---

# Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance

**Conference**: ICLR 2026
**arXiv**: [2510.24711](https://arxiv.org/abs/2510.24711)  
**Code**: [https://github.com/ali-vilab/ProMoE](https://github.com/ali-vilab/ProMoE)  
**Area**: Diffusion Models / Mixture of Experts
**Keywords**: Mixture-of-Experts, DiT, Explicit Routing Guidance, Prototype Routing, Routing Contrastive Loss

## TL;DR

This paper proposes ProMoE, an MoE framework for Diffusion Transformers that introduces a two-stage router (conditional routing + prototype routing) and a routing contrastive loss to provide explicit semantic guidance, promoting expert specialization and significantly outperforming existing MoE and dense models on ImageNet.

## Background & Motivation

While MoE has achieved remarkable success in LLMs, its performance in DiTs has been disappointing:
- DiT-MoE (token-choice routing) performs even worse than dense models
- EC-DiT (expert-choice routing) yields only marginal improvements
- DiffMoE (global token distribution routing) also offers limited gains

**Root Cause Analysis**: Visual tokens differ fundamentally from language tokens:

**High Spatial Redundancy**: Visual tokens are continuous, spatially coupled, and highly redundant (inter/intra-class distance ratio of only 0.748 vs. 19.283 for LLMs), causing experts to learn homogeneous features.

**Functional Heterogeneity**: CFG introduces two functionally distinct input types — conditional and unconditional — which naive MoE fails to differentiate.

## Method

### Overall Architecture

ProMoE comprises a two-stage router combined with routing contrastive learning, aiming to promote:
- **Intra-expert consistency**: each expert consistently processes similar patterns
- **Inter-expert diversity**: different experts specialize in distinct tasks

### Key Design 1: Conditional Routing

Hard routing partitioning based on the functional role of tokens:
- **Unconditional tokens** (image patches under empty labels/text) → $N_u$ unconditional experts
- **Conditional tokens** (image patches under specific conditions) → routing experts (determined in the second stage)

Forward pass:

$$\text{MoE}(\mathbf{x}) = \underbrace{\sum_{i=1}^{N_s} E_i^S(\mathbf{x})}_{\text{Shared}} + \begin{cases}\sum_{j=1}^{N_E}\mathbf{G}_j E_j(\mathbf{x}) & \mathbf{x} \in \mathbf{X}_c \\ \sum_{k=1}^{N_u}E_k^U(\mathbf{x}) & \mathbf{x} \in \mathbf{X}_u\end{cases}$$

### Key Design 2: Prototype Routing

Learnable prototypes $\mathbf{P} \in \mathbb{R}^{N_E \times D}$ are introduced, with each prototype corresponding to one expert. Tokens are assigned via cosine similarity:

$$\mathbf{Z}_{i,j} = \alpha \frac{\mathbf{x}_i \mathbf{p}_j^\top}{\|\mathbf{x}_i\| \|\mathbf{p}_j\|}$$

The identity function $\mathcal{A}(\mathbf{Z}) = \mathbf{Z}$ is used as the activation (outperforming softmax and sigmoid).

### Key Design 3: Routing Contrastive Loss

Explicitly enhances the semantic guidance of prototype routing by pulling each prototype toward the centroid of its positive set and pushing it away from negative centroids:

$$\mathcal{L}_{\text{RCL}} = -\frac{1}{N_a}\sum_{i=1}^{N_a}\log\frac{\exp(\text{sim}(\mathbf{p}_i, \mathbf{m}_i)/\tau)}{\sum_{j=1}^{N_a}\exp(\text{sim}(\mathbf{p}_i, \mathbf{m}_j)/\tau)}$$

where $\mathbf{m}_i$ is the centroid of tokens assigned to expert $E_i$. The repulsion term in RCL also serves as an implicit load balancing mechanism.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{RCL}} \mathcal{L}_{\text{RCL}}$$

## Key Experimental Results

### Main Results: Comparison with Dense Models (Rectified Flow, 500K steps)

| Model | Active Params | Total Params | FID↓ (cfg=1.0) | FID↓ (cfg=1.5) |
|-------|--------------|-------------|---------------|---------------|
| Dense-DiT-B | 130M | 130M | 30.61 | 9.02 |
| ProMoE-B | 130M | 300M | 24.44 | 6.39 |
| Dense-DiT-L | 458M | 458M | 15.44 | 3.56 |
| ProMoE-L | 458M | 1.063B | 11.61 | 2.79 |
| Dense-DiT-XL | 675M | 675M | 13.38 | 3.23 |
| ProMoE-XL | 675M | 1.568B | 9.44 | 2.59 |

ProMoE-L surpasses Dense-DiT-XL (675M) while using fewer active parameters (458M).

### Ablation Study: Semantic Guidance Verification

| Method | FID↓ (cfg=1.5) | IS↑ |
|--------|---------------|-----|
| Dense-DiT-B | 9.02 | 131.13 |
| DiT-MoE-B | 8.94 | 131.66 |
| DiffMoE-B | 8.22 | 137.46 |
| Classification Routing Guidance | **5.91** | **165.45** |
| K-Means Routing Guidance | 6.24 | 159.77 |

Both explicit and implicit semantic guidance yield significant improvements, confirming that visual MoE requires semantic guidance.

### Comparison with MoE Baselines

ProMoE outperforms DiT-MoE, EC-DiT, and DiffMoE across all scales and training paradigms (DDPM/RF).

### Key Findings

- The core bottleneck of visual MoE is expert homogenization (expert subspaces are highly similar without guidance)
- Conditional routing effectively eliminates routing interference caused by functional heterogeneity
- RCL requires no manual labels, is more flexible than classification loss, and more robust than K-Means
- The repulsion term in RCL naturally replaces conventional load balancing losses

## Highlights & Insights

- Provides an in-depth analysis of the root causes behind the performance gap between visual and language MoE
- The two-stage routing + contrastive loss design is concise and effective, and generalizable to other visual MoE frameworks
- Strong parameter efficiency: fewer active parameters surpass larger dense models
- Validated under both DDPM and Rectified Flow paradigms

## Limitations & Future Work

- Evaluation is limited to class-conditional ImageNet; more complex scenarios such as text-to-image generation are not explored
- Conditional routing requires CFG inference and is not applicable to settings without CFG
- Computational overhead of clustering/contrastive learning is not analyzed in detail
- Total parameter count is approximately 2.3× that of the dense counterpart

## Related Work & Insights

- **DiT MoE**: Prior attempts at visual MoE, including DiT-MoE, EC-DiT, and DiffMoE
- **LLM MoE**: Successful applications in the language domain, such as DeepSeek-MoE and Mixtral
- **Diffusion Models**: Transformer-based diffusion models including DiT and SiT

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — In-depth analysis; two-stage routing + RCL combination is novel
- **Technical Rigor**: ⭐⭐⭐⭐ — Rigorous experimental design with thorough ablations
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scale, multi-paradigm validation
- **Impact**: ⭐⭐⭐⭐⭐ — Charts a clear direction for visual MoE research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](../../CVPR2026/image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](../../NeurIPS2025/image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[CVPR 2026\] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation](../../CVPR2026/image_generation/mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
