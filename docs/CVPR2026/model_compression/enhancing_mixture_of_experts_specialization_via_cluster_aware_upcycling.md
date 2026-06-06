---
title: >-
  [Paper Note] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling
description: >-
  [CVPR 2026][Model Compression][Mixture-of-Experts] This paper proposes Cluster-aware Upcycling, which extracts semantic structure from a dense model via spherical k-means clustering to initialize expert and router parame…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Mixture-of-Experts"
  - "sparse upcycling"
  - "expert specialization"
  - "cluster initialization"
  - "self-distillation"
date: 2026-05-08
content_hash: b25644eeb99343ec
---

# Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling

**Conference**: CVPR 2026
**arXiv**: [2604.13508](https://arxiv.org/abs/2604.13508)  
**Code**: N/A  
**Area**: Model Compression / Efficient Models
**Keywords**: Mixture-of-Experts, sparse upcycling, expert specialization, cluster initialization, self-distillation

## TL;DR

This paper proposes Cluster-aware Upcycling, which extracts semantic structure from a dense model via spherical k-means clustering to initialize expert and router parameters in MoE, thereby breaking expert symmetry and promoting early specialization. Combined with an Expert Ensemble Self-Distillation (EESD) loss, the method consistently outperforms existing upcycling approaches on CLIP ViT benchmarks.

## Background & Motivation

- **Background**: Sparse Upcycling initializes MoE by copying pretrained dense model weights, avoiding the high computational cost of training from scratch.
- **Limitations of Prior Work**: Since all experts start from identical weights and routers are randomly initialized, expert symmetry and limited early specialization are inherent problems. Existing symmetry-breaking strategies include noise injection (limited effectiveness) and partial re-initialization (which disrupts pretrained representations), both of which are suboptimal.
- **Key Challenge**: The representations of a pretrained dense model already encode semantic information that can effectively guide expert and router initialization, yet prior methods fail to leverage this structure.
- **Goal**: Exploit the latent semantic structure in dense model activations to initialize both experts and routers in a principled, data-aware manner.

## Method

### Overall Architecture

The proposed method follows a three-step strategy: (1) spherical k-means clustering partitions the activation space of the dense model into semantic clusters; (2) data-aware truncated SVD initializes each expert into the subspace corresponding to its assigned cluster; (3) cluster centroids initialize the router weights. During training, an Expert Ensemble Self-Distillation (EESD) loss provides stable supervision.

### Key Designs

1. **Spherical K-Means on Activation Space**: Cosine similarity is adopted as the clustering objective, directly aligned with the router logit computation ($\mathbf{W}_r \mathbf{x}$ is inherently a directional alignment measure). Input activation vectors from each FFN block are extracted from a calibration dataset and clustered into $N_e$ groups with corresponding centroids.

2. **Data-Aware Truncated SVD for Expert Initialization**: A Cholesky whitening matrix is computed for the activations of each cluster, and truncated SVD is performed in the whitened space, retaining a fraction $\tau$ of the spectral energy. This ensures each expert preserves the principal subspace information under its corresponding cluster's data distribution, while a cross-penalty term discourages experts from collapsing to similar solutions.

3. **Expert Ensemble Self-Distillation (EESD)**: An EMA teacher operating in dense mode (activating all experts) provides stable predictions, offering reliable supervision for tokens with uncertain routing decisions. The sparse MoE student aligns with the teacher's ensemble predictions via the distillation loss.

### Loss & Training

The training objective comprises a task loss $\mathcal{L}_{task}$ (e.g., contrastive loss), a load balancing loss $\mathcal{L}_{lb}$, and the EESD distillation loss. Initializing routers with cluster centroids ensures that early routing decisions are aligned with the semantic structure of the data.

## Key Experimental Results

### Main Results

Evaluated on CLIP ViT-B/32 and ViT-B/16 across zero-shot retrieval and classification benchmarks:

| Benchmark | Sparse Upcycling | Drop-Upcycling | Cluster-aware (Ours) |
|-----------|-----------------|----------------|----------------------|
| MSCOCO I→T R@1 | Baseline | Limited gain | **Best** |
| ImageNet-1k Val | Baseline | Limited gain | **Best** |
| VTAB Natural | Baseline | Limited gain | **Best** |

### Ablation Study

- Cluster initialization vs. random initialization: significantly reduces inter-expert similarity.
- EESD loss yields the greatest improvement for tokens with uncertain routing.
- Data-aware SVD vs. standard SVD: the former better preserves cluster-specific information.

### Key Findings

- Inter-expert similarity is substantially reduced, yielding more diverse and disentangled representations.
- Routing behavior becomes more confident, with more deterministic token assignments.
- Structural improvements directly translate into gains on zero-shot and few-shot generalization benchmarks.

## Highlights & Insights

- Leveraging existing semantic structure to break symmetry—rather than relying on noise or random perturbations—is an elegant and principled design choice.
- The consistent alignment between spherical k-means and the cosine-based router computation reflects careful and coherent design.
- Quantitative analyses confirm that the structural improvements genuinely drive performance gains, rather than being attributable solely to training tricks.

## Limitations & Future Work

- Validation is limited to CLIP ViT-B; the effectiveness on larger-scale models (ViT-L/H) remains unexplored.
- The clustering granularity is tied to the number of experts, limiting flexibility.
- The impact of calibration dataset selection and size on clustering quality is not thoroughly analyzed.

## Related Work & Insights

- The data-aware SVD initialization strategy is generalizable to other model scaling scenarios.
- The dense-teacher–sparse-student distillation paradigm of EESD is applicable to broader MoE training settings.
- Initializing routers with cluster centroids is a simple yet effective approach worth broader adoption.

## Rating

7/10 — The method is elegantly designed with thorough analysis, but the experimental scope is limited (ViT-B only).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](../../ICML2026/model_compression/dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)

</div>

<!-- RELATED:END -->
