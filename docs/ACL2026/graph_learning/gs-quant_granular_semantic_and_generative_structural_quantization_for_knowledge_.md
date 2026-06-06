---
title: >-
  [Paper Note] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion
description: >-
  [ACL 2026][Graph Learning][KGC] GS-Quant quantizes KG entities into "coarse-to-fine" discrete code sequences. By constraining RQ-VAE with a hierarchical clustering tree, shallow codes encode global categories (e.g.…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "KGC"
  - "RQ-VAE"
  - "hierarchical clustering"
  - "codebook"
  - "LLM vocabulary expansion"
date: 2026-05-08
content_hash: f2e74abe6931f753
---

# GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion

**Conference**: ACL 2026  
**arXiv**: [2604.21649](https://arxiv.org/abs/2604.21649)  
**Code**: https://github.com/mikumifa/GS-Quant (Available)  
**Area**: Graph Learning / Knowledge Graph Completion / Quantization  
**Keywords**: KGC, RQ-VAE, hierarchical clustering, codebook, LLM vocabulary expansion

## TL;DR
GS-Quant quantizes KG entities into "coarse-to-fine" discrete code sequences. By constraining RQ-VAE with a hierarchical clustering tree, shallow codes encode global categories (e.g., "Person") while deep codes encode fine-grained attributes (e.g., "Artist"). A GPT-style decoder is used to reconstruct the entity and its ancestors to enforce causal dependencies between codes. These codes are integrated into the LLM vocabulary for LoRA fine-tuning. Experiments on WN18RR and FB15k-237 show that Hits@1 improves by 2.2-2.4 points over the SOTA baseline SSQR.

## Background & Motivation

**Background**: Knowledge Graph Completion (KGC) methods incorporating LLMs are divided into two categories: (1) **text-based** methods linearize triples into natural language prompts (KICGPT, DIFT, KG-FIT, MKGL), which are readable but suffer from token explosion and disrupt graph topology; (2) **embedding-based** methods inject KG embeddings into the LLM latent space (TransE/RotatE + adapter), which are efficient but face a **modality mismatch between continuous embeddings and discrete tokens**.

**Limitations of Prior Work**: (1) Text-based methods use hundreds of tokens per triple, making inference on FB15k-237 inefficient; (2) Embedding methods feed holistic dense vectors into the LLM, whereas LLMs are sequence-to-next-token models, failing to leverage autoregressive strengths; (3) Recent quantization methods (SSQR, ReaLM) quantize entities into code sequences but treat them as **flat numerical compression**—projecting entity embeddings into 4 codes without hierarchical semantic relationships, resembling hash signatures rather than "language."

**Key Challenge**: Human reasoning and LLM generation follow a **coarse-to-fine** hierarchical structure (classification before refinement). In contrast, existing quantization methods flatten all codes and use Euclidean nearest neighbor search, resulting in a **lack of hierarchy and causality between codes**. Consequently, an LLM cannot distinguish which code represents a "category" versus an "instance."

**Goal**: (1) Enable quantized code sequences to be **hierarchical**—shallow codes encode coarse-grained categories and deep codes encode fine-grained attributes; (2) Establish **generative causal dependencies** between codes rather than independence; (3) Verify that injecting these "structured codes" into the LLM vocabulary for KGC is more effective than flat codes.

**Key Insight**: RQ-VAE (Residual Quantization) naturally possesses a numerical recursive hierarchy $\mathbf{r}_{l+1} = \mathbf{r}_l - \mathbf{v}_{c_l}^l$. However, the authors argue that numerical hierarchy is insufficient and **explicit semantic hierarchy** must be injected. The approach involves building a hierarchy tree $\mathcal{H}$ via agglomerative clustering on entity embeddings and aligning each layer of RQ to the corresponding layer of the tree.

**Core Idea**: **Granular Semantic Enhancement (GSE)** injects hierarchical clustering trees into codebook learning; **Generative Structural Reconstruction (GSR)** utilizes a GPT-style decoder to reconstruct entities and ancestors with causal dependencies; both are jointly trained with the RQ commitment loss.

## Method

### Overall Architecture
1. **Preprocessing**: Generate entity relation embeddings $\mathbf{s}_x^{\mathcal{G}}$ using RotatE and text embeddings $\mathbf{s}_x^T$ using a PLM for entity names and descriptions. Fuse them as $\mathbf{s}_x = \rho \mathbf{s}_x^{\mathcal{G}} + (1-\rho) \mathbf{s}_x^T$. Construct a hierarchy tree $\mathcal{H}$ using agglomerative clustering on $\{\mathbf{s}_x\}$.
2. **Quantization Training**: Map $\mathbf{s} \to \mathbf{z} = \text{MLP}(\mathbf{s})$ as $\mathbf{r}_0$. Perform RQ along $m$ codebook layers, selecting $c_l = \arg\min_k \|\mathbf{r}_l - \mathbf{v}_k^l\|_2$ and updating residuals $\mathbf{r}_{l+1} = \mathbf{r}_l - \mathbf{v}_{c_l}^l$. The final output is a code tuple $\mathcal{I} = \{c_i\}_{i=0}^{m-1}$.
3. **Joint Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_Q + \mathcal{L}_{\text{GSE}} + \mathcal{L}_{\text{GSR}}$, where $\mathcal{L}_Q$ is standard commitment loss, $\mathcal{L}_{\text{GSE}}$ injects hierarchical semantics, and $\mathcal{L}_{\text{GSR}}$ adds causal dependencies.
4. **Checkpoint Selection**: Select the model based on maximum Codebook Entropy $\mathcal{Y} = -\frac{1}{M}\sum_m \sum_k p_k^m \log p_k^m$ to avoid codebook collapse.
5. **LLM Fine-tuning**: Add all codes from the codebook as new tokens to the LLM vocabulary. Freeze LLM parameters and only train code embeddings and LoRA adapters (attention + FFN) to select the correct tail/head entity from a KGC candidate list.

### Key Designs

1. **Granular Semantic Enhancement (GSE) —— Injecting clustering trees into the codebook**:

    - **Function**: Aligns the $i$-th layer of RQ code with the clustering center $\boldsymbol{\mu}_e$ of the $i$-th layer of the hierarchy tree, forcing shallow layers to learn coarse concepts and deep layers to learn fine details.
    - **Mechanism**: Calculates a differentiable surrogate $\tilde{\mathbf{v}}_i = \mathbf{r}_i + \operatorname{sg}[\mathbf{v}_{c_i}^i - \mathbf{r}_i]$ to allow gradients through discrete selection. Two contrastive losses are applied:
        - **Coarse-to-Fine Alignment** $\mathcal{L}_1$: Pulls $\tilde{\mathbf{v}}_i$ closer to its cluster center $\boldsymbol{\mu}_e$ at layer $i$, with weight $\lambda_1^{i+1}/m$ ($\lambda_1 \in (0,1)$ decaying exponentially). Higher weights for shallow layers prioritize global category learning.
        - **Hierarchical Separability** $\mathcal{L}_2$: Pushes $\tilde{\mathbf{v}}_i$ away from neighboring centers $\mathbf{n} \in \mathcal{N}_e$ in the tree, with weight $\lambda_2^{m-i}/m$ (reverse decay). Higher weights for deep layers prioritize fine-grained discrimination.
    - **Design Motivation**: Vanilla RQ-VAE layers are numerically recursive but lack semantics, mixing coarse and fine concepts (Fig 4b). GSE constrains the codebook to "aggregate similarities" in shallow layers and "separate differences" in deep layers. Visualization (Fig 4a) shows shallow codes are sparse and global, while deep codes are dense and discriminative, matching human intuition.

2. **Generative Structural Reconstruction (GSR) —— Adding causal dependency via GPT decoder**:

    - **Function**: Transforms $\mathcal{I}$ from "independent codes" into an "ordered, causally dependent semantic sentence," where different combinations encode different semantics.
    - **Mechanism**: Constructs learnable query embeddings $\mathcal{Q} = \{\mathbf{q}_i\}_{i=0}^L$, concatenates $\tilde{\mathbf{v}}$, and feeds them into a Transformer decoder (causal self-attention). Outputs $\{\mathbf{o}_i\}$ target different reconstruction goals: $\mathbf{o}_0$ reconstructs the entity $\mathbf{s}$, and $\{\mathbf{o}_i\}_{i \ge 1}$ reconstruct ancestors $\{\mathbf{h}_i\}$ in $\mathcal{H}$. $\mathcal{L}_{\text{GSR}} = \|\tilde{\mathbf{o}}_0 - \mathbf{s}\|_2^2 + \lambda_s \|\tilde{\mathbf{o}}_1 - \mathbf{h}_0\|_2^2 + \lambda_h \sum_{i=2}^L \|\tilde{\mathbf{o}}_i - \mathbf{h}_{i-1}\|_2^2$, where $\lambda_s \ll \lambda_h$.
    - **Design Motivation**: (i) Causal attention forces code$_l$ to be conditioned on code$_{<l}$, forming a coarse-to-fine dependency isomorphic to LLM autoregression; (ii) Reconstructing ancestors prevents collapse by ensuring code sequences retain multi-granular information; (iii) Ablations show performance drops (0.8% - 1.1%) without GSR, validating that causal dependency makes codes more "language-like."

3. **Codebook Entropy for Checkpoint Selection + Vocab Extension + LoRA**:

    - **Function**: (i) Prevents collapse via codebook utilization signals; (ii) Injects $M \times K$ codes as tokens; (iii) Trains only new embeddings and LoRA to retain general capabilities.
    - **Mechanism**: Codebook entropy $\mathcal{Y} = -\frac{1}{M}\sum_m \sum_k p_k^m \log p_k^m$ is maximized when codes are uniformly activated ($p_k^m = 1/K$). Selecting checkpoints by $\mathcal{Y}$ maximizes expressive power. Table 3 shows $\mathcal{Y}$ correlates positively with MRR/Hits@K.
    - **Design Motivation**: (i) Addresses RQ-VAE collapse (dead codes) using entropy as a simple monitoring signal; (ii) LoRA and new token embeddings allow the LLM to learn KG knowledge without catastrophic forgetting; (iii) During inference, the LLM selects from candidates using templates consistent with DIFT for fair comparison.

### Loss & Training
- **Quantization Stage**: $\mathcal{L}_{\text{total}} = \mathcal{L}_Q + \mathcal{L}_{\text{GSE}} + \mathcal{L}_{\text{GSR}}$, selecting checkpoints via $\mathcal{Y}$.
- **LLM Stage**: Freeze original LLM parameters; update only (i) new code token embeddings and (ii) LoRA adapters. The objective is the language model loss for candidate selection.
- Key Hyperparameters: $\lambda_1 = 0.8, \lambda_2 = 0.4$, small $\lambda_s$, large $\lambda_h$, and $\rho$ to balance graph/text fusion.
- Training Overhead: GSE and GSR contribute 4%-18% additional overhead, which is acceptable.

## Key Experimental Results

### Main Results

KGC main results on WN18RR and FB15k-237 (alignment with DIFT candidates and templates):

| Method | WN18RR MRR | WN18RR H@1 | WN18RR H@3 | WN18RR H@10 | FB15k-237 MRR | FB15k-237 H@1 | FB15k-237 H@3 | FB15k-237 H@10 |
|---|---|---|---|---|---|---|---|---|
| TransE | 0.243 | 0.043 | 0.441 | 0.532 | 0.279 | 0.198 | 0.376 | 0.441 |
| RotatE | 0.476 | 0.428 | 0.492 | 0.571 | 0.338 | 0.241 | 0.375 | 0.533 |
| CompGCN | 0.479 | 0.443 | 0.494 | 0.546 | 0.355 | 0.264 | 0.390 | 0.535 |
| MEM-KGC | 0.557 | 0.475 | 0.604 | 0.704 | 0.346 | 0.253 | 0.381 | 0.531 |
| CoLE | 0.593 | 0.538 | 0.616 | 0.701 | 0.389 | 0.294 | 0.429 | 0.572 |
| KICGPT | 0.564 | 0.478 | 0.612 | 0.677 | 0.412 | 0.327 | 0.448 | 0.581 |
| DIFT (Strong LLM baseline) | 0.617 | 0.569 | 0.638 | 0.708 | 0.439 | 0.364 | 0.468 | 0.586 |
| MKGL | 0.552 | 0.500 | 0.577 | 0.656 | 0.415 | 0.325 | 0.454 | 0.591 |
| SSQR (Prior quantization) | 0.603 | 0.553 | 0.627 | 0.692 | 0.428 | 0.349 | 0.459 | 0.583 |
| **GS-Quant (Ours)** | **0.635** | **0.594** | **0.649** | **0.712** | **0.455** | **0.386** | **0.479** | **0.592** |

Vs DIFT: WN18RR Hits@1 +2.5, FB15k-237 Hits@1 +2.2; vs SSQR: WN18RR Hits@1 +4.1, FB15k-237 Hits@1 +3.7.

### Ablation Study

| Configuration | FB15k-237 MRR | FB15k-237 H@1 | WN18RR MRR | WN18RR H@1 | Notes |
|---|---|---|---|---|---|
| **Ours (full)** | **0.455** | **0.386** | **0.635** | **0.594** | Baseline |
| w/o $\mathcal{L}_1$ | 0.450 (-0.5%) | 0.377 (-0.9%) | 0.629 (-0.5%) | 0.587 (-0.6%) | GSE partially disabled |
| w/o $\mathcal{L}_2$ | 0.450 (-0.5%) | 0.379 (-0.7%) | 0.625 (-0.9%) | 0.577 (-1.6%) | GSE partially disabled |
| w/o $\mathcal{L}_{\text{GSR}}$ | 0.448 (-0.7%) | 0.375 (-1.1%) | 0.627 (-0.7%) | 0.585 (-0.8%) | No causal dependency |
| **w/o Code** | **0.404 (-5.1%)** | **0.303 (-8.3%)** | **0.607 (-2.7%)** | **0.541 (-5.2%)** | Quantized tokens are the main gain |

## Key Findings
- **Quantized code is the primary contributor**: Performance drops 8.3 points on FB15k-237 without codes, proving that injecting KG knowledge as discrete tokens is the core mechanism.
- **GSE and GSR contribute ~1 point each**: While individual losses yield incremental gains (0.5-1.6%), their synergy allows GS-Quant to outperform SSQR by ~4 points in Hits@1.
- **Hyperparameter Robustness**: Performance is stable within $\lambda_1 \in [0.6, 0.9]$ and $\lambda_2 \in [0.2, 0.5]$, with $\lambda_s < \lambda_h$ being consistently optimal.
- **Codebook entropy is a valid selection signal**: The correlation between $\mathcal{Y}$ and downstream metrics allows for best-model selection using self-supervised indicators.
- **Disentanglement validation**: Visualization (Fig 4a) confirms that GSE successfully forces codes to learn hierarchical semantics (sparse shallow codes vs. dense deep codes).
- **Efficiency**: Training time increases only slightly (4-18%) compared to vanilla RQ-VAE, indicating the overhead is not prohibitive.

## Highlights & Insights
- **"Structuring codes like language" is the core philosophy**: Quantization for KGC is redefined from "embedding compression" to "creating a new sub-language for LLMs." This requires codes to possess linguistic properties: hierarchy (coarse-to-fine), causality (left-to-right), and compositionality.
- **Clustering trees as differentiable inductive bias**: Using offline-generated hierarchy trees to guide end-to-end quantization avoids training instability caused by simultaneous structure discovery. This is applicable to any domain requiring hierarchical quantization.
- **Entropy as a portable monitoring trick**: This simple selection strategy can be applied to any VQ-VAE/RQ-VAE framework to mitigate codebook collapse.
- **Multitarget GSR**: Reconstructing hierarchy ancestors rather than just next tokens elegantly unifies sequential generation with hierarchical abstraction.
- **Vocabulary expansion + Frozen backbone**: Training only new embeddings and LoRA adapts the model to specific domains while preserving its general capabilities, providing a recipe for other specialized discrete representations (e.g., molecules, protein sequences).

## Limitations & Future Work
- **Dependency on external KG embeddings**: GS-Quant uses RotatE for $\mathbf{s}^{\mathcal{G}}$; the quantization ceiling is constrained by the quality of the initial backbone.
- **Static hierarchy trees**: Agglomerative clustering is performed once before training and cannot dynamically adapt during the quantization process.
- **Scale validation**: Only WN18RR and FB15k-237 were tested. Scalability to industrial-scale KGs with millions of entities remains unverified.
- **Token cost analysis missing**: While emphasizing efficiency over text-based methods, the paper does not quantify the specific reduction in token usage.
- **Future Directions**: (1) End-to-end learning of the hierarchy tree (differentiable clustering); (2) Scaling studies on Wikidata/NELL; (3) Applying quantized codes to downstream QA or reasoning tasks.

## Related Work & Insights
- **Vs SSQR**: SSQR uses flat VQ with FFN projections. GS-Quant improves Hits@1 by 4.1 points on WN18RR, demonstrating that structural quantization is superior to flat numerical compression.
- **Vs ReaLM**: ReaLM uses RQ without semantic hierarchy. GS-Quant proves that codebook design is the bottleneck for KGC.
- **Vs DIFT**: DIFT injects continuous embeddings. GS-Quant's superior performance (0.594 vs 0.569 Hits@1) suggests that discrete tokens are better aligned with the LLM's generative dynamics.
- **Vs RQ-VAE**: While the original RQ-VAE was designed for image generation, GS-Quant adapts it to structured graph data by injecting semantic alignment.
- **Vs Text-based methods**: GS-Quant's compression of entities into $m$ tokens significantly reduces context length compared to full text descriptions while achieving higher accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ The "code-as-language" framing combined with GSE + GSR is a novel and coherent design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations and visualizations, though lacks large-scale dataset verification.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and convincing visualizations; some slightly redundant terminology regarding "coarse-to-fine."
- Value: ⭐⭐⭐⭐ Open-sourced results and SOTA performance provide a practical reference for LLM-based KGC and other discrete representation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion](collaboration_of_fusion_and_independence_hypercomplex-driven_robust_multi-modal_.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[NeurIPS 2025\] Generative Graph Pattern Machine](../../NeurIPS2025/graph_learning/generative_graph_pattern_machine.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)

</div>

<!-- RELATED:END -->
