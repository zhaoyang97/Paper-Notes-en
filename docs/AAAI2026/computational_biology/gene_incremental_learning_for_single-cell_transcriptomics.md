---
title: >-
  [Paper Note] Gene Incremental Learning for Single-Cell Transcriptomics
description: >-
  [AAAI 2026][Computational Biology][gene incremental learning] This paper proposes a Gene Incremental Learning (GIL) framework that leverages the permutation-invariant nature of single-cell transcriptomics data to extend…
tags:
  - "AAAI 2026"
  - "Computational Biology"
  - "gene incremental learning"
  - "single-cell transcriptomics"
  - "catastrophic forgetting"
  - "token incremental learning"
  - "benchmark"
date: 2026-05-08
content_hash: 7cd996927bffb61b
---

# Gene Incremental Learning for Single-Cell Transcriptomics

**Conference**: AAAI 2026
**arXiv**: [2511.13762](https://arxiv.org/abs/2511.13762)  
**Code**: [GitHub](https://github.com/simpleshinobu/scbenchmark)  
**Area**: Bioinformatics / Incremental Learning
**Keywords**: gene incremental learning, single-cell transcriptomics, catastrophic forgetting, token incremental learning, benchmark

## TL;DR

This paper proposes a Gene Incremental Learning (GIL) framework that leverages the permutation-invariant nature of single-cell transcriptomics data to extend the class incremental learning (CIL) paradigm to the token (gene) dimension. Two baseline methods—gene replay and gene distillation—are designed, and a comprehensive benchmark is established with two evaluation protocols: gene-level regression and gene-level classification.

## Background & Motivation

Class incremental learning (CIL) has been extensively studied in computer vision, with catastrophic forgetting being the central challenge when a model continuously learns new categories. However, tokens—as fundamental units in many domains (NLP, bioinformatics)—also exhibit continuous growth (e.g., new words are coined, new genes are discovered), yet incremental learning in the token dimension has been largely overlooked.

This gap stems primarily from the holistic nature of language data: if words are partitioned across stages (e.g., the word "learning" is absent in a given stage), one must either exclude all texts containing that word (drastically reducing data) or remove the word from existing texts (distorting semantics), rendering both approaches infeasible.

Single-cell transcriptomics data does not suffer from this limitation. In transcriptomics, genes serve as tokens; each sample is represented by a set of gene expression values (analogous to a sentence), and genes carry no relative ordering, allowing free partitioning and permutation. This property makes it possible to construct a gene incremental learning framework.

**Core Idea**: The permutation invariance of transcriptomics data is exploited to circumvent the indivisibility of language data, establishing the first token incremental learning framework. Gene replay and gene distillation are introduced to demonstrate that forgetting analogous to CIL also occurs in this setting.

## Method

### Overall Architecture

The GIL pipeline partitions all genes into a base gene set $B$ (present at every stage) and stage-specific gene sets $T^{s_1}, T^{s_2}, \ldots, T^{s_n}$ (mutually disjoint across stages). Datasets are likewise independently partitioned into $\mathcal{D}^{s_1}, \ldots, \mathcal{D}^{s_n}$. At stage $k$, the model observes only samples containing the base genes and $T^{s_k}$, and is subsequently evaluated on all genes seen so far.

Gene learning is realized via masked value prediction: a subset of gene expression values is randomly masked, and the model is trained to predict the masked values, with training loss $\mathcal{L}_{\text{tran}}(\mathcal{D}, \phi) = \frac{1}{N}\sum_{i=1}^{N}\sum_j \|v_{ij} - \hat{v}_{ij}\|^2$.

### Key Designs

1. **Base Gene Mechanism**:

    - **Function**: A subset of genes is designated as base genes $B$ and appears in every incremental stage.
    - **Mechanism**: Unlike class labels, individual genes cannot independently convey the full semantics of a sample; each gene is merely one component. Base genes ensure that samples at each stage remain biologically meaningful in the transcriptomics context.
    - **Design Motivation**: In CIL, classes and samples correspond directly. In GIL, genes and samples are misaligned—each sample encompasses all genes. Base genes mitigate the risk of producing semantically vacuous samples due to insufficient gene coverage.

2. **Gene Replay**:

    - **Function**: A subset of samples from previous stages is retained and jointly used during training at the current stage.
    - **Mechanism**: $\mathcal{L}_{\text{dr},s_k} = \mathcal{L}_{\text{tran}}(\mathcal{D}^{s_k}, \phi) + \sum_{i=1}^{k-1}\mathcal{L}_{\text{tran}}(\mathcal{D}_{\text{dr}}^{s_i}, \phi)$, where $\mathcal{D}_{\text{dr}}^{s_i} \subset \mathcal{D}^{s_i}$ denotes the retained subset from stage $i$.
    - **Design Motivation**: Directly adapted from data replay strategies in CIL. Performance approaches the Oracle as the replay buffer size increases.

3. **Gene Distillation**:

    - **Function**: Knowledge distillation is applied using the optimal model $\phi_{s_{k-1}}^*$ from the previous stage to regularize the current model.
    - **Mechanism**: $\mathcal{L}_{\text{fd},s_k} = \frac{1}{N_k}\sum_{i=1}^{N_k}(\sum_j \|v_{ij} - \hat{v}_{ij}\|^2 + \lambda\|\hat{\bm{v}}_i - \hat{\bm{v}}_{i,s_{k-1}}^*\|^2)$, augmenting the masked prediction loss with an imitation loss on the outputs of the previous model.
    - **Design Motivation**: Adapted from knowledge distillation in CIL, assuming the previous model can characterize current samples via base genes. New genes introduced at the current stage are excluded from the distillation term, as the previous model has no capacity to predict unseen genes.

4. **Feature Extraction**:

    - Gene embedding: $\bm{e} = \mathbf{E}_\phi(\bm{x}) + \tilde{\bm{v}}\mathbf{L}_{1,\phi}$
    - Transformer encoding: $\bm{e}' = \mathbf{M}_\phi(\bm{e})$
    - Value prediction: $\hat{\bm{v}} = \bm{e}'\mathbf{L}_{2,\phi}$
    - The gene embedding layer maps token IDs to vectors; a linear layer encodes expression values into the embedding space; their sum is fed into the Transformer.

### Evaluation Protocols

1. **Gene-Level Regression**: Directly applies the masked prediction loss $\mathcal{L}_{\text{regress},s_k} = \mathbb{E}[\sum_k \|v_{ik} - \hat{v}_{ik}^*\|^2]$ to assess prediction accuracy for genes specific to a given stage.
2. **Gene-Level Classification**: For each stage, genes critical to a specific downstream classification task are selected; downstream classification accuracy serves as an indirect measure of gene learning and forgetting.

## Key Experimental Results

### Main Results

A scGPT-style Transformer (6 layers, 8 heads, hidden dimension 256) is trained on 906,890 samples.

**2-Stage Gene-Level Regression (Norman–Lupus setting)**:

| Method | Stage | Norman | Lupus | Δ (Forgetting) |
|--------|-------|--------|-------|----------------|
| Baseline | 1 | 0.172 | - | - |
| Baseline | 2 | 0.424 | 0.134 | 0.253 |
| Replay (1000) | 2 | 0.215 | 0.134 | 0.043 |
| Distill (λ=5) | 2 | 0.365 | 0.139 | 0.193 |
| Oracle | - | 0.173 | 0.136 | - |

**2-Stage Gene-Level Classification (Downstream Accuracy %)**:

| Method | Stage | Norman | Lupus | Δ (Forgetting) |
|--------|-------|--------|-------|----------------|
| Baseline | 1 | 37.73 | 67.31 | - |
| Baseline | 2 | 35.59 | 75.39 | -2.14 |
| Replay | 2 | 36.45 | 75.00 | -1.29 |
| Distill | 2 | 34.16 | 72.94 | -3.74 |
| Oracle | - | 38.11 | 75.42 | - |

### Ablation Study

**Hyperparameter Ablation for Gene Replay and Gene Distillation (Norman–Lupus Regression)**:

| Method | Hyperparameter | Norman | Lupus | Δ |
|--------|---------------|--------|-------|---|
| Replay | 50 samples | 0.293 | 0.136 | 0.121 |
| Replay | 100 samples | 0.263 | 0.138 | 0.091 |
| Replay | 1000 samples | 0.215 | 0.134 | 0.043 |
| Replay | 10000 samples | 0.190 | 0.133 | 0.018 |
| Distill | λ=0.5 | 0.420 | 0.134 | 0.248 |
| Distill | λ=5.0 | 0.365 | 0.139 | 0.193 |
| Distill | λ=10.0 | 0.326 | 0.143 | 0.154 |

### Key Findings
- The Baseline exhibits an average regression Δ of 0.279 and an average classification Δ of −1.816% in the 2-stage setting, confirming that gene forgetting is a genuine phenomenon.
- Gene replay improves monotonically with buffer size; the best Δ of 0.018 (10,000 samples) closely approaches the Oracle.
- Gene distillation is effective under regression evaluation (Δ reduced from 0.253 to 0.193) but counterproductive under classification evaluation (Δ deteriorates from −1.816% to −2.473%), suggesting that while distillation reduces average forgetting, it may degrade the quality of learned gene representations.
- Forgetting accumulates across stages in the 3-stage setting.
- Both evaluation protocols consistently show performance degradation as stages progress, though the decline in classification is less pronounced than typically observed in CIL.

## Highlights & Insights
- The paper is the first to conceptualize token-dimension incremental learning, elegantly exploiting the permutation invariance of transcriptomics data to circumvent the indivisibility of language data.
- The base gene mechanism is a concise yet critical design that preserves the semantic integrity of samples at each stage.
- The two evaluation protocols are complementary: regression directly quantifies the degree of forgetting, while classification validates gene memory quality from a downstream-task perspective.
- The finding that gene distillation performs well on regression but poorly on classification is noteworthy, revealing a distinctive aspect of token-level forgetting.

## Limitations & Future Work
- The observed forgetting is weaker than in CIL (classification drop of only 1–3%), because the contribution of any single gene to the overall sample representation is limited.
- No GIL-specific methods are proposed; only existing CIL strategies are adapted, which limits methodological novelty.
- The number of constructible settings is restricted—critical genes for different downstream datasets overlap substantially, making it difficult to design many non-conflicting incremental settings.
- The number of stages is also limited (only 2–3 stages are tested), leaving longer incremental sequences unvalidated.
- The strategy for selecting base genes may significantly affect framework behavior, yet this aspect is not analyzed in depth.
- Input length is capped at 512 genes selected randomly, introducing stochasticity into the evaluation.

## Related Work & Insights
- Classical data replay and knowledge distillation frameworks from CIL transfer naturally to the token dimension, providing a reference for broader incremental learning scenarios.
- Masked prediction frameworks from pretrained models such as scGPT and scBERT offer a mature training paradigm for gene learning.
- The primary contribution of this work lies in problem formulation and benchmark construction rather than problem solving, opening a new direction for future research.
- Insight: Although natural language is unsuitable for token incremental learning, domains such as source code—where tokens carry independent semantics—may be similarly amenable.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](../../ICML2026/computational_biology/scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/computational_biology/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](../../ICML2026/computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](consurv_multimodal_continual_learning_for_survival_analysis.md)

</div>

<!-- RELATED:END -->
