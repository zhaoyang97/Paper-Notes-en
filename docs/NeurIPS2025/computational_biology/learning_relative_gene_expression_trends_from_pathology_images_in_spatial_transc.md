---
title: >-
  [Paper Note] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics
description: >-
  [NeurIPS 2025][Computational Biology][spatial transcriptomics] This paper proposes STRank, a loss function that reformulates gene expression estimation from pathology images as a ranking score estimation task. By modeling the stochastic noise inherent in expression counts via binomial/multinomial distributions, STRank enables models to learn robust relative expression relationships from spatial transcriptomics data subject to batch effects and random fluctuations.
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "spatial transcriptomics"
  - "gene expression estimation"
  - "learning to rank"
  - "pathology images"
  - "batch effects"
date: 2026-05-08
content_hash: d2c97da459c8a81c
---

# Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics

**Conference**: NeurIPS 2025
**arXiv**: [2512.06612](https://arxiv.org/abs/2512.06612)  
**Code**: [GitHub](https://github.com/naivete5656/STRank)  
**Area**: Medical Imaging / Spatial Transcriptomics
**Keywords**: spatial transcriptomics, gene expression estimation, learning to rank, pathology images, batch effects

## TL;DR

This paper proposes STRank, a loss function that reformulates gene expression estimation from pathology images as a ranking score estimation task. By modeling the stochastic noise inherent in expression counts via binomial/multinomial distributions, STRank enables models to learn robust relative expression relationships from spatial transcriptomics data subject to batch effects and random fluctuations.

## Background & Motivation

Spatial transcriptomics technologies (e.g., Visium, Xenium) enable high-resolution gene expression profiling on tissue sections, but sequencing costs remain prohibitive. Directly estimating gene expression from pathology images offers a low-cost alternative; however, two core challenges persist:

**Batch Effects**: Technical factors such as reagent batches and instrument variation introduce systematic biases (scaling and offset) in expression values across tissue samples. Models trained with MSE loss tend to learn these spurious associations rather than genuine biological signals.

**Stochastic Noise**: Due to cellular heterogeneity and temporal dynamics, observed gene expression values fluctuate randomly even for visually identical image patches. Low-expression genes exhibit particularly poor signal-to-noise ratios, and noise can alter the relative ordering of samples.

Existing methods predominantly employ pointwise MSE loss to optimize absolute expression values on a per-sample basis, which cannot accommodate batch effects. While pairwise losses such as Ranking Loss partially mitigate batch effects, they do not account for the probabilistic nature of count data and fail to reliably distinguish signal from noise under low-signal conditions.

The central hypothesis of this work is: **even when absolute expression values are confounded by batch effects and noise, the relative expression trends of genes across image patches remain consistent across independent experiments**. For instance, cancer-specific gene expression is consistently higher in tumor regions than in non-tumor regions.

## Method

### Overall Architecture

The conventional task of "predicting absolute expression values" is reformulated as "predicting ranking scores." The model $f: x^{n,i} \to r^{n,i}$ predicts a scale-invariant ranking score $r$ from image patches, reflecting relative expression relationships within the same tissue. The feature extractor is frozen (using the CONCH pathology foundation model); only the prediction head is trained, isolating the effect of the loss function.

### Key Designs

1. **Pairwise STRank Loss**: Given a pair of image patches $(x^i, x^j)$ from the same tissue, the expression count $e^i_g$ is assumed to follow a binomial distribution $\text{Binomial}(t_g^{i,j}, p_g^i)$, where $t_g^{i,j} = e_g^i + e_g^j$ is the total expression of gene $g$ and $p_g^i$ is the frequency parameter for spot $i$. The model outputs scores $\hat{r}^i, \hat{r}^j$, which are converted to probability estimates $\hat{p}_g^i$ via softmax, and training minimizes the negative log-likelihood:

$$L_{\text{STRank}}^{\text{pair}} = -\sum_{g=1}^{N^g} (e_g^i \log \hat{p}_g^i + e_g^j \log \hat{p}_g^j)$$

where $\hat{p}_g^i = \frac{\exp(\hat{r}_g^i)}{\exp(\hat{r}_g^i) + \exp(\hat{r}_g^j)}$. A key advantage is that when $\hat{r}^i \gg \hat{r}^j$, this loss reduces to the conventional Ranking Loss; when ordering is uncertain, it adaptively weights by count magnitude — penalizing misranking of highly expressed genes more strongly while tolerating greater uncertainty for lowly expressed ones.

2. **Listwise STRank Loss**: The pairwise formulation is extended to lists, assuming expression counts across $N^k$ spots follow a multinomial distribution $\text{Multinomial}(T_g^{(n)}, p_g^i)$. Probabilities across all spots are computed via softmax:

$$L_{\text{STRank}}^{\text{List}} = -\sum_{g}^{N^g} \sum_{i}^{N^k} e_g^i \log p_g^i$$

The listwise formulation captures global expression patterns, outperforming pairwise comparisons.

3. **Library Size Correction**: The total expression count per spot $l^i = \sum_g e_g^i$ is introduced as a correction factor, adjusting probability estimates as $p_g^i = \frac{\exp(\hat{r}_g^i) l^i}{\sum_j \exp(\hat{r}_g^j) l^j}$, accounting for differences in detection capacity across spots while preserving the discrete structure of count data.

### Loss & Training

- Sample pairs are constructed via within-group random permutation; each reference sample is randomly paired with another sample from the same tissue
- Mini-batch losses aggregate relative signals from different tissues
- AdamW optimizer with learning rate $5 \times 10^{-5}$ and batch size 256
- Early stopping with patience = 30 epochs

## Key Experimental Results

### Synthetic Data Experiments

| Loss Type | Method | Uniform SCC ↑ | Imbalanced SCC ↑ |
|-----------|--------|---------------|------------------|
| Pointwise | MSE | 0.748 | 0.583 |
| Pointwise | Poisson | 0.777 | 0.603 |
| Pointwise | Negative Binomial | 0.788 | 0.601 |
| Pairwise | Rank | 0.835 | 0.738 |
| Pairwise | **PairSTRank** | **0.907** | **0.818** |
| Listwise | PCC | 0.858 | 0.560 |
| Listwise | **ListSTRank** | **0.945** | **0.828** |

### Real Dataset Experiments (HEST-1k Benchmark, SCC ↑)

| Loss | IDC | PRAD | PAAD | COAD | READ | ccRCC | IDC-L | Mean |
|------|-----|------|------|------|------|-------|-------|------|
| MSE | 0.393 | 0.484 | 0.307 | 0.556 | 0.140 | 0.093 | 0.168 | 0.306 |
| Rank | 0.317 | 0.317 | 0.181 | 0.566 | 0.047 | 0.059 | 0.110 | 0.228 |
| PCC | 0.472 | 0.459 | 0.307 | 0.640 | 0.105 | 0.102 | 0.198 | 0.326 |
| PairSTRank | 0.494 | 0.458 | 0.346 | 0.613 | 0.136 | 0.127 | 0.228 | **0.343** |
| ListSTRank | **0.510** | 0.459 | 0.343 | 0.597 | 0.140 | 0.125 | 0.238 | **0.345** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| PairSTRank vs. Rank | SCC +0.072/+0.080 | Modeling count distributions outperforms simple ranking |
| ListSTRank vs. PCC | SCC +0.087/+0.268 | Multinomial modeling yields substantial gains under imbalanced conditions |
| Listwise vs. Pairwise | ListSTRank > PairSTRank | Global context benefits scenarios with stronger batch effects |
| With/without library size correction | Marginal improvement | Addresses inter-spot detection capacity differences |

### Key Findings

- Relative expression learning (ranking-based methods) consistently outperforms absolute expression learning (pointwise methods), especially under batch effects
- The probabilistic modeling in STRank provides significant advantages under low-signal conditions — stochastic noise in count data is particularly impactful when gene expression is sparse
- ListSTRank performs best on synthetic data (capturing global patterns) but performs comparably to PairSTRank on real data
- Real-data evaluation is itself noisy, yet STRank achieves the best average performance across benchmarks

## Highlights & Insights

- **Elegant problem reformulation**: Recasting expression estimation as ranking score estimation directly circumvents the root cause of batch effect problems
- **Principled probabilistic modeling**: Modeling count data with binomial/multinomial distributions is a statistically natural choice, enabling the loss to adaptively weight samples by expression magnitude
- **Unification with classical Ranking Loss**: The paper proves that STRank reduces to conventional ranking loss when score differences are sufficiently large, establishing an elegant theoretical connection
- **Compatibility with downstream analyses**: Relative expression relationships are precisely the information required by common downstream tasks such as differential expression analysis

## Limitations & Future Work

- Improvements on real data are modest (average SCC increases from 0.326 to 0.345)
- Evaluation is restricted to 50 highly variable genes; practical applications may involve a far larger gene set
- The feature extractor (CONCH) is frozen; end-to-end training remains unexplored
- Cross-platform generalization (e.g., Visium → Xenium) has not been validated
- The evaluation metric itself is subject to noise, making it difficult to determine the true performance ceiling

## Related Work & Insights

- **Learning to Rank**: A classical information retrieval paradigm introduced here into the spatial transcriptomics domain
- **HEST-1k**: A benchmark dataset for spatial transcriptomics
- **CONCH**: A pathology vision-language foundation model providing feature representations
- Insight: Loss function design is an underappreciated research direction; principled probabilistic modeling can yield substantial gains without modifying model architecture

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of learning-to-rank with probabilistic modeling of count data is a novel loss design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic data validate hypotheses; 7 real datasets are evaluated, though gains are moderate
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and motivation is well-articulated
- **Value**: ⭐⭐⭐⭐ The loss function approach is broadly applicable and readily extensible to other count data scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[CVPR 2026\] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images](../../CVPR2026/computational_biology/from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima.md)
- [\[NeurIPS 2025\] Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow](modeling_microenvironment_trajectories_on_spatial_transcriptomics_with_nicheflow.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](../../ICML2025/computational_biology/scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)

</div>

<!-- RELATED:END -->
