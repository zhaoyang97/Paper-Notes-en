---
title: >-
  [Paper Note] PolyGraph Discrepancy: a classifier-based metric for graph generation
description: >-
  [ICLR 2026][Image Generation][Graph generation] This paper proposes PolyGraph Discrepancy (PGD), which approximates a variational lower bound on the Jensen-Shannon distance by training a classifier to distinguish real gr…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Graph generation"
  - "Jensen-Shannon distance"
  - "classifier-based evaluation"
  - "MMD"
  - "TabPFN"
date: 2026-05-08
content_hash: 21be7192e3016185
---

# PolyGraph Discrepancy: a classifier-based metric for graph generation

**Conference**: ICLR 2026
**arXiv**: [2510.06122](https://arxiv.org/abs/2510.06122)
**Code**: PolyGraph open-source library (mentioned in the paper; to be publicly released)
**Area**: Graph generative model evaluation
**Keywords**: Graph generation, Jensen-Shannon distance, classifier-based evaluation, MMD, TabPFN

## TL;DR

This paper proposes PolyGraph Discrepancy (PGD), which approximates a variational lower bound on the Jensen-Shannon distance by training a classifier to distinguish real graphs from generated ones. PGD addresses three fundamental shortcomings of MMD-based metrics: the lack of an absolute scale, incomparability across descriptors, and high bias and variance under small sample sizes.

## Background & Motivation

Graph generative models are increasingly important in drug discovery, social network modeling, and materials science, yet progress is constrained by deficiencies in evaluation methodology. Current evaluation relies primarily on Maximum Mean Discrepancy (MMD) computed over graph descriptors, which suffers from three fundamental flaws:

**Lack of absolute scale**: The range of MMD is $[0, \infty)$ and depends on the kernel function and feature scaling. Whether MMD = 0.05 is good or bad cannot be determined. Under a linear kernel, multiplying input features by an arbitrary scalar scales MMD proportionally.

**Incomparability across descriptors**: MMD computed from degree histograms and MMD computed from orbit counts are on entirely different scales, making it impossible to compare which descriptor better distinguishes real from generated graphs.

**Severe small-sample problem**: Mainstream benchmarks (Planar, SBM, Lobster) contain only 20–40 test graphs, a scale at which MMD estimates exhibit enormous bias and variance, rendering model rankings unreliable.

| Property | MMD | PGD |
|----------|-----|-----|
| Range | $[0, \infty)$ | $[0, 1]$ |
| Absolute scale | ✗ | ✓ |
| Descriptor comparability | ✗ | ✓ |
| Multi-descriptor aggregation | ✗ | ✓ |
| Unified ranking | ✗ | ✓ |

## Method

### Overall Architecture

The core idea of PGD is to reformulate graph generative model evaluation as a classification problem: can real graphs and generated graphs be distinguished? If a classifier can easily separate them, the generation quality is poor; if separation is difficult, the quality is high.

Theoretically, after fitting a classifier on a fit set, its data log-likelihood on a test set provides a variational lower bound on the Jensen-Shannon divergence. This lower bound has two elegant properties: (1) its range is fixed at $[0, 1]$; (2) it holds for any classifier $D$, with tighter bounds corresponding to better-fitted classifiers.

$$D_{\text{JS}}(P \| Q) = \sup_{D:\mathcal{X}\to[0,1]} \frac{1}{2}\mathbb{E}_{x\sim P}[\log_2 D(x)] + \frac{1}{2}\mathbb{E}_{x\sim Q}[\log_2(1-D(x))] + 1$$

### Key Designs

1. **Single-descriptor JS distance estimation**: Given a reference graph set $P_{\text{ref}}$, a generated graph set $Q_{\text{gen}}$, and a descriptor $d$:

    - Split both sets into fit and test halves
    - Convert graphs in the fit set into vectors using descriptor $d$
    - Train a classifier to distinguish the two classes
    - Compute the log-likelihood on the test set as a lower bound on JS divergence
    - Take the square root to obtain the JS distance

   → Mechanism: easier classification → larger JS distance → worse generative model
   → Design Motivation: separating fit/test prevents overestimation due to overfitting

2. **Classifier choice — TabPFN**: TabPFN is selected as the classifier because it simultaneously satisfies three requirements:

    - **Probabilistic output**: produces class probabilities (required for log-likelihood)
    - **Efficiency**: fast inference, no repeated hyperparameter tuning
    - **No hyperparameters**: transformer-based Bayesian inference that adapts automatically to data

   → Mechanism: replaces conventional classifiers with a foundation model for tabular data
   → Design Motivation: deep networks (slow and require tuning) and SVM/decision trees (non-probabilistic outputs) are excluded

   Comparative experiments with logistic regression demonstrate that TabPFN consistently produces tighter lower bounds by modeling nonlinear decision boundaries.

3. **Tightest lower bound selection across multiple descriptors**: When using $K$ descriptors:

    - Perform 4-fold cross-validation on the fit set for each descriptor
    - Select the descriptor $d^\star$ with the highest mean validation metric
    - Train on the full fit set using $d^\star$ and evaluate on the test set

   → Mechanism: taking the maximum lower bound yields the tightest bound, corresponding to the most discriminative descriptor
   → Design Motivation: the most sensitive descriptor varies across problems (degree, orbit, spectrum, etc.), necessitating a data-driven selection

4. **Descriptor repertoire**:

    - **General descriptors**: degree histogram, clustering coefficient histogram, Laplacian spectrum, 4/5-node orbit counts, GIN embeddings
    - **Molecular descriptors**: topological indices (BertzCT, Kappa, etc.), physicochemical parameters (Lipinski rules), Morgan fingerprints, ChemNet/MolCLR embeddings

   → Mechanism: combination of domain-general and domain-specific descriptors
   → Design Motivation: richer descriptors yield tighter estimates of JS distance

### Loss & Training

PGD is an evaluation metric rather than a generative model. Its "training" involves only fitting the TabPFN classifier — in practice a single forward pass, since TabPFN is an in-context learner.

Key implementation details:
- Fit/test split ratio: 50:50
- Cross-validation: 4 folds
- JS lower bound clipped to $\max(\cdot, 0)$ to ensure non-negativity
- Final result is square-rooted (JS distance is a metric; JS divergence is not)

## Key Experimental Results

### Main Results

**Bias and variance of MMD**

At the common scale of 20–40 test graphs:
- Biased MMD estimates are severely affected by bias (order-of-magnitude differences)
- Unbiased MMD estimates still exhibit sufficient variance to render model comparisons unreliable
- The authors recommend larger datasets (SBM-L, Planar-L, Lobster-L, each with 4096 samples)

**Correlation of PGD with model quality**

| Metric | Planar-L Correlation | SBM-L | Lobster-L |
|--------|---------------------|-------|-----------|
| PGD | **99.52** | 88.07 | 89.32 |
| Orbit RBF | 73.49 | 51.05 | -34.81 |
| Degree RBF | 70.79 | 15.77 | -33.40 |
| Spectral RBF | 73.34 | 36.76 | -22.79 |
| Clustering RBF | 71.48 | 83.97 | 87.05 |
| GIN RBF | 82.78 | 14.12 | -30.31 |

The Pearson correlation between PGD and validity reaches 99.52% on Planar-L, far exceeding all MMD variants.

**Training dynamics tracking**

| Metric | Planar-L Spearman | SBM-L | Lobster-L |
|--------|------------------|-------|-----------|
| Validity | 92.31 | 83.64 | 85.47 |
| **PGD** | **93.71** | **62.73** | **78.19** |
| Orbit RBF | 86.71 | 20.00 | -8.09 |
| Degree RBF | 41.96 | -19.09 | -4.66 |

PGD improves monotonically with training iterations, whereas MMD metrics behave erratically and may even show negative correlation.

**Model benchmarking** (PGD × 100, lower is better)

| Dataset | AutoGraph | DiGress | ESGG | GRAN |
|---------|-----------|---------|------|------|
| Planar-L | **34.0** | 45.2 | 54.2 | 65.1 |
| SBM-L | 30.3 | **23.7** | 35.1 | 50.1 |
| Lobster-L | **16.1** | 18.8 | 51.1 | 56.2 |
| Proteins | 72.5 | **68.2** | — | 73.3 |

### Ablation Study

| Configuration | Key Result | Explanation |
|---------------|-----------|-------------|
| TabPFN vs. logistic regression | TabPFN consistently tighter | Advantage of nonlinear decision boundaries |
| PGD-JS vs. PGD-TV | PGD-JS more robust | Binarization threshold in TV distance introduces noise |
| Sample size < 256 | High variance | PGD stabilizes at 256+ samples |
| Sample size > 256 | Stable | Both mean and variance converge |
| Single descriptor vs. max-aggregation | Aggregation more robust | No single descriptor is optimal across all settings |

### Key Findings

1. **No single descriptor is universally optimal**: The most discriminative descriptor differs across datasets and perturbation types, justifying the need for multi-descriptor and data-driven selection.
2. **Monotonic response of PGD to perturbations**: Regardless of perturbation type (edge deletion/addition/rewiring/swap/mix), PGD increases monotonically with perturbation intensity, with Spearman correlations comparable to MMD.
3. **Unique value of edge-swap perturbations**: The newly proposed perturbation type (degree-preserving edge swaps) cannot be detected by degree- or GIN-based MMD, yet PGD remains robust via its multi-descriptor strategy.
4. **Proteins dataset is most challenging**: PGD values are highest (~70) on Proteins, indicating the greatest difficulty in modeling protein graphs.
5. **5-node orbit counts are strong descriptors**: In model benchmarking, 5-node orbit counts frequently yield the highest PGD values, representing a previously underutilized descriptor.

## Highlights & Insights

1. **Theoretical elegance**: The adversarial training idea from GANs is repurposed for evaluation rather than training — the classifier's log-likelihood provides a variational lower bound on JS distance, a well-known result applied with ingenuity.
2. **High practical value**: Absolute scale, descriptor comparability, and multi-descriptor aggregation together address the three most pressing pain points in graph generation evaluation.
3. **Large-dataset advocacy**: Through detailed bias/variance analysis, a strong case is made that existing benchmark datasets (20–40 graphs) are insufficient for reliable evaluation; SBM-L/Planar-L/Lobster-L are provided as replacements.
4. **Molecule-specific descriptors**: The PGD framework is extended to molecular graph evaluation with a descriptor suite based on RDKit topological indices, physicochemical properties, fingerprints, and learned representations.
5. **Open-source code and data**: The PolyGraph library is committed to public release, lowering the barrier to adoption.

## Limitations & Future Work

1. **Descriptor dependency and information loss**: PGD operates on hand-crafted descriptors rather than raw graphs. If the divergence between descriptor distributions does not tightly approximate the divergence between graph distributions, PGD remains a loose lower bound.
2. **Limitations of max-aggregation**: Taking the maximum may fail to exploit complementary information across descriptors. Future work could explore concatenating features from multiple descriptors and feeding them directly to TabPFN.
3. **Sample size requirement**: PGD requires approximately 256 or more samples for reliable estimation, which may limit applicability in data-scarce settings.
4. **Feature dimensionality constraint of TabPFN**: No more than 500 dimensions are recommended, requiring dimensionality reduction for high-dimensional descriptors.
5. **Unexplored graph types**: Validation is limited to undirected unweighted graphs and molecular graphs; directed, weighted, temporal, and heterogeneous graphs require further investigation.
6. **Connection to GAN evaluation**: The GAN community has developed several classifier-based evaluation methods (FID, C2ST, etc.); the relationship and comparison between PGD and these methods merit deeper treatment.
7. **Computational cost**: PGD computation takes approximately 170 s (Table 15); although this can be amortized alongside descriptor computation, it is substantially slower than a single MMD computation.

## Related Work & Insights

- **MMD framework** (You et al., 2018; O'Bray et al., 2022): The primary comparison target for PGD; the authors analyze various MMD variants in detail (Gaussian TV, RBF, biased/unbiased).
- **Classifier Two-Sample Test** (Lopez-Paz & Oquab, 2017): The direct theoretical predecessor of PGD, using classifier performance as a test statistic for two-sample testing.
- **TabPFN** (Hollmann et al., 2025): A foundation model for tabular data; the practical effectiveness of PGD's classifier choice depends on this model's strong performance.
- **DiGress / AutoGraph**: Current state-of-the-art graph generative models; PGD provides more reliable rankings for these models.
- Insight: Evaluation metric design likewise requires a balance of theoretical grounding and engineering considerations; framing evaluation problems from an information-theoretic perspective tends to yield more principled solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic application of the C2ST idea to graph generation; multi-descriptor aggregation is novel though the underlying theory is established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Exceptionally comprehensive: perturbation experiments, training dynamics, model benchmarking, ablations, and comparisons with multiple MMD variants.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rigorous theoretical derivations, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Poised to become a new standard for graph generation evaluation, addressing core problems that have long challenged the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[ICLR 2026\] GGBall: Graph Generative Model on Poincaré Ball](ggball_graph_generative_model_on_poincaré_ball.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[CVPR 2026\] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric](../../CVPR2026/image_generation/shoe_semantic_hoi_open-vocabulary_evaluation_metric.md)

</div>

<!-- RELATED:END -->
