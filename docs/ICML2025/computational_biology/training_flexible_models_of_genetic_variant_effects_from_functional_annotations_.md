---
title: >-
  [Paper Note] Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra
description: >-
  [ICML2025][Computational Biology][GWAS] This paper introduces DeepWAS (Deep genome Wide Association Studies), which leverages modern fast linear algebra techniques (banded matrix approximation + iterative solvers) to resolve the computational bottleneck of large-scale LD matrix inversion in GWAS. This achieves, for the first time, the training of functional annotation-driven genetic variant effect prediction models by maximizing the full marginal likelihood with large-scale n…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "GWAS"
  - "Deep Learning"
  - "Accelerated Linear Algebra"
  - "Genetic Variant Effect Prediction"
  - "Functional Annotation"
date: 2026-05-08
content_hash: 238c8194cb7c7d56
---

# Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra

**Conference**: ICML2025  
**arXiv**: [2506.19598](https://arxiv.org/abs/2506.19598)  
**Code**: [https://github.com/AlanNawzadAmin/DeepWAS](https://github.com/AlanNawzadAmin/DeepWAS)  
**Area**: Medical Imaging / Computational Genomics  
**Keywords**: GWAS, Deep Learning, Accelerated Linear Algebra, Genetic Variant Effect Prediction, Functional Annotation

## TL;DR

This paper introduces DeepWAS (Deep genome Wide Association Studies), which leverages modern fast linear algebra techniques (banded matrix approximation + iterative solvers) to resolve the computational bottleneck of large-scale LD matrix inversion in GWAS. This achieves, for the first time, the training of functional annotation-driven genetic variant effect prediction models by maximizing the full marginal likelihood with large-scale neural networks. The authors discover that larger models yield improved performance only under full-likelihood training (contrary to traditional summary statistics fitting).

## Background & Motivation

**Background**: Genome-wide association studies (GWAS) aim to model the relationship between genetic variants and phenotypes (e.g., height, diseases like asthma) by analyzing genotype data from hundreds of thousands of individuals ($M \approx 10^6 - 10^8$ variants, $N \approx 10^5$ samples). Geneticists construct "functionally informed priors" utilizing functional genomic features (e.g., DNA accessibility, transcription factor binding sites, cross-species conservation) to better predict the impact of variants on phenotypes.

**Limitations of Prior Work**: Due to linkage disequilibrium (LD) — the strong correlation between adjacent variants in the genome — calculating the marginal likelihood requires inverting and computing the log-determinant of large LD matrices, which incurs a complexity of $\mathcal{O}(M^3)$. This is computationally intractable for millions of variants.

**Key Challenge**: To bypass LD matrix inversion, existing State-of-the-Art (SOTA) methods (such as LD score regression) make two compromises: (a) they employ only simple parameterized models (linear or low-dimensional), failing to capture complex non-linear relationships between functional annotations and phenotypes; (b) they fit summary statistics rather than the full likelihood, sacrificing statistical efficiency. Consequently, model performance fails to scale even as more data and features become available.

**Goal**: How to efficiently train large-scale, flexible neural network models within the full-likelihood framework of GWAS, enabling them to fully exploit the increasingly rich functional genomic features.

**Key Insight**: The authors notice successful experiences in the field of Gaussian process regression, where iterative algorithms reduce the complexity of matrix inversion from $\mathcal{O}(M^3)$ to $\mathcal{O}(M^2 K)$ ($K \ll M$) and drastically reduce the number of iterations by improving the matrix condition number. The LD matrix inherently possesses a banded sparse structure (correlation between distant variants approaches zero) which can be efficiently approximated.

**Core Idea**: Utilizing the banded structure of the LD matrix for block approximation, treating the blocks as mini-batches, and combining iterative linear algebra methods to efficiently calculate the full likelihood and its gradients. This enables the training of large-scale deep neural networks on GWAS data for the first time.

## Method

### Overall Architecture

The overall pipeline of DeepWAS is as follows:
- **Input**: (1) Large-scale functional genomic features (DNA accessibility, transcription factor binding, cross-species conservation, etc.), with a feature vector for a window surrounding each variant site; (2) Publicly available phenotypic association data (GWAS summary statistics and LD matrices).
- **Model**: A parameterized neural network $f_\theta$ that takes the functional annotation features of a variant site as input and outputs the prior effect size of that variant on the phenotype.
- **Training Target**: Maximizing the full marginal likelihood, rather than fitting summary statistics.
- **Output**: The trained model can predict phenotypic effects for any new variant site, which is useful for disease risk prediction and therapeutic target identification.

The core innovation of this entire approach lies in making full-likelihood training computationally feasible. While traditional methods require inverting the entire LD matrix ($M \times M$), DeepWAS decomposes it into manageable subproblems through banded approximation.

### Key Designs

1. **Banded Approximation of the LD Matrix**:

    - **Function**: Approximates the massive LD correlation matrix as a banded matrix, assuming that the correlation between variants separated by more than a certain window size is zero.
    - **Mechanism**: Since variant correlations in the genome primarily stem from sites that are physically close, the LD matrix naturally possesses an approximate banded structure. Utilizing this property, the $M \times M$ matrix can be decomposed into multiple overlapping smaller matrix blocks (slices).
    - **Design Motivation**: The banded structure allows matrix inversion to be performed independently on each slice, significantly reducing the complexity. Simultaneously, these slices naturally serve as mini-batches for stochastic gradient descent, enabling highly efficient batch training.

2. **Likelihood Rearrangement**:

    - **Function**: Reorganizes the mathematical formulation of the marginal likelihood to make it suitable for acceleration via iterative linear algebra algorithms.
    - **Mechanism**: In hierarchical Bayesian models, the marginal likelihood of the phenotype $y$ involves the inversion and log-determinant of the LD matrix $\Sigma$. The traditional approach directly performs Cholesky decomposition with $\mathcal{O}(M^3)$ complexity. DeepWAS rewrites the likelihood into a form suitable for iterative solvers such as the Conjugate Gradient method. By employing preconditioning to improve the matrix condition number, the number of iterations $K$ can be dramatically reduced.
    - **Design Motivation**: Each step of the iterative algorithm only requires matrix-vector multiplication ($\mathcal{O}(M^2)$ or lower when exploiting sparsity). The total complexity for $K$ steps is $\mathcal{O}(M^2 K)$, which far outperforms direct inversion when $K \ll M$.

3. **Large-scale Feature Curation**:

    - **Function**: Carefully curates a vast array of functional genomic features to serve as inputs for the neural network.
    - **Mechanism**: Collects various functional annotations from databases such as ENCODE and FANTOM, including DNA accessibility, histone modifications, transcription factor binding, and evolutionary conservation, to construct high-dimensional feature vectors for each variant site.
    - **Design Motivation**: Richer features provide large models with sufficient information to learn complex feature-to-effect mappings, which forms the empirical basis for DeepWAS's "larger is better" conclusion.

4. **Deep Neural Network as a Functionally Informed Prior**:

    - **Function**: Replaces traditional linear or low-dimensional parametric models with a deep network as the prior.
    - **Mechanism**: Traditional methods (e.g., S-LDSC) use a linear model $\sigma_j^2 = \sum_k a_{jk} \tau_k$ to map functional annotations $a_{jk}$ to the variant effect variance $\sigma_j^2$, using very few parameters. DeepWAS replaces this mapping with a neural network $f_\theta$, enabling it to capture non-linear interactions.
    - **Design Motivation**: Linear models are prone to overfitting when the number of features increases (especially under summary statistics fitting), whereas neural networks under full-likelihood training exhibit better inductive biases, avoiding the "larger is worse" issue.

### Loss & Training

- **Objective Function**: Maximizing the marginal log-likelihood $\log p(y | \theta)$ based on the banded approximate LD matrix. For each mini-batch (corresponding to a slice of the LD matrix), the local likelihood contribution is computed to perform gradient updates.
- **Optimizer**: Standard stochastic gradient descent variants (such as Adam) are used. During each training step, iterative linear algebra (conjugate gradient + preconditioning) is utilized to efficiently compute the likelihood and its gradient with respect to $\theta$.
- **Training Data**: Uses public large-scale GWAS data (e.g., UK Biobank summary statistics), grouped by chromosome to perform leave-one-chromosome-out cross-validation.
- **Baseline Training Setup**: For comparison, a traditional LD score regression training style is also implemented to validate the superiority of the full-likelihood approach.

## Key Experimental Results

### Main Results: Effect of Model Scale and Training Objective on Predictive Performance

| Training Method | Model Type | Model Scale | Fitting Quality (Likelihood/AIC) | Predictive Performance Trend |
|----------|---------|---------|---------------------|-------------|
| LD score regression | Linear | Small (tens of parameters) | Baseline | Baseline |
| LD score regression | Neural Network | Large (thousands of parameters) | $\le$ Baseline | No improvement or decreases |
| DeepWAS (Full Likelihood) | Linear | Small | > Baseline | Outperforms LDSC |
| DeepWAS (Full Likelihood) | Neural Network | Large | >> Baseline | Significantly outperforms small models |

> Core Conclusion: Under summary statistics fitting (LDSC), increasing model scale fails to improve—and can even degrade—performance. However, under DeepWAS full-likelihood training, larger and more flexible models consistently deliver better fitting and predictive performance.

### Ablation Study: Contribution of Key Components

| Configuration | Relative Performance | Description |
|------|---------|------|
| DeepWAS + Large Model + All Features | Optimal | Full model, best prediction |
| DeepWAS + Small Model + All Features | Sub-optimal | Model capacity insufficient to capture non-linearities |
| DeepWAS + Large Model + Few Features | Moderate | Insufficient features limit the model's upper bound |
| LDSC + Large Model + All Features | Poor | Summary statistics fitting fails to leverage large models |
| LDSC + Small Model + Few Features | Baseline | Traditional method |

### Key Findings

- **Model scaling effects are training-objective dependent**: This is the most central finding. Under the LDSC framework, larger models perform worse than smaller ones (overfitting to summary statistics); under the full likelihood, larger models are consistently superior. This demonstrates that the choice of training objective is more critical than model architecture.
- **Feature count and model capacity must be aligned**: More functional features only yield improvements when the model is sufficiently large; small models cannot ingest the extra features.
- **Improvements on held-out data**: The combination of large models and full likelihood achieves superior predictive accuracy on held-out chromosomes compared to all baselines, hinting that larger models and more features still have room for further scaling.
- **Manageable training overhead**: Despite employing large-scale neural networks and full likelihood, DeepWAS constrains computation to a reasonable range on modern GPUs using banded approximation and iterative solvers.

## Highlights & Insights

- **Training objectives matter more than model architecture**: A profound insight — larger models perform worse on simplified objectives (summary statistics) but unlock their true potential only under the correct objective (full likelihood). This finding is valuable not only for GWAS but also for any scenario using proxy objectives.
- **Alignment of LD matrix banded approximation with mini-batches**: Cleverly exploiting the physical structure of the LD matrix (banded sparsity from genetic distance) to decompose it into mini-batches balances statistical rigor and computational efficiency. This paradigm of "co-designing computation using the physical structure of data" is transferable to other large-scale problems with local correlation.
- **Importing iterative linear algebra from the GP community**: Transferring iterative solvers + preconditioning techniques from Gaussian process literature to genetics demonstrates the value of cross-disciplinary method transfer. GPyTorch-style methods have vast application prospects in computational biology.
- **Validating scaling laws in genomics**: The paper indirectly validates a "scaling law" type phenomenon: larger models + more data + better training objectives $\rightarrow$ better performance, provided the training method is mathematically sound.

## Limitations & Future Work

- **Accuracy-efficiency trade-off of banded approximation**: The choice of band width is a hyperparameter. A band that is too narrow may miss long-range LD relationships, while a band that is too wide increases computational cost. The paper does not thoroughly discuss adaptive bandwidth selection.
- **Model interpretability**: Replacing linear models with deep networks enhances predictive performance but makes it harder for biologists to understand "which functional features are important in which contexts."
- **Data dependency**: The model relies heavily on large-scale public GWAS data and functional annotation databases. For underrepresented populations (e.g., non-European ancestries) or rare diseases, limited data availability may restrict the method's applicability.
- **Computational resource requirements**: Although algorithmic optimization reduces complexity, training large neural networks still requires GPU resources, which might not be accessible to all resource-constrained genetics laboratories.
- **Comparison with other Bayesian GWAS methods**: Fine-mapping methods like SuSiE and FINEMAP use different prior assumptions; exploring the complementarity of DeepWAS with these approaches is a promising direction.

## Related Work & Insights

- **vs LD Score Regression (S-LDSC)**: S-LDSC is currently the most widely used method, fitting linear models to summary statistics. The core advantage of DeepWAS lies in full-likelihood training paired with large models, directly overcoming the two central compromises of S-LDSC. One of the strongest conclusions of this paper is that larger models actually perform worse under S-LDSC.
- **vs GPyTorch / Iterative GP methods**: The acceleration strategy of DeepWAS directly draws inspiration from Gardner et al. (2018) in Gaussian process regression. The key difference is that the LD matrices in DeepWAS possess an inherent physical banded structure, which lends itself more naturally to block-wise batch processing.
- **vs PolyFun / BayesR**: These methods also guide priors with functional annotations but are limited by simplified models or heavy computational overhead. DeepWAS breaking through model complexity limits via computational innovation.
- **Insights**: This work demonstrates that in large-scale structured statistical problems, the correct training objective (full likelihood vs. proxy objective) combined with efficient computation (linear algebra acceleration exploiting problem structure) can jointly unlock the benefits of deep learning scaling. This paradigm could inspire other statistical genetics problems facing similar computational bottlenecks, such as multi-trait analysis and cross-population genetic analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ Transferring iterative linear algebra from the GP community to GWAS, with a novel and crucial core insight (training objective dictates scaling behavior).
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple phenotypes with held-out prediction and comparative baselines, though some absolute numerical metrics in caching are missing.
- Writing Quality: ⭐⭐⭐⭐ Clean introductory logic, smooth derivation from problem to solution, and background presentation friendly to non-domain readers.
- Value: ⭐⭐⭐⭐ Highly significant for the computational genetics community; the pairing of full likelihood with deep networks could shift the modeling paradigm in this domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](../../NeurIPS2025/computational_biology/multimodal_3d_genome_pre-training.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/computational_biology/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2026\] Flexible Kernels for Protein Property Prediction](../../ICML2026/computational_biology/flexible_kernels_for_protein_property_prediction.md)

</div>

<!-- RELATED:END -->
