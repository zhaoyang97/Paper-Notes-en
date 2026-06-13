---
title: >-
  [Paper Note] What Makes a Representation Good for Single-Cell Perturbation Prediction?
description: >-
  [ICML2026][Computational Biology][Single-cell] This paper proposes PerturbedVAE, arguing that an effective representation for single-cell perturbation prediction must explicitly decouple dominant perturbation-invariant b…
tags:
  - "ICML2026"
  - "Computational Biology"
  - "Single-cell"
  - "perturbation prediction"
  - "variational autoencoder"
  - "causal representation"
  - "combinatorial generalization"
date: 2026-05-08
content_hash: 28fecddee8325508
---

# What Makes a Representation Good for Single-Cell Perturbation Prediction?

**Conference**: ICML2026  
**arXiv**: [2605.19343](https://arxiv.org/abs/2605.19343)  
**Code**: No public code  
**Area**: Scientific Computing / Single-cell Perturbation Prediction  
**Keywords**: Single-cell, perturbation prediction, variational autoencoder, causal representation, combinatorial generalization  

## TL;DR
This paper proposes PerturbedVAE, arguing that an effective representation for single-cell perturbation prediction must explicitly decouple dominant perturbation-invariant background programs from sparse perturbation-responsive signals, and organize the latter using a causal structure to better generalize to unseen dual-gene combinatorial perturbations.

## Background & Motivation
**Background**: Single-cell perturbation modeling aims to predict how cellular gene expression profiles change after genes are intervened (e.g., via CRISPR). Such models are crucial for drug discovery, understanding gene regulatory mechanisms, and designing combinatorial perturbations. Existing methods generally follow two paths: causal representation learning, which uses latent variables and structural equations to characterize perturbation mechanisms; and single-cell foundation models, which learn universal representations using large-scale transcriptomic data.

**Limitations of Prior Work**: Single-cell expression data exhibits a frequently overlooked imbalance: most expression variance stems from perturbation-invariant factors like cell types, background programs, and technical noise, while signals truly induced by specific perturbations are sparse. To fit the overall distribution, universal foundation models often prioritize encoding the dominant background, suppressing perturbation-specific information. Similarly, causal representation methods may conflate background information into perturbation-related latent variables without explicit separation, leading to impure representation semantics.

**Key Challenge**: Perturbation prediction requires both the retention of background cellular states and the extraction of sparse but critical perturbation-specific signals. Emphasizing only reconstruction allows the model to explain everything with background variables; emphasizing only perturbation variables leads to the loss of the cellular base state. The true difficulty lies in extracting sparse perturbation effects under strong background signals and organizing them into a structure capable of combinatorial generalization.

**Goal**: The authors propose the "perturbation suppression hypothesis" to explain why foundation models and general causal representation methods fail. Subsequently, they design PerturbedVAE, which splits the latent space into an invariant block and a responsive block, supporting this design through contrastive alignment, conditional latent causal models, and identifiability analysis.

**Key Insight**: Starting from the question "what representation is good for perturbation prediction," the answer is not a larger model or a more complex regressor, but that the representation must be perturbation-aware: it must first explicitly extract perturbation-specific information and then utilize causal structures to predict unseen combinatorial interventions.

**Core Idea**: Use control cells to align perturbation-invariant latent variables, ensuring background programs are fixed in $z_\iota$; place the remaining perturbation-responsive signals into $z_\nu$, and use a perturbation-conditioned latent causal structure to generate and combine unseen perturbation effects.

## Method
PerturbedVAE can be viewed as a structured VAE oriented toward single-cell perturbation data. While a vanilla VAE aims to reconstruct expression profiles, here the authors specify roles for latent variables: $z_\iota$ represents perturbation-invariant background programs, and $z_\nu$ represents perturbation-responsive factors. During training, the model observes both perturbed and unperturbed (control) samples, enforcing consistency in $z_\iota$ between the two; thus, background variations are absorbed by $z_\iota$, forcing $z_\nu$ to express the residual changes brought by the perturbation. To predict unseen combinatorial perturbations, the model infers $z_\iota$ from control cells and inputs the dual-gene perturbation vector into a learned perturbation-conditioned mechanism to generate $z_\nu$, which is then decoded into an expression profile.

### Overall Architecture
The input consists of a single-cell expression vector $x$ and a perturbation label $u$, where $u$ can be a one-hot single-gene perturbation or a multi-hot vector for dual-gene combinations. The generative model assumes $x=g(z)$, where $z=(z_\iota,z_\nu)$. $z_\iota$ is perturbation-independent and characterizes background cellular programs; $z_\nu$ depends on $u$ and $z_\iota$, following an unknown DAG representing causal dependencies between perturbation-responsive programs. The variational posterior is decomposed as $q(z_\nu,z_\iota|x,u)=q(z_\nu|x,u)q(z_\iota|x)$, matching the logic that "perturbation response requires labels, while background is inferred from the expression itself."

### Key Designs
1.  **Latent Space Splitting (Invariant vs. Responsive)**:
    - **Function**: Allocates dominant background programs and sparse perturbation effects into different latent variable blocks to avoid mutual contamination.
    - **Mechanism**: $z_\iota$ represents cellular backgrounds stable across perturbations, with a prior independent of $u$; $z_\nu$ represents latent factors that change with perturbations, following a conditional distribution $p(z_\nu|u,z_\iota)$. The reconstruction term of the ELBO ensures both blocks together explain the expression profile, while the KL term limits latent space capacity.
    - **Design Motivation**: Without splitting, models tend to complete reconstruction using large-scale background changes, suppressing perturbation-specific signals; splitting provides the model with a clear semantic division of labor.

2.  **Contrastive Alignment based on Unperturbed Controls**:
    - **Function**: Forces background latent variables to remain consistent between perturbed and unperturbed samples, anchoring invariant information to $z_\iota$.
    - **Mechanism**: For each perturbed sample $(x,u)$, a control profile $x^{(u_0)}$ is sampled to minimize $\mathcal{L}_{contrast}=\|z_\iota-z_\iota^{(u_0)}\|_2^2$. The total objective is $\mathcal{L}=-\mathcal{L}_{ELBO}+\alpha\mathcal{L}_{contrast}$.
    - **Design Motivation**: When optimizing only the ELBO, the reconstruction target might cause $z_\nu$ or $z_\iota$ to absorb incorrect information. Alignment forces $z_\iota$ to explain backgrounds shared across conditions, leaving environment-related changes to $z_\nu$.

3.  **Latent Causal Structure and Identifiability Constraints**:
    - **Function**: Transforms the perturbation-responsive block from a mere compressed representation into a structured mechanism usable for unseen combinatorial interventions.
    - **Mechanism**: The authors model $z_\nu$ as a linear Gaussian structural causal model (SCM) modulated by $u$, with a weight matrix satisfying strict lower triangularity to correspond to a DAG. Theoretical analysis shows that if the generative mapping is invertible and smooth, there is sufficient environmental variation, alignment reaches optimum, and interventions are sufficiently rich, then $z_\nu$ is identifiable up to permutation and scaling, and $z_\iota$ up to linear block transformation.
    - **Design Motivation**: Single-cell data often consists of partial interventions, where traditional CRL assumptions of rich intervention are not met. This analysis shows that under explicit separation and sufficient environmental differences, sparse perturbation variables can still be recovered.

### Loss & Training
The training objective consists of a negative ELBO and contrastive alignment. The ELBO includes a reconstruction term $\mathbb{E}_{q}[\log p(x|z_\nu,z_\iota,u)]$ and the KL divergence from $q(z_\nu,z_\iota|x,u)$ to $p(z_\nu,z_\iota|u)$. Real-world experiments use Norman2019 Perturb-seq: 105,528 K562 cells, 112 target genes, 105 single-gene, and 131 dual-gene conditions. The training set includes controls and 105 single-gene perturbations; 112 dual-gene perturbations are entirely reserved for OOD testing. The optimizer is Adam, batch size 64, epoch 100, hidden dimension 256, learning rate $10^{-4}$, and alignment weight $\alpha=0.05$.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | Ours | Prev. SOTA / Strong Baseline | Gain |
|--------|------|------|----------|------|
| Norman2019 Dual-gene OOD | RMSE ↓ | 0.4474±0.0007 | KNN 0.4894 / ElasticNet 0.4929 / STATE 0.4981 | 0.0420 reduction vs KNN |
| Norman2019 Dual-gene OOD | $R^2$ ↑ | 0.9865±0.0009 | UCE 0.9857 / KNN 0.9843 | Slightly better than best FM/simple baseline |
| Single-gene, z dim 105 | RMSE ↓ | 0.3995±0.0013 | SAMS-VAE 0.4123 / sVAE+ 0.5002 | Significantly lower |
| Dual-gene, z dim 105 | RMSE ↓ | 0.4474±0.0007 | SAMS-VAE 0.4629 / PerturbedVAE w/o Align 0.4623 | OOD more stable after alignment |
| Simulation identifiability | invariant $R^2$ ↑ | 0.97±0.0077 | w/o alignment 0.66±0.0281 | Alignment significantly improves recovery |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| w/o contrastive alignment | Dual-gene RMSE 0.4626±0.0002, $R^2$ 0.9650±0.0002 | Combinatorial generalization drops significantly without alignment |
| with contrastive alignment | Dual-gene RMSE 0.4474±0.0007, $R^2$ 0.9865±0.0009 | Alignment preserves $z_\iota$ information, boosting OOD performance |
| capacity: $z_\nu<z_\iota$ | Single RMSE 0.3995, Dual RMSE 0.4474 | Performance is best when background block capacity is larger |
| capacity: equal split | Single RMSE 0.4084, Dual RMSE 0.4627 | Insufficient invariant background capacity hinders prediction |
| PerturbedVAE(MMD) | RMSE 0.5485, $R^2$ 0.9958, MMD 0.3077 | Even with MMD, it outperforms Discrepancy-VAE, showing gains aren't just from changing loss |

### Key Findings
- Representations from single-cell foundation models do not necessarily retain linearly decodable perturbation labels. Linear probing shows that UCE, scFoundation, and Geneformer have weaker perturbation label decodability than direct PCA, supporting the perturbation suppression hypothesis.
- The alignment term is a key mechanism. In simulations, the $R^2$ of the invariant block improved from 0.66 to 0.97; in real data, the $R^2$ for dual-gene OOD improved from 0.9650 to 0.9865.
- Background capacity should not be too small. The optimal configuration is $z_\nu<z_\iota$, suggesting that while the task focuses on perturbation response, adequately modeling the invariant background is a prerequisite for extracting sparse perturbation signals.
- Additive baselines are strong on pseudobulk average response but yield negative cell-level $R^2$; PerturbedVAE, while not always having the lowest pseudobulk error, preserves explanatory variance at the single-cell level.

## Highlights & Insights
- The best insight of the paper is attributing single-cell perturbation prediction failure to signal imbalance rather than simply stating models are not large enough. The perspective that perturbation-specific signals are sparse while background signals dominate explains different failure modes of foundation models and general CRL methods.
- The structural division of labor in PerturbedVAE is very clear: $z_\iota$ handles background, $z_\nu$ handles response, and contrastive alignment pulls them apart. This design is more interpretable than simply adding larger latent spaces or deeper encoders.
- There is a tight connection between theory and implementation. Although the identifiability theorem relies on strong assumptions, it directly explains the need for environmental diversity, alignment, and perturbation-conditioned Gaussian SCMs.
- The paper does not shy away from the strength of simple additive baselines but distinguishes between pseudobulk average response and single-cell variability. This discussion clarifies the method's value: it is not just regressing to the mean but learning an interpretable perturbation mechanism.

## Limitations & Future Work
- Identifiability analysis relies on strong assumptions, such as invertible smooth generative mappings, sufficient environmental differences, globally optimal alignment, and a common DAG order, which real biological data may not fully satisfy.
- Main real-world experiments focus on Norman2019 and Replogle single-gene screens; validation across cell types, experimental platforms, drug perturbations, and more complex multigene combinations is still needed.
- PerturbedVAE requires unperturbed controls as alignment anchors. If an experimental design has few controls, strong batch effects, or mismatched controls, the alignment term may introduce bias.
- Current biological verification of the learned causal graph consists mainly of plausibility checks; the recovered regulatory edges require more systematic experimental or external database validation.

## Related Work & Insights
- **vs scFoundation / UCE / Geneformer**: These foundation models learn universal expression representations but may suppress perturbation-specific signals; PerturbedVAE is much smaller but more stable on dual-gene OOD due to task-matched structural inductive biases.
- **vs Discrepancy-VAE / SENA / sVAE+ / SAMS-VAE**: These causal or VAE methods do not as clearly distinguish between background and perturbation response, often resulting in entangled invariant information; PerturbedVAE improves this through alignment and capacity allocation.
- **vs additive linear model / GEARS**: Additive baselines are strong for average response in Norman2019, and GEARS directly learns graph mappings from perturbation to expression; PerturbedVAE's advantage lies in simultaneously modeling single-cell variation and latent perturbation mechanisms.
- **Insight**: In other scientific ML intervention prediction tasks (e.g., drug combinations, protein perturbations, or material processing interventions), one could first identify dominant invariant factors and then embed sparse intervention effects into a structured latent mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining the perturbation suppression hypothesis with a structured VAE; the problem definition and motivation are clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes simulations, real Perturb-seq, comparisons with FM/simple baselines/CRL, and multiple ablations, though cross-dataset extrapolation could be stronger.
- Writing Quality: ⭐⭐⭐⭐ Strong connection between theory and experiments; honest discussion of additive baselines. Technical density makes some sections difficult to read.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for single-cell perturbation modeling, especially as a reminder not to blindly trust that universal foundation model representations preserve sparse intervention signals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](../../ICLR2026/computational_biology/scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[ACL 2026\] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](../../ACL2026/computational_biology/aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)

</div>

<!-- RELATED:END -->
