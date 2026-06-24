---
title: >-
  [Paper Note] MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning
description: >-
  [ICML 2025][Computational Biology][Multi-fidelity modeling] The MF-LAL framework is proposed to unify multi-fidelity surrogate models and molecular generative models into a hierarchical latent space. Through active learning, it efficiently integrates molecular docking (low-fidelity) and binding free energy calculation (high-fidelity) oracles, generating candidate drug molecules with significantly improved binding free energies (averaging an approximately 50% improvement in AB…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Multi-fidelity modeling"
  - "active learning"
  - "drug generation"
  - "latent space optimization"
  - "binding free energy"
date: 2026-05-08
content_hash: 1d1a984dffa3e322
---

# MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning

**Conference**: ICML 2025  
**arXiv**: [2410.11226](https://arxiv.org/abs/2410.11226)  
**Code**: [Rose-STL-Lab/MF-LAL](https://github.com/Rose-STL-Lab/MF-LAL)  
**Area**: Medical Image / Drug Discovery  
**Keywords**: Multi-fidelity modeling, active learning, drug generation, latent space optimization, binding free energy

## TL;DR

The MF-LAL framework is proposed to unify multi-fidelity surrogate models and molecular generative models into a hierarchical latent space. Through active learning, it efficiently integrates molecular docking (low-fidelity) and binding free energy calculation (high-fidelity) oracles, generating candidate drug molecules with significantly improved binding free energies (averaging an approximately 50% improvement in ABFE score).

## Background & Motivation

Current molecular generative models in drug discovery primarily rely on **molecular docking** as an oracle to evaluate the activity of generated compounds. However, compounds with high docking scores do not always exhibit real biological activity in actual experiments, limiting the practical utility of these generative models.

A more accurate affinity prediction method—**binding free energy calculation based on molecular dynamics simulations**—is considered the gold standard for predicting affinity, but a single evaluation takes hours to days, which is computationally prohibitive and cannot be directly incorporated into the iterative generator loop.

**Multi-fidelity surrogate modeling** can combine multiple oracles of different accuracy/cost levels. However, existing methods (such as MF-AL-GFN) train the generative model and surrogate model **separately**. This prevents the generative model from recognizing the discrepancy in the optimal query compound distribution across different fidelity levels, thereby limiting query efficiency.

The key insight of MF-LAL is that **the optimal distribution of molecules may vary across different fidelity levels**. Thus, independent latent spaces and decoders are designed for each fidelity level, while cross-fidelity information is shared through a hierarchical network.

## Method

### Overall Architecture

MF-LAL consists of three core components:

1. **Hierarchical Latent Space Encoding** (Left): Encodes molecules into a sequence of latent spaces ordered by fidelity level.
2. **Surrogate Modeling and Back-Optimization** (Center): Learns surrogate models in each latent space to predict oracle outputs and performs gradient-based optimization to generate high-performing molecules.
3. **Active Learning Loop** (Right): Submits generated molecules to the corresponding fidelity oracle, using the results to retrain latent space representations and surrogate models.

This work utilizes 4 levels of fidelity oracles:
- **$f_1$**: Linear regression (~0.1s, ROC-AUC 0.59/0.68)
- **$f_2$**: AutoDock4 (~4s, ROC-AUC 0.73/0.72)
- **$f_3$**: Ensemble AutoDock4 (~44-68s, ROC-AUC 0.80/0.80)
- **$f_4$**: Absolute binding free energy ABFE (~9.3h, ROC-AUC 0.92/0.89)

### Key Designs

#### 1. Multi-Fidelity Latent Space Hierarchy

Use a single probabilistic encoder $q_\phi$ to encode a molecule $x$ (represented as a SELFIES string) into the lowest-fidelity latent space $\mathbf{z}_1 \sim \mathcal{N}(\mu_1, \sigma_1)$.

Information is propagated between adjacent fidelity latent spaces through a set of probabilistic mapping networks $h_{\xi_1}, \ldots, h_{\xi_{K-1}}$:

$$\mathbf{z}_{k+1} \sim \mathcal{N}(\mu_{k+1}, \sigma_{k+1}), \quad (\mu_{k+1}, \sigma_{k+1}) = h_{\xi_k}(\mathbf{z}_k)$$

Each fidelity level has an independent decoder $p_{\theta_k}(\cdot | \mathbf{z}_k)$ for reconstructing molecules. This outperforms shared single latent space approaches in two aspects:
- **Higher generation quality**: Specialized decoders are dedicated to each fidelity level.
- **More accurate surrogate modeling**: Each latent space can be organized independently to optimize prediction at that level.

#### 2. Surrogate Modeling: SVGP

Each fidelity level uses a **Stochastic Variational Gaussian Process (SVGP)** as a surrogate model $\hat{f}_k$ to predict oracle outputs from the corresponding latent vector $\mathbf{z}_k$. A 4-layer Deep Kernel is used to encode inputs, with a Matern kernel as the covariance function. Reasons for choosing SVGP: fast training speed, support for mini-batch training, and capability to produce uncertainty estimates.

#### 3. Novel Likelihood-Constrained Generation

When generating molecules in high-fidelity latent spaces, a **likelihood term** constraint is introduced: ensuring that molecules generated at fidelity $k$ had also scored highly at fidelity $k-1$. Specifically:
- First, generate $M$ high-scoring molecules at fidelity $k-1$.
- Map them to the latent space of fidelity $k$ through $h_{\xi_{k-1}}$ to form a Gaussian mixture distribution.
- Maximize the likelihood of the candidate generation point within this mixture distribution.

$$P(\mathbf{z}_k^{(i)} | \{(\mu_k^{(j)}, \sigma_k^{(j)})\}_{j=1}^M) = \sum_{j=1}^M \frac{1}{\sqrt{2\pi (\sigma_k^{(j)})^2}} \exp\left(-\frac{(\mathbf{z}_k^{(i)} - \mu_k^{(j)})^2}{2(\sigma_k^{(j)})^2}\right)$$

This design dramatically narrows down the chemical space that the high-fidelity oracle needs to search, making expensive ABFE calculations feasible.

#### 4. Stepped Active Learning Strategy

Queries begin from the lowest fidelity $k=1$, and $k$ is permanently incremented when the GP posterior variance $\Sigma_{\lambda_k}(\mathbf{z}_k) < \gamma_k$. The Upper Confidence Bound (UCB) is used as the acquisition function:

$$a(\mathbf{z}_k^{(i)}, k) = m_{\lambda_k}(\mathbf{z}_k^{(i)}) + \beta \cdot \Sigma_{\lambda_k}(\mathbf{z}_k^{(i)}) - \|\mathbf{z}_k^{(i)}\|_2^2$$

During active learning, $\beta=1$ (exploration + exploitation), while during inference, $\beta=0$ (pure exploitation). The L2 regularization term ensures that the generated compounds remain close to the drug-like molecule distribution of the training set.

### Loss & Training

Jointly minimize the ELBO and the GP's marginal log-likelihood (MLL):

$$L(\phi, \xi_{k-1}, \theta_k, \lambda_k; k, x, y) = \underbrace{\mathbb{E}_{\mathbf{z}_k \sim g(\cdot|x)} \log \frac{p_{\theta_k}(x, \mathbf{z}_k)}{g(\mathbf{z}_k|x)}}_{\text{ELBO}} + \underbrace{\int p(y|\hat{f}_k(\mathbf{z}_k)) p(\hat{f}_k(\mathbf{z}_k)|\mathbf{z}_k) d\hat{f}_k}_{\text{MLL}}$$

Key training details:
- Losses are evaluated at fidelity $k$, but **backpropagated through all lower fidelities**.
- At each active learning step, training restarts from scratch to convergence (Adam, lr=0.0001).
- For molecular generation, gradient optimization is performed using Adam (lr=0.1, 100 epochs).
- Encoder/decoder/mapping networks: 3-layer fully connected, ReLU, 512-dimensional hidden layers, 64-dimensional latent spaces.
- Cosine similarity diversity penalty is added during generation.

## Key Experimental Results

### Main Results

Evaluated on two cancer-related human proteins (BRD4(2) and c-MET) under a fixed 7-day computational budget:

| Method | BRD4(2) Mean ABFE | BRD4(2) Top-3 | c-MET Mean ABFE | c-MET Top-3 | Type |
|------|-------------------|---------------|-----------------|-------------|------|
| **MF-LAL (ours)** | **-6.2 ± 3.9** | **-12.0 / -10.2 / -9.8** | **-6.7 ± 3.1** | **-12.9 / -7.9 / -7.7** | Multi-fidelity |
| Pocket2Mol | -4.3 ± 3.8 | -9.8 / -8.7 / -8.0 | -2.2 ± 4.2 | -4.5 / -3.9 / -3.2 | 3D Structure |
| MF-AL-PPO | -2.8 ± 2.5 | -9.2 / -6.5 / -5.2 | -4.2 ± 2.8 | -6.6 / -5.8 / -5.5 | Multi-fidelity |
| REINVENT (ABFE) | -3.9 ± 3.4 | -8.7 / -8.3 / -8.2 | -2.9 ± 3.7 | -6.5 / -5.8 / -5.1 | Single-fidelity |
| DecompDiff | -2.7 ± 4.0 | -8.9 / -8.1 / -7.5 | -1.9 ± 6.4 | -8.0 / -5.1 / -2.7 | 3D Structure |
| MF-AL-GFN | -2.5 ± 2.2 | -6.5 / -5.8 / -5.1 | -3.1 ± 1.8 | -5.5 / -4.5 / -4.1 | Multi-fidelity |

MF-LAL in an extended evaluation of 40 generated compounds: BRD4(2) mean ABFE of **-6.3** (p<0.05), c-MET mean ABFE of **-7.1** (p<0.05), with **8** and **6** active scaffolds respectively, significantly outperforming the baselines.

### Ablation Study

| Configuration | BRD4(2) Mean | BRD4(2) Top-3 | c-MET Mean | c-MET Top-3 | Description |
|------|-------------|---------------|------------|-------------|------|
| MF-LAL (Full) | -6.2 | -12.0/-10.2/-9.8 | -6.7 | -12.9/-7.9/-7.7 | Baseline |
| w/o Fidelity 1 | -6.1 | -7.7/-7.6/-7.4 | -6.0 | -8.8/-7.0/-6.0 | Top severely drops |
| w/o Fidelity 2 | -5.1 | -8.5/-6.5/-6.0 | -5.2 | -8.0/-7.3/-6.1 | Obvious degradation |
| w/o Fidelity 3 | -4.2 | -9.2/-5.9/-5.7 | -4.2 | -9.8/-7.1/-6.1 | Degraded |
| w/o Fidelity 4 | -2.4 | -8.6/-4.3/-3.4 | -3.1 | -7.6/-6.7/-5.1 | Most severe degradation |
| **w/o Likelihood Term** | **-3.4** | -11.9/-9.7/-9.0 | **-3.8** | -10.9/-7.7/-6.3 | **Critical component** |
| Transformer Enc/Dec | -6.1 | -11.5/-9.9/-9.0 | -6.5 | -11.6/-7.6/-6.5 | Minor difference |
| GCN Enc/Dec | -5.9 | -10.9/-10.1/-9.0 | -6.1 | -11.1/-7.5/-6.5 | Slightly worse |

### Key Findings

1. **All fidelity levels contribute**: Removing any fidelity level leads to performance drops, with the removal of the highest fidelity (ABFE) having the greatest impact.
2. **Likelihood constraint is key**: Removing the likelihood term causes the mean ABFE to plummet from -6.2 to -3.4 (on BRD4(2)), indicating that restricting the high-fidelity search space is crucial.
3. **Simple encoders suffice**: Fully connected networks perform comparably to Transformer/GCN encoders/decoders; complex architectures are unnecessary.
4. **Other multi-fidelity methods perform poorly**: MF-AL-GFN and MF-AL-PPO perform similarly to single-fidelity methods, showing that leveraging multi-fidelity requires architectures tailored for multi-fidelity generation like MF-LAL.
5. Realized drug-like properties of generated compounds are good: mean QED of 0.59-0.63, SAscore of 3.5-3.6, and diversity (1 - mean Tanimoto similarity) of 0.81-0.83.

## Highlights & Insights

- **Core Innovation**: Seamlessly transitions the relationship between generative models and multi-fidelity surrogate models from "separated" to "unified", employing independent latent spaces and decoders for each fidelity level while sharing information via a hierarchical mapping network.
- **Provocative Likelihood Constraint**: Exploring the high-fidelity oracle only within regions already proven promising at low fidelity represents an elegant computational budget allocation strategy.
- **Highly Practical**: Successfully integrates ABFE (the gold-standard affinity prediction) into the molecular generation workflow for the first time, achieving results significantly superior to baselines within a fixed 7-day budget.
- **Query Synthesis > Candidate Selection**: The poor performance of the MF-GP + ZINC250k baseline shows that generating new queries is more effective than selecting from a fixed set.

## Limitations & Future Work

1. **Limited pool of oracles**: Only 4 oracles are used; incorporating more intermediate fidelity levels might further enhance performance.
2. **Insufficient synthetic accessibility guarantees**: SAscore is known to be an imperfect evaluation metric for synthetic feasibility.
3. **Overestimated SVGP posterior variance**: SVGP may overestimate posterior variance far from inducing points, biasing generation toward out-of-distribution molecules.
4. **Low-fidelity "bottleneck" effect**: The design may miss scaffolds that score poorly at low fidelity but perform exceptionally well at high fidelity.
5. **Hyperparameter sensitivity**: The ratio of KLD to reconstruction and the diversity coefficient require meticulous tuning.
6. **Evaluation variance**: The 7-day fixed runtime constraint on experiments may result in high variance, as models might not have fully trained to convergence.

## Related Work & Insights

- **LIMO** (Eckmann et al., 2022): Prior work from the same research group using single-fidelity VAE + docking optimization, which is the direct predecessor of MF-LAL.
- **MF-AL-GFN** (Hernandez-Garcia et al., 2023): GFlowNet + separate multi-fidelity GP surrogate, representing the SOTA of the "separated" paradigm.
- **REINVENT** (Olivecrona et al., 2017): A classic method for RL molecular generation.
- **Pocket2Mol / DecompDiff**: 3D structure-driven drug design methods, which do not require oracles but cannot avoid inaccuracies inherent in docking evaluations.
- Insight: The concepts of hierarchical latent spaces and likelihood constraints can be extended to other multi-fidelity scientific computing scenarios (e.g., materials design, protein engineering).

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The design unifying multi-fidelity surrogates and generative models is novel; likelihood-constrained generation is original. |
| Technical Depth | 4 | Involves deep integration across VAE, GP, active learning, and molecular dynamics. |
| Experimental Thoroughness | 4 | Evaluated on two real protein targets, matching against 13 baselines, with extensive ablations and statistical tests. |
| Practical Value | 4 | Successfully leverages gold-standard ABFE in generative models for the first time, offering genuine value in drug discovery. |
| Writing Quality | 4 | Well-structured, detailed description of methodology, with intuitive illustrations. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](../../NeurIPS2025/computational_biology/amortized_active_generation_of_pareto_sets.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](../../NeurIPS2025/computational_biology/prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[ICML 2025\] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)

</div>

<!-- RELATED:END -->
