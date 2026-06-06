---
title: >-
  [Paper Note] CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences
description: >-
  [ICML 2026][Computational Biology][Antibody Evolution] CoSiNE models the antibody affinity maturation process using a neural network-parameterized Conditional Site-Independent Continuous-Time Markov Chain (CTMC). It capt…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Antibody Evolution"
  - "Continuous-Time Markov Chain"
  - "Affinity Maturation"
  - "Variant Effect Prediction"
  - "Classifier-Guided Sampling"
date: 2026-05-08
content_hash: 04121d40cbfc69f5
---

# CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences

**Conference**: ICML 2026  
**arXiv**: [2602.18982](https://arxiv.org/abs/2602.18982)  
**Code**: https://github.com/thematrixmaster/cosine  
**Area**: Scientific Computing/Computational Biology  
**Keywords**: Antibody Evolution, Continuous-Time Markov Chain, Affinity Maturation, Variant Effect Prediction, Classifier-Guided Sampling  

## TL;DR

CoSiNE models the antibody affinity maturation process using a neural network-parameterized Conditional Site-Independent Continuous-Time Markov Chain (CTMC). It captures inter-site epistasis while maintaining tractability and enables antigen-specific antibody optimization via Guided Gillespie sampling, surpassing existing language and evolutionary models in zero-shot variant effect prediction.

## Background & Motivation

**Background**: Deep learning methods in antibody engineering are primarily divided into two categories: protein language models (e.g., ESM-2, AbLang-2), which learn the marginal distribution $p(x)$ and capture complex inter-site epistasis but treat sequences as i.i.d. samples ignoring evolutionary dynamics; and classical phylogenetic models (e.g., WAG, LG), which explicitly model the evolutionary process but assume site independence, failing to capture epistatic interactions.

**Limitations of Prior Work**: The performance of language models stems partially from memorizing conserved germline residues rather than truly understanding affinity maturation. The independent-site assumption in classical evolution models is necessary because matrix exponentiation in the full sequence space $|\mathcal{A}|^L = 20^L$ is computationally infeasible ($O(|\mathcal{S}|^3)$ complexity), but reducing this to $O(L|\mathcal{A}|^3)$ loses epistatic information.

**Key Challenge**: The trade-off between expressivity and computational feasibility—CTMC in the full sequence space captures all epistasis but is infeasible, while independent-site models are feasible but lack expressivity. Language models are expressive but lack evolutionary time modeling.

**Goal**: To design a unified framework that maintains the computational efficiency of independent-site models while capturing epistasis through sequence-context conditioning and explicitly modeling continuous-time evolutionary dynamics.

**Key Insight**: The authors observe that if each site's rate matrix $Q_\ell$ depends on the complete parent sequence $x$ (rather than just the site itself), inter-site dependencies can be implicitly encoded via a neural network while maintaining factorized transition probabilities. Mathematically, this constitutes a first-order approximation of the sequential point mutation process in the full sequence space, where the error grows quadratically with branch length—an ideal fit for antibody affinity maturation, which is characterized by short branches.

**Core Idea**: Use a neural network to output site-specific rate matrices conditioned on the full sequence, achieving a "Conditional Site-Independent" CTMC that merges the temporal dynamics of evolutionary modeling with the epistasis-capturing capabilities of language models.

## Method

### Overall Architecture

The input to CoSiNE is a parent antibody sequence $x$. A neural network (initialized from ESM-2 150M) outputs $L$ site-specific rate matrices $Q_\theta(x)_\ell \in \mathbb{R}^{|\mathcal{A}| \times |\mathcal{A}|}$. Given an evolutionary time $t$, transition probabilities for each site are computed via matrix exponentiation, and their product yields the full sequence transition probability. The model is trained on approximately 2 million evolutionary transitions (parent-child pairs) extracted from about 120,000 clonal trees. During inference, evolutionary trajectories are sampled using the Gillespie algorithm, with antigen-specific optimization enabled via classifier guidance.

### Key Designs

1.  **Conditional Site-Independent Transition Probabilities**:
    - **Function**: Captures inter-site epistasis while maintaining factorized feasibility.
    - **Mechanism**: The transition probability is defined as $p_\theta(y|x,t) = \prod_{\ell=1}^{L} \exp(t Q_\theta(x)_\ell)_{x_\ell, y_\ell}$, where $Q_\theta$ is a neural network conditioned on the full parent sequence $x$. When the rate matrix satisfies $(Q_\theta(x)_\ell)_{x_\ell, y_\ell} = \mathbf{Q}_{x,y}$, this model serves as a first-order approximation of the sequential point mutation process, with an $L_1$ error upper bound of $(\lambda t)^2$ (where $\lambda$ is the maximum exit rate). This implies minimal approximation error for short branches typical in antibody evolution.
    - **Design Motivation**: Traditional independent-site models (WAG/LG) share a single $Q$ across all sites, losing all epistatic information; CoSiNE allows each site's $Q_\ell$ to depend on the full sequence, theoretically capturing epistasis at the first order.

2.  **Selection-Mutation Decoupled Fitness Inference**:
    - **Function**: Extracts pure selection signals from the learned evolutionary model for zero-shot variant effect prediction (VEP).
    - **Mechanism**: Based on the mutation-selection framework, the observed transition rate is decomposed as $Q_{xy} = k \mu_{xy} P_{\text{fix}}(x \to y)$, where $\mu_{xy}$ is the neutral somatic hypermutation (SHM) rate. The selection score is defined as $\text{Score}(x \to y) = \log p_\theta(y|x,t) - \log q(y|x,t) \approx \log P_{\text{fix}}(x \to y) + C$, which is the difference between the CoSiNE log-likelihood and the log-likelihood of a pre-trained SHM model (Thrifty). This removes SHM bias to extract pure natural selection signals.
    - **Design Motivation**: Language models use perplexity to evaluate fitness, which is confounded by germline conservation. DASM requires manual truncation of selection scores; CoSiNE naturally derives selection scores via log-likelihood ratios without heuristic constraints.

3.  **Guided Gillespie Antigen-Specific Sampling**:
    - **Function**: Guides CoSiNE to generate antibody sequences with high affinity for specific antigens during inference.
    - **Mechanism**: Based on classifier guidance theory from discrete diffusion models, the guided rate matrix is defined as $(\mathbf{Q}_z^{(\gamma)})_{x,y} = [p(z|y)/p(z|x)]^\gamma \mathbf{Q}_{x,y}$. By approximating $p(z|y)$ with a Gaussian assumption using a binding affinity predictor and applying Taylor Approximation (TAG), the cost is reduced from $L \times (|\mathcal{A}|-1)$ predictor calls to just 1 gradient calculation per step, achieving a 500x speedup. Adaptive thresholding $r_0 = \mu_{\theta_z}(x)$ is used to prevent vanishing guidance weights.
    - **Design Motivation**: CoSiNE's training data lacks antigen information. Unlike discrete diffusion or flow matching, CTMC has no terminal time constraints, so the predictor does not need training on noisy sequences and can be a standard sequence-property predictor.

### Loss & Training

The model is initialized with an ESM-2 150M checkpoint, replacing the language modeling head with a rate matrix output head using softplus activation. It uses the AdamW optimizer (learning rate $2.5 \times 10^{-4}$), a polynomial decay schedule, and BF16 mixed-precision training, converging in about 1 day on a single A100 GPU. Chain-break tokens are inserted between heavy and light chains to handle paired antibodies.

## Key Experimental Results

### Main Results (Zero-Shot VEP Evaluation)

Evaluated on 4 DMS datasets from the FLAb2 benchmark using Spearman correlation:

| Dataset | CoSiNE | DASM | ESM2-150M | ProGen2-S | PRISM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Koenig Expr (H) | **0.613** | 0.596 | 0.413 | 0.407 | 0.069 |
| Koenig Expr (L) | 0.508 | 0.474 | 0.485 | **0.513** | 0.129 |
| Adams Binding | **0.464** | 0.270 | -0.112 | -0.024 | 0.297 |
| Koenig Bind (H) | **0.456** | 0.415 | 0.112 | 0.098 | 0.005 |
| Koenig Bind (L) | **0.371** | 0.327 | 0.266 | 0.332 | 0.061 |

CoSiNE achieves the best performance on 6 out of 7 datasets, notably leading significantly in cross-species scenarios (Adams mouse antibodies: 0.464 vs runner-up 0.297).

### Ablation Study

| Ablation Config | Effect | Explanation |
| :--- | :--- | :--- |
| No SHM correction (only $\log p_\theta$) | Correlation drops across all datasets | Decoupling mutation-selection is critical for VEP |
| Single chain only (remove paired chain) | Significant drop in some datasets | Inter-chain epistasis contributes to predictions |
| Training from scratch (no ESM2) | Avg $\Delta\rho = 0.041$ | Evolutionary training objective contributes most predictive power |
| Different branch lengths $t \in [0.1, 0.4]$ | $\Delta\rho \leq 0.045$ | Selection scores are robust to choice of $t$ |
| Local CDR optimization (5-mut budget) | $\Delta\text{Bind}_{\text{max}} = 0.395$ | Outperforms Genetic Algorithms and PoE methods |
| Guided Gillespie ($\gamma=5$) | Generated affinity overlaps with real binders | Maintains structural quality (pLDDT) and humanness (OASis) |
| TAG vs. Exact Guidance | 500x speedup, no sig. performance diff | First-order Taylor approximation is effective |

## Highlights & Insights

1.  **Elegant Fusion of Theory and Practice**: Proposition 4.1 provides a rigorous upper bound of $O(t^2)$ for the approximation error of the conditional site-independent model relative to the full sequence CTMC, making it biologically ideal for short-branched antibody evolution.
2.  **Bridging Discrete Diffusion and Classical Evolution**: Transfers classifier guidance from discrete diffusion to the classical CTMC framework, where the predictor requires no retraining on noisy data due to the lack of boundary time constraints in CTMC.
3.  **Categorical Jacobian analysis** reveals learned intra-chain and inter-chain CDR coupling, consistent with the biological structure of antibody-antigen binding pockets.

## Limitations & Future Work

1.  Error for the first-order approximation increases on long branches, limiting applicability to slowly evolving proteins.
2.  The current framework ignores insertions and deletions (indels) and can only process fixed-length sequences—acceptable for antibodies but a bottleneck for general proteins.
3.  Guided Gillespie depends on the quality of the affinity predictor; high guidance strength ($\gamma \geq 10$) may exploit predictor uncertainty to generate over-optimized sequences.

## Related Work & Insights

- **DASM** (Matsen 2025): Also decouples SHM and selection but requires manual truncation of selection scores; CoSiNE naturally maintains mathematical consistency via log-likelihood ratios.
- **SiteRM** (Prillo 2024): Performs well on ProteinGym using per-site independent rate matrices but lacks context conditioning.
- **PRISM** (Kim 2026): Uses auxiliary heads to predict germline/mutant states but simplifies evolution to binary classification.
- **Insight**: The conditional site-independent framework can be extended to evolutionary modeling and design for other protein families.

## Rating

- Novelty: 9/10 — First model to merge neural CTMC with classifier guidance for antibody evolution with solid theoretical contributions.
- Experimental Thoroughness: 9/10 — Comprehensive validation across VEP, guided sampling, ablations, and cross-species generalization.
- Writing Quality: 9/10 — Clear theoretical derivations and systematic experimental organization.
- Value: 8/10 — Direct application value for antibody engineering; the framework is generalizable but currently restricted to antibody contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)
- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICLR 2026\] Intrinsic Lorentz Neural Network](../../ICLR2026/computational_biology/intrinsic_lorentz_neural_network.md)
- [\[ICML 2026\] Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators](protein_language_model_embeddings_improve_generalization_of_implicit_transfer_op.md)
- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)

</div>

<!-- RELATED:END -->
