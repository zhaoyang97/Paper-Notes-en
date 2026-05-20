---
title: >-
  [Paper Note] Hidden Breakthroughs in Language Model Training
description: >-
  [ICLR 2026][Interpretability][Training Dynamics] This paper proposes POLCA (Projection Oriented Loss Change Allocation)—a method that decomposes per-sample loss changes along any orthogonal basis within a low-rank traini…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Training Dynamics"
  - "Hidden Phase Transitions"
  - "Loss Decomposition"
  - "Unsupervised Interpretability"
  - "Hessian Eigenvectors"
date: 2026-05-08
content_hash: 73532c3347ec9a3d
---

# Hidden Breakthroughs in Language Model Training

**Conference**: ICLR 2026
**arXiv**: [2506.15872](https://arxiv.org/abs/2506.15872)  
**Code**: [GitHub](https://github.com/skangasl/POLCA)  
**Area**: Interpretability
**Keywords**: Training Dynamics, Hidden Phase Transitions, Loss Decomposition, Unsupervised Interpretability, Hessian Eigenvectors

## TL;DR

This paper proposes POLCA (Projection Oriented Loss Change Allocation)—a method that decomposes per-sample loss changes along any orthogonal basis within a low-rank training subspace—to reveal numerous hidden conceptual breakthroughs from seemingly smooth training loss curves. The approach inverts the paradigm of training interpretability from "define skills first, then observe" to "decompose first, then discover skills automatically."

## Background & Motivation

**Background**: Language model training is accompanied by various abrupt phase transitions—emergence of in-context learning, acquisition of syntactic structures, appearance of hierarchical generalization, etc. These transitions are valuable for understanding model learning mechanisms and guiding training strategies (e.g., data selection, learning rate scheduling). In practice, however, aggregate loss curves are extremely smooth, and a large number of phase transitions are obscured beneath a single scalar metric.

**Limitations of Prior Work**: Nearly all existing methods for identifying phase transitions adopt a **top-down** paradigm—researchers first predefine a concept or skill (e.g., "carrying," "subject-verb agreement") and then monitor its dynamics during training. This approach fails to discover skills not defined in advance, and cannot handle cases where a single sample depends on multiple skills simultaneously (polygenic scaling effects).

**Key Challenge**: A smooth aggregate loss curve ≠ absence of breakthroughs. Saxe et al. (2019) theoretically predicted that multiple sigmoidal phase transitions occurring at different times superpose into a smooth curve. The missing piece is a tool to recover these hidden phase transitions from the smooth curve.

**Goal**: (1) How can conceptual breakthroughs during training be discovered automatically without predefining skills? (2) How can the entanglement of a single sample undergoing multiple skill breakthroughs simultaneously be handled? (3) How can loss changes be decomposed onto interpretable gradient directions?

**Key Insight**: The authors observe that the training subspace is low-rank and that linearly interpolated checkpoints preserve semantic capabilities, suggesting that linear decomposition is meaningful at the conceptual level. Projecting loss changes onto the high-curvature directions of the Hessian may allow each direction to correspond to the acquisition of a distinct "skill," thereby decoupling a single sample's loss change into contributions along independent directions.

**Core Idea**: Decompose each sample's loss change along an orthogonal basis constructed from Hessian eigenvectors, then cluster the projected loss trajectories to unsupervisedly discover conceptual breakthroughs hidden within smooth training curves.

## Method

### Overall Architecture

The POLCA pipeline consists of three steps: the input is a sequence of training checkpoints and a validation dataset; the output is clusters of data grouped by "shared learning events" and their corresponding breakthrough timestamps. The procedure is: (1) construct an interpretable orthogonal basis from the Hessian matrix; (2) apply POLCA decomposition to project each sample's loss change along this basis; (3) cluster samples by the similarity of their projected loss trajectories.

### Key Designs

1. **Iterative Construction of the Hessian Orthogonal Basis**:

    - Function: Construct a low-rank orthogonal basis that captures the principal gradient movement directions throughout training.
    - Mechanism: At $T$ training checkpoints, the eigenvectors of the Hessian matrix are computed sequentially. At each checkpoint, the Hessian is first projected onto the null space of the existing basis (excluding already-captured directions), and then the top $k$ eigenvectors of the projected Hessian are added to the basis. This yields a $Tk$-dimensional subspace. A critical filtering step removes "oscillating directions"—basis vectors whose average projected loss increases rather than decreases over the course of training—since these represent only local oscillations rather than long-term learning.
    - Design Motivation: The top Hessian eigenvectors correspond to directions of maximum curvature and often represent critical decision boundaries. Null-space projection ensures that each new checkpoint captures novel information, avoiding directional redundancy. Oscillation filtering ensures that the retained directions all represent genuine learning progress.

2. **POLCA Loss Decomposition (First-Order + Second-Order)**:

    - Function: Attribute the loss change of each sample during training to individual basis vector directions.
    - Mechanism: Three key modifications are made relative to classical LCA (Loss Change Allocation). **First**, LCA decomposes along individual parameter axes, whereas POLCA admits arbitrary orthogonal basis vectors $b$. **Second**, LCA aggregates over the entire dataset, whereas POLCA operates at the level of individual samples $x$. **Third**, because the basis is constructed from Hessian eigenvectors (high-curvature directions), the error of a first-order Taylor approximation can be large; a second-order correction term is therefore introduced. The first-order POLCA is $\langle b, \nabla_\theta L(x;\theta_t)\rangle \langle b, \theta_{t+1}-\theta_t\rangle$, and the second-order term approximates the per-sample Hessian by scaling global Hessian eigenvalues proportionally to per-sample loss changes, avoiding the prohibitive cost of computing per-sample Hessians.
    - Design Motivation: Individual parameter axes carry almost no semantic meaning in high-dimensional space, whereas Hessian eigenvector directions have clear geometric interpretations (maximum curvature = potential decision boundaries). Per-sample granularity enables the discovery of breakthroughs that affect only specific data subsets. The second-order correction provides a tighter Lagrange error bound in theory.

3. **HDBSCAN Clustering on Projected Loss Trajectories**:

    - Function: Group data points that undergo similar learning events, automatically discovering subsets of samples sharing a common skill.
    - Mechanism: For each basis vector $b$ and sample $x$, the cumulative projected loss $L_b(x, \theta_t) = \sum_{i=0}^{t-1} \text{POLCA}(x, b; \theta_i)$ is computed, forming a one-dimensional time series. HDBSCAN is applied separately to these trajectories for each basis vector. Before clustering, samples whose projected loss increases are filtered out (as these do not represent positive learning along that direction). After clustering, POS tag templates are used to automatically generate interpretable labels for each cluster.
    - Design Motivation: HDBSCAN handles variable-density clusters (curves that are similar in shape but differ in absolute value) and can identify outliers. Clustering separately per basis vector allows a single sample to be assigned to different clusters along different directions, thereby handling the case where a single token depends on multiple skills simultaneously.

### Formal Definition of Hidden Breakthroughs

A breakthrough point is defined as the time of maximum loss acceleration: $\text{break}(f, x, \Delta) = \arg\max_t [f(x, t+\Delta) - f(x,t)] - [f(x,t) - f(x,t-\Delta)]$. When the average breakthrough onset of a cluster exceeds a threshold $\tau$ (marking the time at which the aggregate loss enters its flat region), the breakthrough is considered "hidden"—a sharp change in projected loss occurring even while the aggregate loss curve is flat.

## Key Experimental Results

### Main Results I: Synthetic Arithmetic Addition Task

A 3-layer, 9M-parameter Transformer is trained on 3-digit addition (e.g., "342+578=920"). The data involves 4 digit-position skills and 1 carrying skill. Digit-position skills exhibit large differences in loss curves (amenable to direct clustering), whereas the carrying skill is invisible in the exact loss.

| Decomposition Strategy | Max Carrying Homogeneity↑ | Hidden Breakthrough Ratio↑ | Recovers Carrying Skill |
|---------|:---:|:---:|:---:|
| Exact Loss | 0.514 | 0.0 | ✗ |
| Exact Loss Change | 0.524 | 0.0 | ✗ |
| LCA (Lan et al., 2020) | 0.792 | 0.019 | Partially |
| **POLCA (Ours)** | **0.973** | **0.355** | **✓** |

POLCA recovers both digit-position skills and the carrying skill using only the first two basis vectors; the carrying homogeneity of the cluster for basis vector #2 reaches 0.90. By contrast, exact-loss clustering entirely fails to distinguish samples that require carrying from those that do not.

### Main Results II: English Language Modeling

A 3-layer, 40M-parameter model is trained on Wikipedia data. Of 30 basis vectors, 26 remain after filtering oscillating directions, and 22 of these yield at least one easily annotatable interpretable cluster.

| Basis Vector | Cluster Label | Example Contents | Breakthrough Type |
|:---:|------|------|------|
| #13 Cluster 1 | Preposition after sentence-initial subordinate clause | "from", "and", "after" | Hidden breakthrough |
| #13 Cluster 2 | Consecutive newlines | "\n\n" at paragraph end | Early breakthrough |
| #13 Cluster 3 | Comma after parenthetical phrase | Enumeration items after comma | Hidden breakthrough |
| #23 Cluster 1 | Appositional noun phrases | e.g., "Air Force Instruction 36-2406: Officer and..." | Hidden breakthrough |
| #23 Cluster 2 | Non-appositional post-comma phrases | List items, year enumerations | Reverse hidden breakthrough |

The most notable finding is a "mirror phenomenon" in basis vector #23: Cluster 1 (appositional noun phrases) and Cluster 2 (non-appositional post-comma phrases) exhibit completely opposite movements in projected loss—the acquisition of the apposition skill is accompanied by a temporary decline in the model's predictive accuracy for non-appositional post-comma tokens, indicating that these two syntactic constructions share the same gradient direction but are learned competitively.

### Ablation Study

| Configuration | Carrying Homogeneity | Notes |
|---------|:---:|------|
| Full POLCA (second-order) | 0.973 | Best |
| POLCA first-order only | ~0.96 | Marginal difference, but theoretically looser bound |
| Without oscillation filtering | Significantly lower | Oscillating directions produce noisy clusters |
| K-Means instead of HDBSCAN | Worse | K-Means cannot handle variable density or outliers |
| LCA (parameter-axis decomposition) | 0.792 | Parameter axes far less semantically meaningful than Hessian directions |

### Key Findings

- **Phase transitions are ubiquitous**: The conjecture of Nanda et al. (2023) is validated. Even in the late stages of training when the aggregate loss is completely flat, 35.5% of POLCA clusters exhibit hidden breakthroughs.
- **Skill separation occurs naturally in gradient space**: Different skills (e.g., carrying vs. digit position) are learned along distinct Hessian directions, providing an operational definition of "skill" in terms of gradient geometry.
- **Competitive learning patterns**: Certain syntactic constructions (apposition vs. enumeration) exhibit antagonistic learning dynamics along the same basis vector, with loss changes moving in opposite directions.
- **Effectiveness of linear decomposition**: Despite relying solely on linear methods, the recovered clusters are highly interpretable (22/26 basis vectors yield annotatable clusters), supporting the hypothesis of linear separability within the training subspace.

## Highlights & Insights

- **Paradigm inversion—from top-down to bottom-up**: The analysis of training dynamics is inverted from "hypothesize skills first, then verify" to "decompose first, then discover automatically." This parallels what Sparse Autoencoders (SAEs) do in representation space: SAEs unsupervisedly discover features in representations, while POLCA unsupervisedly discovers skills in the training process. This symmetry suggests that the two approaches are complementary.
- **Elegance of the POLCA decomposition**: Only three precise modifications are made to LCA (arbitrary basis → per-sample → second-order), each with a clear theoretical motivation, keeping the overall method tractable. The second-order correction term is approximated by scaling global Hessian eigenvalues proportionally to per-sample loss changes, circumventing the catastrophic cost of computing per-sample Hessians.
- **"Mirror cluster" phenomenon**: The projected losses of two clusters on the same basis vector move in opposite directions, indicating that the model temporarily "sacrifices" predictive ability for a related construction while learning a particular syntactic distinction. This finding has important implications for understanding capability trade-offs during training and may guide more refined data curriculum design.

## Limitations & Future Work

- **Model scale bottleneck**: Validation is limited to models of 9M and 40M parameters. Computing Hessian eigenvectors at every checkpoint for modern LLMs with billions of parameters is prohibitively expensive, making direct scaling infeasible. Possible remedies include approximating the Hessian via random projections or restricting decomposition to LoRA subspaces.
- **Limited basis diversity**: Only Hessian eigenvectors are used as the basis. Other candidate bases include PCA principal components (principal directions of the training trajectory), SAE decoder directions, and task gradient directions. Different bases may reveal skills at different levels of granularity.
- **Limitations of the linearity assumption**: The method assumes that each skill corresponds to a single linear direction in parameter space. Complex compositional skills (e.g., abilities requiring simultaneous mastery of syntax and semantics) may span multiple directions, and their interactions cannot be directly captured by a linear basis.
- **Limited coverage of automatic cluster annotation**: The current POS-tag-template-based automatic annotation covers only clusters with simple syntactic patterns; more abstract semantic clusters require manual review. Using an LLM for automatic annotation is a natural extension.
- **Absence of downstream application validation**: The paper mentions that POLCA can guide data selection and learning rate scheduling, but no experiments are conducted to this end. Demonstrating that the breakthrough timestamps discovered by POLCA can improve training efficiency would substantially enhance its practical value.

## Related Work & Insights

- **vs. LCA (Lan et al., 2020)**: LCA decomposes aggregate loss along individual parameter axes with weak semantic grounding and no support for per-sample analysis. POLCA generalizes LCA along three dimensions (arbitrary basis, per-sample, second-order), improving carrying-skill homogeneity from 0.792 to 0.973.
- **vs. Skill-It (Chen et al., 2024b)**: Skill-It analyzes loss curves for different predefined skills and their inter-skill dependencies. The present work requires no predefined skills, constituting a genuinely unsupervised approach. The skill dependency graph of Skill-It could serve as a validation tool for skills discovered by POLCA.
- **vs. SAEs (Sparse Autoencoders)**: SAEs unsupervisedly discover features in the model's representation space; POLCA unsupervisedly discovers skills in the space of training dynamics. The two are complementary—SAEs answer "what representations has the model learned," while POLCA answers "when did the model learn them."
- **vs. Singular Learning Theory (Watanabe, 2010)**: SLT theoretically predicts phase transitions during model training (by analyzing the structure of singular points). POLCA provides a practical empirical tool for discovering these transitions. The multi-scale loss landscape theory of Ma et al. (2022) provides direct theoretical support for POLCA's decomposition-and-deaggregation strategy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Inverts the paradigm from top-down to bottom-up; the three-fold generalization of POLCA is elegantly designed; the concept of hidden breakthroughs is itself highly thought-provoking.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation on the synthetic task is thorough, but the natural language experiment is limited to 40M scale; larger-model and downstream-application validation are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — The theoretical motivation chain is clear and complete; the method's validity is argued from multiple angles; figures and tables are intuitive.
- Value: ⭐⭐⭐⭐ — Opens a new direction for training-time interpretability, though scale limitations currently confine it to the role of an analysis tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[ICLR 2026\] NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ICLR 2026\] Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)

</div>

<!-- RELATED:END -->
