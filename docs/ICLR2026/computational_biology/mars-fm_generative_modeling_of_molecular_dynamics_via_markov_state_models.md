---
title: >-
  [Paper Note] MarS-FM: Generative Modeling of Molecular Dynamics via Markov State Models
description: >-
  [ICLR 2026][Computational Biology][Molecular Dynamics] Instead of learning frame-by-frame MD transition densities with a fixed lag time, this work first uses Markov State Models (MSM) to coarse-grain trajectories into discrete metastable states, then employs Flow Matching to learn "state-to-state" jump distributions. This approach replaces molecular dynamics sampling with a two-order-of-magnitude speedup and enhanced capability for exploring rare large conformational changes.
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "Molecular Dynamics"
  - "Markov State Model"
  - "Flow Matching"
  - "Protein Conformation"
  - "Generative Surrogate"
date: 2026-05-08
content_hash: 05ef5696994d2db0
---

# MarS-FM: Generative Modeling of Molecular Dynamics via Markov State Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jP3HnYXoIp](https://openreview.net/forum?id=jP3HnYXoIp)  
**Code**: [https://github.com/valence-labs/mars-fm](https://github.com/valence-labs/mars-fm)  
**Area**: Computational Biology / Molecular Dynamics Generative Modeling  
**Keywords**: Molecular Dynamics, Markov State Model, Flow Matching, Protein Conformation, Generative Surrogate  

## TL;DR
Instead of learning frame-by-frame MD transition densities with a fixed lag time, this work first uses Markov State Models (MSM) to coarse-grain trajectories into discrete metastable states, then employs Flow Matching to learn "state-to-state" jump distributions. This approach replaces molecular dynamics sampling with a two-order-of-magnitude speedup and enhanced capability for exploring rare large conformational changes.

## Background & Motivation
**Background**: Molecular Dynamics (MD) serves as a "computational microscope" for studying protein conformational ensembles and revealing functional mechanisms. Long trajectories provide sampling of the Boltzmann distribution. However, biological events often occur on millisecond scales, while Langevin integration steps are at the femtosecond level, making long simulations extremely expensive. Consequently, several generative surrogate models have recently emerged to replace MD.

**Limitations of Prior Work**: Prevailing **MD Emulators (MD-Emus)** learn the transition density $p_\tau(y|x)$ under a fixed lag time $\tau$—generating the future frame $x(t+\tau)$ given the current frame $x(t)$—and assemble trajectories autoregressively during inference (e.g., MDGen generates $K$ frames at once). This paradigm has structural flaws: if the lag is too short, acceleration is limited; if too long, important metastable states are skipped. More fundamentally, MD trajectories suffer from **data imbalance**—the vast majority of samples are high-frequency but uninformative transitions "staying within the same energy minimum," while the high-energy barrier crossings that drive exploration (e.g., folding/unfolding) are extremely rare. Training batches are overwhelmed by uninformative intra-state transitions, making it difficult for models to learn rare large conformational changes.

**Key Challenge**: Frame-by-frame modeling with a fixed lag **forcibly binds** the generative target to MD temporal dynamics. This limits the model by data imbalance and leads to accumulated errors and low exploration efficiency due to autoregression.

**Goal**: To **decouple** generative modeling from temporal dynamics, focusing instead on learning macroscopic transitions between metastable states to significantly improve the exploration efficiency and generalization of rare conformations while maintaining long-term statistical correctness.

**Core Idea** (**New MSM Emulators Paradigm**): MSM clusters frames into discrete states and describes inter-state dynamics via a Markov chain matrix $T$, naturally discarding high-frequency noise and guaranteeing long-term statistics. The authors propose a new class of generative models, **MSM-Emus**: instead of learning frame-by-frame transitions, they learn the state-to-state transition distribution $p_T(\cdot|x(t))$ induced by the MSM, with **MarS-FM (Markov Space Flow Matching)** as a representative instance.

## Method

### Overall Architecture
MarS-FM reformulates "surrogate MD sampling" into a three-step pipeline: **offline MSM construction for each training protein → learning MSM-induced state-to-state transition distributions via Flow Matching → parallel energy landscape exploration using tree-based or hybrid sampling during inference**. The key shift is changing the training target from "frame-by-frame transition density $p_\tau$" to "state-mixture distribution $p_T$": given state $S_i$ of the starting frame, a target state $S_j$ is sampled according to $T_{ij}$, and then a target frame is uniformly sampled from the set of MD frames in $S_j$. Any "pair of frames falling into adjacent MSM states" can serve as a training sample, so the quantity and diversity of training transitions are no longer constrained by frame saving intervals. Note that MSM is only used for data preprocessing during training; no MSM information is required during inference.

```mermaid
flowchart TD
    A[MD Trajectories for Each Training Protein] --> B[Dimensionality Reduction: TICA or Rg/Secondary Structure]
    B --> C[k-means Microstates + PCCA+ Clustering into 10 Metastable States]
    C --> D[Estimate Transition Matrix T at Lag Time τ]
    D --> E[Training Pairs: Frame x∈S_i and Target Frame x1∈S_j, j~T_ij]
    E --> F[Flow Matching learns Vector Field vθ: Noise→p_T·|x_t]
    F --> G[Inference: Tree Sampling / MarS-FM⇒MDGen Hybrid]
    G --> H[Parallel Generation of Surrogate Ensembles, Calculate RMSD/Rg/Secondary Structure]
```

### Key Designs

**1. MSM Construction: Compressing high-frequency temporal signals into discrete metastable states.** A separate MSM is constructed offline for each training domain using a fixed set of hyperparameters. States are first defined in a low-dimensional collective variable space: either using **TICA** (Time-lagged Independent Component Analysis, finding directions that maximize the autocorrelation of $w_j^\top x(t)$ and keeping the minimum coordinates covering >95% kinetic variance) or directly clustering physical observables like radius of gyration and secondary structure content. These are clustered into 100 microstates via k-means and then merged into 10 metastable states via **PCCA+** spectral clustering. Finally, the transition matrix is estimated at a specified lag time (100 ps for tetrapeptides, 50 ns for MD-CATH proteins) as $T_{ij} = C_{ij}/\sum_k C_{ik}$, where $C_{ij}=|\{x(t)\in S_i: x(t+\tau)\in S_j\}|$. The induced transition density satisfies $\int_{S_j} p_T(y|x(t))\,dy = T_{ij}$. In its simplest form, the intra-state density is uniform, meaning $p_T$ depends on state identity rather than specific conformation. Since the model learns "inter-state interpolation" rather than "replicating observed paths," a larger $\tau$ can be chosen without sacrificing the amount of training data.

**2. State-to-state Mixture Distribution as Generative Target: Circumventing data imbalance at the root.** For frame $x(t)\in S_i$, the MSM defines a categorical distribution $j\mapsto T_{ij}$ over successor states. The authors interpret $p_T(\cdot|x(t))$ as a **mixture distribution**—first sampling target state $S_j$ via $T_{ij}$, then sampling a conformation from the empirical MD frame ensemble belonging to $S_j$. During training, **uniform sampling is first performed across all states $S_i$**, and then conditioned on a specific frame $x(t)$ within that state. This ensures rare states are encountered with equal probability, whereas standard MD-Emus sampling frames uniformly from trajectories inevitably bias toward high-frequency intra-state transitions. Because each metastable state aggregates frames from different replicas and multiple visits to the same basin, this construction allows the model to see transitions across replicas and multiple energy barrier crossings—even if such events are scarce in a single trajectory.

**3. Flow Matching Training + SE(3) Representation.** Each residue is represented in SE(3) as $T_\alpha(t)=(q_\alpha(t), r_\alpha(t), (\cos\chi_k,\sin\chi_k)_{k=1}^7)$ (quaternion rotation, translation, and 7 torsion angles). Target conformations are represented as roto-translation offsets relative to the input. Output network $v_\theta$ uses the DiT blocks from MDGen, conditioned on the sequence and current conformation via IPA layers. Training uses standard Flow Matching: source distribution $p_0=\mathcal{N}(0,1)$, target $p_1=p_T(\cdot|x(t))$. Noise $x_0$ and target frame $x_1$ are interpolated into $x_s$ to minimize the mismatch between the vector field and the conditional path velocity:
$$\mathcal{L}_{\text{MarS-FM}}(\theta)=\mathbb{E}\,\lVert v_\theta(s,x_s;x(t))-\dot{x}_s\rVert^2$$
Unlike MDGen, which always takes a future frame $x(t+K\tau)$ from the same trajectory, MarS-FM samples $x_1$ via the MSM transition kernel $T$, applying the same FM objective to more diverse, state-conditioned training pairs.

**4. Inference: Tree Sampling and Hybrid Strategies.** MSM information is not used during inference. Given a sequence $a$ and input frame $x(0)$ of an unseen protein, two strategies are used: (i) **Tree Sampling**—parallel generation of $n$ frames $\{y_i\}\sim p_T(\cdot|x(0))$, then parallel generation of $p_T(\cdot|y_i)$ for each $y_i$, deepening the tree according to the sampling budget; (ii) **MarS-FM⇒MDGen Hybrid**—using MarS-FM to sample dispersed conformations, then using MDGen to generate short trajectories from each point. This mirrors the MSM concept of "starting short simulations from different states," restoring local dynamics in workflows requiring temporal fidelity. Both allow the vast majority of conformations to be generated in parallel, reducing autoregressive calls, suppressing cumulative error, and exploring the target distribution more efficiently by decoupling from temporal dynamics.

## Key Experimental Results

### Main Results (MD-CATH, 450 K high-temp replica, strict 20% sequence similarity filter, 495 test domains; 100/1000 conformations)

| Method | Pairwise RMSD r ↑ | Per-target RMSF r ↑ | Rg KL ↓ | Sec. Struct. JSD ↓ | MSM JSD ↓ | ΔG_fold MAE ↓ |
|------|------|------|------|------|------|------|
| MD (Oracle) | 0.65 / 0.89 | 0.77 / 0.92 | 2.19 / 0.32 | 0.22 / 0.05 | 0.49 / 0.12 | 2.40 / 0.80 |
| MDGen-100 | 0.34 / 0.28 | 0.60 / 0.46 | 3.66 / 0.78 | 0.29 / 0.26 | 0.51 / 0.27 | 2.58 / 1.52 |
| MDGen-20 | 0.57 / 0.28 | 0.61 / 0.20 | 2.48 / 1.37 | 0.19 / 0.39 | 0.48 / 0.51 | 1.52 / 2.01 |
| BioEmu | 0.23 / 0.26 | 0.64 / 0.67 | 4.75 / 3.55 | 0.44 / 0.41 | 0.55 / 0.41 | 4.82 / 4.62 |
| **MarS-FM⇒MDGen-20** | 0.59 / 0.64 | 0.79 / 0.84 | 1.99 / 0.77 | 0.18 / 0.11 | 0.43 / 0.23 | 1.38 / 1.02 |
| **MarS-FM** | **0.60 / 0.65** | **0.84 / 0.90** | **1.74 / 0.42** | 0.18 / 0.14 | **0.42 / 0.17** | **1.25 / 1.20** |

MarS-FM significantly outperforms MDGen and BioEmu across nearly all metrics. Some metrics (e.g., per-target RMSF r at 1000 conformations reaches 0.90) approach the MD oracle.

### Ablation Study (Tetrapeptide, 10^4 conformations, JSD ↓)

| Method | Torsions(all) | TICA-0 | MSM states | Macrostate MAE ↓ |
|------|------|------|------|------|
| MD (Oracle) | 0.08 | 0.20 | 0.21 | — |
| MDGen-1000 | 0.11 | 0.23 | 0.23 | 1.13 |
| MDGen-200 | 0.12 | 0.24 | 0.27 | 1.12 |
| MarS-FM⇒MDGen-200 | 0.10 | 0.21 | 0.23 | 0.83 |
| **MarS-FM** | **0.10** | **0.21** | **0.22** | **0.63** |

For tetrapeptides with low chemical diversity and no large conformational movements, MD-Emus and MSM-Emus should theoretically perform similarly; nonetheless, MarS-FM nearly halves the **Macrostate MAE** (1.12→0.63), proving it better captures rare metastable states in TICA space.

### Key Findings
- **Large Conformational Exploration**: Figure 5 shows that secondary structure content of the first 4 MarS-FM samples differs significantly, whereas MDGen samples all fall into the same energy minimum. In TICA plots (Figure 4), MarS-FM explores modes completely ignored by MDGen.
- **Thorough Ablation Control**: The authors specifically tested MDGen-20/100 and a "parallel" MDGen variant (conditioning only on the input frame without autoregression) to prove that MarS-FM's advantages cannot be replicated by simply changing lag times or reducing autoregressive calls.
- **Acceleration**: Achieves over two orders of magnitude speedup compared to implicit/explicit solvent MD sampling.

## Highlights & Insights
- **Paradigm-level Contribution**: Introduces MSM-Emus as a new generative model category, switching "learning temporal dynamics" to "learning state-to-state transitions," addressing MD data imbalance at its source—a deeper insight than mere architectural changes.
- **Reuse of Established Tools**: MSM, TICA, and PCCA+ are standard tools in the MD community. MarS-FM treats them as offline preprocessing for training data, entailing almost zero additional inference overhead, resulting in an elegant engineering solution.
- **Rigorous Generalization Evaluation**: Uses MMseqs2 with maximum sensitivity to enforce ≤20% sequence similarity between test and training sets (stricter than BioEmu's 40%). It also specifically examines large conformational changes like unfolding at 450 K high temperatures, ensuring a robust evaluation protocol.
- **Complementarity of Hybrid Schemes**: Explicitly acknowledges that MSM-Emus are not designed for fine-grained intra-state chronology, offering the MarS-FM⇒MDGen hybrid to combine the strengths of both models, reflecting a clear understanding of the method's boundaries.

## Limitations & Future Work
- **Lack of Fine-grained Kinetic Fidelity**: By design, state-to-state jumps prioritize long-term thermodynamics/dynamics at the expense of intra-state frame-by-frame chronology. Pure MarS-FM is unsuitable for workflows requiring precise local dynamics, necessitating compensation via hybrid schemes.
- **Dependency on MSM Quality**: Performance upper bounds are influenced by offline MSM construction (number of states, lag time, dimensionality reduction features). MSMs are built separately for each domain, and hyperparameters may require dataset-specific tuning; cross-dataset transferability is not yet fully discussed.
- **Coarse Intra-state Uniformity Assumption**: The simplest form assumes uniform intra-state density, which may lose precision for metastable states with complex internal structures.
- **Evaluation Focused on Structural Observables**: While it matches RMSD/Rg/secondary structure distributions, verification of finer kinetic quantities like rate constants and pathway mechanisms remains limited.

## Related Work & Insights
- **MD-Emus Lineage**: Timewarp, Two-for-One, and MDGen learn transition densities with fixed lags. This paper categorizes them as being limited by data imbalance, using MDGen as the primary baseline.
- **Boltzmann Generators**: Noé et al. use Normalizing Flows to sample Boltzmann distributions directly. This work shifts toward an intermediate granularity: "sampling MSM-induced distributions."
- **BioEmu**: A large-scale model for directly predicting conformational ensembles, used as an additional baseline for large systems.
- **Insight**: The idea of "first coarse-graining to find invariant/slow-varying structures, then generating in the coarse-grained space" can be extended to other temporal/dynamical modeling problems (e.g., video, skill discovery in reinforcement learning) to bypass the common issue of high-frequency, uninformative samples dominating training signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Proposes the entirely new MSM-Emus paradigm, switching the generative target from frame-by-frame densities to MSM-induced state transitions; this is a paradigm-level rather than incremental innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Uses large-scale MD-CATH data + strict 20% sequence filtering + high-temp large conformation evaluation + comprehensive ablations (lags, parallel variants, hybrids). Only lacks more verification of fine-grained kinetic quantities.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical chain from motivation to pain points to paradigm shift. Figures 1, 4, and 5 provide intuitive comparisons, and equations/sampling processes are well-detailed.
- **Value**: ⭐⭐⭐⭐⭐ Two orders of magnitude speedup + significantly outperforms existing methods + Open Source; offers direct practical value for protein conformational ensemble sampling and drug discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BioMD: All-atom Generative Model for Biomolecular Dynamics Simulation](biomd_all-atom_generative_model_for_biomolecular_dynamics_simulation.md)
- [\[ICLR 2026\] WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport](wfr-fm_simulation-free_dynamic_unbalanced_optimal_transport.md)
- [\[ICLR 2026\] Align Your Structures: Generating Trajectories with Structure Pretraining for Molecular Dynamics](align_your_structures_generating_trajectories_with_structure_pretraining_for_mol.md)
- [\[ICLR 2026\] Multi-state Protein Sequence Design with DynamicMPNN](multi-state_protein_sequence_design_with_dynamicmpnn.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/computational_biology/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)

</div>

<!-- RELATED:END -->
