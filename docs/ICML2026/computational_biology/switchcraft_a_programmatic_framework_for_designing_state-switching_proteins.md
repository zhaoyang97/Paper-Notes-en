---
title: >-
  [Paper Note] SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins
description: >-
  [ICML 2026][Computational Biology][Multi-state protein design] SwitchCraft formalizes the "design of proteins capable of switching between multiple functional states" as a combinatorial constraint-solving optimization pr…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Multi-state protein design"
  - "Boltz-1"
  - "Differentiable structure prediction"
  - "Allosteric regulation"
  - "Biosensors"
date: 2026-05-08
content_hash: 429626fc2818dace
---

# SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins

**Conference**: ICML 2026  
**arXiv**: [2605.31236](https://arxiv.org/abs/2605.31236)  
**Code**: https://github.com/bjing2016/switchcraft  
**Area**: Protein Design / Scientific Computing / Structural Biology  
**Keywords**: Multi-state protein design, Boltz-1, Differentiable structure prediction, Allosteric regulation, Biosensors  

## TL;DR
SwitchCraft formalizes the "design of proteins capable of switching between multiple functional states" as a combinatorial constraint-solving optimization problem. By backpropagating multiple state-dependent losses (motif, binding, conformational change, contact) through the structure prediction model Boltz-1 to optimize amino acid logits via gradient descent, it achieves the first general computational framework for multi-state protein design. In silico experiments demonstrate de novo design of positive/negative allostery, motif switching, induced binding, ligand modification, ligand discrimination, and cpGFP fluorescent biosensors.

## Background & Motivation

**Background**: Generative protein design is currently dominated by two technical routes: 1) Protein Language Models (PLMs, e.g., ProGen, ESM3), trained on billions of natural sequences to generate new sequences based on family labels or GO terms; 2) Structural generative models (RFDiffusion, Boltz-1, BoltzDesign1), which learn structural distributions from the PDB or backpropagate through structure predictors for binder design and enzyme active site scaffolding.

**Limitations of Prior Work**: Natural proteins are far more than "one static structure corresponding to one static function." Many critical functions (motor proteins walking along microtubules, ATP synthase rotation, polymerase information processing, hemoglobin cooperative binding) depend on **multi-state dynamics**—proteins must switch precisely between multiple conformational or binding states. PLM conditions can only reference existing labels and cannot describe unseen complex functions; structural generators are restricted to single static structures. Neither route can directly express specifications like "folding into conformation 1 in the presence of ligand A and conformation 2 in the presence of ligand B."

**Key Challenge**: Either data-driven but lacking labels (PLM route) or physically controllable but only describing a single state (structural route). A dataset that fully expresses "multiple states + structural constraints for each state" simply does not exist, making purely data-driven paths unfeasible.

**Goal**: Construct a **programmatic framework** that allows designers to specify arbitrary numbers of states and their respective structural constraints—similar to writing a program with branches—and let the optimizer automatically find the amino acid sequence that satisfies all states simultaneously.

**Key Insight**: The authors noted that methods like BoltzDesign1 have proven structure predictors like Boltz-1 can be backpropagated for binder design. Since this can be done for a single state, multiple states can be optimized by summing their losses and backpropagating together. Boltz-1 accepts ligand context as input, naturally supporting multiple forward passes where "the same sequence folds into different structures under different ligand environments."

**Core Idea**: The optimization objective is defined such that the sequence $\mathbf{z}\in\mathbb{R}^{20\times L}$, when folded by Boltz-1 under multiple contexts $\{\mathcal{C}_s\}$, simultaneously satisfies a set of losses $\{\mathcal{L}_n\}$. Gradient descent is performed directly on $\mathbf{z}$, using a straight-through estimator to make the discrete amino acid argmax problem differentiable.

## Method

### Overall Architecture
SwitchCraft decomposes multi-state protein design into two stages:

1.  **Design Specification**: Enumerate states $s=1,\ldots,N_{\text{states}}$, where each state is bound to a folding context $\mathcal{C}_s$ (including small molecules, metal ions, DNA, target peptides, etc.); enumerate losses $\mathcal{L}_n:\mathbb{R}^{20\times L}\to\mathbb{R}$ where each loss depends on the Boltz-1 output of one or more states; declare design masks $\mathbf{m}\in\{0,1\}^L$ and optional fixed motif sequences $\mathbf{s}$.
2.  **Design Optimization**: Using sequence logits $\mathbf{z}$ as optimization variables, a 240-step four-stage schedule is executed. At each step, $\mathbf{z}$ is converted into a "hard-soft-continuous" pseudo-representation $\mathbf{z}_{\text{pseudo}}$ and fed into Boltz-1. All losses are aggregated, and gradients are calculated to update $\mathbf{z}$.

The process is conceptually similar to training a deep learning model: losses are design goals, the optimizer is SGD, and the model weights are the sequence itself.

### Key Designs

1.  **Composable State-Loss Language (Loss DSL)**:
    - **Function**: Translates natural language specifications (e.g., "design a scaffold that presents a motif in the presence of ligand X and disrupts it otherwise") into a set of mathematically well-defined losses differentiable with respect to $\mathbf{z}$.
    - **Mechanism**: Four basic loss primitives are derived from Boltz-1 outputs (distogram, pair representation). **Motif Loss** $\mathcal{L}_{\text{motif}}=\sum_{i,j\in m, i\neq j}\sum_k \frac{p_{ijk}}{|m|(|m|-1)}(d_k-\|\mathbf{r}_i-\mathbf{r}_j\|)^2$ minimizes the squared error of intra-motif residue pair distances weighted by distogram probabilities. The corresponding **Anti-motif Loss** $\mathcal{L}_{\text{anti-motif}}=-0.5\,\mathcal{L}_{\text{motif}}$ actively disrupts the scaffold. **Binding Loss** $\mathcal{L}_{\text{binding}}=\frac{1}{2c}\sum \min_j^{(k=c)}\min_i^{(k=2)} H_{<20\text{Å}}(D_{ij})$ uses truncated entropy from BoltzDesign1 to aggregate proximity probabilities of "top-$c$ ligand tokens and top-2 protein residues," encouraging confident contacts; **Anti-binding Loss** is defined as -0.5 times the binding loss. **Conformational Change Loss** $\mathcal{L}_{\text{conf-change}}(\mathbf{z};\mathcal{C}_1,\mathcal{C}_2)=-\frac{1}{L}\sum_i \max_j \mathrm{JSD}(D^{(1)}_{ij}\|D^{(2)}_{ij})$ maximizes the Jensen-Shannon Divergence (JSD) of the distance distributions for the same residue pairs under two states, forcing significant structural divergence. **Contact Loss** $\mathcal{L}_{\text{contact}}=\frac{1}{L}\sum_j \min_{i:|i-j|\geq 9} H_{<14\text{Å}}(D_{ij})$ ensures the structural confidence of each state.
    - **Design Motivation**: All losses are based on continuous distograms rather than hard distances to ensure differentiability. Symmetric positive/negative designs (motif vs. anti-motif) allow explicit expression of requirements. The JSD-based conformational change loss avoids the difficulty of structural alignment by driving state differences at the distribution level.

2.  **Straight-Through + Four-Stage Annealing Sequence Optimization**:
    - **Function**: Converts the discrete 20-dimensional amino acid argmax problem into gradient descent on continuous logits $\mathbf{z}\in\mathbb{R}^{20\times L}$, transitioning smoothly from "soft continuous exploration" to "hard discrete convergence" within 240 steps.
    - **Mechanism**: Each step calculates three representations—soft distribution $\mathbf{z}_{\text{soft}}=\mathrm{softmax}(\mathbf{z}/\tau)$, hard one-hot $\mathbf{z}_{\text{hard}}=\mathrm{onehot}(\mathrm{argmax}\,\mathbf{z})$, and raw logits $\mathbf{z}$. Using STE, $\mathbf{z}_{\text{st}}=(\mathbf{z}_{\text{hard}}-\mathbf{z}_{\text{soft}})|_{\nabla=0}+\mathbf{z}_{\text{soft}}$ appears hard in the forward pass but uses soft gradients. The input to Boltz-1 is a convex combination $\mathbf{z}_{\text{pseudo}}=\beta\mathbf{z}_{\text{hard}}+(1-\beta)(\gamma\mathbf{z}_{\text{soft}}+(1-\gamma)\mathbf{z})$. Three hyperparameters $\beta,\gamma,\tau$ follow a four-stage schedule: Stage 1 (30 steps, $\beta=0,\gamma=1,\tau=0.5$) for soft exploration; Stage 2 (100 steps, $\gamma$ annealed from 0 to 1) to squeeze out hard decisions; Stage 3 (100 steps, $\tau$ reduced from 0.5 to 0.005) to lower temperature; Stage 4 (10 steps, $\beta=1$) for pure one-hot fine-tuning. Residues at motif positions $\mathbf{m}[i]=0$ are fixed to the motif sequence.
    - **Design Motivation**: Discrete search on a 20D simplex explodes exponentially, while pure continuous relaxation yields non-physical "mixed amino acids." STE + multi-stage annealing allows global exploration with continuous gradients followed by forced convergence to valid residues.

3.  **Multi-Motif Merging and cpGFP Biosensor Assembly Workflow**:
    - **Function**: Extends "motif switching" to scaffolding two different motifs with the same sequence and encapsulates multi-state conformational switchers as assembleable fluorescent biosensor components.
    - **Mechanism**: When a state requires scaffolding multiple motifs, Algorithm 2 merges motif constraints at the residue index level before apply standard motif loss. In the biosensor workflow (Sec 4.6), the sensor is split into a "circularly permuted GFP (cpGFP) reporter + ligand-responsive conformational switch." A switcher is designed to have significant conformational differences between apo/holo states. Residue sites with the largest backbone dihedral angle changes are selected as cpGFP insertion points (Algorithm 3 ensures spatial diversity). The cpGFP is embedded, co-folded with Boltz-1, and designs with significant chromophore contact changes are screened.
    - **Design Motivation**: The mechanism of natural cpGFP biosensors (e.g., nicotine sensor, PDB 7s7u/7s7v) relies on a linker glutamate quenching fluorescence in the apo state and being pulled away in the holo state. The authors reverse-engineered this into computational screening criteria (intraRMSD, crossRMSD, radius of gyration, effector iPTM), allowing de novo sensor design for arbitrary small molecules.

### Loss & Training
The global loss is the sum of all state-loss terms. The optimizer is Adam, with stage-specific learning rates $\alpha\in\{0.1,0.2\}$. Initial $\mathbf{z}$ is sampled from Gumbel-softmax. Extensive independent trajectories (100 to 13,858) are run for each task, with the top sequences evaluated by 5 Boltz-1 structure predictions.

## Key Experimental Results

### Main Results
The authors defined 6 multi-state design primitives of increasing complexity plus 1 biosensor workflow. The table summarizes in silico success rates (percentage of designs passing structural confidence, state divergence, and low intra-state RMSD filters).

| Design Task | Ligand Type | Total Designs | Successful Designs | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| +/- Allostery (Motif ON/OFF) | 5 types × 24 motifs × 100 | 12000 | At least 1 success for 11 motifs | Motif RMSD difference often >5 Å, including fold switching |
| Motif Switching (3IXT↔1YCR) | OQO | 100 | 3 Fully Successful | Most candidates satisfied 3 out of 4 constraints |
| Ligand Modification (heme + O₂) | heme + $O_2$ | 558 | 10 | Oxygen displaced Histidine inducing 3.8 Å rearrangement |
| Induced Binding (Top7 frag + Ca²⁺) | Ca²⁺ | 940 | 8 | Ca²⁺ binding induced 12.50 Å rearrangement to form interface |
| Ligand Discrimination (3 states: apo/OQO/Ca²⁺) | OQO + Ca²⁺ | 465 | 12 | Key loop JSD RMSD ≥1.48 Å between any two states |
| Sensor Switcher (SAM/cGMP/ATP) | 3 small molecules | 13858 | 89 stringent → 44 contact change OK | Replicated nicotine sensor mechanism (Glu74 14.7 Å shift) |

### Ablation Study
| Configuration | Observation | Explanation |
| :--- | :--- | :--- |
| Full 4-stage schedule | Convergence to one-hot in 240 steps | Smooth soft-to-hard transition |
| Motif Loss only (single state) | Degenerates to BoltzDesign1 style | Verifies backward compatibility |
| Removing ConfChangeLoss | Multi-state degenerates to single-state copies | JSD term is critical for state separation |
| Removing ContactLoss | Structural confidence of individual states collapses | Must be maintained separately for each state |
| Anti-motif coeff from -0.5 to -1.0 | Disruption too strong, leading to folding failure | Weight balance of opposing constraints is sensitive |

### Key Findings
- Absolute success rates remain low (11/24 motifs, single-digit percent yields), indicating significant room for improvement in this benchmark. The authors propose the "5 ligands × 24 motifs allostery" task as a standard benchmark for multi-state design.
- The ligand modification task showed "physically implausible" failure modes (requiring unfolding/refolding), suggesting a need for future kinetic constraints.
- Three-state ligand discrimination was achieved using a 50-residue miniprotein, where a key loop formed a salt bridge, a hydrophobic pocket, and a Ca²⁺ coordination site across the three states, hinting at the designability of multi-step enzymes.
- The biosensor workflow generated 44 rational candidates for SAM/cGMP/ATP without natural templates, with the SAM design replicating the "linker glutamate displacement" mechanism.

## Highlights & Insights
- Formalizes multi-state protein design as a "constraint satisfaction problem on sequences." The specification (state + loss + mask) acts as a domain DSL for biologists to "program" functions.
- Symmetric positive/negative losses (motif vs. anti-motif) unify "presence" and "absence" requirements within a single framework.
- ConfChangeLoss uses JSD on distograms rather than "structural alignment + RMSD," avoiding alignment ambiguity and enabling direct differentiation.
- Handling discrete search on a 20D simplex using STE + four-stage annealing is a clean case of pushing "inverse design on gradients" from the structural domain to the sequence domain.

## Limitations & Future Work
- Absolute success rates are low; it is currently a benchmark rather than a production-ready tool.
- Evaluations are entirely in silico. Biases in the structure predictor are amplified; high-confidence predictions may not fold in the wet lab. Preliminary induced binding validation is only mentioned in the Appendix.
- Losses are structural and lack kinetic/energy landscape constraints, leading to physically unreachable transitions. Transition state or kinetic terms are needed.
- Computational cost: Boltz-1 forward passes are expensive; 240 steps across multiple states represent a non-trivial hardware and energy commitment.

## Related Work & Insights
- **vs. BoltzDesign1 (Cho et al. 2025)**: Ours directly inherits its loss and optimization framework for single-state binders but extends it to multiple states, serving as a functional superset.
- **vs. RFDiffusion (Watson et al. 2023)**: RFDiffusion generates single-state structures. Ours reuses its scaffolding tasks but upgrades them to "ligand-responsive switchable scaffolds."
- **vs. ProteinMPNN / DynamicMPNN**: ProteinMPNN performs inverse folding for a given backbone. SwitchCraft optimizes sequence and backbone implicitly through Boltz-1, requiring no pre-defined backbone.
- **vs. ProDiT / ProteinGenerator**: These attempt to generate sequences for multiple backbones but **do not accept ligands as input**, failing to capture "ligand-driven state switching."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First general multi-state protein design framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage across 6 task types, but success rates are low and purely in silico.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear abstraction into specification and optimization stages; loss definitions are rigorous.
- Value: ⭐⭐⭐⭐⭐ Establishes a general benchmark and interface for next-gen functional protein design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Conformational Ensembles of Proteins Based on Backbone Geometry](../../NeurIPS2025/computational_biology/learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [\[NeurIPS 2025\] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](../../NeurIPS2025/computational_biology/barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)
- [\[NeurIPS 2025\] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](../../NeurIPS2025/computational_biology/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)
- [\[ICML 2026\] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)
- [\[ICML 2026\] From Feasible to Practical: Pareto-Optimal Synthesis Planning](from_feasible_to_practical_pareto-optimal_synthesis_planning.md)

</div>

<!-- RELATED:END -->
