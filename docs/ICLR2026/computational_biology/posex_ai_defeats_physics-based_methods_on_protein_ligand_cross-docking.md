---
title: >-
  [Paper Note] PoseX: AI Defeats Physics-based Methods on Protein Ligand Cross-Docking
description: >-
  [ICLR 2026][Computational Biology][cross-docking] PoseX constructs an open-source docking benchmark covering both self-docking and the more realistic cross-docking scenarios. Utilizing 718 + 1312 new crystal structures free of training leakage, evaluation of 23 docking methods across three major categories, a meticulously designed energy relaxation post-processing pip
tags:
  - ICLR 2026
  - Computational Biology
  - cross-docking
  - benchmark
  - AI co-folding
date: 2026-05-08
content_hash: 08248e3c85ba2fe3
---
# PoseX: AI Defeats Physics-based Methods on Protein Ligand Cross-Docking

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=qqzxKudD4T](https://openreview.net/forum?id=qqzxKudD4T)  
**Code**: https://github.com/CataAI/PoseX (Dataset included https://huggingface.co/datasets/CataAI/PoseX)  
**Area**: Computational Biology / Molecular Docking Benchmark  
**Keywords**: Protein-ligand docking, cross-docking, benchmark, energy relaxation, AI co-folding

## TL;DR
PoseX constructs an open-source docking benchmark covering both self-docking and the more realistic cross-docking scenarios. Utilizing 718 + 1312 new crystal structures free of training leakage, evaluation of 23 docking methods across three major categories, a meticulously designed energy relaxation post-processing pipeline, and a real-time leaderboard, it systematically demonstrates that AI methods have comprehensively outperformed traditional physical docking software in the more challenging real-world task of cross-docking.

## Background & Motivation
**Background**: Protein-ligand docking is a core pipeline in drug discovery, predicting how small molecules bind to target proteins to screen drug candidates. Recently, deep learning methods (AI docking such as DiffDock, AI co-folding such as AlphaFold3) have advanced rapidly, and the community has introduced benchmarks like PoseBuster, PoseBench, and PLINDER to evaluate these methods.

**Limitations of Prior Work**: The authors identify three critical flaws in existing benchmarks. First, almost all benchmarks only evaluate **self-docking** (docking a ligand back into its own native co-crystal conformation), which rarely occurs in real drug R&D—medicinal chemists design new molecules and need to dock them into protein conformations obtained from **other previously published compounds**. Second, frameworks like PLINDER are heavyweight, requiring specific data partitioning and training, making them difficult to use. Third, the set of compared models is too narrow; PoseBuster only compares 5 AI and 2 physical methods, while PLINDER focuses primarily on DiffDock.

**Key Challenge**: There is a systematic misalignment between evaluation settings (self-docking) and real-world applications (where protein conformations change due to induced fit by different ligands). In self-docking, the protein conformation is naturally "adapted" to the ligand, leading to overly optimistic results and failing to expose the true capability of methods in handling protein flexibility.

**Goal**: The goal is decomposed into several specific sub-problems: (1) introduce a more realistic cross-docking evaluation; (2) construct a high-quality dataset free of training leakage; (3) conduct a horizontal comparison of as many methods as possible under a unified standard; (4) provide post-processing to fix structural irrationalities in AI predictions; (5) provide a real-time, reproducible public leaderboard.

**Core Idea**: Re-evaluate the full spectrum of docking methods using **cross-docking, a more difficult and realistic task**, coupled with a lightweight "evaluation-not-training" framework and physical relaxation post-processing, to conclude that AI methods have defeated physical methods in real-world scenarios.

## Method

### Overall Architecture
PoseX is not a docking model but a **docking evaluation benchmark and pipeline**, consisting of four modules: **Task Setting → Dataset Construction → Method Pool Evaluation → Relaxation Post-processing + Leaderboard**.

Regarding tasks, PoseX supports two settings: self-docking (docking ligands back to their native co-crystal conformations to verify pose reproduction) and **cross-docking** (docking small molecules from **other complexes of the same protein** into all conformations of that protein except its native co-crystal). The latter requires methods to provide correct poses even when the protein conformation was not optimized for the current ligand, reflecting real-world drug research.

Regarding data, the authors selected crystal structures published between 2022 and January 2025 from the RCSB PDB (later than the training cutoff for all evaluated AI methods to prevent leakage), resulting in PoseX-SD (718 entries) and PoseX-CD (1312 entries, covering 371 structures across 109 protein targets and 362 small molecules).

Regarding methods, 23 docking methods are categorized into three groups for evaluation: 5 **physical methods** (e.g., Schrödinger Glide, Discovery Studio, GNINA), 11 **AI docking methods** (e.g., DiffDock, SurfDock, UMD V2), and 7 **AI co-folding methods** (e.g., AlphaFold3, Chai-1, Boltz-1x). Each prediction undergoes an OpenMM-based energy relaxation post-processing, with results integrated into a real-time public leaderboard.

### Key Designs

**1. Cross-docking Task Setting: Moving Evaluation to Real Drug R&D Scenarios**

This design directly addresses the "unrealistic nature of self-docking." In self-docking, the protein conformation is co-crystallized with the ligand, so methods only need to reproduce the pose in a pre-optimized pocket. In real drug R&D, however, chemists often have protein conformations obtained from other compounds, and the protein may undergo induced fit when a new ligand enters. Cross-docking simulates this by taking all non-native conformations of the same protein and docking corresponding small molecules into them. This explicitly incorporates the realistic difficulty of "mismatched protein conformations," forcing methods to reveal their true ability to handle protein flexibility. Experiments prove this setting is significantly harder and more discriminative.

**2. Leakage-free High-Quality Dataset Construction: Ensuring Fair Comparison**

Most AI methods are trained on PDBBind v2020 (containing 16,379 complexes). If evaluation sets overlap with training sets, results are contaminated by "memorization." The authors use the release date as a natural split—collecting only new crystal structures from 2022 to January 2025 to ensure they are later than the training cutoffs for all AI methods. The entire data processing pipeline is open-sourced (fixing missing chains, capping termini, adding charges, etc.). The authors also quantified leakage risk: in the classic Astex Diverse Set (85 complexes), 43 fall within the PDBBind training set, highlighting that "high scores" on old benchmarks may involve memory components.

**3. Physical Relaxation Post-processing Module: Fixing AI Inconsistencies with Force Fields**

A common issue with AI methods is that predicted poses often have intramolecular or intermolecular atomic clashes, which are geometrically irrational. The authors implemented an automated energy minimization (with short molecular dynamics when necessary) post-processing based on OpenMM: automatically fixing missing chains, capping N/C termini, adding formal charges to proteins and ligands, and applying constraints to backbone atoms (CA, C, N, O) to avoid structural drift. It supports GAFF / OpenFF small molecule force fields and Gasteiger / MMFF94 partial charge calculations. This significantly mitigates irrational conformations and improves PB-Valid (PoseBuster validity) pass rates, revealing a paradigm: **AI modeling + physical post-processing** yields optimal performance.

**4. Dual-Track Evaluation + Pocket Similarity Generalization Analysis: Explaining "Who is better and why"**

Instead of just providing a total success rate, the authors split the evaluation into **Pocket-Given** (provided binding pocket) and **Blind-Docking** (no pocket provided). They found different patterns: for Pocket-Given, AI docking methods like SurfDock and UMD V2 lead; for Blind-Docking, AI co-folding methods like AlphaFold3 excel because they model protein flexibility simultaneously. Success rate is defined as the proportion of top-1 predictions satisfying $RMSD < 2\text{\AA}$ (or with PB-Valid). Furthermore, the authors quantified generalization capability using **pocket similarity** (maximum TM-score with pre-2022 crystal pockets). Most AI methods showed a moderate negative correlation between pocket similarity and ligand RMSD (e.g., Protenix $r=-0.390$), showing how much "high performance" is generalization versus memorization.

## Key Experimental Results

### Main Results
Docking success rates with relaxation across three benchmarks (Astex / PoseX-SD / PoseX-CD) at $RMSD < 2\text{\AA}$ and PB-Valid (mean of three independent runs):

| Benchmark | Best AI Method | Success Rate | Representative Physical Method | Success Rate |
|------|------|------|------|------|
| Astex | UMD V2 / SurfDock | 94.1% | Glide / Discovery Studio | ~56–67% (outperformed by >25%) |
| PoseX-SD | SurfDock | 78.0% | GNINA | 64.4% |
| PoseX-CD | SurfDock | 77.0% | GNINA | 54.1% |

On PoseX-SD, SurfDock (78.0%) ranks first, followed by UMD V2; AI co-folding methods AlphaFold3 (60.5%) and Protenix (56.3%) perform well; early methods like EquiBind and TankBind are below 20%. On PoseX-CD, SurfDock (77.0%) and UMD V2 (69.2%) lead, with AlphaFold3 (68.6%) following closely.

### Difficulty comparison: Cross-docking truly widens the gap between AI and Physics

| Task | Number of AI methods outperforming the leading physical method (GNINA) |
|------|------|
| PoseX-SD | Only 3 AI docking methods |
| PoseX-CD | 9 (4 AI docking + 5 AI co-folding) |

Physical methods collapse on CD: MOE 33.3%, Glide 38.4%, Discovery Studio 43.7%, significantly worse than self-docking. This is the core evidence for the title "AI defeats physics-based methods on cross-docking."

### Dual-track Analysis (PoseX-CD)

| Track | Winner | Success Rate | Note |
|------|------|------|------|
| Pocket-Given | SurfDock | 77.0% | AI docking leads by utilizing explicit pocket info; all physical methods outperformed |
| Blind-Docking | AlphaFold3 | 68.8% | AI co-folding wins by modeling protein flexibility without pocket dependency |

### Key Findings
- **AI Comprehensive Leadership**: Latest AI docking and AI co-folding consistently outperform physical methods in both self/cross-docking, with the largest gap in cross-docking.
- **Critical Contribution of Relaxation**: Force field energy minimization significantly mitigates atomic clashes in AI methods and is a necessary step for high scores in real applications, pointing to an "AI modeling + physical post-processing" optimal combination.
- **Chirality Issues**: Most AI co-folding methods (AlphaFold3, Chai-1) exhibit ligand chirality errors, except for Boltz-1x, which uses physics-inspired potentials during inference to fix hallucinations, significantly improving stereochemical structural rationality.
- **Pocket Information Importance**: DiffDock-Pocket consistently outperforms pocket-less DiffDock on both SD/CD, suggesting that explicit pocket modeling is valuable, especially for AI co-folding.

## Highlights & Insights
- **Countering Data Leakage with "Temporal Splitting"**: Collecting only new crystal structures published after the AI training cutoffs ensures a fair horizontal comparison. The authors also found 43/85 complexes in Astex fall within the PDBBind training set, exposing the memorization risk of old benchmarks.
- **Cross-docking as the True Discriminator**: The precipitous drop of physical methods from self to cross-docking proves more than any single SOTA number that "AI has won in realistic tasks."
- **Transferable "AI Positioning + Physical Refinement" Paradigm**: Decoupling global pose prediction by neural networks from local geometric optimization by force fields is a transferable insight for any structure generation task.
- **Dual-Track + Pocket Similarity Analysis**: Instead of just ranking, the analysis decouples confounding variables like "pocket availability" and "pocket familiarity," allowing readers to see the true capability boundaries of each method.

## Limitations & Future Work
- As a benchmark, results are limited to the 23 selected methods and the temporal window (2022–2025.1). The leaderboard requires continuous updates for new methods and leakage boundaries.
- Success rate is primarily measured by $RMSD < 2\text{\AA}$ + PB-Valid, focusing on geometric pose and structural rationality rather than downstream metrics like binding affinity prediction.
- Cross-docking still relies on the definition of "non-native conformations of the same protein," providing limited coverage for entirely new targets with no known conformations (true apo / large conformational changes).
- Chirality and clash issues reveal systematic weaknesses in AI co-folding. The authors suggest incorporating physics-inspired constraints (like Boltz-1x) and co-folding flexibility modeling as clear future directions.

## Related Work & Insights
- **vs PoseBuster / PoseBench**: These only perform self-docking evaluation, use crude or no relaxation, and include fewer methods (7 each). PoseX adds cross-docking, refined relaxation, expands to 23 methods, and open-sources the data pipeline.
- **vs PLINDER**: PLINDER is heavyweight and only compares DiffDock. PoseX takes a lightweight "only evaluation" approach with lower barriers to entry, wider coverage, and a real-time leaderboard.
- **Insight**: The realism of evaluation settings (cross- vs self-) often drives domain understanding more than chasing SOTA on a single point. Complementing neural networks with physical priors (AI for modeling, force fields for geometry) is likely a robust paradigm for structure prediction.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic large-scale horizontal evaluation of cross-docking + temporal splitting for leakage prevention.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 methods × 3 benchmarks × dual-track × relaxation control × triple independent runs.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of Problem-Solution-Evidence; well-distilled conclusions.
- Value: ⭐⭐⭐⭐⭐ Open-source data + real-time leaderboard + clear paradigm insights; high practical value for the drug discovery community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICLR 2026\] SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion](sigmadock_untwisting_molecular_docking_with_fragment-based_se3_diffusion.md)
- [\[ICLR 2026\] Pallatom-Ligand: an All-Atom Diffusion Model for Designing Ligand-Binding Proteins](pallatom-ligand_an_all-atom_diffusion_model_for_designing_ligand-binding_protein.md)
- [\[ICLR 2026\] SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset](sair_enabling_deep_learning_for_protein-ligand_interactions_with_a_synthetic_str.md)
- [\[ICML 2026\] Protein Circuit Tracing via Cross-layer Transcoders](../../ICML2026/computational_biology/protein_circuit_tracing_via_cross-layer_transcoders.md)

</div>

<!-- RELATED:END -->
