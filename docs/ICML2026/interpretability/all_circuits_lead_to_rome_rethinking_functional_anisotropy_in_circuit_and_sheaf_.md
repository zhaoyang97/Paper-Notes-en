---
title: >-
  [Paper Note] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
description: >-
  [ICML 2026][Interpretability][circuit discovery] This paper systematically falsifies the implicit assumption in the field of mechanistic interpretability—"one LLM capability corresponds to one unique circuit"—by using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It finds that the same task can be supported by multiple circuits or sheaves that have minimal over
tags:
  - ICML 2026
  - Interpretability
  - circuit discovery
  - sheaf discovery
  - functional anisotropy
  - superposition
  - IOI
date: 2026-05-08
content_hash: 0333127f116f55f1
---
# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.12671](https://arxiv.org/abs/2605.12671)  
**Code**: <https://github.com/TonyXiChen/OASR>  
**Area**: LLM Interpretability / Mechanistic Interpretability / Circuit & Sheaf Discovery  
**Keywords**: circuit discovery, sheaf discovery, functional anisotropy, superposition, IOI

## TL;DR
This paper systematically falsifies the implicit assumption in the field of mechanistic interpretability—"one LLM capability corresponds to one unique circuit"—by using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It finds that the same task can be supported by multiple circuits or sheaves that have minimal overlap (IoU ~4–11%) while remaining faithful, sparse, and complete. The "Distributive Dense Circuit Hypothesis" is proposed as a theoretical explanation.

## Background & Motivation

**Background**: Circuit and sheaf discovery (CSD) is a mainstream route in mechanistic interpretability. Methods ranging from ACDC and EAP to EP and DiscoGP encode LLMs as residual stream computation graphs (DAGs) and use heuristic or gradient-based edge mask optimization to identify sparse subgraphs that maintain accuracy on tasks such as IOI, Docstring, and BLiMP.

**Limitations of Prior Work**: Existing evaluation paradigms default to a "unique minimal circuit." They either compare results against preset ground truths using benchmarks like Tracr or pursue the "fewest edges + performance maintenance," implicitly assuming each task is implemented by a structurally unique sub-mechanism. However, the authors discovered that when identifying two sheaves using current methods, their edges barely overlap despite both completing the task with 100% accuracy.

**Key Challenge**: The authors name this implicit assumption the **Functional Anisotropy Hypothesis**—the idea that "model capabilities are anisotropically localized within a specific sub-mechanism." If multiple structurally diverse sheaves can independently support the same task, the act of "finding a unique circuit" itself loses its mechanistic explanatory significance. Previous phenomena such as backup heads and the hydra effect are merely standby mechanisms revealed after ablation; they cannot explain the coexistence of multiple circuits during normal inference.

**Goal**: (1) Design an algorithm capable of actively discovering multiple circuits that are mutually non-overlapping but faithful; (2) Systematically falsify the anisotropy hypothesis across multiple common benchmarks; (3) Provide a theoretical explanation for why this non-uniqueness naturally exists in LLMs.

**Key Insight**: Incorporate an "already discovered circuit repulsion" regularization term into the differentiable sheaf optimization objective of DiscoGP, explicitly transforming the search for the "next sheaf" into "moving away from the previous ones" within the structural space.

**Core Idea**: Reframe CSD from "finding a unique minimal subgraph" to "enumerating multiple solutions within a functional equivalence class" using Overlap-Aware Sheaf Repulsion (OASR), and then use high-dimensional superposition to argue that these solutions are too numerous to be reduced to a single canonical circuit.

## Method

### Overall Architecture
To falsify the idea that "one task corresponds to one unique circuit," simply re-running DiscoGP with random seeds is insufficient, as optimizers often fall back into the same attractor, resulting in highly overlapping sheaves. The OASR approach explicitly rewrites the search for the "next sheaf" as "moving away from all existing sheaves in the structural space." Following the differentiable framework of DiscoGP, each edge $e$ in the residual stream computation graph is assigned a learnable logit $l_e$. These are transformed into continuous scores $s_e=\sigma((l_e-\log(\log\mathcal{U}_1/\log\mathcal{U}_2))/\tau)$ via Gumbel-Sigmoid relaxation and converted to hard masks using a straight-through estimator. A repulsion term is then added to the original fidelity, sparsity, and completeness objectives. By running this cycle $K$ times—each time requiring the new solution to reuse almost no edges from previous ones—a set of mutually non-overlapping but faithful circuits is actively enumerated.

### Key Designs

**1. Overlap-Aware Sheaf Repulsion: Transforming non-uniqueness from a passive phenomenon into active discovery**

The pain point of existing CSD methods is their pursuit of a "unique optimal solution." Even with random re-initialization, discovered sheaves often overlap significantly, failing to prove that task capabilities can be independently carried by structurally distinct subgraphs. OASR explicitly penalizes the reuse of discovered edges during each round. Let $R$ be the set of edges in discovered sheaves; the total loss for a new round is $\mathcal{L}=\mathcal{L}_{fidelity}+\lambda_s\mathcal{L}_{sparsity}+\lambda_c\mathcal{L}_{complete}+\lambda_o\mathcal{L}_{overlap}(R)$. The repulsion term $\mathcal{L}_{overlap}(R)=\frac{1}{|E|}\sum_{e\in R}\sigma(l_e)$ only imposes an expected activation penalty on edges within $R$, which is equivalent to continuously adding gradients in the logit space in the direction of "moving away from existing solutions." Executing this repeatedly yields $K$ non-overlapping sheaves $\{R_1,\dots,R_K\}$, whose pairwise IoU is much lower than that of random restarts—direct evidence that non-uniqueness is an essential property of LLMs rather than optimization noise.

**2. Complement/Complexity Validation Protocol: Excluding the trivial explanation that "circuits are so large they don't overlap just by random slicing"**

Merely having "consistent task accuracy + low IoU" is insufficient to refute anisotropy, as two sheaves could simply be rotations or reparameterizations of the same canonical circuit. To address this, the authors designed a multi-dimensional validation suite. First, they measure the IOI accuracy of each sheaf to confirm task retention. Then, they measure complement accuracy—removing the discovered edge set $E_A$ entirely to see if the remaining graph can still complete the task—to verify that $E_A$ is indeed necessary rather than redundant. They also report edge density and edge count compared to a "random initialization" baseline. Finally, they perform edge-by-edge ablation on an extremely sparse 3-edge sheaf to check if every single edge is indispensable. Combined with layer-wise distribution analysis, the authors confirm that the differences between sheaves are structural rather than superficial edge rearrangements.

**3. Distributive Dense Circuit Hypothesis: Explaining the inevitability of non-uniqueness via superposition**

A theoretical hypothesis is provided to show that the observed non-uniqueness is a structural consequence of LLM representations. The argument is based on the superposition theory by Elhage et al.: high-dimensional residual streams use near-orthogonal directions to represent multiple sets of features. Any specific "computational implementation" is a routing of these directions to the downstream via some linear combination. Given a task behavior $b$, the set of subgraphs satisfying fidelity expands with model depth and width combinations, causing the number of "sparse yet faithful solutions" to grow exponentially. This results in a formal proposition: under mild assumptions, for a sufficiently large model, there exist $\Omega(\exp(d))$ mutually disjoint faithful sheaves. This perspective unifies phenomena like backup heads and the hydra effect into a "globally distributive dense circuit" framework: redundancy is not an anomaly, but the default state.

### Loss & Training
The weighted sum of four loss terms is used: $\mathcal{L}=\mathcal{L}_{fid}+\lambda_s\mathcal{L}_{sp}+\lambda_c\mathcal{L}_{comp}+\lambda_o\mathcal{L}_{overlap}$. The primary subject of the experiment is GPT-2 small (12L × 12H), with hyperparameters following DiscoGP defaults. To obtain 20 sheaves, OASR fixes the $R$ discovered in previous rounds, re-initializes logits, and then jointly optimizes the fidelity, sparsity, completeness, and overlap objectives.

## Key Experimental Results

### Main Results
Using OASR to find two sheaves on GPT-2 small for the IOI task:

| Sheaf | IoI Acc | Comp. Acc | Edge Density | Edge # |
|-------|---------|-----------|--------------|--------|
| A | 100% | 46.20% | 3.56% | 1158 |
| B | 100% | 45.80% | 3.97% | 1289 |
| Overlap | $\|A\cap B\|=96$ | $\|A\cup B\|=2351$ | **IoU = 4.1%** | — |

Discovery of two sheaves for each of the 9 common CSD benchmark tasks:

| Task | Sheaf A Acc | Sheaf B Acc | IoU(A,B) |
|------|------------|-------------|----------|
| IOI | 100% | 100% | 4.1% |
| BLiMP | 96.8% | 92.6% | 5.1% |
| AGA | 96.0% | 95.3% | 6.2% |
| ANA | 98.0% | 91.3% | 5.3% |
| DNA | 100% | 96.2% | 5.8% |
| DNA-i | 100% | 99.0% | 6.2% |
| DNA-a | 98.5% | 97.0% | 7.5% |
| DNA-ia | 100% | 99.0% | 6.4% |
| Docstring | 98.9% | 100% | 11.0% |

### Ablation Study
"Mutual intersection" analysis extended to 20 sheaves (Mutual IoU = Total intersection of 20 sheaves / Total union):

| Task | Method | \|E_∩\| | \|E_∪\| | Mutual IoU | Avg Acc |
|------|--------|--------|--------|------------|----------|
| IOI | Random Init | 20 | 6560 | 0.30% | 99.95% |
| IOI | **OASR** | 11 | 7382 | **0.15%** | 99.59% |
| BLiMP | Random Init | 50 | 4858 | 1.03% | 97.26% |
| BLiMP | **OASR** | 37 | 5289 | **0.70%** | 96.11% |
| ANA | Random Init | 26 | 4531 | 0.57% | 96.40% |
| ANA | **OASR** | 10 | 4890 | **0.20%** | 95.00% |

### Key Findings
- The "common intersection among multiple sheaves" approaches 0 as the number of discovered sheaves increases: IoU drops from ~4–11% for 2 sheaves to <1% for 20 sheaves. This indicates that task capability is not carried by any "essential core."
- An extremely sparse IOI sheaf with only 3 edges was discovered. Edge-by-edge ablation of these 3 edges—removing any one of them—still allowed OASR to find another high-quality sheaf, falsifying the weakened version of the "essential core edge" hypothesis.
- Compared to "re-running DiscoGP with random initialization," OASR achieved lower Mutual IoU across all tasks with almost no drop in accuracy, showing the repulsion term is not a trivial random perturbation.
- Layer-wise analysis (Fig. 2) shows that the distribution of incoming edges for mid-layer MLPs differs the most between two sheaves, confirming the difference is not just surface-level edge reordering.

## Highlights & Insights
- **Paradigm Shock**: The authors directly challenge a core goal the mechanistic interpretability community has pursued for years—"finding the circuit." If a task has infinite faithful solutions, the rationality of minimality-based evaluation (MIB) and comparisons with ground-truth (Tracr) must be reconsidered. This is a discovery at the level of "the problem was defined incorrectly."
- **Generalizability of OASR**: Incorporating "diverse retrieval" into differentiable objectives via repulsion terms is applicable to any problem seeking functional equivalent solutions, such as sparse feature dictionaries, multi-solution NAS, or multi-modal adversarial discovery.
- **Mutual Evidence between Theory and Empirical Results**: The authors do not merely present experimental phenomena; they provide a mathematical explanation using superposition. They unify backup heads, the hydra effect, and multi-circuit phenomena under the "distributive dense circuit" framework, elevating the contribution.
- **Implications of the 3-edge Sheaf**: The existence of an extremely sparse sheaf where no edge is indispensable suggests that solutions found under sparse optimization for "minimality" might be the least interpretable—any single edge can be replaced in other sheaves.

## Limitations & Future Work
- Experiments were mainly conducted on GPT-2 small (the authors admit this is because all CSD baselines only support it). Extended experiments on larger models (Llama/Pythia) are only in Appendix H and do not fully address whether anisotropy returns as scale increases.
- Tasks are limited to short-context, single-step linguistic diagnostics (IOI/BLiMP/Docstring); the non-uniqueness of circuits for real-world tasks like long reasoning, coding, or math has not been verified.
- OASR requires $K$ training iterations; the computational cost grows linearly with $K$, which may be prohibitive for larger models.
- The theoretical Distributive Dense Circuit Hypothesis provides an existence lower bound but lacks a quantitative construction and relies on "mild assumptions" that need to be tested against real transformers.
- Is "circuit non-uniqueness → loss of mechanistic explanation" too pessimistic? The authors have not yet provided a constructive solution for extracting a consensus mechanism in the presence of multiple solutions.

## Related Work & Insights
- **vs Wang et al. 2022a (Original IOI Circuit)**: Proposed the concept of Backup Name-Mover Heads but interpreted them as "backup mechanisms triggered after ablation." This paper proves that backups are active during normal inference and that redundancy is the default state rather than an anomaly.
- **vs ACDC / EAP / EP / DiscoGP**: This paper utilizes the differentiable optimization framework of DiscoGP but changes the objective from "finding the unique optimal solution" to "enumerating multiple solutions," thereby demonstrating that these methods are all affected by the anisotropy hypothesis.
- **vs McGrath et al. 2023 (Hydra effect)**: Hydra describes substitute activation after ablation. This paper shows that multiple circuits are active simultaneously; thus, hydra is merely one facet of a distributive circuit from an ablation perspective.
- **vs Méloux et al. 2025**: They formally proved circuit non-uniqueness in simple models. This paper generalizes the perspective to pre-trained LMs and real tasks, providing a unified explanation from the perspective of superposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Direct challenge to the implicit assumptions of the interpretability community, presenting a new hypothesis + new algorithm + new evaluation protocol. This is a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ The systematic experimental design of 9 tasks × 20 sheaves is convincing, though model scale and task complexity remain focused on GPT-2 small.
- Writing Quality: ⭐⭐⭐⭐⭐ Conceptual naming (Functional Anisotropy / Distributive Dense Circuit) is clear and accurate; the logical structure is excellent, with theory and empirical evidence mutually reinforcing.
- Value: ⭐⭐⭐⭐⭐ Forces the entire mechanistic interpretability community to re-evaluate the explanatory power of their work; the impact is profound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
