---
title: >-
  [Paper Note] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
description: >-
  [ICML 2026][Interpretability][circuit discovery] This paper systematically disproves the implicit assumption in mechanistic interpretability—"one LLM capability corresponds to one unique circuit"—using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It reveals that the same task can be supported by multiple, nearly non-overlapping sheaves (IoU ~4–11%) that satisf
tags:
  - ICML 2026
  - Interpretability
  - circuit discovery
  - sheaf discovery
  - functional anisotropy
  - superposition
  - IOI
date: 2026-05-08
content_hash: 98166c2f85ff6f64
---
# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.12671](https://arxiv.org/abs/2605.12671)  
**Code**: <https://github.com/TonyXiChen/OASR>  
**Area**: LLM Interpretability / Mechanistic Interpretability / Circuit & Sheaf Discovery  
**Keywords**: circuit discovery, sheaf discovery, functional anisotropy, superposition, IOI

## TL;DR
This paper systematically disproves the implicit assumption in mechanistic interpretability—"one LLM capability corresponds to one unique circuit"—using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It reveals that the same task can be supported by multiple, nearly non-overlapping sheaves (IoU ~4–11%) that satisfy requirements for being faithful, sparse, and complete. The authors propose the "Distributive Dense Circuit Hypothesis" as a theoretical explanation.

## Background & Motivation

**Background**: Circuit and Sheaf Discovery (CSD) is a mainstream route in mechanistic interpretability. Methods ranging from ACDC and EAP to EP and DiscoGP encode LLMs into residual stream computational Directed Acyclic Graphs (DAGs). They then use heuristics or gradient-based edge mask optimization to find sparse subgraphs that maintain accuracy on tasks such as IOI, Docstring, and BLiMP.

**Limitations of Prior Work**: Existing evaluation paradigms assume a "unique minimal circuit." They either compare results against preset ground truths (e.g., Tracr) or pursue "minimum edges + performance maintenance," implicitly assuming each task is implemented by a structurally unique sub-mechanism. However, the authors find that when searching for two sheaves using current methods, they can complete the task with 100% accuracy despite having almost no edge overlap.

**Key Challenge**: The authors name this implicit assumption the **Functional Anisotropy Hypothesis**—the idea that "model capabilities are anisotropically localized in specific sub-mechanisms." If multiple structurally diverse sheaves can independently support the same task, the act of "finding the unique circuit" loses its mechanistic significance. Phenomena like backup heads or the hydra effect appear as "standby mechanisms visible only after ablation," failing to explain the coexistence of multiple circuits during normal inference.

**Goal**: (1) Design an algorithm capable of actively discovering multiple "non-overlapping but faithful" circuits; (2) Systematically disprove the anisotropy hypothesis across multiple common benchmarks; (3) Provide a theoretical explanation for why this non-uniqueness naturally exists in LLMs.

**Key Insight**: Incorporate a "repulsion term for discovered circuits" into the differentiable sheaf optimization objective of DiscoGP. This explicitly transforms the task of "finding the next sheaf" into "moving away from previous ones in the structural space."

**Core Idea**: Use Overlap-Aware Sheaf Repulsion (OASR) to reframe CSD from "finding a unique minimal subgraph" to "enumerating multiple solutions within a functional equivalence class." High-dimensional superposition is then used to argue that these solutions are too numerous to be reduced to a single canonical circuit.

## Method

### Overall Architecture
To disprove the idea that "one task corresponds to a unique circuit," merely re-running DiscoGP with different seeds is insufficient, as the optimizer often falls back into the same attractor, resulting in highly overlapping sheaves. OASR explicitly reformulates the search for a new sheaf as "staying away from all previously discovered sheaves in the structural space." Following the differentiable framework of DiscoGP, each edge $e$ in the residual flow DAG is assigned a learnable logit $l_e$. After Gumbel-Sigmoid relaxation, continuous scores $s_e=\sigma((l_e-\log(\log\mathcal{U}_1/\log\mathcal{U}_2))/\tau)$ are obtained and converted into hard masks using a straight-through estimator. A repulsion term is then added to the original fidelity, sparsity, and completeness objectives. By running this loop $K$ times and requiring each new solution to avoid reusing edges from previous iterations, the algorithm actively enumerates a set of non-overlapping but faithful circuits.

### Key Designs

**1. Overlap-Aware Sheaf Repulsion: Turning non-uniqueness from a passive phenomenon into active discovery**

Current CSD methods struggle because they aim for a "unique optimal solution." Even with random re-initialization, the resulting sheaves often show high overlap, failing to prove that task capabilities can be carried by structurally distinct subgraphs. OASR explicitly penalizes the reuse of discovered edges during each round. Let $R$ be the set of edges in discovered sheaves; the total loss for a new round is $\mathcal{L}=\mathcal{L}_{fidelity}+\lambda_s\mathcal{L}_{sparsity}+\lambda_c\mathcal{L}_{complete}+\lambda_o\mathcal{L}_{overlap}(R)$. The repulsion term $\mathcal{L}_{overlap}(R)=\frac{1}{|E|}\sum_{e\in R}\sigma(l_e)$ applies an activation penalty only to edges within $R$, effectively pushing the logits away from existing solutions. Repeating this process yields $K$ non-overlapping sheaves $\{R_1,\dots,R_K\}$ with pairwise IoUs far lower than those from random re-runs, providing direct evidence that non-uniqueness is an inherent property of LLMs rather than optimization noise.

**2. Complementary/Complex Verification Protocol: Ruling out the trivial explanation of "large circuits being sliced"**

Matching accuracy and low IoU are not enough to refute anisotropy, as two sheaves could simply be rotations or re-parameterizations of the same canonical circuit. The authors designed a multi-dimensional verification protocol: first, measure IOI accuracy to confirm task retention; next, measure "complement accuracy"—removing the discovered edge set $E_A$ to see if the remaining graph can still complete the task—to verify that $E_A$ is necessary rather than redundant. Edge density and count are reported and compared against "random re-initialization" baselines. Finally, for an extremely sparse 3-edge sheaf, edge-by-edge ablation is performed to check if every edge is indispensable. Combined with layer-wise distribution analysis, the authors confirm that the differences between sheaves are structural rather than superficial edge rearrangements.

**3. Distributive Dense Circuit Hypothesis: Explaining non-uniqueness via superposition**

To ensure the observed non-uniqueness is not just an artifact of DiscoGP optimization, the authors propose a theoretical hypothesis. Based on the superposition theory by Elhage et al., high-dimensional residual streams use nearly orthogonal directions to represent multiple feature sets. Any "computational implementation" is a linear combination of these directions routed downstream. For a given task behavior $b$, the set of subgraphs satisfying fidelity expands with model depth and width, causing the number of "sparse yet faithful" solutions to grow exponentially. This leads to a formal proposition: under mild assumptions, for a sufficiently large model, there exist $\Omega(\exp(d))$ disjoint faithful sheaves. This perspective unifies backup heads and the hydra effect into a "globally distributive dense circuit" framework where redundancy is the default state, not an anomaly.

### Loss & Training
The four loss components are weighted as $\mathcal{L}=\mathcal{L}_{fid}+\lambda_s\mathcal{L}_{sp}+\lambda_c\mathcal{L}_{comp}+\lambda_o\mathcal{L}_{overlap}$. Experiments were primarily conducted on GPT-2 small (12L × 12H) using default DiscoGP hyperparameters. To obtain 20 sheaves, OASR keeps the $R$ from previous rounds fixed, re-initializes logits, and jointly optimizes the objectives.

## Key Experimental Results

### Main Results
Discovered two sheaves for the IOI task on GPT-2 small using OASR:

| Sheaf | IoI Acc | Comp. Acc | Edge Density | Edge # |
|-------|---------|-----------|--------------|--------|
| A | 100% | 46.20% | 3.56% | 1158 |
| B | 100% | 45.80% | 3.97% | 1289 |
| Overlap | $\|A\cap B\|=96$ | $\|A\cup B\|=2351$ | **IoU = 4.1%** | — |

Discovered two sheaves across 9 common CSD benchmark tasks:

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
Analysis of "Mutual IoU" extended to 20 sheaves (Mutual IoU = total intersection of 20 sheaves / total union):

| Task | Method | \|E_∩\| | \|E_∪\| | Mutual IoU | Avg Acc |
|------|--------|--------|--------|------------|----------|
| IOI | Random Init | 20 | 6560 | 0.30% | 99.95% |
| IOI | **OASR** | 11 | 7382 | **0.15%** | 99.59% |
| BLiMP | Random Init | 50 | 4858 | 1.03% | 97.26% |
| BLiMP | **OASR** | 37 | 5289 | **0.70%** | 96.11% |
| ANA | Random Init | 26 | 4531 | 0.57% | 96.40% |
| ANA | **OASR** | 10 | 4890 | **0.20%** | 95.00% |

### Key Findings
- The "common intersection of multiple sheaves" approaches zero as the number of discovered sheaves increases: IoU ≈ 4–11% for 2 sheaves down to Mutual IoU < 1% for 20 sheaves. This indicates that task capabilities are not carried by any "essential core."
- An extremely sparse IOI sheaf with only 3 edges was discovered. Ablating any one of these 3 edges still allowed OASR to find another high-quality sheaf, disproving the weakened hypothesis of "essential core edges."
- OASR achieved lower Mutual IoU than "random re-initialization" across all tasks with almost no drop in accuracy, proving the repulsion term is not trivial noise.
- Layer-wise analysis (Fig. 2) shows the largest differences between sheaves occur in the input edge distributions of mid-layer MLPs, rather than superficial re-parameterizations.

## Highlights & Insights
- **Paradigm Shift**: The authors directly challenge the core goal of the mechanistic interpretability community—"finding the circuit." If a task has infinite faithful solutions, the validity of minimality-based (MIB) evaluations and ground-truth comparisons (Tracr) must be reconsidered.
- **Generalizability of OASR**: Incorporating "diverse retrieval" via repulsion terms into a differentiable objective is a technique applicable to any problem seeking to enumerate functionally equivalent solutions, such as sparse feature dictionaries or multi-solution NAS.
- **Theoretical and Empirical Synthesis**: Instead of just reporting phenomena, the authors provide a mathematical explanation using superposition. They unify backup heads, the hydra effect, and multi-circuit phenomena under the "distributive dense circuit" framework.
- **Significance of the 3-edge Sheaf**: The existence of extremely sparse sheaves where no single edge is indispensable suggests that the "minimal" solution found via sparse optimization may be the least interpretable, as any of its components can be replaced by other sheaves.

## Limitations & Future Work
- Experiments were mostly limited to GPT-2 small (due to CSD baseline compatibility). Scaling experiments for larger models (Llama/Pythia) are in Appendix H and do not fully address whether anisotropy reverts after scaling.
- Tasks are limited to short-context, single-step linguistic diagnostics (IOI/BLiMP/Docstring); non-uniqueness has not been verified for long reasoning, code, or math tasks.
- OASR requires $K$ training cycles, making computational cost linear with $K$. Setting $K=20$ is expensive and may be infeasible for larger models.
- The Distributive Dense Circuit Hypothesis provides an existence lower bound rather than quantitative construction and relies on "mild assumptions" that require verification in real transformers.
- The conclusion that "non-uniqueness leads to failure of mechanistic explanation" might be too pessimistic; the authors do not provide a constructive way to extract a "consensus mechanism" from multiple solutions.

## Related Work & Insights
- **vs Wang et al. 2022a (Original IOI Circuit)**: They proposed Backup Name-Mover Heads as an "ablation-triggered backup mechanism." This paper proves backups are active during normal inference; redundancy is the default, not an anomaly.
- **vs ACDC / EAP / EP / DiscoGP**: This work utilizes the DiscoGP framework but changes the goal from "finding the unique optimal" to "enumerating multiple solutions," revealing that all these methods are affected by the anisotropy hypothesis.
- **vs McGrath et al. 2023 (Hydra effect)**: While "hydra" describes surrogate activations after ablation, this paper shows multiple circuits are active simultaneously. The hydra effect is just one perspective of the distributive circuit seen through the lens of ablation.
- **vs Méloux et al. 2025**: They formally proved circuit non-uniqueness in simple models; this paper extends the view to pre-trained LMs and real-world tasks with a unified explanation based on superposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges core implicit assumptions and provides a new algorithm and evaluation protocol; a paradigm-shifting contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically designed experiments across 9 tasks and 20 sheaves are convincing, though model scale and task complexity remain focused on GPT-2 small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and accurate conceptual naming (Functional Anisotropy / Distributive Dense Circuit) with excellent logical flow between theory and evidence.
- Value: ⭐⭐⭐⭐⭐ Forces the mechanistic interpretability community to re-evaluate the explanatory power of their work; highly impactful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
