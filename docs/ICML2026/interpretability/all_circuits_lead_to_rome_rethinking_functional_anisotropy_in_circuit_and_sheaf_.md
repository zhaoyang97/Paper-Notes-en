---
title: >-
  [Paper Note] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
description: >-
  [ICML 2026][Interpretability][circuit discovery] This paper systematically falsifies a core implicit assumption in mechanistic interpretability—"one LLM capability corresponds to a unique circuit"—using the Overlap-Aware…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "circuit discovery"
  - "sheaf discovery"
  - "functional anisotropy"
  - "superposition"
  - "IOI"
date: 2026-05-08
content_hash: 43b8c84147631e9f
---

# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.12671](https://arxiv.org/abs/2605.12671)  
**Code**: <https://github.com/TonyXiChen/OASR>  
**Area**: LLM Interpretability / Mechanistic Interpretability / Circuit & Sheaf Discovery  
**Keywords**: circuit discovery, sheaf discovery, functional anisotropy, superposition, IOI

## TL;DR
This paper systematically falsifies a core implicit assumption in mechanistic interpretability—"one LLM capability corresponds to a unique circuit"—using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It finds that the same task can be supported by multiple, nearly non-overlapping (IoU ~4–11%) but all faithful/sparse/complete circuits or sheaves, and proposes the "Distributive Dense Circuit Hypothesis" as a theoretical explanation.

## Background & Motivation

**Background**: Circuit and sheaf discovery (CSD) is a mainstream approach in mechanistic interpretability. From ACDC, EAP to EP, DiscoGP, these methods encode LLMs as residual flow computation DAGs, then use heuristic or gradient-based edge mask optimization to find sparse subgraphs that maintain accuracy on tasks like IOI, Docstring, BLiMP.

**Limitations of Prior Work**: Existing evaluation paradigms assume a "unique minimal circuit"—either by comparing to preset ground-truths (e.g., Tracr) or by seeking "fewest edges + maintained performance," implicitly assuming each task is implemented by a structurally unique submechanism. However, the authors find that when using current methods to find two sheaves, their edges barely overlap yet both achieve 100% task performance.

**Key Challenge**: The authors term this implicit assumption the **Functional Anisotropy Hypothesis**—"model capabilities are anisotropically localized in a particular submechanism." If multiple structurally distinct sheaves can independently support the same task, then "finding the unique circuit" loses mechanistic explanatory value; prior phenomena like backup head/hydra effect are only "revealed after ablation" and cannot explain "multiple circuits coexisting during normal inference."

**Goal**: (1) Design an algorithm that actively discovers "mutually non-overlapping but all faithful" multiple circuits; (2) Systematically falsify the anisotropy hypothesis on several standard benchmarks; (3) Provide a theoretical explanation for why such non-uniqueness naturally exists in LLMs.

**Key Insight**: Add a "repel discovered circuits" regularizer to the differentiable sheaf optimization objective in DiscoGP, explicitly turning "finding the next sheaf" into "structurally distancing from the previous one" in solution space.

**Core Idea**: Use Overlap-Aware Sheaf Repulsion (OASR) to reframe CSD from "finding the unique minimal subgraph" to "enumerating multiple solutions within the functional equivalence class," and use high-dimensional superposition theory to argue that these solutions are so numerous they cannot be reduced to a single canonical circuit.

## Method

### Overall Architecture
The authors adopt DiscoGP's differentiable sheaf discovery framework: each edge $e$ in the residual flow computation graph is associated with a learnable logit $l_e$, relaxed to a continuous score $s_e=\sigma((l_e-\log(\log\mathcal{U}_1/\log\mathcal{U}_2))/\tau)$ via Gumbel-Sigmoid, and a hard mask is obtained via straight-through estimator. The original objective includes fidelity (reproducing task behavior), sparsity (encouraging sparsity), and completeness (mask graph is independently executable). OASR adds a repulsion term, running $K$ rounds of discovery, each time requiring the new sheaf to be structurally almost non-overlapping with all previously found sheaves.

### Key Designs

1. **Overlap-Aware Sheaf Repulsion (OASR) Loss**:

    - **Function**: Explicitly penalizes "reusing edges from previous sheaf" in each new discovery, pushing the optimizer to explore unexplored regions of the structural space.
    - **Mechanism**: Let $R$ be the set of edges in discovered sheaves; the total loss for a new round is $\mathcal{L}=\mathcal{L}_{fidelity}+\lambda_s\mathcal{L}_{sparsity}+\lambda_c\mathcal{L}_{complete}+\lambda_o\mathcal{L}_{overlap}(R)$, where $\mathcal{L}_{overlap}(R)=\frac{1}{|E|}\sum_{e\in R}\sigma(l_e)$ penalizes expected activation only on edges in $R$, effectively adding gradient in logit space away from existing solutions. Repeating this yields $K$ mutually non-overlapping sheaves $\{R_1,\dots,R_K\}$.
    - **Design Motivation**: Simply re-running DiscoGP with random initialization can yield different sheaves, but they may still heavily overlap (optimizer tends to fall into the same attractor). The explicit repulsion term turns "non-uniqueness" from a passive phenomenon into active discovery, enabling solutions with IoU far below random baseline.

2. **Complement/Complex Validation Protocol**:

    - **Function**: Rules out trivial explanations like "low IoU is because circuits are too large and any random cut won't overlap," proving each discovered subgraph is truly "independently competent."
    - **Mechanism**: (a) Evaluate IoI accuracy (task retention); (b) Evaluate complement accuracy—remove the discovered edge set $E_A$, test if the remaining graph can perform the task, verifying $E_A$ is necessary; (c) Report edge density (selected edges/total edges) and edge count, comparing to "random init rerun" baseline; (d) For extremely sparse three-edge sheaves, perform "edge-by-edge ablation" to check if any edge is indispensable.
    - **Design Motivation**: Task accuracy consistency + low IoU alone is insufficient to refute anisotropy, as two sheaves might just be rotations/reparameterizations of the same canonical circuit. Layer-wise distribution analysis + complement tests confirm the differences are structural, not superficial rearrangements.

3. **Distributive Dense Circuit Hypothesis (Theoretical Hypothesis)**:

    - **Function**: Provides a theoretical explanation for "why LLMs must have multiple low-overlap faithful circuits."
    - **Mechanism**: Based on Elhage et al.'s superposition theory—in high-dimensional residual flows, the model uses nearly orthogonal directions to superpose multiple feature sets, and any specific "computation implementation" routes these directions to downstream via some linear combination. For a given task behavior $b$, the set of subgraphs satisfying fidelity grows exponentially with model depth/width, so "sparse yet faithful solutions" proliferate exponentially. The paper formalizes: under mild assumptions, for sufficiently large models, there exist $\Omega(\exp(d))$ mutually disjoint faithful sheaves.
    - **Design Motivation**: Explains why the observed non-uniqueness is not an optimization artifact of DiscoGP, but a structural consequence of LLM representations—unifying backup heads/hydra effect/multi-circuit phenomena under a "global distributive dense circuit" perspective.

### Loss & Training
The four losses are weighted and summed: $\mathcal{L}=\mathcal{L}_{fid}+\lambda_s\mathcal{L}_{sp}+\lambda_c\mathcal{L}_{comp}+\lambda_o\mathcal{L}_{overlap}$. GPT-2 small (12L × 12H) is the main experimental subject, with hyperparameters following DiscoGP defaults. To obtain 20 sheaves, OASR fixes the previously found $R$ each round, reinitializes logits, and optimizes the joint fidelity/sparsity/completeness/overlap objective.

## Key Experimental Results

### Main Results
On GPT-2 small, OASR discovers two sheaves for the IOI task:

| Sheaf | IoI Acc | Comp. Acc | Edge Density | Edge # |
|-------|---------|-----------|--------------|--------|
| A | 100% | 46.20% | 3.56% | 1158 |
| B | 100% | 45.80% | 3.97% | 1289 |
| Overlap | $\|A\cap B\|=96$ | $\|A\cup B\|=2351$ | **IoU = 4.1%** | — |

Across 9 standard CSD benchmark tasks, two sheaves are found per task:

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
Analysis of "mutual intersection" for 20 sheaves (Mutual IoU = total intersection/total union for 20 sheaves):

| Task | Method | \|E_∩\| | \|E_∪\| | Mutual IoU | Mean Acc |
|------|--------|--------|--------|------------|----------|
| IOI | Random Init | 20 | 6560 | 0.30% | 99.95% |
| IOI | **OASR** | 11 | 7382 | **0.15%** | 99.59% |
| BLiMP | Random Init | 50 | 4858 | 1.03% | 97.26% |
| BLiMP | **OASR** | 37 | 5289 | **0.70%** | 96.11% |
| ANA | Random Init | 26 | 4531 | 0.57% | 96.40% |
| ANA | **OASR** | 10 | 4890 | **0.20%** | 95.00% |

### Key Findings
- The "common intersection among multiple sheaves" approaches zero as the number of discovered sheaves increases: from 2 sheaves with IoU ≈ 4–11% to 20 sheaves with Mutual IoU < 1%, indicating that task capability is not carried by any "essential core."
- An extremely sparse IOI sheaf with only 3 edges was found; edge-by-edge ablation shows that removing any edge still allows OASR to find another high-quality sheaf, falsifying the weakened "necessary core edge" hypothesis.
- OASR achieves lower Mutual IoU than "rerunning DiscoGP with random initialization" on all tasks, with almost no drop in accuracy, indicating the repulsion term is not a trivial random perturbation.
- Layer-wise analysis (Fig. 2) shows the two sheaves differ most in mid-layer MLP input edge distributions, not just superficial reparameterization.

## Highlights & Insights
- **Paradigm-level Impact**: The authors directly challenge the mechanistic interpretability community's core goal of "finding the circuit." If a task has countless faithful solutions, then minimality-based evaluation (MIB) and ground-truth comparison (Tracr) must be reconsidered. This is a "problem definition is wrong" level discovery.
- **OASR's Generalizability**: Encoding "diversity search" as a repulsion term in a differentiable objective is applicable to any problem where one wants to enumerate functionally equivalent solutions—e.g., sparse feature dictionaries, multi-solution NAS, adversarial example multi-mode discovery, etc.
- **Theory-Empirics Synergy**: The authors not only present experimental phenomena but also provide a mathematical explanation via superposition, unifying backup head/hydra effect/multi-circuit phenomena under the "distributive dense circuit" framework, elevating the contribution.
- **Significance of 3-edge Sheaf**: The existence of an extremely sparse sheaf with no indispensable edge suggests that "minimal" solutions found by sparse optimization may be the least interpretable—any edge can be replaced in other sheaves.

## Limitations & Future Work
- Experiments are mainly on GPT-2 small (since all CSD baselines only support it); extension to larger models (Llama/Pythia) is only in Appendix H, not fully addressing whether anisotropy returns at scale.
- Tasks are all short-context, single-step linguistic diagnostics (IOI/BLiMP/Docstring); non-uniqueness of circuits for long reasoning, code, math, etc., remains untested.
- OASR requires $K$ rounds of training, with computational cost growing linearly with $K$; 20 sheaves is already expensive, possibly infeasible for larger models.
- The Distributive Dense Circuit Hypothesis provides an existence lower bound but no quantitative construction, and relies on mild assumptions—whether these hold in real transformers remains to be tested.
- Is "circuit non-uniqueness → mechanistic explanation failure" too pessimistic? The authors do not offer constructive approaches for extracting consensus mechanisms in the multi-solution case.

## Related Work & Insights
- **vs Wang et al. 2022a (Original IOI Circuit)**: Proposed the Backup Name-Mover Heads concept, but explained it as "ablation-triggered backup mechanism"; this paper shows backups are active during normal inference, and redundancy is the default, not an exception.
- **vs ACDC / EAP / EP / DiscoGP**: This work adopts DiscoGP's differentiable optimization framework but shifts from "finding the unique optimal solution" to "enumerating multiple solutions," and uses these methods to verify they all suffer from the anisotropy assumption.
- **vs McGrath et al. 2023 (Hydra effect)**: Hydra describes backup activation after ablation; this paper shows multiple circuits are simultaneously active, so hydra is just one aspect of distributive circuits from the ablation perspective.
- **vs Méloux et al. 2025**: They formally proved circuit non-uniqueness in simple models; this paper extends the perspective to pretrained LMs and real tasks, providing a unified explanation via superposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Directly challenges the implicit assumption of the interpretability community, proposing a new hypothesis + algorithm + evaluation protocol—a paradigm-level impact.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic design with 9 tasks × 20 sheaves is convincing, but model scale and task complexity are still limited to GPT-2 small.
- Writing Quality: ⭐⭐⭐⭐⭐ Concept naming (Functional Anisotropy / Distributive Dense Circuit) is clear and precise, logical structure is excellent, with theory and empirics mutually reinforcing.
- Value: ⭐⭐⭐⭐⭐ Forces the mechanistic interpretability community to reassess the explanatory power of its work, with far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICML 2026\] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](../../NeurIPS2025/interpretability/discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
