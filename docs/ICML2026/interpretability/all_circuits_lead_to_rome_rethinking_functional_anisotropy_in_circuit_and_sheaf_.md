---
title: >-
  [Paper Note] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
description: >-
  [ICML 2026][Interpretability][circuit discovery] This paper systematically disproves a hidden assumption in mechanistic interpretability—that "one LLM capability corresponds to a single unique circuit"—using the Overlap-…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "circuit discovery"
  - "sheaf discovery"
  - "functional anisotropy"
  - "superposition"
  - "IOI"
date: 2026-05-08
content_hash: 05387472314ecf89
---

# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**Conference**: ICML 2026  
**arXiv**: [2605.12671](https://arxiv.org/abs/2605.12671)  
**Code**: <https://github.com/TonyXiChen/OASR>  
**Area**: LLM Mechanistic Interpretability / Circuit & Sheaf Discovery  
**Keywords**: circuit discovery, sheaf discovery, functional anisotropy, superposition, IOI

## TL;DR
This paper systematically disproves a hidden assumption in mechanistic interpretability—that "one LLM capability corresponds to a single unique circuit"—using the Overlap-Aware Sheaf Repulsion (OASR) algorithm. It finds that the same task can be supported by multiple circuits or sheaves that exhibit minimal overlap (IoU ~4–11%) while remaining faithful, sparse, and complete. The authors propose the "Distributed Dense Circuit Hypothesis" as a theoretical explanation.

## Background & Motivation

**Background**: Circuit and Sheaf Discovery (CSD) is a mainstream route in mechanistic interpretability. Methods like ACDC, EAP, EP, and DiscoGP represent an LLM as a Directed Acyclic Graph (DAG) of the residual stream and use heuristic or gradient-based edge mask optimization to identify sparse subgraphs that maintain accuracy on tasks such as IOI, Docstring, and BLiMP.

**Limitations of Prior Work**: Existing evaluation paradigms default to a "unique minimal circuit" assumption—either by comparing against a predefined ground-truth (e.g., Tracr) or by pursuing "minimal edge count + performance maintenance." This implicitly assumes each task is implemented by a structurally unique sub-mechanism. However, the authors find that when searching for two sheaves using existing methods, their edges barely overlap despite both being 100% capable of performing the task.

**Key Challenge**: The authors name this implicit assumption the **Functional Anisotropy Hypothesis**—the idea that "model capabilities are anisotropically localized within a specific sub-mechanism." If multiple structurally distinct sheaves can independently support the same task, the goal of "finding the unique circuit" loses its mechanistic significance. Prior redundant phenomena like backup heads or the hydra effect are merely standby mechanisms that "appear only after ablation," which fails to explain the co-existence of multiple circuits during normal inference.

**Goal**: (1) Design an algorithm capable of actively discovering multiple "non-overlapping but faithful" circuits; (2) Systematically disprove the anisotropy hypothesis across multiple common benchmarks; (3) Provide a theoretical explanation for why this non-uniqueness naturally exists in LLMs.

**Key Insight**: Incorporate an "already discovered circuit repulsion" regularization term into the differentiable sheaf optimization objective of DiscoGP. This explicitly reformulates "finding the next sheaf" as "moving away from the previous one in the structural space."

**Core Idea**: Use Overlap-Aware Sheaf Repulsion (OASR) to restructure CSD from "finding the unique minimal subgraph" to "enumerating multiple solutions within a functional equivalence class." High-dimensional superposition is then used to argue that these solutions are too numerous to be reduced to a single canonical circuit.

## Method

### Overall Architecture
The authors build upon the differentiable sheaf discovery framework of DiscoGP. Each edge $e$ in the residual stream computation graph is associated with a learnable logit $l_e$. A continuous score $s_e = \sigma((l_e - \log(\log\mathcal{U}_1/\log\mathcal{U}_2))/\tau)$ is obtained via Gumbel-Sigmoid relaxation, and a hard mask is derived using a straight-through estimator. The original objective includes three terms: fidelity (reproducing task behavior), sparsity (encouraging fewer edges), and completeness (ensuring the masked graph is independently executable). OASR introduces a repulsion term and runs $K$ discovery cycles, requiring each new sheaf to be structurally non-overlapping with all previously discovered sheaves.

### Key Designs

1.  **Overlap-Aware Sheaf Repulsion (OASR) Loss**:
    - **Function**: Explicitly penalizes the "reuse of edges from previous sheaves" during each new discovery cycle, forcing the optimizer to explore untouched regions of the structural space.
    - **Mechanism**: Let $R$ be the set of edges in discovered sheaves. The total loss for a new round is $\mathcal{L} = \mathcal{L}_{fidelity} + \lambda_s\mathcal{L}_{sparsity} + \lambda_c\mathcal{L}_{complete} + \lambda_o\mathcal{L}_{overlap}(R)$, where $\mathcal{L}_{overlap}(R) = \frac{1}{|E|}\sum_{e\in R}\sigma(l_e)$ penalizes the expected activation of edges in $R$. This effectively adds gradients in the logit space to move "away from existing solutions." Repeating this yields $K$ non-overlapping sheaves $\{R_1, \dots, R_K\}$.
    - **Design Motivation**: While re-running DiscoGP with random initialization can yield different sheaves, they often remain highly overlapping as optimization tends to fall into the same attractors. Explicit repulsion turns non-uniqueness into an active discovery process, finding solutions with IoU far below random levels.

2.  **Complementary/Complexity Validation Protocol**:
    - **Function**: Rules out the trivial explanation that "low IoU is due to large circuits being randomly partitioned" and proves each discovered subgraph is truly "independently competent."
    - **Mechanism**: (a) Evaluate IOI accuracy (task maintenance); (b) Evaluate complement accuracy—removing the discovered edges $E_A$ and checking if the remaining graph can perform the task, verifying if $E_A$ is necessary; (c) Report edge density and count, comparing against "random initialization" baselines; (d) Perform "edge-by-edge ablation" on extremely sparse 3-edge sheaves to check if every edge is indispensable.
    - **Design Motivation**: High task accuracy plus low IoU is insufficient to refute anisotropy, as two sheaves might just be rotations or re-parameterizations of the same canonical circuit. Through layer-wise distribution analysis and complement tests, the authors confirm differences are structural rather than superficial rearrangements.

3.  **Distributive Dense Circuit Hypothesis (Theoretical Hypothesis)**:
    - **Function**: Provides a theoretical explanation for why multiple low-overlap faithful circuits must exist in LLMs.
    - **Mechanism**: Based on the superposition theory of Elhage et al.—in high-dimensional residual streams, models represent multiple feature sets via near-orthogonal directional overlays. Any specific "computational implementation" is a linear combination of these directions routed downstream. Given a task behavior $b$, the set of subgraphs satisfying fidelity expands combinatorially with depth and width in high dimensions, causing the number of "sparse yet faithful solutions" to grow exponentially. The paper provides a formal proposition: under mild assumptions, for a sufficiently large model, there exist $\Omega(\exp(d))$ disjoint faithful sheaves.
    - **Design Motivation**: This explains that the observed non-uniqueness is not an optimization artifact of DiscoGP but a structural consequence of LLM representation. It unifies phenomena like backup heads and the hydra effect into a "global distributive circuit" perspective.

### Loss & Training
The weighted sum of four loss terms is used: $\mathcal{L} = \mathcal{L}_{fid} + \lambda_s\mathcal{L}_{sp} + \lambda_c\mathcal{L}_{comp} + \lambda_o\mathcal{L}_{overlap}$. GPT-2 small (12L × 12H) is the primary subject, with hyperparameters following DiscoGP defaults. To obtain 20 sheaves, OASR fixes the $R$ from previous rounds in each iteration, re-initializes logits, and optimizes the joint objective.

## Key Experimental Results

### Main Results
Discovery of two sheaves for the IOI task on GPT-2 small:

| Sheaf | IoI Acc | Comp. Acc | Edge Density | Edge # |
|-------|---------|-----------|--------------|--------|
| A | 100% | 46.20% | 3.56% | 1158 |
| B | 100% | 45.80% | 3.97% | 1289 |
| Overlap | $\|A\cap B\|=96$ | $\|A\cup B\|=2351$ | **IoU = 4.1%** | — |

Discovery of two sheaves across 9 common CSD benchmark tasks:

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
Analysis of "Mutual IoU" extended to 20 sheaves (Mutual IoU = Intersection of 20 sheaves / Union of 20 sheaves):

| Task | Method | \|E_∩\| | \|E_∪\| | Mutual IoU | Avg. Acc |
|------|--------|--------|--------|------------|----------|
| IOI | Random Init | 20 | 6560 | 0.30% | 99.95% |
| IOI | **OASR** | 11 | 7382 | **0.15%** | 99.59% |
| BLiMP | Random Init | 50 | 4858 | 1.03% | 97.26% |
| BLiMP | **OASR** | 37 | 5289 | **0.70%** | 96.11% |
| ANA | Random Init | 26 | 4531 | 0.57% | 96.40% |
| ANA | **OASR** | 10 | 4890 | **0.20%** | 95.00% |

### Key Findings
- The "common intersection among multiple sheaves" approaches zero as the number of discovered sheaves increases: IoU dropped from ~4–11% for 2 sheaves to Mutual IoU < 1% for 20 sheaves, indicating that task capability is not carried by any "essential core."
- An extremely sparse IOI sheaf with only 3 edges was discovered. Removing any one of these 3 edges still allowed OASR to find another high-quality sheaf, disproving the weakened hypothesis of "essential core edges."
- OASR achieved lower Mutual IoU than the "random initialization with DiscoGP" baseline across all tasks while maintaining accuracy, showing the repulsion term is more than trivial random perturbation.
- Layer-wise analysis (Fig. 2) shows that the two sheaves differ most in the distribution of incoming edges to mid-layer MLPs, indicating deeper structural differences rather than surface-level re-parameterization.

## Highlights & Insights
- **Paradigm Shift**: The authors directly challenge the core goal of the mechanistic interpretability community—"finding *the* circuit." If a task has countless faithful solutions, the validity of minimality-based metrics (MIB) and ground-truth comparisons (Tracr) must be reconsidered. This is a "wrong problem definition" level discovery.
- **Generalizable OASR Concept**: Encoding "diverse retrieval" into a differentiable objective via a repulsion term is applicable to any problem seeking to enumerate functionally equivalent solutions—such as sparse dictionaries, multi-solution NAS, or uncovering multiple modes of adversarial samples.
- **Theoretical and Empirical Synergy**: Rather than just reporting experimental phenomena, the authors use superposition to provide a mathematical explanation. They unify backup heads, the hydra effect, and multi-circuits under the "Distributed Dense Circuit" framework.
- **Significance of the 3-edge Sheaf**: The existence of an extremely sparse sheaf where no single edge is indispensable suggests that "minimal" solutions found under sparse optimization might be the least interpretable, as any single edge can be replaced in other sheaves.

## Limitations & Future Work
- Experiments were primarily conducted on GPT-2 small (the authors acknowledge this is due to CSD baseline compatibility). Larger models (Llama/Pythia) were only explored in Appendix H, leaving it partially unresolved whether anisotropy returns as model scale increases.
- Tasks are limited to short-context, single-step linguistic diagnostics (IOI/BLiMP/Docstring). Circuit non-uniqueness for long-context reasoning, coding, or math tasks has not been verified.
- OASR requires $K$ cycles of training, with computational costs scaling linearly with $K$. Setting 20 sheaves is expensive and may be infeasible for larger models.
- The Distributive Dense Circuit Hypothesis provides an existential lower bound but lacks a quantitative construction and relies on "mild assumptions" that require verification in real transformers.
- Is "circuit non-uniqueness → failure of mechanistic interpretation" too pessimistic? The authors suggest no constructive solution for extracting a "consensus mechanism" amidst multiple solutions.

## Related Work & Insights
- **vs Wang et al. 2022a (Original IOI Circuit)**: Proposed "Backup Name-Mover Heads" but interpreted them as mechanisms triggered by ablation. This paper proves backup circuits are active during normal inference; redundancy is the default state rather than an anomaly.
- **vs ACDC / EAP / EP / DiscoGP**: This work utilizes the DiscoGP framework but shifts from "finding the unique optimal" to "enumerating multiple solutions," ultimately proving these methods are affected by the anisotropy assumption.
- **vs McGrath et al. 2023 (Hydra effect)**: While "hydra" describes compensatory activation after ablation, this paper shows multiple circuits are active simultaneously. Thus, the hydra effect is just one perspective of the distributed circuit under ablation.
- **vs Méloux et al. 2025**: They formally proved circuit non-uniqueness in simple models; this paper extends that view to pre-trained LMs and real-world tasks, providing a unified explanation via superposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges the implicit assumptions of the interpretability community with a new hypothesis, algorithm, and evaluation protocol.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic design across 9 tasks and 20 sheaves is convincing, though scale is currently limited to GPT-2 small/medium.
- Writing Quality: ⭐⭐⭐⭐⭐ Terminology (Functional Anisotropy / Distributive Dense Circuit) is clear and accurate, with excellent logical flow between theory and evidence.
- Value: ⭐⭐⭐⭐⭐ Forces the entire mechanistic interpretability community to re-evaluate the explanatory power of their work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)

</div>

<!-- RELATED:END -->
