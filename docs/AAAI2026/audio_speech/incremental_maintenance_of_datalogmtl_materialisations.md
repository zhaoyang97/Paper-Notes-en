---
title: >-
  [Paper Note] Incremental Maintenance of DatalogMTL Materialisations
description: >-
  [AAAI 2026][Audio & Speech][DatalogMTL] This paper proposes the DRed$_{\text{MTL}}$ algorithm, extending the classical Delete/Rederive incremental maintenance technique to DatalogMTL (Datalog with Metric Temporal Logic).…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "DatalogMTL"
  - "incremental reasoning"
  - "materialisation"
  - "Delete/Rederive"
  - "temporal logic"
date: 2026-05-08
content_hash: 2377fbe87298e63e
---

# Incremental Maintenance of DatalogMTL Materialisations

**Conference**: AAAI 2026
**arXiv**: [2511.12169](https://arxiv.org/abs/2511.12169)
**Code**: [GitHub](https://github.com/Horizon12275/DREDmtl-for-DatalogMTL)
**Area**: Audio & Speech
**Keywords**: DatalogMTL, incremental reasoning, materialisation, Delete/Rederive, temporal logic

## TL;DR

This paper proposes the DRed$_{\text{MTL}}$ algorithm, extending the classical Delete/Rederive incremental maintenance technique to DatalogMTL (Datalog with Metric Temporal Logic). By designing novel seminaïve evaluation operators and a periodicity detection algorithm over periodic materialisation representations, the approach enables efficient incremental updates, achieving order-of-magnitude speedups over full rematerialisation.

## Background & Motivation

- DatalogMTL is a temporal extension of Datalog supporting Metric Temporal Logic (MTL) operators, with applications in ontology-mediated query answering, stream reasoning, and financial temporal reasoning.
- In practical scenarios (e.g., anomaly detection in power transformers), data is frequently updated; however, existing DatalogMTL reasoning systems (MeTeoR, vadalog) must recompute materialisations from scratch upon data changes.
- For classical Datalog, incremental maintenance algorithms such as DRed are well established, but extending them to DatalogMTL poses fundamental challenges.

## Core Problem

How to perform incremental materialisation maintenance over the infinite temporal domain of DatalogMTL, avoiding full recomputation upon every data update?

### Key Challenges

1. DatalogMTL supports temporal recursion, and materialisations may span unbounded time intervals, necessitating a **periodic finite representation**.
2. DRed relies on the seminaïve evaluation strategy, but its interaction with the periodic structure of DatalogMTL has not been studied.
3. Termination detection (periodicity recognition) is itself costly, requiring comparison of $O(n)$ facts, and must also be "incrementalised."

## Method

### Periodic Materialisation Representation

Given a bounded program–dataset pair $(\Pi, E)$, its canonical model $\mathfrak{C}_{\Pi,E}$ can be finitely represented via a **saturated interpretation**: left and right periods $\varrho_{\text{left}}, \varrho_{\text{right}}$ are identified such that content outside these intervals repeats periodically. The periodic materialisation $\mathds{I} = \langle I, \varrho_L, \varrho_R \rangle$ recovers the full canonical model via unfolding.

### DRed$_{\text{MTL}}$: Three-Phase Algorithm

1. **Overdeletion**: Starting from the deletion set $E^-$, iteratively applies rules to identify all derived facts depending on deleted facts, using the novel seminaïve operator $\Pi\langle I \vdots \Delta \rangle$ to efficiently locate affected facts.
2. **Rederivation**: Identifies facts that were over-deleted but remain valid, progressively expanding the interval of interest $[t_L, t_R]$ to cover potential recovery ranges.
3. **Insertion**: Propagates the addition set $E^+$, performed in parallel with periodicity detection.

### Key Technical Contributions

- **Novel seminaïve evaluation operator**: Instantiates queries from $\Delta$ (the increment) rather than all facts, enabling lazy unfolding of $\mathds{I}$.
- **Incremental periodicity detection (Pds)**: Searches for periodic structure only within updated facts and their consequences, rather than the entire materialisation (reducing complexity from $O(n)$ to approximately $O(1)$ when the update volume is small).
- **Periodic operators**: Designs Periodic Minus and Periodic Union operations, implemented via Ext (extension to LCM length) and Aln (endpoint alignment).

### Correctness Guarantee

A theorem proves that after executing DRed$_{\text{MTL}}(\Pi, E, \mathds{I}, E^-, E^+)$, $\mathsf{unfold}(\mathds{I}) = \mathfrak{C}_{\Pi, E'}$, where $E' = (E \setminus E^-) \cup E^+$.

## Key Experimental Results

Comparison across three public datasets (DRed$_{\text{MTL}}$ vs. rematerialisation):

| Dataset | \|E\| | Deletions | DRed Delete (s) | Remat Delete (s) | DRed Insert (s) | Remat Insert (s) |
|--------|-------|--------|-------------|------------|-----------|-------------|
| LUBM$_t$ | 630.5k | 100 | 0.7k | 48.6k | 0.4k | 48.5k |
| iTemporal | 46.4k | 100 | 8.1k | 52.7k | 8.7k | 52.8k |
| Meteorological | 62.01M | 100 | 10 | 0.7k | 11 | 0.7k |

- On the Meteorological dataset, DRed$_{\text{MTL}}$ is approximately **70× faster** than rematerialisation.
- Under large deletion volumes on LUBM$_t$ (63.1k deletions), significant speedup is still achieved: 3.4k vs. 45.0k seconds.

## Highlights & Insights

- This is the first incremental maintenance algorithm for DatalogMTL, filling a notable gap in the field.
- The theoretical contributions are rigorous, with complete correctness proofs (including a full technical report).
- The incremental periodicity detection strategy is elegant — searching for periodic structure only within the "diff."
- The implementation is built upon MeTeoR, demonstrating practical applicability.

## Limitations & Future Work

- Only bounded-interval DatalogMTL is supported; the unbounded case is not addressed.
- The overdeletion problem inherent to DRed may be amplified in the MTL setting.
- Incremental advantages diminish when the update volume approaches the full dataset size.
- Extensions of more advanced incremental Datalog algorithms (e.g., B/F or FBF) to the MTL setting remain to be explored.

## Related Work & Insights

| Dimension | MeTeoR | vadalog | DRed$_{\text{MTL}}$ |
|------|--------|---------|-------------------|
| Reasoning Paradigm | Hybrid (bottom-up + top-down) | Bottom-up | Incremental bottom-up |
| Termination Guarantee | Yes (bounded programs) | No (recursion may not terminate) | Yes |
| Incremental Updates | ✗ | ✗ | ✓ |

- The combination of periodic representation and incremental maintenance is generalizable to other temporal reasoning frameworks.
- In streaming temporal data settings, incremental reasoning can substantially reduce computational overhead.

## Rating

⭐⭐⭐⭐ — Theoretically rigorous, algorithmically novel, and experimentally well-validated; a significant contribution to the temporal reasoning literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[AAAI 2026\] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning](towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)

</div>

<!-- RELATED:END -->
