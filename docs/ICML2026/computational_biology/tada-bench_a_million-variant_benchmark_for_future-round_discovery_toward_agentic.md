---
title: >-
  [Paper Note] TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering
description: >-
  [ICML 2026][Computational Biology][protein engineering] TadA-Bench utilizes million-level TadA variant sequences from 31 rounds of real-world wet-lab directed evolution experiments to formalize protein engineering as a f…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "protein engineering"
  - "directed evolution"
  - "future-round discovery"
  - "benchmark"
  - "biological foundation models"
date: 2026-05-08
content_hash: efa8fedf1f589678
---

# TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering

**Conference**: ICML 2026  
**arXiv**: [2606.02624](https://arxiv.org/abs/2606.02624)  
**Code**: Open Source (Hugging Face + GitHub)  
**Area**: Protein Engineering / AI for Science / Benchmark Evaluation  
**Keywords**: protein engineering, directed evolution, future-round discovery, benchmark, biological foundation models  

## TL;DR
TadA-Bench utilizes million-level TadA variant sequences from 31 rounds of real-world wet-lab directed evolution experiments to formalize protein engineering as a fixed-data replay task of "predicting future rounds using preceding ones." Accompanied by the Seq2Graph graph-based label unification pipeline, it reveals that mainstream biological foundation models fail significantly in "future-round discovery."

## Background & Motivation
**Background**: Protein engineering is transitioning from "one-off predictors" to "agentic iterative closed loops," where models read wet-lab history, call analysis tools, recommend next-round variants, and return them for wet-lab verification. This requires evaluation data with three attributes: temporal replayability, exploration scale, and cross-round label consistency.

**Limitations of Prior Work**: Current functional benchmarks (biophysical properties, DMS aggregations like ProteinGym) focus on "width"—maximizing protein families and assays. However, they either lack a real timeline or cover only local fitness landscapes, failing to assess the ranking ability crucial for "predicting future rounds based on the past" in a closed loop. Data specifically for base editor deaminases is highly fragmented, with most focusing on Cas/sgRNA interactions rather than the deaminase itself, and cross-laboratory merging introduces significant batch effects.

**Key Challenge**: Standard random split evaluations measure interpolation capability, whereas real closed-loop protein engineering requires extrapolation. The community lacks a hard benchmark with a "single campaign, deep temporal chain, and unified labels" to determine the magnitude of this gap and whether it can be bridged simply by choosing a superior regression head.

**Goal**: (1) Construct a deep (31 rounds), large-scale (million variants), single directed evolution dataset with a clear temporal chain; (2) Convert "local ranking + cross-round anchors" from multi-round NGS enrichment counts into globally consistent continuous activity labels; (3) Define a fixed past-to-future replay protocol to evaluate DNA/RNA/Protein foundation models using unified metrics.

**Key Insight**: The authors selected TadA (a deaminase for Adenine Base Editors) and performed 31 rounds of PANCE directed evolution. They treated NGS enrichment data from each round as local partial order constraints and used graph-theoretic methods to remove cycles and anchor them to the known TadA8e reference sequence, obtaining cross-round comparable activity labels.

**Core Idea**: The "wet-lab directed evolution trajectory" is treated as a fixed-data replay task. The true recommendation capabilities of current biological foundation models are exposed through future-round ranking and limited-budget selection metrics. By comparing "coverage vs. local density at matching scales," it is demonstrated that "evolutionary coverage is more informative than dense local sampling."

## Method

### Overall Architecture
TadA-Bench consists of three components: (a) **Data Foundation**—NGS sequences from 31 rounds of TadA PANCE directed evolution, providing aligned DNA/RNA/Protein views; (b) **Seq2Graph Label Unification Pipeline**—compresses multi-round enrichment counts into a weighted directed graph, removes cycles, and performs score propagation in the log domain with TadA8e=1.0 as an anchor to output continuous relative activity; (c) **Replay Protocol & Metrics**—fixes rounds 1-27 for training, 28 for validation, and 29-31 for testing with non-overlapping sequences, using Spearman, Recall@10%, and nDCG@10% to measure the capability of "predicting the future from the past." On the model side, a unified protocol of frozen encoders + common downstream regression heads is implemented, supplemented by "adaptation + discovery mode" checks involving full fine-tuning, prompt tuning, and limited-budget candidate selection.

### Key Designs

1.  **Seq2Graph Cross-round Label Unification**:
    - **Function**: Transforms NGS enrichment counts from 31 rounds, each containing batch effects, into a continuous activity label comparable across all variants.
    - **Mechanism**: Each unique DNA sequence is a graph node. Within rounds, variants are ranked by enrichment readings, and edges (high → low) are added only between adjacent variants with the local enrichment ratio as the weight. Cross-round connectivity is achieved through identical sequences acting as anchors. This "local relative comparison + sequence overlap anchoring" avoids cross-round normalization of absolute enrichment, providing resistance to batch effects. The scale supports million-node graphs.
    - **Design Motivation**: Direct concatenation of normalized reads is dominated by batch effects, and global regression is affected by duplicate variants and platform noise. Abstracting information into "which is stronger" via a graph matches the nature of multi-round NGS data.

2.  **Weighted Feedback Arc Set Cycle Removal + Log-Domain Score Propagation**:
    - **Function**: Eliminates inconsistency cycles ($v_i>v_j, v_j>v_k, v_k>v_i$) caused by noise to generate a globally consistent DAG, then diffuses activity scores using TadA8e=1.0 as the anchor.
    - **Mechanism**: Modeling "removing edges with minimum weight to make the graph acyclic" as $$\min_{F\subseteq E}\sum_{e\in F}w_e$$ s.t. $G\setminus F$ is a DAG. FAS is approximated via greedy heuristics within strongly connected components (Eades et al., 1993). After cleaning, log-domain propagation is performed from TadA8e via "fewest-edge paths"—since enrichment ratios are multiplicative, log-domain addition is equivalent to the product along the path. Selecting paths with the fewest edges minimizes noise accumulation.
    - **Design Motivation**: The authors emphasize this as a "data integration pipeline" rather than a "graph learning contribution." They explicitly state that edges and paths serve only for consistency correction and score propagation and **should not be interpreted as biological ancestry**.

3.  **Fixed Future-Round Replay Protocol + DNA/RNA/Protein Tri-view Alignment**:
    - **Function**: Compresses the "past → future" closed-loop protein engineering operation into a reproducible computational evaluation and allows cross-modal foundation models to be compared on the same campaign.
    - **Mechanism**: Using round $k$ as a cutoff, models train only on $D_{\le k}$ and rank variants appearing only in $D_{>k}$. The main benchmark uses 1-27 for training, 28 for validation, and 29-31 for testing. The three views originate from the same NGS: DNA from sequencing, RNA via T→U substitution, and protein via codon translation with activity averaged across synonymous codons. This results in 729k+148k+150k DNA sequences and 256k+45k+108k independent protein sequences.
    - **Design Motivation**: Random splitting allows models to "interpolate" the past via similar modes, masking extrapolation failure. Fixed future rounds and non-overlapping sequence splits force the evaluation of "future-round discovery," the true bottleneck.

### Loss & Training
The main model protocol employs frozen encoders + a unified regression head, utilizing MSE regression for continuous activity on the training set. The validation set (round 28) is used only for learning rate selection. Adaptation checks include full fine-tuning and prompt tuning to rule out "probe weakness" as the cause of poor performance. "Discovery-mode" checks simulate wet-lab budgets: given model top-N candidates, the number of truly high-activity future-round variants captured is measured.

## Key Experimental Results

### Main Results

| View | Model | Spearman ↑ | Recall@10% ↑ | nDCG@10% ↑ |
|------|------|------------|--------------|------------|
| DNA | Evo2-7B | 0.0707 | 0.1005 | 0.3236 |
| DNA | Evo2-40B | 0.0675 | 0.1003 | 0.3244 |
| DNA | NT-500M | 0.0189 | 0.1005 | 0.3079 |
| RNA | OG-46M | 0.0079 | 0.1063 | 0.3158 |
| Protein | ESM2-650M | 0.0479 | 0.1120 | 0.2791 |
| Protein | ESMC-600M | **0.0509** | 0.1180 | 0.2860 |
| Protein | Prot-XLNET | 0.0342 | 0.1175 | 0.2895 |

Main Conclusion: Across DNA/RNA/Protein views, the Spearman $\rho$ for all frozen probes falls below 0.1, significantly lower than correlations observed under random split controls. This indicates that existing biological foundation models possess almost no effective ranking signal for "future rounds," and the bottleneck is not restricted to a specific modality or family.

### Ablation Study

| Configuration | Phenomenon | Meaning |
|------|------|------|
| Random Split (IID) | Protein view Spearman rises significantly to "strong interpolation" levels | The labels are learnable; the issue is not Seq2Graph noise |
| Future-Round (Default) | Spearman ≤ 0.1 | Extrapolation fails; the bottleneck is the "past → future" transition |
| Full fine-tune | Limited improvement; gap remains | Not a matter of probe capacity |
| Prompt tuning | As above | Not a matter of input conditioning |
| Top-N Budget Selection | Recall@10% remains weak | Even with realistic wet-lab budgets, hit rates remain low |
| Match Scale: Coverage vs. Density | Models trained on high evolutionary coverage subsets extrapolate better | "Coverage is more important than density"—benchmarks should preserve campaign structure |

### Key Findings
- **Interpolation Capacity $\neq$ Future-Round Discovery Capacity**: Models appearing "competent" on random splits see Spearman drop to near 0 under the fixed future-round protocol, a gap consistent across DNA/RNA/Protein views.
- **Adaptation Cannot Rescue Performance**: Full fine-tuning and prompt tuning failed to close the gap, implying the issue lies in the lack of extrapolation signals in learned representations rather than "weak probes."
- **Evolutionary Coverage Outperforms Local Density**: At matched training scales, training sets covering multiple lineages are more conducive to future-round extrapolation than repeatedly sampling around known hits, supporting the design principle of retaining campaign structures.
- **Limited-Budget Selection is Weak**: Poor performance in top-N selection indicates that real-world hit rates for recommending high-activity variants to wet labs are limited, representing a clear bottleneck for agentic loops.

## Highlights & Insights
- **Clear Definition of "Wet-lab Replay" Paradigm**: Stripping the core sub-tasks of agentic protein engineering into "fixed data + temporal shifts + ranking metrics" compresses expensive closed-loop problems into reproducible offline protocols.
- **Seq2Graph as Data Infrastructure**: The authors modestly describe it as "data integration rather than graph learning innovation," yet using FAS cycle removal + log-domain propagation to merge multi-round NGS is a transferable toolset for other high-throughput screenings (e.g., Cas9).
- **Tri-view Homologous Alignment**: Generating aligned DNA/RNA/Protein sequences from the same NGS data provides a fair "same problem, different languages" arena for cross-modal foundation models.
- **Conclusions as Research Directions**: Quantifying the failure in "future-round extrapolation" provides immediate insights, such as "coverage is more important," which serves as a methodological guide for agentic AI for Science.

## Limitations & Future Work
- **Single Protein Family**: Although 31 rounds × million variants is rare, the benchmark covers only the TadA protein and one selection-coupled assay. Cross-family generalizability requires further study.
- **Fixed-Data vs. True Closed-Loop**: The protocol does not evaluate proposal, planning, tool-use, or automated wet-lab execution, serving only as a diagnostic for the "ranking sub-module" in an agentic loop.
- **Sequence-Defined vs. Design-Defined Labels**: Activity reflects holistic cellular performance (expression + folding/stability + editing activity) rather than isolated catalytic constants; signals are composite.
- **Path Selection $\neq$ Ancestry**: The authors clarify this distinction, but the community may still misinterpret edges as evolutionary lineages.
- **Future Directions**: Extending to multi-family PANCE data; adding active learning/acquisition function evaluation layers; providing interfaces for candidate generation modules.

## Related Work & Insights
- **vs. ProteinGym / FLIP / ProteinBench (Notin et al., 2023; Dallago et al., 2021; Ye et al., 2025)**: These benchmarks seek "width" through multi-family DMS aggregation, whereas TadA-Bench seeks "depth" through a single campaign and fixed future-round protocols.
- **vs. CRISPRbase and Base Editor Aggregated Datasets (Fan et al., 2023; Dixit et al., 2024)**: Merging across labs introduces batch effects; TadA-Bench uses a single assay source and repairs consistency via Seq2Graph.
- **vs. Biological Foundation Model Evaluations (ESM2/ESMC, Evo 2, NT, OmniGenome)**: Previous papers focused on zero-shot/DMS interpolation; TadA-Bench provides a true extrapolation testbed, challenging the assumption that larger encoders automatically equate to better scientific discovery.
- **vs. ML-guided Directed Evolution / ALDE Methods**: While not evaluating the proposal layer, isolating "ranking" provides a clean sub-module testbed for these methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Task paradigm and label pipeline are rare in biological benchmarks; metrics are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive cross-view testing with 7+ models, random-split controls, and adaptation checks.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts with well-defined limitations (e.g., distinguishing from ancestry/graph learning).
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous benchmark for agentic AI4Science, with conclusions pointing toward future methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/computational_biology/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](../../ICLR2026/computational_biology/how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[ICML 2026\] Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback](influence-guided_symbolic_regression_scientific_discovery_via_llm-driven_equatio.md)
- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](../../ACL2026/computational_biology/toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[ICML 2026\] LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation](lineageflow_flow_matching_for_high-fidelity_family-aware_protein_sequence_genera.md)

</div>

<!-- RELATED:END -->
