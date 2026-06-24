---
title: >-
  [Paper Note] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection
description: >-
  [ACL 2025][LLM Alignment][Instruction Tuning Data Selection] This paper proposes G2IS (Gradient-based Graph Instruction Selection), which models the joint distribution and mutual dependencies among instruction data by constructing a gradient-based instruction graph. Combined with a gradient walk algorithm for data selection, it outperforms full-dataset instruction tuning using only 1% of the data.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Instruction Tuning Data Selection"
  - "Gradient Graph"
  - "Domain Adaptation"
  - "Joint Distribution"
  - "Graph Walk Algorithm"
date: 2026-05-08
content_hash: e3323484a7dadb16
---

# Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection

**Conference**: ACL 2025  
**arXiv**: [2502.11062](https://arxiv.org/abs/2502.11062)  
**Code**: None  
**Area**: LLM Alignment/Instruction Tuning  
**Keywords**: Instruction Tuning Data Selection, Gradient Graph, Domain Adaptation, Joint Distribution, Graph Walk Algorithm

## TL;DR
This paper proposes G2IS (Gradient-based Graph Instruction Selection), which models the joint distribution and mutual dependencies among instruction data by constructing a gradient-based instruction graph. Combined with a gradient walk algorithm for data selection, it outperforms full-dataset instruction tuning using only 1% of the data.

## Background & Motivation

**Background**: Instruction tuning is a key method for adapting LLMs to specific domains, but domain-specific data is often limited. Data selection methods compensate for this by selecting the most relevant samples from large-scale general datasets. Methods like LESS select training data most similar to the validation set based on gradient similarity.

**Limitations of Prior Work**: Existing methods primarily focus on the independent similarity of training samples to the target domain, ignoring the interdependence between instruction data. For example, two instructions teaching "addition" and "carrying" individually might not seem similar enough to a "multi-digit multiplication" target, but combined they provide critical foundational knowledge.

**Key Challenge**: Instruction data forms a joint distribution with complementary, redundant, or conflicting relationships between instructions, whereas existing similarity-based methods treat each sample independently, leading to suboptimal data combinations.

**Goal**: To design a data selection method capable of capturing dependencies among instructions, selecting the subset of instructions from a general dataset that contributes the most to the target task.

**Key Insight**: Gradients naturally encode the impact of training samples on model parameter updates, and the relationships between gradients can reflect the complementary and conflicting dynamics between instructions.

**Core Idea**: Construct an instruction graph using gradient representations (nodes = instruction gradients, edges = gradient cosine similarity), extract core knowledge from validation set gradients using PCA, and then employ a gradient walk algorithm on the graph to select an optimal instruction subset that satisfies three constraints (no knowledge conflict, consistency with core knowledge, and knowledge coherence).

## Method

### Overall Architecture
The input consists of a large-scale general instruction training set and a small-scale target domain validation set, and the output is the selected training subset. The workflow includes: (1) computing gradient representations for all samples; (2) constructing a gradient graph; (3) extracting core knowledge from the validation set; and (4) selecting data based on gradient walk. Finally, the LLM is instruction-tuned on the selected subset.

### Key Designs

1. **Gradient-based Knowledge Representation**:

    - **Function**: Representing the knowledge content of each instruction sample using a gradient vector.
    - **Mechanism**: Apply momentum-adjusted gradients $\nabla\Gamma(z, \theta_t)$ for the training set, initializing the momentum state of the Adam optimizer via a warmup method to ensure gradients capture actual optimization dynamics. Standard SGD gradients are used for the validation set to avoid momentum interference. Gradients from LoRA layers combined with random projection are utilized to reduce the dimensionality to 8192.
    - **Design Motivation**: Standard gradients ignore optimizer states, whereas momentum-adjusted gradients more accurately reflect a sample's contribution to model learning. LoRA gradients combined with random projection significantly reduce computational costs.

2. **Gradient Graph Construction and Core Knowledge Extraction**:

    - **Function**: Modeling the dependencies between instructions and extracting the core capabilities required for the target task.
    - **Mechanism**: Treat the gradient of each training sample as a node $N_z = \nabla\Gamma(z, \theta_t)$, and edge weights as gradient cosine similarity $R_{ij} = \cos(\nabla\Gamma(z_i, \theta_t), \nabla\Gamma(z_j, \theta_t))$. Positive values represent knowledge alignment, and negative values denote conflict. Apply PCA to validation set gradients, extracting the top 50% principal components as core knowledge $K_\mathcal{V}$.
    - **Design Motivation**: PCA filters noise from the validation set to extract the most crucial task-related capabilities. The graph structure allows for explicit modeling of complementary/conflicting relationships between instructions.

3. **Gradient Walk Algorithm**:

    - **Function**: Expanding the instruction subset incrementally on the gradient graph under constraints.
    - **Mechanism**: Starting from anchor points most similar to the core knowledge, new nodes are iteratively selected based on three principles: (1) **No knowledge conflict**—gradient similarity between the new sample and already selected samples is non-negative; (2) **Consistency with core knowledge**—after adding the new sample, the cosine similarity of the subset with core knowledge $K_\mathcal{V}$ is no less than $\delta=0.8$ times the current value; (3) **Knowledge coherence**—the new sample is most gradient-similar to the most recently added sample. This is formalized as $z^* = \arg\max_{z \in \mathcal{Z}} \cos(\nabla\Gamma(z, \theta_t), \nabla\Gamma(s^*, \theta_t))$ satisfying the above constraints.
    - **Design Motivation**: The three constraints ensure that selected data are complementary without conflicts, aligned with the target, and internally coherent.

### Loss & Training
After data selection, LoRA fine-tuning is performed (rank=128, α=512) with a cosine learning rate scheduler for 3 epochs on an A100 GPU cluster.

## Key Experimental Results

### Main Results (1% Data vs. Full Data, Llama3.1-8B + Infinity-Instruct)

| Task | G2IS-1% | LESS-1% | BERT-1% | All(100%) |
|------|---------|---------|---------|-----------|
| BBH | **64.78** | 63.46 | 63.05 | 64.71 |
| GPQA | **31.57** | 29.94 | 30.55 | 29.74 |
| GSM8K | **62.02** | 60.65 | 56.79 | 58.30 |
| Math | **20.96** | 18.66 | 20.22 | 20.26 |
| MMLU | **63.42** | 63.15 | 61.87 | 62.75 |

### Ablation Study (COT Dataset, Llama3.1-8B, 1%)

| Configuration | BBH | GPQA | GSM8K | Math | MMLU | vs Full |
|------|-----|------|-------|------|------|---------|
| G2IS | **65.66** | **32.59** | **62.70** | **21.38** | **64.22** | 1.00 |
| w/o graph | 65.64 | 30.55 | 57.85 | 20.10 | 61.42 | 0.95 |
| w/o gradient | 64.57 | 32.53 | 58.91 | 20.24 | 63.94 | 0.97 |

### Key Findings
- Using only 1% of the data, G2IS outperforms full-dataset instruction tuning on most tasks, notably improving Gemma-7B by 12.66% on GSM8K.
- The graph structure (gradient walk) contributes more to performance than the gradient representation itself (with "w/o graph" exhibiting a larger drop), demonstrating that modeling instruction interdependence is crucial.
- In multi-task optimization, LESS suffers from performance degradation while G2IS remains robust, as graph constraints effectively balance multiple objectives.
- Selecting 5% of the data sometimes yields worse performance than 1%, reinforcing the "less is more" principle as excessive data introduces noise.
- The ratio of PCA principal components has a significant impact on MMLU (multi-domain knowledge, noisier) but a minor effect on GSM8K (mathematical reasoning, less noise).

## Highlights & Insights
- **Paradigm Shift from Independent to Joint Selection**: Elevates data selection from "individual scoring" to "combinatorial optimization," where the graph structure naturally models the synergistic and conflicting relations among samples. This paradigm can be migrated to any scenario requiring data subset selection (e.g., active learning, curriculum learning).
- **1% Data Outperforming 100%**: Consistently surpasses full-dataset training across multiple models and datasets using only 1% of the data. This strongly validates the "quality over quantity" principle and offers practical significance for real-world LLM training.

## Limitations & Future Work
- The approach relies solely on LoRA-layer gradients rather than full-parameter gradients, which might discard some information.
- Experiments are only conducted on 7B-8B models; the effectiveness on larger models (13B/65B/175B) remains unverified.
- Gradient computation still requires forward-backward passes for each sample, making the computational cost on extremely massive datasets a potential concern.
- The threshold $\delta=0.8$ in the three constraints of the gradient walk is fixed; adaptive tuning could potentially yield further improvements.

## Related Work & Insights
- **vs. LESS (Xia et al., 2024)**: LESS selects data independently based on gradient similarity, whereas G2IS introduces a graph structure to model the joint distribution, outperforming LESS across all setups.
- **vs. Sentence-BERT Selection**: Selection based on semantic similarity neglects training dynamics, while the gradient representation in G2IS better reflects the actual contribution to model learning.
- **vs. LIMA (Zhou et al., 2024)**: LIMA demonstrated that a small amount of high-quality data is sufficient; G2IS provides a systematic methodology for data selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of gradient graph + walk algorithm introduces a brand-new paradigm in data selection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 3 models, 3 datasets, 5 benchmarks, multi-task optimization, and comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and experimental designs are reasonable.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for instruction-tuning data selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GIST: Targeted Data Selection for Instruction Tuning with Gradient Subspace Projection](../../ICML2026/llm_alignment/gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)

</div>

<!-- RELATED:END -->
