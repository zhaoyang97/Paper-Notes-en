---
title: >-
  [Paper Note] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery
description: >-
  [NeurIPS 2025][single-cell RNA-seq] This work proposes the scPilot framework and scBench benchmark, enabling LLMs to perform "omics-native reasoning" (ONR) directly on single-cell RNA-seq data—reading marker genes, forming hypotheses, invoking tools for verification, and iteratively refining conclusions—achieving an 11% improvement in cell-type annotation accuracy and a 30% reduction in trajectory inference graph-edit distance.
tags:
  - NeurIPS 2025
  - single-cell RNA-seq
  - LLM reasoning
  - omics-native reasoning
  - cell-type annotation
  - trajectory inference
date: 2026-05-08
content_hash: fb7930d6219dd422
---

# scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery

**Conference**: NeurIPS 2025
**arXiv**: [2602.11609](https://arxiv.org/abs/2602.11609)
**Code**: [https://github.com/maitrix-org/scPilot](https://github.com/maitrix-org/scPilot)
**Area**: Interpretability
**Keywords**: single-cell RNA-seq, LLM reasoning, omics-native reasoning, cell-type annotation, trajectory inference

## TL;DR
This work proposes the scPilot framework and scBench benchmark, enabling LLMs to perform "omics-native reasoning" (ONR) directly on single-cell RNA-seq data—reading marker genes, forming hypotheses, invoking tools for verification, and iteratively refining conclusions—achieving an 11% improvement in cell-type annotation accuracy and a 30% reduction in trajectory inference graph-edit distance.

## Background & Motivation

**Background**: Single-cell RNA-seq analysis relies on fixed pipelines (Scanpy, Seurat), with large amounts of implicit expert reasoning (e.g., differential genes → cell-type judgment) remaining unautomated. Existing LLM applications treat LLMs merely as "code generators" that invoke pre-existing tools.

**Limitations of Prior Work**: (a) Single-cell foundation models (e.g., scGPT) embed gene expression into vector spaces, sacrificing interpretability; (b) LLM code agents only wrap tools with default parameters without performing biological reasoning; (c) the biological logic underlying the analysis process is opaque.

**Key Challenge**: Single-cell analysis demands extensive expert reasoning (identifying cell types from marker genes, inferring developmental relationships from lineage trajectories), yet existing automated tools perform computation without reasoning.

**Goal**: To enable LLMs not merely to invoke tools, but to interpret data, formulate hypotheses, gather evidence, and iteratively refine conclusions in the manner of a biologist.

**Key Insight**: Define the omics-native reasoning (ONR) paradigm—LLMs receive textual summaries of single-cell data, perform explicit reasoning, invoke tools to obtain numerical evidence, and iterate until biological conclusions are reached.

**Core Idea**: Single-cell analysis is formalized as a natural language reasoning problem, where LLMs produce (claim, action) pairs at each step, constituting a dual-track "verbal + computational" proof.

## Method

### Overall Architecture
Three core components: (1) **Problem-to-Text Converter** $\mathcal{C}$: compresses expression matrices of $10^5$–$10^6$ cells into LLM-digestible textual summaries (e.g., cluster sizes, top-$k$ marker genes); (2) **Bio-Tool Library** $\mathcal{T}$: encapsulates Scanpy, Monocle, pySCENIC, and other tools as callable structured APIs; (3) **LLM Reasoner** $\mathcal{R}_\phi$: uses reasoning LLMs such as o1/Gemini as the core, executing closed-loop reasoning $\mathbf{X} \to \text{Prompt} \to \{(\text{Thought}_k, \text{Call}_k)\}_{k=1}^K \to \hat{y}$.

### Key Designs

1. **Omics-Native Reasoning (ONR) Formalization**:

    - Function: Defines bioinformatics analysis tasks as a reasoning sequence $\mathcal{R} = [(c_1,o_1), \ldots, (c_K,o_K)]$
    - Mechanism: At each step, the LLM produces a natural language claim $c_k$ (e.g., "cluster 5 highly expresses CD3D and CD3E, likely T cells") and an action $o_k$ (e.g., "check NK cell marker genes"), with each action updating the data state $S_k = o_k(S_{k-1})$
    - Design Motivation: The key distinction from code agents—the reasoning trace constitutes an auditable biological argument, not merely code and output

2. **Problem-to-Text Compression**:

    - Function: Compresses million-scale cell matrices into text processable within an LLM context window
    - Mechanism: Task-specific compression strategies are designed: Leiden clustering + top-10 marker genes for cell annotation; PAGA graph + pseudotime for trajectory inference; top-150 TF–gene pairs for GRN
    - Design Motivation: Preserving biologically significant information while drastically reducing dimensionality, enabling LLMs to operate in the text domain

3. **scBench Benchmark**:

    - Function: Covers nine datasets across three major tasks (cell-type annotation, trajectory inference, gene regulatory network prediction)
    - Mechanism: Each task includes expert-verified ground truth and automated evaluation metrics (accuracy, graph-edit distance, AUROC)
    - Design Motivation: Existing single-cell benchmarks evaluate only embedding quality or numerical metrics, without assessing the biological validity of reasoning

### Loss & Training
scPilot is a training-free framework that does not fine-tune LLMs. All reasoning capabilities derive from prompt engineering and iterative reasoning strategies. Core design principles: (a) biological context priority; (b) iterative reasoning; (c) minimal hand-crafted heuristics.

## Key Experimental Results

### Main Results

| Task | Dataset | scPilot (o1) | Best Baseline | Gain |
|------|---------|-------------|--------------|------|
| Cell Annotation | PBMC3k | ~0.76 | CellTypist 0.563 | +35% |
| Cell Annotation | Liver | ~0.50 | CellTypist 0.464 | +8% |
| Cell Annotation | Retina | ~0.49 | CellTypist 0.388 | +26% |
| Trajectory Inference | Pancreas | GED reduced 30% | Traditional pipeline | Gemini-2.5-Pro best |
| GRN Prediction | Multi-organ | AUROC +0.03 | pySCENIC direct output | Iterative reasoning gain |

### Ablation Study

| Configuration | Effect | Note |
|--------------|--------|------|
| Direct prompting (no iteration) | Baseline | Single-pass reasoning |
| Iterative reasoning (2–3 rounds) | +11% avg accuracy | Iterative hypothesis refinement |
| Without biological context | Significant drop | Species/tissue information is critical |
| LLM comparison | o1 best for annotation, Gemini best for trajectory | Task-specific LLM capability |

### Key Findings
- Iterative reasoning is critical—LLMs frequently err in the first round (e.g., confusing NK and T cells) but self-correct in the second round upon examining additional marker genes
- LLMs can identify potential issues in expert annotations—in some cases, scPilot's reasoning is more consistent than the original labels
- Different LLMs excel at different tasks: o1's strong reasoning suits annotation, while Gemini's large context window suits trajectory inference
- Reasoning traces are highly interpretable—biologists can audit the logic at every step

## Highlights & Insights
- **Paradigm Shift**: From "LLM invoking tools" to "LLM performing biological reasoning." scPilot automates the expert thought process, not merely the pipeline
- **Scientific Value of Reasoning Traces**: Generated traces expose marker gene ambiguity, tissue-specific expression patterns, and other insights of independent analytical value to biologists
- **Generalizable ONR Framework**: The same "data → textual summary → LLM reasoning → tool verification" paradigm is transferable to proteomics, metabolomics, and other omics domains

## Limitations & Future Work
- Performance depends on the quality of Problem-to-Text compression—information loss may introduce reasoning bias
- Current coverage is limited to three core tasks; spatial transcriptomics, multi-omics integration, and related areas are not addressed
- Full reliance on the LLM's biological knowledge may be insufficient for very novel or rare cell types
- Multi-round LLM reasoning per analysis incurs substantial computational cost (o1 API expenses)

## Related Work & Insights
- **vs. CellAgent**: CellAgent instructs LLMs to write code invoking Scanpy but performs no biological reasoning; scPilot interprets differential genes and formulates hypotheses
- **vs. scGPT/Geneformer**: These models operate in vector space and produce no natural language reasoning; scPilot's traces constitute auditable arguments
- **vs. GPTCellType**: Addresses only cell annotation via direct prompting for a single task; scPilot covers multiple tasks with iterative reasoning

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The formal definition and systematic framework of "omics-native reasoning" is entirely novel, opening a new paradigm for LLMs in computational biology
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine datasets, three tasks, and multiple LLMs and baselines, though gains on the GRN task are modest
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, though the mathematical formalization is somewhat over-notated
- Value: ⭐⭐⭐⭐⭐ Potentially transformative for the computational biology community—demonstrating the feasibility of LLMs as "scientific reasoning partners" rather than "code generators"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)

</div>

<!-- RELATED:END -->
