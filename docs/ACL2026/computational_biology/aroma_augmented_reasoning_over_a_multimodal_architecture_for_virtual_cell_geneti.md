---
title: >-
  [Paper Note] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling
description: >-
  [ACL 2026][Computational Biology][Virtual Cell Modeling] The AROMA framework is proposed, which achieves interpretable and precise genetic perturbation effect prediction through a multimodal architecture integrating text…
tags:
  - "ACL 2026"
  - "Computational Biology"
  - "Virtual Cell Modeling"
  - "Genetic Perturbation Prediction"
  - "Multimodal Fusion"
  - "Knowledge Graph"
  - "Reinforcement Learning Reasoning"
date: 2026-05-08
content_hash: e7a055c64fd4611e
---

# AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling

**Conference**: ACL 2026  
**arXiv**: [2604.20263](https://arxiv.org/abs/2604.20263)  
**Code**: [github](https://github.com/blazerye/AROMA)  
**Area**: Medical Imaging / Bioinformatics  
**Keywords**: Virtual Cell Modeling, Genetic Perturbation Prediction, Multimodal Fusion, Knowledge Graph, Reinforcement Learning Reasoning

## TL;DR

The AROMA framework is proposed, which achieves interpretable and precise genetic perturbation effect prediction through a multimodal architecture integrating textual evidence, knowledge graph (KG) topological information, and protein sequence features, combined with a two-stage training strategy (SFT + GRPO).

## Background & Motivation

**Background**: Virtual cell modeling aims to predict molecular state changes following genetic perturbations, which is crucial for biological mechanism research. Existing methods include general LLMs, domain-fine-tuned language models, cell foundation models, and retrieval-augmented methods.

**Limitations of Prior Work**: (1) General LLMs lack biological constraints, making free-form reasoning unreliable; (2) Existing foundation models only output labels or differential expression scores, lacking human-interpretable reasoning processes; (3) Retrieval signals in current RAG methods are weakly aligned with regulatory topology and fail to model regulatory directionality and multi-step propagation.

**Key Challenge**: Genetic perturbation effects are highly context-dependent and propagate through multi-step regulatory cascades. Simple text similarity retrieval cannot capture mechanistic paths from perturbed genes to target genes.

**Goal**: To build a genetic perturbation prediction framework that provides both accurate predictions and interpretable reasoning.

**Key Insight**: Anchor perturbation prediction on structured, query-specific biological evidence to explicitly model the dependencies between perturbed and target genes.

**Core Idea**: Combine KG retrieval (topology evidence), Graph Neural Network encoders (structural representation), and protein sequence encoders (molecular representation). Model perturbation-target relationships through a cross-modal interaction attention mechanism, followed by two-stage training to optimize prediction and reasoning quality.

## Method

### Overall Architecture

AROMA consists of three stages: (1) Data stage—constructing Gene-KG, Path-KG, and the PerturbReason reasoning dataset; (2) Modeling stage—retrieval-augmented contextualization + multimodal interaction encoding; (3) Training stage—multimodal SFT + GRPO reinforcement learning.

### Key Designs

1.  **Dual Knowledge Graph Construction and Retrieval-Augmented Contextualization**:
    - **Function**: Provide structured biological evidence for perturbation prediction.
    - **Mechanism**: Gene-KG integrates STRING and CORUM to build gene-level association networks (18k nodes, 700k edges); Path-KG integrates GO and Reactome to encode biological process structures (80k nodes, 400k edges). Retrieval includes gene function descriptions, BFS shortest paths ($\le 3$ steps), and cell line descriptions.
    - **Design Motivation**: The two graphs are complementary—Gene-KG provides direct associations between genes, while Path-KG provides higher-level pathway structures, together covering multi-level evidence.

2.  **Multimodal Interaction Encoding Module**:
    - **Function**: Explicitly model the cross-modal interaction between perturbed and target genes.
    - **Mechanism**: Two pre-trained GAT encoders encode Gene-KG and Path-KG subgraphs respectively; ESM-2 is frozen to encode protein sequences. For each modality, cross-attention is used (perturbed gene as Query, target gene as Key/Value), with signals injected into the LLM input via lightweight projectors.
    - **Design Motivation**: Unlike text-only methods, structural and molecular representations enrich the modeling of perturbation-target relationships.

3.  **Two-Stage Optimization Strategy (SFT + GRPO)**:
    - **Function**: First align multimodal information, then optimize reasoning quality.
    - **Mechanism**: The first stage performs multimodal SFT on PerturbReason (498k+ samples), freezing GNN and ESM-2 while using LoRA to fine-tune the LLM. The second stage uses Group Relative Policy Optimization (GRPO) to sample multiple reasoning trajectories, rewarding correct predictions (+5.0) and standard format adherence (+0.5).
    - **Design Motivation**: SFT injects domain knowledge, while GRPO further optimizes the accuracy and consistency of the reasoning process through task-level feedback.

### Loss & Training

The SFT stage uses standard autoregressive language modeling loss. The GRPO stage samples multiple reasoning trajectories per instance. The reward function considers: prediction correctness (5.0/-1.0), reasoning format compliance (+0.5), and answer category uniqueness (+0.5). Advantages are calculated using intra-group normalization.

## Key Experimental Results

### Main Results

| Method | K562 Avg | HepG2 Avg | Jurkat Avg | RPE1 Avg | Total Avg F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-R1 | 0.32 | 0.34 | 0.33 | 0.31 | 0.33 |
| SUMMER | 0.58 | 0.67 | 0.65 | 0.67 | 0.64 |
| GAT | 0.59 | 0.67 | 0.63 | 0.65 | 0.64 |
| **AROMA** | **0.66** | **0.76** | **0.75** | **0.77** | **0.73** |

### Ablation Study

| Configuration | Avg F1 | Description |
| :--- | :--- | :--- |
| Vanilla Qwen3-8B | 0.26 | Lacks domain knowledge |
| + SFT | 0.65 | Domain knowledge injection is key |
| + SFT + GRPO | 0.68 | RL improves reasoning |
| + RAG | 0.71 | Retrieval evidence complement |
| Full Module (AROMA) | 0.73 | Synergistic gains from all components |

### Key Findings
- AROMA consistently outperforms all baseline methods across all 4 cell lines, achieving an average F1 of 0.73, which is 9 percentage points higher than the strongest baseline, SUMMER.
- Zero-shot generalization (RPE1) performance shows only a slight decrease (0.77 $\rightarrow$ 0.73), demonstrating strong cross-distribution generalization capabilities.
- The performance drop on low-popularity and low-connectivity genes is significantly smaller than that of variants without retrieval and multimodal modules, indicating that gains stem from joint modeling rather than memorizing high-frequency genes.
- Performance improves steadily as the number of GRPO sampled trajectories increases from 4 to 16.

## Highlights & Insights
- This is the first work to systematically integrate KG topology, protein sequences, and textual evidence for genetic perturbation prediction.
- The dual KG design is noteworthy: Gene-KG provides local associations, while Path-KG provides global pathway structures.
- The design intuition of the cross-attention mechanism—using the perturbed gene as Query and the target gene as Key/Value—is clear and effective.
- The constructed PerturbReason dataset (498k samples) is a significant resource contribution to the community.

## Limitations & Future Work
- Currently supports only single-gene perturbations; cannot handle multi-gene combinations or chemical interventions.
- Each inference only predicts expression changes for a single target gene, rather than simultaneously predicting multiple downstream genes.
- Dependence on KGs and external text resources implies that prediction quality may degrade for genes with sparse annotations.
- Future work could extend the framework to combinatorial perturbations and chemical intervention scenarios.

## Related Work & Insights
- **vs SUMMER**: SUMMER uses text similarity retrieval; AROMA further introduces topological structure and protein sequences to model perturbation-target interactions.
- **vs GEARS**: GEARS injects graph structure priors but lacks interpretable reasoning; AROMA provides both predictions and reasoning paths.
- **vs SynthPert/rBio-1**: These rely on synthetic reasoning trajectories for training, which may inherit supervisory noise; AROMA optimizes directly from task feedback via GRPO.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Multimodal fusion approach is clear; dual KG and interaction encoding designs are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive analysis across multiple cell lines, zero-shot settings, ablations, and robustness.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-designed illustrations.
- **Value**: ⭐⭐⭐⭐ Significant advancement for the virtual cell modeling field with high resource contribution value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](../../ICLR2026/computational_biology/retrieval-augmented_generation_for_predicting_cellular_responses_to_gene_perturb.md)
- [\[ICLR 2026\] VCWorld: A Biological World Model for Virtual Cell Simulation](../../ICLR2026/computational_biology/vcworld_a_biological_world_model_for_virtual_cell_simulation.md)
- [\[ICML 2026\] What Makes a Representation Good for Single-Cell Perturbation Prediction?](../../ICML2026/computational_biology/what_makes_a_representation_good_for_single-cell_perturbation_prediction.md)
- [\[ICLR 2026\] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](../../ICLR2026/computational_biology/scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)
- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)

</div>

<!-- RELATED:END -->
