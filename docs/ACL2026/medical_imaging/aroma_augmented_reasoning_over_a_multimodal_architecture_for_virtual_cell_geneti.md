---
title: >-
  [Paper Note] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling
description: >-
  [ACL 2026][Medical Imaging][virtual cell modeling] This paper proposes the AROMA framework, which integrates textual evidence, knowledge graph topological information, and protein sequence features within a multimodal architecture, combined with a two-stage training strategy (SFT + GRPO), to achieve interpretable and accurate prediction of genetic perturbation effects.
tags:
  - ACL 2026
  - Medical Imaging
  - virtual cell modeling
  - genetic perturbation prediction
  - multimodal fusion
  - knowledge graph
  - reinforcement learning reasoning
date: 2026-05-08
content_hash: ef9a0c3a32f2cddb
---

# AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling

**Conference**: ACL 2026
**arXiv**: [2604.20263](https://arxiv.org/abs/2604.20263)
**Code**: [github](https://github.com/blazerye/AROMA)
**Area**: Medical Imaging / Bioinformatics
**Keywords**: virtual cell modeling, genetic perturbation prediction, multimodal fusion, knowledge graph, reinforcement learning reasoning

## TL;DR

This paper proposes the AROMA framework, which integrates textual evidence, knowledge graph topological information, and protein sequence features within a multimodal architecture, combined with a two-stage training strategy (SFT + GRPO), to achieve interpretable and accurate prediction of genetic perturbation effects.

## Background & Motivation

**Background**: Virtual cell modeling aims to predict molecular state changes following genetic perturbations, which is critical for understanding biological mechanisms. Existing approaches include general-purpose LLMs, domain-fine-tuned language models, cell foundation models, and retrieval-augmented methods.

**Limitations of Prior Work**: (1) General-purpose LLMs lack biological constraints, making free-form reasoning unreliable; (2) existing foundation models only output labels or differential expression scores, lacking human-interpretable reasoning processes; (3) retrieval-augmented methods suffer from weak alignment between retrieval signals and regulatory topology, failing to model regulatory directionality and multi-step propagation.

**Key Challenge**: Genetic perturbation effects are highly context-dependent and propagate through multi-step regulatory cascades; pure text-similarity-based retrieval cannot capture mechanistic pathways from perturbed genes to target genes.

**Goal**: To construct a genetic perturbation prediction framework that simultaneously achieves accurate prediction and provides interpretable reasoning.

**Key Insight**: Anchor perturbation predictions in structured, query-specific biological evidence, and explicitly model the dependency relationships between perturbed genes and target genes.

**Core Idea**: Combine knowledge graph retrieval (for topological evidence), graph neural network encoders (structural representations), and protein sequence encoders (molecular representations); model perturbation–target relationships via cross-modal interactive attention; then apply a two-stage training strategy to optimize both prediction accuracy and reasoning quality.

## Method

### Overall Architecture

AROMA comprises three stages: (1) Data stage — constructing Gene-KG, Path-KG, and the PerturbReason reasoning dataset; (2) Modeling stage — retrieval-augmented contextualization and multimodal interactive encoding; (3) Training stage — multimodal SFT followed by GRPO reinforcement learning.

### Key Designs

1. **Dual Knowledge Graph Construction and Retrieval-Augmented Contextualization**:

    - Function: Provides structured biological evidence for perturbation prediction.
    - Mechanism: Gene-KG integrates STRING and CORUM to construct a gene-level association network (18k nodes, 700k edges); Path-KG integrates GO and Reactome to encode biological process structures (80k nodes, 400k edges). Retrieval includes gene functional descriptions, BFS shortest paths (≤3 hops), and cell line descriptions.
    - Design Motivation: The two graphs are complementary — Gene-KG captures direct gene–gene associations, while Path-KG provides higher-level pathway structures, jointly covering multi-granularity evidence.

2. **Multimodal Interactive Encoding Module**:

    - Function: Explicitly models cross-modal interactions between perturbed and target genes.
    - Mechanism: Two pretrained GAT encoders separately encode Gene-KG and Path-KG subgraphs; frozen ESM-2 encodes protein sequences. Cross-attention is applied for each modality (perturbed gene as Query, target gene as Key/Value), and representations are injected into the language model input via lightweight projectors.
    - Design Motivation: Unlike text-only approaches, structural and molecular representations enrich the modeling of perturbation–target relationships.

3. **Two-Stage Optimization Strategy (SFT + GRPO)**:

    - Function: First aligns multimodal information, then optimizes reasoning quality.
    - Mechanism: Stage one performs multimodal SFT on PerturbReason (498k+ samples), freezing GNN and ESM-2 while fine-tuning the LLM with LoRA. Stage two applies GRPO reinforcement learning by sampling multiple reasoning trajectories and rewarding correct predictions (+5.0) and format compliance (+0.5).
    - Design Motivation: SFT injects domain knowledge, while GRPO further refines the accuracy and consistency of the reasoning process through task-level feedback.

### Loss & Training

The SFT stage uses the standard autoregressive language modeling loss. In the GRPO stage, multiple reasoning trajectories are sampled per instance; the reward function considers prediction correctness (+5.0/−1.0), reasoning format compliance (+0.5), and answer class uniqueness (+0.5), with advantage values computed via within-group normalization.

## Key Experimental Results

### Main Results

| Method | K562 Avg | HepG2 Avg | Jurkat Avg | RPE1 Avg | Overall Avg F1 |
|--------|----------|-----------|------------|----------|----------------|
| DeepSeek-R1 | 0.32 | 0.34 | 0.33 | 0.31 | 0.33 |
| SUMMER | 0.58 | 0.67 | 0.65 | 0.67 | 0.64 |
| GAT | 0.59 | 0.67 | 0.63 | 0.65 | 0.64 |
| AROMA | **0.66** | **0.76** | **0.75** | **0.77** | **0.73** |

### Ablation Study

| Configuration | Avg F1 | Notes |
|---------------|--------|-------|
| Vanilla Qwen3-8B | 0.26 | Lacks domain knowledge |
| + SFT | 0.65 | Domain knowledge injection is critical |
| + SFT + GRPO | 0.68 | Reinforcement learning improves reasoning |
| + RAG | 0.71 | Retrieved evidence provides further gains |
| Full model (AROMA) | 0.73 | Synergistic gains from all components |

### Key Findings
- AROMA consistently outperforms all baselines across all 4 cell lines, achieving an average F1 of 0.73, surpassing the strongest baseline SUMMER by 9 percentage points.
- Zero-shot generalization performance (RPE1) degrades only slightly (0.77 → 0.73), demonstrating strong cross-distribution generalization.
- Performance degradation on low-prevalence and low-connectivity genes is substantially smaller than in variants without retrieval and multimodal modules, indicating that gains stem from joint modeling rather than memorization of high-frequency genes.
- Performance improves steadily as the number of GRPO sampled trajectories increases from 4 to 16.

## Highlights & Insights
- This work is the first to systematically integrate three modalities — knowledge graph topology, protein sequences, and textual evidence — for genetic perturbation prediction.
- The dual knowledge graph design is noteworthy: Gene-KG provides local gene–gene associations, while Path-KG provides global pathway structures.
- The cross-attention design using the perturbed gene as Query and the target gene as Key/Value is intuitive and mechanistically motivated.
- The PerturbReason dataset (498k samples) constitutes a significant community resource contribution.

## Limitations & Future Work
- The framework currently supports only single-gene perturbations and cannot handle combinatorial multi-gene perturbations or chemical interventions.
- Each inference step predicts expression changes for a single target gene, without extending to simultaneous prediction of multiple downstream genes.
- Dependence on knowledge graphs and external textual resources implies potential performance degradation for poorly annotated genes.
- Future work may extend the framework to combinatorial perturbation and chemical intervention scenarios.

## Related Work & Insights
- **vs. SUMMER**: SUMMER relies on text-similarity-based retrieval, whereas AROMA additionally incorporates topological structure and protein sequence representations to model perturbation–target interactions.
- **vs. GEARS**: GEARS incorporates graph-structural priors but lacks interpretable reasoning; AROMA provides both predictions and reasoning pathways.
- **vs. SynthPert/rBio-1**: These methods rely on synthetic reasoning trajectories for training, potentially inheriting supervision noise; AROMA directly optimizes from task-level feedback via GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐ The multimodal fusion approach is well-motivated; the dual knowledge graph and interactive encoding designs are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple cell lines, zero-shot generalization, ablation studies, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with high-quality figures.
- Value: ⭐⭐⭐⭐ Significant contribution to the virtual cell modeling field with high resource value.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[NeurIPS 2025\] MuSLR: Multimodal Symbolic Logical Reasoning](../../NeurIPS2025/medical_imaging/muslr_multimodal_symbolic_logical_reasoning.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](../../NeurIPS2025/medical_imaging/multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)

<!-- RELATED:END -->
