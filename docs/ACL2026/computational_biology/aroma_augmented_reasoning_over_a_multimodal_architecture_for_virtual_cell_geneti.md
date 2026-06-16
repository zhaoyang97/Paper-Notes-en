---
title: >-
  [Paper Note] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling
description: >-
  [ACL 2026][Computational Biology][Paper Note] Ours proposes the AROMA framework, a multimodal architecture that integrates textual evidence, knowledge graph (KG) topological information, and protein sequence features. Combined with a two-stage training strategy (SFT + GRPO), it achieves interpretable and precise prediction of genetic perturbation effects.
tags:
  - ACL 2026
  - Computational Biology
date: 2026-05-08
content_hash: d424e475c01dca47
---
# AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.20263](https://arxiv.org/abs/2604.20263)  
**Code**: [github](https://github.com/blazerye/AROMA)  
**Area**: Medical Imaging / Bioinformatics  
**Keywords**: Virtual cell modeling, gene perturbation prediction, multimodal fusion, knowledge graph, RL reasoning

## TL;DR

Ours proposes the AROMA framework, a multimodal architecture that integrates textual evidence, knowledge graph (KG) topological information, and protein sequence features. Combined with a two-stage training strategy (SFT + GRPO), it achieves interpretable and precise prediction of genetic perturbation effects.

## Background & Motivation

**Background**: Virtual cell modeling aims to predict changes in molecular states following genetic perturbations, which is crucial for studying biological mechanisms. Existing methods include general LLMs, domain-specific fine-tuned language models, cellular foundation models, and retrieval-augmented methods.

**Limitations of Prior Work**: (1) General LLMs lack biological constraints, making free-form reasoning unreliable; (2) Existing foundation models only output labels or differential expression scores, lacking human-interpretable reasoning processes; (3) Retrieval signals in retrieval-augmented methods are weakly aligned with regulatory topology and fail to model regulatory directionality and multi-step propagation.

**Key Challenge**: Genetic perturbation effects are highly context-dependent and propagate through multi-step regulatory cascades. Simple text-based similarity retrieval cannot capture the mechanistic paths from the perturbed gene to the target gene.

**Goal**: To build a genetic perturbation prediction framework that provides both accurate predictions and interpretable reasoning.

**Key Insight**: Anchor perturbation prediction on structured, query-specific biological evidence to explicitly model the dependencies between perturbed and target genes.

**Core Idea**: Combine KG retrieval (topological evidence), Graph Neural Network encoders (structural representation), and protein sequence encoders (molecular representation). Model the perturbation-target relationship via a cross-modal interaction attention mechanism and optimize prediction and reasoning quality using a two-stage training approach.

## Method

### Overall Architecture

The mechanism of AROMA is to transform the task of "predicting target gene changes under specific perturbations" from simple text matching into reasoning anchored in structured biological evidence. The pipeline consists of: inputting a (perturbed gene, target gene) pair; retrieving gene functional descriptions, shortest regulatory paths, and cell line contexts from two KGs; using a multimodal encoder to feed KG topology, protein sequence features, and text into a language model; and finally, having the LLM output a readable reasoning process along with a directional prediction (upregulated/downregulated/no change). Training is conducted in two stages: large-scale Supervised Fine-Tuning (SFT) for domain knowledge injection, followed by Group Relative Policy Optimization (GRPO) reinforcement learning to refine reasoning quality.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: (Perturbed Gene, Target Gene)"] --> R
    subgraph R["Dual-KG Retrieval"]
        direction TB
        R1["Gene-KG<br/>STRING+CORUM, direct gene-level associations"]
        R2["Path-KG<br/>GO+Reactome, pathway-level propagation backbone"]
    end
    R --> C["Retrieved Evidence<br/>Functional descriptions + Shortest regulatory paths + Cell line context"]
    C --> E
    subgraph E["Multimodal Interaction Encoding"]
        direction TB
        E1["Dual GAT encoding KG subgraphs"]
        E2["Frozen ESM-2 encoding protein sequences"]
        E1 & E2 --> E3["Cross-Attention<br/>Perturbed Gene = Query, Target Gene = Key/Value"]
        E3 --> E4["Projector injection into LLM input"]
    end
    E --> L["Language Model"]
    T["Two-stage Optimization<br/>SFT Knowledge Injection → GRPO Reasoning Refinement"] -.Training.-> L
    L --> O["Reasoning Process + Directional Prediction<br/>Up/Down/No Change"]
```

### Key Designs

**1. Dual-KG Retrieval: Capturing "Perturbation Propagation" with Complementary Structures**

Simple text similarity retrieval fails to grasp mechanistic pathways between perturbed and target genes. AROMA constructs two complementary graphs: Gene-KG (integrating STRING and CORUM, 18k nodes, 700k edges) for direct associations, and Path-KG (integrating GO and Reactome, 80k nodes, 400k edges) for high-level biological processes. Retrieval simultaneously extracts functional descriptions, shortest paths (up to 3) calculated via BFS, and cell line descriptions. This dual-layer approach provides both direct edges and propagation backbones.

**2. Multimodal Interaction Encoding: Explicitly Modeling Perturbation-Target Relationships**

To capture precise relationships, AROMA uses two pre-trained GAT encoders for KG subgraphs and a frozen ESM-2 for protein sequences. For each modality, cross-attention is applied where the perturbed gene acts as the Query and the target gene acts as the Key/Value. A lightweight projector then injects these fused representations into the LLM. This asymmetric design reflects the biological directionality where the perturbation drives the target change.

**3. Two-stage Optimization (SFT + GRPO): Knowledge Injection followed by Reasoning Refinement**

The first stage performs multimodal SFT on PerturbReason (498k+ samples) with frozen GNN/ESM-2 and LoRA on the LLM to inject domain knowledge. The second stage uses GRPO reinforcement learning, sampling multiple trajectories per instance and scoring them based on task-level feedback. This separates "learning the answer" (SFT) from "optimizing the reasoning logic" (GRPO), reducing inconsistencies.

### Loss & Training

The SFT stage uses standard autoregressive loss. In the GRPO stage, multiple trajectories are sampled for each instance. The reward consists of three parts: prediction correctness (correct $+5.0$ / incorrect $-1.0$), format compliance ($+0.5$), and answer uniqueness ($+0.5$). Advantages are computed via intra-group normalization for policy updates.

## Key Experimental Results

### Main Results

| Method | K562 Avg | HepG2 Avg | Jurkat Avg | RPE1 Avg | Overall Avg F1 |
|------|---------|---------|---------|---------|----------|
| DeepSeek-R1 | 0.32 | 0.34 | 0.33 | 0.31 | 0.33 |
| SUMMER | 0.58 | 0.67 | 0.65 | 0.67 | 0.64 |
| GAT | 0.59 | 0.67 | 0.63 | 0.65 | 0.64 |
| **Ours (AROMA)** | **0.66** | **0.76** | **0.75** | **0.77** | **0.73** |

### Ablation Study

| Configuration | Avg F1 | Note |
|------|---------|------|
| Original Qwen3-8B | 0.26 | Lacks domain knowledge |
| + SFT | 0.65 | Domain knowledge injection is key |
| + SFT + GRPO | 0.68 | RL improves reasoning |
| + RAG | 0.71 | Retrieval evidence supplementation |
| Full Model (AROMA) | 0.73 | Synergistic gains from all components |

### Key Findings
- AROMA consistently outperforms all baselines across 4 cell lines, with an average F1 of 0.73, exceeding the strongest baseline (SUMMER) by 9 percentage points.
- Zero-shot generalization (on RPE1) shows only a slight performance drop, demonstrating strong cross-distribution robustness.
- Performance on genes with low popularity or connectivity drops significantly less than variants without retrieval/multimodal modules, indicating gains come from joint modeling rather than memorizing high-frequency genes.
- Performance improves steadily as the number of GRPO sampled trajectories increases from 4 to 16.

## Highlights & Insights
- Systematically integrates KG topology, protein sequences, and textual evidence for the first time in gene perturbation prediction.
- The Dual-KG design (Gene-KG for local associations, Path-KG for global pathways) provides a robust structural prior.
- The cross-attention mechanism using the perturbed gene as the Query provides a clear biological intuition.
- The construction of the PerturbReason dataset (498k samples) serves as a significant resource contribution to the community.

## Limitations & Future Work
- Currently only supports single-gene perturbations; cannot handle combinatorial perturbations or chemical interventions.
- Each inference predicts the expression change of only one target gene, rather than multiple downstream genes simultaneously.
- Dependency on KGs and external text resources implies potential performance degradation for genes lacking annotations.
- Future work could extend the framework to combinatorial perturbations and chemical intervention scenarios.

## Related Work & Insights
- **vs SUMMER**: SUMMER uses text similarity retrieval; AROMA introduces topological structures and protein sequence modeling for perturbation-target interactions.
- **vs GEARS**: GEARS injects graph priors but lacks interpretable reasoning; AROMA providing both predictions and reasoning paths.
- **vs SynthPert/rBio-1**: These rely on synthetic reasoning trajectories for training, which may inherit supervisory noise; AROMA optimizes directly from task feedback via GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear multimodal fusion and innovative Dual-KG/interaction design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across cell lines, zero-shot scenarios, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with high-quality illustrations.
- Value: ⭐⭐⭐⭐ Significant advancement for virtual cell modeling with valuable resource contributions.

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
