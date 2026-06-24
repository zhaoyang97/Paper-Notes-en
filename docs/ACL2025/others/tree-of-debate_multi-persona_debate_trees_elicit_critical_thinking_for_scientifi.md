---
title: >-
  [Paper Note] Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis
description: >-
  [ACL 2025][Multi-agent debate] The Tree-of-Debate (ToD) framework is proposed, which represents scientific papers as LLM personas that engage in tree-structured debates. Through self-deliberation, iterative retrieval, and moderator-guided hierarchical subtopic expansion, ToD generates fine-grained, contextualized comparative summaries of papers, significantly outperforming baseline methods in evaluations by domain experts.
tags:
  - "ACL 2025"
  - "Multi-agent debate"
  - "paper comparative analysis"
  - "tree-structured reasoning"
  - "retrieval-augmented generation"
  - "scientific literature summarization"
date: 2026-05-08
content_hash: 5ca965cce5ab9f6d
---

# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis

**Conference**: ACL 2025  
**arXiv**: [2502.14767](https://arxiv.org/abs/2502.14767)  
**Authors**: Priyanka Kargupta, Ishika Agarwal, Tal August, Jiawei Han (UIUC)
**Code**: [GitHub](https://github.com/pkargupta/tree-of-debate)  
**Area**: Others  
**Keywords**: Multi-agent debate, paper comparative analysis, tree-structured reasoning, retrieval-augmented generation, scientific literature summarization

## TL;DR

The Tree-of-Debate (ToD) framework is proposed, which represents scientific papers as LLM personas that engage in tree-structured debates. Through self-deliberation, iterative retrieval, and moderator-guided hierarchical subtopic expansion, ToD generates fine-grained, contextualized comparative summaries of papers, significantly outperforming baseline methods in evaluations by domain experts.

## Background & Motivation

### Background
With the explosive growth of scientific literature (e.g., over 24,000 papers submitted to arXiv in October 2024 alone), researchers face a major challenge: identifying novel discoveries, incremental contributions, and equivalent ideas across related papers, particularly similar works from different research communities. Automatically generating comparative summaries of papers is highly valuable for literature reviews.

### Limitations of Prior Work
- **Existing comparative summarization methods** (such as the Two-Stage pipeline) typically adopt an "extract-then-compare" two-step strategy, but they only focus on surface-level semantic differences and fail to capture the most relevant deep distinctions.
- **Unstructured LLM generation** tends to produce extractive summaries, listing semantically similar/dissimilar phrases directly as similarities/differences without contextual explanations of "why".
- **Difficulty in handling long texts**: Providing the entire paper directly leads to key details being overwhelmed, while using only titles and abstracts yields only high-level, surface-level comparisons.
- **Lack of datasets for non-co-citing papers**: Prior work neglects comparison scenarios between papers that do not cite each other, which is extremely common in the era of literature explosion.

### Core Motivation
To design a structured multi-persona debate framework that uses a tree structure to decompose paper contributions into independent subtopics. It aims to reveal fine-grained similarities and differences through debate-style critical reasoning while ensuring factual accuracy via iterative retrieval.

## Method

### Overall Architecture
ToD consists of three core roles: two **paper personas** (representing the two papers to be compared) and one **moderator**. Given two papers $p_1, p_2$ and a root topic $n_0$, the framework dynamically constructs a debate tree $T$, where each node represents a debate round on a specific topic contribution.

### Segment-level Retrieval
- A retrieval embedding model is used to partition each paper into segments of approximately three sentences.
- The top $\delta$ most relevant segments are retrieved for each topic $n_i$ based on cosine similarity.
- Query format: "[Topic Name] : [Topic Description]"

### Persona Construction
- **Paper Persona**: Authored with the paper's title, abstract, and retrieved segments related to the current topic, with the role of arguing that its own contribution is superior to the opponent's.
- **Moderator**: Uses the same underlying model and is responsible for (i) identifying key debate subtopics, (ii) judging the progress of the debate, and (iii) synthesizing the debate tree into a comparative summary.

### Tree Node Debate Process (Three Stages)
**Stage 1: Self-Deliberation**
1. Each persona retrieves segments $S_i^a$ related to the current topic $n_i$.
2. Generates $k$ novelty claims $C_i^a$, mapping them to corresponding evidence $E_i^a \subseteq S_i^a$.
3. **Preemption**: Exposes the opponent's claims $C_i^b$ to persona $p_a$, who retrieves preemptive evidence $\widetilde{E}_i^a$ from their own paper and uses an LLM filtering step to judge whether the evidence supports, refutes, or clarifies the opponent's claim.
4. The moderator generates $k$ subtopics based on the claims and evidence from both sides.

**Stage 2: Debate**
For each subtopic node $n_j^i$, each persona sequentially performs:
1. **Present**: Argues that they outperform the opponent on this subtopic.
2. **Respond**: Responds to the opponent's arguments, raising doubts or addressing clarifications.
3. **Revise**: Revises their own arguments based on the interaction.

**Stage 3: Determine Expansion**
The moderator evaluates three conditions:
- Have the arguments progressed sufficiently or introduced new concepts?
- Are there valuable, unanswered questions?
- Is there already a clear "winner," making further decomposition unnecessary?
If either of the first two conditions is met or the third condition does not hold, a new round of self-deliberation is triggered to expand the tree further down.

### Debate Synthesis
Once all debate paths are fully expanded, the moderator synthesizes the entire debate tree into a paragraph-style comparative summary, detailing similarities first and then differences.

## Key Experimental Results

### Experimental Setup
- Base Model: Llama-3.1-Nemotron-70B-Instruct-HF (Open Source)
- Dataset: 100 paper pairs, covering four domains: NLP, Data Mining, Electrical Engineering, and Aerospace, constructed and evaluated by 5 domain experts.
- Paper pair classification: Co-citation (30 pairs) vs Non-co-citation (70 pairs); Method difference (45 pairs) vs Task difference (55 pairs).
- Evaluation metrics: Factuality (0/1 binary sentence-level), Breadth (0-4), Contextualization (0-4).

### Main Results (Table 2, Overall Performance)

| Method | Breadth↑ | Context↑ | Factuality↑ |
|------|----------|----------|-------------|
| Single Stage | 89.04 | 72.80 | 93.59 |
| Two Stage | 86.77 | 75.46 | 94.12 |
| ToD (No Tree) | 80.74 | 70.04 | 89.78 |
| ToD (No SD) | 85.99 | 91.27 | 79.99 |
| **Tree-of-Debate** | **94.92** | **95.28** | **93.87** |

ToD outperforms the best baseline by approximately 6.85% in breadth and 25.98% in contextualization, while maintaining high factuality.

### Domain-level Performance (Table 5)

| Domain | ToD Breadth | ToD Context | ToD Fact |
|------|------------|------------|----------|
| NLP | 95.08 | 95.83 | 94.78 |
| Data Mining | 91.67 | **100.00** | 90.99 |
| Electrical Engineering | 98.08 | 88.46 | 91.36 |
| Aerospace | **100.00** | 90.00 | 88.64 |

ToD demonstrates strong performance across all domains, with the largest gains in NLP and Data Mining. Factuality drops slightly in the Electrical and Aerospace domains, likely because the pre-training knowledge of the model contains relatively less information in these fields.

### Key Findings from Ablation Study
- **Without Tree Structure (No Tree)**: Contextualization and breadth decrease significantly, indicating that structured debates are crucial for fine-grained analysis.
- **Without Self-Deliberation (No SD)**: Contextualization remains relatively high (91.59%), but factuality and breadth decrease drastically, showing that iterative retrieval is key to mitigating hallucination and ensuring broad coverage.
- **Impact of Debate Depth**: As depth increases from 1 to 3, both factuality and contextualization improve significantly. A depth of 1 is prone to producing superficial, overly positive statements.

## Highlights

- **Innovative Design of Tree-Structured Debate**: Decomposing paper contributions into hierarchical subtopics for independent debating avoids the reasoning difficulties caused by entangled multiple arguments in unstructured debates, achieving a contextualization score of 95.28%.
- **Self-Deliberation + Preemptive Rebuttal Mechanism**: Before the debate, personas not only prepare their own arguments but also anticipate the opponent's claims and retrieve counter-evidence, simulating the rigorous preparation process of real academic debates.
- **Iterative Retrieval Ensuring Depth and Accuracy**: Dynamically updating the evidence pool as the debate progresses ensures that the discussion of each subtopic is backed by sufficient paper details, effectively suppressing hallucination.
- **Strong Robustness**: The standard deviation of performance across four settings (co-citing/non-co-citing, method/task differences) is only 2.49, which is much lower than the baseline methods.
- **Expert-Constructed Dataset**: Fills the gap in datasets for comparative analysis of non-co-citing papers, with 100 paper pairs spanning 4 interdisciplinary domains.

## Limitations & Future Work

- **Debate Quality Depends on the Base Model**: Factuality drops in domains where the model's pre-trained knowledge is weaker (such as Electrical Engineering and Aerospace); the quality of the "response" stage in discussions is constrained by the model's capability.
- **Deep Debates May Introduce Noise**: When evidence supporting fine-grained claims is lacking in the papers, personas might "invent" potential future research or new methods, introducing minor noise.
- **Trade-off Between Factuality and Contextualization**: In comparisons involving task differences that require larger reasoning "leaps," ToD may conflate facts between the two papers to identify connections.
- **High Computational Cost**: Multi-round tree-structured debates require intensive LLM inference calls, making the cost substantial when scaling to large-scale literature comparisons.
- **Limited to Pairwise Comparison**: The current framework is restricted to binary debates and has not been extended to group comparisons of multiple papers.
- **Limited Dataset Size**: The evaluation relies on 100 paper pairs and 5 evaluators; the small scale means cross-domain generalizability still requires larger-scale validation.

## Related Work

- **Single Stage / Two Stage Baselines**: Directly using LLMs to generate comparative summaries tends to produce extractive summaries lacking contextual explanations; ToD leads in contextualization by about 20–25 percentage points.
- **Multi-Agent Debate (Liang et al., 2024)**: Employs debates between opposing parties with a judge to mitigate confirmation bias and improve reasoning quality, where debate serves as a means rather than the end product. ToD outputs the debate itself, with a tree structure facilitating a more granular decomposition of contributions.
- **Author Persona (Portenoy et al., 2022)**: Creates personas for author recommendation based on named entity matching. ToD creates paper personas that actively debate and defend each paper's contributions.
- **ContributionSum (Liu et al., 2023) / DIR (Wang et al., 2024a)**: The former generates disentangled contribution summaries but requires fine-tuning, while the latter requires structured fine-tuning based on gold standards. ToD operates entirely at inference time without training, making it domain-agnostic.
- **Graph-Based Methods (Chen et al., 2022; Stroehle et al., 2023)**: Categorize sentences as claims, similarities, or differences and assign scores to generate extractive summaries. ToD generates abstractive, contextualized summaries.

## Rating

- Novelty: ⭐⭐⭐⭐ — The tree-structured debate framework design is novel, and the combination of self-deliberation, preemptive rebuttal, and iterative retrieval is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Domain expert evaluations, multi-dimensional metrics, ablation studies, and case analyses are comprehensive, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, effective diagrams and case presentations, and detailed description of the methodology.
- Value: ⭐⭐⭐⭐ — Provides a practical tool for comparative analysis of scientific literature, and the core concept of the framework can be extended to other complex reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)

</div>

<!-- RELATED:END -->
