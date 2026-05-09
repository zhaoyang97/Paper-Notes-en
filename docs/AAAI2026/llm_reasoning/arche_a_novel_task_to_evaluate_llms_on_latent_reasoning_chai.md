---
title: >-
  [Paper Note] ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction
description: >-
  [AAAI 2026][LLM Reasoning][Latent reasoning chain extraction] This paper proposes the Latent Reasoning Chain Extraction (ARCHE) task, which requires LLMs to decompose scientific paper argumentation into Reasoning Logic Trees (RLTs) grounded in Peirce's three reasoning paradigms. Through two complementary metrics—Entity Coverage (EC) and Reasoning Edge Accuracy (REA)—the study reveals a fundamental trade-off between content completeness and logical correctness across 10 mainstream LLMs.
tags:
  - AAAI 2026
  - LLM Reasoning
  - Latent reasoning chain extraction
  - Peircean reasoning paradigms
  - deduction/induction/abduction
  - reasoning logic tree
  - benchmark
date: 2026-05-08
content_hash: 991a07ead08033f0
---

# ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction

**Conference**: AAAI 2026
**arXiv**: [2511.12485](https://arxiv.org/abs/2511.12485)
**Code**: [GitHub](https://github.com/Linsonng/ARCHEBenchmark/)
**Area**: LLM Evaluation / Scientific Reasoning
**Keywords**: Latent reasoning chain extraction, Peircean reasoning paradigms, deduction/induction/abduction, reasoning logic tree, benchmark

## TL;DR

This paper proposes the Latent Reasoning Chain Extraction (ARCHE) task, which requires LLMs to decompose scientific paper argumentation into Reasoning Logic Trees (RLTs) grounded in Peirce's three reasoning paradigms. Through two complementary metrics—Entity Coverage (EC) and Reasoning Edge Accuracy (REA)—the study reveals a fundamental trade-off between content completeness and logical correctness across 10 mainstream LLMs.

## Background & Motivation

**State of the Field**: LLMs are widely applied in scientific domains—literature review, hypothesis generation, experimental design, etc.—and prompting methods such as CoT can produce "reasoning-like" outputs.

**Limitations of Prior Work**: Reasoning chains produced by CoT are unstructured natural-language narratives lacking formal logical foundations, making it impossible to verify whether a model genuinely understands reasoning paradigms. Most existing benchmarks focus solely on final answer correctness, and deduction, induction, and abduction are evaluated in isolation.

**Root Cause**: Fluent reasoning at the linguistic level $\neq$ structured reasoning at the paradigmatic level. Models can "sound like they are reasoning" without truly mastering the fundamental logical forms of reasoning.

**Paper Goals**: To evaluate whether LLMs can (i) identify three reasoning paradigms within scientific arguments, (ii) assemble them into a coherent reasoning chain, and (iii) anchor each reasoning step to verifiable textual evidence.

**Starting Point**: Drawing from Peircean philosophy, the paper defines a unified framework that integrates deduction, induction, and abduction into a single Reasoning Logic Tree, using real scientific papers as the test medium.

**Core Idea**: RLTs make the latent reasoning chains embedded in scientific text explicit and structured, thereby examining whether LLMs truly understand reasoning paradigms.

## Method

### Overall Architecture

A three-stage pipeline: data processing → RLT generation → evaluation.

- **Input**: The introduction section of a scientific paper, along with viewpoints extracted from the introduction itself and from the abstracts of cited references.
- **Output**: A Reasoning Logic Tree (RLT) represented as a directed acyclic graph (DAG) in DOT graph description language.
- **Evaluation**: Two complementary metrics: EC + REA.

### Key Designs

**Reasoning Logic Tree (RLT) Structure**:
- **Nodes**: Each node contains a viewpoint and its source coordinates $(x, y, z)$, corresponding to three levels: introduction sentence, intra-sentence viewpoint, and cited-reference viewpoint.
- **Edges**: Six types of directed labeled edges, representing fine-grained instantiations of Peirce's three reasoning paradigms:
    - Deduction: Deduction-Rule (DR) and Deduction-Case (DC)
    - Induction: Induction-Common (ICo) and Induction-Case (ICa)
    - Abduction: Abduction-Knowledge (AK) and Abduction-Phenomenon (AP)
- **Constraints**: Single-root DAG; all nodes must be connected to the root; each reasoning step must correspond to exactly one reasoning paradigm and one pair of edge types.
- **Design Motivation**: Tree structures capture branching, convergence, and multi-hop relations in scientific argumentation more effectively than linear CoT; the six edge types operationalize reasoning paradigms as verifiable structural constraints.

**RLT Generation Pipeline** (two stages):
1. Initial extraction: The LLM generates an initial RLT (in DOT format) from the introduction and viewpoints via prompting.
2. Structure repair: An automated validation script checks for multiple roots, cycles, isolated nodes, and illegal labels; if defects are found, the LLM is re-prompted to correct them.

**Evaluation Metrics**:
- **Entity Coverage (EC)**: Core scientific entities are extracted from the paper using the o3 model (typically 8–10 entities); EC measures the proportion of entities covered by valid reasoning steps in the RLT.
- **Reasoning Edge Accuracy (REA)**: A three-model jury (o3 + Claude-Sonnet-4 + Gemini-2.5-Pro) judges the logical validity of each reasoning step by majority vote, achieving an accuracy exceeding 88%.

### Loss & Training

This work presents an evaluation framework rather than a training method; no loss functions are involved. All models are evaluated consistently under zero-shot settings with temperature set to 0.1.

## Key Experimental Results

### Main Results

| Model | REA (Overall) | EC (Overall) |
|------|:---:|:---:|
| Claude-Opus-4 (Thinking) | 24.2% | 69.7% |
| Claude-Sonnet-4 (Thinking) | 28.8% | 53.1% |
| DeepSeek-R1 | 20.1% | 28.7% |
| Doubao-Seed-1.6 (Thinking) | 28.2% | 55.3% |
| Gemini-2.5-Pro | 39.5% | 56.7% |
| **Gemini-2.5-Pro (Thinking)** | **41.4%** | 54.1% |
| GPT-4o | 15.8% | 24.3% |
| Grok-3 | 33.1% | 53.8% |
| Grok-4 | 22.2% | 61.7% |
| o3 | 35.6% | 60.5% |

Best REA: Gemini-2.5-Pro-Thinking (41.4%); best EC: Claude-Opus-4 (69.7%). No model achieves high scores on both dimensions simultaneously.

### Ablation Study

| Model | Abduction Acc. | Deduction Acc. | Induction Acc. | Avg. Total Steps (ATS) | Avg. Effective Steps (AES) |
|------|:---:|:---:|:---:|:---:|:---:|
| Grok-3 | **87.1%** | **74.0%** | **77.9%** | 11.0 | 4.0 |
| GPT-4o | 56.9% | 63.4% | 59.3% | 9.2 | 1.2 |
| Gemini-2.5-Pro | 60.3% | 59.5% | 56.7% | 12.4 | 5.8 |
| Gemini-2.5-Pro (Thinking) | 72.5% | 56.9% | 55.5% | 13.2 | 5.3 |
| o3 | 57.4% | 40.0% | 42.2% | 11.7 | 4.9 |
| Grok-4 | 58.3% | 36.6% | 40.0% | 20.1 | 4.9 |
| Claude-Opus-4 | 58.9% | 42.4% | 57.1% | 11.0 | 3.3 |
| DeepSeek-R1 | 48.8% | 40.6% | 59.0% | 8.9 | 1.9 |

Note: Per-paradigm accuracy is computed only over structurally valid reasoning steps and is therefore higher than the REA reported in Table 1.

### Key Findings

1. **EC–REA Trade-off**: Models with high EC (e.g., Claude-Opus-4 at 69.7% EC) tend to exhibit lower REA (24.2%), indicating that models covering more content simultaneously introduce more logical errors.
2. **Severe Format Violations**: GPT-4o achieves an AES of only 1.2; a large proportion of its outputs contain structural violations (mixing incompatible reasoning types), revealing difficulty in mastering the structural constraints of reasoning paradigms.
3. **Chain Length ≠ Quality**: Grok-4 generates the most steps (ATS = 20.1) yet achieves a comparable number of effective steps to o3 (ATS = 11.7), indicating substantial redundancy.
4. **Thinking Mode Helps but Has Limits**: o3 substantially outperforms GPT-4o, yet even the best model extracts fewer than 6 valid reasoning steps on average from introductions of 30+ sentences.
5. **Performance Boundary**: Top-performing models distribute along a smooth trade-off frontier, suggesting an intrinsic upper bound on reasoning capability imposed by current architectures.

## Highlights & Insights

- **Novel Task Definition**: The paper shifts the evaluation question from "can the model reason?" to "can the model formalize reasoning using standard logical paradigms?", introducing reasoning philosophy into LLM evaluation with a unique and substantive perspective.
- **Elegant Evaluation Design**: EC and REA respectively measure completeness and correctness; three-model voting eliminates single-model bias, achieving accuracy >88%.
- **High-Quality Data**: 70 articles from *Nature Communications* (2025), all peer-reviewed, ensuring the rigor and logical completeness of scientific argumentation.
- **Trade-off Frontier Insight**: Top models distributed along a smooth curve suggest that LLMs of different architectures and training objectives share some intrinsic boundary in reasoning capability—a finding with significant implications for understanding the nature of LLM reasoning.
- **Name Significance**: ARCHE derives from the ancient Greek *arché* (origin/first principle), echoing the paper's aspiration to return to the foundations of reasoning.

## Limitations & Future Work

1. **Small Data Scale**: Only 70 articles, with an evaluation cost of approximately \$4 per article via API, limiting scalability due to economic constraints.
2. **Single Domain**: Coverage is restricted to *Nature Communications*; cross-domain generalizability to chemistry, AI, law, medicine, etc. remains unknown.
3. **Introduction-Only Evaluation**: Methods and Results sections are excluded, potentially underestimating LLM capabilities in experimental reasoning and iterative hypothesis formation.
4. **Blurry Paradigm Boundaries**: The boundaries among deduction, induction, and abduction can be ambiguous in real scientific text, raising potential concerns about annotation consistency.
5. **No Training Exploration**: Only zero-shot evaluation is conducted; whether few-shot prompting or fine-tuning could alleviate the trade-off remains unexplored.

## Related Work & Insights

- **CoT Series** (Wei et al., Kojima et al.): CoT improves reasoning performance but produces untyped narratives; the RLT proposed in this paper additionally requires formalization and paradigm annotation.
- **EntailmentBank** (Dalvi et al.): Requires constructing deductive proof trees, but is limited to deduction alone; ARCHE unifies all three reasoning paradigms.
- **LINC** (Olausson et al.): Translates natural language into formal logic for external solvers, which is brittle; ARCHE requires LLMs to perform structured reasoning autonomously.
- **Insights**: Peircean reasoning paradigms could potentially be incorporated as supervision signals in pre-training or instruction fine-tuning, or used to design paradigm-aligned reward models to guide reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Entirely new task definition unifying three reasoning paradigms within a Reasoning Logic Tree.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 LLMs with multi-dimensional analysis (by reasoning type, step efficiency, and domain), though data volume is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Careful attention to task motivation, formal definitions, and the conceptual significance of the name.
- **Value**: ⭐⭐⭐⭐ — Provides a new perspective for understanding the nature of LLM reasoning; the trade-off frontier finding is thought-provoking.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](../../ICLR2026/llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)

<!-- RELATED:END -->
