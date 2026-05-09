---
title: >-
  [Paper Note] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning
description: >-
  [NeurIPS 2025][Medical Imaging][Clinical Reasoning] CureAgent proposes an Executor-Analyst collaborative framework that decouples precise tool invocation (TxAgent/Llama-8B as Executor) from high-level clinical reasoning (Gemini 2.5 as Analyst). Combined with a Stratified Ensemble Late Fusion topology that preserves evidence diversity, the system achieves 83.8% accuracy on CURE-Bench without end-to-end fine-tuning, and reveals two critical scaling findings: the context–performance paradox and the curse of dimensionality in action space.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Clinical Reasoning
  - Multi-Agent
  - Executor-Analyst
  - Stratified Ensemble
  - Training-Free Architecture Engineering
date: 2026-05-08
content_hash: 4dad6f87ea5e2741
---

# CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2512.05576](https://arxiv.org/abs/2512.05576)
**Code**: [https://github.com/June01/CureAgent](https://github.com/June01/CureAgent)
**Area**: Clinical AI / Multi-Agent Systems
**Keywords**: Clinical Reasoning, Multi-Agent, Executor-Analyst, Stratified Ensemble, Training-Free Architecture Engineering

## TL;DR
CureAgent proposes an Executor-Analyst collaborative framework that decouples precise tool invocation (TxAgent/Llama-8B as Executor) from high-level clinical reasoning (Gemini 2.5 as Analyst). Combined with a Stratified Ensemble Late Fusion topology that preserves evidence diversity, the system achieves 83.8% accuracy on CURE-Bench without end-to-end fine-tuning, and reveals two critical scaling findings: the context–performance paradox and the curse of dimensionality in action space.

## Background & Motivation

**Background**: Large language models show great promise for clinical decision support (Med-PaLM, GPT-4), yet real-world medical reasoning requires actively retrieving and integrating information from continuously updated biomedical sources (FDA labels, OpenTargets, HPO, etc.). The CURE-Bench competition evaluates agent ability to perform clinical reasoning using ToolUniverse (200+ biomedical tools).

**Limitations of Prior Work**: (a) **Context utilization failure**: TxAgent (fine-tuned Llama-3.1-8B) successfully retrieves biomedical evidence but fails to leverage it in final diagnoses, resulting in hallucinations (65.8% of error cases); (b) **Output parsing errors** (19.2%) and **instruction-following failures** (12.3%) stem from inherent limitations of small models; (c) General-purpose closed-source models (Gemini 2.5) possess strong reasoning capabilities but lack precise tool-calling training, yielding zero-shot performance inferior to TxAgent.

**Key Challenge**: Tool invocation demands **syntactic precision** (requiring domain fine-tuning), while clinical reasoning demands **semantic robustness** (requiring large model capacity)—a single model cannot simultaneously satisfy both requirements. TxAgent has tool-calling ability but weak reasoning (8B); Gemini has reasoning ability but poor tool-calling (not fine-tuned).

**Goal**: Rather than end-to-end fine-tuning, decouple and combine the "hands" of tool execution with the "brain" of clinical reasoning through architecture engineering.

**Key Insight**: Error analysis reveals that 65.8% of failures are "retrieval succeeded but reasoning failed"—the bottleneck is reasoning, not retrieval. The solution is to assign retrieval to a dedicated Executor and reasoning to a dedicated Analyst.

**Core Idea**: TxAgent as "hands" for precise retrieval + Gemini as "brain" for deep reasoning + Stratified Ensemble to preserve evidence diversity = training-free SOTA clinical agent.

## Method

### Overall Architecture
**Input**: Clinical question (multiple-choice format requiring retrieval of biomedical evidence). **Output**: Final diagnosis + reasoning chain. **Pipeline**: Three stages—(1) Executors (multiple TxAgent instances in parallel) perform tool calls to collect evidence → (2) Analyst (Gemini 2.5) integrates evidence, supplements via search, generates reasoning chain and preliminary diagnosis → (3) Post-processing module (regex matching + deduplication) ensures output format compliance.

### Key Designs

1. **Executor — Specialized Tool-Retrieval Agent**:

    - Function: Precisely invoke 200+ biomedical tools in ToolUniverse to collect evidence.
    - Mechanism: Uses TxAgent (domain fine-tuned Llama-3.1-8B) to decompose input questions into sub-queries and orchestrate multi-step tool calls and reasoning. Key innovation: **self-consistency mechanism** — $n_1$ Executors run in parallel (temperature $T=0.8$), aggregating the top-$k$ most frequent tool-call results and reasoning trajectories.
    - Design Motivation: Executors do not generate final answers—they are solely responsible for evidence collection. Multi-sampling with majority voting reduces retrieval randomness, ensuring the downstream Analyst receives a comprehensive and robust evidence set.

2. **Analyst — Long-Context Clinical Reasoner**:

    - Function: Synthesize reasoning from the noisy evidence stream produced by Executors to generate reliable clinical diagnoses.
    - Mechanism: Gemini 2.5 (Flash/Pro) serves as the reasoning backbone, relieved of the syntactic burden of tool invocation, and focuses on: (a) cross-referencing tool outputs with patient-specific comorbidities; (b) proactively searching the internet to supplement missing information; (c) filtering irrelevant noise and resolving contradictory data points. Its long context window and "System 2" reasoning capacity are leveraged to generate chain-of-thought reasoning.
    - Design Motivation: Context utilization failure in small models is fundamentally a reasoning capacity deficiency—using a large model for reasoning eliminates this bottleneck entirely.

3. **Stratified Ensemble Topology (Late Fusion)**:

    - Function: Maximize evidence diversity preservation under a fixed computational budget.
    - Mechanism: Two topologies are compared—**Config A (Global Pooling / Early Fusion)**: all Executors pool into a single context → multiple Analysts vote via self-consistency. **Config B (Stratified Ensemble / Late Fusion)**: the Executor budget is divided into $n_2$ parallel subgroups (each with $n_1$ instances); each subgroup independently aggregates → independent Analyst → final Late Fusion vote. Key advantage of Config B: different subgroups may explore distinct retrieval paths, and Late Fusion preserves this diversity.
    - Design Motivation: Early consensus in Config A filters out minority yet critical evidence—rare drug interactions are discarded by majority voting. Config B allows each retrieval path to complete the full reasoning pipeline independently, reducing collective hallucination.

4. **Post-Processing Module**:

    - Function: Ensure output format compliance and deterministic behavior.
    - Mechanism: (a) Format calibration: regular expressions map natural language conclusions to the structured output required by the benchmark; (b) Response deduplication: identical inputs produce identical outputs, eliminating LLM generation randomness.
    - Design Motivation: Clinical decision support systems require deterministic behavior—the result for the same case must be consistent across queries.

### Loss & Training
- **Training-free**: The entire framework requires no end-to-end fine-tuning. TxAgent uses pre-existing fine-tuned weights; Gemini is accessed via API.
- Executor temperature $T=0.8$ (selected via search over $T \in \{0.6, 0.7, 0.8, 0.9\}$), balancing exploration and reliability.
- Computational budget allocation: $N_{\text{total}} = n_1 \times n_2$; Stratified Ensemble uses $n_1=10, n_2=3$.

## Key Experimental Results

### Main Results — CURE-Bench Phase 2

| Architecture | Executor | $n_1$ | Analyst | $n_2$ | Accuracy |
|---|---|---|---|---|---|
| Baseline | gemini-2.5-flash | 1 | — | — | 63.1 |
| Baseline | TxAgent | 1 | — | — | 69.3 |
| SC only | TxAgent | 30 | — | — | 73.5 |
| Config A | TxAgent | 30 | gemini-flash | 3 | 80.5 |
| **Config B** | TxAgent | 10 | gemini-flash | 3 | **81.4** |
| **Config B + search** | TxAgent | 10 | gemini-flash+search | 3 | **83.8** |

### Ablation Study — Impact of Architecture Choices

| Configuration | Accuracy | Notes |
|---|---|---|
| TxAgent alone | 69.3% | Baseline |
| Decoupled (1 Exec + 1 Ana) | 74.7% | Decoupling alone yields +5.4% |
| Config A (30+3) | 80.5% | Early fusion, information bottleneck |
| Config B (10×3) | 81.4% | Late fusion, diversity preserved +0.9% |
| Config B + search | 83.8% | Search supplements missing tool information +2.4% |

### Scaling Findings

| Finding | Data | Implication |
|---|---|---|
| Context–performance paradox | Accuracy drops from 94% to 87.93% when reasoning context exceeds 12k tokens | Excessive raw evidence introduces noise that overwhelms the attention mechanism |
| Curse of dimensionality in action space | ToolUniverse v1→v2 (200→600 tools): accuracy drops from 92.0% to 87.5% | Increased tool count degrades retrieval precision |

### Key Findings
- **Decoupling is the largest source of gain**: A single Executor + single Analyst (74.7%) already outperforms both TxAgent (69.3%) and Gemini (63.1%).
- **Topology matters**: Under the same computational budget, Config B (81.4%) > Config A (80.5%); Late Fusion preserves diversity.
- Self-consistency converges rapidly: performance improves quickly for $n<15$ and plateaus beyond $n>20$ (approx. 74.2%).
- Temperature $T=0.8$ is optimal: excessively high temperature ($0.9 \to 56.7\%$) renders outputs too stochastic.
- Gemini 2.5 Pro (81.3%, post-competition model) + search suggests future foundation models may reduce reliance on the Executor.

## Highlights & Insights
- **"Hand-brain separation" as an architecture engineering philosophy**: Rather than end-to-end fine-tuning, specialized models are assigned distinct roles—small fine-tuned models perform precise tool invocation while large models handle deep reasoning. This principle is broadly applicable to all tool-augmented agent systems.
- **Late Fusion preserves evidence diversity**: The information bottleneck of Early Fusion is clearly quantified (+0.9%), with the core insight that "premature consensus discards rare but critical evidence."
- **Context–performance paradox**: Performance degrades beyond 12k tokens, demonstrating that more retrieval is not always better in RAG systems—information compression or early rejection strategies are needed.
- **Full modularity**: The modular design allows independent upgrading of the Executor and Analyst (e.g., substituting stronger models) without retraining.

## Limitations & Future Work
- The method is essentially systems engineering (multi-agent orchestration + voting), offering limited technical novelty.
- High computational cost: $n_1 \times n_2 = 30$ LLM calls per question incurs substantial API expenses.
- The context–performance paradox is observed but not resolved—confidence-based filtering strategies (e.g., DeepConf) are needed.
- Tool-count scaling degradation (4.5% drop with 600 tools) requires hierarchical retrieval or RAG over tool documentation.
- Reliance on closed-source APIs (Gemini) limits reproducibility and deployment flexibility.

## Related Work & Insights
- **vs. TxAgent (single model)**: Fine-tuned 8B model handles the full pipeline; strong in tool-calling but weak in reasoning. CureAgent's decoupled design yields +14.5%.
- **vs. Gemini-2.5-Pro (single model + search)**: Search provides broad knowledge but lacks the precision of specialized tool invocation; CureAgent combines both advantages for +9%.
- **vs. ReAct**: ReAct interleaves reasoning and action within a single model; CureAgent delegates reasoning and action to different models, better suited to asymmetric capability scenarios.
- CureAgent's Stratified Ensemble paradigm generalizes to any multi-agent RAG system: independent retrieval → independent reasoning → final vote.

## Rating
- Novelty: ⭐⭐⭐ Multi-agent decoupling and ensemble are established paradigms; innovation lies in systematic design and empirical analysis tailored to clinical scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rich ablations, multi-model comparisons, and scaling analyses, though evaluation is limited to the single CURE-Bench benchmark.
- Writing Quality: ⭐⭐⭐⭐ Error-analysis-driven motivation chain is clear; figures and tables are professional; quantitative analysis is thorough.
- Value: ⭐⭐⭐⭐ Training-free architecture engineering holds high practical value for clinical AI; scaling findings are informative for the broader community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [\[NeurIPS 2025\] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training](qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](../../ICLR2026/medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[NeurIPS 2025\] MuSLR: Multimodal Symbolic Logical Reasoning](muslr_multimodal_symbolic_logical_reasoning.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2026/medical_imaging/a_semisupervised_framework_for_breast_ultrasound_s.md)

<!-- RELATED:END -->
