---
title: >-
  [Paper Note] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering
description: >-
  [ACL 2026][LLM Agent][Table Question Answering] This paper proposes MATA, a multi-agent framework for table question answering that employs a scheduler to prioritize reasoning paths (CoT/PoT/text2SQL), a confidence checker to filter candidate answers, and a judge agent for arbitration. The framework achieves model-agnostic, efficient, and accurate TableQA, with an average EM improvement of 40.1% across 10 LLMs.
tags:
  - ACL 2026
  - LLM Agent
  - Table Question Answering
  - Multi-Agent Framework
  - Multi-Reasoning Paths
  - Model-Agnostic
  - LLM Efficiency
date: 2026-05-08
content_hash: 48696bcb2c89201f
---

# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

**Conference**: ACL 2026
**arXiv**: [2602.09642](https://arxiv.org/abs/2602.09642)
**Code**: [GitHub](https://github.com/AIDASLab/MATA)
**Area**: LLM Agent
**Keywords**: Table Question Answering, Multi-Agent Framework, Multi-Reasoning Paths, Model-Agnostic, LLM Efficiency

## TL;DR
This paper proposes MATA, a multi-agent framework for table question answering that employs a scheduler to prioritize reasoning paths (CoT/PoT/text2SQL), a confidence checker to filter candidate answers, and a judge agent for arbitration. The framework achieves model-agnostic, efficient, and accurate TableQA, with an average EM improvement of 40.1% across 10 LLMs.

## Background & Motivation

**Background**: LLMs have significantly advanced table question answering (TableQA), enabling natural language interaction with structured tables. Existing methods typically leverage reasoning strategies such as CoT, PoT (Program-of-Thought), or text2SQL to generate answers.

**Limitations of Prior Work**: (1) Most high-performing methods rely on closed-source LLMs (e.g., GPT-4o), making them unsuitable for privacy-sensitive or cost-constrained scenarios, with their reliability on small open-source models insufficiently validated; (2) To improve answer reliability, existing frameworks frequently invoke LLM inference (e.g., Self-Consistency), resulting in high computational costs and even accuracy degradation due to over-prompting; (3) Most frameworks only exploit CoT and PoT, failing to fully leverage the complementary diversity of CoT, PoT, and text2SQL.

**Key Challenge**: A fundamental trade-off exists between reasoning diversity and efficiency—adding more reasoning paths improves accuracy but incurs additional LLM inference overhead, while blindly executing all paths wastes resources and may introduce noise.

**Goal**: To build a model-agnostic TableQA framework that maintains high accuracy across diverse open-source and closed-source LLMs, while minimizing LLM invocations through intelligent scheduling.

**Key Insight**: Reasoning diversity does not require a fixed inference budget—lightweight controllers can determine which reasoning branches are necessary and when validation can be terminated early.

**Core Idea**: Lightweight tool models (Scheduler, Confidence Checker, Format Matcher) coordinate reasoning path selection and answer verification across multiple LLM agents, achieving an optimal balance between reasoning diversity and efficiency.

## Method

### Overall Architecture
MATA takes a table $T$ and question $Q$ as input and produces a final answer through a three-stage pipeline: (1) **Agent Selection**: the Scheduler determines the execution priority of PoT and text2SQL while CoT agents run in parallel; (2) **Code Generation & Debugging**: PoT/text2SQL agents generate code, which is iteratively repaired by a Debug Agent; (3) **Final Answer Decision**: the Confidence Checker evaluates the confidence of candidate answers and invokes the Judge Agent for arbitration when necessary.

### Key Designs

1. **Scheduler**:

    - **Function**: Determines whether to prioritize PoT or text2SQL based on table features and question semantics.
    - **Mechanism**: Built on MobileBERT with a two-layer MLP (only 24.65M parameters). It takes table meta-features (size, schema, data types) and question semantics as input and outputs probabilities for PoT and text2SQL. The higher-probability path is executed first; if its answer agrees with CoT, the other path is skipped and the framework proceeds directly to answer selection.
    - **Design Motivation**: The relative advantage of different reasoning paths depends on the underlying model characteristics and question type. Intelligent scheduling avoids unnecessary LLM calls. Training data is sourced from inference results of three LLMs on WikiTQ/TabMWP/TabFact.

2. **Confidence Checker (CC)**:

    - **Function**: Computes a confidence score for each candidate answer to determine whether additional Judge Agent arbitration is required.
    - **Mechanism**: Fine-tuned on DeBERTaV3-large (~435M parameters). It takes the table, question, and candidate answers as input and outputs confidence scores for each reasoning path. If the highest confidence score exceeds a threshold $\theta = 0.1$, that answer is selected directly; otherwise, the Judge Agent is invoked for a comprehensive decision.
    - **Design Motivation**: Avoids invoking the expensive LLM judge for every query, as the lightweight model suffices for high-quality answer selection in most cases.

3. **Code Generation & Debugging Loop**:

    - **Function**: Iteratively repairs syntax and logical errors in code generated by PoT/text2SQL agents.
    - **Mechanism**: PoT/text2SQL agents generate and execute code; errors are corrected by the corresponding Debug Agent (PDA/SDA) for up to $N=3$ iterations. An early stopping condition is introduced: if the new code is highly similar to the previous version and yields the same execution result, the loop terminates.
    - **Design Motivation**: Code-based reasoning is naturally prone to syntax errors, while iterative refinement of text-based reasoning (CoT) yields minimal benefit. Debugging is therefore applied only to code paths to balance cost and effectiveness.

### Loss & Training
The Scheduler and Confidence Checker are each trained on 173,664 samples. Training labels for the Scheduler indicate whether the PoT or text2SQL path is correct; labels for the CC indicate the correctness of each of the three paths. All LLM agents share the same backbone model and are differentiated only through role prompts.

## Key Experimental Results

### Main Results

| Benchmark | Metric | MATA | MixSC | SynTQA | TabLaP |
|-----------|--------|------|-------|--------|--------|
| Penguins (avg) | EM | 0.881 | 0.626 | 0.810 | 0.524 |
| Penguins (avg) | F1 | 0.881 | 0.637 | 0.811 | 0.544 |
| TableBench (avg) | EM | 0.451 | 0.286 | 0.322 | 0.260 |
| TableBench (avg) | F1 | 0.482 | 0.331 | 0.362 | 0.307 |

### Ablation Study

| Configuration | Penguins EM | TableBench EM | Note |
|---------------|-------------|---------------|------|
| MATA (full) | 0.881 | 0.451 | Complete framework |
| w/o Scheduler | ~0.86 | ~0.43 | All paths executed; LLM calls increase |
| w/o CC (JA only) | ~0.85 | ~0.42 | Judge Agent invoked every time |
| w/o Debug | ~0.82 | ~0.38 | No code debugging |

### Key Findings
- MATA yields the most significant gains on small models (3B–7B): qwen2.5-3b improves from 0.163 EM (TabLaP) to 0.291, and mistral-7b improves from 0.036 to 0.294.
- MATA maintains advantages on large models, though the margin narrows due to their stronger inherent reasoning capabilities.
- The Scheduler effectively reduces LLM calls by approximately 30–40% while maintaining or improving accuracy.

## Highlights & Insights
- The design of lightweight tool models (totaling under 1B parameters) coordinating LLM agents is highly practical—the Scheduler and CC act as "gatekeepers" to prevent unnecessary expensive inference calls.
- The complementarity of three reasoning paths is well validated: CoT excels at ambiguous or intuitive questions, PoT at numerical computation, and text2SQL at structured queries.
- The model-agnostic design allows the framework to be directly transferred to any new LLM, which is particularly valuable given the rapid iteration of open-source models.

## Limitations & Future Work
- Training of the Scheduler and CC relies on specific datasets (WikiTQ/TabMWP/TabFact), which may limit generalization to tables with substantially different domain characteristics.
- The current framework supports only single-table reasoning; multi-table join question answering has not been addressed.
- The maximum debug iteration count $N=3$ is empirically determined; tasks of varying complexity may require adaptive adjustment.

## Related Work & Insights
- **vs. MixSC**: MixSC integrates only CoT and Python paths with self-consistency voting, lacking text2SQL and intelligent scheduling. MATA outperforms it by an average EM margin of 25.5%.
- **vs. SynTQA**: SynTQA integrates text2SQL and end-to-end TQA but does not support model switching. MATA's model-agnostic design confers a substantial advantage on small models.
- **vs. TabLaP**: TabLaP relies on multiple heterogeneous LLMs and supports only specific models, whereas MATA achieves superior performance using a unified backbone.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The architecture combining lightweight tools with multi-agent coordination is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 10 LLMs, two benchmarks, and three metrics with broad experimental scope.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and detailed algorithmic descriptions.
- **Value**: ⭐⭐⭐⭐ The model-agnostic framework has direct reference value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](../../CVPR2026/llm_agent/nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](../../AAAI2026/llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)

</div>

<!-- RELATED:END -->
