---
title: >-
  [Paper Note] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models
description: >-
  [ACL 2026][LLM Agent][implicit memory] This paper introduces ImplicitMemBench, the first benchmark for systematically evaluating implicit memory in LLMs. It comprises 300 test items across three cognitive paradigms—procedural memory, priming effects, and classical conditioning—and reveals severe limitations across 17 models: the best-performing model achieves only 66% overall accuracy, far below the human baseline.
tags:
  - ACL 2026
  - LLM Agent
  - implicit memory
  - behavioral adaptation
  - procedural memory
  - priming effect
  - classical conditioning
date: 2026-05-08
content_hash: 746a12dd351f2215
---

# ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.08064](https://arxiv.org/abs/2604.08064)
**Code**: [https://github.com/ImplicitMemBench](https://github.com/ImplicitMemBench)
**Area**: LLM Agent / LLM Evaluation
**Keywords**: implicit memory, behavioral adaptation, procedural memory, priming effect, classical conditioning

## TL;DR
This paper introduces ImplicitMemBench, the first benchmark for systematically evaluating implicit memory in LLMs. It comprises 300 test items across three cognitive paradigms—procedural memory, priming effects, and classical conditioning—and reveals severe limitations across 17 models: the best-performing model achieves only 66% overall accuracy, far below the human baseline.

## Background & Motivation

**Background**: LLM memory evaluation benchmarks (e.g., LoCoMo, LongMemEval, MemBench) have matured considerably, yet nearly all of them assess explicit memory—fact retrieval triggered by active querying.

**Limitations of Prior Work**: Existing benchmarks uniformly adopt a QA format that explicitly prompts models to recall target information, thereby neglecting implicit memory—the conversion of experience into automatic behavior rather than conscious recollection. An effective AI assistant should be able to autonomously execute learned procedures and avoid previously failed actions without explicit reminders.

**Key Challenge**: There is a fundamental gap between explicit memory evaluation ("what do you remember?") and real-world application requirements ("what do you execute automatically?"). The QA format of existing benchmarks actively cues target information, emphasizes storage capacity over first-attempt triggering, and involves costly evaluation pipelines.

**Goal**: Drawing on the cognitive science taxonomy of non-declarative memory, this work constructs the first benchmark for systematically evaluating implicit memory in LLMs.

**Key Insight**: Three classical implicit memory paradigms from cognitive science—procedural memory, priming effects, and classical conditioning—are mapped onto text-based agent scenarios via functional isomorphism.

**Core Idea**: A unified "learning/priming–interference–testing" protocol combined with a first-attempt scoring mechanism shifts evaluation from "what can the model recall?" to "what can the model execute automatically?"

## Method

### Overall Architecture
The benchmark consists of 300 test items spanning three implicit memory paradigms. Each item follows a unified three-phase protocol (learning → interference → testing). A hybrid evaluation framework combining rule-based validators and LLM judges is employed. Seventeen closed-source and open-source models are evaluated.

### Key Designs

1. **Procedural Memory Evaluation**:

    - Function: Tests whether models can internalize novel behavioral rules from minimal demonstrations and execute them automatically after interference.
    - Mechanism: Tasks are designed across five domains (tool/API usage, linguistic formatting, logical operations, abstract rules, and creative constraints), each requiring the model to suppress pretrained behaviors and internalize new rules. The learning phase provides 1–3 examples; the interference phase inserts 10–15 turns of misleading content; the test phase requires first-attempt success. Evaluation uses deterministic parsers combined with LLM judges.
    - Design Motivation: To distinguish "proceduralization" from "memorization"—models must convert explicit instructions into automatic behaviors that survive interference.

2. **Priming Effect Evaluation**:

    - Function: Measures the unconscious influence of prior topic exposure on subsequent creative generation tasks.
    - Mechanism: A paired experimental–control design is used. The experimental group is first exposed to rich thematic passages (e.g., deep-sea exploration), while the control group is exposed to neutral technical text; both groups then receive the same creative generation task. The priming effect is quantified by comparing thematic bias differences between the two groups' outputs. Topics span diverse conceptual domains including Arctic exploration, volcanic eruptions, and Renaissance alchemy.
    - Design Motivation: Priming effects are a central manifestation of unconscious context sensitivity; effective assistants need to absorb contextual cues without explicit instruction.

3. **Classical Conditioning Evaluation**:

    - Function: Tests whether models can form automatic protective responses through CS–US pairing experience.
    - Mechanism: Tasks are designed across three domains (tool safety, dialogue adaptation, and system protection). The learning phase consists of four CS–US pairings (e.g., a specific API keyword triggering an error); the interference phase inserts two turns of unrelated dialogue; the test phase reintroduces the CS to observe first-attempt behavioral responses. The evaluation assesses whether models automatically avoid harmful patterns without reminders.
    - Design Motivation: Automatic defensive learning is critical for safe agents—learning to avoid harm from experience rather than relying on explicit instructions.

### Evaluation Metrics
First-Attempt Accuracy (FTA) is used for procedural memory and classical conditioning. Priming Impact Score (PIS), computed via an LLM judge comparing experimental and control condition outputs, is used for priming effects.

## Key Experimental Results

### Main Results
Overall performance across 17 models:

| Model | Overall Accuracy | Procedural Memory | Priming Effect | Conditioning |
|-------|-----------------|-------------------|----------------|--------------|
| DeepSeek-R1 | 65.3% | Top tier | Moderate | Low |
| Qwen3-32B | 64.1% | High | Moderate | Low |
| GPT-5 | 63.0% | High | Moderate | Low |
| Human Baseline | Far above all models | High | High | High |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Inhibition vs. preference | Inhibitory learning: 17.6% vs. preference learning: 75.0% (large asymmetry) |
| Memory-augmented agents | External memory modules do not consistently improve implicit memory performance |
| Cross-paradigm correlation | Procedural memory performance does not predict classical conditioning performance |

### Key Findings
- Severe ceiling effect: No model exceeds 66% overall accuracy; even the best model remains far below the human baseline.
- Paradigm asymmetry: Procedural memory is most tractable; classical conditioning constitutes a fundamental bottleneck; priming effect scores cluster in the moderate range.
- Extreme inhibition–preference asymmetry: Models strongly favor positive learning (75.0%) while struggling with inhibitory learning (17.6%).
- Memory-augmented agents (with explicit storage and retrieval) do not consistently improve implicit memory, demonstrating that implicit memory cannot be reduced to explicit retrieval.

## Highlights & Insights
- The paradigm shift from "what is remembered?" to "what is executed automatically?" carries profound implications and identifies a fundamental blind spot in current LLM evaluation.
- The functional isomorphism mapping of three cognitive science paradigms is elegantly designed, preserving causal structure while achieving textual operationalization.
- The extreme inhibition–preference asymmetry is a significant finding, suggesting architectural-level deficiencies in LLMs' capacity for forgetting and inhibition.

## Limitations & Future Work
- The dataset contains only 300 items; though carefully designed, its scale is limited.
- Context length is approximately 500 tokens, leaving long-term cross-session persistence of implicit memory untested.
- Non-associative learning paradigms (habituation/sensitization) are not included.
- Future work should explore architectural innovations—rather than parameter scaling—to improve implicit memory capabilities.

## Related Work & Insights
- **vs. LoCoMo/LongMemEval**: These benchmarks assess active retrieval of explicit memory, whereas this work evaluates passive triggering of implicit memory.
- **vs. MemoryAgentBench**: That benchmark evaluates capabilities such as retrieval, learning, and forgetting but remains within an explicit memory framework; this work fills the implicit memory gap.
- **vs. memory-augmented agents**: External memory modules cannot resolve implicit memory deficiencies; architectural-level innovation is required.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First implicit memory benchmark; innovative evaluation paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across 17 models, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Solid cognitive science foundations; rigorous experimental design logic.
- Value: ⭐⭐⭐⭐⭐ Reveals fundamental capability deficiencies in LLMs and provides important guidance for future research directions.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)

<!-- RELATED:END -->
