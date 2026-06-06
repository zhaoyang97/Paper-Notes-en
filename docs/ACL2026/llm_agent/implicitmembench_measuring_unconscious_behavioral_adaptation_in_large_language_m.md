---
title: >-
  [Paper Note] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models
description: >-
  [ACL 2026][LLM Agent][Implicit Memory] Proposes ImplicitMemBench, the first benchmark to systematically evaluate implicit memory in LLMs. It covers 300 test items across three cognitive paradigms: procedural memory…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Implicit Memory"
  - "Behavioral Adaptation"
  - "Procedural Memory"
  - "Priming Effect"
  - "Classical Conditioning"
date: 2026-05-08
content_hash: 6402e35036dc0a09
---

# ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.08064](https://arxiv.org/abs/2604.08064)  
**Code**: [https://github.com/ImplicitMemBench](https://github.com/ImplicitMemBench)  
**Area**: LLM Agent / LLM Evaluation  
**Keywords**: Implicit Memory, Behavioral Adaptation, Procedural Memory, Priming Effect, Classical Conditioning

## TL;DR
Proposes ImplicitMemBench, the first benchmark to systematically evaluate implicit memory in LLMs. It covers 300 test items across three cognitive paradigms: procedural memory, priming effects, and classical conditioning. Evaluations across 17 models reveal significant limitations, with the best model achieving only 66% overall accuracy, far below the human baseline.

## Background & Motivation

**Background**: LLM memory evaluation benchmarks (e.g., LoCoMo, LongMemEval, MemBench) have matured, but almost all focus on explicit memory—fact retrieval triggered through active queries.

**Limitations of Prior Work**: Existing benchmarks uniformly adopt a QA format, using explicit prompts to trigger the model to recall target information. This ignores implicit memory—where experience transforms into automatic behavior rather than conscious recall. Effective AI agents should automatically execute learned procedures or avoid failed operations without requiring explicit reminders.

**Key Challenge**: There is a fundamental gap between explicit memory evaluation ("what do you remember") and practical application requirements ("what do you automate"). The QA format in existing benchmarks actively prompts for target information, emphasizes storage capacity over first-attempt triggers, and involves high-cost evaluation pipelines.

**Goal**: Construct the first benchmark for systematically evaluating implicit memory in LLMs based on the non-declarative memory classification system from cognitive science.

**Key Insight**: Functionally map three classic implicit memory paradigms from cognitive science (procedural memory, priming effect, classical conditioning) onto text-based agent scenarios through functional isomorphism.

**Core Idea**: Shift the evaluation from "what the model can recall" to "what the model can automate" using a unified "learning/priming-interference-test" protocol and a first-attempt scoring mechanism.

## Method

### Overall Architecture
The benchmark consists of 300 test items covering three implicit memory paradigms. Each item follows a unified three-stage protocol (learning → interference → testing). A hybrid evaluation framework utilizing rule-based validators and LLM-as-a-Judge is employed to test 17 closed-source and open-source models.

### Key Designs

1. **Procedural Memory Evaluation**:

    - **Function**: Tests whether the model can internalize new behavioral rules from minimal demonstrations and execute them automatically after interference.
    - **Mechanism**: Tasks are designed across five domains (tool/API usage, language format, logical operations, abstract rules, creative constraints). Each task requires the model to suppress pre-trained behaviors and internalize new rules. The learning phase provides 1-3 examples, the interference phase inserts 10-15 rounds of misleading content, and the testing phase requires success on the first attempt. Verification is performed using deterministic parsers and LLM judges.
    - **Design Motivation**: Distinguish "proceduralization" from "memorization"—the model must transform explicit instructions into automatic behaviors capable of withstanding interference.

2. **Priming Effect Evaluation**:

    - **Function**: Measures the unconscious influence of prior thematic exposure on subsequent creative tasks.
    - **Mechanism**: Utilizes a paired experimental-control design. The experimental group is exposed to thematically rich passages (e.g., deep-sea exploration), while the control group is exposed to neutral technical text. Both are then given the same creative generation task. The priming effect is quantified by comparing the thematic bias differences between the outputs. Themes cover various conceptual domains such as Arctic exploration, volcanic eruptions, and Renaissance alchemy.
    - **Design Motivation**: Priming is a core manifestation of unconscious context sensitivity; effective assistants need to absorb environmental cues without explicit instructions.

3. **Classical Conditioning Evaluation**:

    - **Function**: Tests whether models can form automatic protective responses through CS-US pairing experiences.
    - **Mechanism**: Tasks are designed across three domains (tool safety, dialogue adaptation, system protection). The learning phase involves 4 rounds of CS-US pairing (e.g., a specific API keyword triggering an error), followed by 2 rounds of irrelevant dialogue in the interference phase. The testing phase re-introduces the CS to observe the first behavioral response. The evaluation determines if the model automatically avoids harmful patterns without reminders.
    - **Design Motivation**: Automatic defensive learning is critical for safety agents—learning to automatically avoid hazards based on experience rather than relying solely on instructions.

### Metrics
First-Attempt Accuracy (FTA) is used for procedural memory and classical conditioning. Priming Influence Score (PIS) is calculated using LLM judges to compare differences between experimental and control conditions for priming effects.

## Key Experimental Results

### Main Results
Overall performance of 17 models:

| Model | Overall Accuracy | Procedural Memory | Priming Effect | Conditioning |
|------|-----------|-----------|---------|---------|
| DeepSeek-R1 | 65.3% | Top Group | Medium | Low |
| Qwen3-32B | 64.1% | High | Medium | Low |
| GPT-5 | 63.0% | High | Medium | Low |
| Human Baseline | Far above all models | High | High | High |

### Ablation Study

| Analysis Dimension | Finding |
|---------|------|
| Suppression vs Preference | Inhibitory learning 17.6% vs Preferential learning 75.0% (Huge asymmetry) |
| Memory-Augmented Agents | External memory modules do not consistently improve implicit memory performance |
| Paradigm Correlation | Proficiency in procedural memory does not predict conditioning performance |

### Key Findings
- **Severe Ceiling Effect**: No model exceeds 66% overall accuracy; the best-performing models remain far below the human baseline.
- **Paradigm Asymmetry**: Procedural memory is the most solvable, classical conditioning constitutes a fundamental bottleneck, and priming effects cluster in the medium range.
- **Extreme Suppression-Preference Asymmetry**: Models show a strong preference for positive learning (75.0%) but struggle significantly with inhibitory learning (17.6%).
- **Memory-Augmented Agents** (using explicit storage and retrieval) do not consistently improve implicit memory, suggesting that implicit memory cannot be reduced to explicit retrieval.

## Highlights & Insights
- The shift in evaluation paradigm from "what is remembered" to "what is automated" is significant, highlighting a fundamental blind spot in current LLM evaluation.
- The functional isomorphism mapping of cognitive science paradigms is ingeniously designed, maintaining causal structures while enabling textual evaluation.
- The extreme asymmetry between suppression and preference is a critical finding, suggesting that LLM "forgetting/suppression" capabilities have architectural deficiencies.

## Limitations & Future Work
- The dataset contains only 300 items; while carefully designed, the scale is limited.
- Context length is limited to ~500 tokens, leaving long-term, cross-session implicit memory persistence untested.
- Non-associative learning paradigms (habituation/sensitization) are not included.
- Future research needs to explore architectural innovations (rather than mere parameter scaling) to improve implicit memory.

## Related Work & Insights
- **vs LoCoMo/LongMemEval**: These assess active retrieval of explicit memory, whereas Ours evaluates passive triggering of implicit memory.
- **vs MemoryAgentBench**: While it evaluates retrieval, learning, and forgetting, it remains within an explicit framework; Ours fills the gap in implicit memory.
- **vs Memory-Augmented Agents**: External memory modules fail to solve implicit memory issues, indicating a need for architecture-level innovation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First implicit memory benchmark; innovative evaluation paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of 17 models, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Solid cognitive science foundation with logically rigorous experimental design.
- **Value**: ⭐⭐⭐⭐⭐ Reveals fundamental capability flaws in LLMs, providing significant guidance for future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)

</div>

<!-- RELATED:END -->
