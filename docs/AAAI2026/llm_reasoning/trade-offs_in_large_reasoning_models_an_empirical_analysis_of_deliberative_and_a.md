---
title: >-
  [Paper Note] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities
description: >-
  [AAAI 2026][LLM Reasoning][Large Reasoning Models] This paper systematically evaluates the negative impact of deliberative reasoning on foundational capabilities (helpfulness and harmlessness) in Large Reasoning Models (…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Large Reasoning Models"
  - "Reasoning–Foundational Capability Trade-offs"
  - "Adaptive Reasoning"
  - "Zero-Thinking"
  - "Safety"
date: 2026-05-08
content_hash: 506e840af9ca62d3
---

# Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities

**Conference**: AAAI 2026
**arXiv**: [2503.17979](https://arxiv.org/abs/2503.17979)  
**Code**: [https://github.com/SCIR-SC-Qiaoban-Team/FreeEvalLM](https://github.com/SCIR-SC-Qiaoban-Team/FreeEvalLM)  
**Area**: LLM Reasoning
**Keywords**: Large Reasoning Models, Reasoning–Foundational Capability Trade-offs, Adaptive Reasoning, Zero-Thinking, Safety

## TL;DR
This paper systematically evaluates the negative impact of deliberative reasoning on foundational capabilities (helpfulness and harmlessness) in Large Reasoning Models (LRMs) such as DeepSeek-R1, QwQ, and OpenThinker. It finds that deliberative reasoning significantly degrades instruction-following and safety, and proposes adaptive reasoning modes—Zero-Thinking, Less-Thinking, and Summary-Thinking—that effectively mitigate these deficiencies.

## Background & Motivation
**Background**: LRMs represented by OpenAI o1/o3 and DeepSeek-R1 have achieved substantial progress on specialized reasoning tasks such as mathematics and coding through long chain-of-thought reasoning. Community attention has largely focused on the performance, efficiency, and robustness of LRMs on reasoning tasks.

**Limitations of Prior Work**: The effect of acquiring deliberative reasoning capability on foundational capabilities—general task performance, instruction-following, and safety—has received almost no systematic study. Cognitive science suggests that human reasoning ability is closely coupled with overall cognitive function; an analogous relationship should hold for LRMs.

**Key Challenge**: The process of acquiring reasoning capability through distillation or reinforcement learning may come at the cost of the model's original helpfulness and harmlessness. This trade-off is critical for real-world deployment yet remains unquantified.

**Goal**
- RQ1: Which foundational capabilities are most severely degraded after LRMs acquire reasoning ability?
- RQ2: How does inference-time compute affect LRM performance on general tasks?
- Can degradation of foundational capabilities be mitigated by controlling the reasoning mode?

**Key Insight**: By inserting special tokens (e.g., `<think></think>`) into the LRM's thinking process, the reasoning depth is manually controlled to realize three adaptive reasoning modes: Zero-Thinking, Less-Thinking, and Summary-Thinking.

**Core Idea**: Deliberative reasoning in LRMs substantially harms foundational capabilities while improving specialized reasoning; adaptive reasoning modes that dynamically allocate inference-time compute represent a critical direction for future LRM development.

## Method

### Overall Architecture
This paper presents a systematic empirical study rather than a new model. It comprises:
- **Evaluation targets**: LRMs from three model families (DeepSeek / Qwen / LLaMA) at multiple scales (7B–671B) and their corresponding chat counterparts.
- **Evaluation dimensions**: Helpfulness (MMLU-Pro, Live-Bench, IFEval, MT-Bench) and Harmlessness (StrongReject, WildJailbreak).
- **Controlled variables**: Systematic comparison across reasoning modes (Deliberative / Zero / Less / Summary-Thinking).

### Key Designs

1. **Foundational Capability Evaluation Framework**:

    - *Function*: Comprehensively assess changes in foundational performance after LRMs acquire reasoning ability.
    - *Mechanism*: Each distilled LRM is directly compared with its source chat model (e.g., OpenThinker-7B vs. Qwen2.5-7B-Instruct), measuring performance differences across six benchmarks. For RL-trained LRMs (e.g., QwQ-32B), capability changes are inferred indirectly by modifying the reasoning mode.
    - *Design Motivation*: No prior work systematically compared LRMs and chat models on non-reasoning tasks. Evaluating only reasoning tasks leads to the erroneous conclusion that LRMs are uniformly superior.

2. **Adaptive Reasoning Modes**:

    - *Function*: Regulate inference-time compute by controlling the length and content of the LRM's thinking process.
    - *Mechanism*:
        - **Zero-Thinking**: Appends `</think>` immediately after the input, forcing the model to skip the reasoning process entirely.
        - **Less-Thinking**: Inserts `</think>` at the $p\%$ position of the reasoning process to terminate thinking early ($p$ = 10%, 20%, 50%, 60%, 80%, 90%).
        - **Summary-Thinking**: Uses GPT-4o to compress the full reasoning process into a summary, which is inserted between `<think>` and `</think>`.
        - **Summary-Thinking-Plus**: Retains the first sentence of the original reasoning plus the compressed summary, as the opening sentence pattern has a significant impact on accuracy.
    - *Design Motivation*: Different tasks require different reasoning depths. General and safety tasks may not require long-chain reasoning, and excessive reasoning may be actively harmful.

3. **Thought Safety Analysis**:

    - *Function*: Analyze whether the LRM's thinking process itself is safe.
    - *Mechanism*: LRM outputs are classified according to a 2×2 matrix—safe/unsafe thought × safe/unsafe response—with GPT-4o serving as the safety judge.
    - *Design Motivation*: Even when an LRM's final response refuses a malicious query, the thinking process may contain substantial unsafe content. This constitutes a previously overlooked safety risk.

### Loss & Training
This paper is a purely evaluative study and involves no training. All LRMs are served using vLLM on 8×A100 GPUs; decoding parameters and prompt formats strictly follow official configurations.

## Key Experimental Results

### Main Results
Foundational capability comparison between distilled LRMs and their corresponding chat models:

| Model | MMLU-Pro | Live-Bench | IFEval | MT-Bench | StrongReject↑ | WildJailbreak↑ |
|-------|----------|------------|--------|----------|--------------|----------------|
| Qwen2.5-7B-Instruct | 54.44 | 36.34 | 67.84 | 7.94 | 95.21 | 10.70 |
| OpenThinker-7B | 39.04↓ | 20.81↓ | 34.20↓ | 7.33↓ | 37.29↓ | 12.45 |
| Qwen2.5-32B-Instruct | 67.07 | 53.85 | 77.26 | 8.32 | 95.00 | 13.30 |
| s1.1-32B | 43.77↓ | 34.42↓ | 37.34↓ | 7.98↓ | 49.38↓ | 4.90↓ |
| Llama-3.3-70B-Instruct | 70.54 | 60.30 | 89.83 | 8.11 | 95.63 | 19.50 |
| R1-Distill-Llama-70B | 71.57 | 54.09↓ | 76.89↓ | 8.03 | 89.17↓ | 28.25 |

s1.1-32B suffers a 47.38% drop on IFEval while incurring a 250% increase in reasoning cost.

### Ablation Study (Adaptive Reasoning Modes)

| Model + Mode | IFEval | StrongReject↑ | WildJailbreak↑ |
|--------------|--------|--------------|----------------|
| s1.1-32B (original) | 37.34 | 49.38 | 4.90 |
| + Zero-Thinking | 42.33 | 64.79 | 11.15 |
| + Summary-Thinking | **54.16** | 53.96 | 4.70 |
| QwQ-32B (original) | 75.60 | 95.00 | 10.65 |
| + Zero-Thinking | 64.51 | **98.33** | **59.65** |
| + Summary-Thinking | 77.26 | 93.33 | 11.60 |
| R1-Distill-Llama-70B (original) | 75.60 | 89.17 | 28.25 |
| + Zero-Thinking | 63.22 | **99.17** | **89.10** |

### Key Findings
- **Longer thoughts correlate with worse performance**: In win/loss analyses, samples where performance degrades exhibit significantly longer thoughts than samples where performance improves, indicating that excessive reasoning actively harms general task performance.
- **Zero-Thinking dramatically improves safety**: Skipping the reasoning process raises jailbreak resistance in R1-Distill-Llama-70B from 28.25 to 89.10 on WildJailbreak, surpassing even the original chat model.
- **Summary-Thinking substantially improves instruction-following**: s1.1-32B improves on IFEval from 37.34 to 54.16 (+45%); similar gains are observed on QwQ.
- **Unsafe thoughts are the primary cause of unsafe responses**: Over 80% of R1-Distill-Llama-70B's thinking processes contain unsafe content, even when the final response appears safe.
- **The optimal Less-Thinking ratio is task-dependent**: No universal thinking ratio exists; different benchmarks reach their optimum at different truncation percentages.

## Highlights & Insights
- **First systematic quantification of the reasoning–foundational capability trade-off in LRMs**: The community has over-focused on improvements in reasoning benchmarks while neglecting the foundational capabilities essential for real-world deployment. This paper fills an important gap.
- **The finding that "thoughts are also unsafe" is highly valuable**: Even when a response is filtered by a safety mechanism, unsafe content in the thinking process itself constitutes a security risk (potentially observable or exploitable by adversaries). This has important implications for LRM safety alignment research.
- **Simplicity and effectiveness of adaptive reasoning**: Rich reasoning-mode control is achieved merely by inserting `</think>` tokens. The approach is extremely simple yet yields significant effects, providing a practical foundation for future "think when needed" adaptive inference paradigms.

## Limitations & Future Work
- **Limited evaluation scope**: Coverage is restricted to general tasks, instruction-following, and safety; multi-modal scenarios and open-ended dialogue closer to real-world applications are not addressed.
- **Incomplete model coverage**: Only specific checkpoints are evaluated; applicability to MoE architectures or multi-modal LLMs remains unknown.
- **Adaptive reasoning is manually controlled**: Zero/Less/Summary-Thinking requires the mode to be specified in advance; no method is proposed to automatically determine when reasoning is needed and how deep it should be.
- **Future directions**: Training LRMs to automatically assess input difficulty and dynamically allocate inference-time compute; mixing long and short CoT data during distillation; incorporating reasoning length as a reward factor in RL.

## Related Work & Insights
- **vs. Reasoning efficiency studies (e.g., o1-preview analyses)**: Prior work addresses "over-thinking on reasoning tasks"; this paper extends the analysis to "the broad negative impact of reasoning on non-reasoning tasks," adopting a wider perspective.
- **vs. Safety studies such as SafeChain**: SafeChain focuses on embedding safety reasoning within CoT; this paper reveals that CoT itself is a source of unsafe content—the two lines of work are complementary.
- **Implications for LRM deployment**: Deliberative reasoning should not be applied uniformly in production settings; the reasoning depth must be adaptively selected according to query type.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of the trade-off between reasoning and foundational capabilities in LRMs; a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model families, multiple scales, six benchmarks, four reasoning modes, fine-grained analysis, and case studies—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; findings are convincing; case studies effectively support the conclusions.
- Value: ⭐⭐⭐⭐⭐ Carries an important cautionary message for the LRM community—gains in reasoning capability may come at the cost of foundational capabilities, making adaptive reasoning an essential requirement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](text-to-scene_with_large_reasoning_models.md)
- [\[AAAI 2026\] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)
- [\[NeurIPS 2025\] Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost](../../NeurIPS2025/llm_reasoning/are_large_reasoning_models_good_translation_evaluators_analysis_and_performance_.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](../../ICLR2026/llm_reasoning/towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)

</div>

<!-- RELATED:END -->
