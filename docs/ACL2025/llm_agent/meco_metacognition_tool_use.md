---
title: >-
  [Paper Note] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger
description: >-
  [ACL 2025][LLM Agent][Tool use] Proposed MeCo (Meta-Cognition Trigger), which extracts "meta-cognitive signals"—the model's self-assessment of its own capabilities—from within the LLM utilizing representation engineering. This adaptively determines whether to call external tools without the need for fine-tuning and with minimal computational overhead, significantly improving the accuracy of tool-use decision-making across multiple backbone models and benchmarks.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Tool use"
  - "meta-cognition"
  - "self-assessment"
  - "representation engineering"
  - "adaptive decision-making"
date: 2026-05-08
content_hash: aac1fae68439cbf0
---

# Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger

**Conference**: ACL 2025  
**arXiv**: [2502.12961](https://arxiv.org/abs/2502.12961)  
**Code**: None (Huawei)  
**Area**: LLM Agent  
**Keywords**: Tool use, meta-cognition, self-assessment, representation engineering, adaptive decision-making  

## TL;DR
Proposed MeCo (Meta-Cognition Trigger), which extracts "meta-cognitive signals"—the model's self-assessment of its own capabilities—from within the LLM utilizing representation engineering. This adaptively determines whether to call external tools without the need for fine-tuning and with minimal computational overhead, significantly improving the accuracy of tool-use decision-making across multiple backbone models and benchmarks.

## Background & Motivation
**Background**: LLMs extend their capabilities through external tools (such as search engines, calculators, and code interpreters). Existing research focuses on expanding tool libraries and optimizing tool usage workflows but neglects the decision-making problem of "when to use tools."

**Limitations of Prior Work**: (a) **Indiscriminate tool invocation** leads to unnecessary delays, as LLMs can answer many queries on their own without tools; (b) **Risk of tool errors**, where external APIs might fail or return incorrect results, and unnecessary calls increase the probability of errors; (c) Lack of lightweight tool-use decision-making mechanisms.

**Key Challenge**: LLMs should answer directly when they "know" (fast and accurate) and call tools when they "do not know" (slower but complementary)—but how does the model determine whether it "knows"?

**Goal**: To enable LLMs to self-assess their capability boundaries and adaptively decide whether external tools are needed.

**Key Insight**: Introducing "meta-cognition" (cognition about cognition) from cognitive science into LLMs—utilizing representation engineering to extract signals of "the model knowing whether it knows" from intermediate layers.

**Core Idea**: Extracting meta-cognitive signals from the representation space of the LLM $\to$ determining whether a tool is needed $\to$ adaptive decision-making.

## Method

### Overall Architecture
MeCo consists of three components: (1) **Meta-cognitive signal extraction**, which uses representation engineering (RepE) to extract "confident/unconfident" cognitive signals from the intermediate layers of the LLM; (2) **Meta-cognitive probe training**, which trains a lightweight linear classifier on a small amount of annotated data to quantify signals into scores; (3) **Double-threshold decision strategy**, which uses high and low thresholds to distinguish between three states: "confident no tool is needed," "confident a tool is needed," and "uncertain."

### Key Designs

1. **Meta-Cognition via RepE**:

    - Function: Extracting "self-knowing" signals from the hidden representations of the LLM
    - Mechanism: Constructing contrastive data—questions the LLM can answer correctly (high meta-cognition/no tool needed) vs. questions it cannot answer correctly (low meta-cognition/tool needed), and extracting the difference direction in the intermediate layer representations as the "meta-cognition direction"
    - Design Motivation: Similar to GLoRE using RepE to extract reasoning patterns, this extracts the "confident/unconfident" patterns.

2. **Meta-Cognition Probe**:

    - Function: Mapping hidden layer representations to meta-cognitive scores
    - Mechanism: Training a linear regressor/classifier on the representations extracted from the intermediate layers to predict "whether the LLM can correctly answer this query"
    - Design Motivation: Linear probes are highly efficient and do not require modifying model parameters.

3. **Double-Threshold Decision Strategy**:

    - Function: Making tool invocation decisions based on the meta-cognitive score
    - Mechanism: Setting a high threshold $\tau_h$ and a low threshold $\tau_l$. When the score exceeds $\tau_h$, the model is "confident that no tool is needed" and answers directly; when it is below $\tau_l$, it is "confident that a tool is needed" and invokes a tool; for the intermediate region, additional judgment (such as multi-sample consensus) is performed
    - Design Motivation: The double-threshold strategy allows for more cautious handling of uncertain regions, avoiding a "one-size-fits-all" approach.

### Loss & Training
- Probes are trained using linear regression loss with minimal data (hundreds of questions).
- **No LLM fine-tuning is required**—serving completely as a lightweight plugin during inference.
- Treating adaptive RAG as a special case of tool use—deciding whether retrieval is needed.

## Key Experimental Results

### Main Results

| Method | Correct Decision Rate (↑) | Latency Reduction (↑) | Final Accuracy |
|------|------------|----------|----------|
| Always Call Tool | 100% Invocation | 0% | Baseline |
| Never Call Tool | 0% Invocation | Maximum | Low (some questions cannot be answered) |
| Rule-based Threshold (perplexity, etc.) | Medium | Medium | Medium |
| **MeCo** | **Highest** | **High** | **Highest** |

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| Different Intermediate Layers | ~50% layer is optimal | Consistent with GLoRE's findings |
| Single vs. Double Threshold | Double threshold is superior | Uncertain regions require special handling |
| Cross-task Generalization | Good | Meta-cognitive signals are generic |
| As Adaptive RAG | Effective | "When to retrieve" is inherently a tool-use decision |

### Key Findings
- MeCo significantly reduces unnecessary tool invocations while maintaining or even improving final accuracy.
- Meta-cognitive signals are strongest in intermediate layers and are consistent across tasks, validating that meta-cognition is a generic capability of LLMs.
- Performs better than existing methods in adaptive RAG scenarios, indicating that "when to retrieve" and "when to use tools" are inherently the same decision.
- Minimal computational overhead—requiring only one additional linear calculation.

## Highlights & Insights
- **The transfer of the "meta-cognition" concept from cognitive science to LLMs** is the core contribution, empowering LLMs with the ability to "know whether they know."
- Similar to GLoRE in using representation engineering but with different goals—GLoRE activates reasoning capability, while MeCo activates self-assessment capability—demonstrating the versatility of RepE.
- **The double-threshold strategy** is more practical than a single threshold—in practice, the "uncertainty" region requires more cautious handling.
- Adaptive RAG is treated as a special case of tool use, unifying two seemingly different problems.
- Direct value for deployed LLM Agent systems by reducing unnecessary API calls to lower latency and costs.

## Limitations & Future Work
- Probe training requires annotations on "whether the LLM can answer"—acquiring these labels requires running the LLM.
- The stability of meta-cognitive signals may change with model updates.
- Evaluated only single-tool decisions; multi-tool selection scenarios have not been tested.
- Double thresholds require manual tuning.

## Related Work & Insights
- **vs. Self-RAG**: Self-RAG trains LLMs to generate reflection tokens to decide whether to retrieve; MeCo uses RepE to bypass training—making it more efficient.
- **vs. GainRAG**: GainRAG judges whether paragraphs provide gain; MeCo judges whether retrieval is needed—making an upstream decision.
- **vs. GLoRE**: Both use RepE but with different objectives—GLoRE activates reasoning, while MeCo activates self-assessment.
- Serves as an important reference for the tool selection mechanism of LLM Agents.

## Rating
- Novelty: ⭐⭐⭐⭐ The introduction of the meta-cognition concept + the adaptation of the RepE method is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model + multi-task + ablation + RAG scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, intuitive double-threshold strategy.
- Value: ⭐⭐⭐⭐⭐ Direct value for the practical deployment of LLM Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](../../ACL2026/llm_agent/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2025\] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](fact_audit_factcheck.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)

</div>

<!-- RELATED:END -->
