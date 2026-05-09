---
title: >-
  [Paper Note] Reasoning Models Better Express Their Confidence
description: >-
  [NeurIPS 2025][LLM Reasoning][confidence calibration] This paper systematically demonstrates that reasoning models (with extended CoT) exhibit significantly better confidence calibration than non-reasoning models, and identifies "slow-thinking" behaviors—exploring alternatives, backtracking, and verification—as the fundamental source of this calibration improvement.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - confidence calibration
  - reasoning models
  - chain-of-thought
  - slow thinking
  - verbalized confidence
  - ECE
  - Brier Score
date: 2026-05-08
content_hash: f999a2305073c461
---

# Reasoning Models Better Express Their Confidence

**Conference**: NeurIPS 2025
**arXiv**: [2505.14489](https://arxiv.org/abs/2505.14489)
**Code**: [GitHub](https://github.com/MattYoon/reasoning-models-confidence)
**Area**: LLM Reasoning
**Keywords**: confidence calibration, reasoning models, chain-of-thought, slow thinking, verbalized confidence, ECE, Brier Score

## TL;DR
This paper systematically demonstrates that reasoning models (with extended CoT) exhibit significantly better confidence calibration than non-reasoning models, and identifies "slow-thinking" behaviors—exploring alternatives, backtracking, and verification—as the fundamental source of this calibration improvement.

## Background & Motivation
**State of the Field**: LLMs are increasingly deployed in high-stakes decision-making scenarios, where the ability to accurately express uncertainty (i.e., confidence calibration) is critical for trustworthy AI deployment.

**Limitations of Prior Work**: Prior studies have identified overconfidence in verbalized confidence of LLMs, but these investigations largely target conventional (non-reasoning) models and have not systematically examined the calibration behavior of reasoning models.

**Core Observation**: Reasoning models engage in explicit extended chain-of-thought before producing an answer, encompassing exploration, backtracking, and verification—a "slow-thinking" process that mirrors the human intuition of deliberating more carefully under uncertainty before committing to a response.

**Paper Goals**: (1) Are reasoning models better calibrated than non-reasoning models? (2) What is the source of calibration improvement—differences in model capability or the slow-thinking process itself?

**Starting Point**: Six paired reasoning vs. non-reasoning models are compared on multiple knowledge QA benchmarks, with CoT unfolding analysis and ablation studies used to localize the source of calibration gains.

**Core Idea**: The slow-thinking process of reasoning models—exploring alternatives, backtracking, and self-verification—naturally enables models to more accurately perceive their own uncertainty.

## Method

### Experimental Design
- **Reasoning models (6)**: R1-Distill-Qwen-32B, QwQ-32B-Preview, OR1-Preview, GLM-Z1-Air-0414, EXAONE-Deep-32B, Qwen3-235B-A22B-Thinking
- **Non-reasoning counterparts**: Each reasoning model is paired with a comparable non-reasoning model from the same family and scale (e.g., Qwen2.5-32B-Instruct vs. R1-Distill-Qwen-32B)
- **Datasets (6)**: TriviaQA, NonambigQA, MMLU-Pro-Math, MMLU-Pro-NonMath, SuperGPQA-Math, SuperGPQA-NonMath

### Confidence Elicitation
- **Verbalized confidence** is adopted: confidence is discretized into 10 intervals (from "Almost no chance 0–0.1" to "Almost certain 0.9–1.0")
- After answering, each model is prompted to select one of the 10 intervals as its confidence expression
- The midpoint of each interval is used as the numeric confidence value (e.g., 0–0.1 → 0.05)

### Calibration Metrics
1. **ECE (Expected Calibration Error)**: Measures the discrepancy between predicted confidence and actual accuracy; lower is better
2. **Brier Score**: Jointly measures calibration and resolution; lower is better
3. **AUROC**: Measures the model's ability to discriminate between correct and incorrect answers; higher is better

### CoT Unfolding Analysis
- The reasoning process is divided into equal segments by token position
- The model is prompted to express confidence after each truncated CoT segment
- Calibration metrics are tracked as a function of CoT progression

### Slow-Thinking Behavior Analysis
Three key slow-thinking behaviors are defined:
1. **Exploring Alternatives**: Considering multiple possible answers or solution strategies
2. **Backtracking**: Rejecting prior reasoning steps and correcting course
3. **Verification**: Checking and confirming one's own answers

Ablation experiments remove segments containing these behaviors from the CoT and measure the resulting change in calibration.

### Non-Reasoning Models + Slow-Thinking ICL
- In-context learning is used to present non-reasoning models with demonstrations exhibiting slow-thinking behaviors
- The resulting calibration improvement in non-reasoning models is evaluated

## Key Experimental Results

### Main Results: Reasoning vs. Non-Reasoning Model Calibration
- Reasoning models outperform their non-reasoning counterparts in **33 out of 36** settings (6 models × 6 datasets) across calibration metrics
- Reasoning models consistently perform better on all three metrics (ECE↓, Brier Score↓, AUROC↑)

### Typical Metric Differences

| Metric | Reasoning Models (avg.) | Non-Reasoning Models (avg.) | Improvement |
|--------|-------------------------|-----------------------------|-------------|
| ECE ↓ | Lower | Higher | Reasoning models significantly better |
| Brier Score ↓ | Lower | Higher | Reasoning models significantly better |
| AUROC ↑ | Higher | Lower | Reasoning models significantly better |

### CoT Unfolding Trend
- **Reasoning models**: Calibration improves steadily as CoT unfolds (p<0.05); ECE and Brier Score decrease monotonically while AUROC increases
- **Non-reasoning models**: No such trend is observed; calibration metrics remain largely unchanged throughout CoT generation

### Slow-Thinking Ablation
- Removing slow-thinking structures (exploring alternatives, backtracking, verification) from the CoT causes significant calibration degradation in reasoning models
- This confirms that calibration gains originate from slow-thinking behaviors rather than from other capability differences between models

### Non-Reasoning Models + Slow-Thinking ICL
- Non-reasoning models guided to perform slow thinking via ICL also exhibit calibration improvements
- This further supports a causal role of slow thinking in calibration improvement

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic study linking extended CoT in reasoning models to confidence calibration, identifying slow thinking as a causal source
- **Method Rigor**: ⭐⭐⭐⭐ — Comprehensive comparison across 6 paired models, 6 datasets, and 36 settings; CoT unfolding and ablation designs are rigorous
- **Practical Value**: ⭐⭐⭐⭐ — Direct implications for deploying LLMs in high-stakes decisions; the ICL-guided slow-thinking approach is immediately applicable to non-reasoning models
- **Overall**: ⭐⭐⭐⭐

## Highlights & Insights
1. **Overwhelming 33/36 advantage**: Reasoning models outperform non-reasoning models in nearly all settings, yielding highly robust conclusions
2. **Causal analysis**: The paper establishes causality through both ablation (removing slow thinking) and intervention (injecting slow thinking via ICL), going beyond mere correlation
3. **CoT unfolding analysis**: Reveals that calibration improves progressively throughout the reasoning process, offering a novel perspective on the internal mechanisms of reasoning models
4. **Transferable finding**: Non-reasoning models can also achieve calibration gains through guided slow thinking, broadening the applicability of the findings
5. **Alignment with human intuition**: "The more one deliberates, the more accurately one judges one's own uncertainty"—this finding is highly consistent with human cognitive intuition

## Limitations & Future Work
1. **Verbalized confidence only**: Token-level logit-based confidence is not examined (reasoning model APIs typically do not expose logits), potentially missing complementary signals
2. **Model scale limitations**: Experiments are primarily conducted on open-source 32B-scale models; closed-source reasoning models such as GPT-o1/o3 and Claude are not included
3. **Narrow task coverage**: Focus is on knowledge-based QA; calibration on reasoning-intensive tasks (mathematical proof, code generation, etc.) is not examined
4. **Coarse-grained behavior taxonomy**: Slow-thinking behaviors are categorized into only three types (exploration, backtracking, verification); finer-grained classification may reveal additional calibration mechanisms
5. **Scalability of ICL guidance**: The effectiveness of injecting slow thinking into non-reasoning models via ICL may be sensitive to demonstration selection and prompt design
6. **Lack of theoretical grounding**: The mechanism by which slow thinking improves calibration is not explained at the mathematical or information-theoretic level

## Related Work & Insights
- **vs. Probing-based confidence estimation (Kadavath et al.)**: Requires access to internal hidden states, limiting applicability; the proposed method relies solely on verbalized outputs and is fully black-box compatible
- **vs. Consistency-based sampling (Self-CheckGPT et al.)**: Multiple sampling incurs high computational cost (N× inference overhead); the present approach achieves better calibration with a single inference pass
- **vs. Concurrent work (Zhang et al., reasoning probes)**: Their approach trains probes from hidden states to optimize CoT generation, whereas this paper focuses on analyzing why slow thinking naturally improves calibration; the two works are complementary

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic demonstration of reasoning models' calibration advantage, with attribution to slow-thinking mechanisms
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 models × 6 datasets × multi-dimensional ablation + ICL validation; highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logical flow (phenomenon → attribution → ablation → validation); figures and tables are well-designed
- **Value**: ⭐⭐⭐⭐ — Direct guidance for reliability assessment and deployment of reasoning models

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](../../ICLR2026/llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](sprint_enabling_interleaved_planning_and_parallelized_execution_in_reasoning_mod.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] Mapping Faithful Reasoning in Language Models](mapping_faithful_reasoning_in_language_models.md)

<!-- RELATED:END -->
