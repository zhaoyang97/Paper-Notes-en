---
title: >-
  [Paper Note] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning
description: >-
  [ACL2026][LLM Efficiency][Large-Small Model Collaboration] Tandem enables LLMs to generate only four types of short reasoning clues (Goal, Planning, Retrieval, Action)…
tags:
  - "ACL2026"
  - "LLM Efficiency"
  - "Large-Small Model Collaboration"
  - "Structured Thought Prompting"
  - "Reasoning Acceleration"
  - "Uncertainty Estimation"
  - "Cost-Aware Routing"
date: 2026-05-08
content_hash: 948394edbd0efb65
---

# Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning

**Conference**: ACL2026  
**arXiv**: [2604.23623](https://arxiv.org/abs/2304.23623)  
**Code**: https://github.com/Applied-Machine-Learning-Lab/ACL2026_Tandem  
**Area**: LLM Reasoning / Model Collaboration / Inference Efficiency  
**Keywords**: Large-Small Model Collaboration, Structured Thought Prompting, Reasoning Acceleration, Uncertainty Estimation, Cost-Aware Routing

## TL;DR
Tandem enables LLMs to generate only four types of short reasoning clues (Goal, Planning, Retrieval, Action), while SLMs judge clue sufficiency via perplexity and entropy to complete the answer. It achieves or exceeds standalone LLM performance on MATH, GSM8K, and HumanEval with approximately 60% of the computational cost.

## Background & Motivation
**Background**: LLM reasoning has transitioned from simple answering to an explicit thinking paradigm. Typical models expand long reasoning chains before generating final answers. This explicit thinking enhances the interpretability of complex mathematics, scientific reasoning, and code generation, making models more robust for difficult problems.

**Limitations of Prior Work**: The primary cost of explicit thinking stems from generation length. The paper notes that the reasoning traces of thinking models are often 5 to 10 times longer than standard outputs, creating latency and API cost pressures in deployment. Existing methods, such as reinforcement fine-tuning to make models "think less," require modifying LLM weights—unusable for closed-source API models—and may harm general capabilities.

**Key Challenge**: High-quality reasoning requires the abstract planning and key insights of LLMs, but a complete reasoning chain contains significant exploration, explanation, and repetitive steps. That is, the real expense is not "obtaining the key idea," but forcing the LLM to write out the entire solution process.

**Goal**: The authors aim to compress LLM reasoning capabilities into lightweight guidance for cheaper SLMs to execute without training or modifying the mentor LLM, while dynamically deciding the required level of LLM guidance for problems of varying difficulty.

**Key Insight**: The paper uses a mentor-intern analogy: the LLM acts as a mentor responsible for providing goals, plans, retrieved knowledge, and key actions; the SLM acts as an intern responsible for completing specific reasoning based on these clues. Whether to continue consulting the mentor is not determined by a fixed token budget, but by the distributional uncertainty of the SLM when processing current clues.

**Core Idea**: Deconstruct long LLM thinking into phased, structured insights and train a sufficiency classifier using SLM PPL/entropy features to determine when to stop LLM generation and allow the SLM to complete the answer.

## Method
The Tandem method does not focus on model debate or one-time query routing. Instead, it splits reasoning into "guidance generation" and "answer completion" roles. The LLM produces increasingly detailed reasoning clues, while the SLM judges its own ability to complete the task while reading them. If current clues are deemed sufficient, the LLM stops early, and the SLM generates the final answer.

### Overall Architecture
Given a question $Q$, the mentor LLM generates reasoning insights across three effort levels. Each phase includes four categories: Goal (stating the final objective), Planning (providing strategies), Retrieval (extracting relevant facts), and Action (giving key calculations or logic).

New insights at stage $t$ are denoted as $\Delta I^t$, with accumulated clues $I^t = I^{t-1} \oplus \Delta I^t$. After each stage, the intern SLM reads $Q \oplus I^t$, extracts statistical features of token-level perplexity and entropy, and passes them to an MLP classifier to output a sufficiency score $s^t$.

If $s^t$ exceeds the threshold $\tau^t$ for the current stage, the system stops LLM generation and lets the SLM output the final answer based on $Q \oplus I^t$. If no stage is judged sufficient, the system selects $t^*$ with the highest sufficiency score to complete the response rather than blindly using the final stage.

Two details are notable: first, LLM insights are constrained by prompts into structured content rather than raw long-chain thinking. Second, the classifier input comes from the SLM distribution, judging "whether this small model is confident given these clues" rather than generic problem difficulty.

### Key Designs
1. **Structured Generation of Four Thinking Clues**:

    - **Function**: Compresses verbose LLM chain-of-thought into high-level guidance that is easier for SLMs to utilize.
    - **Mechanism**: Drawing from cognitive modularity and LLM agent workflows, reasoning is broken into Goal, Planning, Retrieval, and Action. Goal clarifies the objective, Planning provides sub-problem decomposition, Retrieval adds necessary knowledge, and Action provides critical steps. Each effort level covers these categories with increasing depth and token budget.
    - **Design Motivation**: Feeding complete LLM thinking to an SLM is uneconomical and may exceed its comprehension. Structured insights retain the "why" backbone while removing trial-and-error and explanatory redundancy, providing stronger guidance than standard prompts but shorter than full thinking chains.

2. **PPL / Entropy-based Sufficiency Classifier**:

    - **Function**: Determines if current insights are sufficient for the SLM, deciding whether to continue LLM calls.
    - **Mechanism**: When the SLM processes $Q \oplus I^t$, it generates a predictive distribution for each token. The paper calculates the PPL and entropy of each token, followed by statistical features: mean, standard deviation, median, max, min, 25/75 percentiles, and the trend difference between the last 20 and first 20 tokens. The intuition: if clues help the SLM form stable predictions, entropy will be lower and distribution statistics will resemble "solvable" samples.
    - **Design Motivation**: Fixed budgets waste resources on easy tasks and may fail on hard ones. Using the SLM's own uncertainty as a signal binds resource allocation to model capability: the same insight might be sufficient for a strong SLM but insufficient for a weaker one.

3. **Phased Early Exit and Fallback Strategy**:

    - **Function**: Provides a fine-grained trade-off between performance and cost rather than a binary choice between SLM and LLM.
    - **Mechanism**: The LLM generates low-effort insights first; if the classifier deems them sufficient, it stops. If not, it generates medium, then high effort. If all stages fail the threshold, the stage with the highest sufficiency score is used.
    - **Design Motivation**: Many problems only need goal clarification or simple planning; generating action details adds unnecessary cost. Phased design positions Tandem between LLM cascading and full LLM reasoning.

### Loss & Training
Tandem does not train the mentor LLM nor fine-tune the SLM's generation capability. Only the sufficiency classifier needs training: for each problem and effort level in the training set, if the SLM answers correctly given current insights, it is labeled "sufficient"; otherwise, it is "insufficient."

The classifier is a two-layer MLP (hidden layers 64 and 32) with ReLU, 0.3 dropout, and Adam optimizer ($10^{-4}$ learning rate). It is trained for a maximum of 3 epochs with early stopping. Thresholds $\tau^t$ are determined via grid search from 0.05 to 0.95.

Experiment LLMs include DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B, GPT-4o-mini, and gpt-oss-120b. SLMs include DeepSeek-7B and Qwen3-8B. Primary datasets are MATH and GSM8K, with HumanEval for cross-domain code generation transfer.

## Key Experimental Results

### Main Results
The MATH main table shows that with DeepSeek-32B as mentor and DeepSeek-7B as intern, Tandem outperforms both the standalone 32B and fixed high-effort collaboration while significantly reducing TFLOPs cost.

| Method | MATH Avg Acc | Avg Gen Length | Cost (TFLOPs) | Rel. 32B Cost | Key Conclusion |
|------|----------------|--------------|-----------------|---------------|----------|
| 7B Standalone | 77.14 | 2,732 | 38.25 | 22.7% | Low cost, but lacks complex problem capability |
| 32B Standalone | 80.90 | 2,630 | 168.35 | 100% | Strong baseline, but full thinking is expensive |
| 7B+32B low | 78.74 | 2,735 | 44.76 | 26.6% | Guidance too short, limited gain |
| 7B+32B medium | 80.36 | 2,853 | 71.96 | 42.7% | Close to 32B, but still lower |
| 7B+32B high | 83.18 | 2,930 | 104.62 | 62.1% | Fixed long guidance improves accuracy |
| **Tandem** | **83.46** | **2,916** | **99.72** | **59.2%** | Highest accuracy, cost reduced by ~41% |

Cross-family experiments demonstrate that insight formats are not model-specific. DeepSeek-7B reading Qwen3-32B insights achieved 79.96 on MATH, significantly exceeding DeepSeek-7B (76.92) and Qwen3-32B (69.50).

| SLM + LLM | MATH Acc. | MATH Cost | GSM8K Acc. | GSM8K Cost | Observation |
|-----------|-----------|-----------|------------|------------|------|
| Qwen3-8B Standalone | 60.86 | 51.15 | 89.61 | 31.86 | Weak SLM lacks math capability |
| Qwen3-32B Standalone | 69.50 | 193.41 | 94.01 | 104.00 | Qwen3-32B weaker than DeepSeek-R1 at MATH |
| DeepSeek-7B Standalone | 76.92 | 37.25 | 87.11 | 15.74 | Strong math, slightly weaker GSM8K |
| DeepSeek-32B Standalone | 80.76 | 163.84 | 94.47 | 67.14 | Strong mentor baseline |
| DeepSeek-7B + Qwen3-32B | 79.96 | 58.06 | 94.62 | 76.87 | Cross-family insights are usable |
| DeepSeek-7B + DeepSeek-32B | 83.34 | 97.95 | 95.45 | 52.66 | Best when gap is moderate |

### Ablation Study
The paper evaluates Tandem's stability across model sizes, API mentors, cross-domain transfer, and efficiency baselines.

| Analysis Dimension | Setting | Key Result | Implication |
|----------|----------|----------|------|
| Model Scale | DeepSeek Counting & Prob. | 7B+32B high hits 82.49 vs 7B's 75.95; 14B+32B only goes from 79.96 to 80.38 | Gains limited if gap is too small; SLM can't digest guidance if too weak |
| API mentor | DeepSeek-7B + GPT-oss-120B | Algebra 95.79, Counting 86.71, exceeding both standalone models | No weight access needed; suitable for closed APIs |
| Domain Transfer | MATH classifier on HumanEval | Tandem 85.37 vs 7B+32B high 83.54 (LLM: 89.02) | PPL/entropy features are somewhat domain-agnostic |
| Efficiency Baseline | vs Budget Forcing & Cascade | Tandem 83.46 / 99.72 TFLOPs vs Cascade 82.60 / 95.33 | Tandem is more accurate than truncation and more granular than routing |

### Key Findings
- Tandem's primary gain comes from "dynamically selecting guidance depth," not simply making the 32B model write more or less; it improves accuracy while reducing costs even when the fixed high effort baseline is already strong.
- Structured insights are transferable across model families, but the SLM cannot be too weak. A 1.5B model getting 32B guidance cannot approach 32B standalone performance.
- API experiments are practical: GPT-oss-120B as a remote mentor can lower localized costs by outputting short insights for a local 7B model to complete.
- HumanEval results show the classifier learns distribution patterns of "is the SLM stable after reading this guidance" rather than just math heuristics.

## Highlights & Insights
- The paper reframes LLM reasoning efficiency from "how to make LLMs think less" to "what thinking content must be provided by the LLM." This reformulation is vital as it avoids LLM training and is compatible with closed APIs.
- The four insight categories (Goal/Planning/Retrieval/Action) are more controllable than standard hints. They explicitly decompose cognitive modules needed for problem-solving.
- The sufficiency classifier uses the SLM's own distribution features, making it personalized. The same clue may be "sufficient" for different SLMs at different levels, which is more execution-centric than problem-difficulty routing.
- Tandem provides a lightweight paradigm for model collaboration: collaboration does not require multi-round debate; short guidance plus local execution generates high cost-performance.

## Limitations & Future Work
- Primarily covers math and HumanEval; has not fully verified open-domain QA, common sense reasoning, or long-context RAG.
- The sufficiency classifier still requires training data with labels. While cross-domain transfer is good, cold-starting on low-resource tasks remains a challenge.
- Currently a fixed dual-model structure. Complex scenarios might benefit from dynamic selection of mentors or multiple expert models.
- The method assumes insights are reliable. If a mentor LLM provides incorrect Retrieval or Action data, the SLM might be confidently misled; future work could introduce verifiers.

## Related Work & Insights
- **vs Budget Forcing**: Budget Forcing truncates thinking, which might cut critical steps. Tandem changes roles through structured insights.
- **vs LLM Cascade / FrugalGPT**: Cascading usually decides at the query level. Tandem's decision happens between reasoning stages, allowing incremental guidance for a single query.
- **vs Speculative Decoding**: Speculative decoding uses SLMs to accelerate LLM token generation but adheres to LLM output. Tandem lets the SLM generate the answer, yielding different cost savings.
- **vs Multi-agent Debate**: Debate increases communication overhead. Tandem's "communication" is short insights, resembling mentor-execution handover in production systems.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Compressing long reasoning into structured insights using SLM uncertainty for early exit is elegant and addresses cost.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers major datasets and model combinations; open-domain complex tasks need more validation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology and well-structured RQ-based experiments.
- Value: ⭐⭐⭐⭐⭐ High engineering value for reducing LLM reasoning costs and providing a reusable system design for collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](lizard_an_efficient_linearization_framework_for_large_language_models.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)

</div>

<!-- RELATED:END -->
