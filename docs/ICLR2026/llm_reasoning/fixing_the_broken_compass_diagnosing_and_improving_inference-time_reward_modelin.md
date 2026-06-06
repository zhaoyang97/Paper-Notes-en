---
title: >-
  [Paper Note] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling
description: >-
  [ICLR 2026][LLM Reasoning][Reward Model] This paper systematically diagnoses three failure modes of inference-time reward models (RMs)—performance degradation on easy problems…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Reward Model"
  - "inference-time scaling"
  - "CRISP"
  - "Best-of-N"
  - "MCTS"
date: 2026-05-08
content_hash: eed8f4be17bdee5d
---

# Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling

**Conference**: ICLR 2026
**arXiv**: [2503.05188](https://arxiv.org/abs/2503.05188)  
**Code**: [GitHub](https://github.com/BugMakerzzz/CRISP)  
**Area**: LLM Reasoning
**Keywords**: [Reward Model, inference-time scaling, CRISP, Best-of-N, MCTS]

## TL;DR

This paper systematically diagnoses three failure modes of inference-time reward models (RMs)—performance degradation on easy problems, diminished discriminability as sample size increases, and accuracy loss under high search diversity—and proposes CRISP, an algorithm that mitigates these issues via answer-clustering-based reward aggregation and stepwise prefix guidance, achieving accuracy improvements of up to 5%.

## Background & Motivation

Inference-time scaling techniques (e.g., OpenAI o1, DeepSeek-R1) enhance LLM reasoning by increasing test-time compute. Current research focuses primarily on training-time optimization (RL/SFT), while inference-time reward-model-based methods remain comparatively underexplored. Meanwhile, R1-series models suffer from overthinking and limited task generalization.

Taking CSQA commonsense reasoning as an example: DeepSeek-R1-7B achieves an accuracy of 64.8 with an average of 3,613 tokens, whereas the inference-time method proposed in this paper reaches 72.0 on the base model Qwen2.5-Math-7B using only 1,100 tokens. This demonstrates that optimizing inference-time RMs remains a critical direction.

However, preliminary experiments show that advanced RMs yield limited improvements on downstream reasoning tasks: BoN improves over Self-Consistency (SC) by less than 5% on most LLMs, while the Oracle (directly recalling correct answers from samples) far outperforms other methods, indicating that **the bottleneck lies in the RM's discriminative capacity rather than the LLM's generative ability**.

## Method

### Overall Architecture

This paper first models the RM inference process as a function of three components: input question $q$, sample count $n$, and search parameters $\Phi$. RM behavior is systematically probed by fixing two of these while varying the third. After diagnosing three major failure modes, the paper proposes CRISP (Clustered Reward Integration with Stepwise Prefixing), an iterative framework comprising five modules.

### Key Designs

1. **Systematic Diagnosis of Three RM Failure Modes**: Function—conducting controlled experimental analyses of key factors affecting RM inference performance. Mechanism—(Cl.1) Problems are partitioned into 5 difficulty levels by pass@1; BoN/MCTS-RM is found to underperform Self-Consistency on easy problems (Levels 1–2). (Cl.2) Tracking the frequency of RM-highest-scored incorrect answers reveals an "inverse long-tail" phenomenon: low-frequency incorrect answers (appearing $<5$ times) are more likely to receive high scores, as larger $n$ introduces more out-of-distribution low-frequency samples that degrade discriminability. (Cl.3) Increasing temperature or MCTS width/depth (search diversity) consistently degrades RM performance, with moderate diversity being optimal. Design Motivation—Understanding the specific failure mechanisms of RMs at inference time, thereby providing principled guidance for targeted algorithm design.

2. **CRISP: Clustered Reward Aggregation with Stepwise Prefix Iteration**: Function—designing a five-module iterative framework to specifically mitigate the three identified RM failure modes. Mechanism—(a) **Path Generation**: complete reasoning paths are generated based on a prefix set $\mathcal{P}$ (rather than step-by-step as in MCTS), controlling search diversity (addressing Cl.3); (b) **State Aggregation**: paths are clustered by final answer $\psi: \mathcal{R} \to \mathcal{C}$ (addressing the low-frequency error issue in Cl.2); (c) **Reward Evaluation**: rewards are aggregated at the cluster level $\mathcal{F}(\mathcal{C}_j) = \sum_{x \in \mathcal{C}_j} f(x)$, incorporating frequency information to prevent low-frequency errors from receiving high scores; (d) **Early Termination**: when the cluster count $<2$, the problem is deemed easy and SC is applied directly (addressing Cl.1); (e) **Prefix Extraction**: the first $i$ steps of the best path from the highest-scoring cluster are extracted as the prefix for the next iteration, progressively narrowing the search space. Design Motivation—Each module is designed to address a specific diagnostic finding, forming a systematic solution.

### Loss & Training

CRISP is a purely inference-time method requiring no training. Policy models used are Qwen2.5-3B and Llama3.1-8B; reward models used are Skywork ORM and Skywork-o1 PRM. BoN uniformly samples $n=32$; MCTS uses 32 rollouts. Temperature and the number of iterations are tunable hyperparameters.

## Key Experimental Results

### Main Results

| Method | Qwen2.5-3B GSM8K | Qwen2.5-3B MATH | Qwen2.5-3B Olympiad | Llama3.1-8B MATH |
|:---|:---|:---|:---|:---|
| CoT | 0.78 | 0.46 | 0.24 | 0.38 |
| Self-Consistency | 0.83 | 0.64 | 0.31 | 0.57 |
| BoN + PRM | 0.87 | 0.61 | 0.34 | 0.62 |
| MCTS + PRM | 0.95 | 0.71 | 0.31 | 0.57 |
| Beam Search | 0.95 | 0.73 | 0.34 | 0.56 |
| **CRISP + PRM** | **0.96** | **0.76** | **0.39** | **0.67** |

### Ablation Study

| Comparison with R1 Models | MATH Acc / Tokens | CSQA Acc / Tokens | SIQA Acc / Tokens | LogiQA Acc / Tokens |
|:---|:---|:---|:---|:---|
| Qwen2.5-Math-7B Chat | 0.74 / 1855 | 0.58 / 1479 | 0.58 / 1388 | 0.49 / 2133 |
| R1-Distill-7B | **0.88** / 9626 | 0.65 / 3612 | 0.66 / 2920 | 0.50 / 6492 |
| **CRISP** | 0.79 / **987** | **0.72** / **1100** | **0.66** / **1059** | **0.59** / **2058** |

### Key Findings

- CRISP achieves up to 5% improvement on MATH-500 (Llama3.1-8B: 0.62 → 0.67) and 5% on OlympiadBench.
- Compared to R1 models: average accuracy on non-mathematical tasks is 10% higher, with token consumption reduced by up to 90%.
- Ablation experiments confirm that each module contributes independently: removing clustering aggregation, early termination, or prefix guidance all lead to performance degradation.
- CRISP is robust across different RMs: even with the weaker Shepherd PRM (BoN at only 0.47), high accuracy is maintained.
- Inference time: CRISP 91.0s vs. MCTS 211.3s vs. Beam Search 268.7s, demonstrating substantially greater efficiency.

## Highlights & Insights

- The three diagnostic findings are systematic and insightful; in particular, the "inverse long-tail" phenomenon—where RMs assign high scores to low-frequency incorrect answers—represents an important behavioral insight into RMs.
- Cluster-level reward aggregation is an elegant design that naturally incorporates frequency information into scoring without modifying the RM itself.
- The early termination mechanism cleanly resolves the issue of RMs being counterproductive on easy problems.
- The underperformance of R1 models on non-mathematical tasks (compounded by high token costs) underscores the sustained value of inference-time optimization.

## Limitations & Future Work

- Clustering relies on exact matching of final answers, making it not directly applicable to open-ended generation tasks such as text summarization.
- The number of steps extracted for prefix guidance grows linearly with iteration count, which may be overly constraining for long reasoning chains.
- Validation is limited to mathematical and commonsense reasoning; more complex multi-step reasoning tasks (e.g., programming, planning) remain unexplored.
- The early termination threshold (cluster count $<2$) is hard-coded and may require adjustment for different tasks.
- The possibility of iteratively improving the RM itself through its own reasoning process during inference is not considered.

## Related Work & Insights

- BoN Weighted (Snell et al., 2024) and MCTS (Hao et al., 2023) are the primary competing methods; CRISP builds upon both by introducing clustering and prefix mechanisms.
- DeepSeek-R1 avoids RM reward hacking through rule-based rewards, while this paper mitigates RM discriminability deficiencies from an inference-time perspective.
- The "inverse long-tail" phenomenon may have implications for the distributional design of RM training data—greater coverage of low-frequency error patterns is needed.
- The idea of cluster-level aggregation is generalizable to other multi-candidate scoring scenarios, such as test-case aggregation in code generation.

## Rating

⭐⭐⭐⭐ — The problem diagnosis is systematic and thorough; CRISP is purposefully designed and experimentally well-validated, offering significant practical value for inference-time optimization. However, its applicability is constrained by the requirement for exact answer matching, limiting generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](../../NeurIPS2025/llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ICLR 2026\] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](../../AAAI2026/llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)

</div>

<!-- RELATED:END -->
