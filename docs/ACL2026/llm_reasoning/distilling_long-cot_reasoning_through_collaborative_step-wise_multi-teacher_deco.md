---
title: >-
  [Paper Note] Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (CoRD)
description: >-
  [ACL 2026][LLM Reasoning][Multi-teacher distillation] The authors propose CoRD (Collaborative Reasoning Decoding), transforming multi-teacher Long-CoT reasoning distillation from a "generate full trajectories then select…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Multi-teacher distillation"
  - "Long-CoT"
  - "step-wise decoding"
  - "beam search"
  - "predictive perplexity"
date: 2026-05-08
content_hash: 46469cb3fdb937db
---

# Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (CoRD)

**Conference**: ACL 2026  
**arXiv**: [2605.02290](https://arxiv.org/abs/2605.02290)  
**Code**: TBD (not directly provided in the paper)  
**Area**: Model Compression / Distillation / Long-CoT Reasoning  
**Keywords**: Multi-teacher distillation, Long-CoT, step-wise decoding, beam search, predictive perplexity

## TL;DR
The authors propose CoRD (Collaborative Reasoning Decoding), transforming multi-teacher Long-CoT reasoning distillation from a "generate full trajectories then select post-hoc" approach into "step-wise collaborative decoding." In each step, multiple LRMs propose candidate steps, and predictive perplexity from a meta-prover is used for scoring. Combined with beam search to maintain Top-B partial trajectories, the resulting 32B student model exceeds all single teachers on AIME24/25 (79.6 / 70.2 vs 78.9 / 67.9).

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) like DeepSeek-R1 have achieved breakthroughs through test-time scaling and Long-CoT, but their deployment costs are extremely high. Distilling the reasoning capabilities of LRMs into smaller models is a mainstream direction. Representative methods include "curation-based" approaches like S1 and LIMO, where multiple teachers generate complete reasoning trajectories (thousands of tokens) independently, and the highest-scoring one is selected via heuristics as training data.

**Limitations of Prior Work**: Current approaches face three fundamental shortcomings:
1. **PRM / MCTS are unsuitable for Long-CoT**: Process reward models (PRMs) may prematurely prune branches that seem suboptimal but are necessary paths for an "Aha moment"; MCTS search space explodes exponentially on long trajectories.
2. **Curation wastes compute**: Each teacher generates a complete long trace, but only one is kept while others are discarded, and post-hoc selection cannot dynamically adjust the exploration direction.
3. **Lack of collaboration between teachers**: Multiple teachers are sampled independently and maxed, failing to combine complementary strengths (e.g., R1-Qwen's problem formulation vs. Phi4's conclusion synthesis) into a superior trajectory that no single teacher could achieve alone.

**Key Challenge**: "Aha moments" in Long-CoT reasoning emerge dynamically. Post-hoc curation eliminates the possibility of cross-teacher temporal stitching, such as combining a weak step from teacher A with a strong reflection from teacher B at the next step.

**Goal**: To enable multiple teachers to make collaborative decisions at **every step**, treating the reasoning process itself (rather than the full trajectory) as the minimum unit for distillation.

**Key Insight**: Analogize the reasoning process to auto-regressive decoding—each step is a "token," the set of steps proposed by teachers is the "decoding vocabulary," and beam search can be used for exploration at the step level.

**Core Idea**: Use (i) prompt-guided step segmentation to align step boundaries across different LRMs, (ii) predictive perplexity to evaluate the "predictability of the correct answer given the current prefix" as a short-term quality signal, and (iii) beam search to maintain Top-B partial trajectories at the step level to avoid greedy myopia.

## Method

### Overall Architecture

Formalization: For a problem $x$ and $K$ teacher LRMs $\mathcal{T}$, traditional curation is $\tau(x_i)^* = \arg\max_{\tau^{(k)}} Q(x_i, \tau^{(k)})$ (selecting the max among $K$ complete trajectories). CoRD changes this to a step-wise approach:

$$\tau(x_i)^* = \{(s_1^*, \dots, s_T^*) \mid s_t^* = \arg\max_{s_t \in \{s_t^{(1)}, \dots, s_t^{(K)}\}} S(\tau_{<t} \oplus s_t^{(k)})\}$$

In each step, each teacher proposes a candidate step $s_t^{(k)}$ conditioned on a shared prefix $\tau_{<t}$, and the best one is selected by a scoring function $S(\cdot)$. This is "step-wise autoregressive decoding" where the decoding vocabulary is the set of teacher proposals.

### Key Designs

1. **Prompt-guided step segmentation**:
    - **Function**: Semantically align the Long-CoT outputs of different LRMs at the "step unit" to facilitate cross-model comparison.
    - **Mechanism**: Embed a `<think> ### Step` template in the prompt to guide the LRM to actively output in the format "### Step 1. Understanding... ### Step 2. Recalling...". This ensures that shallow markers like `\n\n` or `wait` fall within steps rather than at boundaries, preventing mis-segmentation. Compared to line-break units or prefix units (which vary by model), prompt-guided segmentation offers the highest fairness.
    - **Design Motivation**: Different LRMs have vast differences in "line-break habits" and "reflection cue" frequency. Simply cutting by physical markers could result in some steps having dozens of tokens while others have hundreds, making horizontal comparison impossible. Prompt-guided segmentation gives control back to the generating LRM, forcing "logical functional" division where each step is a "sub-task" (e.g., problem understanding / theorem recall), thereby supporting cross-teacher step replacement.

2. **Predictive perplexity step selection**:
    - **Function**: Given the current prefix and candidate step, estimate "the model's ability to predict the correct answer after adding this step."
    - **Mechanism**: Introduce an independent meta-prover (QwQ-32B is used in experiments) to calculate $S(\tau_{<t} \oplus s_t^{(k)}) = \exp(\frac{1}{M} \log p_{\text{meta}}(A \mid \tau_{<t} \oplus s_t^{(k)}))$, where $A$ is the ground-truth answer sequence and $M$ is the number of answer tokens. This represents the average conditional probability per answer token by the meta-prover, normalized to $[0, 1]$. This is a forward-looking signal—it evaluates how much a step makes the final answer more predictable, rather than just how "reasonable" the step is locally.
    - **Design Motivation**: Compared to PRM (which looks at local correctness and might prune "seemingly wrong but self-correcting" paths) or binary judgment (which is too sparse), predictive perplexity is (i) a bounded continuous score capturing fine-grained quality differences; (ii) globally informed by the meta-prover's likelihood of the answer; and (iii) does not require training an additional reward model. In experiments, perplexity improved accuracy on AIME24 from 75.0 (PRM) and 77.7 (binary judgment) to 79.6.

3. **Beam search step-wise decoding**:
    - **Function**: Maintain Top-B partial trajectories at the step level to avoid greedy myopia and premature commitment to suboptimal branches.
    - **Mechanism**: Starting from step $t$ with beam $\mathcal{B}_{t-1} = \{\tau_{<t}^{(b)}\}_{b=1}^B$, each teacher proposes a candidate step for each prefix, resulting in $B \times K$ candidates $\mathcal{C}_t$. The Top-$B$ are selected based on predictive perplexity to form $\mathcal{B}_t$. The complexity is $\mathcal{O}(TKMB)$, lower than MCTS's $\mathcal{O}(TK \log(TMB))$, but $B$ times higher than greedy search ($B=1$). $B=4$ is used in experiments.
    - **Design Motivation**: In Long-CoT, "strategic shifts" and "self-corrections" often appear suboptimal at one step but show strength a few steps later; greedy search would lose these. Beam search is the sweet spot—preserving 4 candidates allows for exploration without losing control. Analysis (Fig 5) shows that while MCTS tends to favor a globally strong teacher, beam search allows R1-Qwen-32B to excel in the early phase (formulation) and Phi4-Reasoning-Plus to excel in the late phase (synthesis).

### Loss & Training
The student model is trained using pure SFT. Teacher pool: QwQ-32B + R1-Distill-Qwen-32B + Phi4-Reasoning-Plus (heterogeneous) or a single QwQ-32B with different temperatures (homogeneous). Meta-prover: QwQ-32B. Beam width $B = 4$. Datasets: LIMO-v1 (817), S1k-1.1 (1000), LIMO-v2 (800). Students: R1-Qwen-7B/14B/32B. Training: 8×H100, bs=8, 5 epochs, lr=5e-6, max seq=20480, DeepSpeed Stage-3. Max output = 20,480 tokens.

## Key Experimental Results

### Main Results: AIME24/25 Student Pass@1 (Heterogeneous teachers)

| Model / Method | AIME24 | AIME25 |
| :--- | :--- | :--- |
| Teacher: R1-Qwen-32B | 71.6 | 53.8 |
| Teacher: QwQ-32B | 77.9 | 66.7 |
| Teacher: Phi4-Reasoning-Plus | 78.9 | 67.9 |
| Student R1-Qwen-32B w/o distill | 71.6 | 53.8 |
| Student 32B + Curation-Hetero | 75.0 | 62.1 |
| Student 32B + Integration-Hetero | 12.7 | 9.0 |
| **Student 32B + CoRD-Hetero** | **79.6** | **70.2** |
| Student 7B + Curation-Hetero | 56.6 | 42.1 |
| **Student 7B + CoRD-Hetero** | **60.8** | **45.6** |
| Student 14B + CoRD-Hetero | **74.8** | **62.3** |

The 32B student distilled via CoRD **surpasses** the strongest teacher (Phi4-Reasoning-Plus) on both benchmarks, proving that collaborative step-wise composition generates trajectories that teachers cannot achieve independently. The Integration baseline (where GPT-4o-mini merges trajectories) performs poorly (9-12 points) as it compresses Long-CoT into short-form, losing supervision signals.

### Ablation Study

**(a) Step segmentation (Heterogeneous, R1-Qwen-32B student)**

| Method | Acc | PP | AIME24 | AIME25 |
| :--- | :--- | :--- | :--- | :--- |
| Line-break | 88.4 | 0.734 | 76.7 | 67.7 |
| Prefix | 91.3 | 0.747 | 77.1 | 67.3 |
| **Prompt-guide** | **93.1** | **0.774** | **79.6** | **70.2** |

**(b) Step selection criterion**

| Method | Acc | PP | AIME24 | AIME25 |
| :--- | :--- | :--- | :--- | :--- |
| Random | 80.4 | 0.494 | 69.0 | 61.9 |
| Max-length | 80.0 | 0.502 | 68.8 | 59.0 |
| PRM (Qwen2.5-Math-PRM-72B) | 82.6 | 0.591 | 75.0 | 64.6 |
| Binary Judgment (LLM) | 91.7 | 0.626 | 77.7 | 66.3 |
| **Predictive Perplexity** | **93.1** | **0.774** | **79.6** | **70.2** |

**(c) Decoding strategy**

| Method | Acc | PP | AIME24 | AIME25 | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy ($B=1$) | 81.6 | 0.719 | 76.7 | 66.5 | – |
| MCTS | 89.6 | 0.755 | 75.8 | 66.3 | 589.2 |
| **Beam Search ($B=4$)** | **93.1** | **0.774** | **79.6** | **70.2** | **288.7** |
| Curation Baseline | 84.8 | 0.652 | 75.0 | 62.1 | 168.3 |
| Curation×2 (Equal Compute) | 90.3 | 0.712 | 74.6 | 63.8 | 336.6 |

### Key Findings
- **CoRD 32B Student Surpasses All 32B Teachers**: 79.6 vs Phi4's 78.9 (AIME24); 70.2 vs Phi4's 67.9 (AIME25). Reasonings are distilled that teachers could not perform.
- **Predictive perplexity correlates strongly with student performance, while answer accuracy is unreliable**: The Integration baseline had 91.2% answer accuracy during reasoning but only 0.223 perplexity (indicating compressed trajectories), resulting in a student score of only 12.7. This proves that the quality of the "reasoning process" is key; final answer correctness alone can be misleading.
- **Heterogeneous > Homogeneous**: Heterogeneous teachers improved the 32B student from 75.8 to 79.6 on AIME24. Diversity stems from architecture rather than sampling temperature.
- **Teacher Specialization Automatically Emerges**: Under beam search, R1-Qwen-32B/QwQ-32B dominate the early phase (≤40% progress, problem formulation), while Phi4-Reasoning-Plus dominates the late phase (≥80%, conclusion synthesis). MCTS tends to collapse to the globally strongest teacher.
- **Curation cannot catch up even with equal compute**: Curation×2 (336.6s vs CoRD 288.7s) yielded only 74.6 / 63.8, significantly lower than CoRD, proving step-wise composition is indispensable.
- **Generalization beyond AIME**: MATH500 94.8, TaTQA 95.2 (tabular reasoning, OOD), PubMedQA 91.8 (biomedical QA).
- **8B Students Benefit**: R1-Llama-8B + CoRD-Hetero reached 54.0 on AIME24, proving the method works for non-Qwen families.

## Highlights & Insights
- **Conceptual Shift to "Reasoning as Tokens"**: Traditional KD is at the token level, curation is at the trajectory level; CoRD operates at the step level—adjusting the grain size of reasoning to allow cross-model swapping and "teacher collaborative synthesis."
- **Predictive Perplexity as an Underrated Reward Signal**: It is forward-looking. Instead of judging if a step is locally "correct," it evaluates if the correct answer is more predictable after that step. This naturally accommodates "Aha" paths that might look wrong initially but lead to better outcomes, avoiding the pitfalls of PRM.
- **Transferable Trick of Prompt-guided Segmentation**: Using `### Step N.` templates to control LRM output boundaries is a zero-cost standardization tool applicable to any multi-model collaboration or step-level evaluation scenario.
- **Emergent Specialization**: Teachers showing their strengths in different phases emerges naturally from predictive perplexity scoring and beam-level diversity preservation without manual division-of-labor prompts. This resembles MoE but is realized at inference-time.
- **MCTS Limitations in Long-CoT**: MCTS tends to favor the globally strongest teacher because trajectory-level rewards cause the search to converge to the "overall best" branch, losing local complementarity. This is a warning for Long-CoT search design—wrong granularity loses collaboration.

## Limitations & Future Work
- **Domain Scope**: Validated only on Math/English; multi-lingual reasoning distillation remains unknown.
- **Training Strategy**: Used only SFT; step-level preference data generated by CoRD could naturally be used for step-DPO.
- **Meta-prover Dependency**: Performance drops significantly with a weak meta-prover (70.2 to 53.2 on AIME25), indicating reliance on having at least one strong model in the pool.
- **Efficiency**: At 288.7s per problem (70% slower than curation), cost is high for large datasets.
- **Token-level Comparison**: Performance relative to pure token-level logit matching (white-box KD) was not compared.
- **Future Directions**: (i) Distilling lighter meta-provers; (ii) extending to multi-modal/agent/code reasoning; (iii) step-level preference learning.

## Related Work & Insights
- **vs S1 / LIMO**: CoRD consistently outperforms these curation-based datasets on AIME benchmarks, proving step-wise quality exceeds manual/post-hoc curation.
- **vs PRM-based RL**: While PRM evaluates local correctness, CoRD evaluates predictive power for the final answer, which is better for the "self-correction" nature of Long-CoT.
- **vs MCTS-based Reasoning (rStar, ToT)**: CoRD is twice as fast and more effective because perplexity implicitly accounts for "future informedness" without explicit rollouts.
- **vs Mixture-of-Agents**: CoRD operates at the step level rather than response level, providing finer granularity and trajectory-level supervision signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Redefining distillation as collaborative decoding)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Broad benchmarks, ablation, and generalization)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear formulas and engineering references)
- **Value**: ⭐⭐⭐⭐⭐ (Students surpassing teachers is a significant result in KD)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)

</div>

<!-- RELATED:END -->
