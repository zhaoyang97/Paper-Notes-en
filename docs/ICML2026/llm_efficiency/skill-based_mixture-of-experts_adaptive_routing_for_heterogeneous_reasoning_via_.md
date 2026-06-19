---
title: >-
  [Paper Note] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills
description: >-
  [ICML 2026][LLM Efficiency][Paper Note] SKILL-MOE proposes a training-free symbolic MoE framework that uses "skills" as routing signals. It extracts required skills for each question, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and merges multiple CoT solutions into a final answer using a task-level optimal aggreg
tags:
  - ICML 2026
  - LLM Efficiency
date: 2026-05-08
content_hash: e77f68bc16bbb9b4
---
# Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills

**Conference**: ICML2026  
**arXiv**: [2503.05641](https://arxiv.org/abs/2503.05641)  
**Code**: https://github.com/dinobby/Skill-MoE (Available)  
**Area**: LLM Efficiency / Mixture-of-Experts / Multi-agent Reasoning  
**Keywords**: Symbolic MoE, Skill Routing, Instance-level Expert Selection, Aggregator Selection, Batch Inference

## TL;DR
SKILL-MOE proposes a training-free symbolic MoE framework that uses "skills" as routing signals. It extracts required skills for each question, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and merges multiple CoT solutions into a final answer using a task-level optimal aggregator. Coupled with expert-bucketed batch inference, it allows 16 7-8B models to run on a single GPU, outperforming the strongest multi-agent baselines by 8.15% on average.

## Background & Motivation

**Background**: Current approaches for collaborative reasoning using multiple pre-trained LLMs primarily follow two paths: multi-agent debate (Debate / ReConcile / MoA / Self-MoA), which uses a fixed set of models for multi-round discussion; or training MoE into a single large model where experts are parameter subsets requiring end-to-end joint training. The former binds "which models to use" at the task level, while the latter cannot directly leverage existing LLM pools.

**Limitations of Prior Work**: Task-level model selection is too coarse-grained—even within mathematics, an algebra problem and a probability problem require different experts. Multi-round discussions are also extremely expensive, requiring 6-9 LLM calls per instance. Furthermore, scaling the candidate pool to 16 7-8B models would normally require one GPU per model, making deployment impractical.

**Key Challenge**: The need to balance "instance-level dynamic expert recruitment for fine-grained capability matching" with "running a large heterogeneous model pool on a single GPU." Fixed expert sets sacrifice granularity, while naive dynamic scheduling suffers from high latency due to frequent model loading/unloading.

**Goal**:  
(1) Design a gradient-free routing mechanism to select experts at the instance level based on skills.  
(2) Design an inference scheduling strategy that enables 16 7-8B models to achieve throughput comparable to a 4-GPU MoA setup on a single GPU.  
(3) Solve the problem of aggregator selection and determine if multi-round discussions can be bypassed.

**Key Insight**: Rather than training a router in parameter space, LLMs can exchange information via "natural language" as a common protocol. A lightweight "skill vector"—the cumulative score of each model on each skill—serves as a symbolic router. Skill descriptions can be inferred from questions via keywords or aligned between test samples and profiles using Sentence-BERT.

**Core Idea**: Shift MoE routing from "hidden states" to "discrete skills" and replace "parameter subsets" with "complete pre-trained LLMs." Use expert-bucketed batch inference to make dynamic recruitment feasible on a single GPU.

## Method

### Overall Architecture
SKILL-MOE manages 16 independently trained 7-8B heterogeneous LLMs. It dynamically selects the most suitable experts for each reasoning problem and merges their solutions into a final answer, all without parameter training and within a single GPU. The process has two phases. In the pre-processing phase, offline statistics are gathered from ~350 validation samples: Qwen2.5-7B-Instruct acts as a "Keyword LLM" to extract required skills for each problem; the 16 models generate CoT solutions, receiving +1 for a correct answer and −1 for an incorrect one per involved skill to build a skill profile $P_i$ for each model $M_i$ (e.g., $\{\text{Algebra}: 10, \text{Biology}: 3, \text{Chemistry}: -6, \dots\}$). Simultaneously, a "synthetic task" of selecting the correct CoT from three candidates is used to identify the best task-level aggregator $A^*$. During inference, skills are extracted from test samples, aligned with profile skills via Sentence-BERT cosine similarity, and $k=3$ experts are sampled based on matching scores. Finally, $A^*$ merges the $k$ CoT outputs—supported by a "bucket-based batch inference" system that makes this dynamic recruitment efficient on a single card.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph PRE["Pre-processing (~350 validation samples, offline, gradient-free)"]
        direction TB
        P1["Keyword LLM extracts skills<br/>+ 16 LLMs run CoT<br/>Correct: skill +1 / Incorrect: −1"]
        P3["Skill Profile P_i<br/>Skill score dictionary for each model"]
        P4["Synthetic Task: Pick correct CoT from 3<br/>Select task-level aggregator A*"]
        P1 --> P3
    end
    Q["Batch of test samples"] --> R
    P3 --> R
    R["Skill Profile + Local/Global Weighted Routing<br/>Sentence-BERT alignment → softmax sampling k=3 experts"]
    R --> BATCH["Expert-bucketed Batch Inference Scheduling<br/>Group samples by expert; load model once per batch"]
    BATCH --> E["k experts each generate 1 CoT"]
    E --> AGG["Task-level Aggregator A* merges k CoTs"]
    P4 --> AGG
    AGG --> OUT["Final Answer"]
```

### Key Designs

**1. Skill Profile + Local/Global Weighted Instance-level Routing: Matching the right model to the right problem**  
Task-level selection is too coarse. SKILL-MOE routes at the instance level: for a question $q$, it extracts a set of skills $K_q$. The "local adaptation score" for model $M_i$ is the sum of its scores for those skills: $S(M_i, q) = \sum_{k_j \in K_q} s^{(i)}_{k_j}$. To prevent weak models with lucky high scores on niche skills from being selected, this is multiplied by a "global competence" $\gamma_i$—the ratio of the model's total profile score to the pool's total score. The final relevance score $w^{(i)}_q = \gamma_i \cdot S(M_i, q)$ is processed via softmax (temperature 0.5) to sample $k$ experts with replacement, filtering out experts appearing in $<5\%$ of the test set to avoid noise. This balances "relative advantage on the specific sample" with "overall strength on the task." Ablation studies on GPQA show that Top-3 / Top-5 / Random fixed experts achieve only 52.86% / 47.68% / 42.61%, whereas instance-level routing reaches 57.78% (Table 5).

**2. Task-level Aggregator Selection: Using a fixed "best judge" rather than instance-level switching or majority voting**  
After selecting experts, $k$ heterogeneous CoTs must be merged. SKILL-MOE creates synthetic tasks (1 correct CoT + 2 incorrect CoTs) on the validation set for each question. Candidate models act as aggregators to identify the correct answer, and the one with the highest hit rate is chosen as $A^*$. Inference follows $y = A^*(\bigoplus_{i=1}^k y_0^{(i)})$, where $\oplus$ denotes concatenation. A counter-intuitive finding is that "reasoning ability $\neq$ aggregation ability"—a random aggregator on MMLU-Pro yields 52.29%, an instance-level adaptive aggregator 57.12%, while a task-specific aggregator reaches 63.71% (Table 3). Crucially, Table 7 shows that once a good aggregator is selected, adding 3 rounds of discussion provides almost zero gain (63.83 vs. 63.71), allowing for the removal of expensive multi-round interactions.

**3. Expert-bucketed Batch Inference Scheduling: Turning dynamic recruitment back into static loading**  
Dynamic recruitment has a high system cost: adjacent samples may require entirely different model sets. Naive instance-by-instance scheduling leads to constant loading/offloading, taking 196.92 s/sample on GPQA. SKILL-MOE pre-calculates routing for all samples in a batch, then groups all samples requiring the same expert together. Each activated expert is loaded only once per batch. This reduces single-GPU latency to 25.76 s/sample—44% lower than 1-GPU MoA (45.98 s) and comparable to 4-GPU MoA (21.66 s).

### Loss & Training
Completely gradient-free. All experts, aggregators, and Keyword LLMs are frozen pre-trained LLMs. "Training" consists only of calculating skill scores and aggregation hit rates on ~350 validation samples. Each test instance involves $k=3$ experts + 1 aggregator, totaling 4 LLM calls—comparable to Self-Consistency $\times 5$ and more efficient than MoA (6 calls) or ReConcile (9 calls).

## Key Experimental Results

### Main Results
Evaluation was conducted on four heterogeneous reasoning datasets: MMLU-Pro, AIME 2024, GPQA Diamond, and MedMCQA. The model pool consists of 16 LLMs (3.5B–12B), mostly in the 7-8B range.

| Dataset | Metric | SKILL-MOE | Strongest Multi-agent Baseline | Gain |
|--------|------|-----------|------------------|------|
| AIME 2024 | Acc. | 68.88 | 55.56 (Self-MoA) | +13.32 |
| MMLU-Pro | Acc. | 63.71 | 61.78 (MoA) | +1.93 |
| MedMCQA | Acc. | 59.35 | 60.74 (ReConcile) | −1.39 |
| GPQA Diamond | Acc. | 57.78 | 52.86 (MoA / Self-MoA) | +4.92 |
| **Average** | Acc. | **62.43** | 54.28 (Strongest baseline avg) | **+8.15** |

SKILL-MOE is robust across datasets; its average score exceeds Qwen2.5-72B (54.28) and Llama3.3-70B (53.18), and it is more stable than QwenR1-32B (56.94), which excels on AIME (76.67%) but fails on MedMCQA (24.70%).

### Ablation Study

| Configuration | Key Metric (GPQA) | Description |
|------|----------------|------|
| Full SKILL-MOE | 57.78 | Skill profile routing + task-level aggregator |
| Random Aggregator + Recruited Experts | 51.52 | Aggregator quality is critical |
| Task-specific Aggregator + Random Experts | 31.82 | Weak experts drag down strong aggregators |
| Majority Vote + Recruited Experts | 53.54 | Majority vote as a fallback without aggregator |
| Top-3 Fixed Experts | 52.86 | Task-level coarse selection vs. instance-level (−4.92) |
| Adaptive Aggregator (MMLU-Pro) | 57.12 | Switching aggregators per instance loses 6.59 |
| Task Aggregator + 3-round Discussion | 63.83 (MMLU-Pro) | Discussion adds negligible gain |

### Key Findings
- **Synergy between selection and aggregation**: Neither component alone is sufficient; both must be optimized to reach 57.78% performance.
- **Reasoning $\neq$ Aggregation**: Models specialized in judged task-specific CoTs perform better than general reasoning models for merging answers.
- **Strong cross-domain generalization**: A skill profile built on MMLU-Pro performs 14.81% better than the Debate baseline when transferred to OmniMATH. Skills are more transfer-friendly than "task-model" mappings.

## Highlights & Insights
- **Symbolic vs. Hidden State Routing**: Replacing neural routers with "skill-dictionary weighted sampling" allows independent pre-trained LLMs to be used in an MoE framework without joint training.
- **Systems-Algorithm Co-design**: Bucket-based batch inference solves the practical bottleneck of "dynamic model loading," enabling 16 7-8B models to fit on one GPU.
- **De-emphasizing Multi-round Debate**: The paper suggests that multi-agent debate gains often come from an implicit good aggregator; once an explicit optimal aggregator is found, multi-round costs can be eliminated.

## Limitations & Future Work
- **Validation Set Dependency**: Skill profiles and aggregator rankings are derived from ~350 samples, making the strategy sensitive to validation set bias.
- **Batching Requirement**: The efficiency gains rely on samples arriving in batches, which is less suited for real-time, single-query stream applications.
- **Specialized Domains**: On MedMCQA, SKILL-MOE slightly underperformed ReConcile, suggesting that for narrow, highly specialized domains with sparse expert coverage, single-round aggregation may be insufficient compared to debate.

## Related Work & Insights
- **vs MoA / Self-MoA**: MoA uses fixed task-level top-k models and multi-round discussion. SKILL-MOE switches experts per sample, removes discussion, reduces calls from 6-9 to 4, and improves average accuracy by 8.15%.
- **vs Traditional Sparse MoE**: Traditional MoE uses parameter subsets and end-to-end training. SKILL-MOE treats complete models as experts and uses natural language communication, making the expert pool "hot-pluggable."

## Rating
- Novelty: ⭐⭐⭐⭐ Shifting MoE routing to a symbolic skill space for full LLMs is a clear and effective paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive benchmarks across 4 datasets, 8 baselines, and comprehensive ablations on cross-domain generalization and efficiency.
- Writing Quality: ⭐⭐⭐⭐ Clear frameworks and precise algorithms, though some notation varies slightly across sections.
- Value: ⭐⭐⭐⭐⭐ Highly practical for researchers with limited GPU resources to leverage large heterogeneous model ensembles.

## Related Papers

- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2025\] Cooperation of Experts: Fusing Heterogeneous Information with Large Margin](../../ICML2025/llm_efficiency/cooperation_of_experts_fusing_heterogeneous_information_with_large_margin.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2026\] KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)
- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)

</div>

<!-- RELATED:END -->
