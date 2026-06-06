---
title: >-
  [Paper Note] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills
description: >-
  [ICML2026][LLM Efficiency][Symbolic MoE] SKILL-MOE proposes a training-free, symbolic MoE framework that uses "skills" as routing signals: it extracts required skills from each problem…
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "Symbolic MoE"
  - "skill routing"
  - "instance-level expert selection"
  - "aggregator selection"
  - "batch inference"
date: 2026-05-08
content_hash: 06245ec0782e0cd7
---

# Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills

**Conference**: ICML2026  
**arXiv**: [2503.05641](https://arxiv.org/abs/2503.05641)  
**Code**: https://github.com/dinobby/Skill-MoE (Available)  
**Area**: LLM Efficiency / Mixture-of-Experts / Multi-agent Reasoning  
**Keywords**: Symbolic MoE, skill routing, instance-level expert selection, aggregator selection, batch inference

## TL;DR
SKILL-MOE proposes a training-free, symbolic MoE framework that uses "skills" as routing signals: it extracts required skills from each problem, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and merges multiple CoT paths into a final answer using a task-level optimal aggregator. Coupled with expert-wise batch inference, it enables running 16 7-8B models on a single GPU, achieving an average accuracy 8.15% higher than the strongest multi-agent baselines.

## Background & Motivation

**Background**: Current approaches for solving reasoning problems using multiple pre-trained LLMs primarily follow two paths: multi-agent debate (Debate / ReConcile / MoA / Self-MoA), which uses a fixed set of models for multi-round discussion; or integrating MoE into a single large model, where experts are parameter subsets requiring end-to-end joint training. The former ties "which models to use" to the task level, while the latter cannot directly reuse existing LLM pools.

**Limitations of Prior Work**: Task-level model selection is too coarse—two math problems might require different experts if one involves algebra and the other probability. Furthermore, multi-round discussion is extremely expensive, requiring 6–9 LLM calls per sample. Deploying a candidate pool of 16 7-8B models would typically require one GPU per model, making it unfeasible for standard setups.

**Key Challenge**: To achieve a balance between "instance-level dynamic expert recruitment for fine-grained capability matching" and "supporting a large heterogeneous model pool on a single GPU." Fixed expert sets sacrifice granularity, while naive dynamic scheduling suffers from high latency due to frequent model loading/unloading.

**Goal**:
(1) Design a training-free routing mechanism that selects experts at the instance level based on skills.
(2) Design an inference scheduling strategy that allows 16 7-8B models to run on a single GPU with throughput comparable to 4-GPU MoA.
(3) Determine how to select the optimal aggregator and whether multi-round discussions can be eliminated.

**Key Insight**: Rather than training a router in the parameter space, LLMs can exchange information through the common protocol of "natural language." A lightweight "skill vector"—the cumulative score of each model across various skills—can then serve as a symbolic router. Skill descriptions can be inferred from problems via a "Keyword LLM" or aligned using Sentence-BERT between test samples and profiles.

**Core Idea**: Shift MoE routing from "hidden states" to "discrete skills" and replace experts from "parameter subsets" with "full pre-trained LLMs." Use expert-wise batch inference to make dynamic recruitment viable on a single GPU.

## Method

### Overall Architecture
SKILL-MOE consists of two phases. **Pre-processing Phase** (based on ~350 validation samples): Qwen2.5-7B-Instruct serves as a "Keyword LLM" to extract required skills (e.g., Algebra, Biology, Chemistry) for each problem. Each of the 16 LLMs in the pool attempts to solve these problems using CoT. If correct, the score for each involved skill for that model is incremented (+1); if incorrect, it is decremented (−1). This results in a skill profile $P_i$ for each model $M_i$, e.g., $\{\text{Algebra}: 10, \text{Biology}: 3, \text{Chemistry}: -6, \dots\}$. Simultaneously, a synthetic task of "selecting the correct CoT" is constructed to identify the model with the strongest aggregation capability for each dataset as the task-level aggregator $A^*$. **Inference Phase**: Skills are extracted for each test sample and aligned with profile skills using Sentence-BERT cosine similarity. Based on skill matching, $k=3$ experts are sampled to generate CoT paths, which $A^*$ then concatenates to produce the final answer.

### Key Designs

1. **Skill Profile + Local/Global Weighted Instance-Level Routing**:

    - **Function**: Selects the $k$ most suitable experts from the 16 models for each test sample, ensuring that "algebra-strong models handle algebra problems" and vice-versa.
    - **Mechanism**: For a query $q$, the required skill set $K_q$ is extracted. A model's "local adaptation score" is the sum of its scores on these skills: $S(M_i, q) = \sum_{k_j \in K_q} s^{(i)}_{k_j}$. The "global competence" $\gamma_i$ is calculated as the ratio of the model's total profile score to the pool's total score, reflecting its overall strength. The final relevance score $w^{(i)}_q = \gamma_i \cdot S(M_i, q)$ is passed through a softmax (temperature 0.5) for weighted sampling of $k$ experts. Experts appearing in less than 5% of the test set are filtered to reduce noise.
    - **Design Motivation**: Pure local scoring might allow a weak model with a lucky high score in a niche skill to be selected, while pure global scoring reverts to "task-level top-k," losing granularity. Multiplying the two balances "relative advantage on the specific sample" with "overall reliability on the task." In ablations, Top-3 / Top-5 / Random baselines on GPQA achieved 52.86% / 47.68% / 42.61%, while this design reached 57.78%, validating the effectiveness of fine-grained routing (Table 5).

2. **Task-Level Aggregator Selection (vs. Instance-Level or Majority Vote)**:

    - **Function**: Merges $k$ heterogeneous CoT paths into one high-quality answer using an aggregator that remains fixed across the task.
    - **Mechanism**: On the validation set, 1 correct CoT and 2 incorrect CoTs are sampled per problem. Candidate models act as aggregators to identify the correct answer. The model with the highest hit rate for each dataset is selected as $A^*$. During inference, $y = A^*(\bigoplus_{i=1}^k y_0^{(i)})$, where $\oplus$ denotes concatenation.
    - **Design Motivation**: The authors found that "models capable of reasoning are not necessarily good at aggregation"—a Random aggregator reached 52.29% on MMLU-Pro and an Adaptive instance-level aggregator reached 57.12%, while the Task-specific aggregator achieved 63.71% (Table 3). Table 7 further shows that once a good aggregator is selected, the gains from multi-round discussion are negligible (63.83 vs. 63.71), allowing for the elimination of expensive interactions.

3. **Expert-Wise Batched Inference Scheduling**:

    - **Function**: Enables 16 7-8B models to run on a single GPU with latency close to that of a fixed model set.
    - **Mechanism**: Routing is pre-calculated for a batch of samples. Samples requiring the same expert are grouped into a sub-batch. Models are loaded in a round-robin fashion—each activated expert is loaded only once per batch, avoiding the high latency of repeated loading/offloading in naive implementations.
    - **Design Motivation**: Dynamic recruitment means adjacent samples might require completely different model sets. Naive sample-by-sample scheduling on GPQA takes 196.92 s/sample. With grouping, this drops to 25.76 s on a single card (44% lower than MoA's 45.98 s on 1 GPU), matching 4-GPU MoA's 21.66 s. When scaled to 4 GPUs, SKILL-MOE's latency drops to 10.85 s, a nearly 2× speedup (Table 6). This assumes samples arrive in batches, which is naturally compatible with vLLM/ChatGPT/Gemini inference APIs.

### Loss & Training
Completely gradient-free. All experts, aggregators, and the Keyword LLM are frozen pre-trained LLMs. The "training" involves using ~350 validation samples to compute skill scores and aggregator hit rates. Each test sample involves $k=3$ expert calls plus 1 aggregator call, totaling 4 LLM calls—comparable to Self-Consistency $\times 5$ and fewer than MoA (6) or ReConcile (9).

## Key Experimental Results

### Main Results
Evaluation was conducted on 4 heterogeneous reasoning datasets: MMLU-Pro (14 subjects, 2100 problems), AIME 2024 (Mathematics Olympiad), GPQA Diamond (Science), and MedMCQA (Medical exams). The pool consisted of 16 LLMs (3.5B–12B), mostly 7-8B.

| Dataset | Metric | SKILL-MOE | Best Multi-Agent Baseline | Gain |
|---------|--------|-----------|---------------------------|------|
| AIME 2024 | Accuracy | 68.88 | 55.56 (Self-MoA) | +13.32 |
| MMLU-Pro | Accuracy | 63.71 | 61.78 (MoA) | +1.93 |
| MedMCQA | Accuracy | 59.35 | 60.74 (ReConcile) | −1.39 |
| GPQA Diamond | Accuracy | 57.78 | 52.86 (MoA / Self-MoA) | +4.92 |
| **Average** | **Accuracy** | **62.43** | **54.28 (Avg. of best baseline)** | **+8.15** |

Robustness across datasets: No single baseline consistently ranked second, whereas SKILL-MOE's average score surpassed Qwen2.5-72B (54.28) and Llama3.3-70B (53.18). It was also more stable than QwenR1-32B (56.94), which scored 76.67% on AIME but only 24.70% on MedMCQA.

### Ablation Study
| Configuration | Key Metric (GPQA) | Notes/Description |
|---------------|-------------------|-------------------|
| Full SKILL-MOE | 57.78 | Skill profile routing + Task-level aggregator |
| Random Aggregator + Recruited Experts | 51.52 | Aggregator quality is crucial |
| Task-specific Aggregator + Random Experts | 31.82 | Weak experts degrade even strong aggregators (Table 4) |
| Majority Vote + Recruited Experts | 53.54 | Majority vote acts as a fallback without aggregator |
| Top-3 Fixed Experts | 52.86 | Task-level coarse selection vs. instance-level (−4.92) |
| Top-5 Fixed Experts | 47.68 | Larger pools are more prone to noise from weak models |
| Adaptive Aggregator (MMLU-Pro) | 57.12 | Changing aggregators at instance-level actually −6.59 |
| Task-specific Aggregator + 3 Rounds | 63.83 (MMLU-Pro) / 57.72 (GPQA) | Discussion yields negligible or negative gains (Table 7) |

### Key Findings
- **Expert and Aggregator Selection are Synergistic**: Random experts with a strong aggregator yielded only 31.82%, while strong experts with a random aggregator yielded 51.52%; both are necessary to reach 57.78%.
- **Task-Level Aggregators Outperform Instance-Level**: "Ability to reason" $\neq$ "Ability to judge which CoT is correct." Dynamically selecting aggregators per sample is less effective than identifying the "best judge" for the overall task.
- **Strong Cross-Domain Generalization**: Skill profiles from MMLU-Pro transferred to OmniMATH (Math Olympiad) yielded 49.32%, 14.81% higher than the strongest Debate baseline. AIME profiles transferred to the same task still outperformed Self-MoA by 3.28% (Table 2), suggesting skills are more transfer-friendly than task-model pairings.
- **Efficiency Gains without Performance Loss**: Single-GPU 25.76 s/sample is 44% faster than MoA on one GPU with higher accuracy; 4-GPU setup provides nearly 2× speedup.

## Highlights & Insights
- **Symbolic Routing over Latent Routing**: By replacing neural network routers with "skill dictionary weighted sampling," 16 independently pre-trained LLMs can be integrated into an MoE framework without joint training. Profiles can be updated simply by re-running the validation set when models are upgraded.
- **Batch Inference as Key Engineering Support**: Dynamic expert recruitment has been explored algorithmically, but engineering constraints like VRAM limits and loading overhead remained. Grouping by experts transposes "dynamic" back into "static" batches, enabling a single card to hold 16 7-8B models. This synergy between algorithm and system constraints is broadly applicable to "on-demand LLM cluster" scenarios.
- **Aggregator Counter-Intuition**: The authors demonstrate that reasoning capability does not equate to aggregation capability. Once a correct aggregator is chosen, multi-round discussion provides almost zero benefit, suggesting that performance gains in many multi-agent frameworks stem from implicit aggregation rather than the discussion process itself.

## Limitations & Future Work
- **Dependency on Validation Set Distribution**: Profiles and aggregator rankings are derived from ~350 validation samples. Biases in the validation set propagate to the routing strategy. While MMLU-Pro profiles generalized to OmniMATH, profiles may fail if the target task's skill space differs significantly from training/validation data.
- **Requirement for Batched Arrival**: Expert-wise batching depends on pre-calculating routing for a group of samples. It is not directly applicable to single-query streaming or real-time low-latency scenarios (e.g., chat) without first aggregating skills at the frontend.
- **Keyword LLM Bias**: Although switching the Keyword LLM had minimal impact (Appendix), the study primarily used Qwen models. Robustness in entirely different domains (e.g., Law, Finance) requires further validation.
- **Slight Underperformance on MedMCQA**: SKILL-MOE was outperformed by ReConcile (59.35 vs. 60.74). The authors attribute this to the specialized nature of medical questions, where multi-round discussion may compensate for expert uncertainty in domains with sparse coverage.

## Related Work & Insights
- **vs. MoA / Self-MoA (Wang 2024a / Li 2025)**: MoA uses task-level fixed top-k models and 2 rounds of discussion; Self-MoA repeatedly calls a single strong model. SKILL-MOE changes experts per sample, eliminates discussion rounds, reduces calls from 4–9 to 4, and is 44% faster on a single GPU.
- **vs. ReConcile / Multi-Agent Debate (Chen 2024b / Du 2023)**: These rely on multi-round debate for consensus, running 6–9 calls on 3 fixed models. SKILL-MOE uses symbolic routing to automatically select models and relies on aggregators rather than debate.
- **vs. LLM-Blender / Router-R1 / DER**: These also focus on model selection/ranking, but LLM-Blender trains a ranking/fusion model, while Router-R1 and DER use RL/MDP for routing. SKILL-MOE is gradient-free; new models can be integrated via a few forward passes.
- **vs. Traditional Sparse MoE (Shazeer 2017)**: Traditional experts are parameter subsets; they require end-to-end training and have fixed scales. SKILL-MOE experts are full models communicating via language, allowing for hot-swappable scaling—effectively upgrading the "expert" concept to the "agent" scale.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Scaling MoE routing to symbolic skill spaces and experts to full LLMs is a clear contribution, though individual components (skill extraction, aggregator selection) have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 heterogeneous datasets, 8 baselines, extensive ablations, and zero-shot transfer tests on OmniMATH.
- **Writing Quality**: ⭐⭐⭐⭐ Clear frameworks, complete narrative, and precise algorithms, though some symbols are slightly inconsistent across sections.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical for researchers with limited resources (single-GPU support for heterogeneous pools) and offers a reusable engineering solution (expert-wise batching).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)

</div>

<!-- RELATED:END -->
