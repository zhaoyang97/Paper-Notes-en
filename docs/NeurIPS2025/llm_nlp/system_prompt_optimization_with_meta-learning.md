---
title: >-
  [Paper Note] System Prompt Optimization with Meta-Learning
description: >-
  [NeurIPS 2025][LLM/NLP][system prompt optimization] This paper formulates system prompt optimization as a bilevel problem and proposes MetaSPO…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "system prompt optimization"
  - "meta-learning"
  - "bilevel optimization"
  - "prompt engineering"
  - "cross-task generalization"
date: 2026-05-08
content_hash: c9e190bc41741b15
---

# System Prompt Optimization with Meta-Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.09666](https://arxiv.org/abs/2505.09666)  
**Code**: [GitHub](https://github.com/Dozi01/MetaSPO)  
**Area**: LLM/NLP
**Keywords**: system prompt optimization, meta-learning, bilevel optimization, prompt engineering, cross-task generalization

## TL;DR

This paper formulates system prompt optimization as a bilevel problem and proposes MetaSPO, a meta-learning framework that optimizes system prompts for cross-task generalization in the outer loop while optimizing task-specific user prompts in the inner loop. The resulting system prompts significantly outperform baselines across 14 unseen tasks.

## Background & Motivation

LLM inputs consist of two components: the **system prompt** (task-agnostic, defining the behavioral framework of the LLM) and the **user prompt** (task-specific, addressing particular queries). However, existing prompt optimization research has focused almost exclusively on user prompts, largely neglecting system prompt optimization.

The distinctive value of system prompt optimization:
- **Optimize once, deploy everywhere**: A well-optimized system prompt can generalize across multiple tasks and domains.
- **Synergy with user prompts**: An optimized system prompt establishes a robust behavioral framework for the LLM, complementing user prompts.
- **Reduced adaptation cost**: A good system prompt accelerates user prompt optimization for new tasks, requiring fewer iterations and less data.

Limitations of prior work:
- APE (2022), OPRO (2023), and TextGrad (2023) optimize only user prompts.
- SPRIG (2024) optimizes system prompts but ignores their interaction with user prompts and lacks a meta-learning framework.
- No systematic study exists on the cross-task generalization of system prompts.

## Method

### Overall Architecture

System prompt optimization is formalized as a **bilevel optimization** problem:

**Outer objective** (system prompt): $\mathbf{s}^* = \arg\max_{\mathbf{s}} \mathbb{E}_{T_i \sim \mathcal{T}}[\mathbb{E}_{(\mathbf{q},\mathbf{a}) \sim T_i}[f(\text{LLM}(\mathbf{s}, \mathbf{u}_i^*, \mathbf{q}), \mathbf{a})]]$

**Inner objective** (user prompt): $\mathbf{u}_i^* = \arg\max_{\mathbf{u}} \mathbb{E}_{(\mathbf{q},\mathbf{a}) \sim T_i}[f(\text{LLM}(\mathbf{s}, \mathbf{u}, \mathbf{q}), \mathbf{a})]$

### Key Designs: MetaSPO (Meta-level System Prompt Optimizer)

**Inner loop (user prompt optimization)**:
1. Evaluate the current user prompt on the target task and identify mispredicted samples.
2. Feed the current user prompt along with error samples into the LLM for **failure analysis**.
3. Generate multiple candidate user prompts based on the analysis.
4. Evaluate all candidates and retain the top-$k$ prompts.

**Outer loop (system prompt optimization)**:
1. Evaluate the current system prompt across **all** source tasks and aggregate error samples from each task.
2. Perform cross-task failure analysis on the system prompt.
3. Generate multiple candidate system prompts.
4. Evaluate each candidate system prompt across all tasks using the respective optimized user prompts.
5. Retain the top-$k$ system prompts with the best cross-task performance.

The two loops alternate for 3 iterations.

### Implementation Details

- **Base model**: Llama 3.2 (3B) for generating responses.
- **Optimizer model**: GPT-4o mini for generating candidate prompts.
- **Inner loop**: 3 candidate user prompts generated and retained per round; 3 error samples used for analysis.
- **Outer loop**: 9 candidate system prompts generated, 1 retained per round; 2 error samples per task used for analysis.
- **Temperature**: 0 for the base model (ensuring consistency), 1 for the optimizer model (encouraging diversity).

### Loss & Training

Optimization is performed entirely at inference time (gradient-free) without modifying model parameters. The task metric function $f$ (accuracy, F1, etc.) serves as the evaluation signal. The entire optimization process relies on the LLM's text generation capability — both prompt analysis and candidate generation are realized through carefully designed meta-prompts.

## Key Experimental Results

### Main Results: Generalization to Unseen Tasks

Average scores across 14 unseen target tasks spanning 5 domains (medical, review analysis, reasoning, safety, fact-checking):

| Method | Medical | Review | Reasoning | Safety | Grounding | Avg |
|--------|---------|--------|-----------|--------|-----------|-----|
| Default ("You are a helpful...") | 33.3 | 37.4 | 42.8 | 25.0 | 13.4 | 32.2 |
| CoT ("Let's think step by step") | — | — | — | — | — | 33.2 |
| Service (commercial system prompt) | — | — | — | — | — | 34.2 |
| SPRIG (genetic algorithm) | 37.0 | 56.8 | 38.7 | 28.1 | 14.1 | 35.0 |
| **MetaSPO (Domain)** | **48.9** | **62.7** | **52.2** | **36.5** | **16.4** | **44.5** |

MetaSPO achieves an **average improvement of 12.3 points** over the default system prompt and 7–9 points over the strongest baseline SPRIG.

### Test-Time Adaptation

With the system prompt fixed and user prompts additionally optimized:

| Method | Medical | Review | Reasoning | Safety | Grounding | Avg |
|--------|---------|--------|-----------|--------|-----------|-----|
| Default | 45.1 | 68.9 | 64.0 | 59.9 | 17.5 | 51.1 |
| SPRIG | 45.4 | 69.3 | 65.3 | 64.7 | 17.7 | 52.5 |
| **MetaSPO** | **45.6** | **71.4** | **67.3** | **67.2** | **19.9** | **54.3** |

### Ablation Study

| Experiment | Key Findings |
|------------|-------------|
| Outer loop only (no inner loop) | 38.2 vs. MetaSPO 42.2, demonstrating the importance of joint user prompt optimization |
| MetaSPO w/ APE | 38.9; alternative prompt optimization strategies can be plugged into the framework |
| MetaSPO w/ EVO | 40.2; evolutionary algorithm variants are also effective |
| Number of source tasks | Scaling from 1 to 6 source tasks yields +17.1% on Review and +8.26% on Reasoning |
| Cross-model transfer | System prompts optimized for Llama 3.2 transfer effectively to Llama 3.1, Qwen 2.5, and GPT-4o mini |
| Separated vs. unified input | Placing system/user prompts in their respective roles (separated input) outperforms merging them into the user role |

### Efficiency Analysis

- MetaSPO reaches the final performance of Default using **80% fewer optimization iterations** and **75% less data**.
- 85% of user prompts show improved performance under MetaSPO system prompts.

### Key Findings

1. System prompt optimization is a significantly underexplored direction — even a straightforward MetaSPO implementation yields substantial gains.
2. The synergy between inner and outer loops is critical — the outer loop alone yields considerably weaker results.
3. Optimized system prompts exhibit cross-model transferability, suggesting they encode generalizable task-solving knowledge.
4. LLM attention scores confirm that MetaSPO system prompts receive higher attention weights.

## Highlights & Insights

1. **Novel problem formulation**: Bilevel system prompt optimization is a natural yet previously unaddressed problem with strong practical relevance.
2. **Natural fit for meta-learning**: System prompts encode cross-task shared knowledge while user prompts handle task-specific adaptation — a perfect match for the meta-learning paradigm.
3. **Plug-and-play design**: The MetaSPO framework is agnostic to the specific prompt optimization method and allows flexible replacement of inner and outer loop strategies.
4. **Cross-model generalization**: System prompts optimized for one model remain effective on others, suggesting that good system prompts are universally applicable.
5. **Practical intuition**: Optimized system prompts typically endow the LLM with more specific role descriptions and behavioral guidelines.

## Limitations & Future Work

1. **Optimizer model dependency**: Performance is bounded by the capability of the optimizer LLM (GPT-4o mini here); weaker models may be insufficient.
2. **Limited evaluation tasks**: The 14 target tasks provide insufficient coverage, with notable omissions such as code generation and mathematical reasoning.
3. **Small base model**: Experiments are conducted primarily on Llama 3.2 (3B); the effect on larger models is not systematically verified.
4. **Computational overhead**: Multiple rounds of LLM calls are required to generate and evaluate candidate prompts, incurring non-negligible costs.
5. **Safety risks**: Optimized system prompts could be misused to steer LLMs toward harmful outputs.
6. **Absence of fine-tuning comparison**: Whether system prompt optimization can substitute for some degree of fine-tuning remains an open question.

## Related Work & Insights

- **Distinction from SPRIG (Wen et al., 2024)**: SPRIG uses a genetic algorithm to optimize system prompts but disregards user prompt interaction and lacks a meta-learning framework.
- **Relationship to MAML (Finn et al., 2017)**: MetaSPO draws on the "learning to learn" philosophy of meta-learning but operates in the text space.
- **Relationship to TextGrad (Pryzant et al., 2023)**: The inner loop employs a similar textual gradient concept.
- **Broader inspiration**: The bilevel optimization framework may be applicable to other LLM configuration optimization settings, such as RAG retrieval strategies and tool-use rules.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel problem formulation and well-motivated bilevel framework, though the underlying optimization techniques are relatively standard.
- **Theoretical Depth**: ⭐⭐ — Primarily an engineering and empirical contribution; theoretical analysis is lacking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 domains, 14 tasks, multi-model evaluation, extensive ablations, and attention analysis; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-illustrated, carefully designed experiments, and in-depth analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Directly applicable to improving LLM deployments; code is publicly available.
- **Overall**: ⭐⭐⭐⭐ (8/10) — A highly practical contribution that fills a clear gap in system prompt optimization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)
- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)

</div>

<!-- RELATED:END -->
