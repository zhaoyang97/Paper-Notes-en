---
title: >-
  [Paper Note] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes
description: >-
  [ACL 2026][LLM/NLP][LLM routing attack] This paper proposes R2A (Route to Rome Attack), which constructs a hybrid ensemble surrogate router in a black-box setting and optimizes universal adversarial suffixes to redirect LLM router decisions from cheap weak models toward expensive strong models — achieving an average attack success rate improvement of 49% across 7 open-source routers and 2 commercial routers (GPT-5-Auto, OpenRouter), with inference costs increasing by 2.7–2.9×.
tags:
  - ACL 2026
  - LLM/NLP
  - LLM routing attack
  - adversarial suffix
  - black-box attack
  - surrogate router
  - inference cost
date: 2026-05-08
content_hash: 7f0c76554519944d
---

# Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes

**Conference**: ACL 2026
**arXiv**: [2604.15022](https://arxiv.org/abs/2604.15022)
**Code**: [GitHub](https://github.com/thcxiker/R2A-Attack)
**Area**: LLM/NLP
**Keywords**: LLM routing attack, adversarial suffix, black-box attack, surrogate router, inference cost

## TL;DR

This paper proposes R2A (Route to Rome Attack), which constructs a hybrid ensemble surrogate router in a black-box setting and optimizes universal adversarial suffixes to redirect LLM router decisions from cheap weak models toward expensive strong models — achieving an average attack success rate improvement of 49% across 7 open-source routers and 2 commercial routers (GPT-5-Auto, OpenRouter), with inference costs increasing by 2.7–2.9×.

## Background & Motivation

**State of the Field**: To balance performance and cost, cost-aware LLM routing directs simple queries to cheap weak models and complex queries to expensive strong models. This routing strategy has been adopted by commercial systems (OpenRouter, GPT-5-Auto). Routers optimize the trade-off between quality loss and cost via $\mathcal{R}(q) = \arg\min_{M_i} [\ell(q, M_i) + \lambda \cdot C(q, M_i)]$.

**Limitations of Prior Work**: (1) Routing strategies introduce a new attack surface — adversaries may manipulate routers into consistently selecting expensive models to inflate operator costs; (2) the existing routing attack method Rerouting relies on white-box access (requiring gradients and architecture information), making it unsuitable for commercial black-box scenarios; (3) LifeCycle uses heuristic prompt templates (extracted from high-win-rate queries) without rigorous optimization, resulting in inconsistent effectiveness across different routers.

**Root Cause**: An adversary can only observe the router's final routing decision (which model was selected), with no access to internal logits, parameters, or gradients. Under this strict black-box setting, how can one learn a universal adversarial suffix within a limited query budget (120 queries) that consistently misleads routers of diverse architectures?

**Paper Goals**: (1) In a black-box setting where only routing decisions are observable, find a universal adversarial suffix that biases the router toward selecting expensive models; (2) the suffix must generalize across datasets, including out-of-distribution data.

**Starting Point**: Drawing on black-box adversarial attack techniques from computer vision — training surrogate models to simulate target model behavior, then optimizing adversarial examples on the surrogate and transferring them. The key challenge is the diversity of router architectures (embedding-based, LLM-based, etc.), meaning a single-architecture surrogate may not match the target router.

**Core Idea**: Employ a hybrid ensemble surrogate router (multiple open-source routers + a lightweight trainable router) to cover diverse routing mechanisms, and search for universal adversarial suffixes effective against unknown target routers via gradient-normalized ensemble suffix optimization.

## Method

### Overall Architecture

R2A proceeds in two stages: (1) surrogate router training — collecting 120 routing labels from the target router to train a hybrid ensemble surrogate that mimics target behavior; (2) suffix optimization — searching for adversarial suffixes on the surrogate using a modified GCG algorithm to bias routing probability toward strong models. The input is an original query; the output is a query appended with a fixed suffix that causes the router to select the expensive model.

### Key Designs

1. **Hybrid Ensemble Surrogate Router**:

    - Function: Faithfully simulate an unknown-architecture target router within a limited query budget.
    - Mechanism: Combines two types of routers — (a) $K$ pretrained open-source routers $\{\mathcal{R}_o^{(1)}, \dots, \mathcal{R}_o^{(K)}\}$ covering diverse routing mechanisms, aligned to the target model pool via zero-padding and linear mapping $\mathbf{W}_o$; (b) a lightweight trainable router $\mathcal{R}_l$ that encodes queries with all-MiniLM-L6-v2 and maps them to target model logits via LoRA-style low-rank decomposition $\mathbf{z}_l = E(q) \mathbf{W}_l^1 \mathbf{W}_l^2$ ($r \ll d=384$). Ensemble output: $\hat{y} = \text{softmax}(\alpha_0 \mathbf{z}_l + \sum_{i=1}^K \alpha_i \mathbf{z}_o^{(k)})$, with learnable weights $\alpha_i$.
    - Design Motivation: Open-source routers provide "rapid matching" — if the target router resembles a known open-source implementation, the ensemble can quickly lock in; the lightweight router provides "compensatory capacity" for target routers that differ from all open-source members. Low-rank decomposition reduces the number of queries required.

2. **Ensemble Gradient-Normalized Suffix Optimization**:

    - Function: Efficiently optimize discrete adversarial suffixes over a multi-encoder ensemble.
    - Mechanism: Optimization objective $\min_s \mathcal{L}_A = -\mathbb{E}_q \sum_{M \in \mathcal{M}_{strong}} p(\hat{y}=M|q \oplus s)$. Directly applying GCG is problematic — token gradients $\delta_i^{(k)} = \partial \mathbf{z}^{(k)} / \partial s_i$ from different ensemble members differ vastly in magnitude, causing one router to dominate optimization if gradients are summed directly. Solution: apply min-max normalization to each router's gradients $\tilde{\delta}_i^{(k)} = (\delta_i^{(k)} - \delta_{min}^{(k)}) / (\delta_{max}^{(k)} - \delta_{min}^{(k)})$, then aggregate with learned weights $\tilde{g}_i = \sum_{k=0}^K \alpha_k \cdot \tilde{\delta}_i^{(k)} \cdot \frac{\partial \mathcal{L}_A}{\partial \mathbf{z}_{total}}$.
    - Design Motivation: Gradient scales across router architectures (embedding-based vs. LLM-based) may differ by orders of magnitude; normalization ensures each ensemble member contributes equitably to suffix optimization.

3. **Progressive Query Activation**:

    - Function: Improve the generalizability of the optimized suffix.
    - Mechanism: Suffix optimization begins with the first query and admits the next query only after the current suffix successfully attacks all activated queries — this curriculum-like strategy progressively expands the query distribution the suffix must cover. At each iteration, the Top-K candidate tokens are selected for each suffix position, $B$ variants are sampled, and the one with the lowest loss is used to update the suffix.
    - Design Motivation: Optimizing over all queries simultaneously tends to get stuck in local optima; progressive activation allows the suffix to first identify effective patterns on simpler queries and then gradually generalize to harder ones.

### Loss & Training

Surrogate training: cross-entropy loss $\mathcal{L}_S = \frac{1}{Q}\sum_{i=1}^Q l(\hat{y}(q_i), \mathcal{R}_t(q_i))$ with query budget $Q=120$. Suffix optimization: maximize the probability of routing to strong models. When the target router is present in the ensemble pool, it is removed to prevent data leakage.

## Key Experimental Results

### Main Results

**Attack Success Rate ASR (average over 6 datasets, in-distribution + out-of-distribution)**

| Target Router | Clean | LifeCycle(W) | Rerouting | CoT | **R2A** | Δ vs Clean |
|---|---|---|---|---|---|---|
| RouteLLM-Bert | 0.40 | 0.69 | 0.77 | 0.52 | **0.89** | +0.49 |
| GraphRouter | 0.64 | 0.69 | 0.63 | 0.65 | **0.87** | +0.23 |
| RouteLLM-MF | 0.56 | 0.77 | 0.88 | 0.54 | **0.95** | +0.39 |
| OpenRouter* | 0.27 | 0.44 | 0.44 | 0.42 | **0.74** | +0.47 |

### Ablation Study

**Ablation of R2A Core Components (average over in-distribution datasets)**

| Configuration | RouterDC | CausalLLM | RouteLLM-MF | SW |
|---|---|---|---|---|
| R2A (full) | 0.83 | 0.83 | 0.95 | 0.81 |
| w/o lightweight router | 0.30 | 0.75 | 0.70 | 0.61 |
| w/o gradient normalization | 0.33 | 0.78 | 0.49 | 0.63 |

### Key Findings

- R2A raises ASR on the OpenRouter commercial router from 0.27 to 0.74 (+0.47), with effectiveness maintained on out-of-distribution data.
- In terms of inference cost, per-million-token costs increase by approximately 2.7× on MMLU and 2.9× on RouterArena.
- For GPT-5-Auto: fingerprinting analysis reveals that Thinking-likeness of responses increases significantly after attack; LLM-based evaluation shows that post-attack responses achieve 64–72% win rates on dimensions such as comprehensiveness and diversity.
- Query budget analysis: near-saturating performance is achieved with only 120 queries, demonstrating high sample efficiency.
- Under whitespace defense, ASR decreases only marginally (e.g., RouteLLM-Bert MT-Bench: 0.95→0.93), indicating robustness against simple defenses.

## Highlights & Insights

- The hybrid ensemble surrogate router design is practically motivated — open-source routers provide "prior knowledge" while the lightweight router provides "adaptive compensation," with the two being complementary. This design principle generalizes to other black-box attack scenarios.
- Gradient normalization is a seemingly simple but critical detail — ablations show that without it, ASR on the MF router drops precipitously from 0.95 to 0.49.
- The attack cost is extremely low (\$0.98 in query fees), yet induces a cost increase of up to 2.9× — highlighting the importance of routing systems as a security boundary.

## Limitations & Future Work

- A separate adversarial suffix must be trained for each target router; cross-router universal suffixes have not been achieved.
- The work focuses solely on routing toward expensive models and does not explore other attack objectives (e.g., targeting specific models, bypassing safety filters).
- The threat model assumes the adversary knows the candidate model pool and which model is selected per query — some deployments may not expose this information.
- Evaluation of GPT-5-Auto can only be conducted indirectly (routing decisions are not observable), with results dependent on the accuracy of fingerprinting.

## Related Work & Insights

- **vs. Rerouting (Shafran et al.)**: Requires white-box access (gradients/architecture); R2A requires only 120 black-box queries.
- **vs. LifeCycle**: Uses heuristic prompt templates, yielding inconsistent results across routers (e.g., only 0.69 on GraphRouter vs. R2A's 0.87).
- **vs. CoT baseline**: "Let's think step by step" is occasionally effective but unstable, and on some routers performs even worse than clean (e.g., RouteLLM-MF: 0.54 vs. clean 0.56).

## Rating

- Novelty: ⭐⭐⭐⭐ First to achieve LLM routing attacks in a strictly black-box setting; hybrid ensemble surrogate and gradient normalization are practical innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7+2 routers × 6 datasets + GPT-5 case study + cost analysis + defense evaluation + ablation + budget sensitivity.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; threat model is well-specified.
- Value: ⭐⭐⭐⭐⭐ Exposes a critical security vulnerability in LLM routing systems with direct implications for commercial deployments.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](../../AAAI2026/llm_nlp/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[ACL 2026\] Detoxification for LLM from Dataset Itself](detoxification_for_llm_from_dataset_itself.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](../../ICLR2026/llm_nlp/predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](../../ICLR2026/llm_nlp/pt2-llm_post-training_ternarization_for_large_language_models.md)

<!-- RELATED:END -->
