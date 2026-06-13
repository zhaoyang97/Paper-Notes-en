---
title: >-
  [Paper Note] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes
description: >-
  [ACL 2026][LLM Safety][LLM routing attack] This paper proposes R2A (Route to Rome Attack), which constructs a hybrid ensemble surrogate router under a black-box setting and optimizes universal adversarial suffixes to red…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM routing attack"
  - "adversarial suffix"
  - "black-box attack"
  - "surrogate router"
  - "inference cost"
date: 2026-05-08
content_hash: 1bed71f968456f01
---

# Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes

**Conference**: ACL 2026  
**arXiv**: [2604.15022](https://arxiv.org/abs/2604.15022)  
**Code**: [GitHub](https://github.com/thcxiker/R2A-Attack)  
**Area**: LLM/NLP  
**Keywords**: LLM routing attack, adversarial suffix, black-box attack, surrogate router, inference cost

## TL;DR

This paper proposes R2A (Route to Rome Attack), which constructs a hybrid ensemble surrogate router under a black-box setting and optimizes universal adversarial suffixes to redirect LLM routing decisions from cheap weak models to expensive strong models—improving the average attack success rate by 49% across 7 open-source and 2 commercial routers (GPT-5-Auto, OpenRouter) and increasing inference costs by 2.7-2.9 times.

## Background & Motivation

**Background**: To balance performance and cost, cost-aware LLM routing directs simple queries to cheap weak models and complex queries to expensive strong models. This strategy has been adopted by commercial systems (OpenRouter, GPT-5-Auto). Routers optimize $\mathcal{R}(q) = \arg\min_{M_i} [\ell(q, M_i) + \lambda \cdot C(q, M_i)]$ to trade off quality loss and cost.

**Limitations of Prior Work**: (1) Routing strategies introduce a new security surface—attackers may manipulate routers to consistently select expensive models to increase operator costs; (2) Existing routing attack methods like Rerouting rely on white-box access (requiring gradients and architectural information), which is unsuitable for commercial black-box scenarios; (3) LifeCycle uses heuristic prompt templates (extracted from high-win-rate queries) without rigorous optimization, leading to unstable performance across different routers.

**Key Challenge**: Attackers can only observe the final routing decision (which model was selected) without access to internal logits, parameters, or gradients. In this strict black-box setting, how can a universal adversarial suffix be learned with a limited query budget (120 queries) to consistently mislead routers of different architectures?

**Goal**: (1) Identify universal adversarial suffixes under black-box settings that bias routers toward selecting expensive models; (2) Ensure suffixes generalize across datasets (including out-of-distribution data).

**Key Insight**: Draw inspiration from black-box adversarial attacks in computer vision—train a surrogate model to simulate the target model's behavior, optimize adversarial samples on the surrogate, and then transfer. The key challenge lies in the diversity of router architectures (embedding-based, LLM-based, etc.), where a single-architecture surrogate may not match the target router.

**Core Idea**: Use a hybrid ensemble surrogate router (multiple open-source routers + a lightweight trainable router) to cover various routing mechanisms, finding universal adversarial suffixes effective for unknown target routers through ensemble suffix optimization with gradient normalization.

## Method

### Overall Architecture

R2A consists of two stages: (1) Surrogate router training—collecting training labels from the target router using 120 queries to train the hybrid ensemble surrogate router to simulate target behavior; (2) Suffix optimization—searching for adversarial suffixes on the surrogate router using an improved GCG algorithm to bias routing probabilities toward strong models. The input is the original query, and the output is the query with a fixed suffix attached to force the selection of expensive models.

### Key Designs

1.  **Hybrid Ensemble Surrogate Router**:
    *   **Function**: Faithfully simulate target routers of unknown architectures within a limited query budget.
    *   **Mechanism**: Combine two types of routers: (a) $K$ pre-trained open-source routers $\{\mathcal{R}_o^{(1)}, \dots, \mathcal{R}_o^{(K)}\}$ (covering different routing mechanisms), aligned to the target model pool via zero-padding and linear mapping $\mathbf{W}_o$; (b) A lightweight trainable router $\mathcal{R}_l$, which encodes queries using all-MiniLM-L6-v2 and maps them to target model logits via LoRA-style low-rank decomposition $\mathbf{z}_l = E(q) \mathbf{W}_l^1 \mathbf{W}_l^2$ ($r \ll d=384$). The ensemble output is $\hat{y} = \text{softmax}(\alpha_0 \mathbf{z}_l + \sum_{i=1}^K \alpha_i \mathbf{z}_o^{(k)})$, where weights $\alpha_i$ are learnable.
    *   **Design Motivation**: Open-source routers provide "fast matching"—if the target router resembles an open-source implementation, the ensemble can quickly lock in; the lightweight router provides "compensatory capability" for targets unlike any open-source router. Low-rank decomposition reduces the required number of queries.

2.  **Ensemble Gradient Normalization Suffix Optimization**:
    *   **Function**: Effectively optimize discrete adversarial suffixes across a multi-encoder ensemble.
    *   **Mechanism**: Optimization target: $\min_s \mathcal{L}_A = -\mathbb{E}_q \sum_{M \in \mathcal{M}_{strong}} p(\hat{y}=M|q \oplus s)$. Using GCG directly is problematic as token gradients $\delta_i^{(k)} = \partial \mathbf{z}^{(k)} / \partial s_i$ from different routers vary significantly in magnitude, leading to one router dominating. Solution: Apply min-max normalization to each router's gradient $\tilde{\delta}_i^{(k)} = (\delta_i^{(k)} - \delta_{min}^{(k)}) / (\delta_{max}^{(k)} - \delta_{min}^{(k)})$, then perform weighted aggregation $\tilde{g}_i = \sum_{k=0}^K \alpha_k \cdot \tilde{\delta}_i^{(k)} \cdot \frac{\partial \mathcal{L}_A}{\partial \mathbf{z}_{total}}$.
    *   **Design Motivation**: Gradient scales across different router architectures (embedding-based vs LLM-based) can differ by orders of magnitude; normalization ensures every member router contributes fairly to suffix optimization.

3.  **Progressive Query Activation**:
    *   **Function**: Improve suffix universality.
    *   **Mechanism**: Suffix optimization starts with the first query and adds the next query only after the suffix successfully attacks all currently activated queries—this curriculum-like strategy gradually expands the query distribution the suffix must cover. Each iteration selects Top-K candidate tokens for each suffix position, samples $B$ variants, and updates with the lowest loss.
    *   **Design Motivation**: Optimizing for all queries simultaneously is prone to local optima; progressive activation allows suffixes to find effective patterns in simple queries first, then gradually generalize to harder ones.

### Loss & Training

Surrogate training: Cross-entropy loss $\mathcal{L}_S = \frac{1}{Q}\sum_{i=1}^Q l(\hat{y}(q_i), \mathcal{R}_t(q_i))$ with query budget $Q=120$. Suffix optimization: Maximize the probability of routing to strong models. If the target router is in the ensemble pool, it is removed to prevent data leakage.

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR) (average across 6 datasets, In-Distribution + Out-of-Distribution)**

| Target Router | Clean | LifeCycle(W) | Rerouting | CoT | **R2A** | Gain vs Clean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RouteLLM-Bert | 0.40 | 0.69 | 0.77 | 0.52 | **0.89** | +0.49 |
| GraphRouter | 0.64 | 0.69 | 0.63 | 0.65 | **0.87** | +0.23 |
| RouteLLM-MF | 0.56 | 0.77 | 0.88 | 0.54 | **0.95** | +0.39 |
| OpenRouter* | 0.27 | 0.44 | 0.44 | 0.42 | **0.74** | +0.47 |

### Ablation Study

**Ablation of R2A Core Components (average across In-Distribution datasets)**

| Configuration | RouterDC | CausalLLM | RouteLLM-MF | SW |
| :--- | :--- | :--- | :--- | :--- |
| **R2A Full** | 0.83 | 0.83 | 0.95 | 0.81 |
| w/o Lightweight Router | 0.30 | 0.75 | 0.70 | 0.61 |
| w/o Gradient Normalization | 0.33 | 0.78 | 0.49 | 0.63 |

### Key Findings

*   R2A increases ASR on the OpenRouter commercial router from 0.27 to 0.74 (+0.47) and remains effective on out-of-distribution data.
*   In terms of inference costs, the cost per million tokens increases by approximately 2.7x on MMLU and 2.9x on RouterArena.
*   Attacking GPT-5-Auto: Fingerprinting analysis shows that "Thinking-likeness" in responses significantly increases after the attack; LLM judges indicate that post-attack response win rates reach 64-72% in dimensions such as comprehensiveness and diversity.
*   Query budget analysis: Performance near-saturation is achieved with only 120 queries, demonstrating high sample efficiency.
*   ASR drops only slightly under white-space defense (e.g., RouteLLM-Bert MT-Bench: 0.95→0.93), showing robustness to simple defenses.

## Highlights & Insights

*   The design of the hybrid ensemble surrogate router is highly practical—open-source routers provide "prior knowledge" while the lightweight router provides "adaptive compensation." This approach can be generalized to other black-box attack scenarios.
*   Gradient normalization is a seemingly simple but critical detail—ablation shows that without it, the ASR of the MF router plummets from 0.95 to 0.49.
*   The attack cost is extremely low (approx. $0.98 in query fees), yet it causes a cost increase of up to 2.9x—highlighting the importance of routing systems as a security boundary.

## Limitations & Future Work

*   A separate adversarial suffix must be trained for each target router; universal suffixes across different routers have not been achieved.
*   Focuses only on directing routing to expensive models; other attack targets (e.g., specifying specific models, bypassing safety filters) have not been explored.
*   Assumes the attacker knows the candidate model pool and the model selected for each query—some deployments may not disclose this information.
*   GPT-5-Auto can only be indirectly evaluated (as routing decisions cannot be observed), making results dependent on fingerprinting accuracy.

## Related Work & Insights

*   **vs Rerouting (Shafran et al.)**: Requires white-box access (gradients/architecture), whereas R2A requires only 120 black-box queries.
*   **vs LifeCycle**: Uses heuristic prompt templates, which are unstable across different routers (e.g., 0.69 on GraphRouter vs 0.87 for R2A).
*   **vs CoT baseline**: "Let's think step by step" is occasionally effective but unstable, and even performs worse than "clean" on some routers (e.g., RouteLLM-MF: 0.54 vs clean 0.56).

## Rating

*   Novelty: ⭐⭐⭐⭐ First to achieve LLM routing attacks under strict black-box settings; hybrid ensemble surrogate and gradient normalization are practical innovations.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7+2 routers × 6 datasets + GPT-5 case study + cost analysis + defense evaluation + ablation + budget sensitivity.
*   Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the threat model is standardized.
*   Value: ⭐⭐⭐⭐⭐ Reveals a key security vulnerability in LLM routing systems with direct implications for commercial deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](../../CVPR2026/llm_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)

</div>

<!-- RELATED:END -->
