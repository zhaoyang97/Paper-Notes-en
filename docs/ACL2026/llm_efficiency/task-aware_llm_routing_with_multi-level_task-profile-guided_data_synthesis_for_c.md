---
title: >-
  [Paper Note] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios
description: >-
  [ACL 2026][LLM Efficiency][LLM Routing] This paper proposes a multi-level task-profile-guided data synthesis framework to address the cold-start problem in LLM routing. It designs TRouter—a routing method that treats tas…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "LLM Routing"
  - "Cold-Start"
  - "Data Synthesis"
  - "Task Awareness"
  - "Cost-Performance Trade-off"
date: 2026-05-08
content_hash: d7e2899e4a1ae74c
---

# Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios

**Conference**: ACL 2026  
**arXiv**: [2604.09377](https://arxiv.org/abs/2604.09377)  
**Code**: [GitHub](https://github.com/less-and-less-bugs/ColdStartLLMRouter)  
**Area**: LLM Evaluation  
**Keywords**: LLM Routing, Cold-Start, Data Synthesis, Task Awareness, Cost-Performance Trade-off

## TL;DR
This paper proposes a multi-level task-profile-guided data synthesis framework to address the cold-start problem in LLM routing. It designs TRouter—a routing method that treats task types as latent variables and models the query-cost-performance relationship through variational inference, achieving effective routing in both cold-start and in-domain settings.

## Background & Motivation

**Background**: LLM routing aims to select the optimal model from a candidate pool for each user query to balance performance and cost. Prevailing methods are categorized into classification-based (predicting the best model directly) and regression-based (predicting cost and performance to maximize a utility function), typically requiring small routers (e.g., BERT) trained on in-domain data.

**Limitations of Prior Work**: (1) Real-world deployments often encounter cold-start scenarios where no in-domain labeled data is available for training routers; (2) Pre-trained routers exhibit poor generalization across domains, sometimes performing worse than simple rule-based baselines (Adaptive LLM); (3) Relying directly on LLMs for model selection is unreliable due to the difficulty in accurately characterizing the capability boundaries of candidate models.

**Key Challenge**: LLM routing depends on labeled data, which is unavailable in cold-start scenarios; meanwhile, out-of-distribution shifts render cross-domain trained routers ineffective.

**Goal**: (1) Design a data synthesis method without human annotation to approximate the query distribution at test time; (2) Construct a task-aware router to enhance cross-domain robustness.

**Key Insight**: It is observed that the cost and performance of LLMs are intrinsically linked to task categories and difficulty—different task types/difficulties impose significantly different requirements on models. Based on this, a hierarchical task classification system can organize synthetic data, and implicit task type information can be leveraged during routing.

**Core Idea**: Use a hierarchical task taxonomy (domain $\rightarrow$ subcategory $\rightarrow$ difficulty) to guide synthetic data generation and incorporate task types as latent variables into a regression-based routing framework.

## Method

### Overall Architecture
The system consists of two major modules: (1) A multi-level task-profile-guided data synthesis framework, which iteratively builds a hierarchical task taxonomy starting from seed domain descriptions to generate diverse QA pairs as routing training data; (2) TRouter—a task-type-aware router that introduces latent task variables and jointly models the conditional distribution of performance and cost via variational inference.

### Key Designs

1.  **Hierarchical Task Taxonomy Generation (Task Type Generator + Quality Evaluator)**:
    - **Function**: Automatically expands a small number of seed domain descriptions into a complete three-level taxonomy (domain $\rightarrow$ subcategory $\rightarrow$ difficulty).
    - **Mechanism**: The Task Type Generator uses parent category descriptions as prompt conditions to recursively generate sub-types (each level including name, definition, and examples). The Task Type Quality Evaluator performs self-review on the generated sub-types to check for redundancy, specificity, and completeness, iterating until no modifications are made for three consecutive rounds. GPT-4.1 was used to synthesize 10 domains, 103 subcategories, 447 difficulty nodes, and 17,880 QA pairs.
    - **Design Motivation**: The hierarchical structure enables fine-grained control and efficient sampling coverage, while the quality evaluator ensures the cohesion and diversity of the taxonomy.

2.  **QA Pair Generation and Deduplication (Question-Answer Pair Generator)**:
    - **Function**: Generates diverse QA pairs for each difficulty-level task profile to serve as routing training data.
    - **Mechanism**: Uses task profiles (containing descriptions of the current and parent task types) as conditions for batch QA pair generation. A sentence-transformer calculates semantic similarity between new and existing QA pairs, filtering out near-duplicates with a maximum similarity $>0.9$. Generation iterates until each profile reaches the target quantity (40 pairs per profile, batch=8).
    - **Design Motivation**: Ensures synthetic data approximates the diversity of real test distributions, while the deduplication mechanism avoids data redundancy.

3.  **TRouter: Task-Type-Aware Router**:
    - **Function**: Introduces latent task type variables to enhance routing robustness and generalization.
    - **Mechanism**: Decomposes $p(h|q,m)$ as $\sum_t p(h|t,m) \cdot p(t|q)$, where $h$ is the evaluation metric and $t$ is the latent task type. The Task Recognition Module encodes the query and all task type descriptions, concatenating them through an MLP+softmax to predict the task distribution $q_\phi(t|q)$, constrained by cross-entropy and KL divergence from a prior. The Metric Prediction Module produces final predictions for each metric-model pair using task-distribution-weighted predictions for each type, trained with MSE loss. At inference, the optimal model is selected via the utility function $U(m,q)=\mu_r \cdot r(m,q) - \mu_c \cdot c(m,q)$.
    - **Design Motivation**: Directly predicting cost/performance from query features is prone to superficial feature influence; introducing task types as intermediate representations decouples the impact of task semantics, improving cross-domain robustness.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|\mathcal{M}||\mathcal{H}|} \sum_m \sum_h \mathcal{L}_{MSE}^{h,m}$, where the cross-entropy loss corresponds to the KL term of the ELBO and the MSE loss corresponds to the reconstruction term. Queries and task types are mapped to 256 dimensions using all-MiniLM-L6-v2. In cold-start settings, 30 training and 10 validation QA pairs are used per type.

## Key Experimental Results

### Main Results

| Setting | Method | Cost-first Utility | Balanced Utility | Perf-first Utility | Utility Sum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cold-start | Adaptive LLM | 0.0217 | 0.1809 | 0.2887 | 0.4913 |
| Cold-start | RouterDC⋆ | 0.0197 | 0.1490 | 0.2989 | 0.4676 |
| Cold-start | Ours▲ (GPT-4.1 Synth) | **0.0355** | **0.1811** | 0.3108 | **0.5274** |
| Cold-start | Ours∙ (Gemini Synth) | 0.0352 | 0.1809 | **0.3221** | **0.5382** |
| In-domain | MetricRouter | 0.0442 | 0.1911 | 0.3388 | 0.5741 |
| In-domain | Ours▲ | **0.0518** | **0.1949** | 0.3447 | **0.5914** |

### Ablation Study

| Configuration | Utility Sum | Description |
| :--- | :--- | :--- |
| TRouter (Full) | 0.5382 | Full model (Gemini Synth) |
| w/o Task Variables | ~0.52 | Degenerates to standard regression routing |
| w/o Data Synthesis | 0.4913 | Degenerates to rule-based baseline |
| w/o Quality Evaluator | ~0.51 | Decline in taxonomy quality |

### Key Findings
- In cold-start scenarios, TRouter's Utility Sum exceeds all baselines and even approaches the performance of in-domain methods.
- The synthesis framework is effective using both GPT-4.1 and Gemini-2.5-flash, verifying its versatility.
- In in-domain settings, TRouter also outperforms regression baselines like MetricRouter, proving that task type modeling gains are not limited to cold-start.
- Traditional cross-domain trained routers (RouterDC⋆, MetricRouter⋆) perform poorly under cold-start, sometimes failing to beat the Adaptive LLM rule-based baseline.

## Highlights & Insights
- **The design of task type as a latent variable is ingenious**: Extending the task taxonomy from the data synthesis stage to the routing modeling stage creates a closed loop of "synthetic data $\rightarrow$ routing prior." This adds a layer of structured inductive bias compared to simply training a standard router on synthetic data.
- **Problem definition and solution for cold-start are transferable**: The core idea of the data synthesis framework—using hierarchical classification to guide the generation of diverse samples—is applicable to any model selection or scheduling scenario lacking labeled data.
- **The variational inference framework provides interpretability**: The task distribution $q_\phi(t|q)$ is not only used for prediction but also informs the user about the type of task a query belongs to, enhancing the explainability of routing decisions.

## Limitations & Future Work
- Synthetic data remains dependent on powerful LLMs (GPT-4.1 or Gemini); applicability may be limited in scenarios where these models are unavailable.
- Seed domains for the task taxonomy must be manually specified (expanded from 6 to 10), and adaptation capability to entirely new domains is unverified.
- The candidate model pool in experiments is relatively small (6 open-source + 5 commercial); routing efficiency and scalability with larger model pools require further validation.
- Routing latency is not discussed—whether the inference time of the router itself offsets the efficiency gains of model selection in real-world deployments remains to be seen.

## Related Work & Insights
- **vs GraphRouter**: GraphRouter models routing as edge prediction on a heterogeneous graph; TRouter is more concise using latent task variables and shows a clear advantage in cold-start scenarios.
- **vs MetricRouter**: Both are regression-based; whereas MetricRouter predicts metrics directly from query embeddings, TRouter introduces additional task type decomposition, outperforming it in both in-domain and cold-start settings.
- **vs Adaptive Rules**: Adaptive LLM linearly selects models based only on user cost tolerance. Its relative robustness compared to most learning-based methods in cold-start highlights the severity of the cold-start problem.

## Rating
- Novelty: ⭐⭐⭐⭐ Valuable cold-start problem definition; clever integration of data synthesis and latent variable routing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both cold-start and in-domain settings with multi-LLM pool validation, though ablation studies could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, intuitive framework diagrams, and well-defined problem statements.
- Value: ⭐⭐⭐⭐ Cold-start routing is a genuine pain point in actual deployment; the synthesis framework offers good generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings](mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/llm_efficiency/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)

</div>

<!-- RELATED:END -->
