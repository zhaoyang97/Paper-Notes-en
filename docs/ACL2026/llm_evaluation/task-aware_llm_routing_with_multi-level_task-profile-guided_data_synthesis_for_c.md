---
title: >-
  [Paper Note] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios
description: >-
  [ACL 2026][LLM Evaluation][LLM routing] This paper proposes a multi-level task-profile-guided data synthesis framework to address the cold-start problem in LLM routing, and introduces TRouter—a routing method that treats task type as a latent variable—which models the query-cost-performance relationship via variational inference, achieving effective routing in both cold-start and in-domain settings.
tags:
  - ACL 2026
  - LLM Evaluation
  - LLM routing
  - cold-start
  - data synthesis
  - task-awareness
  - cost-performance trade-off
date: 2026-05-08
content_hash: 4c33429048ba9ab9
---

# Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios

**Conference**: ACL 2026
**arXiv**: [2604.09377](https://arxiv.org/abs/2604.09377)
**Code**: [GitHub](https://github.com/less-and-less-bugs/ColdStartLLMRouter)
**Area**: LLM Evaluation
**Keywords**: LLM routing, cold-start, data synthesis, task-awareness, cost-performance trade-off

## TL;DR
This paper proposes a multi-level task-profile-guided data synthesis framework to address the cold-start problem in LLM routing, and introduces TRouter—a routing method that treats task type as a latent variable—which models the query-cost-performance relationship via variational inference, achieving effective routing in both cold-start and in-domain settings.

## Background & Motivation

**Background**: LLM routing aims to select the optimal model from a candidate pool for each user query in order to balance performance and cost. Mainstream approaches are divided into classification-based (directly predicting the best model) and regression-based (predicting cost and performance to maximize a utility function) methods, typically training a small router (e.g., BERT) on in-domain data.

**Limitations of Prior Work**: (1) Real-world deployments frequently encounter cold-start scenarios where no in-domain labeled data is available for training the router; (2) pre-trained routers generalize poorly across domains and can even underperform simple rule-based baselines (e.g., Adaptive LLM); (3) using LLMs directly for model selection is unreliable, as accurately characterizing the capability boundaries of each candidate model is difficult.

**Key Challenge**: LLM routing depends on labeled data, which is unavailable in cold-start scenarios; meanwhile, distribution shift across domains renders cross-domain trained routers ineffective.

**Goal**: (1) Design a data synthesis approach that requires no human annotation to approximate the query distribution at test time; (2) build a task-type-aware router to enhance cross-domain robustness.

**Key Insight**: The observation that LLM cost and performance are intrinsically correlated with task category and difficulty—different task types and difficulty levels impose substantially different demands on models. This motivates organizing synthetic data via a hierarchical task taxonomy and incorporating implicit task-type information into routing.

**Core Idea**: Guide synthetic data generation with a hierarchical task taxonomy (domain → subcategory → difficulty), and model task type as a latent variable within a regression-based routing framework.

## Method

### Overall Architecture
The system consists of two main modules: (1) a multi-level task-profile-guided data synthesis framework—starting from seed domain descriptions, it iteratively constructs a hierarchical task taxonomy and generates diverse QA pairs as routing training data; (2) TRouter—a task-type-aware router that introduces an implicit task-type variable and jointly models the conditional distributions of performance and cost via variational inference.

### Key Designs

1. **Hierarchical Task Taxonomy Generation (Task Type Generator + Quality Evaluator)**:

    - Function: Automatically expands a small set of seed domain descriptions into a complete three-level taxonomy (domain → subcategory → difficulty).
    - Mechanism: The Task Type Generator recursively generates subtypes conditioned on parent-class descriptions (each node includes a name, definition, and examples). The Task Type Quality Evaluator performs self-review on the generated subtype sets, checking for redundancy, specificity, and completeness, and iteratively revises until no modifications occur for three consecutive rounds. Using GPT-4.1, this process yields 10 domains, 103 subcategories, 447 difficulty nodes, and 17,880 QA pairs.
    - Design Motivation: The hierarchical structure enables fine-grained control and efficient sampling coverage, while the quality evaluator ensures cohesion and diversity within the taxonomy.

2. **QA Pair Generation and Deduplication (Question-Answer Pair Generator)**:

    - Function: Generates diverse QA pairs for each difficulty-level task profile to serve as routing training data.
    - Mechanism: QA pairs are batch-generated conditioned on task profiles (which include descriptions of the current task type and its ancestor types). Sentence-transformer embeddings are used to compute semantic similarity between newly generated and existing QA pairs, filtering out near-duplicates with a maximum similarity above 0.9. Generation iterates until each profile reaches the target count (40 pairs per profile, batch size = 8).
    - Design Motivation: Ensures that synthetic data approximates the diversity of the real test distribution; the deduplication mechanism prevents data redundancy.

3. **TRouter: Task-Type-Aware Router**:

    - Function: Introduces an implicit task-type variable to enhance routing robustness and generalization.
    - Mechanism: $p(h|q,m)$ is decomposed as $\sum_t p(h|t,m) \cdot p(t|q)$, where $h$ is the evaluation metric and $t$ is the implicit task type. The Task Recognition Module encodes the query and all task-type descriptions, concatenates them, and passes them through an MLP with softmax to predict the task distribution $q_\phi(t|q)$, with a cross-entropy term constraining the KL divergence from the prior. The Metric Prediction Module uses a task-distribution-weighted combination of per-type predictions for each metric-model pair as the final prediction, trained with MSE loss. At inference time, the optimal model is selected via the utility function $U(m,q)=\mu_r \cdot r(m,q) - \mu_c \cdot c(m,q)$.
    - Design Motivation: Predicting cost/performance directly from query features is susceptible to surface-level artifacts; introducing task type as an intermediate representation decouples the influence of task semantics and improves cross-domain robustness.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|\mathcal{M}||\mathcal{H}|} \sum_m \sum_h \mathcal{L}_{MSE}^{h,m}$, where the cross-entropy loss corresponds to the KL term of the ELBO and the MSE loss corresponds to the reconstruction term. Queries and task types are encoded with all-MiniLM-L6-v2 and projected to 256 dimensions. In the cold-start setting, 30 training and 10 validation QA pairs per task type are used.

## Key Experimental Results

### Main Results

| Setting | Method | Cost-first Utility | Balanced Utility | Perf-first Utility | Utility Sum |
|------|------|---------------------|-------------------|---------------------|-------------|
| Cold-start | Adaptive LLM | 0.0217 | 0.1809 | 0.2887 | 0.4913 |
| Cold-start | RouterDC⋆ | 0.0197 | 0.1490 | 0.2989 | 0.4676 |
| Cold-start | Ours▲ (GPT-4.1 synthesis) | **0.0355** | **0.1811** | 0.3108 | **0.5274** |
| Cold-start | Ours∙ (Gemini synthesis) | 0.0352 | 0.1809 | **0.3221** | **0.5382** |
| In-domain | MetricRouter | 0.0442 | 0.1911 | 0.3388 | 0.5741 |
| In-domain | Ours▲ | **0.0518** | **0.1949** | 0.3447 | **0.5914** |

### Ablation Study

| Configuration | Utility Sum | Note |
|------|-------------|------|
| TRouter (full) | 0.5382 | Full model (Gemini synthesis) |
| w/o task-type variable | ~0.52 | Degrades to standard regression routing |
| w/o data synthesis | 0.4913 | Degrades to rule-based baseline |
| w/o quality evaluator | ~0.51 | Taxonomy quality degrades |

### Key Findings
- In cold-start scenarios, TRouter's Utility Sum surpasses all baselines and approaches the performance of in-domain methods.
- The synthesis framework is effective with both GPT-4.1 and Gemini-2.5-flash, validating its generality.
- In the in-domain setting, TRouter also outperforms regression baselines such as MetricRouter, demonstrating that the gains from task-type modeling extend beyond cold-start scenarios.
- Traditional cross-domain trained routers (RouterDC⋆, MetricRouter⋆) perform poorly in cold-start settings, with some even underperforming the Adaptive LLM rule-based baseline, highlighting the severity of the cold-start problem.

## Highlights & Insights
- **The design of task type as a latent variable is particularly elegant**: the task taxonomy is extended from the data synthesis stage into the routing modeling stage, forming a closed loop of "synthetic data → routing prior." This introduces an additional layer of structured inductive bias compared to simply training a standard router on synthetic data.
- **The cold-start problem formulation and solution are transferable**: the core idea of the data synthesis framework (using hierarchical taxonomy to guide the generation of diverse samples) is applicable to any model selection or scheduling scenario lacking labeled data.
- **The variational inference framework simultaneously endows the router with interpretability**: the task distribution $q_\phi(t|q)$ is not only used for prediction but also informs users of "what type of task this query belongs to," enhancing the interpretability of routing decisions.

## Limitations & Future Work
- The data synthesis pipeline still relies on powerful LLMs (GPT-4.1 or Gemini), limiting applicability in settings where such models are unavailable.
- The seed domains for the task taxonomy must be manually specified (6 domains expanded to 10); adaptability to entirely new domains has not been validated.
- The candidate model pool in experiments is relatively small (6 open-source + 5 commercial models); routing efficiency and scalability under a larger pool remain to be verified.
- Routing latency is not discussed—in practical deployment, whether the router's own inference time offsets the efficiency gains from model selection warrants investigation.

## Related Work & Insights
- **vs. GraphRouter**: GraphRouter models routing as edge prediction on a heterogeneous graph; TRouter's implicit task-type variable approach is more parsimonious and shows a clear advantage in cold-start settings.
- **vs. MetricRouter**: Both are regression-based routers; MetricRouter directly predicts metrics from query embeddings, whereas TRouter additionally introduces task-type decomposition, yielding improvements in both in-domain and cold-start settings.
- **vs. Adaptive LLM rule baseline**: Adaptive LLM selects models linearly based on user cost tolerance alone, yet proves more robust than most learning-based methods under cold-start conditions, underscoring the severity of the cold-start problem.

## Rating
- Novelty: ⭐⭐⭐⭐ The cold-start problem formulation is valuable, and the combination of data synthesis with latent-variable routing is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both cold-start and in-domain settings with multi-LLM-pool validation, though ablation studies could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, framework diagrams are intuitive, and problem definitions are precise.
- Value: ⭐⭐⭐⭐ Cold-start routing is a genuine pain point in real-world deployment, and the synthesis framework exhibits strong generality.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)

<!-- RELATED:END -->
