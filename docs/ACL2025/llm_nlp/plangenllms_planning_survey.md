---
title: >-
  [Paper Note] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities
description: >-
  [LLM (Other)] This is the first survey on LLM planning capabilities that introduces a six-dimensional evaluation framework (completeness, executability, optimality, representation, generalization, efficiency) based on classical planning theory (Kartam & Wilkins 1990). It systematically reviews foundational paradigms ranging from task decomposition to search algorithms, while identifying key unresolved directions such as multi-agent planning, hallucinations…
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: ffdb99cce3da91d2
---

# PlanGenLLMs: A Modern Survey of LLM Planning Capabilities

| Conference | arXiv | Code | Area | Keywords |
|------------|-------|------|------|----------|
| ACL 2025 | [2502.11221](https://arxiv.org/abs/2502.11221) | [GitHub](https://github.com/wll199566/Awesome-LLM-Planning-Capability) | llm_nlp | LLM Planning, Survey, Planning Evaluation, Agentic Workflow, Task Decomposition |

## TL;DR

This is the first survey on LLM planning capabilities that introduces a six-dimensional evaluation framework (completeness, executability, optimality, representation, generalization, efficiency) based on classical planning theory (Kartam & Wilkins 1990). It systematically reviews foundational paradigms ranging from task decomposition to search algorithms, while identifying key unresolved directions such as multi-agent planning, hallucinations, and alignment with human preferences.

## Background & Motivation

**Background**: LLMs have demonstrated great potential in planning tasks and have been widely applied in domains such as web navigation, travel planning, database querying, and robotic manipulation. A large body of research has explored utilizing LLMs for various planning tasks, ranging from simple step decomposition to complex multi-step decision-making.

**Limitations of Prior Work**: Existing LLM planning systems face three key issues: (1) Most systems are customized for specific problems, making it difficult to compare the strengths and weaknesses of different planners across domains; (2) There is a lack of clear and consistent evaluation standards, as different papers use different metrics and datasets, preventing fair comparison; (3) Prior surveys (e.g., Huang et al. 2024, Li et al. 2024) only approach the topic from the perspective of methodological classification or evaluation benchmarks, lacking systematic dimensions for performance evaluation.

**Key Challenge**: The contradiction between the rapid growth of LLM planning research and the fragmentation of evaluation frameworks makes it difficult for researchers to answer the core question of "which planning method to choose for a new task," due to the lack of a unified performance evaluation framework to measure different methods across various dimensions.

**Goal**: (1) To establish a unified evaluation framework derived from classical AI planning theory; (2) To conduct a comprehensive and systematic classification and analysis of existing LLM planning methods; (3) To identify key gaps and future directions in current research.

**Key Insight**: The authors trace back to the classical planning evaluation framework of Kartam & Wilkins (1990), readapting its evaluation dimensions of performance, representation, and communication to the LLM era and proposing six key performance criteria. This perspective of "examining new problems on classical foundations" offers greater theoretical depth than purely classifying from a methodological standpoint.

**Core Idea**: To uniformly measure LLM planning capabilities using a six-dimensional evaluation framework from classical AI planning theory (completeness, executability, optimality, representation, generalization, efficiency), providing a systematic comparison benchmark for a fragmented research field.

## Method

### Overall Architecture

As a survey paper, the core contribution is proposing a structured analytical framework. The overall organization is structured as: Foundational Paradigms (§2) → Detailed Analysis of Six Performance Dimensions (§3-§8) → Evaluation Methods & Metrics (§9) → Future Directions (§10). The strengths and weaknesses of representative works are thoroughly analyzed under each dimension.

### Key Designs

1. **Four Foundational Planning Paradigms**:

    - **Function**: To define the foundational methodology of LLM planning
    - **Mechanism**: (a) **Task Decomposition**—decomposing abstract goals into concrete sub-goals, supporting sequential, parallel, and asynchronous modes (e.g., DEPS, ADaPT recursively decomposing until executable); (b) **LLM + Classical Planner**—LLMs are responsible for translating natural language into formal representations like PDDL, while classical solvers (e.g., Fast Downward) guarantee correctness; (c) **Search Algorithms**—treating planning as a search problem and systematically exploring it using BFS/DFS/MCTS, with LLMs providing expansion and evaluation (e.g., Tree of Thoughts, RAP); (d) **Fine-Tuning**—fine-tuning LLM parameters on planning-specific or general Agent data
    - **Design Motivation**: To cover the complete methodological spectrum from pure prompting to parameter updating, and from end-to-end to hybrid systems

2. **Six-Dimensional Performance Evaluation Criteria**:

    - **Function**: To provide a unified evaluation framework for LLM planners
    - **Mechanism**:
        - **Completeness**: Whether the planner can generate correct plans (to reach the target state) + identify unreachable goals (even GPT-4 and o1 still fail severely due to hallucinations when identifying unreachable goals)
        - **Executability**: Whether the generated plans can be executed in the given environment, implemented via four safeguarding approaches: object grounding, action grounding, sampling-filtering, and closed-loop systems
        - **Optimality**: Whether the optimal (rather than just feasible) plan can be found, with LLM + optimizer and A* search being the two main paradigms
        - **Representation**: The format of input and output (Natural Language vs. PDDL vs. Python); LLMs are highly sensitive to prompt formatting
        - **Generalization**: Whether the systems can generalize to more complex, out-of-distribution (OOD) scenarios, enhanced through three approaches: fine-tuning, generalized planning, and skill libraries
        - **Efficiency**: Reducing the number of LLM calls, shortening input/output length, and utilizing smaller models
    - **Design Motivation**: A modern adaptation of the classical Kartam & Wilkins (1990) evaluation framework, where each dimension is orthogonal and complementary

3. **Panoramic Analysis of the Evaluation Toolkit**:

    - **Function**: To organize the complete ecosystem of datasets, evaluation methods, and metrics
    - **Mechanism**: Categorizing datasets into planning-specific (Embodied, scheduling, games, task decomposition) and downstream tasks (reasoning, tool-use, programming, Web); evaluation methods are split into verifier validation, human evaluation, and LLM-as-Judge; metrics cover success rate, goal-conditioned recall rate, executability rate, optimality rate, etc.
    - **Design Motivation**: To provide researchers with clear guidelines for choosing evaluation schemes

## Key Experimental Results

### Method Coverage Statistics

As a survey, the core data is reflected in the covered literature and taxonomic statistics:

| Dimension | Representative Methods Count | Key Findings |
|-----------|------------------------------|--------------|
| Task Decomposition | 6+ (DEPS, ProgPrompt, ADaPT...) | Recursive decomposition performs best in long-term planning |
| LLM + Classical Planner | 7+ (LLM+P, LLM-DP, SimPlan...) | Guarantees correctness but relies on formal translation quality |
| Search Algorithms | 9+ (ToT, RAP, LATS, MCTS...) | Systematic exploration but with high computational overhead |
| Fine-Tuning | 6+ (RobLM, Agent-FLAN, ETO...) | General Agent fine-tuning outperforms planning-specific fine-tuning |

### Performance Criteria vs. Methods Comparison

| Performance Criteria | Main Challenges | Current Best Practices | Gap |
|----------------------|-----------------|------------------------|-----|
| Completeness | LLMs cannot identify unreachable goals | LLM + Classical solver | GPT-4/o1 still suffer from severe hallucinations |
| Executability | Action/Object mismatch | Closed-loop systems + Feedback | High overhead; implicit vs. explicit repair remains ununified |
| Optimality | Finding optimal instead of just feasible solutions | A* search + LLM heuristics | Requires admissible heuristics, hard to guarantee |
| Representation | Format choice has high impact | Python > NL > PDDL (in some experiments) | Lacks systematic format comparison studies |
| Generalization | Out-of-distribution (OOD) transfer | Skill libraries + Generalized planning | Only explored by a few works like VOYAGER |
| Efficiency | Excessive token consumption | One-shot generation + Prefix caching | Search methods still require massive redundant queries |

### Key Findings

- **Insufficient Independent Planning Ability of LLMs**: Multiple studies (Valmeekam et al. 2023, Kambhampati 2024) have demonstrated that LLMs perform poorly acting as standalone planners; combining them with classical planners remains a more reliable alternative.
- **Hallucinations as the Core Bottleneck**: Even top-tier LLMs fail to reliably identify unreachable goals, frequently generating non-existent actions and objects.
- **The Double-Edged Sword of Fine-Tuning**: Planning-specific fine-tuning may jeopardize general capabilities, whereas general Agent fine-tuning (e.g., Agent-FLAN) yields better results.
- **Overlooked Representation Formats**: While most studies rely on natural language by default, Python formats perform better in some tasks. This direction remains significantly under-researched.

## Highlights & Insights

- **Modern Resurgence of Classical Theory**: Readapting the 1990 evaluation framework to the LLM era—a "standing on the shoulders of giants" approach—offers both theoretical depth and practical utility, establishing a unified language for a fragmented research domain.
- **Orthogonal Design**: The orthogonal relationships among the six dimensions are clearly defined (e.g., executability is orthogonal to completeness—one executable plan is not necessarily correct, and vice versa). This precise distinction helps researchers understand the multi-faceted nature of the problem.
- **Implicit vs. Explicit Classification of Closed-Loop Systems**: Closed-loop repair is categorized into implicit (only fixing the failed steps) and explicit (regenerating the entire plan) approaches. The former is highly efficient but susceptible to error accumulation, while the latter is computationally heavy but more robust; this classification provides direct guidance for practical implementation.
- **Four-Component Decomposition of Search Algorithms**: Systematically decomposing search methods into four components—search strategy, expansion, world model, and evaluation—facilitates straightforward comparison of design choices across different methods.

## Limitations & Future Work

- **Coverage Bias of the Survey**: It primarily focuses on single-agent scenarios (robotics, household chores, web tasks), while offering insufficient coverage of emerging fields such as multi-agent planning, natural sciences, and IoT.
- **Lack of Quantitative Horizontal Comparison**: Although the strengths and weaknesses of various methods are analyzed, there is no quantitative experimental comparison across a unified benchmark.
- **Omission of Safety and Personalization**: Safety and personalized planning are only briefly mentioned in the discussion section without in-depth analysis.
- **Insufficient Actionability of Future Directions**: While directions like multi-agent planning and hallucination are identified, the paper lacks concrete roadmaps to address them.
- **Timeliness Issues**: Submitted in February 2025, it does not cover subsequent rapidly evolving Agent frameworks (such as the newer OpenAI Agent SDK).

## Related Work & Insights

- **vs. Huang et al. (2024) Survey**: While they classify from a methodological perspective (decomposition, selection, reflection, memory), this paper approaches from the performance evaluation perspective, focusing more on "how to evaluate" rather than "how to execute." The two perspectives are highly complementary.
- **vs. Li et al. (2024d) LASP Survey**: While they focus on the classification of evaluation benchmarks, this work builds upon that to introduce a unified performance criteria framework.
- **vs. Kambhampati et al. (2024) "LLMs Can't Plan"**: They proposed the LLM-Modulo framework, arguing that LLMs cannot plan independently. The analysis in this paper supports this view and offers more comprehensive evidence.
- **Important Implications for Current Agent Research**: The six evaluation dimensions can be directly applied to evaluate the quality of planning modules within Agent systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic nature of the six-dimensional evaluation framework is a highlight, though methodological innovation is inherently limited as a survey.
- Experimental Thoroughness: ⭐⭐⭐ No proprietary experiments are conducted as a survey paper, but literature coverage is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The structure is clear, the taxonomy is professionally presented, and the literature is well-organized.
- Value: ⭐⭐⭐⭐ Holds high reference value for LLM planning researchers and Agent developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)

</div>

<!-- RELATED:END -->
