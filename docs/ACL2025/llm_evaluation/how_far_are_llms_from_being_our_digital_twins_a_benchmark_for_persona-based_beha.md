---
title: >-
  [Paper Note] How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation
description: >-
  [ACL 2025 (Findings)][LLM Evaluation][Digital Twin] This paper proposes the BehaviorChain benchmark to evaluate LLMs' ability to simulate sequential human behaviors for the first time. Containing 15,846 behavior samples under 1,001 persona profiles, the study reveals that even state-of-the-art models perform poorly in sequential behavior simulation.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Evaluation"
  - "Digital Twin"
  - "Persona-Driven Behavior Simulation"
  - "Behavior Chain"
  - "Benchmark"
date: 2026-05-08
content_hash: f6321b45bb6c6dc8
---

# How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.14642](https://arxiv.org/abs/2502.14642)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Digital Twin, Persona-Driven Behavior Simulation, Behavior Chain, Benchmark, LLM Evaluation

## TL;DR

This paper proposes the BehaviorChain benchmark to evaluate LLMs' ability to simulate sequential human behaviors for the first time. Containing 15,846 behavior samples under 1,001 persona profiles, the study reveals that even state-of-the-art models perform poorly in sequential behavior simulation.

## Background & Motivation

**Background**: The potential of large language models (LLMs) as "human digital twins"—aiming to build virtual agents capable of replicating individual traits and making decisions and reasoning on behalf of users—has garnered increasing attention. Prior work primarily evaluates LLMs' human simulation capabilities through persona-based dialogues, where models role-play specific characters.

**Limitations of Prior Work**: Current evaluations focus overly on dialogue simulation while neglecting behavior simulation, which is crucial for digital twins. A true digital twin must not only "talk like a human" but also "act like a human," making coherent sequential behavioral decisions consistent with a specific persona. For instance, what action sequence would a cautious individual take when facing a risk? Existing benchmarks cannot evaluate this dimension.

**Key Challenge**: There is a fundamental difference between dialogue simulation and behavior simulation. Dialogue is more susceptible to superficial language patterns, whereas behavior simulation requires a deep understanding of how personality traits map to concrete action choices. Furthermore, behaviors are often continuous, chained processes (behavior chains) where preceding actions influence subsequent steps.

**Goal**: To construct the first benchmark for evaluating LLMs' continuous behavior simulation capabilities, answering the core question: "How far are LLMs from being our digital twins?"

**Key Insight**: Focusing on the perspective of the "Behavior Chain"—evaluating the model's capacity to consecutively infer a series of contextually relevant behaviors under a given persona and dynamic scenario, rather than predicting isolated single actions.

**Core Idea**: Introducing a three-dimensional information framework of persona, historical behaviors, and situational contexts, enabling LLMs to iteratively infer the next action in a behavior chain and evaluating their digital twin similarity against real human behaviors.

## Method

### Overall Architecture

The pipeline of BehaviorChain construction and evaluation consists of: (1) collecting high-quality persona profiles and histories, compiling metadata such as demographics, personality traits, and backgrounds; (2) constructing a behavior chain—a sequence of chronological behaviors in specific scenarios—for each persona; (3) injecting persona metadata into LLMs during evaluation as role setups, providing scenarios and partial behavioral histories to let models iteratively infer subsequent actions; (4) comparing model-predicted behaviors with real behavior chains to compute matching scores.

### Key Designs

1. **Persona-driven behavior chain dataset construction**:

    - **Function**: To provide high-quality and diverse persona-behavior alignment data.
    - **Mechanism**: The dataset consists of 1,001 unique personas, each containing detailed profiles (demographics, personality, hobbies) and history. Behavior chains are constructed chronologically (15,846 total behaviors), with causal or contextual connections between actions—e.g., "shopping at the grocery store" $\rightarrow$ "organizing shopping bags" $\rightarrow$ "cooking at home". Data sources are rigorously filtered for diversity and authenticity.
    - **Design Motivation**: Modeling behavior chains instead of isolated single actions captures the continuity and dependencies of human actions, which is a core requirement for digital twins.

2. **Iterative behavior inference evaluation protocol**:

    - **Function**: To evaluate models' capability to infer behaviors in dynamically evolving scenarios.
    - **Mechanism**: An iterative inference protocol is adopted: at each step, the model is fed persona metadata, current scenario, and past behavior history to predict the next action. After prediction, the ground-truth action is appended to history for the next step. This rolling prediction simulates dynamic real-world decision-making. Evaluation metrics combine behavioral type matching, detailed consistency, and contextual coherence.
    - **Design Motivation**: One-off generation of complete behavior chains oversimplifies the problem, whereas iterative inference closer aligns with real-world applications (e.g., virtual assistants making real-time decisions).

3. **Multi-dimensional persona information injection**:

    - **Function**: To analyze how persona information granularity impacts behavior simulation.
    - **Mechanism**: The experiments ablate the injection of persona descriptions across multiple levels: (a) basic profile only (age, gender, etc.); (b) profile + personality description; (c) profile + personality + behavior history. Performance gains under different dimensions are evaluated to understand which aspect matters most.
    - **Design Motivation**: Identifying which information is most vital for behavior prediction is critical for designing practical digital twin systems.

### Loss & Training

This work is evaluative and does not involve model training. All models are evaluated in a zero-shot setting via prompting.

## Key Experimental Results

### Main Results

| Model | Behavior Prediction Accuracy | Behavior Chain Consistency | Description |
|------|-------------|------------|------|
| GPT-4o | Highest but still insufficient | Medium | Best commercial model |
| GPT-4 | High | Medium | Slightly inferior to GPT-4o |
| Claude 3 | Medium | Medium-Low | Obvious degradation on long chains |
| LLaMA 3 (70B) | Medium | Medium-Low | Best open-source model |
| LLaMA 3 (8B) | Low | Low | Small models are significantly insufficient |
| Qwen2 | Medium | Medium-Low | Comparable to LLaMA |
| Mistral | Low-Medium | Low | Poor persona adherence |

### Ablation Study

| Information Granularity | Behavior Prediction Improvement | Description |
|---------|------------|------|
| No Persona | Baseline | Scenario-only prediction |
| Basic Profile | +Small margin | Demographical info provides limited help |
| Profile + Personality | +Medium margin | Personality descriptions are helpful |
| Profile + Personality + History | +Largest margin | Historical behaviors serve as the strongest indicator |

### Key Findings

- **Even state-of-the-art models fail to accurately simulate sequential human behaviors**: Although GPT-4o performs the best in single-step predictions, error accumulation over long chains limits its overall consistency. Models tend to generate generic and "safe" behaviors instead of specific, persona-consistent activities.
- **Behavioral history yields the strongest signal**: Static personality descriptions (e.g., "extroverted", "adventurous") provide limited utility, whereas past behavioral logs act as the most predictive cues. This aligns with psychological findings that "past behavior is the best predictor of future behavior".
- **Performance drops rapidly as behavior chains grow longer**: Accuracy degrades significantly with the step distance, showing that models lack a deep understanding of continuity and causation in human behavior.
- **Models exhibit behavior homogenization**: Model predictions tend to converge across different personas, lacking individual variance. For instance, given the scenario of "visiting a grocery store", predictions for highly distinct personas remain highly similar.

## Highlights & Insights

- **Excellent concept definition of 'digital twin'**: Clearly distinguishing between "dialogue simulation" and "behavior simulation" highlights a heavily overlooked evaluation metric. This framework can inspire future works.
- **The behavior chain design** successfully captures the dynamics and inter-dependencies of human actions, which is far more realistic than single-step predictions. This evaluation paradigm can scale to other sequential decision-making domains such as virtual agent generation and caregiving robots.
- **The insight that past behaviors surpass personality descriptions** offers highly practical guidance for digital twin deployment. Priority should be given to sourcing and leveraging user behavior logs instead of relying solely on self-reported personality profiles.

## Limitations & Future Work

- **Dataset Diversity**: Although containing 1,001 personas, it may not span sufficient cultural backgrounds and everyday scenarios. Cross-cultural behavioral variance remains a severe challenge for digital twins.
- **Subjectivity of Evaluation Metrics**: In many cases, behavioral correctness lacks a unique ground truth, as one persona may behave in multiple valid ways in the same context. Current evaluation strategies might underestimate LLMs' capabilities.
- **Zero-shot Only**: The paper does not investigate whether fine-tuning or few-shot in-context learning could significantly boost predictions. Can a few behavioral examples of a specific personal profile boost consistency?
- **Missing Human Baseline**: Can humans accurately predict a target subject's behavior chain solely from their persona profile? What is the upper bound of human performance?
- **Future Work**: Integrating retrieval-augmented generation (RAG) to locate past activities in similar contexts from user behavior logs to boost LLMs' predictive abilities.

## Related Work & Insights

- **vs Dialogue Simulation like PersonaChat**: While dialogue simulations assess whether models can "talk" like a given role, this work benchmarks whether they can "act" accordingly, which is inherently harder due to behavioral causal chains.
- **vs Generative Agents (Park et al., 2023)**: Generative Agents employ LLMs to drive virtual characters in Sims-like contexts but lack systematic quantitative evaluation. BehaviorChain supplies the first standardized benchmark.
- **vs Role-playing Evaluations (e.g., RPBENCH)**: Role-play benchmarks heavily prioritize dialogue stylistic consistency, whereas this work evaluates behavioral decision alignment, serving as complementary evaluation dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ First LLM benchmark focusing on sequential behavior chains with a novel task definition.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive multi-model evaluation but lacks human baselines and fine-tuning trials.
- Writing Quality: ⭐⭐⭐⭐ Robust motivation and clear task formulation.
- Value: ⭐⭐⭐⭐ Introduces the first systematic benchmark for digital twins, an increasingly popular topic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](../../ACL2026/llm_evaluation/beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)
- [\[ACL 2025\] CFBench: A Comprehensive Constraints-Following Benchmark for LLMs](cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)

</div>

<!-- RELATED:END -->
