---
title: >-
  [Paper Note] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States
description: >-
  [ACL2025][LLM Evaluation][Theory of Mind] This paper proposes the DynToM benchmark, evaluating LLM capabilities in tracking the temporal evolution of human mental states across 5,500 temporally linked scenes within 1,100 social contexts and 78,100 questions, revealing that models lag behind humans by an average of 44.7%.
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "Theory of Mind"
  - "Dynamic Reasoning"
  - "Mental State Tracking"
  - "Social Cognition"
date: 2026-05-08
content_hash: 0fdcb0d4e184e2b0
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States

**Conference**: ACL2025  
**arXiv**: [2505.17663](https://arxiv.org/abs/2505.17663)  
**Code**: [GitHub & HuggingFace](https://github.com/)  
**Area**: LLM Evaluation  
**Keywords**: Theory of Mind, Dynamic Reasoning, Mental State Tracking, Social Cognition, LLM Evaluation

## TL;DR
This paper proposes the DynToM benchmark, evaluating LLM capabilities in tracking the temporal evolution of human mental states across 5,500 temporally linked scenes within 1,100 social contexts and 78,100 questions, revealing that models lag behind humans by an average of 44.7%.

## Background & Motivation
1. **Theory of Mind is the foundation of social interaction**: Theory of Mind (ToM) is the core capacity to understand others' beliefs, emotions, and intentions. As LLMs increasingly participate in human-computer interactions (such as psychological support dialogues), evaluating their ToM capability becomes crucial.
2. **Existing evaluations are limited to static snapshots**: Benchmarks such as SocialIQA, BigToM, and TOMBENCH mainly evaluate static mental states in isolated scenarios, overlooking the key trait in real-world social interactions where mental states **persistently evolve over time**.
3. **Lack of temporal-dimension ToM evaluation**: No benchmarks systematically capture mental state transitions across multiple consecutive scenarios—for example, how a user's belief gradually shifts due to a sequence of conversational events.
4. **Complex causal relationships exist among mental states**: Psychological research shows that beliefs affect emotions, beliefs and emotions jointly influence intentions, and all three collectively drive behaviors—requiring a framework to model the temporal evolution of these interdependent relationships.
5. **Compositional reasoning is a weakness of LLMs**: Tracking mental state changes across multiple scenarios requires compositional reasoning. Prior studies indicate that Transformer performance degrades significantly as compositional complexity grows.
6. **Dynamic understanding is indispensable in practical applications**: Applications such as AI counseling, emotional companionship, and social simulation require models to understand how a user's mental state changes as the interaction progresses; static evaluations fail to measure this capability.

## Method

### Four-Step Construction Framework

**Step 1: Social Context Construction**
- Collected 261 social locations across 13 categories (such as workplaces, educational locations, etc.).
- Sampled character attributes from the US census data pool: name, gender, occupation, education, race, and personality traits (7 dimensions).
- Each social context contains 1 location + 2 characters + character relationships.
- GPT-4-Turbo generated character relationships based on 4 exemplars, verified by 4 annotators, achieving a retention rate of 92%.

**Step 2: Mental State Trajectory Design**
- Tracked 4 mental states: belief, emotion, intention, and action.
- Modeled three causal relationships based on psychological theory (D'Andrade, 1995): belief $\rightarrow$ emotion; belief + emotion $\rightarrow$ intention; belief + emotion + intention $\rightarrow$ action.
- Designed mental state evolution trajectories across 5 temporal scenarios for each context, including specific cues triggering state transitions.
- 4 annotators rated trajectories on a 5-point scale across three dimensions: coherence, rationality, and realism. Trajectories scoring below 4.0 were filtered out, with a retention rate of 85.4%.

**Step 3: Scenario Generation**
- Based on the trajectories from Step 2, GPT-4-Turbo generated background descriptions and natural dialogue for each scenario.
- The dialogue format naturally portrays changes in character mental states.
- 4 annotators evaluated consistency, coherence, and realism. Scenes with scores below 4.0 were regenerated, resulting in a retention rate of 88.7%.

### Four Types of Questions Design

| Question Type | Evaluation Goal | Difficulty |
|---------|---------|------|
| **Understanding** | Identify specific mental states within a single scenario | Basic |
| **Transformation-1** | Detect whether states change between adjacent scenarios | Intermediate |
| **Transformation-2** | Understand causal mechanisms behind state transitions | Challenging |
| **Transformation-3** | Track the state evolution sequence across all scenarios | Most Difficult |

- Option design leverages trajectory information: the correct answer is derived from the target state, while distractors come from other states in the same scenario or the same state type in other scenarios.
- Ultimately verified by annotators in terms of clarity and answerability, a total of **78,100** multiple-choice questions were gathered.

### Dataset Scale
- 1,100 social contexts, 2,200 characters, and 261 locations.
- 5,500 scenarios, with an average scenario length of 457.9 words.
- 78,100 questions: Understanding 28.2%, Transformation-1 22.5%, Transformation-2 43.7%, Transformation-3 5.6%.

## Key Experimental Results

### Table 1: LLM Performance on DynToM (Accuracy %)

| Model | Understanding Avg | Transformation Avg | Overall Avg |
|------|-------------------|-------------------|--------|
| **Human Baseline** | 82.3 | 76.6 | **77.7** |
| GPT-4o | 88.8 | 49.5 | **64.0** |
| Llama-3.1-70B | 83.6 | 42.5 | **57.1** |
| Qwen2-72B | 81.7 | 32.3 | **48.5** |
| GPT-4-Turbo | 72.5 | 34.5 | **47.6** |
| Llama-3.1-8B | 30.2 | 17.5 | **22.3** |
| DeepSeek-V2 | 4.5 | 7.6 | **7.2** |

**Key Findings**: The average LLM performance is only 33.0%, lagging behind humans by 44.7%. Performance drops precipitously from Understanding to Transformation (on average $48.2\% \rightarrow 24.7\%$), showing that models can identify static states but struggle to track dynamic changes.

### Table 2: GPT-4o Error Type Analysis

| Error Type | Ratio | Meaning |
|---------|------|------|
| Full error | 50-58% | Failure in both state identification and transition reasoning |
| Local error | 13-18% | Correct state identification but failure in transition reasoning |
| Restoration error | 8-16% | Correct transition reasoning but failure in state identification (surface pattern matching) |
| Fully correct | 13-17% | Both correct |

**Key Findings**: Full errors dominate, with the belief state showing the highest error rate (58%). The presence of Restoration errors suggests models rely on shallow pattern matching rather than genuine understanding.

### "Lost in the Middle" Phenomenon
- Models perform worst in the middle scenarios (spans 2-3, 3-4), showing a U-shaped curve.
- In sequences of 7 scenarios, the accuracy for span 3-4 is only 26%.
- After truncating subsequent scenarios, the accuracy in the middle improved by 21 percentage points ($26\% \rightarrow 47\%$).

## Highlights & Insights
1. **First Dynamic ToM Benchmark**: Moves from static snapshot evaluations to temporal evolution assessments, filling a crucial gap in evaluating dynamic mental state tracking.
2. **Systematic Four-Step Construction Framework**: Social contexts $\rightarrow$ psychological trajectories $\rightarrow$ scenario generation $\rightarrow$ question design, with rigorous human validation at each step.
3. **Large Scale of 78,100 Questions**: Far exceeding existing ToM benchmarks (e.g., TOMBENCH's 2,860 questions, BigToM's 600 questions), offering stronger statistical power.
4. **Ingenious Progressive Question Design**: Gradually scaling from Understanding $\rightarrow$ Transformation-1/2/3, enabling precise pinpointing of where the model's reasoning chain breaks.
5. **Discovery of "Lost in the Middle" in ToM**: Validates this phenomenon in social cognition tasks for the first time, providing crucial evidence for long-context ToM reasoning research.

## Limitations & Future Work
1. **Limited Model Coverage**: Only 10 models were evaluated, lacking the Claude family and recent open-source models (e.g., Qwen2.5, DeepSeek-V3).
2. **Single Prompting Method**: Only vanilla and CoT prompting were used, without exploring methods like Think-Twice or Self-Consistency, which may be more suitable for ToM tasks.
3. **Extendable Mental State Types**: Covers only belief, emotion, intention, and action, leaving cognitive dimensions like knowledge or desire unaddressed.
4. **Fixed Scenario Count of 5**: Although adjustable as mentioned in the paper, the systematic impact of lengths other than 5 on performance was not explored (with only preliminary 6/7 scenario experiments).
5. **Text-Only Modality**: Real-world social interactions convey mental states through multimodal signals such as facial expressions and tone, which are currently not covered by this benchmark.
6. **Reliance on GPT-4-Turbo for Data Generation**: Potential biases of the generative model itself may be introduced, and the naturalness of some scenarios could be limited.

## Related Work & Insights

### vs. TOMBENCH (Chen et al., 2024)
TOMBENCH contains 2,860 questions evaluating basic ToM capabilities, but **only tests static mental states in isolated scenarios**. DynToM utilizes 5 temporally linked scenarios to form a continuous narrative, evaluating the **dynamic evolution and causal chains** of mental states, which aligns closer with real-world social interactions. While most LLMs exhibit near-perfect performance on TOMBENCH, the best model on DynToM achieved only 64%.

### vs. BigToM (Gandhi et al., 2023)
BigToM consists of only 600 questions, containing social locations and character relationships but **without tracking temporal changes in mental states**. DynToM's scale (78,100 questions) and dynamic evaluation design far exceed those of BigToM, revealing critical deficiencies of LLMs in state tracking through Transformation questions.

### vs. OpenToM (Xu et al., 2024)
OpenToM includes 2,384 questions, featuring social locations and interdependent mental states, but **lacks dynamic mental states and progressive question design**. DynToM is the only benchmark that simultaneously incorporates detailed character profiles, character relationships, interdependent mental states, dynamic evolution, and a large-scale question bank.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically evaluates LLM capabilities in dynamic Theory of Mind for the first time; both framework design and problem stratification are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 models + human baseline + in-depth error analysis + Lost-in-the-Middle validation.
- Writing Quality: ⭐⭐⭐⭐ Structure is complete and clear; the design driven by psychological theory is highly convincing.
- Value: ⭐⭐⭐⭐⭐ Unveils fundamental shortcomings of LLMs in dynamic social cognition, offering significant insights for the field of human-computer interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](../../ICML2025/llm_evaluation/position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)

</div>

<!-- RELATED:END -->
