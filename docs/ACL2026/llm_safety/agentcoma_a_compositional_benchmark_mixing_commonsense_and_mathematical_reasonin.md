---
title: >-
  [Paper Note] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
description: >-
  [ACL2026][LLM Safety][Mixed-type Compositional Reasoning] AgentCoMa constructs an agentic benchmark that forcibly combines commonsense selection with single-step mathematical operations. Evaluations across 61 LLMs reveal that while models typically solve both sub-problems independently (80%), the average accuracy drops to 51% when combined, exposing significant vulnerabilities in mixed-type compositional reasoning.
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Mixed-type Compositional Reasoning"
  - "Commonsense Reasoning"
  - "Mathematical Reasoning"
  - "Agent Evaluation"
  - "Model Vulnerability"
date: 2026-05-08
content_hash: 49cc3129173822e3
---

# AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios

**Conference**: ACL2026  
**arXiv**: [2508.19988](https://arxiv.org/abs/2508.19988)  
**Code**: https://agentcoma.github.io  
**Area**: LLM Safety / LLM Reasoning Evaluation  
**Keywords**: Mixed-type Compositional Reasoning, Commonsense Reasoning, Mathematical Reasoning, Agent Evaluation, Model Vulnerability  

## TL;DR
AgentCoMa constructs an agentic benchmark that forcibly combines commonsense selection with single-step mathematical operations. Evaluations across 61 LLMs reveal that while models typically solve both sub-problems independently (80%), the average accuracy drops to 51% when combined, exposing significant vulnerabilities in mixed-type compositional reasoning.

## Background & Motivation
**Background**: Current LLMs have numerous benchmarks for both commonsense and mathematical reasoning. Commonsense benchmarks focus on spatial, temporal, social, and causal knowledge in daily scenarios, while mathematical benchmarks cover difficulties from elementary word problems to competition math. Additionally, agentic benchmarks often incorporate complex factors like tool calls, long task chains, and dynamic environments to measure the overall capabilities of LLM agents.

**Limitations of Prior Work**: While these evaluations are valuable, they struggle to answer a specific question: when a task requires both commonsense judgment and mathematical computation, is the model failing at "composition" itself, or is it merely burdened by extra factors like long contexts, tool calls, or environmental changes? If a benchmark contains too many distractors, it cannot precisely locate the source of error.

**Key Challenge**: Real-world agent tasks frequently require switching across reasoning types—for instance, using commonsense to identify items that can be stored at room temperature before calculating their total price. However, most existing compositional benchmarks combine steps of the same reasoning type, such as two knowledge retrieval steps or two math steps. High performance on a single reasoning type does not guarantee that a model can stably link different types of reasoning.

**Goal**: The authors aim to construct a controlled evaluation environment where every sample can be decomposed into a commonsense sub-problem and a math sub-problem, both of which are easy for humans and strong models. In this way, if a model fails the composite task, the failure can be clearly attributed to the "cross-type composition" itself rather than the difficulty of individual steps.

**Key Insight**: The paper selects commonsense and math as two complementary reasoning types: the former resembles fast, intuitive System 1 thinking, while the latter resembles slow, explicit System 2 calculation. Placing both into the same realistic agent scenario tests whether the model can truly mobilize different capabilities within a single reasoning chain.

**Core Idea**: By using manually constructed tasks of "commonsense selection + single-step arithmetic," the authors isolate mixed-type compositional reasoning in agent scenarios from other complexities. They then locate LLM compositional vulnerabilities through sub-problem controls, human experiments, and interpretability analysis.

## Method

### Overall Architecture

Rather than proposing a new model, this paper introduces a benchmark and a diagnostic protocol to answer the core question: when a task involves both commonsense and math, does the model fail due to "compositional failure" or other overheads? Each problem includes a "composite problem" and "two independently evaluable sub-problems," allowing experiments to compare model performance across three input forms: commonsense only, math only, and the combined two-step task.

In AgentCoMa, the input is a realistic task description for an LLM agent, and the output is typically a numerical answer. A sample contains four core objects: the composite problem, the commonsense sub-problem, the math sub-problem, and their respective ground truths. The composite problem first requires the model to make a commonsense selection from multiple options, then perform an arithmetic operation on the selected objects. For example, in a garage tool organization task, the model must first identify that a power drill, extension cords, and leaf blower are electrical appliances, then sum their quantities. The commonsense sub-problem only asks "which items belong in the waterproof cabinet," while the math sub-problem explicitly states the selection results and only requires addition. This separates "knowledge selection failure" from "composition failure." The dataset covers five agentic scenarios: house working, web shopping, science experiments, smart assistant, and travel agent. The math step involves only one basic arithmetic operation, and scenarios/operation types are balanced across development and test sets. The final scale is 260 samples (80 dev + 180 test), with no training set as it targets pre-trained or instruction-tuned models.

### Key Designs

**1. Decomposable Mixed-Reasoning Samples: Explicitly splitting composite problems into commonsense and math sub-problems for a three-way comparative evaluation.**

Traditional benchmarks only report failure without specifying if the model lacks commonsense, arithmetic skills, or the ability to link them. AgentCoMa ensures that the commonsense step determines the operands for calculation, and the math step remains a simple numerical operation. Sub-problems remove the burden of the opposite reasoning type. If a model solves both sub-problems but fails the composite one, the failure is clearly rooted in cross-type composition rather than single-step incapacity—offering greater diagnostic power than final accuracy alone.

**2. Manual Curation and Multi-round Expert Verification: Ensuring the "simple sub-problems vs. difficult composition" relationship holds.**

The core findings rely entirely on the premise that sub-problems are easy. If samples are ambiguous or contain implicit multi-step math, the comparison fails. Thus, quality control is part of the methodology. All samples are handcrafted by experts rather than generated by LLMs, followed by binary checks, independent solving, answer comparison, and ambiguity feedback. Any failure in these checks requires rewriting and re-verification. To mitigate author bias, all verifications (except for one evaluation step) were performed by experts other than the sample authors.

**3. Behavioral-to-Mechanical Diagnostic Chain: Probing the source of the compositional gap beyond accuracy reporting.**

Accuracy drops could be hastily attributed to context length or poor prompting. The authors compare the AgentCoMa gap with Bamboogle and MultiArith and check if extra context explains the performance drop. They then use Min-K%++ to estimate the similarity of mixed-type problems to the training distribution, utilize lookback attention ratio to analyze context utilization, and use QRNCA to compare overlapping activated neurons between composite and sub-problems. This multi-layered evidence supports a stronger explanation: mixed-type tasks are rarer in training distributions, and during reasoning, models tend to activate math-related circuits without simultaneously mobilizing commonsense-related circuits.

### Loss & Training
No new models were trained; thus, no model loss function is defined. The experiment employs a unified reasoning and evaluation strategy: all LLMs use two-shot chain-of-thought (CoT) prompts and greedy decoding. Numerical answers are extracted from CoT outputs using regex for exact matching with ground truths. Non-numerical answers for commonsense sub-problems are judged by an LLM-as-a-judge against standard answers. The authors also tested self-ask decomposition prompting, finding that the average composition gap remains similar to the CoT setting, suggesting simple decomposition prompts do not eliminate the issues exposed by AgentCoMa.

Human control experiments involved 45 non-expert crowdworkers with high school education and English fluency, performing calculations manually without tools. Each annotator answered 12 questions; the composite problem and sub-problems of the same sample were never assigned to the same person to prevent answer leakage.

## Key Experimental Results

### Main Results
The paper evaluates 61 LLMs, highlighting 16 recent models covering instruction-tuned, SFT reasoning, and RL reasoning strategies, with sizes ranging from 3B to 141B. The core finding is that while models can solve both independent sub-steps for 80% of samples on average, the average accuracy for composite problems is only 51%, resulting in a 29% average compositionality gap.

| Target / Benchmark | Both Sub-steps Correct Individually | Composite Accuracy | Composition Gap / Description |
|--------------------|-------------------------------------|-------------------|-------------------------------|
| Avg. 16 Representative LLMs on AgentCoMa | 80% | 51% | 29% average gap; main issue is composition, not single-step ability |
| Non-expert Humans on AgentCoMa | 78.9% | 82.8% | No significant collapse; composite accuracy even slightly higher |
| Avg. LLMs on Bamboogle | 53% | 52% | Homogeneous knowledge reasoning; negligible gap |
| Avg. LLMs on MultiArith | Near perfect | Near perfect | Homogeneous math reasoning; gap < 1% |

Even strong models are not immune. For example, Llama3.3 70B IT shows a 90.0% individual sub-step success rate vs. 73.3% composite accuracy; Qwen2 14B shows 88.9% vs. 60.6%; SimpleRL 32B shows 93.9% vs. 66.7%. The collapse is more pronounced in smaller or weaker models; SimpleRL 8B solves sub-steps individually 56.7% of the time but achieves only 25.0% composite accuracy.

### Ablation Study
The "ablations" in this paper function as diagnostic experiments, systematically ruling out context length, prompt decomposition, and homogeneous composition difficulty while observing attention and neuron patterns.

| Analysis Item | Key Result | Description |
|---------------|------------|-------------|
| Failure Source Decomposition | ~0.74 of AgentCoMa failures come from samples where both sub-steps were solved individually | Failure is not due to lack of single-step skills, but the inability to reliably combine different types of reasoning |
| Self-Ask Prompting | Avg. gap ~27% (vs. 29% in CoT) | Explicit decomposition prompts provide only minor relief, not a solution |
| Lookback Attention | Commonsense: 71.49, Math: 72.20, Composition: 70.75 | Models show lower lookback attention to context in composite problems, leading to more context hallucinations |
| Neuron Overlap: Llama 3.1 8B | Comp-Math: 39%, Comp-Commonsense: 3% | Composite problems tend to activate mathematical circuits rather than simultaneously activating commonsense circuits |
| Neuron Overlap: GeneralReasoner 4B | Comp-Math: 54%, Comp-Commonsense: 10% | Reasoning models show a similar bias, indicating the problem persists beyond standard instruction-tuned models |

### Key Findings
- The gap in AgentCoMa is significantly larger than in homogeneous reasoning benchmarks. MultiArith (math + math) and Bamboogle (knowledge + knowledge) show composite accuracies close to their joint sub-step probabilities, whereas AgentCoMa shows a clear disconnect.
- Reasoning optimization via RL or SFT does not eliminate the problem. The paper notes that reasoning models exhibit large compositional gaps similar to instruction-tuned models, suggesting that "long-chain reasoning" does not equate to "cross-type composition."
- Training distribution similarity analysis supports the explanation that mixed commonsense and math tasks are relatively rare. Min-K%++ scores suggest composite problems are less similar to typical training patterns than standalone commonsense or math problems.
- Mechanism analysis provides a concrete failure picture: when facing composite problems, models often follow math-related neuron patterns while commonsense-related neurons are not sufficiently activated, leading to formally fluent but contextually incorrect reasoning.

## Highlights & Insights
- The most valuable design is the triplet structure: "Composite Problem + Two Sub-problems." It transforms ambiguous error diagnosis into a quantifiable comparison: models do not fail because they don't know how to shop or how to add, but because they fail to link "what to buy" with "how much to pay."
- The difficulty of AgentCoMa lies in switching reasoning types rather than absolute task difficulty. This is crucial for agent evaluation, as real-world failures often stem from failing to combine simple capabilities in the correct order rather than a lack of advanced math or planning.
- The paper goes beyond leaderboards by connecting behavioral gaps to training distributions, attention, and neuron overlap. While these analyses are not causal proofs, they make the conclusion that "mixed-type tasks are under-learned patterns" highly credible.
- It provides insights for safety and reliability: if a model cannot stably combine commonsense and arithmetic in a controlled, low-risk, short-context setting, additional step-by-step verification and intermediate state checks are needed when deploying models as automated decision-makers in real agent scenarios.

## Limitations & Future Work
- AgentCoMa is a deliberately simplified controlled experiment. Each problem only combines two reasoning types, and math is limited to basic operations. While useful for pinpointing issues, it does not cover long task chains, multiple constraints, multi-turn interactions, or tool calls in real agent tasks.
- The data scale is relatively small, with 180 test samples and 260 total. This is comparable to classic benchmarks like Bamboogle and MultiArith, but larger datasets are needed to systematically analyze scenarios, arithmetic types, reasoning orders, and linguistic variants.
- The paper only briefly explores the impact of reasoning order. The authors mention that constructing a full reversed-order dataset requires extensive expert annotation, so the difference between "Math then Commonsense" and "Commonsense then Math" remains to be systematically answered.
- Automatic evaluation of commonsense sub-problems relies on LLM-as-a-judge. Although validated against human assessments, judge bias might still affect fine-grained conclusions in more open-ended response spaces.
- Future work could extend to more mixed types, such as commonsense + constrained planning, spatial reasoning + arithmetic, or social commonsense + resource allocation, and integrate the benchmark into actual agent traces to see if controlled gaps predict real-world failures.

## Related Work & Insights
- **vs. Commonsense Benchmarks**: Where traditional benchmarks test if models possess daily knowledge, this work treats commonsense as one link in a chain to see if it can synergize with explicit computation.
- **vs. Math Benchmarks**: MultiArith and GSM-style tasks examine math skills or word problem solving. In AgentCoMa, the math itself is easy; the true test is whether the model uses commonsense to select the correct operands first.
- **vs. Bamboogle / MultiArith Compositional Reasoning**: These benchmarks also have multi-step structures but usually combine the same type of reasoning. The key difference here is cross-type composition, which reveals vulnerabilities invisible in homogeneous tasks.
- **vs. Agentic Benchmarks**: Many agent benchmarks include tool calls, long horizons, and dynamic environments. They are closer to real deployment but harder for attribution. AgentCoMa sacrifices some realism for a clearer causal diagnostic window.
- **Insights for Future Research**: Training and inference methods should not only reward final answers or long CoTs but should explicitly supervise intermediate type-switching. For instance, training models to first label the required reasoning type or output selected objects before calculating, or using verifiers to ensure commonsense selections are correctly passed to math steps.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Selects "Commonsense + Math" mixed-type composition as a controlled benchmark; the problem definition is clear and distinct.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid evaluation across 61 LLMs, human controls, benchmarking against similar tasks, and various interpretability analyses, though scale and reversed-order analysis could be expanded.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, logically progressing from data construction to behavioral results to mechanical analysis. Main conclusions are well-supported by evidence.
- Value: ⭐⭐⭐⭐⭐ High value for LLM agent reliability assessment, reminding us not to mistake individual reasoning skills for stable compositional ability in real scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers](../../ICLR2026/llm_safety/mcp-safetybench_a_benchmark_for_safety_evaluation_of_large_language_models_with_.md)
- [\[ICLR 2026\] CIMemories: A Compositional Benchmark For Contextual Integrity In LLMs](../../ICLR2026/llm_safety/cimemories_a_compositional_benchmark_for_contextual_integrity_in_llms.md)
- [\[ICLR 2026\] MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement](../../ICLR2026/llm_safety/mscr_exploring_the_vulnerability_of_llms_mathematical_reasoning_abilities_using_.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](../../ICLR2026/llm_safety/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)

</div>

<!-- RELATED:END -->
