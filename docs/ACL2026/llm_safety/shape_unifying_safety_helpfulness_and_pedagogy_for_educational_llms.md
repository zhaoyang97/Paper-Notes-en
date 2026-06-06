---
title: >-
  [Paper Note] SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs
description: >-
  [ACL2026][LLM Safety][Educational LLMs] This paper unifies safety, helpfulness, and pedagogy for educational LLMs onto a knowledge mastery graph. It proposes the SHAPE benchmark to evaluate whether models…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Educational LLMs"
  - "Knowledge Mastery Graph"
  - "Pedagogical Safety"
  - "Answer Induction"
  - "Personalized Tutoring"
date: 2026-05-08
content_hash: 02a5d9532c75abde
---

# SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs

**Conference**: ACL2026  
**arXiv**: [2604.22134](https://arxiv.org/abs/2604.22134)  
**Code**: https://github.com/MAPS-research/SHaPE  
**Area**: LLM Alignment / Educational Safety  
**Keywords**: Educational LLMs, Knowledge Mastery Graph, Pedagogical Safety, Answer Induction, Personalized Tutoring

## TL;DR
This paper unifies safety, helpfulness, and pedagogy for educational LLMs onto a knowledge mastery graph. It proposes the SHAPE benchmark to evaluate whether models, under the pressure of answer induction, can still choose between "guidance" or "direct answering" based on the student's mastery state. A graph-augmented gating pipeline is introduced to significantly improve robustness.

## Background & Motivation
**Background**: LLMs have been widely utilized for intelligent tutoring, lesson planning, and learning support. The goal of general-purpose LLMs is typically to resolve user queries as quickly as possible. However, a good teacher in an educational setting does not always provide answers directly; instead, they decide whether to guide, probe, or explain directly based on whether the student has mastered prerequisite knowledge.

**Limitations of Prior Work**: Current educational LLMs often utilize system prompts, pedagogical fine-tuning, or Socratic-style prompting to avoid providing direct answers. However, these methods suffer from two types of failure. First, they lack personalization: repeatedly probing a student on content they already master wastes time. Second, they are easily bypassed by answer-induction prompts, losing pedagogical constraints when students demand "only the final answer."

**Key Challenge**: Educational safety is not simple refusal. Giving a direct answer to a student who has not mastered prerequisite knowledge is unsafe; however, continuing to refuse or beat around the bush for a student who has already mastered relevant concepts is unhelpful. Safety, helpfulness, and pedagogy must be jointly defined relative to the student's knowledge state.

**Goal**: The authors aim to establish a formal definition and evaluation benchmark for educational LLMs to systematically measure whether a model can maintain appropriate pedagogical behavior under both normal requests and answer-induction pressure. They also propose an architecture that enhances robustness without the need for model retraining.

**Key Insight**: The paper uses a knowledge mastery graph to represent concepts and their prerequisite relationships, representing the student's mastery state as a set of concepts. For each question, the system first identifies the concepts required for the solution and their prerequisite scope, then decides whether to allow a direct answer based on the student's mastery state.

**Core Idea**: Transform "whether to provide a direct answer" from a soft rule at the prompt level into explicit gating on a graph structure. When a student lacks relevant concepts, the model can only generate heuristic guidance targeting the missing concepts; when the student has mastered all required concepts, a direct answer becomes the appropriate behavior.

## Method

### Overall Architecture
The paper is divided into two parts. The first part proposes the SHAPE benchmark: based on linear algebra problems from Big-Math, a manually constructed knowledge concept DAG, and simulated student mastery states, it generates 9,087 student-question pairs to evaluate educational LLMs across Safety, Helpfulness, and Pedagogy. The second part introduces a graph-augmented pedagogical pipeline: it parses the knowledge points required for the student's question and compares them with the student's mastery state. If missing concepts exist, it routes to a pedagogical guidance node; otherwise, it routes to a direct answer node.

The key to this framework is not filtering inputs, but passing the same complete user input to the pipeline, explicitly defining at the architectural level which component judges knowledge gaps and which generates the response. This verifies whether improvements stem from knowledge graph gating rather than simply stripping away inductive text.

### Key Designs
1. **Knowledge Mastery Graph-Based Safety Definition**:
    - **Function**: Redefines educational safety from "do not output certain content" to "do not disclose answers directly when the student has not mastered the prerequisites."
    - **Mechanism**: For a question $q$, let $Req(q)$ be the set of concepts and their ancestors required to solve the problem, and $s$ be the student's mastery state. The gating function is $g(q,s)=\mathbb{I}[Req(q)\subseteq s]$. When $g=0$, providing a direct solution is deemed unsafe; when $g=1$, a direct solution is permitted and helpful.
    - **Design Motivation**: The appropriate behavior for the same question varies across different students. A safety definition without mastery state would conflate "pedagogical guidance" with "over-refusal."

2. **Pedagogy Constraints and Teaching Frontier**:
    - **Function**: Defines what makes a safe answer truly pedagogical.
    - **Mechanism**: When a student has knowledge gaps, the concepts mentioned in the output $\phi(Y)$ should fall within the question-induced subgraph $G_q$, the pedagogical target $\tau(Y)$ should belong to the set of unknown concepts, and it should at least hit the "teaching frontier." The frontier represents concepts that are currently missing but whose direct prerequisites have been mastered, making them the most suitable entry points for the next step of teaching.
    - **Design Motivation**: Safe refusal does not equal pedagogy. If a model pivots to irrelevant concepts, repeats mastered content, or simply states "I cannot provide the answer," it should not be considered pedagogical.

3. **Graph-Augmented Routing Pipeline**:
    - **Function**: Implements the formal definition into an executable pedagogical workflow without fine-tuning the base model.
    - **Mechanism**: The pipeline consists of question decomposition/concept mapping nodes, a mastery comparison node, a conditional router, a direct answer node, and a pedagogical response node. The parsing node maps the question to 1-6 solution steps and corresponding knowledge points; the comparison node calculates the set of missing knowledge; if empty, it generates a direct solution, otherwise, it generates heuristic questions for the missing concepts.
    - **Design Motivation**: General system prompts are easily overwritten by strong user goals. Explicit routing moves the judgment of "whether to answer directly" to a structured decision-making stage, reducing the burden on the generative model to perform mid-generation trade-offs.

### Loss & Training
The primary method is a training-free pipeline and does not involve model parameter updates. For benchmark construction, the authors manually built 211 linear algebra concept nodes and selected 1,786 problems from Big-Math where the Llama-8B success rate was between 20%-80%, covering 92 concepts. Through enumeration and prerequisite consistency filtering, they obtained 9,087 valid student-question pairs. For evaluation, 200 pairs were sampled, repeated with multiple random seeds, and judged for Safety, Helpfulness, and Pedagogy using manual inspection combined with a GPT-5 evaluator.

## Key Experimental Results

### Main Results
The baseline evaluation covers models such as Claude, Gemini, GPT-5, Qwen3, EduChat, and SocraticLM, reporting three metrics across default requests and two types of answer-induction settings. Representative models are selected in the table below.

| Model | Default Safety | Worst Induction Safety | Default Helpfulness | Default Pedagogy | Worst Induction Pedagogy | Observation |
|------|----------------|---------------------|---------------------|------------------|------------------------|------|
| GPT-5 | 91.21 | 74.35 | 100.00 | 96.35 | 88.50 | Most stable, but safety still declines |
| GPT-5 mini | 94.77 | 36.10 | 100.00 | 94.49 | 93.42 | Safety drops sharply under strong induction |
| Gemini 2.5 Pro | 99.05 | 4.28 | 98.32 | 80.58 | 27.78 | High default safety, almost fails under induction |
| Qwen3-80B | 99.29 | 0.00 | 1.12 | 70.33 | 0.00 | High default safety stems from severe over-refusal |
| EduChat-32B | 89.12 | 6.12 | 20.75 | 56.49 | 50.00 | Pedagogical fine-tuning does not solve the problem stably |
| SocraticLM-8B | 4.76 | 2.04 | 98.11 | 85.71 | 22.22 | Socratic style does not guarantee safety |

The results indicate that many models appear safe in default scenarios but provide answers directly once students apply strong goal pressure. Some models achieve high safety through over-refusal at the cost of helpfulness.

### Ablation Study
Strictly speaking, the paper does not perform traditional module ablation but compares the robustness changes of the graph-augmented pipeline versus vanilla prompting under worst-case induction conditions.

| Model | Vanilla Worst Safety | Ours Worst Safety | Safety Gain | Vanilla Worst Pedagogy | Ours Worst Pedagogy | Pedagogy Change |
|------|----------------------|-------------------|-------------|-------------------------|---------------------|----------------|
| Qwen3-80B | 0.00 | 92.25 | +92.25 | 0.00 | 70.99 | +70.99 |
| Gemini 2.5 Flash-Lite | 12.35 | 90.85 | +78.50 | 86.54 | 83.72 | -2.82 |
| Gemini 2.5 Pro | 4.28 | 77.46 | +73.18 | 27.78 | 72.18 | +44.40 |
| Claude Opus 4.5 | 24.23 | 92.25 | +68.02 | 82.35 | 68.70 | -13.65 |
| GPT-5 mini | 36.10 | 88.46 | +52.36 | 93.42 | 89.13 | -4.29 |
| GPT-5 | 74.35 | 85.92 | +11.57 | 88.50 | 94.31 | +5.81 |
| Qwen3-32B | 1.66 | 7.04 | +5.38 | 0.00 | 50.00 | +50.00 |

| Analysis Item | Value | Significance |
|--------|------|------|
| SHAPE benchmark Scale | 9,087 pairs | Covers valid student mastery states |
| Concept Graph Scale | 211 nodes | Derived from linear algebra topics and chapters |
| Evaluation Problems | 1,786 | Medium difficulty problems (Llama-8B success 20%-80%) |
| Evaluation Sampling | 200 pairs × 3 seeds | Controls for random error |
| Average Token Cost | baseline 943.25 vs pipeline 1135.15 | Pipeline is more stable but slightly more expensive |

### Key Findings
- The graph-augmented pipeline significantly improves safety without resorting to brainless refusal. The paper reports that under default non-induction settings, the pipeline brings model helpfulness close to or at 100%, indicating it can more accurately distinguish when to give direct answers.
- A slight decrease in the Pedagogy conditional ratio for some models does not mean pedagogical behavior has decreased. For example, Gemini Flash-Lite's Pedagogy ratio dropped by 2.82, but Safety rose by 78.50, meaning the absolute number of pedagogical safe answers still increased significantly.
- Model scale and instruction-following ability impact pipeline effectiveness. Qwen3-80B saw massive gains, while Qwen3-8B/32B remained unstable under strong induction, showing that structured routing still requires the base model to execute node-specific instructions.
- Educational fine-tuned models are not inherently safe. EduChat and SocraticLM still failed significantly on certain metrics, suggesting that style fine-tuning to "talk like a teacher" cannot substitute for knowledge state modeling.

## Highlights & Insights
- The most important contribution of the paper is the redefinition of educational safety: direct answers are not always bad, and refusal is not always good; the key depends on whether the student has mastered prerequisite concepts.
- The definition of the "teaching frontier" has a strong pedagogical flavor. It prevents the model from jumping to overly advanced missing concepts and avoids repeating mastered content, essentially formalizing the "Zone of Proximal Development" (Vygotsky) onto a knowledge graph.
- This paper demonstrates that architectural-level constraints are more reliable than simple system prompts. Expecting an LLM to simultaneously judge safety, helpfulness, and pedagogy during generation is too difficult; offloading the judgment to a graph and router is more stable.
- Alignment research often discusses the helpfulness-safety trade-off. This paper illustrates that the educational context requires a third dimension, pedagogy, otherwise the model might be both safe and useless.

## Limitations & Future Work
- The benchmark only covers linear algebra, which has a relatively clear prerequisite DAG. Knowledge relationships in fields like philosophy, writing, or history may not be suitable for expression via strict directed prerequisite graphs.
- Student mastery states are simplified into binary variables and assume prerequisite consistency; real-world learning involves partial mastery, misconceptions, forgetting, and "leapfrog mastery."
- Concept extraction and pedagogy judgment rely on GPT-5 evaluator approximations; automatic evaluation itself still has errors.
- The current pedagogical strategy is to ask questions one by one for missing concepts; multi-turn pedagogical paths have not yet been optimized. Future work could integrate knowledge tracing or reinforcement learning to dynamically update student states.
- The average token cost of the pipeline is higher than the baseline; practical deployment requires a trade-off between robustness, cost, and latency.

## Related Work & Insights
- **vs LearnLM / Study Mode prompts**: These methods primarily rely on system instructions to shape pedagogical style. SHAPE binds pedagogical decisions to a student mastery graph and explicit gating.
- **vs SocraticLM**: SocraticLM emphasizes Socratic dialogue but lacks a formalized mastery state. This paper points out that a fixed Q&A style cannot handle the direct solution needs of students who have already mastered the content.
- **vs General LLM Safety**: General safety focuses on harmful content, while educational safety focuses on direct answers that are harmful to the learning process; the objects of restriction differ.
- **Inspiration for ITS**: LLM tutors should not just have a "prompt persona"; they also require auditable student models, knowledge graphs, and routing strategies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Standardizing safety, helpfulness, and pedagogy using knowledge mastery graphs is very distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad model coverage and clear metrics, though subject range and real student experiments are still lacking.
- Writing Quality: ⭐⭐⭐⭐ Solid definitions and motivation. The appendix contains many attack evaluation details, requiring prioritized reading.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for educational LLM product design, evaluation, and alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Responsible Federated LLMs via Safety Filtering and Constitutional AI](responsible_federated_llms_via_safety_filtering_and_constitutional_ai.md)
- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs](parasite_conditional_system_prompt_poisoning_to_hijack_llms.md)
- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)

</div>

<!-- RELATED:END -->
