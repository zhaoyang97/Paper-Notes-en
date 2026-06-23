---
title: >-
  [Paper Note] Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment
description: >-
  [ICLR 2026][Interpretability][Paper Note] This paper moves beyond merely evaluating whether the "answers" of legal LLMs are correct. Instead, it **faithfully decomposes the model's score for each judgment into a set of AND/OR interaction patterns between input phrases**. Sixteen legal experts then labeled these phrases as "relevant / irrelevant / forbidden" to
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: ff1d3ec208a318df
---
# Can LLMs Reason Soundly in Law? Auditing Inference Patterns for Legal Judgment

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=5T0BXtJxzN](https://openreview.net/forum?id=5T0BXtJxzN)  
**Code**: To be confirmed  
**Area**: Interpretability / Legal LLM Auditing  
**Keywords**: Interaction interpretability, AND-OR logic model, Legal judgment, Reasoning reliability, Identity bias

## TL;DR
This paper moves beyond merely evaluating whether the "answers" of legal LLMs are correct. Instead, it **faithfully decomposes the model's score for each judgment into a set of AND/OR interaction patterns between input phrases**. Sixteen legal experts then labeled these phrases as "relevant / irrelevant / forbidden" to quantify "what logic the model actually uses to reach its decision." The results reveal that even when four major LLMs (including legal-specific ones) provide correct judgments, **more than half of the reasoning interactions are based on irrelevant or even incorrect grounds**—such as attributing one person's criminal behavior to another defendant or being biased by professional identity.

## Background & Motivation
**Background**: Legal Judgment Prediction (LJP) is a typical high-risk application for LLMs. Specialized benchmarks and models such as LawBench, LegalBench, SaulLM, and BAI-Law have been developed. The dominant evaluation method is to verify the accuracy of the generated results (judgment categories)—high accuracy is equated with reliability.

**Limitations of Prior Work**: The authors find that "correct output tokens" and "correct reasoning processes" are two different matters. Even if a top-tier LLM generates the correct judgment token, the internal logic it relies on to calculate that token may be deeply flawed. In legal contexts, this problematic logic can steer the model's choice between multiple "seemingly plausible" judgments, which should ideally be left to judicial discretion. Once a model employs biased grounds, it introduces unfairness and risk, which are entirely masked by the "correct result."

**Key Challenge**: Traditional evaluations only consider language generation results (a scalar score or category) and cannot answer the question: "What logic did the model use to arrive at this result?" Auditing the reasoning process faces a fundamental hurdle: can LLM predictions be **faithfully** decomposed into discrete reasoning units that are human-readable and mathematically guaranteed to be equivalent? If the decomposition itself is unfaithful, the audit is effectively meaningless.

**Goal**: (1) Identify a theoretically guaranteed method to decompose LLM judgments into discrete reasoning patterns; (2) Use legal domain knowledge to define "reliable vs. unreliable" reasoning; (3) Quantify the extent of unreliable reasoning and its impact on judgments; (4) Categorize common representational defects.

**Key Insight**: The authors' team previously proved in interpretability theory that deep network output scores can be **faithfully represented by the interactions between input features**, with mathematical guarantees of "universal matching" and "sparsity." They adapted this interaction explanation tool to legal LLMs, treating each interaction as a "reasoning pattern" utilized by the model.

**Core Idea**: Use **AND-OR interactions** to faithfully decompose judgment scores into discrete reasoning patterns. By leveraging legal experts' labels of phrases as "relevant / irrelevant / forbidden," these interactions are partitioned into **reliable effects** and **unreliable effects**. Finally, a ratio metric $s_{\text{reliable}}$ is used to measure "how much of the model's judgment basis aligns with human experts."

## Method

### Overall Architecture
The method is an auditing pipeline of "**faithful decomposition, domain adjudication, and quantification**." It does not involve training a better model but rather performing a "logic checkup" on existing LLMs. Given a legal case $x$ (segmented into $n$ semantic phrases) and the model's output score $v(x)$ for a specific judgment (e.g., "Assault"), the process is:

1.  **Interaction Extraction**: Decompose $v(x)$ into an AND-OR logic model $h(x)$, consisting of weighted interaction patterns $I_S^{\text{AND}}, I_S^{\text{OR}}$. Theoretically, this matches $v$ across all $2^n$ possible states of input masking, allowing each interaction to be viewed as an actual reasoning pattern used by the model.
2.  **Phrase Labeling**: Legal experts categorize input phrases into Relevant $R$, Irrelevant $I$, and Forbidden $F$.
3.  **Reliability Decomposition**: Based on the phrases involved in each interaction, its numerical effect is split into reliable effects ($R_S^{\text{AND}}, R_S^{\text{OR}}$) and unreliable effects ($U_S^{\text{AND}}, U_S^{\text{OR}}$).
4.  **Metric Quantification**: Reasoning quality is characterized through the ratio of reliable effects $s_{\text{reliable}}$, the distribution of interaction complexity (order), and the degree of conflict cancellation $s_{\text{conflict}}$. Representational defects are visualized in specific cases.

The key is that the decomposition is a "post-hoc audit" without modifying the model; the validity of this audit rests on the mathematical guarantees of the interaction representation (universal matching + sparsity).

### Key Designs

**1. Faithfully decomposing judgment scores into reasoning patterns via AND-OR interactions**

For an audit to be valid, the decomposition must be faithful. The authors use an AND-OR logic model to represent output scores. Judgment scores are defined as the sum of log-odds for the target token sequence: $v(x)=\sum_{t=1}^{m}\log\frac{p(y=y_t\mid x, Y_t^{\text{prev}})}{1-p(y=y_t\mid x, Y_t^{\text{prev}})}$. The logic model is constructed as: $h(x_{\text{mask}})=h(b)+\sum_{S\in\Omega^{\text{AND}}}\mathbb{1}^{\text{AND}}(S\mid x_{\text{mask}})\,I_S^{\text{AND}}+\sum_{S\in\Omega^{\text{OR}}}\mathbb{1}^{\text{OR}}(S\mid x_{\text{mask}})\,I_S^{\text{OR}}$. An AND trigger function returns 1 only if **all** phrases in set $S$ are present; an OR trigger function returns 1 if **any** phrase in $S$ is present. $I_S$ is the scalar weight.

Its efficacy is guaranteed by two theorems: **Universal Matching** (Theorem 1) proves that if weights are set as $I_S^{\text{AND}}=\sum_{T\subseteq S}(-1)^{|S|-|T|}v^{\text{and}}(x_T)$, then $v(x_T)=h(x_T)$ for all $2^n$ masking states—meaning interactions are not approximations but exactly equivalent across exponential perturbations. **Sparsity** ensures only $O(n^p)$ interactions have significant numerical values (empirically $1.5\le p\le2.0$), allowing the removal of near-zero weights. Together, these allow the model's reasoning to be represented by a few human-readable interactions, distinguishing this work from standard attribution methods like Shapley values or saliency maps.

**2. Phrase labeling (Relevant/Irrelevant/Forbidden) to ground adjudication in domain knowledge**

Interactions lack inherent "correctness" without human legal knowledge. Sixteen legal experts and volunteers categorized phrases $N$ into three disjoint sets $R\cup I\cup F=N$: **Relevant phrases $R$** directly constitute the grounds for judgment (e.g., for "Assault," phrases like "chased / with an axe / bit / slightly injured" are direct grounds); **Irrelevant phrases $I$** describe the defendant but do not decide the judgment (e.g., "morning / had an argument"); **Forbidden phrases $F$** are sensitive and misleading (e.g., phrases describing **another person's** actions, like "hit / with a shovel / death" describing Bob's act rather than Andy's).

**3. Decomposing interaction effects and defining the reliability ratio**

The third step "projects" phrase labels onto interaction effects. For **AND interactions**, a binary criterion is used: it is reliable if and only if $S$ contains no forbidden phrases and at least one relevant phrase; otherwise, the entire interaction is unreliable:
$$\text{if } S\cap F=\varnothing,\ S\cap R\neq\varnothing:\ R_S^{\text{AND}}=I_S^{\text{AND}},\ U_S^{\text{AND}}=0;\quad \text{otherwise } R_S^{\text{AND}}=0,\ U_S^{\text{AND}}=I_S^{\text{AND}}$$
For **OR interactions**, effects are distributed using an "even split": $R_S^{\text{OR}}=\frac{|S\cap R|}{|S|}I_S^{\text{OR}}$, $U_S^{\text{OR}}=\left(1-\frac{|S\cap R|}{|S|}\right)I_S^{\text{OR}}$. The **Reliability Ratio** is defined as:
$$s_{\text{reliable}}=\frac{\sum_{S\in\Omega^{\text{AND}}}|R_S^{\text{AND}}|+\sum_{S\in\Omega^{\text{OR}}}|R_S^{\text{OR}}|}{\sum_{S\in\Omega^{\text{AND}}}|I_S^{\text{AND}}|+\sum_{S\in\Omega^{\text{OR}}}|I_S^{\text{OR}}|}\in[0,1]$$
A value closer to 1 indicates the model's reasoning aligns better with expert logic.

**4. Interaction complexity and conflict cancellation metrics**

The **Interaction Complexity (Order)**, $|S|$, reveals the "shape" of reasoning. Statistical analysis shows that models **strongly prefer low-order interactions**, suggesting they rely on a few local phrases for heuristic guessing (similar to a bag-of-words) rather than a comprehensive synthesis of case elements. This challenges the assumption that CoT prompting enables long-chain reasoning. The **Conflict Cancellation Degree**, $s_{\text{conflict}}=1-\frac{\sum_{op}|\sum_{S\in\Omega^{op}}I_S^{op}|}{\sum_{op}\sum_{S\in\Omega^{op}}|I_S^{op}|}$, measures the proportion of positive and negative effects that cancel each other out. Experiments show over 60% of effects are canceled, indicating significant contradictory noise in model judgments.

## Key Experimental Results

### Main Results
Four models were evaluated: Qwen2.5-14B-Base, Deepseek-R1-Distill-Qwen-14B, SaulLM-7B-Instruct (English legal), and BAI-Law-13B (Chinese legal), using benchmarks like LexGLUE, LegalBench, and CAIL2018.

| Dimension | Key Finding | Description |
|-----------|-------------|-------------|
| Reliability Ratio $s_{\text{reliable}}$| Low across all models; **>50% of interactions are unreliable** | Correct judgment ≠ Correct reasoning |
| Interaction Complexity | Strong preference for low-order interactions | Suggests heuristic guessing, not synthesis |
| Conflict Cancellation $s_{\text{conflict}}$ | $>60\%$ of effects are self-canceling | Significant noise and contradiction within judgments |

### Key Findings
- **Correct results mask reasoning errors**: All four models can output the correct judgment category, but over half of the underlying interactions rely on irrelevant or forbidden phrases—a risk missed by traditional accuracy metrics.
- **"Entity Misattribution" is a pervasive defect**: Models tend to associate sensitive tokens (e.g., weapons, death) with judgment outcomes without distinguishing "who did what," often attributing one person's actions to another.
- **Professional identity biases judgment**: Changing a victim's occupation from "judge" to "programmer" can cause a judgment to shift from "Robbery" to "Not mentioned," exposing deep-seated biases.
- **Long-chain reasoning is largely an illusion**: The dominance of low-order interactions suggests models perform local heuristic matching rather than integrating all fact elements.

## Highlights & Insights
- **Shifting evaluation from outcome to reasoning**: This work quantifies "why a model judges" rather than just "if it judged correctly," using mathematically guaranteed decomposition.
- **Theoretical grounding as a foundation**: Universal matching and sparsity ensure the audit targets the model's actual logic, not just an approximation. This framework is transferable to other high-risk domains like finance or medicine.
- **Expert domain interface**: The "relevant/irrelevant/forbidden" labeling paradigm provides a clean interface to inject human knowledge into the interaction analysis.

## Limitations & Future Work
- **Risk warning, not a benchmark**: The authors state that current LLM representation quality is too low to support a simple ranking benchmark; the goal is to reveal the existence of defects.
- **Manual labeling dependency**: The study relies on human experts for phrase categorization, which is hard to scale and subject to the specific legal frameworks of different jurisdictions.
- **Improvement direction**: Audit signals could be used for training or alignment (e.g., penalizing unreliable interactions) to enhance reasoning quality.

## Related Work & Insights
- **Compared to traditional evaluation**: While benchmarks like LegalBench focus on outcome accuracy, this paper exposes the "correct result, wrong process" blind spot.
- **Compared to standard attribution (Shapley/Saliency)**: Standard methods lack the "universal matching" guarantee and cannot distinguish between AND and OR logic types as this framework does.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Lifts auditing from outcomes to reasoning patterns with theoretical guarantees)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers 4 models and 6 datasets; however, limited by manual labeling scale)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear theoretical derivation and visualizations)
- **Value**: ⭐⭐⭐⭐⭐ (Highlights critical risks in high-stakes legal LLM applications)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping](reforming_the_mechanism_editing_reasoning_patterns_in_llms_with_circuit_reshapin.md)
- [\[ICLR 2026\] The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities](the_achilles_heel_of_llms_how_altering_a_handful_of_neurons_can_cripple_language.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](../../ICML2026/interpretability/interpretability_can_be_actionable.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](../../ACL2026/interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
