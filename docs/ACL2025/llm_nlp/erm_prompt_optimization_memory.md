---
title: >-
  [Paper Note] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection
description: >-
  [ACL 2025][LLM (Other)][prompt optimization] Proposes the ERM method, which enhances feedback quality by generating exemplars with detailed problem-solving processes guided by meta-prompts, and introduces Feedback Memory and Exemplar Factory as two long-term memory mechanisms to efficiently store and reuse historical feedback and exemplars, surpassing SOTA prompt optimization methods on multiple tasks with approximately half the optimization steps.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "prompt optimization"
  - "feedback memory"
  - "exemplar retrieval"
  - "automatic prompt engineering"
  - "LLM"
date: 2026-05-08
content_hash: a469ec65404b9171
---

# Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection

**Conference**: ACL 2025  
**arXiv**: [2411.07446](https://arxiv.org/abs/2411.07446)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: prompt optimization, feedback memory, exemplar retrieval, automatic prompt engineering, LLM  

## TL;DR
Proposes the ERM method, which enhances feedback quality by generating exemplars with detailed problem-solving processes guided by meta-prompts, and introduces Feedback Memory and Exemplar Factory as two long-term memory mechanisms to efficiently store and reuse historical feedback and exemplars, surpassing SOTA prompt optimization methods on multiple tasks with approximately half the optimization steps.

## Background & Motivation
**Background**: Automatic prompt optimization aims to find the optimal prompt without human intervention. Mainstream methods include evolutionary (EvoPrompt), trajectory-based (OPRO, GPO), and feedback-based (ProTeGi) approaches.

**Limitations of Prior Work**: Feedback-based methods suffer from two core problems: (a) they only utilize the feedback from the current step, discarding historical and non-selected feedback, which leads to more optimization steps required for convergence; (b) during inference, exemplar retrieval is solely based on semantic similarity, without evaluating its impact on actual task performance.

**Key Challenge**: Valuable feedback information is wasted, and the selection of exemplars is disconnected from task performance.

**Goal**: How to efficiently utilize all historical feedback? How to select exemplars that truly contribute to task performance?

**Key Insight**: Leveraging the human memory mechanism (Ebbinghaus forgetting curve), the proposed method establishes long-term memory storage for feedback and exemplars with priority scores, dynamically adjusting priority and selectively forgetting based on effectiveness evaluation.

**Core Idea**: Manage feedback and exemplars using memory mechanisms, ensuring that valuable information is continuously utilized while invaluable information is forgotten.

## Method

### Overall Architecture
ERM (Exemplar-Guided Reflection with Memory) contains three core components: the input is a set of error samples $\mathcal{B}$ and the current prompt $p^t$, which passes through Exemplar-Guided Reflection to generate exemplars and feedback, stored in two memory storage systems—Feedback Memory and Exemplar Factory—ultimately outputting the optimized prompt $p^{t+1}$. During inference, exemplars are retrieved from the Exemplar Factory and concatenated with the prompt to improve prediction accuracy.

### Key Designs
1. **Exemplar-Guided Reflection**:

    - Function: Designing a guided meta-prompt to direct the prompt optimizer to select typical samples from error instances and provide detailed problem-solving processes (CoT style), then generating more informative feedback based on these exemplars.
    - Mechanism: $\mathcal{E}^t = M_e(p^t, \mathcal{B}; p^{meta}_{ref*})$ is first used to generate a set of exemplars (including question, answer, and CoT), and then feedback $\mathcal{F}^t = M_e(p^t, \mathcal{B}, \mathcal{E}^t; p^{meta}_{ref*})$ is generated based on the exemplars.
    - Design Motivation: Traditional methods generate limited feedback directly on error samples. Incorporating detailed problem-solving processes makes the feedback more targeted, offering a more precise direction for subsequent prompt optimization.

2. **Feedback Memory**:

    - Function: Storing historical feedback, assigning a priority score to each, and periodically retrieving high-priority feedback to guide prompt optimization.
    - Mechanism: Filtering during storage (only storing feedback that brings performance gains + deduplication); sampling by priority probability $P_f = \text{softmax}(\{e^{s_p(f_i)/\tau_f}\})$ during retrieval; updating priority after use $s_p^t(f) = (1-\beta)s_p(f)^{t-1} + \beta \mathbb{I}(f)$, and feedback scoring below threshold $\theta$ is forgotten.
    - Design Motivation: Avoiding the loss of valuable historical feedback while ensuring only effective information is retained in memory through a selective forgetting mechanism, thereby accelerating optimization convergence.

3. **Exemplar Factory**:

    - Function: Storing, evaluating, and retrieving exemplars, and choosing the optimal exemplar to concatenate with the prompt to enhance predictions during inference.
    - Mechanism: Also managed using priority scores; retrieval synthetically considers priority and semantic similarity to the current question $P_e^r = \text{softmax}(\{e^{s_p(e_i) \cdot s_s^j(e_i)/\tau_e}\})$. Correctness of the problem-solving process is validated and deduplicated during storage, and priorities are updated after use based on their helpfulness to predictions.
    - Design Motivation: Retrieval purely based on semantic similarity does not guarantee selecting the most helpful exemplars for the task. Thus, the retrieval strategy needs to be optimized through real-world performance feedback.

### Loss & Training
- Employs beam search to select the $k$ best-performing candidate prompts on the validation set for the next optimization step.
- Similarity computation uses the BGE-M3 model.
- The task model uses Doubao-Pro, and the prompt optimizer uses GPT-4o.

## Key Experimental Results

### Main Results

| Dataset | Metric | ERM | Prev. SOTA | Gain |
|--------|------|-----|----------|------|
| LIAR | F1 | 68.6 | 58.5 (ProTeGi) | +10.1 |
| BBH | F1 | 86.1 | 81.9 (CoT) | +4.2 |
| ETHOS | F1 | 98.0 | 96.5 (ProTeGi) | +1.5 |
| WebNLG | Rouge-L | 59.6 | 55.7 (ProTeGi) | +3.9 |
| GSM8K | Acc. | 93.3 | 91.7 (Promptbreeder) | +1.6 |
| WSC | Acc. | 86.0 | 84.0 (GPO) | +2.0 |

### Ablation Study

| Configuration | LIAR F1 | BBH F1 | Description |
|------|---------|--------|------|
| Baseline (ProTeGi) | 58.5 | 73.6 | No components |
| +Exemplar-Guided Reflection | 62.9 | 75.7 | +4.4 |
| +Reflection +Feedback Memory | 67.2 | 84.7 | +8.7 |
| +Reflection +Exemplar Factory | 66.6 | 82.6 | +8.1 |
| Full ERM | 68.6 | 86.1 | +10.1 |

### Key Findings
- Feedback Memory contributes the most (+5.7 on LIAR), demonstrating that the reuse of historical feedback is robustly important.
- Both filtering and selective forgetting are indispensable in Exemplar Factory; retrieving without filtering is actually ineffective.
- ERM requires approximately half the optimization steps of ProTeGi to achieve optimal performance (LIAR: 7 steps vs. 13 steps).
- Consistently and significantly outperforms other methods in the few-shot setting.

## Highlights & Insights
- Introduces the concept of the human memory "forgetting curve" into prompt optimization, dynamically managing the lifecycle of feedback/exemplars with priority scores; this approach can be transferred to other scenarios requiring long-term knowledge management.
- The design of Exemplar-Guided Reflection is elegant: it first generates CoT-style exemplary solutions and then uses them to assist feedback generation, forming a closed loop of mutual reinforcement.

## Limitations & Future Work
- The scenario of human-in-the-loop optimization is not explored (when the model continues to fail, human-provided solutions might be more efficient).
- Experiments are constrained by budget, resulting in a limited variety of task types.
- Hyperparameters ($\beta$, $\theta$, $\tau$) in Feedback Memory and Exemplar Factory require tuning.

## Related Work & Insights
- **vs ProTeGi**: ProTeGi only uses feedback from the current step, whereas ERM reuses historical feedback via the memory mechanism, achieving +10.1 on LIAR.
- **vs OPRO/GPO**: Trajectory-based methods optimize based on historical prompts and scores but fail to learn concrete improvement directions from errors.
- **vs EvoPrompt**: Evolutionary methods randomly mutate prompts without targeted feedback guidance.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying the memory mechanism to prompt optimization is an innovative combination, though the individual components are not entirely novel on their own.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets, extensive ablation studies, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-illustrated figures.
- Value: ⭐⭐⭐⭐ Highly practical, with a significant increase in prompt optimization efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Meta-Reflection: A Feedback-Free Reflection Learning Framework](meta-reflection_a_feedback-free_reflection_learning_framework.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)

</div>

<!-- RELATED:END -->
