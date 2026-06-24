---
title: >-
  [Paper Note] Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm
description: >-
  [ACL 2025][LLM (Other)][cognitive evaluation] Through a systematic analysis of cognitive task performance across multiple LLMs using the N-back paradigm, this study reveals that poor performance is primarily caused by insufficient task comprehension and task-set maintenance failure rather than working memory capacity constraints. Supported by curriculum learning, the best model (Llama 3.1 70b) can even perform the 10-back task (accuracy of 84.75%).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "cognitive evaluation"
  - "N-back"
  - "working memory"
  - "task comprehension"
  - "LLM"
date: 2026-05-08
content_hash: 45d6d2e9c9ce146b
---

# Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm

**Conference**: ACL 2025  
**arXiv**: [2412.18120](https://arxiv.org/abs/2412.18120)  
**Code**: [https://github.com/hxiaoyang/lm-nback](https://github.com/hxiaoyang/lm-nback)  
**Area**: LLM/NLP - Cognitive Evaluation  
**Keywords**: cognitive evaluation, N-back, working memory, task comprehension, LLM

## TL;DR
Through a systematic analysis of cognitive task performance across multiple LLMs using the N-back paradigm, this study reveals that poor performance is primarily caused by insufficient task comprehension and task-set maintenance failure rather than working memory capacity constraints. Supported by curriculum learning, the best model (Llama 3.1 70b) can even perform the 10-back task (accuracy of 84.75%).

## Background & Motivation

**Background**: Experimental paradigms from cognitive psychology (e.g., the N-back task) are increasingly applied to evaluate LLMs' cognitive capabilities, including theory of mind, analogical reasoning, and working memory. These assessments aim to understand whether LLMs possess human-like cognitive constructs.

**Limitations of Prior Work**: Gong et al. (2024) applied the N-back task to GPT-3.5 and deduced that LLMs' working memory capacity is about 3 (similar to humans) after observing a sharp drop in 2-back and 3-back performance. This interpretation has two major issues: (1) human working memory constraints cannot be assumed to exist in LLMs; and (2) poor performance may originate from task comprehension failure rather than memory capacity limitations.

**Key Challenge**: When a model performs poorly, is it because the tested cognitive capability is limited, or because the model did not understand what task to execute in the first place? Prior research has failed to distinguish between these two possibilities.

**Goal**: Systematically differentiate whether LLMs' performance degradation in the N-back task is driven by insufficient task comprehension or working memory limitations.

**Key Insight**: Conduct N-back testing on multiple open-source models across different performance tiers, introduce counterfactual measures to analyze which m-back task the models are actually executing, and validate task comprehension through attention analysis.

**Core Idea**: Differences in LLMs' performance on the N-back task primarily reflect variations in task comprehension capabilities rather than working memory capacities.

## Method

### Overall Architecture
By utilizing standard N-back datasets (50 trials per n-back task, with each trial containing a sequence of 24 letters), this study systematically evaluates several open-source instruction-tuned models and GPT-3.5. The evaluation dimensions include: (1) baseline performance layering, (2) task comprehension analysis, (3) task-set maintenance analysis, (4) performance testing with higher n-values, (5) curriculum learning, (6) interactive demonstrations, and (7) attention analysis.

### Key Designs

1. **Counterfactual Measures**:

    - **Function**: Evaluate the response probability of models on all possible m-back ($m \neq n$) tasks simultaneously given the n-back instruction.
    - **Mechanism**: Define $\mathsf{P}_{n,m}$ as the average log probability of m-back retrieval under n-back instructions and demonstrations; define $\mathsf{P}^{-}_{n,m}$ as the corresponding value when only instructions (no demonstrations) are provided. If $\max_m \mathsf{P}_{n,m} = \mathsf{P}_{n,n}$, it indicates that the model truly understands the n-back task.
    - **Design Motivation**: Traditional evaluations only assess accuracy, failing to differentiate between making mistakes in the correct task and performing a completely different task correctly (i.e., 'doing the wrong task right').

2. **Three Performance Tiers**:

    - **Function**: Categorize models into three tiers based on their retrieval accuracy in 2-back and 3-back tasks: T1 (>80%), T2 (~50%), and T3 (≤20%).
    - **Mechanism**: T3 models (e.g., Qwen 1.5 14b) completely misunderstand the task, performing 1-back even when instructed with 2-back; T2 models (e.g., Gemma 2 27b, GPT-3.5) drift from the correct task to an incorrect task over time; T1 models (e.g., Llama 3.1 70b) consistently comprehend and execute the assigned tasks correctly.
    - **Design Motivation**: Different tiers reflect varying levels of task comprehension capability rather than disparate working memory capacities.

3. **Task Set Maintenance**:

    - **Function**: Track changes in the models' task consistency over time within a 24-step trial.
    - **Mechanism**: Define $\mathsf{A}_{n,\cdot}(m,i)$ as the average accuracy of m-back consistent retrieval at the $i$-th step. Investigate error accumulation effects by providing T2 models with m-back consistent historical responses and observing behavioral changes in subsequent steps.
    - **Design Motivation**: Identify that the performance decline in T2 models stems from task drift driven by error accumulation.

4. **In-context Curriculum Learning**:

    - **Function**: Provide a complete sequence of demonstrations from 1-back to (n-1)-back before testing on the n-back task.
    - **Mechanism**: Progressively increase task difficulty to guide the model toward understanding more complex N-back tasks.
    - **Design Motivation**: This strategy enables Llama 3.1 70b to achieve task accuracies of 90.08%, 90.08%, and 84.75% on 8-back, 9-back, and 10-back tasks, respectively.

5. **Attention Analysis**:

    - **Function**: Calculate the mean relative attention (MRAT) of each retrieval token to the correct source token (n steps behind).
    - **Mechanism**: High-performing models are expected to pay more attention to the token located n steps behind during retrieval. The maximum MRAT of Qwen 2 72b reached 71.98%, whereas that of Qwen 1.5 14b was only 38.95%.
    - **Design Motivation**: Provide mechanistic evidence for task comprehension.

## Key Experimental Results

### Main Results

| Model | Tier | 1-back | 2-back | 3-back |
|------|------|--------|--------|--------|
| Qwen 1.5 14b Chat | T3 | 1.00 | 0.09 | 0.08 |
| Llama 3.1 8b Instr. | T3 | 1.00 | 0.14 | 0.17 |
| Gemma 2 27b Instr. | T2 | 1.00 | 0.57 | 0.36 |
| GPT 3.5 Turbo | T2 | 1.00 | 0.51 | 0.43 |
| Qwen 2 72b Instr. | T1 | 1.00 | 0.81 | 0.84 |
| Llama 3.1 70b Instr. | T1 | 1.00 | 0.99 | 0.93 |

### Ablation Study (Curriculum Learning vs. Direct Testing, Llama 3.1 70b)

| N Value | Direct Testing - Retrieval Accuracy | Curriculum Learning - Retrieval Accuracy | Direct Testing - Task Accuracy | Curriculum Learning - Task Accuracy |
|-----|---------------------|---------------------|---------------------|---------------------|
| 8 | 75.25% | 79.83% | 83.33% | 90.08% |
| 9 | 66.08% | 80.17% | 78.25% | 90.08% |
| 10 | 57.10% | 71.67% | 71.92% | 84.75% |

### Key Findings
- When T3 models are given 2-back instructions, $\mathsf{P}_{2,1} > \mathsf{P}_{2,2}$, showing that the models fully misunderstand the task and are actually executing 1-back.
- T2 models (including GPT-3.5) initially understand the task but gradually drift towards 1-back behavior due to error accumulation.
- T1 models are capable of correctly inferring the task even when provided only with instructions (no demonstrations): $\mathsf{P}^{-}_{2,2} > \mathsf{P}^{-}_{2,1}$.
- Interactive demonstrations prove less effective than standard demonstrations; for Llama 3.1 70b, the 3-back performance drops from 0.93 to 0.62.
- A clear attention-performance correspondence is observed within the Qwen model family, whereas the attention of Llama models appears more dispersed.

## Highlights & Insights
- Provides a systematic methodology to distinguish between 'task comprehension failure' and 'insufficient cognitive capability', which is valuable for all cognitive evaluation studies.
- The counterfactual measurement method is clever—it checks not only what the model does correctly, but also what the model "assumes it is doing".
- The success of curriculum learning demonstrates that LLMs possess sequence memory capabilities far exceeding prior expectations, provided they receive correct guidance on task comprehension.

## Limitations & Future Work
- The internal circuit mechanisms of the models (e.g., which layers/components are responsible for task comprehension) are not thoroughly analyzed.
- The underlying cause of the abnormally dispersed attention patterns in Llama models remains unexplained.
- There is still room to optimize prompt selection, as more effective prompting strategies might exist to enhance task comprehension.

## Related Work & Insights
- **vs. Gong et al. (2024)**: Directly challenges their conclusion that "LLMs have a working memory capacity of approximately 3", proving that performance degradation is caused by task comprehension rather than memory constraints.
- **vs. Zhang et al. (2024)**: While they acknowledge that smaller models might not understand task intent, they fail to control for task comprehension as a confounding variable.
- **vs. Biran et al. (2024)**: A decline in LLMs' reasoning capability during subsequent steps is also identified in multi-hop QA, resembling the task drift phenomenon observed in N-back.

## Rating
- Novelty: ⭐⭐⭐⭐ The counterfactual measurement methodology is novel, though the research question itself leans towards verification.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis across seven dimensions with multi-tier model comparisons, covering 1-10 back.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, progressive experimental design, and intuitive charts/tables.
- Value: ⭐⭐⭐⭐ Holds significant guiding value for LLM cognitive assessment methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)
- [\[ACL 2025\] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)

</div>

<!-- RELATED:END -->
