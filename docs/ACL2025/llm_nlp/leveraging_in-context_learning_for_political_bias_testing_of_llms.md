---
title: >-
  [Paper Note] Leveraging In-Context Learning for Political Bias Testing of LLMs
description: >-
  [ACL 2025][LLM (Other)][Political bias detection] This paper proposes "Questionnaire Modeling" (QM), a novel probing task that utilizes human survey data as in-context examples to improve the stability of LLM political bias detection. The study finds that instruction tuning can alter the direction of bias, and larger models can more effectively leverage in-context examples and exhibit smaller bias scores.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Political bias detection"
  - "Large Language Models"
  - "In-context learning"
  - "Questionnaire modeling"
  - "Instruction tuning"
date: 2026-05-08
content_hash: 518bcc12ea7b376b
---

# Leveraging In-Context Learning for Political Bias Testing of LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.22232](https://arxiv.org/abs/2506.22232)  
**Code**: Yes (noted as publicly available in the paper)  
**Area**: LLM/NLP  
**Keywords**: Political bias detection, Large Language Models, In-context learning, Questionnaire modeling, Instruction tuning

## TL;DR

This paper proposes "Questionnaire Modeling" (QM), a novel probing task that utilizes human survey data as in-context examples to improve the stability of LLM political bias detection. The study finds that instruction tuning can alter the direction of bias, and larger models can more effectively leverage in-context examples and exhibit smaller bias scores.

## Background & Motivation

**Background**: An increasing number of studies evaluate the latent biases of LLMs by posing politically relevant questions. A common approach is to directly prompt the models with political questionnaires (such as question sets from "voting advice applications") and calculate their position on the political spectrum based on the model responses.

**Limitations of Prior Work**: This simple "direct prompting" probing method suffers from severe stability issues: (1) the same model may give different answers to the same question across different runs; (2) minor changes in question phrasing can flip the bias evaluation results; (3) comparing biases across different models becomes unreliable. These issues make it difficult for researchers to draw robust conclusions about "which side" an LLM actually leans toward.

**Key Challenge**: When directly presented with political questions, LLMs lack sufficient context to "understand" the specific meaning of the task—they do not know which survey the questions originate from, how others answered them, or what "agree/disagree" specifically means in that context. This ambiguity leads to unstable responses.

**Goal**: Design a more stable LLM bias probing method to make comparisons between models more reliable, while also shedding light on the impact of instruction tuning on model bias.

**Key Insight**: The authors' key insight is that if LLMs are provided with real human respondents' answers as in-context examples, they can better understand the task context, leading to more stable answers. This essentially reframes bias detection as a "questionnaire modeling" task.

**Core Idea**: Utilize human survey data as in-context learning examples to transform LLM bias probing into a "Questionnaire Modeling" task, significantly improving probing stability and enabling reliable comparisons between instruction-tuned and base models.

## Method

### Overall Architecture

The input consists of a set of political questionnaire questions and response data from actual human surveys (e.g., the Smartvote voting advice platform). For each test question, several examples (comprising question-answer pairs) are sampled from the human survey data as context, guiding the LLM to answer the target question. By repeatedly sampling different sets of in-context examples and averaging the results, stable preference scores for each question are obtained, which are finally mapped onto the political spectrum to calculate bias scores.

### Key Designs

1. **Questionnaire Modeling (QM) Task**:

    - **Function**: Upgrades LLM bias detection from "direct prompting" to a conditional generation task with context.
    - **Mechanism**: Given a political question $q_i$, $k$ other questions and their corresponding answers $\{(q_j, a_j)\}_{j \neq i}$ are sampled from human survey data as in-context examples. The LLM is then prompted to predict the response to $q_i$. Bootstrap sampling over different context sets is repeated multiple times, and the average is taken to obtain a stable estimate. The core lies in providing a "respondent persona profile" for the model to simulate.
    - **Design Motivation**: In-context examples provide the model with two levels of information: the "survey scenario" and the "respondent persona," reducing the randomness and ambiguity of the model's responses.

2. **In-Context Sampling Based on Human Survey Data**:

    - **Function**: Provides high-quality in-context examples for the QM task.
    - **Mechanism**: Uses real candidate response data from the Swiss voting advice platform Smartvote. Each candidate answered a set of political questions (Likert scale), and these real answers serve as the sampling pool. During each test, a candidate's responses are randomly selected as the in-context template.
    - **Design Motivation**: Real data is more representative than synthetic data. Politicians' responses possess internally consistent political stances, providing a coherent "persona" for the model to simulate.

3. **Stability Evaluation Framework for Bias Scores**:

    - **Function**: Quantitatively evaluates the improvement in stability of QM compared to direct prompting.
    - **Mechanism**: Multiple rounds of tests (direct prompting vs. QM) are conducted on the same model and the same question, and the variance across rounds is calculated as the stability metric. Comparisons include: (1) consistency of answers across runs; (2) confidence intervals of the overall bias scores; (3) correlation between the conditional bias scores under a given candidate context and that candidate's real political stance.
    - **Design Motivation**: The reliability of bias detection is a long-neglected issue in this field; stability must be resolved before meaningful comparisons between models can be made.

### Loss & Training

This work does not involve training but performs zero-shot and few-shot testing on existing models. Evaluation metrics include bias scores (position on the political spectrum), cross-run variance (stability), and differences in bias direction between instruction-tuned and base models.

## Key Experimental Results

### Main Results

| Model | Direct Prompting Bias Variance | QM Bias Variance | Variance Reduction | Bias Direction |
|------|-----------------|-----------|---------|---------|
| GPT-4 | High | Low | Significant | Moderate-Left |
| Llama-2-70B-chat | High | Low | Significant | Moderate-Left |
| Llama-2-13B-chat | Medium | Relatively Low | Moderate | Unstable → Stable Left |
| Llama-2-7B-chat | High | Medium | Moderate | Uncertain Direction |
| Small Models (<7B) | Very High | Still Relatively High | Small | Almost Random |

### Ablation Study

| Configuration | Stability | Description |
|------|--------|------|
| QM (Full) | Highest | Uses real human context, averages over multiple sampled runs |
| Direct Prompting (No Context) | Lowest | Model responses are highly unstable |
| Instruction Tuning vs. Base Model | N/A | QM effectively distinguishes differences in bias direction between the two |
| Different Context Sizes k | Stability improves as k increases | k=5-10 provides a good balance |

### Key Findings

- **QM significantly improves stability**: After introducing human survey data as context, the cross-run variance of bias measurements drops substantially, making comparisons across models reliable.
- **Instruction tuning alters bias direction**: Within the QM framework, it is clearly observed that instruction tuning sometimes flips the model's bias from one direction to another—a phenomenon that could not be reliably detected with direct prompting due to high variance.
- **Scaling effect**: Larger models are better at utilizing in-context examples (exhibiting stronger in-context learning capabilities), thus showing lower bias score magnitude and higher stability in QM.
- **Small models struggle with QM**: Small-scale models lack sufficient in-context learning capabilities, resulting in limited improvements from QM.

## Highlights & Insights

- **Solving evaluation stability with ICL**: Framing in-context learning as part of the evaluation methodology rather than merely a task execution tool represents a novel perspective. Its core idea—"providing the model with more context to stabilize its behavior"—can be transferred to any task requiring a stable evaluation of LLM subjective preferences.
- **Interdisciplinary design of questionnaire modeling**: Combining social science survey methodology with NLP, using real candidate data to construct context, ensures ecological validity while maintaining a controlled experimental design.
- **Revealing the double-edged sword of instruction tuning**: Empirical evidence shows that instruction tuning can not only amplify bias but also alter its direction, which has important implications for safety alignment research.

## Limitations & Future Work

- Only tested within the Swiss political context; the definition of the political spectrum and questionnaire questions are culturally specific, requiring validation in other countries/contexts.
- The human responses in the context are from political candidates, whose positions may be more polarized than those of ordinary citizens, potentially influencing the model's simulation behavior.
- Only the bias of text model responses is evaluated; differences in refusal rates and their impact on bias estimation are not considered.
- The definition of "bias" is narrow (position on the political spectrum) and does not cover deeper ideological biases.

## Related Work & Insights

- **vs. Political Compass test**: The Political Compass directly prompts the model, whereas the QM method in this paper significantly improves stability using in-context examples, representing a methodological upgrade.
- **vs. OpinionQA (Santurkar et al.)**: OpinionQA evaluates the opinion distribution of LLMs, whereas this paper focuses more on the reliability of bias measurement and proposes a concrete mitigation strategy (QM).
- **vs. Persona-based probing**: Some works prompt LLMs to role-play specific political personas to evaluate bias; QM guides the model through implicit context rather than explicit persona instructions, closely mimicking natural behavior.

## Rating

- Novelty: ⭐⭐⭐⭐ QM is a simple and effective new paradigm, and using ICL for evaluation stability is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on multiple models of varying scales with quantitative stability analysis, though limited to a single political context.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems and concise method descriptions.
- Value: ⭐⭐⭐⭐ Makes an important contribution to the methodology of LLM bias evaluation, which should be valued by the safety research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)

</div>

<!-- RELATED:END -->
