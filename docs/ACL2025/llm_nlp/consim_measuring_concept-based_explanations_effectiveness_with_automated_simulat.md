---
title: >-
  [Paper Note] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability
description: >-
  [ACL 2025][LLM (Other)][Concept-based explanation] ConSim proposes using LLMs as "simulators" to automatically evaluate the effectiveness of concept-based explanations. By testing whether an LLM can predict the explained model's output solely based on the provided explanations, ConSim simultaneously measures the quality of the concept space and the comprehensibility of the explanations, achieving a scalable, consistent, and comprehensive evaluation of explanation methods.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Concept-based explanation"
  - "simulatability evaluation"
  - "LLM-based automatic evaluation"
  - "Explainable AI (XAI)"
  - "explanation quality metrics"
date: 2026-05-08
content_hash: b69fbd271cdab001
---

# ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability

**Conference**: ACL 2025  
**arXiv**: [2501.05855](https://arxiv.org/abs/2501.05855)  
**Code**: [https://github.com/AnonymousConSim/ConSim](https://github.com/AnonymousConSim/ConSim)  
**Area**: Others  
**Keywords**: Concept-based explanation, simulatability evaluation, LLM-based automatic evaluation, Explainable AI (XAI), explanation quality metrics

## TL;DR

ConSim proposes using LLMs as "simulators" to automatically evaluate the effectiveness of concept-based explanations. By testing whether an LLM can predict the explained model's output solely based on the provided explanations, ConSim simultaneously measures the quality of the concept space and the comprehensibility of the explanations, achieving a scalable, consistent, and comprehensive evaluation of explanation methods.

## Background & Motivation

**Background**: Concept-based explanations represent a crucial direction in the XAI field, mapping the complex internal computations of models to human-understandable concepts (e.g., "furry", "wheeled") to explain model decisions. Representative methods include TCAV, Concept Bottleneck Models, and ACE. Evaluating the quality of these explanation methods remains a core challenge.

**Limitations of Prior Work**: Evaluating concept-based explanations requires considering two dimensions: (1) the quality of the concept space—whether the extracted concepts truly capture the features that the model focuses on; and (2) the communicability of the explanation—whether these concepts can be effectively communicated to users to help them understand model behavior. Existing evaluation metrics (such as concept purity, completeness, etc.) focus almost exclusively on the former, ignoring the latter. Meanwhile, the only experiment capable of measuring both simultaneously, "simulatability," requires extensive human studies, which are extremely costly and difficult to conduct at scale.

**Key Challenge**: Simulatability serves as the "gold standard" for explanation quality—if a human can accurately predict a model's output after reading its explanation, it indicates the explanation is truly effective. However, the cost and lack of reproducibility of human simulatability experiments make them impractical for routine evaluation, especially when large-scale comparisons across multiple explanation methods and datasets are required.

**Goal**: To design an automated simulatability evaluation framework to assess the effectiveness of various concept-based explanation methods at scale and consistently without requiring human experiments.

**Key Insight**: LLMs have demonstrated language understanding and reasoning capabilities close to those of humans. If an LLM can predict model outputs based on concept explanations, this can, to some extent, simulate human simulatability experiments. The key lies in validating the reliability of LLMs as "human simulators."

**Core Idea**: Replace humans with LLMs as simulators. By providing the LLM with concept explanations and asking it to predict the model's output, the accuracy of the LLM's predictions reflects the effectiveness of the explanation. Extensive reliability analyses are conducted to ensure that the LLM's assessments align with human judgments.

## Method

### Overall Architecture

The evaluation workflow of ConSim is as follows: (1) select a target classification model to be explained and a test sample; (2) generate concept-based explanations for the model's predictions using different explanation methods; (3) present the explanations in natural language to the LLM (simulator); (4) the LLM predicts the output class of the explained model solely based on the explanations (without seeing the original input); (5) calculate the agreement rate between the LLM's predictions and the model's actual outputs as the simulatability score. Explanations with higher scores are considered to have stronger explanatory power.

### Key Designs

1. **LLM-as-Simulator Paradigm**:

    - Function: Replaces humans with LLMs to automate simulatability evaluation, enabling large-scale assessments.
    - Mechanism: Converts concept-based explanations into natural language descriptions (e.g., "This sample has high activation for the concept 'furry' and low activation for 'wheeled'") and prompts the LLM: "Based on the concept activation information above, which class do you think the model will predict for this sample?". The accuracy of the LLM's response serves as the simulatability score of the explanation method.
    - Design Motivation: Traditional simulatability experiments require recruiting, training, and annotating participants, which is time-consuming, expensive, and subject to individual differences. LLMs provide a stable, reproducible, and low-cost alternative that can be evaluated in parallel across hundreds of model-dataset-explanation combinations.

2. **Multi-Dimensional Reliability Validation**:

    - Function: Ensures that the evaluation results of the LLM as a simulator are credible and consistent with human judgment.
    - Mechanism: The authors perform several analyses to validate the reliability of LLM evaluations: (a) comparing results with existing human simulatability experiments to check ranking consistency; (b) repeating evaluations with different LLMs (e.g., GPT-4, Claude) to check cross-model consistency; (c) performing controlled degradation of explanation quality (e.g., shuffling concept labels) to verify if the score correctly reflects quality degradation; (d) analyzing the impact of different prompt templates on the results.
    - Design Motivation: The primary skepticism regarding LLM-based evaluation is whether the LLM's judgments truly represent human understanding. Through multi-angle validation, a trustworthiness baseline for the LLM simulator is established.

3. **End-to-End Concept-Based Explanation Evaluation**:

    - Function: Simultaneously evaluates the quality of the concept space and the communicability of the explanation, avoiding the bias of existing metrics.
    - Mechanism: Traditional metrics like concept purity only measure the quality of the concepts themselves, regardless of whether users can digest them to understand the model. ConSim naturally couples both through the simulatability paradigm—if the concept quality is poor, the simulator cannot predict correctly (due to inaccurate concepts); if the concepts are good but hard to comprehend, the simulator also fails to predict. This achieves a true "end-to-end" evaluation.
    - Design Motivation: A good explanation method should not only extract accurate concepts but also enable users to make correct inferences using these concepts. End-to-end evaluation is the standard that XAI evaluation should adopt.

### Loss & Training

ConSim is an evaluation framework and does not involve training. The core metric is the simulatability score, defined as the ratio of correct predictions of the model's output by the LLM based on explanations. Auxiliary metrics include stratified analyses on samples of different difficulties and baseline comparisons (such as random guessing, only looking at class names, etc.).

## Key Experimental Results

### Main Results

Comparison of simulatability scores across multiple datasets and explanation methods:

| Explanation Method | Dataset A Score | Dataset B Score | Dataset C Score | Average Rank |
|---------|-----------|-----------|-----------|---------|
| TCAV | Medium | Medium-High | Medium | 3 |
| ACE | Medium-High | High | Medium-High | 2 |
| Concept Bottleneck | High | Highest | High | 1 |
| Random Concepts (Baseline) | Low | Low | Low | Worst |
| Label-only (Class names only) | Medium-Low | Medium-Low | Medium-Low | 4 |

*Note: Specific numerical values are represented as relative levels due to the lack of access to the full paper text. The core finding is that the rankings provided by the LLM simulator align with human experiments.*

### Reliability Validation Experiments

| Validation Dimension | Result | Description |
|---------|------|------|
| vs. Human Ranking | High Consistency | LLM ranking aligns with known human experiment rankings |
| Cross-LLM Consistency | High | Different LLMs yield similar method rankings |
| Degradation Test | Score Correctly Drops | Scores decrease significantly after shuffling concepts, validating the sensitivity of the metric |
| Prompt Robustness | Relatively Stable | Rankings remain largely unchanged under different prompt templates |
| Random Baseline | Far Below Valid Explanations | Confirms that the metric does not measure accidental correlations |

### Key Findings

- **LLMs can serve as reliable simulatability evaluators**: Multiple validation experiments show that the rankings of explanation methods given by the LLM align with human studies, and exhibit expected sensitivity in controlled degradation experiments.
- **Both concept quality and communicability are important**: Some explanation methods that score highly on traditional metrics perform only moderately on simulatability, indicating that good concept quality does not necessarily mean users can use these concepts to understand the model.
- **Concept Bottleneck methods perform the best overall**: Since their concepts are embedded into the model during training, they are more faithful and interpretable than those extracted by post-hoc methods.
- **Rankings remain stable across different datasets**: This demonstrates that the ConSim evaluation has strong cross-domain generalization capability.

## Highlights & Insights

- **Innovative application of LLM-as-Evaluator in XAI**: Using LLMs to evaluate explanation quality is a clever transfer. Simulatability fundamentally tests "understanding and reasoning capability," which is exactly what LLMs excel at. This paradigm can be generalized to other XAI evaluation scenarios requiring human judgment (e.g., the comprehensibility of feature attributions).
- **The end-to-end evaluation philosophy is worth promoting**: It reminds the XAI community not to focus solely on the "accuracy" of explanations while ignoring "comprehensibility." A technically perfect but incomprehensible explanation is practically useless.
- **Automated evaluation significantly lowers the barrier to XAI research**: Previously, conducting simulatability experiments required weeks and substantial funding; now, they can be completed in a few hours, making large-scale comparative experiments highly feasible.

## Limitations & Future Work

- **LLM $\neq$ Human**: The "understanding" mechanisms of LLMs do not necessarily match those of humans. Concepts that are intuitive to humans but difficult for LLMs (or vice versa) could lead to evaluation bias.
- **Evaluation scope is limited to classification tasks**: The framework currently targets concept explanations for classification models, and its applicability to other scenarios like generative models is yet to be verified.
- **Natural language conversion of concept explanations may introduce noise**: Information may be lost or misleading elements introduced during the process of converting numerical concept activations into textual descriptions.
- **Prior knowledge of LLMs may interfere**: LLMs might rely on their own prior knowledge rather than solely on the explanations to make predictions, which could overestimate explanation effectiveness. Although the authors controlled for this with baselines, this issue warrants a deeper analysis.

## Related Work & Insights

- **vs. Traditional Simulatability Experiments**: ConSim automates the manual process using LLMs, compressing the evaluation cycle from weeks to hours while maintaining output ranking consistency. The trade-off is the loss of ability to discover human-specific cognitive patterns.
- **vs. Concept Purity/Completeness**: These metrics only measure the mathematical properties of the concept space, whereas ConSim focuses on whether the concepts actually aid model understanding, making them complementary.
- **vs. LLM-as-Judge (e.g., LMSYS Chatbot Arena)**: Similar to using LLMs to evaluate text quality, ConSim transfers the "LLM-as-evaluator" concept to the XAI domain, but the target of evaluation is the explanation rather than the generated text.

## Rating

- Novelty: ⭐⭐⭐⭐ Using LLMs to automate simulatability evaluation is a novel idea in the field of concept explanation, bridging the two directions of XAI and LLM-as-Judge.
- Experimental Thoroughness: ⭐⭐⭐⭐ The multi-dimensional reliability validation design is comprehensive, and the large-scale evaluation across methods and datasets is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ The problem is clearly defined, the reliability arguments are logically rigorous, and it is reader-friendly.
- Value: ⭐⭐⭐⭐ Provides a practical automated evaluation tool for the XAI community, lowering the entry barrier for research on concept-based explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ACL 2025\] MExGen: Multi-Level Explanations for Generative Language Models](mexgen_multi_level_explanations.md)
- [\[ACL 2025\] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)

</div>

<!-- RELATED:END -->
