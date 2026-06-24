---
title: >-
  [Paper Note] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles
description: >-
  [ACL 2025][LLM Evaluation][Calibration] This paper systematically investigates three major factors influencing LLM calibration: multi-model response agreement, loss function selection, and prompt style. It proposes the Calib-n framework, which trains an auxiliary model to aggregate responses from multiple LLMs to estimate confidence, and reveals that response agreement and focal loss significantly improve calibration performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Calibration"
  - "Confidence Estimation"
  - "Auxiliary Model"
  - "Loss Function"
  - "Prompt Style"
date: 2026-05-08
content_hash: b6dda2be79d7b651
---

# Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles

**Conference**: ACL 2025  
**arXiv**: [2501.03991](https://arxiv.org/abs/2501.03991)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Calibration, Confidence Estimation, Auxiliary Model, Loss Function, Prompt Style  

## TL;DR

This paper systematically investigates three major factors influencing LLM calibration: multi-model response agreement, loss function selection, and prompt style. It proposes the Calib-n framework, which trains an auxiliary model to aggregate responses from multiple LLMs to estimate confidence, and reveals that response agreement and focal loss significantly improve calibration performance.

## Background & Motivation

**Background**: Calibration—the alignment between a model's predicted confidence and its actual accuracy—is crucial for the reliable deployment of LLMs. Modern methods for obtaining LLM confidence primarily fall into three categories: (1) internal logits (logit-based); (2) verbalized confidence (direct self-reporting of confidence by the model); and (3) external confidence estimation based on auxiliary models.

**Limitations of Prior Work**: Existing calibration methods suffer from two overlooked evaluation issues: first, a lack of generalizability testing across different prompt styles (methods may only succeed under specific prompts); second, a lack of systematic evaluation across LLMs of different scales. Furthermore, current auxiliary model methods typically rely only on a single LLM's response, ignoring the consensus information across multiple models.

**Key Challenge**: The output probabilities or verbalized confidence of a single LLM are inherently unreliable (large models often exhibit overconfidence). Meanwhile, existing auxiliary model approaches fail to fully leverage information complementarity across multiple models, and systematic investigation into the choice of loss functions is still lacking.

**Goal**: (1) To define a controlled experimental framework covering 12 LLMs and 4 prompt styles; (2) to verify whether multi-model response agreement improves calibration; (3) to explore the effects of focal loss and surrogate AUC loss on calibration; and (4) to analyze the impact of prompt styles on calibration methods.

**Key Insight**: The authors observe that if multiple LLMs provide the same answer to a single question, that answer is more likely to be correct. This cross-model agreement signal can serve as an additional input feature for auxiliary models to improve confidence estimation.

**Core Idea**: Construct the Calib-n framework, which uses an auxiliary model to aggregate response features (including answer text, answer agreement, verbalized confidence, etc.) from n LLMs. Combined with focal loss to optimize calibration, this achieves more robust calibration estimation than internal probabilities and verbalized confidence.

## Method

### Overall Architecture

The pipeline of Calib-n consists of three steps: (1) Collection phase—for each question, obtain responses from n LLMs under a given prompt style; (2) Feature construction—extract the response text, verbalized confidence, and cross-model answer agreement scores of each LLM; (3) Training the auxiliary model—using the constructed features as inputs and the binary correctness labels as targets, train a lightweight auxiliary model (such as logistic regression or a small MLP). The output probability of this auxiliary model serves as the calibrated confidence.

### Key Designs

1. **Response Agreement Feature**:

    - **Function**: Captures the degree of consensus among multiple LLMs on the same question.
    - **Mechanism**: For each question, obtain answers from $n$ LLMs and calculate the match ratio of the target LLM's answer with those of other LLMs to serve as the agreement score: $\text{agree}(q) = \frac{1}{n-1}\sum_{i \neq t} \mathbb{1}[a_i = a_t]$. This score, along with the text embeddings of the target LLM's response, is fed as input to the auxiliary model.
    - **Design Motivation**: Multi-model consensus serves as a "collective wisdom" signal. If the majority of models generate the exact same answer, that answer is highly likely to be correct. This is far more robust than relying solely on a single model's logits.

2. **Focal Loss and Surrogate AUC Loss**:

    - **Function**: Improves the optimization target of calibration, and addresses the shortcomings of standard cross-entropy.
    - **Mechanism**: Standard binary cross-entropy (BCE) treats all samples equally. However, in calibration tasks, gradients contributed by "easy samples" (where the model is highly certain and correct/wrong) dominate, overshadowing the signal of "hard samples" (uncertain, boundary cases). Focal loss downweights easy samples using a modulating factor $(1-p_t)^\gamma$: $\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$. Surrogate AUC loss, on the other hand, directly optimizes ranking quality.
    - **Design Motivation**: Calibration fundamentally requires the model to be accurate in its confidence ranking—high-confidence samples indeed ought to be more likely correct than low-confidence ones. Focal loss places more focus on boundary cases, which is precisely the most critical region for calibration.

3. **Systematic Evaluation of Four Prompt Styles**:

    - **Function**: Evaluates the generalizability of calibration methods across different prompt styles.
    - **Mechanism**: Four prompt styles are designed: zero-shot, few-shot, chain-of-thought (CoT), and few-shot-CoT. All calibration methods are then evaluated under each style separately. The auxiliary model is trained and tested independently on each prompt style.
    - **Design Motivation**: In practical deployment, prompt styles vary widely. If a calibration method only works under a specific prompt, its utility is significantly diminished.

### Loss & Training

The auxiliary model is trained using three loss functions: BCE (baseline), Focal Loss ($\gamma=2$), and surrogate AUC loss. The training data comes from LLM responses and their correctness labels on the training set. The model architecture is a lightweight classifier (Text Embedding + Agreement Score $\rightarrow$ MLP $\rightarrow$ Confidence).

## Key Experimental Results

### Main Results

| Method | TriviaQA (ECE↓) | CoQA (ECE↓) | SQuAD (ECE↓) | Natural Questions (ECE↓) |
|------|-----------------|-------------|-------------|------------------------|
| Internal Prob (Llama-70B) | 0.182 | 0.201 | 0.175 | 0.193 |
| Verbalized Conf | 0.156 | 0.178 | 0.162 | 0.171 |
| Calib-1 (BCE) | 0.098 | 0.112 | 0.095 | 0.108 |
| Calib-n (BCE) | 0.082 | 0.096 | 0.081 | 0.094 |
| Calib-n (Focal) | **0.071** | **0.085** | **0.073** | **0.082** |

### Ablation Study

| Configuration | Average ECE↓ | Description |
|------|----------|------|
| Calib-n + Focal (Full) | 0.078 | Optimal configuration |
| Calib-1 + Focal (No Agreement) | 0.091 | Removing multi-model agreement yields a 17% ECE increase |
| Calib-n + BCE (No Focal) | 0.088 | Reverting to standard BCE yields a 13% ECE increase |
| Calib-n + AUC Loss | 0.084 | AUC loss shows some effect but is inferior to Focal |
| Internal Prob Only | 0.188 | Worst baseline |

### Key Findings

- **Multi-model response agreement yields around 17% calibration improvement** (ECE reduction), confirming the effectiveness of cross-model consensus.
- **Focal loss consistently outperforms BCE and AUC loss**, especially on datasets with numerous boundary cases.
- **Few-shot prompt is the optimal pairing for auxiliary model methods**—zero-shot provides too few signals, while CoT, despite improving accuracy, shows limited benefits for calibration.
- **The auxiliary model remains stable even when the accuracy of LLMs fluctuates**, whereas internal probabilities and verbalized confidence fluctuate significantly alongside accuracy.

## Highlights & Insights

- **Using multi-model consensus as a calibration signal** is a simple but effective idea. In practical settings, more reliable confidence estimates can be achieved through a few additional API calls, maintaining acceptable costs while yielding substantial gains.
- **The systematic analysis of prompt style impacts** fills an important gap in calibration research. Prior works typically only evaluated methods under a single prompt, leaving the generalizability of their findings questionable.
- **The adaptation of Focal loss in calibration tasks** can be generalized to other tasks requiring attention to boundary conditions, such as OOD detection and uncertainty quantification.

## Limitations & Future Work

- The auxiliary model needs to be retrained for each prompt style and dataset, so its transferability warrants further verification.
- Multi-model agreement relies on querying multiple LLMs, which linearly increases inference cost and is unfavorable for latency-sensitive applications.
- The method is evaluated only on QA tasks; calibration scenarios in generative tasks (e.g., summarization, translation) have not been explored.
- The agreement score relies on exact matching, which demands more flexible matching paradigms for open-ended generation scenarios.

## Related Work & Insights

- **vs Temperature Scaling**: Temperature scaling is the simplest post-processing calibration method but requires logit access, making it inapplicable to closed-source APIs. The auxiliary model approach of Calib-n is more versatile.
- **vs Verbalized Confidence (Lin et al. 2022)**: Verbalized confidence asks the LLM to self-report its confidence, but the model tends to be overconfident. Calib-n avoids this issue through an external auxiliary model.
- **vs Self-Consistency (Wang et al. 2023)**: Self-consistency estimates confidence by repeatedly sampling from the same model, whereas Calib-n leverages diversity across different models, yielding stronger information complementarity.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing multi-model response agreement for calibration is a novel perspective, though the overall framework is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The cross-evaluation involving 12 LLMs, 4 prompts, 4 datasets, and 3 loss functions is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-organized ablation analysis.
- Value: ⭐⭐⭐⭐ High practical guidance value for LLM calibration, despite moderate methodological novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)
- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [\[ICLR 2026\] Textual Bayes: Quantifying Prompt Uncertainty in LLM-based Systems](../../ICLR2026/llm_evaluation/textual_bayes_quantifying_prompt_uncertainty_in_llm-based_systems.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)

</div>

<!-- RELATED:END -->
