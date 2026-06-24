---
title: >-
  [Paper Note] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis
description: >-
  [ACL 2025][Interpretability][Data Contamination] This paper proposes to locate "shortcut neurons" in contaminated models through comparative analysis and causal analysis, and suppress these neurons via activation patching to achieve more trustworthy LLM evaluation, achieving a Spearman correlation coefficient of over 0.95 with MixEval.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Data Contamination"
  - "Shortcut Neuron"
  - "Activation Patching"
  - "Trustworthy Evaluation"
  - "Benchmark Fairness"
date: 2026-05-08
content_hash: a1e488b46621cd2e
---

# Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis

**Conference**: ACL 2025  
**arXiv**: [2506.04142](https://arxiv.org/abs/2506.04142)  
**Code**: [https://github.com/GaryStack/Trustworthy-Evaluation](https://github.com/GaryStack/Trustworthy-Evaluation)  
**Area**: Interpretability  
**Keywords**: Data Contamination, Shortcut Neuron, Activation Patching, Trustworthy Evaluation, Benchmark Fairness  

## TL;DR

This paper proposes to locate "shortcut neurons" in contaminated models through comparative analysis and causal analysis, and suppress these neurons via activation patching to achieve more trustworthy LLM evaluation, achieving a Spearman correlation coefficient of over 0.95 with MixEval.

## Background & Motivation

**Background**:
   - With the rapid development of LLMs, evaluation has become increasingly crucial.
   - Most evaluations rely on public benchmarks (e.g., GSM8K, MATH), but the large-scale and opaque nature of training data leads to severe data contamination issues.
   - Data contamination causes artificially inflated scores on benchmarks, severely undermining evaluation fairness.
   - Current dominant methods to address contamination focus on building dynamic benchmarks, which is costly and does not cure the root cause.

**Limitations of Prior Work**:
   - **Behavioral Shortcuts (A1)**: End-to-end LLMs might take "shortcut" reasoning instead of true reasoning processes, leading to distrust in their reasoning capabilities.
   - **Input Format Shortcuts (A2)**: Models might overfit to the specific input formats of benchmarks rather than truly learning problem-solving skills.
   - Building dynamic benchmarks is highly expensive, and new benchmarks still face the risk of contamination.
   - Existing methods focus on external benchmark replacement, lacking mechanisms to understand and resolve contamination from the model's internal perspective.

**Key Challenge**:
   - The need to enable a contaminated model to demonstrate its true capabilities as if it "had never seen" the benchmark data, rather than simply replacing the test data.
   - How to precisely suppress the "shortcuts" acquired through contamination without affecting the model's true capabilities?

**Goal**:
   - To understand and mitigate the effects of data contamination from the perspective of internal model neurons, thereby establishing a more trustworthy evaluation methodology.

**Key Insight**:
   - Starting from the Transformer neuron mechanism, it is found that sparse "shortcut neurons" exist in contaminated models.
   - By locating these neurons through comparative and causal analysis, the shortcuts can be suppressed by patching the activations with those of the base model.

**Core Idea**:
   - Contamination causes the model to acquire a small number of "shortcut neurons" to solve problems via shortcuts. Locating and suppressing these neurons can restore the model's true performance.

## Method

### Overall Architecture

The methodology is divided into two phases:
1. **Locate Phase**: Locating shortcut neurons through comparative analysis + causal analysis.
2. **Patch Phase**: Replacing the activation values of shortcut neurons in the model under test with those from the base model.

### Key Designs

1. **Comparative Analysis**:

    - Function: Compare the difference in neuron activation between the contaminated model M_con and the uncontaminated model M_un when processing the same benchmark samples.
    - Core Formula: S_i^l = sqrt(Σ(a_i^l(x_T|M_con) - a_i^l(x_T|M_un))² / |D|)
    - Using the activation of the last token (which is more effective than the average).
    - Design Motivation: Neurons with large activation differences are more likely to be associated with memory shortcuts.
    - Implementation: For each neuron in each layer, calculate its RMS activation difference on the contaminated benchmark data.

2. **Causal Analysis**:

    - Function: Verify the causal effect of candidate neurons via activation patching.
    - Mechanism: A true shortcut neuron should satisfy two conditions:
        - (a) patching significantly reduces the score of the contaminated model (disrupting shortcut reasoning).
        - (b) patching has minimal impact on the score of the uncontaminated model (not affecting genuine capabilities).
    - Causal Score Formula: C_N = [a(M_con) - a_patch(M_con|M_0)] + [1 - (a(M_un) - a_patch(M_un|M_0))]
    - Design Motivation: Relying solely on comparative analysis may introduce noise; causal analysis provides more precise validation.

3. **Dynamic Patching**:

    - Function: Perform activation patching token-by-token during the generation process.
    - Mechanism:
        - Step 1: Run the patching model (base model) and cache the activation values of designated neurons.
        - Step 2: Run the patched model and replace the corresponding neuron activations.
        - Step 3: Predict the next token, append it to the prompt, and repeat the above steps.
    - Design Motivation: Traditional patching methods target short-output tasks, whereas open-ended tasks like mathematical reasoning require progressive, dynamic patching.

4. **Trustworthy Evaluation Framework**:

    - Function: Patch the model under test M_e using the shortcut neuron activations from the base model M_0.
    - Mechanism:
        - If M_e is contaminated $\rightarrow$ the score drops significantly after patching (shortcuts are suppressed).
        - If M_e is not contaminated $\rightarrow$ the score remains largely unchanged after patching (no shortcuts to suppress).
    - Practical Application: Directly patching the model yields a more trustworthy score without needing to know whether the model is contaminated beforehand.

### Loss & Training

- The method itself **does not require training**. The core operation is activation replacement during inference.
- Prerequisites to prepare:
    - A base model M_0 of the same architecture (e.g., LLaMA2-7B base).
    - A contaminated model M_con and an uncontaminated model M_un (used to locate shortcut neurons).
- Hyperparameters: temperature=1, top-p=1, top-k=50.
- Neuron grouping: Every 512 adjacent neurons are grouped to calculate causal effects.

## Key Experimental Results

### Main Results

**GSM8K Trustworthy Evaluation Results** (LLaMA2-7B):

| Model Variant | Ref Score | Original Score | TE Score | Δ_acc |
|----------|---------|---------|---------|-------|
| Vanilla | 16.7 | 18.5 | 18.5 | — |
| +GSM-i (Contaminated) | 26.7 | 40.5 | 27.0 | **-13.5** |
| +5×GSM-i (Heavily Contaminated)| 23.7 | 80.0 | 30.2 | **-49.8** |
| +OpenOrca (Uncontaminated)| 21.0 | 20.2 | 21.5 | +1.3 |
| +GSM8K Train | 24.6 | 35.0 | 28.5 | -6.5 |
| +MATH (Uncontaminated)| 20.6 | 19.5 | 19.0 | -0.5 |

- For the heavily contaminated model (+5×GSM-i), the original score of 80.0 is reduced to a TE score of 30.2, largely eliminating the inflation.
- Uncontaminated models (+OpenOrca, +MATH) maintain almost unchanged scores after patching.

**Similar results observed on Mistral-7B**: 5×GSM-i dropped from 88.7 to 45.6 (-43.1).

### Key Findings

1. **Shortcut neurons are sparse**: Only about 5,000 neurons (accounting for 1.4% of total neurons in LLaMA2-7B) are sufficient to effectively eliminate contamination.
2. **Saturation beyond 5,000**: Patching more neurons begins to degrade the model's normal capabilities.
3. **High correlation with MixEval**: On real-world models, the Spearman correlation between the patched evaluation scores and the MixEval reference scores is > 0.95.
4. **Input format shortcuts are also eliminated**: Format shortcuts are detected and eliminated even if the model is only fine-tuned on the GSM8K training set (not direct data contamination).
5. **No impact on general capabilities**: The performance of the patched model on MAWPS and MMLU shows no significant change.
6. **Cross-architecture generalizability**: The effectiveness of the method is validated across both LLaMA2 and Mistral architectures.

## Highlights & Insights

- **Addressing the problem from the mechanism of contamination**: Distinct from "defensive" ideas of constructing new benchmarks, this paper "cures" existing contamination directly from the model's internal mechanism.
- **Profound impact of the sparsity finding**: Only 1.4% of neurons determine the inflation caused by contamination, implying that LLM "memories" are highly localized.
- **Dual causal validation**: Requiring both a major impact on the contaminated model and zero impact on the normal model to avoid collateral damage.
- **Practical application prospects**:
    - No access to the model training data is required.
    - No need to know whether the model is contaminated.
    - Can be implemented using only a base model of the same architecture.
- **High correlation with MixEval** provides strong external validation.

## Limitations & Future Work

1. **Requirement of uncontaminated models**: Locating shortcut neurons requires a pair of contaminated and uncontaminated models of the same architecture, which may be difficult to obtain in practice.
2. **Dependence on base models**: Patching requires activation values from a base model of the exact same architecture, limiting applicability to closed-source models.
3. **Validation limited to mathematical reasoning benchmarks**: The effectiveness on other benchmark datasets such as code or reading comprehension remains to be validated.
4. **Grouped evaluation (512 neurons per group)** may lose granularity.
5. **Inference cost of dynamic patching**: Running two models simultaneously doubles the inference cost.
6. **Optimal number of neurons may require adjustment** depending on the severity of contamination.

## Related Work & Insights

- **Knowledge Neurons** (Dai et al., 2021): Discovered neurons that store factual knowledge; this paper analogously discovers neurons storing "shortcut knowledge."
- **Skill Neurons** (Wang et al., 2022): Neurons associated with specific language skills.
- **Activation Patching** (Meng et al., 2022; Vig et al., 2020): The standard method for causal intervention.
- **MixEval** (Ni et al., 2024): A trustworthy benchmark aligned with real-world user queries, serving as an external reference in this study.
- **Insight**: Understanding and intervening in LLM behavior at the neuron level can be extended to more scenarios (e.g., safety alignment, hallucination mitigation).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first work to analyze and eliminate the impact of data contamination from the perspective of neuron mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulation of contamination combined with real-world model evaluation, though with a limited range of benchmark types.
- Writing Quality: ⭐⭐⭐⭐ — Clear framework diagrams and rigorous mathematical formalization.
- Value: ⭐⭐⭐⭐⭐ — Significant contribution to the fairness of LLM evaluation, providing a practical and scalable approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Neuron-Level Analysis of Cultural Understanding in Large Language Models](../../ICLR2026/interpretability/neuron-level_analysis_of_cultural_understanding_in_large_language_models.md)
- [\[NeurIPS 2025\] Rectifying Shortcut Behaviors in Preference-based Reward Learning](../../NeurIPS2025/interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[ICML 2026\] Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis](../../ICML2026/interpretability/accurate_evaluation_of_quickest_changepoint_detectors_via_non-parametric_surviva.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)

</div>

<!-- RELATED:END -->
