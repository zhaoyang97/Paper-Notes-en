---
title: >-
  [Paper Note] Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models
description: >-
  [ACL2026][LLM Safety][machine unlearning] This paper demonstrates that unlearned LLMs may rely more on dataset shortcut tokens for decision-making even while maintaining low calibration error. Consequently, using ECE…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "machine unlearning"
  - "calibration"
  - "reliability paradox"
  - "shortcut learning"
  - "Integrated Gradients"
date: 2026-05-08
content_hash: 85833775a4dba8fc
---

# Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models

**Conference**: ACL2026  
**arXiv**: [2605.20915](https://arxiv.org/abs/2605.20915)  
**Code**: https://github.com/Exploration-Lab/Unlearning-Reliability-Paradox  
**Area**: LLM Security / Machine Unlearning / Reliability Evaluation  
**Keywords**: machine unlearning, calibration, reliability paradox, shortcut learning, Integrated Gradients

## TL;DR
This paper demonstrates that unlearned LLMs may rely more on dataset shortcut tokens for decision-making even while maintaining low calibration error. Consequently, using ECE, MCE, or Brier scores alone is insufficient to determine whether an unlearned model is reliable.

## Background & Motivation
**Background**: Machine unlearning aims to remove the influence of specific training data from a model while preserving remaining knowledge and reliable behavior. Existing evaluations often focus on whether forgetting occurs on the forget split, whether performance is maintained on the retain split, and whether model confidence is calibrated. Calibration metrics such as ECE, MCE, and Brier scores are frequently used as proxies for reliability.

**Limitations of Prior Work**: Calibration only indicates whether the probabilities provided by the model match empirical accuracy; it does not explain why the model made a certain decision. A model may have accurate confidence levels while actually relying on dataset spurious correlations, option formats, or high-frequency words (shortcuts) rather than semantic evidence within the question.

**Key Challenge**: Unlearning explicitly modifies model parameters, which may alter internal decision rules. The fact that a model appears well-calibrated on the retain split does not guarantee it still uses reasonable features to answer; it may maintain probabilistic performance through increased shortcut reliance. This is the reliability paradox that the authors extend to the machine unlearning scenario.

**Goal**: The authors aim to simultaneously evaluate two types of reliability in generative decoder-only LLMs: probabilistic reliability (whether confidence matches accuracy) and decision-rule reliability (whether the model relies on semantically meaningful tokens rather than dataset-level shortcuts). The experiments are conducted on Llama-3.1-8B using TOFU/RELU MCQA with various unlearning algorithms.

**Key Insight**: The paper utilizes the multiple-choice QA (MCQA) format of RELU as a bridge. Fixed options allow LLMs to output normalizable probabilities for calibration calculation, while MCQA inputs enable the use of Integrated Gradients to analyze token contributions to the predicted option logit.

**Core Idea**: Combine calibration metrics with shortcut detection based on attribution and Local Mutual Information (LMI) to investigate whether a "low ECE but high shortcut proportion" paradox emerges after unlearning.

## Method
The paper does not propose a new unlearning algorithm but introduces a reliability evaluation framework. It compares pretrained, full-finetuned, retained, and various approximate unlearned models, reporting task performance, calibration error, and shortcut usage proportion on both forget and retain splits.

### Overall Architecture
Experiments start with the TOFU dataset, partitioned into forget and retain data at 1%, 5%, and 10% forget ratios. A Llama-3.1-8B model is first fine-tuned on the complete TOFU dataset to obtain the full-finetuned model. A retained model is trained separately on the retain split as an ideal unlearning approximation. Several approximate unlearning methods, including Gradient Ascent, Gradient Difference, Negative Preference Optimization (NPO), and Direct Preference Optimization (DPO), are executed starting from the full-finetuned model.

For evaluation, RELU converts TOFU QA into four-option MCQA. The model calculates likelihoods for each option, normalized into a probability distribution. These probabilities are used to compute accuracy, F1, Brier, ECE, and MCE. The authors then use Integrated Gradients to identify the top-10 tokens influencing the predicted option logit and use LMI to identify the top 5% of tokens highly correlated with labels. If a predicted high-attribution token overlaps with a high-LMI token for the corresponding label, it is recorded as a shortcut-cued prediction.

### Key Designs
1. **Separation of Probabilistic and Decision Reliability**:
    - **Function**: Avoids misinterpreting low calibration error as "reliable model decision-making."
    - **Mechanism**: Probabilistic reliability is measured by ECE, MCE, and Brier; decision reliability is measured by shortcut proportion $P_{SC}$ and trade-off score $T_{SC}$. Reporting them separately allows for the observation of low ECE coupled with high $P_{SC}$.
    - **Design Motivation**: The goal of unlearning is not just to ensure the model answers correctly with reasonable confidence on the retain split, but also to ensure it does not rely on non-generalizable spurious correlations to maintain performance.

2. **Integrated Gradients + LMI Shortcut Detection**:
    - **Function**: Identifies whether the model relies on dataset-level tokens correlated with answer labels that may lack semantic meaning.
    - **Mechanism**: IG measures the influence of each input token on the predicted logit from the model side; LMI measures the statistical correlation between tokens and labels from the corpus side. If a token has both high IG and high LMI, the model is using a label-predictive correlation cue. The authors define $P_{SC}=\frac{\text{shortcut-cued predictions}}{\text{total predictions}}$ as the shortcut reliance metric.
    - **Design Motivation**: Attribution alone does not indicate if a token is semantic evidence, and LMI alone does not indicate if the model used it; their intersection is better suited for analyzing actual reliance on dataset shortcuts.

3. **Multi-forgetting Ratios and Multi-algorithm Comparison**:
    - **Function**: Observes whether the reliability paradox is specific to certain algorithms or forget ratios.
    - **Mechanism**: Experiments cover forget ratios of 1%, 5%, and 10%, comparing retained, GradAscent, GradDiff, NPO, and DPO states. All results emphasize the retain split, as its reliability should be preserved after unlearning.
    - **Design Motivation**: Examining a single setting makes it difficult to distinguish accidental fluctuations from structural phenomena. Multi-setting comparison reveals systematic changes in calibration and shortcut reliance relative to unlearning intensity.

### Loss & Training
Full fine-tuning utilizes AdamW with a learning rate of $1\times10^{-5}$, a linear scheduler, and 5 training epochs. Retained models are trained on 99%, 95%, and 90% retain splits respectively with the same settings. Approximate unlearning starts from the full-finetuned model using recommended hyperparameters for Gradient Ascent, Gradient Difference, NPO, and DPO; 4-bit quantized LoRA is used to reduce costs.

Calibration is calculated using 10 equal-width confidence bins. IG attribution calculates gradients for the predicted answer option logit using a zero embedding baseline, 50-step Riemann sum approximation, and absmax for subword aggregation, selecting the top-10 tokens per sample. LMI selects the top 5% of highly correlated words for each label as shortcut candidates.

## Key Experimental Results

### Main Results
The following table extracts the retain 90% setting, which best illustrates the reliability paradox at higher forgetting ratios.

| Model State | Acc | F1 | Brier | ECE | MCE | $P_{SC}$ | $T_{SC}$ | Interpretation |
|----------|-----|----|-------|-----|-----|----------|----------|------|
| Pretrained | 0.266 | 0.146 | 1.144 | 0.516 | 0.665 | 80.0 | 0.047 | Near random, poor calibration |
| Full finetuned | 0.694 | 0.699 | 0.410 | 0.039 | 0.070 | 85.0 | 0.431 | Good performance and calibration, but high shortcut |
| Retained | 0.639 | 0.640 | 0.478 | 0.038 | 0.081 | 90.0 | 0.393 | ECE similar to full, but higher shortcut |
| GradAscent | 0.235 | 0.155 | 0.841 | 0.209 | 0.383 | 92.5 | 0.073 | Degraded performance and calibration |
| GradDiff | 0.476 | 0.472 | 0.689 | 0.133 | 0.221 | 92.5 | 0.255 | Decreased performance, high shortcut |
| NPO | 0.527 | 0.525 | 0.601 | 0.101 | 0.159 | 92.5 | 0.304 | Moderate performance, high shortcut |
| DPO | 0.483 | 0.434 | 0.648 | 0.052 | 0.089 | 92.5 | 0.266 | ECE remains low, but shortcut significantly high |

Comparing the retain split for full-finetuned and DPO across different forget ratios shows that while DPO maintains low ECE, $P_{SC}$ increases as the forget ratio rises.

| Forget / Retain | Model | Retain ECE | Retain $P_{SC}$ | Key Information |
|-----------------|------|------------|------------------|----------|
| 1% / 99% | Full | 0.040 | 85.0 | High calibration after fine-tune |
| 1% / 99% | DPO | 0.033 | 85.0 | No significant shortcut increase at low ratios |
| 5% / 95% | Full | 0.040 | 85.0 | Full model remains stable |
| 5% / 95% | DPO | 0.046 | 87.5 | Shortcut begins to rise |
| 10% / 90% | Full | 0.039 | 85.0 | Full model low ECE |
| 10% / 90% | DPO | 0.052 | 92.5 | Low ECE coexists with high shortcut |

### Ablation Study
While there is no traditional module ablation, the paper provides qualitative examples of shortcuts. In the retain 90% setting, a DPO unlearned model selects the correct answer A, but attribution shows it relies on functional words rather than the semantic entities in the question.

| Question Segment | Pred / Truth | Shortcut token | Attribution | LMI | Meaning |
|----------|-------------|----------------|-------------|-----|------|
| How does Andres Santiago Cruz's family background influence... | A / A | does | 0.0140 | 0.0578 | High contribution and label-related, but not semantic |
| Same question | A / A | about | 0.0103 | 0.0543 | Model may answer based on format/functional cues |

### Key Findings
- Pretrained models are near random on TOFU fictitious facts, with ECE typically above 0.5; after full fine-tuning, the retain split ECE drops to approximately 0.04.
- After unlearning, calibration error can remain low; for example, DPO at retain 90% has an ECE of 0.052, but $P_{SC}$ reaches 92.5%.
- Low ECE does not imply low shortcut reliance. In several model states, models with better calibration actually use more shortcut tokens.
- $T_{SC}$ often decreases after unlearning, but because F1 also decreases, the authors argue that this trade-off score alone cannot judge reliability.
- Qualitative examples show that unlearned models may rely on grammatical or formatting cues like "does" or "about" rather than semantic tokens like author names or specific topics.

## Highlights & Insights
- The primary value of the paper is decomposing "reliability" in unlearning. Forgetting target data and maintaining accuracy is the first level; whether the model answers using reasonable evidence is a deeper level of reliability.
- The calibration paradox is transferred to decoder-only LLMs and machine unlearning, serving as a practical safety warning. Deployment pipelines often treat low ECE as a proxy for trustworthiness, but this paper shows ECE only reflects probability matching, not causality or semantic evidence.
- The combination of IG + LMI is concise: one measures what the model actually depends on, while the other measures which words correlate with labels in the dataset. While not perfect, it is more interpretable than just looking at attention or token frequency.
- Results suggest that some unlearning algorithms may use "shortcut compensation" to maintain surface reliability. After forgetting certain facts, the model may shift toward shallower decision rules.

## Limitations & Future Work
- Evaluation is limited to Llama-3.1-8B; calibration and shortcut behavior may differ across other model sizes, architectures, or instruction tuning styles.
- Experiments are restricted to TOFU/RELU MCQA settings. While convenient for probability and attribution calculations, they do not fully represent forgetting and reliability in open-ended generation.
- Shortcut detection is an approximation. Integrated Gradients is affected by baselines, tokenization, and aggregation methods; LMI only captures dataset-level statistical correlations.
- Calibration is only calculated on multiple-choice probabilities; token-level or sequence-level calibration for open-ended generation remains an unsolved problem.
- The paper does not evaluate robustness or OOD performance. Future work should investigate whether shortcut reliance leads to generalization and safety issues in real-world deployment.

## Related Work & Insights
- **vs. TOFU / RELU**: While TOFU and RELU provide the forgetting and retention protocols, this paper adds calibration and shortcut attribution analysis on top of them.
- **vs. Bihani and Rayz’s Reliability Paradox**: Prior work demonstrated that calibration does not equal reliable decision-making in encoder classification models; this paper extends it to decoder-only LLMs and machine unlearning.
- **vs. Conventional Unlearning Evaluation**: Simply looking at forget/retain accuracy ignores how those scores are achieved. This paper shows models can still take shortcuts even when accuracy and ECE on the retain split appear good.
- **Insights for Safety Evaluation**: Future unlearning benchmarks should simultaneously report performance, calibration, attribution, shortcuts, robustness, and OOD behavior, rather than treating low ECE as a definitive proof of trustworthiness.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Successfully introduces the reliability paradox to the machine unlearning field.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Multi-ratio and multi-algorithm comparisons are valuable, but the model and benchmark range is narrow, and open-ended generation is not covered.
- Writing Quality: ⭐⭐⭐⭐☆ Background, metrics, and result interpretations are clear and the claims are measured.
- Value: ⭐⭐⭐⭐☆ Provides a direct warning for LLM safety, unlearning evaluation, and the interpretation of calibration metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_safety/revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ACL 2026\] Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks](before_forgetting_learn_to_remember_revisiting_foundational_learning_failures_in.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
