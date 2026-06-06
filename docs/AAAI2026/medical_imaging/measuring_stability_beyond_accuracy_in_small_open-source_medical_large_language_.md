---
title: >-
  [Paper Note] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology
description: >-
  [AAAI 2026][Medical Imaging][Medical LLM] This paper systematically evaluates six small open-source medical LLMs (<10B parameters) in pediatric endocrinology…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Medical LLM"
  - "Small Model Evaluation"
  - "Stability"
  - "Reproducibility"
  - "Prompt Sensitivity"
  - "Pediatric Endocrinology"
  - "Self-Evaluation Bias"
date: 2026-05-08
content_hash: 54f64244f8847e60
---

# Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology

**Conference**: AAAI 2026
**arXiv**: [2601.11567](https://arxiv.org/abs/2601.11567)  
**Code**: [GitHub](https://github.com/vanessadamario/SmallMedLLMs-PedEndo)  
**Area**: Medical AI / LLM Evaluation
**Keywords**: Medical LLM, Small Model Evaluation, Stability, Reproducibility, Prompt Sensitivity, Pediatric Endocrinology, Self-Evaluation Bias

## TL;DR
This paper systematically evaluates six small open-source medical LLMs (<10B parameters) in pediatric endocrinology, demonstrating that accuracy alone is insufficient to characterize model reliability: semantically neutral prompt variations lead to significant output shifts (Stuart-Maxwell $p < 10^{-4}$), high consistency does not imply correctness, and even differences in CUDA versions can induce statistically significant output distribution changes.

## Background & Motivation

**Background**: Large medical LLMs (e.g., GPT-4-class, 70B+ parameters) have demonstrated strong performance across diverse clinical tasks; however, these models are typically closed-source, computationally expensive, and inaccessible in low-resource or low-income healthcare settings. Small open-source medical LLMs (<10B parameters) offer a lightweight, locally deployable, and transparent alternative.

**Limitations of Prior Work**:
- Existing evaluations of medical LLMs rely predominantly on MCQ accuracy, neglecting **consistency**, **robustness**, and **reasoning quality**.
- Sensitivity of LLMs to prompt variations has been reported in non-medical domains but lacks systematic investigation in medical LLMs.
- Whether small models encode sufficient clinical knowledge to support real-world deployment remains an open question.
- Reproducibility concerns — hardware/software stack differences may yield clinically divergent outputs.

**Why Pediatric Endocrinology?** The field faces growing patient volumes, extended wait times, and a shortage of specialists, making it a promising target for AI-assisted decision support. Yet no LLM evaluation exists for this specialty.

**Core Problem**: How stable and reproducible are small open-source medical LLMs beyond accuracy? Does accuracy alone reflect actual model reliability?

## Method

### Overall Architecture
A multi-dimensional evaluation framework comprising three core experiments plus a numerical stability analysis.

### Evaluation Data
- **Pediatric ESAP 2021–2022**: Pediatric Endocrinology Self-Assessment Program, 100 clinical case/knowledge questions (MCQ format, 5-choice single-answer).
- 9 questions requiring images/figures were excluded, yielding 91 items.
- Each item includes a clinical vignette, question stem, five options, correct answer, and expert gold-standard explanation.

### Evaluated Models (6 Small Open-Source Medical LLMs)

| Model | Base Model | Parameters |
|-------|-----------|------------|
| HuatuoGPT-o1-8B | LLaMA 3.1 8B | ~8B |
| Diabetica-7B | Qwen2-7B | ~7B |
| Diabetica-o1 | Self-distilled from Diabetica-7B | ~7B |
| Meditron3-8B | LLaMA 3.1-8B | ~8B |
| MedFound-7B | BLOOM-7B | ~7B |
| ClinicalGPT-base-zh | BLOOM-7B | ~7B |

### Experimental Design

**Experiment 1: Accuracy and Prompt Stability (Deterministic Setting, $T=0$)**
- Three prompt strategies: original prompt A, grammatical variant prompt B, prompt A with option letters removed.
- Evaluation dimensions: accuracy, McNemar test (accuracy change), Stuart-Maxwell test (output distribution change), Cohen's $\kappa$ (consistency), match rate.

**Experiment 2: Stability Under Non-Deterministic Settings**
- The top four models from Experiment 1 are run 10 times each at $T = 0.3 / 0.6 / 1.0$.
- The relationship between consistency (majority vote frequency) and correctness is assessed.

**Experiment 3: Self-Evaluation Bias and Gold-Standard Reasoning Discrimination**
- Models are presented with two candidate explanations (their own vs. the expert gold standard) and asked to select the superior one.
- Position bias is tested by swapping the order of the two explanations.
- A pediatric endocrinology expert manually reviews cases where HuatuoGPT-o1 answered incorrectly.

**Numerical Stability Analysis**
- Identical inference is run on two machines with different GPU/CUDA versions (CUDA 11.8 vs. CUDA 12.8).
- The effect of hardware–software stack differences on model outputs is examined.

## Key Experimental Results

### Experiment 1: Accuracy and Prompt Stability

| Model | Prompt A Acc. | Prompt B Acc. | No-Letter Acc. | Usability |
|-------|--------------|--------------|----------------|-----------|
| HuatuoGPT-o1-8B | **0.35** | **0.35** | **0.33** | 100% |
| Diabetica-o1 | 0.33 | 0.34 | 0.27 | 100% |
| Meditron3-8B | 0.33 | No output | 0.34 | 97.8% |
| Diabetica-7B | 0.30 | 0.32 | 0.27 | 98.9% |
| ClinicalGPT-base-zh | 0.20 | 0.20 | 0.20 | 79.1% |
| MedFound-7B | 0.04 | 0.12 | 0.04 | 18.6% |

**Key Findings**:
- McNemar tests show no significant accuracy changes across all models ($p > 0.4$).
- However, Stuart-Maxwell tests reveal **significant output distribution shifts** ($p < 10^{-4}$), i.e., accuracy stability $\neq$ behavioral stability.
- Cohen's $\kappa \leq 0.4$ when option letters are removed, indicating that models rely heavily on letter tokens rather than semantic understanding.
- Meditron3-8B fails to produce any valid output under prompt B.

### Experiment 2: Consistency vs. Correctness Under Non-Deterministic Settings

| Model | T=0.3 Consistent & Correct | T=0.3 Consistent & Incorrect | T=1.0 Consistent & Correct | T=1.0 Consistent & Incorrect |
|-------|---------------------------|------------------------------|---------------------------|-------------------------------|
| HuatuoGPT-o1 | 14 | 11 | 10 | 4 |
| Diabetica-7B | 13 | 16 | 3 | 7 |
| Diabetica-o1 | 5 | 4 | 5 | 3 |
| Meditron3-8B | 2 | 0 | 0 | 0 |

**Key Findings**:
- **High consistency does not imply correctness**: Diabetica-7B at $T=0.3$ produces more consistent-but-incorrect cases (16) than consistent-and-correct ones (13), which could lead to systematic recommendation of inappropriate treatments in clinical settings.
- HuatuoGPT-o1 is the only model where correct answers outnumber incorrect ones among highly consistent responses.
- At higher temperatures, majority votes from Meditron3-8B collapse to "hallucination/no output" (71/91 cases).

### Experiment 3: Self-Evaluation Bias

- Among 59 incorrect cases for HuatuoGPT-o1: 27 selected the gold standard in both positions, 19 in only one position, and 8 never selected it.
- Diabetica-o1 performs worse: 19 cases consistently selected the gold standard, 19 never did.
- **Position bias** is evident: the presentation order of explanations influences model selection.
- Expert review of 12 incorrect HuatuoGPT-o1 cases yielded Likert scores of 2–4; only 2 cases exhibited clear factual errors, with most exhibiting incomplete reasoning.

### Numerical Stability

| Model | $\Delta$ Accuracy | Stuart-Maxwell $p$ | Cohen's $\kappa$ | Match Rate |
|-------|-------------------|-------------------|-----------------|------------|
| HuatuoGPT-o1 | +0.044 | $<10^{-4}$ | 0.51 | 0.60 |
| Diabetica-o1 | −0.022 | $<10^{-4}$ | 0.53 | 0.63 |
| Diabetica-7B | +0.011 | $<10^{-4}$ | 0.68 | 0.75 |
| Meditron3-8B | +0.011 | $<10^{-4}$ | 0.31 | 0.45 |

- **CUDA version differences alone induce statistically significant output distribution shifts**, even when accuracy differences fall within confidence intervals.
- Meditron3-8B exhibits the lowest cross-system consistency ($\kappa = 0.31$), producing different answers for 45% of cases.

## Highlights & Insights

1. **"Accuracy stability $\neq$ behavioral stability"**: McNemar tests pass while Stuart-Maxwell tests fail, exposing blind spots in conventional evaluation metrics.
2. **"Consistency $\neq$ correctness"**: Models may be highly consistent in producing incorrect answers, potentially leading to systematic clinical misguidance.
3. **Letter-token dependence over semantic understanding**: Removing option letters substantially degrades consistency ($\kappa \leq 0.4$), revealing reliance on surface-level cues.
4. **Hardware–software stack affects clinical decisions**: CUDA version differences alone can change clinical recommendations in 25–55% of cases, posing a serious reproducibility challenge.
5. **Methodological contribution**: A multi-dimensional diagnostic framework (prompt sensitivity + stochastic stability + reasoning consistency + numerical stability) is proposed and is generalizable to other settings.
6. **Complementarity of manual and automated evaluation**: GPT-4-based automatic evaluation is abandoned due to its own inconsistency; a mixed non-expert and expert review strategy is adopted instead.

## Limitations & Future Work

1. **Low absolute performance**: The best model achieves only 35% accuracy (vs. 18.2% random chance), and no human baseline is provided for comparison.
2. **MCQ format only**: Open-ended question answering and scenario-based evaluation are not considered; MCQ format may mask reasoning inconsistencies.
3. **Limited hyperparameter exploration**: Only prompt and temperature are varied; other inference hyperparameters (top-$p$, repetition penalty, etc.) are held fixed.
4. **Limited numerical stability analysis**: Only two machines are compared; a more systematic survey of hardware configurations is not conducted.
5. **Non-expert review may miss subtle medical errors**: Expert review is used as a supplement, but comprehensive expert evaluation is resource-constrained.
6. **Small dataset**: Only 91 cases, limiting statistical power.

## Related Work & Insights

- **Medical LLM evaluation**: MedQA, MedMCQA, PubMedQA benchmarks; HealthBench (OpenAI).
- **LLM stability research**: Prompt sensitivity (li2024can, khatun2024study), self-evaluation bias (xu2024pride).
- **Small open-source medical LLMs**: HuatuoGPT-o1, Meditron3, MedFound, ClinicalGPT.
- **Reproducibility concerns**: CUDA non-determinism (he2025nondeterminism), inference pipeline variability.

## Rating ⭐⭐⭐⭐

This is the first work to systematically evaluate small medical LLMs from the perspective of stability and reproducibility, with significant methodological contributions. In particular, the findings that "accuracy stability $\neq$ behavioral stability" and "CUDA version differences affect clinical outputs" carry important cautionary implications. However, the small dataset and low absolute performance limit the generalizability of the conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)

</div>

<!-- RELATED:END -->
