---
title: >-
  [Paper Note] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models
description: >-
  [ACL 2025][Multimodal VLM][unsolvable problem detection] This work propounds the Unsolvable Problem Detection (UPD) task to systematically evaluate whether Large Multimodal Models (LMMs) can correctly refuse to answer when faced with unanswerable MCQA questions across three types of unsolvable problems (absent answers, incompatible options, and image-text mismatches), revealing a dimension of trustworthiness overlooked by existing benchmarks.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "unsolvable problem detection"
  - "trustworthiness"
  - "multimodal evaluation"
  - "answer refusal"
  - "MCQA"
date: 2026-05-08
content_hash: f75948cf99e7603e
---

# Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models

**Conference**: ACL 2025  
**arXiv**: [2403.20331](https://arxiv.org/abs/2403.20331)  
**Code**: [https://github.com/AtsuMiyai/UPD](https://github.com/AtsuMiyai/UPD)  
**Area**: Multimodal VLM  
**Keywords**: unsolvable problem detection, trustworthiness, multimodal evaluation, answer refusal, MCQA

## TL;DR

This work propounds the Unsolvable Problem Detection (UPD) task to systematically evaluate whether Large Multimodal Models (LMMs) can correctly refuse to answer when faced with unanswerable MCQA questions across three types of unsolvable problems (absent answers, incompatible options, and image-text mismatches), revealing a dimension of trustworthiness overlooked by existing benchmarks.

## Background & Motivation

**Background**: Multiple-Choice Question Answering (MCQA) is the mainstream approach to evaluate the comprehension capabilities of LMMs, with benchmarks like MMBench and MMMU being widely adopted. Currently, LMMs perform exceptionally well on these benchmarks, with many models achieving accuracies exceeding 80%.

**Limitations of Prior Work**: High accuracy does not imply that the models truly understand the answers. Models might merely select the "least absurd" option among the choices rather than genuinely knowing the correct answer. When the correct answer is absent from the options, the models will still "force" themselves to select an option, exposing their lack of genuine comprehension.

**Key Challenge**: Existing evaluations only consider ideal scenarios where "answers exist and are solvable," neglecting unsolvable situations that may arise in reality. While prior work in the LLM domain has investigated refusal capabilities, the types of unsolvable problems in multimodal scenarios are more diverse (e.g., image-text mismatches) and lack systematic evaluation.

**Goal**: (1) Define a classification system for unsolvable problems in multimodal scenarios; (2) build a rigorous evaluation benchmark; (3) systematically evaluate the refusal capabilities of existing LMMs and analyze their bottlenecks.

**Key Insight**: Categorize unsolvable problems in multimodal MCQA into three classes: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD), covering all possible inconsistencies among the three elements: images, questions, and options.

**Core Idea**: Validate whether models "truly understand" the answers instead of merely performing option elimination by constructing three types of unsolvable problems.

## Method

### Overall Architecture

Based on MMBench, the MM-UPD Bench is constructed, consisting of three sub-benchmarks: MM-AAD (820 questions), MM-IASD (919 questions), and MM-IVQD (356 questions), totaling 2095 questions. The input consists of a multiple-choice question with an image. The model must determine whether the problem is solvable; if solvable, it answers normally, and if unsolvable, it refuses to answer. The evaluation employs Dual Accuracy as the core metric, which requires the model to answer correctly on the standard problem *and* correctly refuse on the corresponding unsolvable problem to be considered successful.

### Key Designs

1. **Definition of Three Unsolvable Problem Types**:

    - Function: Systematically cover all unsolvable scenarios in multimodal MCQA.
    - Mechanism: AAD removes the correct option to examine whether the model can recognize missing answers; IASD replaces the options with a completely irrelevant set to check if the model can identify option-question mismatches; IVQD shuffles image-text pairings to evaluate whether the model can identify image-text irrelevance.
    - Design Motivation: The three types of problems test different levels of understanding—AAD tests fine-grained judgment, IASD tests basic semantic matching, and IVQD tests visual-text alignment.

2. **Three-Stage Benchmark Construction Pipeline**:

    - Function: Ensure high-quality and unambiguous evaluation data.
    - Mechanism: (1) Filter out questions that can be answered without images using a text-only GPT-4 combined with CircularEval; (2) generate unsolvable problems by removing correct options (AAD), shuffling option sets (IASD), and shuffling image-question pairs (IVQD); (3) conduct manual review to eliminate ambiguous samples.
    - Design Motivation: If a question can be answered without relying on the image, multimodal understanding cannot be evaluated effectively. Manual review ensures that unsolvable problems are genuinely unsolvable.

3. **Dual Accuracy Evaluation Metric**:

    - Function: Comprehensively measure model performance on both solvable and unsolvable problems.
    - Mechanism: Correctness is credited only when the model answers correctly on the standard problem **and** correctly refuses on the corresponding unsolvable problem. $\text{Dual Acc} = \mathbb{1}[\text{Standard correct}] \times \mathbb{1}[\text{UPD correct}]$
    - Design Motivation: Analyzing Standard Accuracy or UPD Accuracy in isolation is insufficient; the former neglects the refusal ability, while the latter may award high scores to models that simply refuse everything.

### Loss & Training

As this work focuses on evaluation, it does not involve training losses. The evaluation setup includes three prompting strategies: Base (no prompt), Option (adding a "None of the above" option), and Instruction (explicitly instructing that refusal is allowed). Three improvement strategies—CoT, Self-reflection, and fine-tuning—are additionally explored.

## Key Experimental Results

### Main Results

| Model | AAD-Base Dual | IASD-Base Dual | IVQD-Base Dual | AAD-Inst Dual | IASD-Inst Dual |
|------|-------------|---------------|---------------|-------------|---------------|
| LLaVA-OV-7B | 4.5 | 5.5 | 2.5 | 25.9 | 27.1 |
| InternVL2-8B | 28.5 | 30.1 | 28.4 | 34.0 | 56.5 |
| InternVL2-40B | 43.5 | 45.0 | 42.7 | 67.9 | 75.7 |
| Qwen2.5-VL-7B | 32.2 | 46.1 | 71.1 | 58.5 | 70.4 |
| GPT-4o | 45.6 | 56.1 | 65.2 | 59.3 | 68.0 |

(Dual Accuracy %. While these models all achieve >80% accuracy on the original MMBench, the UPD Base accuracy can drop below 5%.)

### Ablation Study (CoT / Self-Reflection Effects)

| Model | Method | AAD Dual | IASD Dual | IVQD Dual |
|------|------|---------|----------|----------|
| LLaVA-OV-7B | Base | 4.5 | 5.5 | 2.5 |
| LLaVA-OV-7B | CoT | 37.9 | 36.7 | 14.9 |
| LLaVA-OV-7B | Self-Reflection | 27.6 | 35.4 | 31.7 |
| GPT-4o | Base | 45.6 | 56.1 | 65.2 |
| GPT-4o | CoT | 47.7 | 48.4 | 57.2 |
| GPT-4o | Self-Reflection | 55.2 | 57.9 | 57.9 |

### Key Findings

- Almost no correlation exists between MMBench accuracy and UPD performance (the correlation coefficient between UPD Accuracy and Original Standard accuracy is as low as 6.5%), indicating that existing benchmarks completely fail to measure this dimension.
- A massive gap exists between open-source and closed-source models: under the Base setting, most open-source models achieve a UPD accuracy of <10%, whereas GPT-4o reaches 90.2% on IVQD-Base. This gap stems from the fact that closed-source models undergo explicit refusal training.
- CoT and Self-Reflection are effective for models with language-side bottlenecks (e.g., LLaVA-OV), but show limited efficacy for models that already possess strong refusal capabilities.
- Bottleneck Analysis: Even when directly provided with the correct answers, LLaVA-OV and Qwen2VL still fail to correctly choose "None of the above," suggesting that the primary bottleneck lies in the LLM's refusal capability rather than visual comprehension.
- Qwen2.5-VL-7B is the most balanced model in terms of UPD performance among open-source 7B-class models.

## Highlights & Insights

- **Novel and Practical Bottleneck Diagnosis**: Distinguishing whether the bottleneck lies in the visual or the language component by "directly telling the model the answer + observing whether it can still refuse" is simple and effective. This methodology can be transferred to any scenario that requires diagnosing component bottlenecks in multimodal models.
- **Ingenious Discriminative Design of the Three UPD Tasks**: The contrast between AAD and IASD directly reveals whether the model "cannot distinguish extremely fine-grained options" or "fundamentally lacks refusal awareness"—if a model fails even on IASD (completely irrelevant options), it indicates that the model simply does not know how to say "I don't know."
- **Generalizable Dual Accuracy Metric**: This "bidirectional verification" paradigm can be extended to hallucination detection (correct judgment on both faithful and hallucinated questions), safety evaluation (simultaneous assessment under normal usage and adversarial attacks), etc.

## Limitations & Future Work

- This work focuses on evaluation design and does not propose new methodologies tailored for UPD; the fine-tuning experiments are merely preliminary explorations.
- The benchmark is constructed based on MMBench, where the question difficulty is relatively low. Future work requires more challenging foundational problems (at the MMMU level).
- Only single-image, single-turn MCQA is considered; "refusal to answer" in multi-image reasoning, open-ended QA, and multi-turn dialogues remains unaddressed.
- Improving UPD through fine-tuning may come at the expense of performance on general tasks; how to balance the two remains an open question.

## Related Work & Insights

- **vs Wang et al. (2025) LLM Refusal Research**: While they only investigated AAD in LLMs, this work extends the scope to the multimodal domain and introduces two new problem types, IASD and IVQD, providing more fine-grained diagnosis.
- **vs SQuAD 2.0**: SQuAD 2.0 introduced unanswerable questions in reading comprehension. This work extends a similar concept to multimodal MCQA and designs richer types of unsolvable scenarios.
- **vs Hallucination Detection**: UPD focuses on "whether the model knows it cannot answer," whereas hallucination focuses on "the model answered but answered incorrectly." The two are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The taxonomy of three unsolvable problem types and the design of the Dual Accuracy metric are novel, though the core ideas inherit from LLM refusal research.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated 20+ models, incorporating multi-angle experiments including bottleneck analysis, CoT/Self-Reflection, and fine-tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem definitions are clear, charts are intuitive, and findings are summarized progressively.
- Value: ⭐⭐⭐⭐ Unveils a dimension of trustworthiness neglected by existing benchmarks, offering insights for LMM safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ACL 2025\] SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](sciver_evaluating_foundation_models_for_multimodal_scientific_claim_verification.md)

</div>

<!-- RELATED:END -->
