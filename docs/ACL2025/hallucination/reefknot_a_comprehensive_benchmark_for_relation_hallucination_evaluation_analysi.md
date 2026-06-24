---
title: >-
  [Paper Note] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models
description: >-
  [Hallucination Detection] This paper proposes Reefknot, the first comprehensive benchmark to systematically evaluate relation-level hallucination in Multimodal Large Language Models (MLLMs), consisting of over 20k samples across three tasks. Based on confidence-entropy detection, a Detect-then-Calibrate mitigation strategy is proposed, which reduces the average hallucination rate by 9.75%.
tags:
  - "Hallucination Detection"
date: 2026-05-08
content_hash: bf59e8557ae1b2ad
---

# ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models

- **Conference**: ACL 2025
- **arXiv**: [2408.09429](https://arxiv.org/abs/2408.09429)
- **Code**: [JackChen-seu/Reefknot](https://github.com/JackChen-seu/Reefknot)
- **Area**: Multimodal VLM
- **Keywords**: Relation Hallucination, Multimodal Large Language Models, Hallucination Benchmark, Confidence Calibration, Scene Graph

## TL;DR

This paper proposes Reefknot, the first comprehensive benchmark to systematically evaluate relation-level hallucination in Multimodal Large Language Models (MLLMs), consisting of over 20k samples across three tasks. Based on confidence-entropy detection, a Detect-then-Calibrate mitigation strategy is proposed, which reduces the average hallucination rate by 9.75%.

## Background & Motivation

The hallucination problem in Multimodal Large Language Models (MLLMs) can be categorized into three levels of granularity: object-level, attribute-level, and relation-level. Existing works (such as POPE, MME, AMBER, etc.) primarily focus on whether an object exists or whether its attributes are correct, whereas **relation-level hallucination** (i.e., the model's incorrect description of the relationship between two or more objects in an image) remains heavily understudied. The unique challenges of relation hallucination include:

1. It involves at least two entities, making the reasoning complexity significantly higher than single-entity problems.
2. Existing benchmarks only provide simple Yes/No discriminative evaluation, lacking comprehensive multi-task assessment.
3. Dataset construction generally relies on post-processing or automatic annotation, which introduces systematic biases.
4. There are almost no mitigation methods specifically targeting relation hallucination.

The authors compared POPE (object-level) and Reefknot (relation-level) under the same settings and found that **relation hallucination is much more severe than object hallucination**, highlighting the urgent need for systematic research.

## Method

### Overall Architecture

The construction and evaluation pipeline of Reefknot consists of six stages:

1. **Triplet Identification**: Extract visual relation triplets (subject, relation, object) for 11,084 images from the Visual Genome scene graph dataset.
2. **Triplet Filtering**: Remove redundant, incorrect, or noisy descriptions.
3. **Semantic Triplet Extraction**: Standardize the subject-predicate-object structure.
4. **Relation Classification**: Categorize relations into perceptive (e.g., spatial prepositions like on/in/behind) and cognitive (e.g., action phrases like eating/watching), containing 56 perceptive relations and 152 cognitive relations.
5. **Question Construction**: Generate three types of tasks—Yes/No discrimination, Multiple Choice Question (MCQ), and open-ended Visual Question Answering (VQA).
6. **Multi-round Expert Verification**: At least 3 rounds of review by 4 domain experts to filter out meaningless questions.

The final dataset contains 21,880 questions, including 13,260 perceptive and 8,600 cognitive questions.

### Key Design 1: Unified Evaluation Metric $R_{score}$

The authors propose a unified metric, $R_{score}$, to comprehensively evaluate model performance across the three tasks:

$$R_{score} = \text{Avg}\left[\sum_{i=1}^{3}(1 - Halr_i)\right]$$

where $Halr_i$ is the hallucination rate of the $i$-th task. For discriminative tasks (Y/N and MCQ), $Halr$ is the complement of accuracy; for the generative VQA task, bidirectional entailment matching using the DeBERTa model is employed to determine answer correctness. This metric has the advantage of simultaneously considering both discriminative and generative dimensions of capability.

### Key Design 2: Detect-then-Calibrate Mitigation Method

Core discovery: When a model generates relation hallucinations, the probability of its responses drops significantly (around 95% under normal conditions, but only around 70% during hallucinations). Based on this observation, a two-step strategy is proposed:

**Detection Phase**: Calculate the information entropy of the generated tokens:

$$E(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)$$

Set an entropy threshold $\gamma$; when $E(r_0) \geq \gamma$, it is flagged as a potential hallucination.

**Calibration Phase**: Utilize intermediate hidden states to calibrate the final output:

$$r = \begin{cases} \arg\max \log \frac{(1+\alpha) \cdot \text{softmax}(\phi(h_t^n))}{\alpha \cdot \text{softmax}(\phi(h_t^{n-\lambda}))} & \text{if } E_t > \gamma \\ \arg\max(\text{softmax}(\phi(h_t^n))) & \text{otherwise} \end{cases}$$

where $\lambda$ controls the depth of the intermediate layer, and $\alpha$ adjusts the calibration strength. The key idea is to **only calibrate answers with high uncertainty** to avoid miscorrecting correct answers. In the experiments, $\lambda=2$, $\alpha=0.1$, and $\gamma=0.9$ are used.

### Key Design 3: Layer-wide Probability Analysis

The authors project the hidden state $\mathcal{H}_j$ of each layer through the language model head $\phi(\cdot)$ to obtain the next-token probability distribution for each layer:

$$\mathbb{P}(\mathcal{H}_j | \mathcal{H}_{j-1}) = \text{softmax}(\phi(\mathcal{H}_{j-1}))$$

By visualizing 32-layer MiniGPT4-v2 and 40-layer LLaVA-13B, they discover that: the shallow layers (0-20) show unchanged probabilities (information aggregation stage), while the deep layers begin to converge on answers—**hallucinations occur in the final few layers**, where the model suddenly becomes uncertain in the final decoder blocks.

## Key Experimental Results

### Table 1: Hallucination rates (%, ↓ better) and integrated scores (↑ better) of mainstream MLLMs on Reefknot

| Model | Params | Perceptive-Y/N | Perceptive-MCQ | Perceptive-VQA | Cognitive-Y/N | Cognitive-MCQ | Cognitive-VQA | $R_{score}$ |
|------|--------|----------|----------|----------|----------|----------|----------|-------------|
| GPT-4o | - | 32.56 | 40.93 | 42.70 | 26.27 | 11.53 | 48.78 | **68.32** |
| MiniCPM | 7B | 31.93 | 48.65 | 47.63 | 27.65 | 16.71 | 45.96 | 65.73 |
| Yi-VL | 34B | 32.79 | 44.19 | 57.67 | 33.75 | 14.85 | 52.72 | 62.61 |
| GLM4V | 9B | 34.09 | 50.47 | 58.09 | 27.08 | 16.87 | 56.47 | 62.03 |
| Phi-3 | 4.2B | 39.88 | 57.07 | 50.98 | 33.97 | 21.35 | 49.45 | 60.30 |
| LLaVA | 13B | 40.70 | 59.35 | 48.93 | 34.19 | 29.19 | 54.45 | 57.47 |
| CogVLM | 19B | 37.23 | 47.95 | 70.14 | 29.89 | 18.54 | 66.18 | 57.10 |
| MiniGPT4-v2 | 7B | 46.70 | 78.00 | 61.30 | 43.73 | 68.50 | 65.88 | 39.88 |

### Table 2: Comparison of Detect-then-Calibrate with other mitigation methods (LLaVA-13B, hallucination rate %↓)

| Method | Reefknot | MMRel | R-bench |
|------|----------|-------|---------|
| Baseline | 37.06 | 40.43 | 29.52 |
| + VCD | 38.32 | 41.96 | 22.05 |
| + DoLa | 36.96 | 39.68 | 23.52 |
| + OPERA | 35.73 | 39.22 | 26.73 |
| + **Detect-then-Calibrate (Ours)** | **34.50** | **21.73** | **22.02** |

The proposed method achieves the best results across all three datasets, dropping the hallucination rate by 19.7 percentage points compared to the baseline on MMRel.

## Key Findings

1. **Perceptive Hallucination > Cognitive Hallucination**: Across all models, perceptive relation hallucinations are consistently about 10% higher than cognitive ones, with the gap reaching up to 30.16% in extreme cases (LLaVA-13B MCQ). This is hypothesized to be because action descriptions (cognitive) are richer in pre-training data, while spatial relations (perceptive) are often neglected in annotations.
2. **Models favor answering "Yes" in Y/N tasks**: The number of No $\rightarrow$ Yes misclassifications is twice that of Yes $\rightarrow$ No, exposing an imbalanced data distribution during training.
3. **Models favor selecting "D" in MCQ tasks**: This is likely caused by data distribution bias in the instruction-tuning phase.
4. **Hallucinations occur in deep layers**: Shallow layers aggregate information, while deep layers cause relation hallucinations due to the inclusion of large amounts of parameterized knowledge.
5. **Confidence is a reliable signal of hallucination**: The average probability stands at only around 70% during hallucinations, whereas it reaches up to 95% during non-hallucinatory generation.

## Highlights & Insights

- **Filling the Gap**: This is the first comprehensive benchmark focusing on relation-level hallucination, with three complementary task formats (discriminative + generative).
- **Data Quality**: Built on the original scene graphs of Visual Genome, avoiding post-processing or synthesis, and verified through multi-round expert review to ensure quality.
- **In-depth Mechanical Analysis**: Beyond evaluation, this work explains the location and mechanism of hallucination generation through layer-wise probability changes.
- **Concise and Effective Mitigation**: Detect-then-Calibrate is a training-free inference-time method that requires no extra training, and its selective calibration avoids interfering with correct answers.
- The classification perspective of perceptive vs. cognitive relations is inspiring, suggesting that the bottlenecks of MLLMs in reasoning vary across different relationship types.

## Limitations & Future Work

1. The mitigation method is only verified on discriminative tasks (Y/N). The quantitative evaluation and mitigation of relation hallucinations in open-ended VQA remain unresolved.
2. Single data source (Visual Genome), which may not sufficiently cover more complex, real-world scenario relations.
3. Determinstic decoding with a temperature of 0 was used during model evaluation, leaving the performance differences under random sampling unexplored.
4. The sensitivity of hyper-parameters ($\gamma$, $\alpha$, $\lambda$) across different models/datasets has not been fully discussed.
5. The calibration strategy relies on accessing the internal probabilities of models, making it inapplicable to black-box APIs (e.g., GPT-4o).

## Related Work & Insights

- **POPE** and **AMBER** focus on object-level hallucination. Reefknot shifts the granularity to the relation level, and the three form a complementary evaluation framework.
- **DoLa** first proposed utilizing contrastive decoding between layers to improve factuality. Reefknot's innovation lies in **only calibrating highly uncertain answers**, avoiding incorrect corrections of already correct answers.
- **VCD** mitigates object hallucination through visual contrastive decoding, but its effectiveness on the relation level is inferior to Detect-then-Calibrate.
- The concept of semantic entropy (Kuhn et al.) aligns with the information entropy detection direction of this work, but this work limits the analysis to the candidate answer vocabulary, making it more practical.
- This work inspires future search in directions such as: (1) finer-grained scene graph annotation, (2) layer-specific calibration strategies, and (3) explicit introduction of relation reasoning supervision during the training phase.

## Rating

⭐⭐⭐⭐ — The dataset is systematically constructed and sizable, the evaluation dimensions are comprehensive, and the mechanism analysis is profound. The mitigation method is simple yet effective, but its applicability is limited (discriminative only), with extension to open-ended scenarios serving as the main drawback.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](../../ACL2026/hallucination/halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](../../CVPR2025/hallucination/ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[ACL 2025\] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models](cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)

</div>

<!-- RELATED:END -->
