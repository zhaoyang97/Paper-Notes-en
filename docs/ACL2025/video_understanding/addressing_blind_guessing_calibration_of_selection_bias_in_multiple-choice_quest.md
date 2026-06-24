---
title: >-
  [Paper Note] Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models
description: >-
  [ACL 2025][Video Understanding][Selection Bias] This study presents the first systematic exploration of selection bias in multiple-choice question answering (MCQA) with Video Language Models (VLMs). By analyzing bias sources through task decomposition, it proposes BOLD, a post-processing calibration technique that reduces bias while simultaneously improving model performance.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Selection Bias"
  - "Video Language Models"
  - "Multiple-Choice Questions"
  - "Debiasing Calibration"
  - "MCQA"
date: 2026-05-08
content_hash: f0c3d63653a577e3
---

# Addressing Blind Guessing: Calibration of Selection Bias in Multiple-Choice Question Answering by Video Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.14248](https://arxiv.org/abs/2410.14248)  
**Code**: [github.com/ologin/BOLD](https://github.com/ologin/BOLD)  
**Area**: Video Understanding  
**Keywords**: Selection Bias, Video Language Models, Multiple-Choice Questions, Debiasing Calibration, MCQA

## TL;DR

This study presents the first systematic exploration of selection bias in multiple-choice question answering (MCQA) with Video Language Models (VLMs). By analyzing bias sources through task decomposition, it proposes BOLD, a post-processing calibration technique that reduces bias while simultaneously improving model performance.

## Background & Motivation

Multiple-Choice Question Answering (MCQA) is widely used to evaluate the reasoning capabilities of Video Language Models (VLMs) due to its standard evaluation and convenience. However, VLMs exhibit severe **selection bias** when answering MCQA; they tend to disproportionately favor certain option positions based on positional patterns learned during training, rather than performing genuine reasoning based on content.

While selection bias has been extensively studied in textual LLMs and partially addressed in image-text VLMs, it remains almost entirely unexplored in **video language models**. Video language models present unique challenges as they rely on complex visual inputs and require spatio-temporal reasoning. Existing debiasing methods (such as option shuffling) require multiple inference passes across all permutations, which is computationally prohibitive.

The core contributions of this work are: (1) providing the first comprehensive analysis of selection bias in VLM-MCQA, and (2) proposing BOLD, a computationally efficient post-processing debiasing method.

## Method

### Overall Architecture

The framework consists of two stages: first, systematically analyzing bias patterns across different VLMs and datasets through 11 dataset modification settings; second, introducing BOLD (Bias Optimisation Leveraging Decomposition), a debiasing method based on task decomposition.

### Key Designs

1. **Systematic Bias Analysis (11 Dataset Modification Settings)**: MCQA is decomposed into three core components: videos, questions, and answer options. Eleven modification configurations are designed:
   
   **Video Modifications**: Ground truth frames (providing only frames where the answer lies), empty frames (black frames).
   
   **Question Modifications**: Paraphrased questions (rephrased 5 times by Llama3, selecting one randomly), empty questions (empty string).
   
   **Answer Modifications**: Option shuffling, correct answer placed at each position, fixed correct answer + shuffling, adding blank options, all options identical, all options correct, empty options.

   Key discovery: The option distribution under accuracy-related settings (ground-truth frames, paraphrasing, shuffling) is almost identical to the default setting, indicating the model relies more on option position than content; accuracy-unrelated settings (identical options, empty settings) clearly expose the bias patterns.

2. **Evaluation Metrics for Bias and Fairness**: Metrics from fairness studies are adapted to VLM-MCQA evaluation:

    - **F1_std**: Standard deviation of F1 scores across options; a high value indicates inconsistent performance across different option positions.
    - **Recall_std**: Standard deviation of recall across options; a high value indicates that correct answers at specific positions are easier to identify.
    - **JS_std**: Standard deviation of the Jensen-Shannon divergence between predictions and ground-truth distributions, detecting distribution inconsistency.

3. **BOLD Debiasing Method**: The core idea is to decompose the observed prediction distribution $P_o$ into a prior bias distribution $P_p$ and a debiased distribution $P_d$:
   
    $$P_o(d_i|T) = \frac{1}{Z_T} P_p(d_i|T) \times P_d(d_i|T)$$
   
   **Bias Estimation**: Three "attacks" are introduced to make the task ill-defined (removing videos, questions, or options). Under these conditions, an ideal model should make uniform selections. This is used to estimate the prior bias:
   
    $$\tilde{P}_p(d_i|T) = \text{softmax}\left(\sum_j P_p(d_i|A_j(T))\right)$$
   
   **Global Prior Calculation**: $K = k \times ||D||$ samples are drawn from the dataset ($k=0.5$ is optimal). Each sample is subjected to the three attacks, and the results are averaged to obtain the global prior.
   
   **Debiased Inference**:
    $$P_d(d_i|T) = \text{softmax}\left(\log P_o(d_i|T) - \log \tilde{P}_p(d_i)\right)$$

4. **Weighted_BOLD Extension**: Learnable weights $w_j$ are introduced to the prior decomposition across the three attack dimensions. Weights are optimized via 5-fold cross-validation using the COBYLA optimizer to accurately estimate the bias direction. The weights are constrained to $0 \leq w_i \leq 1$ or $|w_i| \leq 1$.

### Loss & Training

BOLD is a post-processing method and does not involve model retraining. The weight optimization for Weighted_BOLD uses the COBYLA constrained optimization algorithm, with the objective of minimizing the Recall_std on the validation set.

## Key Experimental Results

### Main Results

| Model | Dataset | Accuracy | F1 Mean |
|------|--------|----------|---------|
| Video-LLaMA | NExT-QA | 40.85 | 40.85 |
| Video-LLaVA | NExT-QA | 49.96 | 49.81 |
| SeViLA | NExT-QA | 63.78 | 63.88 |
| Video-LLaMA | STAR | 36.59 | 31.86 |
| Video-LLaVA | Perception Test | 40.73 | 35.69 |
| SeViLA | Video-MME | 39.85 | 39.82 |

### Ablation Study

**Debiasing Performance of BOLD/Weighted_BOLD ($k=0.5$, Video-LLaVA):**

| Dataset | Method | Acc Gain | F1 Gain | Recall_std↓ | F1_std↓ |
|--------|------|---------|--------|-------------|---------|
| NExT-QA | BOLD | +3.72% | +3.82% | -18.63% | -18.42% |
| NExT-QA | W-BOLD | +4.39% | +4.49% | -22.02% | -19.83% |
| STAR | BOLD | +7.01% | +12.37% | -26.85% | -27.15% |
| STAR | W-BOLD | +7.94% | +13.69% | -30.15% | -28.03% |
| Perc. Test | W-BOLD | +3.02% | +10.58% | -28.26% | -30.53% |

### Key Findings

- **Bias patterns vary across models**: Video-LLaMA strongly prefers $a1$ and ignores subsequent options; Video-LLaVA shows an even stronger preference for $a1$; SeViLA is the most robust, showing a near-uniform 20% distribution on NExT-QA.
- **Debiasing improves performance simultaneously**: Notably, BOLD not only reduces bias metrics but also increases Accuracy and F1 in most cases.
- **Weighted_BOLD outperforms BOLD**: The weighted decomposition prior captures the bias structure more precisely.
- **Robustness of SeViLA**: Due to its argmax function and specialized QA training, SeViLA naturally exhibits less bias, making the debiasing effect on it relatively marginal.
- **$k=0.5$ is the optimal sample ratio**: Estimating bias on 50% of the samples is sufficient to achieve solid performance, avoiding the need for the full dataset.
- **Consistency of bias across datasets**: The same model exhibits similar but not completely identical bias patterns across different datasets.

## Highlights & Insights

- **First Systematic Study**: It fills the gap in studying selection bias in video MCQA, with a comprehensive experimental design utilizing 11 modification configurations.
- **Debiasing as Performance Enhancement**: It challenges the intuition that "reducing bias equals sacrificing accuracy", demonstrating that bias indeed hinders true reasoning.
- **Computationally Efficient**: Unlike conventional shuffling methods which require $n!$ inference permutations, BOLD only requires 3 extra inferences (empty video/question/options) on 50% of the sample.
- **Strong Theoretical Foundation**: It decomposes bias into projections across three orthogonal planes, providing a mathematically clear framework.
- **Use of Unconventional Option Identifiers**: Using a0/a1/a2 instead of A/B/C to minimize token bias demonstrates rigorous experimental design.

## Limitations & Future Work

- It only tests 3 VLMs, leaving out recent large-scale closed-source VLMs (such as GPT-4o, Gemini).
- It assumes mutual independence and no logical overlaps among options, which might not entirely hold in certain real-world benchmarks.
- Weighted_BOLD requires cross-validation to optimize weights, slightly increasing implementation complexity.
- The debiasing effect on SeViLA is limited or marginally lower, indicating limited applicability to models with inherently low bias.
- The relationship between bias, model scale, and training data distribution remains unexplored.
- The mechanisms of bias emergence during the fine-tuning phase could be investigated further.

## Related Work & Insights

- PriDe (Zheng et al., 2024a) isolates bias from unbiased results via shuffling, but at a high computational cost.
- Zhang et al. (2024) analyze bias in image MCQA by cutting off visual input; this study extends it to video and adds multi-plane decomposition.
- Wang et al. (2024a) address bias through data augmentation, but this requires generating more QA pairs.
- The decomposition reasoning in this study can be generalized to fairness analysis of other multimodal tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First to study MCQA bias in video VLMs; the decomposition-based debiasing method offers theoretical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 settings $\times$ 3 models $\times$ 4 datasets; highly exhaustive analysis.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, rich in figures/tables, and mathematically well-formulated.
- Value: ⭐⭐⭐⭐ Raises important concerns regarding the reliability of MCQA evaluation, with highly practical debiasing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](../../CVPR2025/video_understanding/question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](../../CVPR2025/video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[ICLR 2026\] Video-LevelGauge: Investigating Contextual Positional Bias in Video Language Models](../../ICLR2026/video_understanding/video-levelgauge_investigating_contextual_positional_bias_in_video_language_mode.md)

</div>

<!-- RELATED:END -->
