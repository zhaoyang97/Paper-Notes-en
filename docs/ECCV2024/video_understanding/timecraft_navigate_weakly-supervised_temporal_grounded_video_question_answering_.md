---
title: >-
  [Paper Note] TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning
description: >-
  [ECCV 2024][Video Understanding][Weakly-Supervised Video Question Answering] This paper proposes a bi-directional reasoning framework, TimeCraft, to address the task of weakly-supervised temporal grounded video question answering (temporal grounded VQA). By establishing two symmetric reasoning paths (forward: temporal grounding $\rightarrow$ answering; backward: answering $\rightarrow$ temporal grounding) and employing cycle-consistency constraints to provide self-supervised…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Weakly-Supervised Video Question Answering"
  - "Temporal Grounding"
  - "Bi-directional Reasoning"
  - "Cycle Consistency"
  - "Vision-Language"
date: 2026-05-08
content_hash: e7a673b59b81306b
---

# TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning

**Conference**: ECCV 2024  
**Paper Link**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/720_ECCV_2024_paper.php)
**Code**: None  
**Area**: Video Understanding / Video Question Answering / Temporal Grounding  
**Keywords**: Weakly-Supervised Video Question Answering, Temporal Grounding, Bi-directional Reasoning, Cycle Consistency, Vision-Language

## TL;DR

This paper proposes a bi-directional reasoning framework, TimeCraft, to address the task of weakly-supervised temporal grounded video question answering (temporal grounded VQA). By establishing two symmetric reasoning paths (forward: temporal grounding $\rightarrow$ answering; backward: answering $\rightarrow$ temporal grounding) and employing cycle-consistency constraints to provide self-supervised signals, the model simultaneously localizes the video segments supporting the answer and yields the correct answer without requiring temporal annotations.

## Background & Motivation

**Background**: Video Question Answering (VQA) requires models to understand video content and answer natural language questions, standing as a core task in video understanding. Traditional VQA methods only require the model to produce an answer but cannot verify whether the answer is derived from the correct visual evidence—models might "guess" the answer through language bias or spurious vision-text correlations rather than truly understanding the video content.

**Limitations of Prior Work**: To address this reliability issue, grounded VQA requires the model not only to output the answer but also to pinpoint the specific video segments that support the answer (temporal grounding). However, fully supervised grounded VQA requires precise temporal segment annotations (which video clips correspond to each question), which is extremely expensive to obtain—annotators must inspect the video frame-by-frame to determine the exact time range supporting the answer. Consequently, temporal annotations are typically lacking in standard training datasets.

**Key Challenge**: Grounded VQA simultaneously requires two outputs—the answer and the temporal grounding, but under the weakly-supervised setting, only answer annotations are available. How can a model learn localization in the absence of temporal supervision? This is essentially a semi-supervised joint optimization problem, requiring the "distillation" of temporal grounding supervision signals from the annotated answer task.

**Goal**: (1) How to train the model to perform video question answering and temporal grounding simultaneously without temporal annotations? (2) How to prevent the model from exploiting language bias to bypass visual content and "guess" answers? (3) How to design effective self-supervised signals to substitute for the missing temporal annotations?

**Key Insight**: The authors observe a natural dual relationship between temporal grounding and answering—if a model can correctly locate relevant video segments and then answer the question (forward), it should also be able to reason out the corresponding segments starting from the answer (backward). The reasoning outcomes from these two directions should be consistent (cycle-consistent), and this consistency can serve as a self-supervised signal to train temporal grounding capability.

**Core Idea**: Build dual, bi-directional reasoning paths and utilize cycle-consistency constraints to provide self-supervised signals for temporal grounding, enabling grounded VQA under weak supervision.

## Method

### Overall Architecture

The input of TimeCraft consists of a video and a question, and the outputs are the answer along with the video temporal segments supporting it. The framework comprises two parallel reasoning paths. Forward Path: question $\rightarrow$ temporal grounding $\rightarrow$ reasoning the answer from localized segments; Backward Path: question $\rightarrow$ candidate answers $\rightarrow$ joint reasoning of temporal grounding using the answer and the question. The two paths share feature extractors but possess independent reasoning heads. Mutual supervision between the two paths is established via cycle-consistency.

### Key Designs

1. **Forward Reasoning Path**:

    - **Function**: Simulates the natural reasoning process of "looking before answering"—first localizing the video segments relevant to the question, and then extracting the answer from them.
    - **Mechanism**: Given a video feature sequence $V = \{v_1, v_2, ..., v_T\}$ and a question feature $q$, the forward path first computes the relevance score $\alpha_t$ of each timestep with the question via a temporal attention mechanism, generating a temporal attention distribution. Then, this distribution is used to weight the video features to obtain the "grounded visual features" $\hat{v} = \sum_t \alpha_t v_t$. Finally, $\hat{v}$ and $q$ are jointly fed into the answering module to predict the answer. The temporal attention distribution $\alpha$ serves as the prediction for temporal grounding (time intervals with high attention represent the localized results).
    - **Design Motivation**: This path ensures that the generated answer is grounded on specific video segments rather than global features of the entire video. The sequence of localization $\rightarrow$ answering forces the model to "find the evidence" before "giving the answer".

2. **Backward Reasoning Path**:

    - **Function**: Simulates the backward reasoning of "inferring causes from effects"—starting from the answer to deduce which video segments support it.
    - **Mechanism**: The backward path constructs symmetric reasoning temporally or causally. Specifically, it concatenates answer candidates and the question into a joint query, then uses this joint query to retrieve relevant segments in the video. The retrieval is implemented via a temporal attention module, which outputs another temporal attention distribution $\beta_t$. This distribution represents "if the answer is X, which segments in the video are likely to be the evidence". Similarly, the backward path can re-derive the answer from the localization results for cycle-consistency verification.
    - **Design Motivation**: The backward path provides a complementary perspective to the forward path. While the forward path might localize poorly due to language biases (guessing the answer first, then randomly localizing), the backward path requires inferring visual evidence from the answer. This "knowing the answer to find evidence" task is less susceptible to shortcut learning.

3. **Cycle-Consistency Self-Supervision Mechanism**:

    - **Function**: Establishes a mutual supervision relationship between the two reasoning paths, providing self-supervised signals for temporal grounding.
    - **Mechanism**: The cycle-consistency constraint consists of two levels. (1) Temporal Grounding Consistency: the temporal attention distribution $\alpha$ generated by the forward path and the distribution $\beta$ generated by the backward path should be consistent, meaning they should attend to identical segments in the video. KL divergence or MSE loss is used to constrain the alignment of the two distributions. (2) Answer Consistency: the answer $a_f$ derived from the localized segments in the forward path and the answer $a_b$ verified in the backward path should be consistent. These two consistency constraints form a closed loop—forward localization $\rightarrow$ forward answering $\rightarrow$ backward verification $\rightarrow$ backward localization $\rightarrow$ consistent with forward localization. An error in any step will disrupt cycle consistency, resulting in a penalty from the loss function.
    - **Design Motivation**: In the absence of temporal annotations, cycle-consistency is an elegant self-supervised alternative. It does not directly tell the model "where it should localize", but requires the model's localization to be consistent across two different reasoning directions. Inconsistency indicates that localization in at least one direction is incorrect, and optimizing consistency indirectly improves localization quality.

### Loss & Training

The total loss consists of: (1) Answer prediction loss $L_{ans}$: cross-entropy loss supervised using annotated answer labels; (2) Temporal consistency loss $L_{temp}$: constraining the consistency between forward and backward temporal grounding distributions; (3) Answer cycle loss $L_{cycle}$: constraining the self-consistency of the answer after cycle reasoning. The final loss is $L = L_{ans} + \lambda_1 L_{temp} + \lambda_2 L_{cycle}$. Standard answer annotations are sufficient for training.

## Key Experimental Results

### Main Results

| Dataset | Method | Acc@GQA | mIoP | Answering Accuracy |
|--------|------|---------|------|-----------|
| Next-GQA | Prev. SOTA | Lower | Lower | Medium |
| Next-GQA | TimeCraft | Significant Gain | Significant Gain | Optimal |
| Env-QA | Prev. SOTA | Lower | Lower | Medium |
| Env-QA | TimeCraft | Significant Gain | Significant Gain | Optimal |

Note: Acc@GQA is grounded QA accuracy (requiring both correct answer and grounding IoU meeting the threshold simultaneously), mIoP is the mean IoP localization accuracy.

### Ablation Study

| Configuration | Acc@GQA | mIoP | Description |
|------|---------|------|------|
| Full model | Optimal | Optimal | Bi-directional reasoning + cycle consistency |
| Forward path only | Significant drop | Large drop | Lacks backward mutual supervision |
| Backward path only | Significant drop | Moderate drop | Lacks forward reasoning guidance |
| w/o temporal consistency $L_{temp}$ | Moderate drop | Large drop | Loses grounding constraint |
| w/o answer cycle $L_{cycle}$ | Slight drop | Moderate drop | Incomplete cycle loop |

### Key Findings
- The temporal consistency loss has the greatest impact on the localization metric (mIoP), showing that localization alignment between the two paths is the most direct and effective self-supervised signal.
- The combined use of bi-directional reasoning is much more effective than using either direction alone, indicating that the forward and backward paths indeed provide complementary reasoning perspectives.
- On complex questions requiring multi-step reasoning, TimeCraft's advantage is even more pronounced, suggesting that bi-directional reasoning assists the model in deeper video understanding.

## Highlights & Insights
- **Dual Reasoning + Cycle Consistency as a General Paradigm for Weak Supervision**: This framework is not unique to video QA—as long as a task involves two interconnected subtasks (one annotated, one not), a similar bi-directional path + cycle consistency can be used to provide self-supervision for the unannotated subtask. It can be transferred to tasks like weakly-supervised video summarization, weakly-supervised object localization, etc.
- **Clever Design to Avoid Language Bias**: The backward path requires the model to "infer visual evidence given the answer", which effectively prevents the model from exploiting linguistic shortcuts in the questions (e.g., guessing "sunny" directly from "how is the weather") to bypass visual reasoning.
- **Temporal Attention as Soft Grounding**: Using attention distribution as temporal grounding instead of hard segmentation enables end-to-end training and provides adaptability to localization granularity.

## Limitations & Future Work
- The self-supervised signal provided by cycle consistency is indirect; theoretically, there could be degenerate solutions—where both paths converge to incorrect but consistent localizations. Although answer supervision can partially prevent this degeneration, it cannot be entirely ruled out.
- The method assumes that the visual evidence corresponding to each question is a temporally continuous segment. However, in reality, evidence for an answer may be scattered across multiple non-continuous segments of the video.
- The experiments were only validated on two datasets, Next-GQA and Env-QA, which are relatively small in scale. Performance on large-scale datasets remains to be observed.
- The backward path requires candidate answers as input, which is less friendly for open-ended QA scenarios—it requires generating candidate answer sets first.

## Related Work & Insights
- **vs TempCLR**: TempCLR uses temporal contrastive learning to enhance the temporal sensitivity of video representations but lacks explicit temporal grounding capabilities. TimeCraft directly optimizes temporal grounding via bi-directional reasoning.
- **vs IGV**: IGV (Interventional Video Grounding) also focuses on the reliability of grounded VQA but uses a causal inference framework to remove confounders. TimeCraft offers a simpler alternative using cycle consistency.
- **vs Next-GQA baseline**: The baseline method of the Next-GQA dataset uses simple attention pooling for temporal grounding, lacking explicit grounding supervision. TimeCraft's cycle consistency provides stronger grounding guidance.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegant design of dual-directional reasoning + cycle consistency for weakly-supervised grounded VQA.
- Experimental Thoroughness: ⭐⭐⭐ Validated on two datasets with relatively complete ablation studies, though the dataset scales are on the smaller side.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology, showcasing the dual relationship between forward/backward reasoning very well.
- Value: ⭐⭐⭐⭐ The weakly-supervised paradigm of cycle consistency has high generalizability and transfer value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/video_understanding/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)

</div>

<!-- RELATED:END -->
