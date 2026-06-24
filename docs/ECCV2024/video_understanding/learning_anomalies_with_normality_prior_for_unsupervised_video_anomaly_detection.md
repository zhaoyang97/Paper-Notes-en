---
title: >-
  [Paper Note] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection
description: >-
  [ECCV 2024][Video Understanding][Unsupervised Video Anomaly Detection] This paper proposes LANP, an unsupervised video anomaly detection method based on a normality prior (where the beginning and ending segments of a video are typically normal events). The normalness of unlabeled segments is estimated through normality propagation, combined with a loss re-weighting strategy to mitigate the negative influence of mispropagated labels, achieving superior performance on ShanghaiT…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Unsupervised Video Anomaly Detection"
  - "Normality Prior"
  - "Label Propagation"
  - "Loss Re-weighting"
  - "Pseudo Label"
date: 2026-05-08
content_hash: 102e18afe7dd0189
---

# Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection

**Conference**: ECCV 2024  
**Code**: [https://github.com/shyern/LANP-UVAD](https://github.com/shyern/LANP-UVAD)  
**Area**: Video Understanding / Anomaly Detection  
**Keywords**: Unsupervised Video Anomaly Detection, Normality Prior, Label Propagation, Loss Re-weighting, Pseudo Label

## TL;DR

This paper proposes LANP, an unsupervised video anomaly detection method based on a normality prior (where the beginning and ending segments of a video are typically normal events). The normalness of unlabeled segments is estimated through normality propagation, combined with a loss re-weighting strategy to mitigate the negative influence of mispropagated labels, achieving superior performance on ShanghaiTech and UCF-Crime compared to existing methods.

## Background & Motivation

**Background**: Unsupervised Video Anomaly Detection (UVAD) aims to detect anomalous events in videos without using any annotations. Existing methods are mostly purely data-driven, identifying patterns that deviate from the normal distribution through clustering, autoencoder reconstruction errors, or contrastive learning. These methods rely heavily on learned feature representations and data distributions to distinguish between normal and anomalous events.

**Limitations of Prior Work**: Purely data-driven methods depend excessively on the quality of feature representations and data distribution. Consequently, they can only detect "salient anomalies" that differ significantly from normal events (e.g., violent fights), easily missing "subtle anomalies" that exhibit indistinct differences from normal events (e.g., slow tailgating). More fundamentally, in a completely unsupervised setting, the model lacks a prior knowledge anchor for "what is normal," resulting in extremely weak learning signals.

**Key Challenge**: Unsupervised settings imply that no annotations are available, yet constructing normal/anomalous boundaries automatically and solely from data is highly challenging. Existing methods overlook the data-irrelevant prior knowledge inherent in video data itself—for instance, the beginning and ending of a surveillance video are highly likely to be normal.

**Goal**: (1) How to provide a reliable normality anchor for the model under unannotated conditions? (2) How to extend limited prior knowledge to all segments of the entire video? (3) How to handle the inevitable label errors during the propagation process?

**Key Insight**: The authors propose a simple yet effective prior assumption—the beginning and ending segments of a video are highly likely to be normal events. This assumption is based on the statistical properties of surveillance videos (where anomalous events occur sparsely in the time series). Leveraging this prior, normality is propagated through inter-segment relationships, spreading the "known normal" information to unlabeled segments.

**Core Idea**: Utilizing the prior knowledge that "the beginning and ending of a video are normal events," normality estimation is extended to the entire video via graph propagation. A loss re-weighting strategy is then employed to mitigate the impact of propagation errors on anomaly detection learning.

## Method

### Overall Architecture

The LANP framework consists of three core components. First, a normality prior is introduced, treating the start and end segments of a video as normal samples that serve as seeds for propagation. Second, a video segment relationship graph is constructed to spread the seed labels to all unlabeled segments through normality propagation, estimating the "normality score" for each segment. Finally, based on the propagated pseudo-labels, an anomaly detection model is trained using the Multiple Instance Learning (MIL) framework, employing a loss re-weighting strategy to reduce the negative impact of propagated erroneous labels on training.

### Key Designs

1. **Normality Prior**:

    - Function: Provides an initial anchor for completely unsupervised anomaly detection.
    - Mechanism: Based on the statistical patterns of surveillance videos, it is assumed that the first $k$ and last $k$ segments of each video are normal events. Although this assumption is not universally true, it holds statistically over a large collection of videos—anomalous events (such as fights or thefts) usually occur in the middle of a video rather than at the boundaries. These segments are labeled as "normal" (with a normality score of 1) and the remaining segments are labeled as "unknown" (with a score of 0), forming the initial normality distribution $\mathbf{y}^0$.
    - Design Motivation: Compared to pure data-driven methods that learn signals from scratch, prior knowledge provides a strong initialization. While simple and coarse, this prior is statistically sufficient to improve the learning starting point, and subsequent propagation and re-weighting mechanisms can correct occasional errors.

2. **Normality Propagation**:

    - Function: Propagates normality information from the start and end segments to all segments of the video.
    - Mechanism: First, feature vectors for each segment are obtained using a pre-trained video feature extractor. A similarity graph between segments is then constructed by calculating the cosine similarity of features between all segment pairs, yielding an affinity matrix $\mathbf{W}$. Using the iterative label propagation algorithm: $\mathbf{y}^{(t+1)} = \alpha \mathbf{W} \mathbf{y}^{(t)} + (1-\alpha) \mathbf{y}^{0}$, normality scores are propagated and diffused along feature-similar segments. Upon convergence, normality estimation scores for all segments are obtained. Segments with high normality scores are highly likely to be normal, while those with low scores are potential anomaly candidates.
    - Design Motivation: Based on the assumption that "similar segments share the same label," feature-similar segments should possess close normality values. Label propagation is a well-established technique in semi-supervised learning and naturally fits this scenario. The parameter $\alpha$ controls the diffusion strength of propagation versus the retention of the initial prior.

3. **Loss Re-weighting**:

    - Function: Lowers the negative impact of potentially incorrect labels in the propagated pseudo-labels on training.
    - Mechanism: The propagated normality scores are not fully reliable—some real anomalous segments might be mispropagated as "normal" due to feature similarity with normal segments. Therefore, the training loss of each segment is re-weighted: samples with high-confidence normality scores (close to 0 or 1) are assigned higher weights, while low-confidence samples (close to 0.5) receive lower weights. Specifically, the weight is computed as the distance to 0 or 1: $w_i = |y_i - 0.5| \cdot 2$. Furthermore, complementary positive/negative sample losses are designed, allowing the model to learn simultaneously from reliable normal and anomalous judgments.
    - Design Motivation: Erroneous labels are inevitable in propagation; rather than rigidly trusting all propagation results, it is better to flexibly adjust contributions based on confidence. This is conceptually similar to sample weighting strategies in noisy label learning, preventing the model from overfitting in uncertain regions.

### Loss & Training

The Multiple Instance Learning (MIL) framework is adopted. Segments with normality propagation scores higher than a threshold are grouped into a positive bag, and those below the threshold into a negative bag. The MIL loss is formulated as a weighted classification loss: $\mathcal{L} = \sum_i w_i \cdot \ell(f(x_i), \hat{y}_i)$, where $w_i$ denotes the weight based on propagation confidence, and $\hat{y}_i$ is the propagated pseudo-label. A temporal smoothing regularization $\mathcal{L}_{smooth}$ is added to encourage smooth transitions in anomaly scores between adjacent segments.

## Key Experimental Results

### Main Results

| Dataset | Metric | LANP | GCL (CVPR'22) | EVAL (ICCV'23) | Gain |
|--------|------|------|---------------|----------------|------|
| ShanghaiTech | AUC(%) | 79.13 | 71.04 | 74.80 | +4.33 |
| UCF-Crime | AUC(%) | 75.56 | 71.04 | 72.16 | +3.40 |

### Ablation Study

| Configuration | ShanghaiTech AUC | Description |
|------|-----------------|------|
| Full model (LANP) | 79.13 | Full method |
| w/o Normality Prior | 71.82 | Performance degrades to purely data-driven without the prior, dropping by 7.3% |
| w/o Normality Propagation | 74.35 | Using only start/end labels without propagation, dropping by 4.8% |
| w/o Loss Re-weighting | 76.91 | No weight adjustment, dropping by 2.2% |
| Fixed propagation steps = 1 | 76.02 | Insufficient propagation, dropping by 3.1% |

### Key Findings
- The normality prior contributes the most (dropping by 7.3% AUC when removed), indicating that data-irrelevant prior knowledge is crucial for unsupervised anomaly detection.
- Propagation steps significantly affect performance; insufficient propagation occurs with too few steps, while noise accumulates and leads to performance degradation with too many steps.
- Although loss re-weighting has a smaller standalone contribution (+2.2%), it behaves highly effectively when combined with propagation, demonstrating its role as a key complement to propagation.
- The improvement is particularly pronounced in subtle anomaly scenarios (e.g., minor physical altercations), confirming the method's capability to detect non-salient anomalies.

## Highlights & Insights
- **The normality prior assumption is simple yet highly effective**—The common-sense prior that "the beginning and ending of videos are highly likely to be normal" provides a highly valuable anchor in a completely unsupervised setting. This approach inspires looking for similar domain priors in other unsupervised tasks.
- **The application of label propagation to video anomaly detection is ingenious**—Creatively employing a classic semi-supervised learning algorithm (label propagation) in an unsupervised scenario, by "generating" a sparse set of initial labels through prior assumptions and then propagating them, is an effective cross-paradigm combination.
- The complementary design of the loss re-weighting mechanism and label propagation is highly educational. This "propagate first, denoise later" two-stage strategy serves as a good reference for any scenarios involving pseudo-label quality (e.g., semi-supervised learning, self-training).

## Limitations & Future Work
- The normality prior assumption does not hold in certain scenarios—such as when a video begins directly with an anomalous event (e.g., recordings of emergency incidents), where the prior introduces errors.
- Normality propagation relies on the quality of feature similarity between segments; if pre-trained features fail to distinguish subtle differences between normal and anomalous events, the propagation effect is limited.
- The method is sensitive to hyperparameters: the header/tail segment count $k$, propagation diffusion factor $\alpha$, and re-weighting threshold require careful tuning.
- Integration with large foundation models is yet to be explored—models like CLIP may provide superior semantic-level anomaly priors, presenting potential when combined with the propagation strategy of this work.

## Related Work & Insights
- **vs RTFM**: RTFM assumes video-level anomaly labels (weakly supervised), while this work is completely unsupervised. Approaching weakly supervised performance under unannotated conditions represents a major breakthrough.
- **vs GCL**: GCL also focuses on unsupervised anomaly detection but distinguishes normal/anomalous features in a purely data-driven manner via graph contrastive learning; LANP introduces prior knowledge to provide stronger signals, offering high complementarity.
- **vs EVAL**: EVAL handles uncertainty through evidential learning, whereas LANP estimates normality through propagation. The uncertainty modeling philosophies of both approaches could potentially be fused.

## Rating
- Novelty: ⭐⭐⭐⭐ The normality prior assumption is simple and novel, whilst creatively applying label propagation to an unsupervised scenario.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two standard benchmarks with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the rationality of the prior assumption is thoroughly analyzed.
- Value: ⭐⭐⭐⭐ Introduces a brand-new "prior + propagation" paradigm for unsupervised anomaly detection, providing broad general inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[ECCV 2024\] Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)
- [\[AAAI 2026\] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment](../../AAAI2026/video_understanding/learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)
- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)

</div>

<!-- RELATED:END -->
