---
title: >-
  [Paper Note] Self Iterative Label Refinement via Robust Unlabeled Learning
description: >-
  [NeurIPS 2025][Medical Imaging][self-refinement] This paper proposes an iterative pipeline that leverages a robust unlabeled-unlabeled (UU) learning framework to refine LLM-generated pseudo-labels…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "self-refinement"
  - "pseudo-labels"
  - "UU learning"
  - "LLM alignment"
  - "weakly supervised classification"
date: 2026-05-08
content_hash: fa773208b6b0ebe5
---

# Self Iterative Label Refinement via Robust Unlabeled Learning

**Conference**: NeurIPS 2025
**arXiv**: [2502.12565](https://arxiv.org/abs/2502.12565)
**Code**: [GitHub](https://github.com/HikaruAsano/self-iterative-label-refinement)
**Area**: Medical Imaging
**Keywords**: self-refinement, pseudo-labels, UU learning, LLM alignment, weakly supervised classification

## TL;DR

This paper proposes an iterative pipeline that leverages a robust unlabeled-unlabeled (UU) learning framework to refine LLM-generated pseudo-labels, surpassing the self-refinement approaches of GPT-4o and DeepSeek-R1 on both classification and generative safety alignment tasks with minimal human annotation.

## Background & Motivation

Large language models (LLMs) demonstrate strong performance across diverse downstream tasks, yet further improving their capabilities typically requires substantial high-quality human feedback (e.g., RLHF). Although RLAIF attempts to reduce annotation costs by substituting model-generated signals for human labels, models exhibit inherent biases when evaluating their own outputs. In particular, self-refinement methods often fail to yield improvements—or even degrade performance—in domains where the model's internal knowledge is insufficient.

The authors identify two key observations: (1) large quantities of unlabeled data are readily available in modern settings, whereas high-quality labeled data remains costly to obtain; and (2) existing LLM self-refinement methods fundamentally rely on the model's internal knowledge to generate feedback and corrections, causing them to fail when such knowledge is lacking. Accordingly, the paper proposes decoupling the refinement process from the LLM's internal knowledge, instead exploiting data-driven features through a UU learning framework to iteratively denoise and improve pseudo-labels.

## Method

### Overall Architecture

The pipeline consists of three steps repeated iteratively: (1) **Initial LLM annotation**: an LLM generates initial pseudo-labels for an unlabeled corpus; (2) **Robust UU learning**: the corpus is partitioned into pseudo-positive and pseudo-negative subsets, and a classifier is trained via robust UU learning; (3) **Re-annotation**: the trained classifier re-labels the entire dataset, updating the pseudo-positive and pseudo-negative sets for the next iteration.

### Key Designs

1. **Unlabeled-Unlabeled (UU) Learning Framework**: The core idea is to train a classifier using two unlabeled datasets with different positive-class prior ratios. Given two unlabeled corpora $\widetilde{\mathcal{C}}_p$ and $\widetilde{\mathcal{C}}_n$ with positive-class proportions $\theta_p$ and $\theta_n$ respectively ($\theta_p > \theta_n$), the UU learning risk is defined as:

    $R_{\text{uu}}(g) = aR_{\tilde{p}}^+(g) - bR_{\tilde{p}}^-(g) - cR_{\tilde{n}}^+(g) + dR_{\tilde{n}}^-(g)$

   where coefficients $a, b, c, d$ are derived from $\pi_+, \theta_p, \theta_n$. When $\theta_p=1, \theta_n=0$, this reduces to standard supervised learning. The two subsets induced by LLM pseudo-labels naturally satisfy the UU learning conditions, as the pseudo-positive subset contains a higher proportion of true positives.

2. **Robust UU Learning**: The original UU risk contains negative terms (e.g., $-bR_{\tilde{p}}^-(g)$) that are prone to overfitting. The robust variant introduces a generalized Leaky ReLU function $f$ to regulate negative risk:

    $R_{\text{ruu}}(g) = f(aR_{\tilde{p}}^+(g) - cR_{\tilde{n}}^+(g)) + f(dR_{\tilde{n}}^-(g) - bR_{\tilde{p}}^-(g))$

   where $f(x) = x$ when $x > 0$ and $f(x) = \lambda x$ when $x < 0$ ($\lambda < 0$). This preserves positive risk values while converting negative risk into positive values, effectively mitigating overfitting.

3. **Iterative Re-labeling and Convergence**: After training classifier $g^{(t)}$ at each round, the entire dataset is re-labeled as $\tilde{y}_i^{(t)} = \text{sign}(g^{(t)}(x_i))$, yielding updated pseudo-positive and pseudo-negative sets. Ideally, as iterations proceed, the true positive proportion in the pseudo-positive set converges toward 1 and that of the pseudo-negative set toward 0, eventually approximating standard supervised learning.

### Loss & Training

- The classifier appends an affine layer over the Transformer's final hidden state to produce a scalar score.
- QLoRA is used for 4-bit quantized fine-tuning with the AdamW optimizer (learning rate $1.0 \times 10^{-4}$, batch size 16, 3 epochs).
- $\lambda$ is fixed at $-0.001$.
- Class prior estimation: the oracle setting uses ground-truth values directly; the few-labeled setting uses only 50 or 100 annotated samples to estimate $\hat{\theta}_p$ and $\hat{\theta}_n$.

## Key Experimental Results

### Main Results (Simple Tasks, RQ1)

| Dataset | Model | Initial LLM Accuracy | Ours (50-labeled) Final | Gain |
|--------|------|--------------|----------------------|------|
| Fake News | Meta-Llama-3-8B | ~0.75 | ~0.90 | +15% |
| Saroco | Llama-3.2-1B | 0.576 | ~0.80 | +22% |
| Safety | Multiple models | ~0.65 | ~0.85 | +20% |

On simple tasks, PIE and CCP completely fail on Saroco and Safety, whereas the proposed method consistently improves performance across all tasks. With only 50 labeled samples, the method converges rapidly to the oracle upper bound.

### Hard Tasks (RQ2)

| Dataset | Method | Final Accuracy | vs. Initial |
|--------|------|-----------|-----------|
| Corona Sentiment | GPT-4o self-refinement | Marginal gain | Weak improvement |
| Corona Sentiment | DeepSeek-R1 | Stagnant | No gain |
| Corona Sentiment | Ours (Oracle) | Steady improvement | Significant gain |
| Green Patent | GPT-4o self-refinement | Performance degradation | Negative gain |
| Green Patent | Ours (Oracle) | Steady improvement | Significant gain |
| Protein Structure | GPT-4o self-refinement | Performance degradation | Negative gain |
| Protein Structure | Ours (Oracle) | Steady improvement | Significant gain |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| PN (standard supervised) | Low accuracy | Completely fails under heavy pseudo-label noise |
| UU (non-robust) | Improved but below Ours | Negative risk terms cause overfitting |
| Ours (Oracle) | Highest | Theoretical upper bound |
| Ours (50-labeled) | Close to Oracle | Requires only 50 labeled samples |

### Key Findings

- The proposed method achieves steady improvement even on hard tasks where GPT-4o and DeepSeek-R1 self-refinement completely fail.
- The few-labeled variant with 50 annotated samples performs nearly on par with the oracle, demonstrating UU learning's robustness to class prior estimation errors.
- In safety alignment (RQ3), using the refined classifier as a reward model for RLAIF substantially improves the safety of generated responses, whereas vanilla RLAIF even underperforms the SFT baseline.

## Highlights & Insights

- Introducing UU learning—a classical weakly supervised technique—into the LLM self-refinement paradigm is both conceptually clean and practically effective.
- The method requires minimal human annotation (50 samples) and does not rely on external tools or knowledge bases.
- By decoupling the refinement process from the LLM's internal knowledge and relying on data-driven features, the approach is well-suited to specialized domains where LLM knowledge is insufficient.
- Demonstrated effectiveness on both classification and generative alignment tasks highlights strong generality.

## Limitations & Future Work

- Performance is limited when initial pseudo-label noise is extremely high and the class priors of the two subsets are nearly identical.
- Factors beyond initial label quality—such as the intrinsic separability of the task—also affect performance; future work may explore modeling instance-level separability.
- Only binary classification scenarios have been validated; extension to multi-class settings remains to be explored.
- Incorporating auxiliary information (e.g., reasoning chains, retrieval context) could further improve classification accuracy.

## Related Work & Insights

- UU learning originates from the weakly supervised learning literature; this paper innovatively combines it with LLM pseudo-label refinement.
- Compared to PIE (progressive confidence-based label selection) and CCP (contrastive semi-supervised learning), the proposed method is more robust to noisy labels generated by LLMs.
- This work may inspire broader adoption of classical weakly supervised and semi-supervised techniques in the context of LLM self-improvement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of UU learning and LLM pseudo-label refinement is novel, though the methodological modifications are relatively incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets spanning simple and hard tasks, covering both classification and generative alignment, with comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and research-question-driven experimental design.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for LLM self-improvement in low-resource and specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2026/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)

</div>

<!-- RELATED:END -->
