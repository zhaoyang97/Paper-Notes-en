---
title: >-
  [Paper Note] An Information Theoretic Evaluation Metric for Strong Unlearning
description: >-
  [AAAI 2026][AI Safety][Machine Unlearning] This work reveals a fundamental flaw in existing black-box unlearning evaluation metrics (such as MIA and JSD)—modifying only the final layer can satisfy all black-box metrics while the intermediate layers completely retain information about the forgotten data. To address this, the authors propose IDI, a white-box metric that quantifies unlearning effectiveness by estimating the difference in mutual information between each layer and…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Machine Unlearning"
  - "Mutual Information"
  - "White-box Evaluation"
  - "IDI Metric"
  - "COLA Method"
date: 2026-05-08
content_hash: 3daed5b0e9a77a3f
---

# An Information Theoretic Evaluation Metric for Strong Unlearning

**Conference**: AAAI 2026  
**arXiv**: [2405.17878](https://arxiv.org/abs/2405.17878)  
**Code**: To be confirmed  
**Area**: AI Safety/Machine Unlearning  
**Keywords**: Machine Unlearning, Mutual Information, White-box Evaluation, IDI Metric, COLA Method

## TL;DR

This work reveals a fundamental flaw in existing black-box unlearning evaluation metrics (such as MIA and JSD)—modifying only the final layer can satisfy all black-box metrics while the intermediate layers completely retain information about the forgotten data. To address this, the authors propose IDI, a white-box metric that quantifies unlearning effectiveness by estimating the difference in mutual information between each layer and the forgotten labels via InfoNCE. They also propose the COLA method, which achieves IDI scores close to retraining (Retrain) on CIFAR-10/100 and ImageNet-1K.

## Background & Motivation

- **Background**: Machine Unlearning (MU) aims to remove the influence of specific data from trained models to comply with regulations like the "right to be forgotten". Ideal strong unlearning requires the unlearned model to be indistinguishable from a model retrained from scratch (without the forgotten data). However, retraining from scratch is computationally expensive, shifting the research focus toward approximate unlearning methods.
- **Limitations of Prior Work**: Existing evaluations primarily rely on black-box metrics, such as Membership Inference Attacks (MIA) and accuracy comparison. These metrics only examine model outputs and fail to capture residual information of the forgotten data in intermediate layers. More crucially, there is a lack of reliable white-box metrics to verify strong unlearning.
- **Key Challenge**: This work reveals a fundamental problem through a simple experiment: Head Distillation (HD), which modifies only the final classification head (freezing all encoder layers) and aligns the output distribution with a retrained model via distillation, performs excellently on all black-box metrics (even achieving the best MIA score). However, since the encoder remains identical to the original model, 100% of the forgotten data's information is retained. This demonstrates that black-box metrics fail to assess strong unlearning.
- **Key Insight**: Inspired by the information bottleneck principle (information gets more compressed in deeper layers of a DNN), mutual information is used to quantify the residual information between the features of each layer and the forgotten labels. If unlearning is successful, the mutual information of the intermediate layers should be close to that of the retrained model.

## Method

### Overall Architecture

Two core contributions are proposed: (1) IDI (Information Difference Index), a white-box metric that estimates the mutual information between model features and forgotten labels layer by layer to calculate the proportion of information removed in the unlearned model relative to the original model; (2) COLA (COLapse-and-Align), an unlearning method that first collapses the features of the forget set to make them indistinguishable, and then aligns the features of the retain set to restore performance.

### Key Designs

1. **Mutual Information Estimation for the IDI Metric**
    - **Function**: Estimate the mutual information $I(\mathbf{Z}_\ell; Y)$ between each layer's features $\mathbf{Z}_\ell$ and the forgotten labels $Y$.
    - **Mechanism**: Utilize an InfoNCE lower-bound estimator, defining critic functions $f_{\nu_\ell}$ and $g_{\eta_\ell}$ for each layer. $f_{\nu_\ell}$ reuses the $(\ell+1)$-th to $L$-th layers of the model plus a projection layer as a feature extractor, while $g_{\eta_\ell}$ models binary labels as two trainable vectors. Maximizing the InfoNCE loss yields the MI estimate.
    - **Design Motivation**: The model-specific critic design (reusing subsequent layers) ensures that the MI estimation process is model-agnostic—the identical estimation pipeline can be applied to different architectures like ResNet and ViT without requiring per-architecture redesigns.

2. **IDI Score Computation**
    - Compute the information difference for the unlearned model $\theta_u$: $ID(\theta_u) = \sum_{\ell=1}^{L} \max(0, I_{\theta_u}(\mathbf{Z}_\ell; Y) - I_{\theta_r}(\mathbf{Z}_\ell; Y))$
    - Compute $ID(\theta_o)$ similarly for the original model $\theta_o$.
    - IDI Score = $ID(\theta_u) / ID(\theta_o)$, with a range of $[0, 1]$. A value closer to 0 indicates more thorough unlearning.
    - The theoretical IDI for Retrain is 0; the IDI for the original model is 1.

3. **Head Distillation Experiment Revealing Flaws in Black-box Metrics**
    - Freeze the encoder and only retrain the final classification head.
    - Use KL divergence distillation to match the output distribution of a pseudo-retrained model where the logit of the forgotten class is set to negative infinity.
    - Results: This approach ranks among the best across black-box metrics like MIA and JSD, but yields an IDI of 1.000 (indicating complete information retention).
    - Further verification: Retraining the classification head using the frozen encoder of the unlearned model with only 2% of the training data manages to restore over 82% accuracy on the forgotten class (vs. only 41% for Retrain), directly confirming information leakage.

4. **COLA Unlearning Method**
    - **Function**: Eliminate forget set information at the feature level.
    - **Collapse Phase**: Collapse the features of the forget set at the encoder layers to make the features of forget set samples indistinguishable from those of the retain set.
    - **Align Phase**: Align the features of the retain set to restore task performance.
    - On CIFAR-10/ResNet-18, COLA achieves IDI = 0.010 (vs. 0.0 for Retrain) and MIA = 12.64 (close to 10.64 for Retrain).

### Loss & Training

- The critic networks for InfoNCE in the IDI calculation are trained independently (using SGD optimizer) with the forget set ($Y=1$) and retain set ($Y=0$).
- The COLA method operates at the feature level without requiring full retraining.
- Experiments cover: CIFAR-10/100 + ImageNet-1K, using ResNet-18/50 + ViT architectures.
- Unlearning scenarios: Single-class unlearning + multi-class unlearning (5/10 classes) + random data unlearning.

## Key Experimental Results

### Main Results (CIFAR-10 Single-Class Unlearning, ResNet-18)

| Method | UA↑ | TA | MIA (Closer to Retrain is better) | IDI (↓, 0 is optimal) | RTE |
|------|-----|-----|-----|-----|------|
| Retrain | 100.0 | 95.64 | 10.64 | 0.0 | 154.56min |
| HD | 100.0 | 95.22 | 2.05 | **1.000** | 0.10min |
| SALUN | 100.0 | 95.42 | 0.01 | 0.936 | 3.54min |
| RL | 99.93 | **95.66** | 0.0 | 0.830 | 3.09min |
| SCRUB | 100.0 | 95.37 | 19.73 | **-0.056** | 3.49min |
| **COLA** | 100.0 | 95.36 | **12.64** | **0.010** | 4.91min |

### Ablation Study — MI Visualization Across Layers

| Unlearning Method | Shallow Layer MI Features | Deep Layer MI Features | IDI Evaluation |
|---------|----------|----------|--------|
| Retrain | Low MI (Scattered information) | Low MI | 0.0 (Baseline) |
| Original/HD | High MI (Fully retained) | High MI | 1.0 (Not unlearned) |
| GA | Low MI (Close to Retrain) | Low MI | 0.334 (Relatively good) |
| SALUN | High MI (Close to Original) | High MI | 0.936 (Barely unlearned) |
| COLA | Low MI (Close to Retrain) | Low MI | **0.010** (Best) |

### Key Findings

- **HD is "perfect" on black-box metrics but yields IDI = 1.0**: This provides the strongest evidence that black-box metrics are unreliable—merely looking at outputs cannot determine whether a model has truly unlearned.
- **SALUN achieves excellent black-box performance yet yields IDI = 0.936**: This indicates that SALUN's unlearning primarily occurs at the output layer, while intermediate layers almost completely retain the forgotten data's information.
- **Only 2% of the data is required to recover 82% accuracy from the "unlearned" encoder**: For methods like SALUN and RL, freezing the encoder and retraining with a minimal amount of data recovers forgotten performance, which poses a severe risk to safety compliance.
- **COLA's IDI of 0.010 is close to Retrain's 0.0**: This proves that feature-level collapse strategies can effectively eliminate information residue in intermediate layers.
- **IDI exhibits consistent behavior across architectures (ResNet/ViT) and datasets (CIFAR/ImageNet)**: This demonstrates the strong generalizability of the metric.

## Highlights & Insights

- **The "HD Experiment" is the most insightful part of this work**: With a minimal operation (modifying only the classification head), it exposes weaknesses in the entire field's evaluation paradigm. When black-box metrics can be so easily deceived, any claims of unlearning effectiveness based solely on them become untrustworthy.
- **The naturalness of Mutual Information as an unlearning evaluation metric**: MI directly measures "how much information about the forgotten data is still retained in the model," which aligns precisely with the definition of strong unlearning. Black-box metrics are merely incomplete proxies of this goal.
- **Insights from the COLA method**: To achieve thorough unlearning at the feature level, merely performing gradient ascent (GA) or applying saliency masks (SALUN) is insufficient; explicitly collapsing the feature representations is necessary.

## Limitations & Future Work

- IDI computation requires training separate critic networks for each layer, which incurs significant overhead (each setting requires averaging over 5 trials).
- InfoNCE, being a lower-bound estimator of MI, might not be sufficiently tight, especially when the true MI is high.
- The collapse operation of the COLA method may have limited efficacy in random data unlearning (non-class-level unlearning) scenarios.
- This work only discusses classification tasks; unlearning evaluation for generative models (e.g., diffusion models, LLMs) is not covered.

## Related Work & Insights

- **vs. Black-box MIA Metric**: MIA only detects "whether model outputs expose membership information" and cannot detect information residue in intermediate layers. IDI directly measures mutual information at each layer, which is more fundamental.
- **vs. SOTA Unlearning Methods (e.g., SALUN, L1-sparse)**: While these methods perform well on black-box metrics, IDI reveals that their encoders still heavily retain forgotten information, providing an important warning to the machine unlearning community.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The HD experiment revealing flaws in black-box metrics is a major finding, and the IDI metric fills the gap in white-box evaluations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Compiles 14 unlearning methods × 3 datasets × 2 architectures, comprehensively covering single-class, multi-class, and random unlearning.
- Writing Quality: ⭐⭐⭐⭐ Shows a clear logical chain from problem exposure to metric design and method proposal.
- Value: ⭐⭐⭐⭐⭐ Exerts a substantial impact on the evaluation paradigm of the machine unlearning domain—all future unlearning methods should report IDI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MLUBench: A Benchmark for Lifelong Unlearning Evaluation in MLLMs](../../ICML2026/ai_safety/mlubench_a_benchmark_for_lifelong_unlearning_evaluation_in_mllms.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[ICLR 2026\] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](../../ICLR2026/ai_safety/why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](../../ICLR2026/ai_safety/prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)

</div>

<!-- RELATED:END -->
