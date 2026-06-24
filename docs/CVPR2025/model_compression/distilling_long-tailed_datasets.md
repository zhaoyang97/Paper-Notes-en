---
title: >-
  [Paper Note] Distilling Long-tailed Datasets
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] This work presents the first systematic study on long-tailed dataset distillation. It reveals that existing methods degrade severely in long-tailed scenarios (even underperforming random selection) and proposes two strategies: Distribution-agnostic Matching (DAM) and Expert Decoupling (ED). The proposed method significantly outperforms existing approaches on CIFAR-10/100-LT and Tiny-ImageNet-LT (e.g.…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Long-tailed Distribution"
  - "Trajectory Matching"
  - "Distribution-agnostic Matching"
  - "Expert Decoupling"
date: 2026-05-08
content_hash: 80e691b7d4c31ce7
---

# Distilling Long-tailed Datasets

**Conference**: CVPR 2025  
**arXiv**: [2408.14506](https://arxiv.org/abs/2408.14506)  
**Code**: [https://github.com/ichbill/LTDD](https://github.com/ichbill/LTDD)  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Long-tailed Distribution, Trajectory Matching, Distribution-agnostic Matching, Expert Decoupling

## TL;DR
This work presents the first systematic study on long-tailed dataset distillation. It reveals that existing methods degrade severely in long-tailed scenarios (even underperforming random selection) and proposes two strategies: Distribution-agnostic Matching (DAM) and Expert Decoupling (ED). The proposed method significantly outperforms existing approaches on CIFAR-10/100-LT and Tiny-ImageNet-LT (e.g., surpassing DATM by 19.7% at an imbalance factor of 100).

## Background & Motivation

**Background**: Dataset distillation has achieved remarkable results on balanced datasets (e.g., CIFAR-10/100, ImageNet). However, real-world data typically exhibits a long-tailed distribution (e.g., medical images), where a few head classes contain a vast majority of samples, while most tail classes have very few samples.

**Limitations of Prior Work**: Directly applying existing DD methods to long-tailed datasets leads to a drastic drop in performance—DATM achieves only 40.1% accuracy at an imbalance factor of 200, which is worse than the 49.9% achieved by random selection. The root causes of this issue are twofold: (1) Experts trained on long-tailed data generate biased gradients skewed toward head classes, and this bias is transferred to the synthetic dataset during distillation, causing tail-class images to contain far less useful information than head-class ones; (2) Long-tailed experts provide unreliable predictions for tail classes (confidence of only 0.38 vs. 0.97 for head classes), leading to poor-quality soft label initialization.

**Key Challenge**: During distillation, the student is trained on the balanced synthetic data but needs to match the expert trained on long-tailed data. The fundamentally different weight distributions between the two create a conflict in the bi-level optimization objective.

**Goal**: To distill balanced and highly informative synthetic datasets from long-tailed training data.

**Key Insight**: To address this mismatch, the student is guided to "simulate" long-tailed training behavior to bridge the distribution gap with the expert, and decoupled trained experts are used to provide more reliable supervision signals.

**Core Idea**: A long-tailed-aware loss function is utilized to encourage the student's weight distribution to naturally mimic the long-tailed expert, while decoupled representation and classifier experts provide supervision for the backbone and classifier trajectory matching, respectively.

## Method

### Overall Architecture
Based on the trajectory matching framework (e.g., DATM), two improvements are introduced: (1) DAM modifies the inner-loop training loss of the student, enabling the student to generate a weight distribution similar to long-tailed training even on balanced synthetic data; (2) ED trains two decoupled experts—a representation expert (trained on the entire long-tailed data) and a classifier expert (fine-tuned on oversampled balanced data with the backbone frozen)—to match the trajectories of the backbone and classifier layers, respectively.

### Key Designs

1. **Distribution-agnostic Matching (DAM)**:

    - **Function**: Eliminates the mismatch in weight distribution between the student and the expert caused by different training data distributions.
    - **Mechanism**: During the student's inner-loop training, a modified long-tailed distribution loss $\mathcal{L}^c$ replaces the standard cross-entropy loss. Inspired by Balanced Softmax, this loss weighs each class based on its sample count in the original long-tailed dataset: assigning larger weights to head classes and smaller weights to tail classes. Consequently, when trained on balanced synthetic data, the student's weight distribution naturally aligns with that of the long-tailed expert, reducing their trajectory distance. The key equation uses $-\lambda \log(s_{y_i})$ to adjust logits, with $g(s_{y_i})$ normalizing the class weights.
    - **Design Motivation**: Directly matching a biased expert transfers the bias to the synthetic data. DAM prompts the student to actively "simulate" the bias, keeping the bias within the student's weights rather than contaminating the synthetic data.

2. **Expert Decoupling (ED)**:

    - **Function**: Provides more accurate distillation supervision and high-quality soft label initialization.
    - **Mechanism**: The experts are trained in two stages—the representation expert trains the entire model on the original long-tailed data to learn feature representations, while the classifier expert freezes the backbone and fine-tunes the classifier on oversampled balanced data. During distillation, joint matching is performed: the backbone trajectory of the student is guided by the representation expert, while the classifier trajectory is guided by the classifier expert. Soft labels are generated by the classifier expert because it retains high confidence even for tail classes.
    - **Design Motivation**: During representation learning, long-tailed data actually helps in learning good feature representations (as more diverse head data does not harm the representation quality), but the classifier becomes heavily biased. Decoupling allows each expert to leverage its respective strengths.

3. **Joint Trajectory Matching**:

    - **Function**: Optimizes the matching of both the backbone and the classifier simultaneously.
    - **Mechanism**: The total loss is defined as $\mathcal{L} = \lambda_{\text{rep}} \mathcal{L}_{\text{match}}(\text{backbone}) + \lambda_{\text{cls}} \mathcal{L}_{\text{match}}(\text{classifier})$, where the representation expert handles backbone matching and the classifier expert handles classifier matching.
    - **Design Motivation**: Matching either expert alone is suboptimal because each expert has its own bias or limitation.

### Loss & Training
DAM uses a long-tailed distribution-weighted softmax loss (inspired by Balanced Softmax), and trajectory matching employs the standard normalized L2 distance. $\lambda$ controls the smoothness of the logit adjustments, while $\lambda_{\text{rep}}$ and $\lambda_{\text{cls}}$ control the weight ratio between the two matching losses.

## Key Experimental Results

### Main Results

| Dataset | IF | IPC | Ours | DATM | MTT | Random |
|--------|-----|-----|------|------|-----|--------|
| CIFAR-10-LT | 10 | 50 | 70.5% | 66.7% | 62.0% | 51.9% |
| CIFAR-10-LT | 100 | 50 | 64.0% | 44.3% | 47.8% | 52.6% |
| CIFAR-10-LT | 200 | 50 | 62.3% | 40.1% | 23.9% | 49.9% |
| CIFAR-100-LT | 10 | 50 | 34.8% | - | - | 32.1% |
| Tiny-ImageNet-LT | 10 | 50 | - | - | - | 20.6% |

*\*IF = Imbalance Factor (higher IF denotes greater imbalance). At IF = 200, DATM and MTT collapse severely (even worse than random selection), while Ours maintains 62.3%.*

### Ablation Study

| Configuration | CIFAR-10-LT IF=100 IPC=50 | Description |
|------|--------------------------|------|
| DAM + ED (Full) | 64.0% | Full method |
| DAM Only | 57.1% | Drops by 6.9% without ED |
| ED Only | 54.8% | Drops by 9.2% without DAM |
| Baseline (DATM) | 44.3% | Without any modifications |

### Key Findings
- As the imbalance level increases (IF from 10 to 200), the performance of existing methods continuously declines. However, the performance drop of the proposed method is much smaller, demonstrating strong robustness.
- Both DAM and ED are indispensable: DAM addresses the distribution bias issue (+12.8% vs. baseline), and ED resolves the weak supervision issue (+10.5% vs. baseline). Their combination yields the optimal result.
- Visualizations show that the proposed method leads to a balanced weight distribution in the classifier (with weight norms close across all classes), whereas the baseline's classifier weights are heavily skewed toward head classes.
- This work is the first to demonstrate that dataset distillation can achieve nearly lossless compression in long-tailed scenarios (approaching full-dataset training performance under certain IF and IPC configurations).

## Highlights & Insights
- **Pioneering Problem Definition**: Long-tailed dataset distillation is a completely neglected yet highly practical problem (since medical and autonomous driving data are naturally long-tailed). This paper is the first to systematically define and address it.
- **Counter-Intuitive DAM Design**: Intuition suggests that the student should learn the "correct" balanced distribution. However, DAM forces the student to simulate the long-tailed bias instead, containing the bias within the student's weights rather than polluting the synthetic data. This "fight fire with fire" approach is highly ingenious.
- **Complementary Decoupled Experts**: Rather than simply using a single better expert, the method assigns tasks to two experts based on what they do best (representation vs. classification). This division-of-labor paradigm has broad transfer value.

## Limitations & Future Work
- The class distribution of the original dataset must be known to construct the DAM loss; if the distribution is unknown, it requires additional estimation.
- Expert Decoupling requires training two sets of expert trajectories, which doubles the computational cost.
- The method was validated only on CIFAR and Tiny-ImageNet scales, and lacks evaluation on large-scale long-tailed distributions like ImageNet-1K.
- In extreme cases where the imbalance factor is exceptionally high and IPC is extremely low, some tail classes might have fewer original samples than the IPC itself; this boundary condition has not been fully discussed.

## Related Work & Insights
- **vs. DATM**: DATM is the state-of-the-art on balanced datasets but collapses to 40.1% at IF = 200. The proposed method achieves 62.3% under the same setting through DAM+ED, yielding a 22.2 percentage point improvement.
- **vs. Long-Tailed Learning Methods**: Long-tailed learning techniques like Balanced Softmax are creatively integrated into the distillation framework. Instead of being used merely during training, they are integrated into the student's inner-loop to bridge the distribution gap.
- **vs. IDM**: While IDM is relatively stable in long-tailed scenarios (as distribution-matching methods are generally more robust than trajectory-matching ones), the proposed method still outperforms IDM by approximately 10%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define and solve the long-tailed DD problem; the DAM design is counter-intuitive and highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across four datasets, multiple IF and IPC settings, and detailed ablations, though lacking large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ Deep problem analysis with clear illustrations (especially Figures 3 and 5).
- Value: ⭐⭐⭐⭐⭐ Fills the gap of DD in long-tailed scenarios, possessing high practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](../../ICML2026/model_compression/mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)
- [\[CVPR 2025\] Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs](chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](../../ACL2025/model_compression/law_of_capacity_gap_distilling_language_models.md)

</div>

<!-- RELATED:END -->
