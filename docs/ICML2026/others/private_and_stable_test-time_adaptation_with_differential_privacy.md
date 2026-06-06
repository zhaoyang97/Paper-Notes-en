---
title: >-
  [Paper Note] Private and Stable Test-Time Adaptation with Differential Privacy
description: >-
  [ICML 2026][Test-Time Adaptation] This paper is the first to indicate that Test-Time Adaptation (TTA) causes model parameters to leak private information about test data. It systematically transforms five mainstream TTA…
tags:
  - "ICML 2026"
  - "Test-Time Adaptation"
  - "Differential Privacy"
  - "DP-SGD"
  - "Per-Sample Clipping"
  - "ImageNet-C"
date: 2026-05-08
content_hash: 683b8e77a38937a7
---

# Private and Stable Test-Time Adaptation with Differential Privacy

**Conference**: ICML 2026  
**arXiv**: [2606.01908](https://arxiv.org/abs/2606.01908)  
**Code**: None  
**Area**: AI Security / Differential Privacy / Test-Time Adaptation  
**Keywords**: Test-Time Adaptation, Differential Privacy, DP-SGD, Per-Sample Clipping, ImageNet-C  

## TL;DR
This paper is the first to indicate that Test-Time Adaptation (TTA) causes model parameters to leak private information about test data. It systematically transforms five mainstream TTA methods (Tent, EATA, SAR, DeYO, and COME) into Differential Privacy (DP) versions using per-sample gradient clipping and Gaussian noise. On ImageNet-C, the method provides provable $(\epsilon,\delta)$-DP guarantees and unexpectedly discovers that "clipping itself" enhances TTA accuracy by $0.1\%$–$4.1\%$.

## Background & Motivation

**Background**: TTA continues to update model parameters during the deployment phase using unlabeled test samples (typically updating only the affine parameters of normalization layers) to combat distribution shifts through entropy minimization, filtering, or reweighting. Tent represents entropy minimization; EATA adds reliability filtering and Fisher regularization; SAR employs sharpness-aware optimization; DeYO utilizes patch shuffle to calculate Pseudo-Label Probability Difference (PLPD); and COME replaces entropy with Dirichlet uncertainty.

**Limitations of Prior Work**: All existing methods implicitly assume that test data does not require protection. However, test images may contain sensitive information such as medical imaging, faces, or location trajectories. TTA "welds" the information of these samples into the model parameters. Once the model or its outputs are queried or shared, attackers can launch membership inference or reconstruction attacks to recover individual test samples from the updates.

**Key Challenge**: Directly applying DP-SGD to TTA faces several issues: (1) TTA batches are often as small as 1, causing DP noise to be amplified relative to the signal; (2) TTA methods rely heavily on data-dependent filtering/reweighting, which are essentially "queries" in a privacy sense that break DP and stability if implemented naively; (3) Traditional DP-SGD analysis is built on sampling and leave-one-out adjacency, whereas TTA is a single-epoch stream where each sample is seen only once, requiring a different accounting framework.

**Goal**: (a) Provide a general recipe for DP-TTA; (b) Implement it across five representative TTA methods; (c) Systematically characterize the "privacy budget vs. adaptation accuracy" curve and identify which TTA designs are naturally more DP-friendly.

**Key Insight**: The authors realize that the streaming nature of TTA actually allows for cleaner DP analysis—each sample is processed only once in a single step. Therefore, no composition is required across steps; as long as the sensitivity of a single step is controlled, the global privacy is closed by post-processing.

**Core Idea**: Use "per-sample gradient clipping + Gaussian noise" as a mandatory privacy interface for TTA. Non-DP-friendly filtering/reweighting operators are either removed or converted into DP post-processing forms. Simultaneously, per-sample clipping is discovered to be a "free lunch" for improving TTA accuracy even with zero noise.

## Method

### Overall Architecture

Let the source model be $f_{\theta_0}$, the test stream be $\{B_t\}_{t=1}^T$, and the adaptable parameters be the affine subset of normalization layers $\theta^a \subset \theta$. The standard TTA update is $\theta_{t+1} = \theta_t - \eta \Delta_t$, where $\Delta_t = \frac{1}{|B_t|}\sum_{x_i \in B_t} w_t^i g_t(x_i)$ and $g_t(x_i) = \nabla_\theta \ell_\text{tta}(x_i,\theta_t)$. DP-TTA replaces this by first performing $L_2$ clipping on each sample gradient: $\bar g_t(x_i) = g_t(x_i)/\max(1,\|g_t(x_i)\|_2/C)$, and then injecting Gaussian noise: $\Delta_t^{DP} = \frac{1}{|B_t|}(\sum_i \bar g_t(x_i) + \mathcal{N}(0,C^2\sigma^2 I_d))$. BatchNorm is disabled at the architectural level (as cross-sample coupling violates the per-sample sensitivity assumption), and ViT-Base/16 with LayerNorm is used uniformly.

### Key Designs

1. **DP-TTA Privacy Analysis (Streaming + Change-one Adjacency)**:

    - **Function**: Provides a tight $(\epsilon,\delta)$ guarantee for each DP-TTA method without needing multi-epoch sub-sampling composition used in DP-SGD.
    - **Mechanism**: Since TTA is single-epoch and each sample is seen once, once a single step guarantees $\mu$-GDP, all subsequent steps are post-processing of DP results, requiring no composition across steps. "Change-one" (replacing one sample) adjacency is used instead of "leave-one-out" (as batch sizes are fixed in streaming), which doubles sensitivity from $C$ to $2C$. Thus, DP-Tent satisfies $\mu$-GDP with $\mu = 2/\sigma$, leading to $\delta(\epsilon) = \Phi(-\sigma\epsilon/2 + 1/\sigma) - e^\epsilon \Phi(-\sigma\epsilon/2 - 1/\sigma)$.
    - **Design Motivation**: Align the analysis with the actual usage of TTA rather than forcing DP-SGD training assumptions; convert the "streaming without resampling" disadvantage into an advantage by eliminating expensive composition losses.

2. **De-filtering / Post-processing (DP-EATA / DP-SAR / DP-DeYO / DP-COME)**:

    - **Function**: Ensures that data-dependent operators in non-trivial TTA methods satisfy DP by either internalizing them into clipping or moving them outside the privacy boundary.
    - **Mechanism**: Entropy threshold filtering and diversity filtering in EATA are removed because they cause effective batch size drift and require private statistics. However, Fisher regularization $\mathcal{R}(\theta_t,\theta_0)$ is retained as it is DP post-processing. In SAR, the two-point sharpness update consumes double the privacy budget; this is replaced by a private SAM variant using the previous step's private gradient $\tilde g_{t-1}$ to construct the perturbation $\tilde \epsilon_t = \rho \tilde g_{t-1}/\|\tilde g_{t-1}\|_2$. DeYO's PLPD term $e^{\text{PLPD}_\theta(x_i,x_i')}$ is integrated into the loss to pass through the per-sample clipping channel. COME uses Dirichlet uncertainty $\ell_\text{COME} = -\sum_k b_k\log b_k - u\log u$, requiring no additional modification.
    - **Design Motivation**: Every TTA method has operators that "query the test set unauthorized." It is necessary to identify which can be moved after DP results (free), which must be absorbed into clipping (increasing noise), and which are too costly and should be removed.

3. **Per-sample clipping as a Free Stabilizer for TTA**:

    - **Function**: Using per-sample gradient clipping alone can stabilize and improve TTA accuracy even without adding noise ($\sigma=0$).
    - **Mechanism**: While per-batch clipping was previously though to be ineffective for TTA, per-sample clipping preserves the individual direction of each sample while capping its norm. Applying per-sample clipping without DP noise to EATA/SAR/DeYO/COME results in an average adaptation gain increase from $0.1\%$ to $4.1\%$, with up to a $14\%$ boost on ImageNet-R.
    - **Design Motivation**: Pseudo-label gradients in TTA are high-variance and easily dominated by outliers. Per-sample clipping acts as a "hard" directional sparsification and outlier suppression, preventing bad samples from causing model collapse in continual streams.

### Loss & Training

Only affine parameters of normalization layers are updated. DP-EATA retains the $\lambda \nabla_\theta \mathcal{R}(\theta_t,\theta_0)$ term (Fisher regularization does not enter the clipping channel). Hyperparameters are selected via cross-validation with $\eta \in \{10^{-4}, 5\cdot 10^{-4}, \dots, 1\}$ and $C \in \{1,5,10,15\}$. The batch size is fixed at 64. Noise levels $\sigma \in \{8.594,1.966,1.084,0.777,0.619\}$ correspond to $\epsilon = 1,5,10,15,20$ at $\delta=10^{-6}$.

## Key Experimental Results

### Main Results

Accuracy of DP-Tent under different privacy budgets on ImageNet-C (severity 5, continual setup, ViT-B/16, average of 5 seeds):

| Setup | $\epsilon$ | Avg Top-1 (%) | Description |
|------|------------|----------------|------|
| Non-private Tent | $\infty$ | 60.8 | Original baseline |
| DP-Tent | 20 | 62.9 | 2.1% higher than non-private |
| DP-Tent | 15 | 62.6 | Still exceeds non-private |
| DP-Tent | 10 | 62.1 | Still exceeds non-private |
| DP-Tent | 1 | 58.5 | Only 2.3% drop under strong privacy |

Accuracy gap compared to non-private versions for other methods at $\epsilon=20$: DP-EATA $-2.9\%$, DP-SAR $-1.2\%$, DP-DeYO $-2.4\%$, DP-DeYO-COME $-1.7\%$. These losses are significantly smaller than the accuracy drops typically seen in DP-SGD training.

### Ablation Study: Contribution of Per-sample Clipping Alone

Comparison of "with vs. without per-sample clipping" (no DP noise) on ImageNet-C (continual, ViT-B/16):

| Configuration | Avg Gain | Key Findings |
|------|----------|----------|
| Original TTA (No clip) | $+0.1\%$ | Average of 5 methods relative to source model |
| Original TTA + per-sample clip | $+4.1\%$ | 4 methods improved, only 1 slight decrease of $-0.3\%$ |
| DeYO-COME + clip | $67.5\%$ | Highest absolute accuracy in continual setup |
| Tent / EATA + clip (ConvNeXt) | $+1\%$ to $+5\%$ | Consistent across continual & episodic setups |
| ImageNet-R + clip | Up to $+14\%$ | Larger gains on more challenging data |

### Key Findings

- **Per-sample clipping is a free lunch for TTA**: Even without privacy requirements, clipping alone stabilizes continual TTA and prevents collapse—a granularity previously overlooked by studies using per-batch clipping.
- **Streaming TTA makes DP costs exceptionally cheap**: Due to the single-epoch nature, change-one adjacency, and post-processing closure, there is no composition loss. This makes "moderate privacy" like $\epsilon=10$ almost free on ImageNet-C, sometimes outperforming non-private baselines.
- **More filtering correlates with less DP-friendliness**: Elaborate filters in EATA/SAR/DeYO mostly need to be removed during DP conversion (as they break sensitivity bounds or require expensive private statistics). Surprisingly, the simplest method, Tent, is the most resilient to DP conversion.
- **Architectural constraints are rigid**: BatchNorm violates sensitivity independence under per-sample clipping. Deployment requires switching to LayerNorm-based models (like ViT), meaning the architecture choice itself becomes a privacy decision.

## Highlights & Insights
- **Shifting the "threat model" to TTA is a critical paradigm shift**: Previous TTA research focused solely on accuracy and stability. This paper demonstrates that parameter updates during deployment constitute a privacy surface and provides a provable solution, completing the spectrum from DP-SGD to DP fine-tuning.
- **The "Remove, Preserve, Internalize" classification is reusable**: The framework for handling data-dependent operators based on their sensitivity cost provides a clear engineering template for making any new TTA method DP-compliant.
- **Independent value of per-sample clipping**: This finding can be cited by non-privacy TTA works as a "near-zero cost" trick to stabilize continual TTA, potentially gaining wider community adoption than the DP contribution itself.

## Limitations & Future Work
- **Cost of removing filters**: Removing reliability filters from EATA/DeYO only drops accuracy by 1–3% in these experiments, but their long-term stability contributions in longer sequences or extreme OOD scenarios might be undervalued.
- **Batch=1 remains unresolved**: While the authors claim DP-TTA is robust to batch size, experiments were fixed at 64. The trade-off between strong privacy and the extreme $B=1$ case common in edge deployment requires further quantification.
- **Lack of empirical privacy auditing**: DP provides an upper bound. This paper does not use membership inference attacks to verify the lower bound, leaving it unclear if the actual leakage is significantly lower than $\epsilon$ suggests.
- **Architecture limited to LayerNorm**: Excluding BatchNorm models limits the scope. Future research into ghost normalization or private BN estimation is needed to relax this constraint.

## Related Work & Insights
- **vs DP-SGD (Abadi et al., 2016)**: DP-SGD assumes training phases, multi-epoch, sub-sampling, and leave-one-out adjacency. This work proves streaming single-epoch TTA avoids composition losses but requires change-one adjacency (sensitivity $2C$).
- **vs Original Tent / EATA / SAR / DeYO / COME**: These works treat accuracy and stability as the only goals; this paper adds a privacy dimension and finds that the simplest method (Tent) is the most robust under DP.
- **vs DP-SAM (Park et al., 2023)**: This work adapts the technique of using the previous private gradient as a perturbation direction to reduce DP-SAR from two gradient evaluations to one.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)

</div>

<!-- RELATED:END -->
