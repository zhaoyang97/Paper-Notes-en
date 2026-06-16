---
title: >-
  [Paper Note] Feature-aware Modulation for Learning from Temporal Tabular Data
description: >-
  [NeurIPS 2025][Signal & Communication][temporal shift] This paper argues that the core challenge in temporal tabular learning is not simply "adding a time embedding…
tags:
  - "NeurIPS 2025"
  - "Signal & Communication"
  - "temporal shift"
  - "feature modulation"
  - "concept drift"
  - "Yeo-Johnson"
  - "tabular deep learning"
date: 2026-05-08
content_hash: a3beb6f1c9ced535
---

# Feature-aware Modulation for Learning from Temporal Tabular Data

**Conference**: NeurIPS 2025
**arXiv**: [2512.03678](https://arxiv.org/abs/2512.03678)  
**Code**: [https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation)  
**Area**: Temporal Tabular Learning / Signal & Communication / Temporal Distribution Shift
**Keywords**: temporal shift, feature modulation, concept drift, Yeo-Johnson, tabular deep learning

## TL;DR

This paper argues that the core challenge in temporal tabular learning is not simply "adding a time embedding," but rather that the semantics of many features drift over time. To address this, the paper proposes feature-aware modulation, which uses temporal context to dynamically generate per-feature shift, scale, and nonlinear shape parameters, re-aligning cross-temporal semantics. The approach enables deep models to consistently outperform GBDT on average rank for the first time on the TabReD benchmark.

## Background & Motivation

Tabular learning has long been dominated by GBDT.

Even as deep models such as FT-Transformer, TabR, TabM, and ModernNCA have advanced rapidly in recent years, tree-based models remain more stable in real-world business scenarios affected by temporal distribution shift.

The root cause is that most tabular learning methods assume i.i.d. data.

In reality, time alters the relationship between features and labels.

Income is affected by inflation. House coordinates remain unchanged, but the meaning of "prime location" evolves with urban development. User behavior, policy environments, medical workflows, and financial risk preferences all cause the same numerical values to carry different meanings across different years.

The authors summarize this phenomenon as "feature semantic evolution."

More specifically, features carry both *objective semantics* and *subjective semantics*. Objective semantics refer to the inherent meaning of a value, such as geographic coordinates or absolute salary. Subjective semantics refer to meaning relative to a distributional context, such as "high income," "popular area," or "anomalously high risk." Concept drift is typically driven by the latter.

This also explains why naively concatenating a time embedding often falls short. Feeding time information directly into the model and expecting it to learn how to interpret change is problematic: if the semantics of input features are not first corrected, the model may entangle the temporal modality with the input modality, leading to overfitting on short-term patterns and poor generalization to the future.

The authors therefore raise a more fundamental question: rather than having the model memorize patterns at each time point, is it possible to first "translate" features from different time periods into a more semantically consistent space? If so, the downstream backbone can then learn decision boundaries in a more stable representation space. This is the motivation behind feature-aware modulation.

## Method

### Overall Architecture

The input is a timestamped tabular sample $(\mathbf{x}, t)$.

The model first obtains a temporal embedding $\psi(t)$ via a time encoder.

A lightweight modulator then reads $\psi(t)$ and produces three groups of parameters for each feature dimension:

$\gamma$, controlling scale;

$\beta$, controlling translational shift;

$\lambda$, controlling nonlinear shape change.

Each feature $x_i$ is then transformed via a Yeo-Johnson transform combined with affine modulation:

$$\tilde{x}_i = \gamma_i(\psi(t)) \cdot \text{YJ}(x_i;\, \lambda_i(\psi(t))) + \beta_i(\psi(t)).$$

The modulated features are subsequently fed into an arbitrary backbone, such as MLP or TabM.

Modulation can be applied not only at the input layer but also at intermediate representation layers and at the output logits layer.

The key design principle is that time no longer serves as a parallel input feature concatenated with raw features, but instead acts as a conditioning signal that governs "how features should be interpreted."

### Key Designs

1. **Shifting from "temporal modeling" to "semantic alignment"**

    - Function: Reformulates the temporal distribution shift problem as a semantic calibration problem in the representation space.
    - Mechanism: The authors argue that temporal tabular shift is fundamentally neither pure covariate shift nor pure label shift, but rather concept drift in a large number of subjectively interpreted features.
    - Design Motivation: If the backbone directly learns "how to make decisions at a given year," it memorizes local temporal patterns. If feature semantics are first aligned and the backbone then learns a unified decision boundary, generalization is more robust.

2. **Three-parameter feature modulation: mean, scale, and skewness**

    - Function: Captures the most critical temporal distribution changes with minimal degrees of freedom.
    - Mechanism: The authors observe that temporal drift commonly manifests as three types of statistical change: mean shift, std shift, and skewness shift, corresponding respectively to $\beta$, $\gamma$, and $\lambda$.
    - Design Motivation: Rather than an unconstrained hypernetwork, this is a lightweight modulator with strong inductive bias, making it considerably less prone to overfitting.

3. **Yeo-Johnson transform in place of plain FiLM**

    - Function: Enables differentiable nonlinear reshaping of feature distributions.
    - Mechanism: FiLM is limited to scale and shift, affecting only mean and variance. The addition of the Yeo-Johnson transform allows dynamic modification of distribution shape, which is especially suited to skewed, heavy-tailed, or temporally deforming features.
    - Design Motivation: Many tabular features are non-Gaussian, and temporal drift does not manifest solely as linear translation; shape-level modulation is therefore necessary.

4. **Multi-layer modulation with input-layer priority**

    - Function: Provides temporal adaptation at different levels of representation.
    - Mechanism: Modulation can be applied at the raw input, intermediate representations, and output logits. Experiments show that enabling all three layers is optimal, but input-layer modulation alone already captures the majority of gains.
    - Design Motivation: The input layer possesses the most complete and undistorted original information and is therefore the most natural location for semantic alignment; deeper modulation supplements more abstract temporal adaptation.

5. **Lightweight implementation decoupled from the backbone**

    - Function: Minimally disrupts existing tabular model architectures.
    - Mechanism: The modulator generates only $3m$ parameters, where $m$ is the feature dimension, rather than producing all network weights as in a hypernetwork.
    - Design Motivation: This allows the method to be inserted into models such as MLP and TabM with minimal overhead, at a fraction of the cost of a fully dynamic network.

### Loss & Training

Cross-entropy is used for classification and MSE for regression.

The optimizer is AdamW with early stopping.

Hyperparameter search uses Optuna with 100 trials; each configuration is averaged over 15 random seeds.

The temporal embedding dimension is fixed at 128 and includes both periodic and trend components.

Notably, the paper introduces no complex training tricks in the optimization objective; the core innovation is entirely concentrated in the modulation mechanism applied prior to the representation layers. This is one of the paper's most convincing aspects: the gains are stable under a standard training protocol without relying on any specialized loss formulation.

## Key Experimental Results

### Main Results

Experiments are conducted on TabReD, covering 8 real-world tabular datasets with temporal distribution shift. Evaluation metrics are AUC for classification and RMSE for regression, aggregated into average rank.

The most important result is that, with temporal modulation, TabM achieves an average rank of 3.500, surpassing CatBoost at 4.375. This marks the first time a deep learning method systematically outperforms GBDT under this temporal shift setting.

| Method | Type | Representative Results | Avg. Rank ↓ | Notes |
|--------|------|------------------------|-------------|-------|
| CatBoost | Static GBDT | HI 0.9639, CT 0.4792 | 4.375 | Strong traditional baseline, consistently stable |
| TabM | Static Deep | HI 0.9640, CT 0.4813 | 7.250 | Strong deep model, but still lags under temporal shift |
| MLP + Time Embedding | Adaptive Deep | HI 0.9471, CT 0.4801 | 14.375 | Direct time concatenation yields limited gains |
| TabM + Time Embedding | Adaptive Deep | HI 0.9629, CT 0.4791 | 5.125 | Better than static deep, but not best |
| **TabM + Temporal Modulation** | **Ours** | **HI 0.9641, CT 0.4773** | **3.500** | First to surpass GBDT in average rank |
| **MLP + Temporal Modulation** | **Ours** | **HI 0.9593, CT 0.4782** | **11.000** | Lightweight backbone also benefits substantially |

Two observations merit particular attention.

First, modulation is not merely incremental improvement on a strong model. Even when applied to a plain MLP, it outperforms many more complex static or time-embedding-based methods.

Second, time embedding is not useless, but its effect is clearly weaker than modulation. This supports the authors' thesis: the question is not whether to condition on time, but whether time should be directly mixed with raw features.

### Ablation Study

The authors systematically compare modulation positions. The conclusion is clear: full-layer modulation is best, but input-layer modulation alone already captures the vast majority of the gains.

| Input | Middle | Output | Avg. Gain | Fraction of Full Gain | Avg. Rank ↓ |
|-------|--------|--------|-----------|----------------------|-------------|
| ✗ | ✗ | ✗ | 0.00% | 0% | 5.500 |
| ✓ | ✗ | ✗ | 1.83% | 87.4% | 3.250 |
| ✓ | ✗ | ✓ | 1.54% | 73.6% | 3.625 |
| ✓ | ✓ | ✗ | 1.62% | 77.2% | 3.750 |
| ✓ | ✓ | ✓ | **2.09%** | **100%** | **2.500** |

The authors further note that removing input-layer modulation leaves only 56.8% of the full gains intact. This points to a simple but important principle: semantic alignment is most effective when performed early. Once raw features are misinterpreted at an early stage, corrective interventions at later layers become increasingly difficult.

| Observation | Empirical Finding | Implication |
|-------------|-------------------|-------------|
| Input-only modulation | Achieves 87.4% of full gains | Lowest-cost, most transferable deployment |
| Full-layer modulation | Avg. gain of 2.09% | Multi-layer modulation is complementary |
| No input-layer modulation | Only 56.8% of gains remain | Early representations are most critical |
| Larger time embedding dimension | Traditional methods degrade | Confirms the reality of modality entanglement and scaling issues |

### Key Findings

- The paper provides one of the clearest characterizations to date of the difficulty in temporal tabular learning, attributing it to "feature semantic drift" rather than broadly invoking non-i.i.d. assumptions.
- Input-layer modulation contributes the vast majority of gains, supporting the view that aligning semantics before learning discriminative boundaries is a more natural approach than direct time concatenation.
- Modulation benefits both MLP and TabM, indicating it is not a backbone-specific trick but a general-purpose front-end.
- A pilot study shows that after modulation, feature distributions across different time periods, while not perfectly i.i.d., are sufficiently aligned for the model to learn consistent decision boundaries.

## Highlights & Insights

- **The "objective vs. subjective semantics" analytical framework is highly valuable.** It translates the vague notion of temporal drift in business applications into an interpretable learning objective.
- **The choice of modulation statistics is admirably restrained.** Targeting only three types of distributional change—mean, std, and skewness—yet achieving strong results demonstrates that well-chosen structural priors outweigh brute-force model scaling.
- **The introduction of the Yeo-Johnson transform is well-motivated.** Unlike plain FiLM, the method genuinely allows distribution shape to vary over time, which is particularly important for tabular data.
- **87.4% of gains are recoverable from input-layer modulation alone.** This makes the method highly appealing from an engineering standpoint due to its low deployment cost.
- **The paper can be interpreted as automating interpretable temporal feature engineering.** This is a perspective worth generalizing broadly.

## Limitations & Future Work

First, full-stage modulation is incompatible with PLR embeddings, which limits deeper integration with some of the strongest tabular backbones.

Second, experiments are primarily conducted on TabReD. While TabReD is an important benchmark for temporal tabular learning, validation across more industries and larger-scale tabular settings remains necessary.

Third, the paper focuses on three distributional statistics: mean, std, and skewness. While sufficient for most forms of drift, more complex temporal semantic changes—such as multi-modal distribution switching or structural changes in sparse patterns—may require a richer family of modulation functions.

Fourth, the temporal embedding remains a predefined structure. For event-driven settings without clear periodicity, whether such prior-based encodings are sufficient requires further investigation.

Fifth, while the authors provide an intuitive pilot study visualization, the learned values of $\gamma$, $\beta$, and $\lambda$ themselves lack deeper interpretation.

Several directions for future work are promising: designing modulation variants compatible with PLR embeddings; extending modulation to tabular-text, multimodal tabular, or graph-structured temporal settings; and conditioning modulation parameters not only on time but also on group, region, or environmental variables to produce finer-grained context-aware tabular learners.

## Related Work & Insights

- **vs. direct time embedding**: The latter treats time as an additional input for the model to absorb end-to-end; the present work instead uses time to rewrite how features are interpreted. The two approaches differ fundamentally in their inductive biases.
- **vs. FiLM / Hypernetwork**: This paper extends FiLM with nonlinear distributional shape modulation, while remaining far more lightweight than a full hypernetwork—occupying a practical middle ground.
- **vs. GBDT**: Tree models tend to be more stable under temporal drift partly because they are more robust to changes in feature thresholds. The present work attempts to explicitly restore this source of stability within deep models.
- **Transferable insight**: In temporal tabular modeling, one should first ask "do these features still mean the same thing today as they did last year" before stacking complex temporal architectures.
- **Application domains**: Financial risk control, longitudinal medical follow-up, recruitment profiling, insurance pricing, and real estate valuation are all settings where temporally conditioned feature semantic alignment would be a natural and valuable first step.

## Rating

- Novelty: ⭐⭐⭐⭐☆ Reframes temporal tabular learning from a semantic drift perspective and realizes it via feature modulation—a genuinely fresh angle.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results, pilot study, ablation, and extended analyses are comprehensive; the primary limitation is reliance on a single benchmark.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation is clearly articulated, examples are intuitive, and the method is straightforward to understand.
- Value: ⭐⭐⭐⭐⭐ Lightweight, transferable, and empirically robust; highly impactful for practical temporal tabular learning.

---

## Related Papers

- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[CVPR 2025\] Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ICML 2026\] Joint Model and Data Sparsification via the Marginal Likelihood](../../ICML2026/signal_comm/joint_model_and_data_sparsification_via_the_marginal_likelihood.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)

</div>

<!-- RELATED:END -->
