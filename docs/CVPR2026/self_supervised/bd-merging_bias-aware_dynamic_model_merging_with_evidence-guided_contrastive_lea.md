---
title: >-
  [Paper Note] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning
description: >-
  [CVPR 2026][Self-Supervised Learning][Model Merging] This paper proposes the BD-Merging framework, which trains a debiased router via Dirichlet evidential modeling, Adjacency Discrepancy Score (ADS)…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Model Merging"
  - "Multi-Task Learning"
  - "Evidential Deep Learning"
  - "Distribution Shift"
  - "Contrastive Learning"
  - "uncertainty estimation"
date: 2026-05-08
content_hash: a82023b498606492
---

# BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning

**Conference**: CVPR 2026
**arXiv**: [2603.03920](https://arxiv.org/abs/2603.03920)
**Code**: Not available
**Area**: Self-Supervised Learning
**Keywords**: Model Merging, Multi-Task Learning, Evidential Deep Learning, Distribution Shift, Contrastive Learning, uncertainty estimation

## TL;DR

This paper proposes the BD-Merging framework, which trains a debiased router via Dirichlet evidential modeling, Adjacency Discrepancy Score (ADS), and discrepancy-aware contrastive learning to adaptively assign model merging weights, significantly improving the robustness and generalization of merged models under test-time distribution shifts and on unseen tasks.

## Background & Motivation

**Rise of Model Merging (MM)**: Multi-task learning demands substantial data and computation, and data sharing is often prohibited by privacy constraints. Model merging integrates independently fine-tuned checkpoints to achieve multi-task capability without retraining, making it an efficient alternative.

**Neglected Reliability under Distribution Shift**: Existing MM methods generally assume that test data follows the same distribution as training or auxiliary data. In practice, however, sensor noise, transmission distortion, and environmental variation cause test-time input distribution shifts that severely degrade the performance of merged models.

**Test-time Bias**: Experiments demonstrate that even mild natural perturbations cause significant accuracy drops across all existing methods (e.g., Task Arithmetic drops 16.8% at corruption level L3), revealing a lack of robustness to input-level noise in current approaches.

**Insufficient Generalization to Unseen Tasks**: AdaMerging achieves 90.79% on seen tasks but collapses to 49.83% on unseen tasks, exposing severe overfitting. Methods relying on auxiliary data further amplify the distribution gap when that auxiliary data is mismatched.

**Lack of Fine-Grained Sample-Level Alignment**: Existing methods adjust weights only at the global or task level and cannot capture sample-level distributional discrepancies, leaving them unable to address conflicting knowledge and biased aggregation induced by heterogeneous distribution shifts.

**Core Insight**: Leveraging evidential uncertainty to capture distributional discrepancies and using it to guide adaptive representation alignment is the key breakthrough for addressing distribution shift in MM.

## Method

### Overall Architecture

BD-Merging comprises three core modules: (1) a joint evidential head based on the Dirichlet distribution that models uncertainty over a unified label space; (2) the Adjacency Discrepancy Score (ADS), which quantifies evidential alignment among neighboring samples; and (3) a discrepancy-aware contrastive learning mechanism that guides a debiased router to adaptively assign merging weights per sample. The entire pipeline requires no labeled data and operates in a fully unsupervised setting.

### Key Design 1: Joint Evidential Head

- **Function**: An evidential head is appended to the pretrained backbone to output Dirichlet concentration parameters $\boldsymbol{\alpha}$ for each sample over the unified label space $\mathcal{Y} = \bigcup_{k=1}^{K} \mathcal{Y}_k$, from which belief mass $b_c$, uncertainty $u$, and predictive probability $p_c$ are derived.
- **Mechanism**: An Inter-class Evidence Contrastive (IEC) metric $\nu = (S / \alpha_{\hat{c}^{(1)}}) \cdot (L / S) \cdot (\alpha_{\hat{c}^{(2)}} / \alpha_{\hat{c}^{(1)}})$ is introduced to jointly account for prediction concentration, inter-class competition, and semantic ambiguity, compensating for the inability of conventional EDL metrics (total evidence and top-class confidence) to characterize cross-task semantic shift.
- **Design Motivation**: Test-time distribution shift amplifies semantic ambiguity in overlapping label spaces, and a single uncertainty measure is insufficient to distinguish different types of prediction failure. The IEC provides finer-grained uncertainty estimates by capturing inter-class dependencies.

### Key Design 2: Adjacency Discrepancy Score (ADS)

- **Function**: For each sample $x_i$, a neighborhood set $\mathcal{A}_r(x_i)$ of radius $r$ is constructed in feature space, and the ADS $d_{ik}$ is computed by combining three complementary factors.
- **Mechanism**: ADS is the product of three terms:
    - **Prediction Sharpness**: $\mathrm{Sharp}(x_i) = \mathbb{E}_{x_j \in \mathcal{A}_r} \log(S_j / \max_c \alpha_{jc} - 1)$, measuring overall epistemic uncertainty in the neighborhood;
    - **Semantic Divergence**: $\mathrm{Div}(x_i) = \mathbb{E}_{x_j \in \mathcal{N}_r} \| \boldsymbol{\alpha}_i / S_i - \boldsymbol{\alpha}_j / S_j \|_1$, quantifying class-level distributional deviation between the target sample and its neighbors;
    - **Opinion Conflicts**: $\mathrm{Conf}(x_i, x_k) = \sum_c |p_{ic} - p_{kc}| \cdot (1-u_i)(1-u_k)$, characterizing belief conflicts among high-confidence samples.
- **Design Motivation**: No single metric can comprehensively characterize local distributional discrepancy. The three-factor joint evaluation provides a unified view of local discrepancy from the dimensions of global uncertainty, semantic divergence, and confidence conflict, effectively distinguishing in-distribution samples from perturbed or unseen-task inputs.

### Key Design 3: Discrepancy-Aware Contrastive Learning and Debiased Router

- **Function**: Based on ADS threshold $\epsilon$, the neighborhood is partitioned into a positive set $\mathcal{M}_r^+(i)$ ($d_{ik} < \epsilon$) and a negative set $\mathcal{M}_r^-(i)$ ($d_{ik} \geq \epsilon$). A contrastive loss is constructed to pull consistent samples together and push conflicting ones apart. Simultaneously, a debiased router (two-layer MLP) is trained to compute per-sample task-level and layer-level merging weights from token embeddings of the pretrained backbone.
- **Mechanism**: The router outputs $\{w_k\} = \mathrm{softmax}(R(\mathbf{H}))$, and the merged parameters are $\theta^* = \theta_0 + \sum_k w_k \cdot \tau_k$. Dynamic weight assignment replaces fixed-weight merging, allowing different inputs to receive different weight combinations.
- **Design Motivation**: Fixed-weight merging tends to produce interference across heterogeneous tasks. The debiased router adaptively adjusts based on input features, reducing task interference while improving adaptability to unknown distributions.

## Loss & Training

Training proceeds in two stages:

**Stage 1: Evidential Head Training**

$$\mathcal{L}_{\mathrm{Head}} = \mathcal{L}_{\mathrm{Ent}} + \gamma \mathcal{L}_{\mathrm{Inv}}$$

- $\mathcal{L}_{\mathrm{Ent}}$: Entropy minimization with KL divergence regularization toward a non-informative prior, encouraging sharp predictions while avoiding overconfidence.
- $\mathcal{L}_{\mathrm{Inv}}$: Inverse correlation loss, constraining uncertainty $u$ and IEC $\nu$ to maintain an inverse relationship.

**Stage 2: Router Training**

$$\mathcal{L}_{\mathrm{BD}} = \mathcal{L}_{\mathrm{Unsup}} + \eta \mathcal{L}_{Dis}$$

- $\mathcal{L}_{\mathrm{Unsup}}$: Shannon entropy minimization to enhance the determinacy of merged predictions.
- $\mathcal{L}_{Dis}$: Discrepancy-aware contrastive loss using ADS-partitioned positive and negative sample pairs.

All regularization coefficients ($\lambda, \gamma, \eta$) are set to 0.1. Training runs for 300 epochs with a batch size of 16.

## Key Experimental Results

### Main Results

**Table 1: Performance under Test-Time Bias (ViT-B/32, average accuracy over 8 tasks)**

| Method | Clean | L1 (↓) | L2 (↓) | L3 (↓) |
|------|-------|--------|--------|--------|
| Task Arithmetic | 69.09 | 64.07 (−7.3%) | 60.30 (−12.7%) | 57.50 (−16.8%) |
| Ties-Merging | 72.92 | 68.14 (−6.6%) | 64.34 (−11.8%) | 61.51 (−15.6%) |
| AdaMerging (Layer) | 74.33 | 69.00 (−7.2%) | 64.46 (−13.3%) | 61.22 (−17.6%) |
| Twin-Merging | 84.10 | 79.13 (−5.9%) | 74.17 (−11.8%) | 70.88 (−15.7%) |
| AdaMerging w/Surgery | 84.40 | 79.02 (−6.4%) | 74.33 (−11.9%) | 70.97 (−15.9%) |
| **BD-Merging (Layer)** | **87.15** | **83.31 (−4.4%)** | **78.78 (−9.6%)** | **75.36 (−13.5%)** |

BD-Merging achieves the best performance across all corruption levels with the smallest performance degradation (only −4.4% at L1 vs. −5.9%–7.3% for other methods).

**Table 2: Seen vs. Unseen Task Generalization (4 seen + 4 unseen)**

| Method | Seen Task Avg | Unseen Task Avg |
|------|-------------|-------------|
| AdaMerging | 90.79 | 49.83 |
| Twin-Merging | 93.06 | 53.03 |
| **BD-Merging** | **94.53** | **55.01** |

BD-Merging improves accuracy on both seen and unseen tasks simultaneously, achieving the best generalization–specialization balance.

### Ablation Study

**(Clean / Corrupted L2)**

| Variant | Clean | Corrupted |
|------|-------|-----------|
| BD-Merging | 87.15 | 78.78 |
| w/o Router | 78.31 (−8.84) | 67.25 (−11.53) |
| w/o ADS | 84.48 (−2.67) | 75.44 (−3.34) |
| w/o $\mathcal{L}_{Dis}$ | 83.34 (−3.81) | 74.26 (−4.52) |
| w/o Div(·) | 85.36 (−1.79) | 76.28 (−2.50) |

The debiased router is the most critical component (removal causes ~9/11 point drops), and among the three ADS factors, Div(·) contributes the most.

## Highlights & Insights

1. **Precise Problem Formulation**: This work is the first to systematically study the reliability of model merging under test-time distribution shift, explicitly identifying test-time bias and unseen-task generalization as two major challenges.
2. **Elegant Integration of Evidential Modeling and Contrastive Learning**: Uncertainty signals from EDL guide the positive/negative sample partition in contrastive learning, forming a closed loop — evidential modeling detects discrepancy → ADS quantifies discrepancy → contrastive learning exploits discrepancy.
3. **Efficiency–Performance Balance**: BD-Merging approaches the performance of individually fine-tuned models while incurring substantially lower computational overhead than methods such as AdaMerging w/Surgery.
4. **Interpretable Router**: The routing weight distributions for different unseen tasks exhibit clear task-specific patterns (e.g., concentrated allocation for Cars vs. uniform allocation for SUN397), providing intuitive interpretability.

## Limitations & Future Work

1. **Validation Limited to Image Classification**: All eight datasets are image classification benchmarks; validation on NLP, multimodal, and other task types is absent, leaving generalizability in question.
2. **Neighborhood Construction Overhead**: ADS requires computing neighborhood sets in feature space, which may introduce additional computational cost for large-scale datasets; scalability is not discussed in the paper.
3. **Hyperparameter Sensitivity**: Neighborhood radius $r$, threshold $\epsilon$, and multiple loss coefficients all require tuning. The paper uniformly sets all coefficients to 0.1, but careful adjustment may be necessary in different scenarios.
4. **Limited Gains on Unseen Tasks**: Although BD-Merging outperforms baselines on unseen tasks (55.01% vs. 53.03%), the absolute performance still falls well below the pretrained model (56.99%), leaving significant room for improvement.
5. **Fixed Router Architecture**: The two-layer MLP router is relatively simple; more expressive structures such as attention mechanisms or mixture-of-experts could be explored.

## Related Work & Insights

- **Task Arithmetic / Ties-Merging / DARE**: Static weight merging methods that do not consider input-level adaptation and exhibit poor robustness under distribution shift. BD-Merging surpasses this fixed-weight paradigm through dynamic routing.
- **AdaMerging**: Learns task- and layer-adaptive weights but optimizes on auxiliary data, leading to overfitting when the auxiliary distribution mismatches the test distribution. BD-Merging's evidence-guided mechanism models uncertainty directly on test features, reducing dependence on the auxiliary data distribution.
- **Twin-Merging**: Dynamically integrates modular knowledge with high efficiency but limited accuracy. BD-Merging achieves a better trade-off between accuracy and efficiency.
- **Surgery**: Improves merging quality through surgical adjustments but at high computational cost. BD-Merging achieves comparable accuracy with lower overhead.
- **Evidential Deep Learning**: BD-Merging extends EDL from conventional OOD detection to the model merging setting, opening a new application direction for EDL.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing evidential learning into model merging and designing a closed loop of ADS and contrastive learning is conceptually novel and technically complete.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of multi-level corruption, seen/unseen tasks, ablations, router analysis, and multi-backbone validation; non-visual tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete mathematical derivations, and rich figures; problem formulation and method presentation are well connected.
- **Value**: ⭐⭐⭐⭐ — The first systematic study of distribution-shift robustness in MM, with important reference value for real-world deployment scenarios.

## Related Papers

- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](unigeoclip_geospatial_contrastive.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](../../ICML2025/self_supervised/what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[CVPR 2026\] MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](unigeoclip_geospatial_contrastive.md)
- [\[CVPR 2026\] MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)

</div>

<!-- RELATED:END -->
