---
title: >-
  [Paper Note] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis
description: >-
  [NeurIPS 2025][Interpretability][Text image forgery] This paper proposes FSTS, a Fourier-series-inspired forgery synthesis framework that models the "invisible distribution" (the high-dimensional distribution of forgery operation parameters) from 16,750 real-world forgery instances collected from 67 human participants, generating synthetic training data that more closely approximates real-world forgeries and substantially improving the generalization of text image forgery loc…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Text image forgery"
  - "synthetic data"
  - "invisible distribution modeling"
  - "Fourier series"
  - "forgery parameters"
date: 2026-05-08
content_hash: 4b0f187bb10bc806
---

# Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2511.12658](https://arxiv.org/abs/2511.12658)  
**Code**: [Project Page](https://github.com/)（The paper mentions a Project Page for dataset release）  
**Area**: Interpretability
**Keywords**: Text image forgery, synthetic data, invisible distribution modeling, Fourier series, forgery parameters

## TL;DR

This paper proposes FSTS, a Fourier-series-inspired forgery synthesis framework that models the "invisible distribution" (the high-dimensional distribution of forgery operation parameters) from 16,750 real-world forgery instances collected from 67 human participants, generating synthetic training data that more closely approximates real-world forgeries and substantially improving the generalization of text image forgery localization models.

## Background & Motivation

Text images (e.g., documents, invoices, news screenshots) contain highly sensitive information and are frequent targets of forgery. Existing text image forgery localization (T-IFL) methods rely on large-scale, high-quality annotated data, yet real-world forgery datasets remain limited in scale (e.g., FindIt contains only 240 images; STFD contains only 4,094 images), making it difficult to meet the data demands of deep learning.

Existing synthesis approaches (e.g., DocTamper) attempt to generate forgery data automatically, but they primarily focus on the **visible distribution**—surface-level attributes such as scene diversity, data scale, and linguistic variety. Real-world forgeries, however, involve complex **invisible distributions**: forgers select different forgery types depending on the scene and execute a series of primary operations (region selection, text insertion, geometric transformation) and post-processing steps (blurring, filtering, color adjustment, JPEG compression, etc.). The high-dimensional parameter vectors underlying these operations are imperceptible to the human eye, yet they are critical to the diversity of forgery artifacts and the generalization capacity of detection models.

The core challenges are: (1) how to effectively collect real forgery parameters, given that most forged images retain only the final result with no recoverable operation history; and (2) how many samples are sufficient to adequately model the distribution, since exhaustive collection is infeasible and generalization from limited samples is required.

## Method

### Overall Architecture

FSTS proceeds in three stages: collecting real forgery parameters → hierarchically modeling the parameter distribution → sampling from the distribution to generate synthetic data. The core idea is inspired by the Fourier series — just as a complex waveform can be decomposed into a weighted sum of basis functions, complex forgery behavior can be decomposed into a weighted combination of basic operation–parameter configurations.

### Key Designs

1. **Structured forgery parameter collection pipeline**: 67 experts and volunteers were recruited to perform 5 forgery types (copy-move, splicing, removal, insertion, replacement) across diverse scene categories (photographs, screenshots, scanned documents) using Photoshop, yielding 16,750 forgery instances. Parameters for each editing action were automatically captured via screen recordings, PSD files, and operation logs in multiple formats. This resolves the problem of "how to collect $t$" — rather than reverse-engineering forged images, parameters are recorded directly during the forgery process.

2. **Individual-level distribution modeling**: Analysis reveals that individual forgers tend to reuse similar parameter configurations (e.g., Forger 1 applied Content-Aware Fill in 67% of replacement samples and Gaussian blur in 41.4%). Accordingly, the forgery distribution of individual $i$ is modeled as a weighted combination of $K$ forgery types $\phi_k$:

$$P_S^{(i)}(t) = \sum_{k=1}^{K} a_k^{(i)} \phi_k(t_k^{(i)})$$

where $a_k^{(i)}$ is the frequency weight of forgery type $\phi_k$ and $t_k^{(i)}$ is the representative operation–parameter configuration (selected as the most frequent configuration exceeding a 2% usage threshold). As sample size increases, the statistical characteristics of each type stabilize, allowing approximation by a single representative configuration.

3. **Population-level distribution modeling**: Shared preferences are observed across forgers (e.g., 61.7% use Content-Aware Fill; 39.7% use Gaussian blur). The population-level distribution is therefore obtained by aggregating all individual distributions:

$$P_S(t) = \sum_{k=1}^{K} a_k \phi_k(t_k)$$

where $a_k = \sum_{i=1}^{I} a_k^{(i)}$ and $t_k$ is selected from $\{t_k^{(i)}\}$ across all individuals as configurations shared by at least 5% of participants. Under the approximation $t_k \approx \hat{t}_k$, the optimization objective simplifies to aligning the synthetic weights $\{a_k\}$ with the ground-truth weights $\{\hat{a}_k\}$.

### Loss & Training

FSTS is a data synthesis framework and does not involve additional model training losses. Downstream detection models are trained using their respective default configurations — Protocols 1/2 train for 50 epochs and Protocols 3/4 train for 25 epochs. The synthetic image generation process samples operation–parameter configurations from the modeled distribution and applies the corresponding forgery operations to the source image $I^o$:

$$I^s = \text{Generator}(I^o | \{a_k, t_k, \phi_k\}_{k=1}^K)$$

## Key Experimental Results

### Main Results (Protocol 2: Synthetic Training → Real-world Testing)

| Method | Training Data | Avg. F1 (Real Datasets) | Avg. AUC (Real Datasets) | F1 Gain over DocT-T |
|--------|--------------|------------------------|--------------------------|---------------------|
| RRU-Net | DocT-T | .199 | .765 | — |
| RRU-Net | FSTS-T | .342 | .864 | +.143 |
| DFCN | DocT-T | .102 | .782 | — |
| DFCN | FSTS-T | .327 | .889 | +.225 |
| MVSS-Net | DocT-T | .168 | .697 | — |
| MVSS-Net | FSTS-T | .386 | .812 | +.218 |
| TruFor | DocT-T | .198 | .785 | — |
| TruFor | FSTS-T | .477 | .912 | +.279 |
| STFL-Net | DocT-T | .205 | .781 | — |
| STFL-Net | FSTS-T | .399 | .892 | +.194 |

All methods trained on FSTS-T substantially outperform their DocT-T counterparts on real-world test sets.

### Ablation Study (Protocol 4: Synthetic Pre-training + Real-world Fine-tuning)

| Configuration | Avg. F1 | Avg. AUC | Notes |
|---------------|---------|---------|-------|
| Direct (real data only) | .405 (MVSS-Net) | .758 | Baseline; poor cross-dataset generalization |
| DocT-T pre-training + fine-tuning | .406 | .758 | Negligible gain; indicates large distribution gap in DocTamper |
| FSTS-T pre-training + fine-tuning | .434 | .807 | Consistent improvement; confirms pre-training value of FSTS synthetic data |

### Key Findings

- Under Protocol 2, models trained on FSTS-T achieve average F1 improvements exceeding 14% on real datasets, with some models exceeding 21%
- Models trained on FSTS-T synthetic data can even surpass those trained directly on real data in cross-domain generalization (Protocol 3 vs. Protocol 2)
- DocT-T pre-training frequently leads to performance degradation (negative gain), whereas FSTS-T pre-training almost always yields positive gain
- Across the 5 forgery types, the population-level parameter distributions exhibit clear shared preference patterns, validating the hierarchical modeling design

## Highlights & Insights

- The conceptual shift from **visible distribution** to **invisible distribution** is particularly insightful. Existing synthesis methods focus solely on image-level diversity while neglecting the parameter distribution of the forgery operations themselves — a latent variable that is critical for generating realistic training data
- The Fourier series analogy, while not a strict mathematical correspondence, provides an intuitively elegant framework: complex forgery behavior = weighted superposition of basic operation patterns
- The data collection pipeline has strong practical value: acquiring ground-truth parameters by recording human editing processes (rather than reverse-engineering forged images) is a principled and scalable approach

## Limitations & Future Work

- The current framework covers only 5 forgery types and does not address novel forgery techniques based on deep generative models (GANs/Diffusion models)
- The assumption $t_k \approx \hat{t}_k$ (that synthetic base configurations approximate real base configurations) lacks theoretical guarantees
- Whether a pool of 67 participants is sufficiently representative of global forger behavior patterns remains to be verified
- Some methods still underperform on specific datasets (e.g., PSCC-Net on AFAC), indicating that synthetic data cannot fully compensate for architectural limitations

## Related Work & Insights

- The key distinction from DocTamper: DocTamper generates forgeries using predefined rules, whereas FSTS models the parameter distribution from real human behavior
- The underlying idea is transferable to other domains: any scenario requiring synthetic training data can benefit from "modeling latent parameter distributions from human operation logs"
- Connection to computational forensics: invisible distribution modeling can be viewed as the forward-modeling dual of reverse digital forensics

## Rating

- **Novelty**: ⭐⭐⭐⭐ The invisible distribution modeling perspective is novel, and the Fourier series analogy is creative
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four evaluation protocols, seven baseline methods, five real-world datasets — the experimental design is systematic and comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ The logic is clear, with a well-structured progression from problem formulation to method derivation
- **Value**: ⭐⭐⭐⭐ Direct applicability to document security and digital forensics; the dataset will be publicly released

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering](../../ICLR2026/interpretability/real_reading_out_transformer_activations_for_precise_localization_in_language_mo.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](../../ACL2025/interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](../../CVPR2025/interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[NeurIPS 2025\] Geometric Priors for Generalizable World Models via Vector Symbolic Architecture](geometric_priors_for_generalizable_world_models_via_vector_symbolic_architecture.md)

</div>

<!-- RELATED:END -->
