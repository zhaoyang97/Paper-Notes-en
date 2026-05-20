---
title: >-
  [Paper Note] CAD-VAE: Leveraging Correlation-Aware Latents for Comprehensive Fair Disentanglement
description: >-
  [AAAI 2026][Image Generation][fair disentanglement] CAD-VAE introduces a correlation-aware latent code to capture shared information between target and sensitive attributes…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "fair disentanglement"
  - "VAE"
  - "conditional mutual information"
  - "correlation-aware"
  - "counterfactual fairness"
date: 2026-05-08
content_hash: 53be015e6e24850d
---

# CAD-VAE: Leveraging Correlation-Aware Latents for Comprehensive Fair Disentanglement

**Conference**: AAAI 2026
**arXiv**: [2503.07938](https://arxiv.org/abs/2503.07938)  
**Code**: None  
**Area**: Image Generation / Fairness
**Keywords**: fair disentanglement, VAE, conditional mutual information, correlation-aware, counterfactual fairness

## TL;DR
CAD-VAE introduces a correlation-aware latent code to capture shared information between target and sensitive attributes, achieves disentanglement by directly minimizing conditional mutual information, and employs a relevance-driven optimization strategy to precisely regulate the shared code, attaining state-of-the-art performance on fair representation learning, counterfactual generation, and fair image editing.

## Background & Motivation

### State of the Field
**Background**: Deep generative models (particularly VAEs) have achieved remarkable success in representation learning, yet they may inherit or amplify biases in training data—entanglement between sensitive attributes (e.g., gender, race) and target labels gives rise to fairness concerns. Existing fair disentanglement methods fall into two categories: invariant learning (removing sensitive attributes via adversarial training or regularization) and disentanglement methods (partitioning the latent space into target and sensitive codes in pursuit of statistical independence).

### Limitations of Prior Work
**Limitations of Prior Work**: (1) **The assumption of complete independence is unrealistic**—multiple studies (Jang & Wang 2024; Park et al. 2020) demonstrate that target and sensitive attributes are inherently correlated in real data (e.g., "beard" is associated with both gender and attractiveness), and enforcing complete independence inevitably sacrifices predictive accuracy; (2) **Causal graph methods require domain knowledge**—causal-graph-based disentanglement requires manually constructing causal structures, which are difficult to obtain in practice and erroneous assumptions can lead to worse outcomes; (3) **Indirect methods may leak information**—approaches such as FADES approximate conditional mutual information indirectly via group sampling, yielding limited control.

**Key Challenge**: A fundamental fairness–utility trade-off—the shared information between target and sensitive attributes is simultaneously a source of bias and a useful predictive signal, requiring fine-grained control rather than blunt removal.

### Paper Goals
**Goal**: To address the above core problem by proposing a novel approach that achieves significant improvements on key metrics.

**Core Idea**: CAD-VAE introduces a correlation-aware latent code to capture shared information between target and sensitive attributes, and achieves disentanglement by directly minimizing conditional mutual information.

## Method

### Overall Architecture
CAD-VAE augments the standard VAE disentanglement framework with a third type of latent code—the correlation code $z_c$ (in addition to the target code $z_y$ and the sensitive code $z_s$)—to capture shared information between target and sensitive attributes. Given $z_c$, the conditional mutual information $I(z_y; z_s | z_c)$ between $z_y$ and $z_s$ is directly minimized to achieve conditional independence.

### Key Designs

1. **Correlated Latent Code ($z_c$)**:

    - **Function**: Explicitly models the shared information between target and sensitive attributes.
    - **Mechanism**: The VAE encoder outputs three latent codes $z_y, z_s, z_c$, where $z_c$ is dedicated to capturing the variation factors shared by target and sensitive attributes. The conditional mutual information $I(z_y; z_s | z_c) = E_{p(z_c)}[D_{KL}(p(z_y, z_s|z_c) \| p(z_y|z_c)p(z_s|z_c))]$ should equal zero given $z_c$—meaning that once shared information is extracted, the target and sensitive codes are conditionally independent.
    - **Design Motivation**: Unlike pursuing marginal independence $I(z_y; z_s)=0$ directly, conditional independence $I(z_y; z_s|z_c)=0$ permits the two codes to share information through $z_c$, which better reflects the correlation structure of real-world data.

2. **Direct Minimization of Conditional Mutual Information**:

    - **Function**: Achieves precise disentanglement without domain knowledge.
    - **Mechanism**: The CMI is converted into an optimizable loss via a variational upper bound. Specifically, auxiliary distributions $q(z_y|z_c)$ and $q(z_s|z_c)$ are introduced to approximate the conditional marginals, yielding the upper bound $I(z_y;z_s|z_c) \le E[D_{KL}(p(z_y|z_s,z_c)\|q(z_y|z_c))] + E[D_{KL}(p(z_s|z_y,z_c)\|q(z_s|z_c))]$. The auxiliary distributions are parameterized by small MLPs and trained jointly.
    - **Design Motivation**: Directly minimizing CMI is more precise than indirect methods (e.g., the group sampling of FADES) and provides stronger theoretical guarantees.

3. **Relevance-Driven Optimization**:

    - **Function**: Ensures that $z_c$ captures precisely the necessary shared information without redundancy.
    - **Mechanism**: A relevance measure $R(z_c)$ is introduced to quantify the degree of correlation between $z_c$ and the target/sensitive attributes, imposing an information bottleneck on $z_c$—it should contain just enough shared information to minimize $I(z_y;z_s|z_c)$, but no more. This is implemented in practice via KL-divergence regularization between the posterior and prior of $z_c$.
    - **Design Motivation**: Prevents $z_c$ from degenerating into an "absorb-everything" code—if $z_c$ contains excessive information, $z_y$ and $z_s$ will lose predictive capacity.

### Loss & Training
Total loss = VAE reconstruction loss + KL regularization (for all three latent codes) + CMI upper bound minimization + relevance bottleneck regularization. The model is trained end-to-end without additional causal graph assumptions or domain knowledge.

## Key Experimental Results

### Main Results: Fair Classification Performance (CelebA Dataset)

| Method | Accuracy ↑ | Demographic Parity ↓ | Equalized Odds ↓ | Fairness–Utility Trade-off |
|--------|-----------|---------------------|------------------|---------------------------|
| β-VAE | 82.3% | 0.15 | 0.12 | Poor |
| FFVAE | 83.1% | 0.10 | 0.09 | Moderate |
| FADES | 83.5% | 0.08 | 0.07 | Good |
| **CAD-VAE** | **84.2%** | **0.05** | **0.04** | **Best** |

### Counterfactual Generation Quality

| Method | FID ↓ | Counterfactual Consistency ↑ | Attribute Preservation ↑ |
|--------|-------|------------------------------|--------------------------|
| FactorVAE | 45.2 | 0.72 | 0.81 |
| FADES | 38.7 | 0.79 | 0.85 |
| **CAD-VAE** | **32.4** | **0.86** | **0.91** |

### Ablation Study

| Configuration | Accuracy | DP | EO |
|---------------|----------|-----|-----|
| Full CAD-VAE | 84.2% | 0.05 | 0.04 |
| w/o $z_c$ | 83.0% | 0.11 | 0.09 |
| w/o relevance optimization | 83.8% | 0.07 | 0.06 |
| w/o CMI minimization | 83.3% | 0.09 | 0.08 |

### Key Findings
- $z_c$ is the most critical component—removing it causes fairness metrics to degrade by more than 100%.
- CAD-VAE simultaneously surpasses baselines on both accuracy and fairness metrics, improving fairness without sacrificing utility.
- The method generalizes to VLM settings, demonstrating strong generalizability.

## Highlights & Insights
- The method is elegantly designed with a clear core idea, addressing a key pain point in the field.
- Experiments comprehensively cover multiple datasets and scenarios, validating the effectiveness and robustness of the approach.
- Ablation studies clearly demonstrate the independent contribution of each module.

## Limitations & Future Work
- Generalizability to larger-scale data and more complex scenarios warrants further investigation.
- Computational efficiency could be further optimized to support real-time applications.
- In-depth comparison and complementarity analysis with other related methods remains to be explored.

## Related Work & Insights
- The proposed method demonstrates clear improvements and innovations over representative methods in the field.
- The technical approach provides important reference value for subsequent related work.
- The core module design is extensible to a broader range of application scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The method makes a unique contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Well-organized and clearly presented.
- Value: ⭐⭐⭐⭐ Advances the state of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek](../../ICLR2026/image_generation/seek-cad_a_self-refined_generative_modeling_for_3d_parametric_cad_using_local_in.md)
- [\[AAAI 2026\] CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images](causalclip_causally-informed_feature_disentanglement_and_filtering_for_generaliz.md)
- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](../../CVPR2026/image_generation/da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](../../CVPR2026/image_generation/vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)

</div>

<!-- RELATED:END -->
