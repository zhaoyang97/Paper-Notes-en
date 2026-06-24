---
title: >-
  [Paper Note] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization
description: >-
  [ECCV 2024][Causal Inference][Markov Blanket] This work proposes the CMBRL framework to discover Markov Blanket (MB) features—the minimal sufficient statistics of the target variable—within the latent space. This replaces the convention of selecting only causal or anti-causal variables in existing methods, constructing an invariant prediction mechanism to achieve cross-domain generalization.
tags:
  - "ECCV 2024"
  - "Causal Inference"
  - "Markov Blanket"
  - "causal representation learning"
  - "domain generalization"
  - "invariant prediction"
  - "latent variables"
date: 2026-05-08
content_hash: f72b5b0f10690f28
---

# Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Markov Blanket, causal representation learning, domain generalization, invariant prediction, latent variables

## TL;DR

This work proposes the CMBRL framework to discover Markov Blanket (MB) features—the minimal sufficient statistics of the target variable—within the latent space. This replaces the convention of selecting only causal or anti-causal variables in existing methods, constructing an invariant prediction mechanism to achieve cross-domain generalization.

## Background & Motivation

**Background**: The goal of Domain Generalization (DG) is to enable a model trained on source domains to maintain performance on unseen target domains. Causal domain generalization methods are currently a research hotspot, where the core idea is to identify the latent causal variables that generate the input data and construct an invariant causal mechanism for prediction, thereby maintaining robustness under distribution shifts.

**Limitations of Prior Work**: Existing causal DG methods lack consensus on which latent causal variables should be selected for prediction. Some methods select "causes of the target" (causal variables), while others select "effects of the target" (anti-causal variables). But both strategies exhibit clear limitations: selecting only causal variables may miss discriminative features that are highly correlated with the target but lack causal relationships; selecting only anti-causal variables may introduce spurious correlations. A more fundamental issue is that the dichotomy of "causal" or "anti-causal" is too coarse, ignoring other features that are equally invariant and discriminative.

**Key Challenge**: Existing methods suffer from a theoretical gap regarding "which variables to select for prediction"—the division into causal and anti-causal variables is not sufficiently granular, failing to guarantee that the selected variable set is the optimal feature subset for the prediction target. An ideal feature subset should be both minimal (avoiding redundancy) and sufficient (containing all necessary information).

**Goal**: (1) How to define and discover a theoretically optimal subset of predictive features in the latent causal space? (2) How to ensure that this feature subset maintains invariance across different domains? (3) How to simultaneously achieve causal representation learning and optimal feature selection within an end-to-end framework?

**Key Insight**: The authors introduce a classic concept from probabilistic graphical models—the Markov Blanket (MB). In a Bayesian network, the Markov Blanket of a variable is the minimal set consisting of its parents, its children, and its children's other parents. It possesses the following property: given the MB, the target variable is conditionally independent of all other variables. This implies that the MB is the "minimal variable set sharing the maximum mutual information with the target"—precisely matching the requirements for optimal predictive features.

**Core Idea**: To discover the Markov Blanket features of the target variable within the latent causal space, and construct an invariant prediction mechanism using these minimal sufficient features, achieving superior domain generalization compared to using only causal or anti-causal variables.

## Method

### Overall Architecture

CMBRL consists of three core modules: (1) a causal representation learning module, which maps high-dimensional inputs to a structured latent causal space; (2) a Markov Blanket discovery module, which identifies the MB features of the target variable in the latent space; and (3) an invariant prediction module, which constructs a cross-domain invariant predictor based on the MB features. The input image is mapped to a latent representation via an encoder, the latent representation is processed through causal structure learning to obtain a causal graph, the MB features are then identified within the causal graph, and finally, the MB features are used for prediction.

### Key Designs

1. **Causal Representation Learning Module**:

    - **Function**: To recover structured latent causal variables from high-dimensional observational data.
    - **Mechanism**: A VAE-based architecture is utilized, where the encoder maps the input $x$ to latent variables $z = (z_1, z_2, ..., z_d)$. A sparse causal adjacency matrix $A$ is introduced in the latent space to describe the causal relationships among the latent variables. The changes in data distribution provided by multiple source domains are leveraged as "intervention" signals to identify the causal structure—different domains exert different "soft interventions" on the generative mechanisms of the latent variables, rendering the causal relationships identifiable. The learning process adopts a Structural Causal Model (SCM) framework, jointly learning the latent variables and the causal structure by maximizing the likelihood of the observational data while minimizing the sparsity of the causal graph.
    - **Design Motivation**: A causal graph must be established in the latent space in order to subsequently identify the MB. Existing methods typically assume a known causal structure or adopt simple independence assumptions, whereas CMBRL learns the causal structure end-to-end, which aligns better with the complexity of real-world data.

2. **Markov Blanket Discovery Module**:

    - **Function**: To identify the Markov Blanket of the target variable $Y$ within the learned causal graph.
    - **Mechanism**: Given the causal adjacency matrix $A$, MB($Y$) consists of three parts—the parents of $Y$ (direct causes), the children of $Y$ (direct effects), and the other parents of $Y$'s children (co-parents). These nodes are extracted from $A$ via graph operations: $MB(Y) = Pa(Y) \cup Ch(Y) \cup Pa(Ch(Y))$. To make MB selection differentiable during training, soft attention masks are used instead of hard selection—computing the MB relationship weights of each latent variable based on the causal graph structure to obtain continuous MB probability assignments.
    - **Design Motivation**: The theoretical guarantee of MB lies in being the minimal set of variables that satisfies $Y \perp\!\!\!\perp Z_{others} | Z_{MB}$. It is neither missing (sufficiency) nor redundant (minimality). Compared to selecting only causal variables $Pa(Y)$, MB also includes $Y$'s effect variables and their co-parents—although these variables are not direct causes of $Y$, they contain additional discriminative information regarding $Y$.

3. **Invariant Predictor**:

    - **Function**: To build cross-domain invariant classification/regression heads based on MB features.
    - **Mechanism**: The core assumption of invariant prediction is that $P(Y|Z_{MB})$ remains consistent across all domains (invariance condition). During training, prediction losses are computed for each domain separately, and an invariance regularization is added—constraining the differences between the prediction distributions $P_d(Y|Z_{MB})$ across different domains to be minimized. An Invariant Risk Minimization (IRM)-style penalty term is utilized to ensure that MB features are the optimal inputs for the predictor across all domains.
    - **Design Motivation**: Only MB features can satisfy both the "information sufficiency" and "cross-domain invariance" conditions. Pure causal variables may be invariant but lack sufficient information, whereas using all features (which might include spurious correlations) may provide sufficient information but fail to remain invariant across domains. MB achieves a theoretically optimal balance between the two.

### Loss & Training

Total Loss: $L = L_{recon} + \alpha L_{sparsity} + \beta L_{pred} + \gamma L_{inv}$

- $L_{recon}$: VAE reconstruction loss, ensuring that latent variables retain input information
- $L_{sparsity}$: Causal graph sparsity penalty ($L1$ regularization), encouraging a concise causal structure
- $L_{pred}$: Prediction loss based on MB features
- $L_{inv}$: Invariance regularization, constraining prediction consistency across domains
- The training adopts a two-stage strategy: first learning causal representations and structure, and then freezing the causal graph to perform MB discovery and invariant prediction.

## Key Experimental Results

### Main Results

| Dataset | Metric | CMBRL | SOTA (Prev. SOTA) | Gain |
|--------|------|-------|-----------------|------|
| PACS | Acc (%) | 88.7 | 87.4 (CIRL) | +1.3 |
| VLCS | Acc (%) | 79.3 | 78.1 (CIRL) | +1.2 |
| OfficeHome | Acc (%) | 70.5 | 69.6 (IRM) | +0.9 |
| TerraIncognita | Acc (%) | 49.8 | 48.5 (CIRL) | +1.3 |
| DomainNet | Acc (%) | 43.2 | 42.6 | +0.6 |

### Ablation Study

| Configuration | PACS Acc | VLCS Acc | Description |
|------|----------|----------|------|
| Full CMBRL (MB features) | 88.7 | 79.3 | Full model |
| Causal variables only Pa(Y) | 86.9 | 77.8 | Lost information of children and co-parents |
| Anti-causal variables only Ch(Y) | 87.1 | 78.0 | Lost parent node information |
| All latent variables (no selection) | 86.2 | 76.5 | Introduced spurious correlation features |
| w/o causal structure learning | 85.8 | 76.9 | Random MB leads to inaccurate selection |

### Key Findings

- MB features outperform using causal or anti-causal variables alone, validating the theoretical advantage of MB as the "minimal sufficient feature set" (outperforming them by 1.8 and 1.6 percentage points on PACS, respectively).
- Dispensing with feature selection (using all latent variables) yields the worst performance, indicating that redundant and spurious variables indeed damage generalization performance.
- The causal structure learning module is crucial—without an accurate causal graph, it is impossible to accurately identify the MB.
- The improvement is most significant on datasets with particularly large domain discrepancies like TerraIncognita, demonstrating that the invariance of MB is more advantageous under large distribution shifts.

## Highlights & Insights

- **Introduction of Markov Blanket fills the theoretical gap in variable selection**: Previous causal DG methods debated between selecting causal vs. anti-causal variables, and MB offers a theoretically optimal solution as a unified framework. The "minimal sufficiency" of MB ensures that it neither introduces noise like using all variables nor loses useful information like using causal variables only.
- **Bringing a classic concept from graphical models into deep learning**: MB is a classic concept in probabilistic graphical models but is systematically applied in deep domain generalization for the first time. This combination of "classic theory + modern deep learning" is highly instructive—many elegant concepts in graphical models may find new life in deep learning scenarios.
- **End-to-end causal discovery and feature selection**: This method does not require prior causal knowledge, directly learning the causal structure and identifying MB simultaneously from multi-domain data. This end-to-end design makes the approach highly practical.

## Limitations & Future Work

- The dimension $d$ of latent causal variables needs to be preset, where different values of $d$ may lead to different causal graphs and MBs. Automatically determining the latent space dimension remains an open challenge.
- Causal structure learning can be inaccurate under limited data, especially when the number of domains is small. The method relies on "intervention" signals provided by multi-domain data, and an insufficient number of domains may render the causal structure unidentifiable.
- The two-stage training strategy may cause early causal structure errors to propagate to the subsequent MB discovery. Joint optimization might be preferable but harder to train.
- The domain discrepancies of the experimental datasets are relatively moderate; the performance under more extreme distribution shifts (such as sim-to-real) remains unknown.
- High computational overhead: Causal structure learning itself is an NP-hard problem, requiring a trade-off between the efficiency and accuracy of approximation algorithms.

## Related Work & Insights

- **vs CIRL (Causal Invariant Representation Learning)**: CIRL selects only causal variables for prediction. CMBRL selects a more complete feature set via MB, outperforming CIRL across multiple datasets.
- **vs IRM (Invariant Risk Minimization)**: IRM pursues a cross-domain invariant predictor without explicitly modeling the causal structure. CMBRL combines causal modeling with invariant prediction, offering a stronger theoretical foundation than IRM.
- **vs CausalVAE**: CausalVAE conducts causal discovery in the latent space but lacks targeted feature selection. CMBRL adds an MB discovery module on top of it, achieving selective feature utilization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Bringing the Markov Blanket into domain generalization is a brand new perspective with a solid theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers 5 standard DG benchmarks, with well-designed ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Theory and practice are closely integrated, and concepts are explained clearly.
- Value: ⭐⭐⭐⭐ It provides a superior theoretical framework of variable selection for causal domain generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[CVPR 2026\] CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization](../../CVPR2026/causal_inference/cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali.md)
- [\[ICLR 2026\] CARL: Preserving Causal Structure in Representation Learning](../../ICLR2026/causal_inference/carl_preserving_causal_structure_in_representation_learning.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)

</div>

<!-- RELATED:END -->
