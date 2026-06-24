---
title: >-
  [Paper Note] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning
description: >-
  [ICML2025][Self-Supervised Learning][Contrastive Learning] GloFND is proposed to learn a dynamic threshold for each anchor sample, discovering and filtering global false negatives in real-time during training. This improves the representation quality in contrastive learning with low computational overhead.
tags:
  - "ICML2025"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "False Negative Discovery"
  - "Self-supervised Representation Learning"
  - "Global Threshold Optimization"
  - "SogCLR"
date: 2026-05-08
content_hash: ebcf03491892cc84
---

# Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning

**Conference**: ICML2025  
**arXiv**: [2502.20612](https://arxiv.org/abs/2502.20612)  
**Code**: [vibalcam/GloFND](https://github.com/vibalcam/GloFND)  
**Area**: Contrastive Learning / Self-Supervised Learning  
**Keywords**: Contrastive Learning, False Negative Discovery, Self-supervised Representation Learning, Global Threshold Optimization, SogCLR

## TL;DR

GloFND is proposed to learn a dynamic threshold for each anchor sample, discovering and filtering global false negatives in real-time during training. This improves the representation quality in contrastive learning with low computational overhead.

## Background & Motivation

In self-supervised contrastive learning (such as SimCLR, MoCo, SogCLR), positive sample pairs consist of different augmented views of the same image, while negative sample pairs are randomly sampled from the dataset (excluding the anchor). However, this random sampling process incorrectly labels semantically similar samples as negatives, known as **false negatives (FN)**.

**The harm of false negatives**: Taking SogCLR pre-training on ImageNet100 as an example, about 1% of negative pairs are false negatives—approximately 20,000 per batch when batch size = 1024, and about 325 when batch size = 128. These false negatives force the encoder to discard crucial semantic information, resulting in a drop of up to **10%** in linear classifier accuracy under semi-supervised settings.

**Limitations of Prior Work**:

- **Local methods** (WCL, FNC): Search for false negatives only within the mini-batch; top-k similar samples are unreliable when the batch size is small.
- **Global methods** (IFND): Perform k-means clustering on the entire dataset at specific epochs, which is computationally expensive on large-scale datasets.

→ **Core Motivation**: There is a need for a global (dataset-wise), dynamic (on-the-fly), and batch-size-independent false negative discovery method.

## Method

### Core Idea

GloFND learns a **dynamic threshold** $\lambda_i$ for each anchor sample $\mathbf{x}_i$, which is used to filter out the top-$\alpha$% most similar negative samples to the anchor as false negatives.

### Threshold Learning via Convex Optimization

The problem of finding the $(1-\alpha)$-quantile is modeled as the following convex optimization problem:

$$\lambda_i = \arg\min_{\nu \in [-1,1]} \nu\alpha + \frac{1}{|R_i|}\sum_{r \in R_i}(r - \nu)_+$$

where $R_i = \{\text{sim}(E_\mathbf{w}(\mathbf{x}_i), E_\mathbf{w}(\mathbf{x})) \mid \mathbf{x} \in \mathcal{S}_i^-\}$ is the set of cosine similarities between the anchor and all negative samples.

**Lemma 3.1**: The solution $\lambda_i$ is exactly the $k = \lceil \alpha|\mathcal{S}_i^-| \rceil$-th largest value in $R_i$ (or lies between the $k$-th and $(k+1)$-th largest values), thereby precisely selecting the top-$\alpha$% similar negative samples.

### Threshold Update via SGD

In each iteration, the stochastic subgradient is computed for the anchor $\mathbf{x}_i$ within the mini-batch:

$$\hat{\nabla}_{\lambda_i} = \alpha - \frac{1}{|\mathcal{B}_i^-|}\sum_{\mathbf{x} \in \mathcal{B}_i^-} \mathbb{I}(\text{sim}(\mathbf{z}_i, E_\mathbf{w}(\mathbf{x})) > \lambda_i)$$

Then, the updated value is projected as:

$$\lambda_i \leftarrow \Pi_{[-1,1]}[\lambda_i - \theta \hat{\nabla}_{\lambda_i}]$$

where $\theta$ is the learning rate for the threshold. The thresholds $\lambda_j$ for samples not in the current batch remain unchanged.

### Modified Contrastive Loss

The global contrastive loss after removing false negatives is:

$$\ell(\mathbf{w}, \lambda_i; \mathbf{x}_i) = -\text{sim}(\mathbf{z}_i, \mathbf{z}_i') + \tau \log(|\tilde{\mathcal{S}}_i^-| \cdot g(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{S}}_i^-))$$

where $\tilde{\mathcal{S}}_i^- = \{\mathbf{x} \mid \mathbf{x} \in \mathcal{S}_i^-, \text{sim}(\mathbf{z}_i, E_\mathbf{w}(\mathbf{x})) \leq \lambda_i\}$ is the filtered "clean" negative sample set.

### Moving Average Estimator (Inherited from SogCLR)

To avoid the need for large batch sizes, the moving average strategy from SogCLR is adopted to estimate $g$:

$$u_i \leftarrow (1-\gamma)u_i + \gamma \hat{g}(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{B}}_i^-)$$

Encoder gradient estimation:

$$\hat{\nabla}_\mathbf{w} = \frac{1}{|\mathcal{B}|}\sum_{\mathbf{x}_i \in \mathcal{B}} -\nabla_\mathbf{w}\text{sim}(\mathbf{z}_i, \mathbf{z}_i') + \frac{\tau \nabla_\mathbf{w}\hat{g}(\mathbf{w}, \lambda_i; \mathbf{x}_i, \tilde{\mathcal{B}}_i^-)}{u_i}$$

### Bimodal Extension

GloFND can be directly extended to CLIP-style image-text contrastive learning: maintaining two thresholds $\lambda_{I,i}$ (for image anchors) and $\lambda_{T,i}$ (for text anchors) for each instance, and filtering false negatives for the two modalities separately.

### Algorithmic Pipeline (SogCLR + GloFND)

1. Initialize the encoder $\mathbf{w}$, moving average $\mathbf{u}$, and threshold $\boldsymbol{\lambda}$
2. In each iteration: sample batch $\mathcal{B}$, and for each anchor $\mathbf{x}_i \in \mathcal{B}$:
    - Update $\lambda_i$ via SGD (threshold step)
    - Filter false negatives using $\lambda_i$ to obtain $\tilde{\mathcal{B}}_i^-$
    - Update the moving average $u_i$
3. Compute the gradient $\hat{\nabla}_\mathbf{w}$ and update the encoder using Adam/SGD
4. The extra overhead is only $O(B^2)$, which is much smaller than forward/backward propagation

## Key Experimental Results

### Unimodal Contrastive Learning

| Method | Dataset | Setting | Key Results |
|------|--------|------|----------|
| SogCLR + GloFND | ImageNet100 | Semi-supervised 1% Label | Accuracy improvement up to **~10%** vs. no FN processing |
| SogCLR + GloFND | ImageNet100 | Linear Evaluation | Consistently outperforms the SogCLR baseline |
| SimCLR + GloFND | Multiple Datasets | Linear Evaluation | Compatible with various CL methods, achieving consistent improvements |

### Bimodal Contrastive Learning (CLIP-style)

- Integrating GloFND on image-text data improves both retrieval and classification performance.

### Key Findings

- t-SNE visualizations show: after filtering false negatives with GloFND, representation clusters for different categories become more compact and distinct.
- $\alpha$ is a crucial hyperparameter: controlling the false negative ratio, it adapts to both coarse-grained (e.g., cars vs. animals) and fine-grained (e.g., dog breed classification) tasks.
- Extremely low computational overhead: GloFND requires only $O(B^2)$ extra computation, which is negligible compared to forward/backward propagation.

## Highlights & Insights

1. **Convex Optimization Perspective for Quantile Estimation**: Formulating the top-$\alpha$% filtering as a convex optimization problem (Ogryczak & Tamir, 2003) avoids the high computational overhead of full-dataset sorting or clustering.
2. **Global Discovery, Local Computation**: The threshold $\lambda_i$ implicitly reflects the similarity distribution of the anchor in the entire dataset, yet each update only requires samples within the mini-batch.
3. **Plug-and-Play**: GloFND is decoupled from specific contrastive losses—it can be integrated into various frameworks such as SogCLR, SimCLR, and CLIP by simply adding threshold update steps.
4. **Flexible Handling of False Negatives**: Although this work only explores the "filtering" strategy, GloFND can also support treating false negatives as extra positive pairs (FN attraction), opening avenues for future research.
5. **Theoretical Guarantee**: Lemma 3.1 guarantees that the solution of the optimization problem corresponds exactly to the top-$k$ selection.

## Limitations & Future Work

1. **Simplistic False Negative Utilization**: This work only validates the strategy of "filtering" false negatives, without deeply exploring whether converting false negatives to positive pairs (attraction) is superior.
2. **Hyperparameter $\alpha$ Tuning Required**: $\alpha$ depends on prior knowledge or the granularity of downstream tasks, lacking an adaptive adjustment mechanism.
3. **Warm-up Phase Required**: GloFND relies on "sufficiently good" initial representations to bootstrap false negative discovery; under a cold start, the threshold estimation might be inaccurate.
4. **Neglect of Hypergradient**: Directly setting the hypergradient of $\lambda$ with respect to $\mathbf{w}$ to zero (similar to the first-order approximation in MAML) is theoretically sub-optimal.
5. **Lack of Large-scale Experiments**: Validation is mainly performed on medium-scale datasets like ImageNet100, without demonstrating performance on truly large-scale settings like ImageNet-1K/21K.

## Related Work & Insights

- **SogCLR** (Yuan et al., 2022): Global contrastive learning with stochastic optimization, serving as the foundational framework for GloFND.
- **FNC** (Huynh et al., 2022): Within-batch top-$k$ false negative discovery, a representative of local methods.
- **IFND** (Chen et al., 2022): A global method using dataset-wide clustering, which incurs high computational costs.
- **iSogCLR** (Qiu et al., 2023): Learning individualized temperatures to address false negatives, offering a complementary perspective.
- **SupCon** (Khosla et al., 2021): Incorporating ground-truth labels to avoid false negatives, serving as an upper-bound reference.
- **CVaR / Quantile Optimization** (Ogryczak & Tamir, 2003): The mathematical tool source for threshold learning in GloFND.

## Rating

- Novelty: ⭐⭐⭐⭐ — Global false negative discovery from an optimization perspective is a novel and elegant approach.
- Experimental Thoroughness: ⭐⭐⭐ — Covers unimodal, bimodal, and semi-supervised settings, but lacks large-scale datasets and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, standardized algorithm descriptions.
- Value: ⭐⭐⭐⭐ — A plug-and-play false negative filtering module with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](../../ICLR2026/self_supervised/on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](../../ACL2025/self_supervised/whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)

</div>

<!-- RELATED:END -->
