---
title: >-
  [Paper Note] Feedforward Few-shot Species Range Estimation
description: >-
  [ICML 2025][Species Distribution Modeling] Proposes FS-SINR (Few-shot Spatial Implicit Neural Representations), a Transformer-based feedforward few-shot species range estimation model. Without requiring retraining for new species, it predicts spatial distributions in a single forward pass from a few (or even zero) observation locations, outperforming retraining-based methods like LE-SINR on IUCN and S&T benchmarks with only 2-6% of the computational time.
tags:
  - "ICML 2025"
  - "Species Distribution Modeling"
  - "Few-shot Learning"
  - "Transformer"
  - "Spatial Implicit Neural Representation"
  - "Multimodal Fusion"
date: 2026-05-08
content_hash: 50b29363372792ca
---

# Feedforward Few-shot Species Range Estimation

**Conference**: ICML 2025  
**arXiv**: [2502.14977](https://arxiv.org/abs/2502.14977)  
**Code**: [GitHub](https://github.com/Chris-lange/fs-sinr)  
**Area**: LLM Evaluation  
**Keywords**: Species Distribution Modeling, Few-shot Learning, Transformer, Spatial Implicit Neural Representation, Multimodal Fusion

## TL;DR

Proposes FS-SINR (Few-shot Spatial Implicit Neural Representations), a Transformer-based feedforward few-shot species range estimation model. Without requiring retraining for new species, it predicts spatial distributions in a single forward pass from a few (or even zero) observation locations, outperforming retraining-based methods like LE-SINR on IUCN and S&T benchmarks with only 2-6% of the computational time.

## Background & Motivation

**Background**: Most species on Earth lack sufficient observation data to accurately estimate their spatial distribution. The iNaturalist platform has recorded 130 million observations for approximately 373,000 species, but more than half of the species have fewer than 10 observation records. Deep learning methods, such as SINR (Cole et al., 2023), have made progress by jointly representing tens of thousands of species in a single model, but still require a large number of training samples per species.

**Limitations of Prior Work**: When a new species not present in the training set appears, existing methods (such as SINR and LE-SINR) must retrain an embedding vector for the new species (e.g., via logistic regression), which is highly inefficient for interactive exploration and large-scale applications. Furthermore, the reality that most species only have very few observation records significantly compromises the efficacy of standard methods.

**Key Challenge**: Species range estimation requires the capability of "inferring global distribution from a few observations." However, existing methods follow a "train-then-query" paradigm, which cannot generalize to unseen species. Few-shot species range estimation also faces unique challenges: fixed input domain (Earth's surface), multi-label settings (a single location can have multiple species), immense label space (tens of thousands of species), and presence-only data (no confirmed absences).

**Goal**: (1) Design a feedforward model that generates range estimations for unseen species in a single forward pass; (2) significantly improve estimation accuracy under extremely few-shot settings (<10 observations); (3) flexibly integrate auxiliary metadata (text, images, etc.) to further enhance performance.

**Key Insight**: Shift the species embedding from a "learnable vector per species" to being "generated in real-time from a set of observation locations by a Transformer." Consequently, the embedding of a new species can be obtained in a single forward pass without retraining.

**Core Idea**: Utilize a Transformer to encode a variable-length set of observation locations into a species embedding vector, replacing the fixed embedding learned individually for each species in SINR.

## Method

### Overall Architecture

The input consists of a set of context locations $\mathcal{C}^t = \{c_1, \ldots, c_k\}$ for the target species (where each $c_i$ represents latitude and longitude coordinates) as well as optional text or image metadata. First, each location is encoded into a $d$-dimensional embedding vector (token) via a shared spatial encoder $f_\theta$. These tokens, along with a CLS token and a REG token, are fed into a Transformer encoder $m_\psi$. The CLS token output by the Transformer is processed by a species decoder MLP $s()$ to obtain the species embedding. For a query location $x$, its spatial embedding $f_\theta(x)$ is dot-produced with the species embedding, and a sigmoid function is applied to yield the probability of the species' presence at $x$.

### Key Designs

1. **Spatial Encoder + Transformer Architecture**:

    - **Function**: Maps a variable-length set of observation locations to a fixed-dimensional species embedding vector.
    - **Mechanism**: The spatial encoder employs a multi-layer fully connected network (with residual connections) from SINR, pre-trained on large-scale data in the SINR manner, after which the classification head is discarded. The Transformer consists of 4 encoder layers and does not use positional encoding (since the input set is unordered). Instead, it adds a learned "embedding type" vector to each token to distinguish between location, text, image, CLS, and REG tokens.
    - **Design Motivation**: Set inputs naturally require permutation invariance, which is precisely satisfied by the self-attention mechanism of the Transformer. The CLS token serves as a global aggregator, compressing the variable-length sequence into a fixed-dimensional species representation.

2. **Multimodal Context Fusion (Text + Image)**:

    - **Function**: Optionally provides textual descriptions (e.g., "this species is distributed in tropical rainforests") or species images as additional context alongside spatial observations.
    - **Mechanism**: Texts are extracted into embeddings via a frozen GritLM, and images are extracted via a frozen EVA-02 ViT (pre-trained on iNat). Both are then mapped to the same space as the spatial tokens via two-layer MLPs. During training, text/image tokens are randomly dropped out with a probability of 0.5, and location tokens with a probability of 0.1, ensuring model robustness under various input combinations.
    - **Design Motivation**: For species with highly scarce observations, habitat descriptions in texts (such as "high-altitude mountains" or "deserts") can provide ecological priors that spatial data alone cannot offer, significantly narrowing the search space. Images provide species appearance clues, though the information density is limited.

3. **In-batch Loss Function $\mathcal{L}_{\text{AN-full-b}}$**:

    - **Function**: Adapts the all-species assume-negative loss of SINR to the feedforward architecture.
    - **Mechanism**: Since FS-SINR lacks a per-species weight matrix $W$, it cannot calculate the loss for all species simultaneously. Instead, the loss is computed over the $s^b$ species in the batch: $\mathcal{L}_{\text{AN-full-b}}(\hat{y}, z^b) = -\frac{1}{s^b}\sum_{j=1}^{s^b}[\mathbb{1}_{[z^b=j]}\lambda\log(\hat{y}_j) + \mathbb{1}_{[z^b \neq j]}\log(1-\hat{y}_j) + \log(1-\hat{y}'_j)]$, where $\hat{y}'_j$ represents the prediction of a random pseudo-absence location.
    - **Design Motivation**: It retains the core structure of presence + pseudo-absence contrast while adapting to the characteristics of the feedforward architecture. A batch size of 2048 ensures that each batch contains enough species to form meaningful negative samples.

### Loss & Training

An in-batch assume-negative-full loss with a batch size of 2048 is utilized. The training data comprises 35.5 million records from iNaturalist (representing 44,422 species, excluding evaluation species), 127k text descriptions, and 200k images. Each training sample provides 20 context locations. The total number of parameters is 8.2M (compared to 11.9M for SINR, as per-species embeddings are no longer required).

## Key Experimental Results

### Main Results

| Method | IUCN MAP (1-shot) | IUCN MAP (5-shot) | IUCN MAP (10-shot) | S&T MAP (1-shot) | S&T MAP (5-shot) | Requires Retraining |
|------|-------------------|-------------------|--------------------|--------------------|-------|----------|
| SINR | ~0.15 | ~0.30 | ~0.38 | ~0.30 | ~0.55 | Yes |
| LE-SINR (RT) | ~0.25 | ~0.40 | ~0.48 | ~0.45 | ~0.65 | Yes |
| Active SINR | ~0.10 | ~0.20 | ~0.25 | ~0.25 | ~0.40 | No |
| **FS-SINR** | ~0.22 | ~0.38 | ~0.45 | ~0.45 | ~0.68 | **No** |
| **FS-SINR (RT)** | ~0.35 | ~0.48 | ~0.53 | ~0.55 | ~0.72 | **No** |

### Ablation Study

| Configuration | IUCN MAP (0-shot) | S&T MAP (0-shot) | Description |
|------|-------------------|------------------|------|
| SINR (TST, Upper Bound) | 0.67 | 0.77 | Training set contains evaluation species |
| FS-SINR (No Metadata) | 0.05 | 0.18 | Only CLS token output |
| FS-SINR (Habitat Text) | 0.33 | 0.53 | Habitat descriptions |
| FS-SINR (Range Text) | 0.52 | 0.64 | Range descriptions |
| FS-SINR (Image) | 0.19 | 0.38 | Image only |
| FS-SINR (Image + RT) | 0.46 | 0.64 | Images sometimes introduce interference |

### Key Findings

- **Text >> Image**: Range text provides far richer distribution information than a single image (0.52 vs 0.19 on IUCN). This is intuitive—"distributed in the South American Andes" provides a more direct spatial prior than a photo of a bird.
- **Images can introduce negative effects**: On IUCN, Image + Range Text (0.46) is lower than pure Range Text (0.52), as images might introduce incorrect spatial biases.
- **Huge speed advantage**: On identical hardware, FS-SINR generates range estimations for all evaluation species using only 2% (CPU) or 6% (GPU) of the computation time required by LE-SINR, due to the elimination of per-species retraining.
- **Reasonable estimations even with only 1 observation point**: The model learns strong spatial priors—from a single African observation point, it can infer that the species is likely distributed across Sub-Saharan Africa.

## Highlights & Insights

- **The elegant design of "set-to-embedding"**: It transforms few-shot learning from "per-species optimization" to "feedforward encoding." The core lies in replacing the learnable per-species weight vector with a Transformer CLS token. This paradigm can be generalized to any scenario requiring class embedding inference from a few exemplars (e.g., few-shot image classification, user profiling).
- **Efficacy of text as a spatial prior**: Simple text descriptions (such as "distributed in deserts/rainforests/high mountains") combined with a single observation location can significantly shift model predictions, demonstrating the practical value of language model knowledge in ecology. This inspires a new paradigm: leveraging LLM-generated descriptions as weak supervision signals to assist in low-annotation scenarios.
- **Fewer parameters (8.2M vs 11.9M)**: Although a Transformer module is introduced, the elimination of the per-species embedding matrix $W \in \mathbb{R}^{d \times s}$ (where $s$ is the number of species) results in a 30% reduction in total parameters.

## Limitations & Future Work

- **Deterministic outputs**: Given the same input, the model always outputs the same distribution map. However, in few-shot scenarios, the same set of observations may correspond to multiple plausible distributions. Introducing stochasticity (such as latent sampling) to generate multiple candidate distributions and quantify uncertainty is an important direction for expansion.
- **Presence-only limitations**: The model cannot utilize confirmed absence information (i.e., knowing a species is not present in a certain place). Adding an "absence" embedding type for different token types is a natural improvement.
- **Training data bias**: The iNaturalist data is heavily biased toward regions with active citizen science like North America and Europe, causing poorer performance in Africa and Asia. The paper analyzes this bias in the appendix but does not propose a solution.
- **Evaluation constraints**: Expert range maps from IUCN and S&T also contain errors and primarily cover vertebrates and birds, leaving their applicability to plants and invertebrates uncertain.

## Related Work & Insights

- **vs SINR (Cole et al., 2023)**: SINR learns a fixed embedding $w_j$ for each species, failing to handle unseen species. FS-SINR dynamically generates embeddings from observation sets using a Transformer, achieving true generalization. Nonetheless, SINR remains competitive when data is abundant (>50 observations).
- **vs LE-SINR (Hamilton et al., 2024)**: LE-SINR utilizes text information to improve zero-shot/few-shot estimation but still requires retraining a classifier for each new species. FS-SINR comprehensively outperforms LE-SINR under the same metadata conditions while running 16-50 times faster during inference.
- **vs Prototypical Networks (Snell et al., 2017)**: The Proto SINR baseline, which simply averages the embeddings of observation positions as the species embedding, performs significantly worse than FS-SINR. This suggests that the attention mechanism of the Transformer learns a more effective aggregation strategy than simple averaging when integrating multiple observation points.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing the feedforward few-shot concept to species distribution modeling is a major breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on two benchmark datasets, comparison with multiple baselines, and extensive ablation studies and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivations, concise method descriptions, and excellent visualizations.
- **Value**: ⭐⭐⭐⭐ Possesses practical application value for ecological conservation, with pronounced advantages in the feedforward paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[ACL 2025\] ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment](../../ACL2025/others/algen_few-shot_inversion_attacks_on_textual_embeddings_via_cross-model_alignment.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](../../ICCV2025/others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[CVPR 2026\] Data-Centric Meta-Learning for Robust Few-Shot Generalization](../../CVPR2026/others/data-centric_meta-learning_for_robust_few-shot_generalization.md)

</div>

<!-- RELATED:END -->
