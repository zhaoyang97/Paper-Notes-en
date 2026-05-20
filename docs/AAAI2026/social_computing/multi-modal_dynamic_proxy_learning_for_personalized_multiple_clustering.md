---
title: >-
  [Paper Note] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering
description: >-
  [AAAI 2026][Social Computing][Multiple Clustering] This paper proposes the Multi-DProxy framework, which leverages learnable textual proxies for personalized multiple clustering through three key innovations: gated cross…
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Multiple Clustering"
  - "Cross-modal Fusion"
  - "Proxy Learning"
  - "Dynamic Candidate Management"
  - "CLIP"
date: 2026-05-08
content_hash: 37eabd546d51b709
---

# Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering

**Conference**: AAAI 2026
**arXiv**: [2511.07274](https://arxiv.org/abs/2511.07274)  
**Code**: None (the paper mentions an anonymous code repository in the Supplementary Material)  
**Area**: Social Computing
**Keywords**: Multiple Clustering, Cross-modal Fusion, Proxy Learning, Dynamic Candidate Management, CLIP

## TL;DR
This paper proposes the Multi-DProxy framework, which leverages learnable textual proxies for personalized multiple clustering through three key innovations: gated cross-modal fusion, dual-constraint proxy optimization, and dynamic candidate management, achieving state-of-the-art performance on all public benchmarks.

## Background & Motivation

**Background**: Clustering is a cornerstone of unsupervised learning, aiming to discover latent data structures based on intrinsic similarity. Traditional clustering yields only a single partition, overlooking the inherent complexity that data can be meaningfully grouped from multiple perspectives. Multiple clustering seeks to discover complementary and diverse partitions; however, existing methods exhaustively enumerate all possible clusterings without regard to user interests, forcing users to manually sift through results—a significant practical bottleneck.

**Limitations of Prior Work**: Recent methods such as Multi-MaP and Multi-Sub exploit CLIP for proxy learning, using textual prompts to guide interest-biased embedding extraction. Nevertheless, these approaches suffer from two fundamental deficiencies:

**Static Semantic Rigidity**: Predefined candidate words (e.g., "red," "blue," "green" generated for the concept "color") cannot adapt to dataset-specific concepts, introducing semantic alignment bias when LLM suggestions do not match actual categories.

**Inflexible Feature Fusion**: Fixed fusion strategies (concatenation or simple averaging) ignore the continuously evolving feature interactions between modalities, yielding suboptimal joint representations.

**Key Challenge**: Users expect clustering results aligned with their interests via simple concept keywords (e.g., "color"), yet static textual proxies and fixed fusion strategies fail to capture dataset-specific semantic structure and dynamic cross-modal interactions.

**Key Insight**:
- Replace static candidate words with **learnable** textual proxies
- Employ a **gating mechanism** for adaptive cross-modal fusion
- Use **iterative feedback** to allow the candidate set to evolve dynamically with the clustering structure

## Method

### Overall Architecture
Multi-DProxy builds upon frozen CLIP encoders (visual $f_v(\cdot)$ and textual $f_t(\cdot)$), with the following core pipeline:
1. The user specifies an interest concept $u$ (e.g., "color")
2. GPT-4 generates an initial candidate word set $\mathcal{C}$
3. A learnable proxy $\mathbf{w}_i$ is initialized for each image
4. A joint representation $\mathbf{F}$ is obtained via gated cross-modal fusion
5. Proxy embeddings are optimized under dual constraints
6. Dynamic candidate management updates the candidate set every $R$ epochs
7. K-means clustering is applied to the fused features $\mathbf{F}$ as the final step

### Key Designs

#### 1. **Gated Cross-Modal Fusion**

The core mechanism dynamically synthesizes discriminative joint representations through hierarchical bidirectional attention and adaptive feature recalibration.

**Bidirectional Cross-Attention**: Visual features attend to textual features and vice versa:
$$\mathbf{V}_{\text{attn}}^l = \text{MultiHead}(\mathbf{V}^{l-1}, \mathbf{T}^{l-1}, \mathbf{T}^{l-1})$$
$$\mathbf{T}_{\text{attn}}^l = \text{MultiHead}(\mathbf{T}^{l-1}, \mathbf{V}^{l-1}, \mathbf{V}^{l-1})$$

**Gated Residual Fusion**: A sigmoid gate controls the degree to which attended information is incorporated:
$$\mathbf{V}^l = \mathbf{V}^{l-1} + \sigma(\mathbf{W}_g^{\mathbf{V}}[\mathbf{V}^{l-1}; \mathbf{V}_{\text{attn}}^l]) \odot \mathbf{V}_{\text{attn}}^l$$

**Adaptive Feature Fusion**: Temperature-scaled cosine similarity dynamically balances the contribution of each modality:
$$\mathbf{F} = \lambda \mathbf{T}^L + (1-\lambda)\mathbf{V}^L, \quad \lambda = \sigma\left(\frac{\langle \mathbf{T}^L, \mathbf{V}^L \rangle}{\tau}\right)$$

**Design Motivation**: Fixed fusion strategies (concatenation/averaging) cannot capture sample-wise variations in modality importance. The gating mechanism enables the model to dynamically adjust modality weights based on cross-modal consistency, allowing $\lambda$ to adapt throughout training.

#### 2. **Dual-Constraint Proxy Optimization**

**User Interest Constraint**: Ensures proxies remain semantically consistent with domain concepts. Each proxy is composed via attention-weighted combination of candidate word embeddings:
$$\mathbf{w}_i = \sum_{k=1}^{|\mathbf{C}|} \alpha_{ik} \mathbf{c}_k, \quad \alpha_{ik} = \frac{\exp(\mathbf{w}_i^{\prime \top} \mathbf{c}_k / \tau_\alpha)}{\sum_j \exp(\mathbf{w}_i^{\prime \top} \mathbf{c}_j / \tau_\alpha)}$$

The semantic consistency loss minimizes deviation of proxies from the candidate word centroid:
$$\mathcal{L}_u = \frac{1}{D} \sum_{i=1}^{D} \|\mathbf{w}_i - \bar{\mathbf{c}}\|_2^2$$

**Concept Discriminability Constraint**: Contrastive learning enhances clustering separability:
$$\mathcal{L}_c = \frac{1}{B} \sum_{i=1}^{B} \log \sum_{j \neq i} \exp(\mathbf{f}_i^{\top} \mathbf{w}_j / \sigma)$$

**Design Motivation**: $\mathcal{L}_u$ anchors proxies within a meaningful semantic space (preventing drift), while $\mathcal{L}_c$ enlarges representational distances between different clusters through hard negative mining. The two objectives are complementary: the former ensures semantic relevance, and the latter ensures clustering discriminability.

#### 3. **Dynamic Candidate Management**

An update is performed every $R$ epochs:
1. Collect all proxy embeddings $\mathbf{W}$
2. Apply K-means to the proxies to obtain $M$ cluster centers
3. Compute the average cosine similarity of each candidate word to all cluster centers
4. Retain the Top-K ($K=|\mathcal{C}|/2$) highest-scoring candidates
5. Recompute candidate word embeddings

The initial candidate set contains $2^\beta M$ words ($\beta = E/R$); after $E$ epochs of training, the set converges to $M$ candidates, naturally aligned with the true number of categories.

**Design Motivation**: LLM-generated candidate words may include concepts irrelevant to the dataset. Through iterative alignment with the clustering structure, irrelevant candidates are progressively pruned, retaining only semantically meaningful, dataset-specific concepts.

### Loss & Training
The unified loss function is:
$$\mathcal{L} = \underbrace{\frac{1}{D}\sum_{i=1}^{D}(1-\cos(\mathbf{f}_i, \mathbf{v}_i))}_{\text{Cross-modal Alignment } \mathcal{L}_a} + \alpha(t)\mathcal{L}_u + \beta(t)\mathcal{L}_c$$

Constraint weights follow adaptive scheduling:
- $\alpha(t) = \min(0.5, 0.1 + 0.4 \cdot t/E)$: linear growth, gradually strengthening the semantic constraint
- $\beta(t) = 0.1 \times (1 - \cos(\pi t / E))$: cosine schedule, smoothly intensifying the discriminability constraint

Training runs for 1,000 epochs using the Adam optimizer (momentum 0.9) on an RTX 4090 GPU.

## Key Experimental Results

### Main Results (NMI / RI metrics, higher is better)

| Method | Fruit-Color NMI | Fruit-Species NMI | Card-Suits NMI | CMUface-Identity NMI | CIFAR10-Type NMI |
|------|----------------|-------------------|----------------|---------------------|------------------|
| MSC | 0.6886 | 0.1627 | 0.0497 | 0.3892 | 0.1547 |
| ENRC | 0.7103 | 0.3187 | 0.0676 | 0.5607 | 0.1826 |
| AugDMC | 0.8517 | 0.3546 | 0.0873 | 0.5875 | 0.2855 |
| Multi-MaP | 0.8619 | 1.0000 | 0.2734 | 0.6625 | 0.4969 |
| Multi-Sub | 0.9693 | 1.0000 | 0.3104 | 0.7441 | 0.5271 |
| **Multi-DProxy** | **1.0000** | **1.0000** | **0.5008** | **0.7609** | **0.5863** |

### Ablation Study

| Configuration | Modification | Observation |
|------|---------|------|
| w/o-Dynamic | Remove dynamic candidate management | Performance drops; candidates fail to adapt to data |
| w/o-UConstraints | Remove user interest constraint | Proxies lack semantic anchoring |
| w/o-CConstraints | Remove concept discriminability constraint | Insufficient clustering discriminability |
| w/o-GFusion | Replace gated fusion with concatenation | **Largest performance drop**, confirming the centrality of cross-modal fusion |
| -T (text only) | Use text modality only | Unimodal clustering is feasible but limited |
| -V (visual only) | Use visual modality only | Fused representations substantially outperform unimodal baselines |

### Zero-shot Comparison

| Method | Fruit-Color | Stanford Cars-Color | CIFAR10-Type |
|------|------------|--------------------|----|
| CLIP_GPT (zero-shot) | 0.7912 | 0.6539 | 0.4935 |
| CLIP_label (ground-truth labels) | 0.8629 | 0.6830 | 0.5087 |
| **Multi-DProxy** | **1.0000** | **0.7610** | **0.5863** |

### Key Findings
1. Multi-DProxy consistently surpasses existing state-of-the-art methods across all datasets and clustering dimensions, with substantial improvements in both NMI and RI.
2. Gated cross-modal fusion is the most critical component—its removal results in the largest performance degradation.
3. Even compared to CLIP zero-shot baselines utilizing ground-truth labels, Multi-DProxy achieves superior performance in most settings, demonstrating that the learned representations are more comprehensive.
4. Dynamic candidate management progressively converges from general LLM-generated concepts to dataset-specific semantics, effectively resolving the static rigidity problem.
5. Theoretical analyses establish the stability of proxy updates (Proposition 1) and the mechanism by which visual features gate textual representation learning (Theorem 1).

## Highlights & Insights
1. **The first unified framework integrating learnable proxies, dynamic candidate management, and adaptive fusion for multiple clustering**, with the three innovations mutually reinforcing one another.
2. **Rigorous theoretical analysis**: Proposition 1 quantifies how candidate word updates constrain proxy drift; Theorem 1 reveals how visual features act as gating signals to modulate textual representation learning.
3. **A coarse-to-fine candidate evolution process**: the initial set of $2^\beta M$ candidates is halved every $R$ epochs, naturally converging to $M$ candidates aligned with the true number of categories.
4. **Adaptive weight scheduling** eliminates manual hyperparameter tuning, yielding consistent performance across diverse datasets.

## Limitations & Future Work
1. The framework depends on GPT-4 for initial candidate generation, making candidate quality contingent on LLM capability.
2. Users are required to specify the number of clusters $M$, which may be unknown in real-world applications.
3. Training for 1,000 epochs incurs considerable computational cost.
4. Validation is limited to visual datasets; extension to text or other modalities for multiple clustering remains unexplored.
5. The candidate update interval $R$ requires manual specification; an adaptive adjustment strategy warrants investigation.

## Related Work & Insights
- The proxy learning paradigm is generalizable to other unsupervised tasks requiring alignment with user intent.
- The generate–evaluate–filter loop of dynamic candidate management can inspire other LLM-assisted feature learning approaches.
- The gated cross-modal fusion mechanism offers a useful reference for other multimodal tasks such as retrieval and classification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](../../NeurIPS2025/social_computing/evaluating_multiple_models_using_labeled_and_unlabeled_data.md)

</div>

<!-- RELATED:END -->
