---
title: >-
  [Paper Note] EdgeCAPE：边权预测用于类别无关姿态估计
description: >-
  [ICLR 2026][Human Understanding][Paper Note] EdgeCAPE introduces a learnable **weighted pose graph prediction** mechanism for Category-Agnostic Pose Estimation (CAPE) for the first time. By predicting edge weights and new edges for the skeleton graph, and incorporating Markov Attention Bias to enhance spatial dependency modeling, it achieves SOTA on the MP-100 be
tags:
  - ICLR 2026
  - Human Understanding
date: 2026-05-08
content_hash: bf56c1dc5cbd8fcd
---
# EdgeCAPE: Edge Weight Prediction for Category-Agnostic Pose Estimation

**Conference**: ICLR 2026  
**Paper**: [OpenReview](https://openreview.net/forum?id=f0iKV3cKFE)  
**Code**: https://github.com/orhir/edge_cape  
**Area**: 3D Vision / Pose Estimation / Few-Shot Learning  
**Keywords**: Category-Agnostic Pose Estimation, Graph Prediction, Edge Weight Learning, Graph Transformer  
**Authors**: Or Hirschorn, Shai Avidan (Tel Aviv University)  

## TL;DR

EdgeCAPE introduces a learnable **weighted pose graph prediction** mechanism for Category-Agnostic Pose Estimation (CAPE) for the first time. By predicting edge weights and new edges for the skeleton graph, and incorporating Markov Attention Bias to enhance spatial dependency modeling, it achieves SOTA on the MP-100 benchmark, with a 1.99% Gain over GraphCape in 1-shot scenarios.

## Background & Motivation

**Background**

Pose estimation has evolved from early single-class, fixed keypoint definitions (e.g., human keypoint detection) through traditional methods by Fang et al. (2022) and Cao et al. (2019) to data-driven deep learning. Traditional methods are limited to specific categories and cannot generalize to new objects; this issue is particularly prominent in industrial applications—every new object category requires re-annotation and re-training.

In 2022, Xu et al. first proposed the concept of **Category-Agnostic Pose Estimation (CAPE)**, aiming to allow a single model to locate arbitrary keypoints of any object using only a few support images, breaking the dependence on category-specific training data. Subsequent work like CapeFormer (Shi et al., 2023) improved the initial approach using a DETR-style framework. However, these early works treated keypoints as **isolated entities**, ignoring the importance of object structure.

The turning point occurred in 2024: GraphCape (Hirschorn & Avidan, 2024) first introduced the **skeleton graph** (structural relationships between keypoints) as a prior, utilizing Graph Convolutional Networks (GCN) to propagate information between connected keypoints. This improvement significantly enhanced robustness to occlusion and symmetrical structures. EdgeCAPE takes a critical step forward from this foundation: **from a fixed graph to a learned weighted graph**, marking the upgrade of graphs in category-agnostic vision tasks from static priors to adaptive, instance-specific dynamic constraints.

**Limitations of Prior Work**

Although GraphCape is effective, it relies on **manually defined fixed unweighted graphs**. This presents three deep-seated issues:

1.  **Ambiguity of Graph Definition**: Even for the same object, relationships between keypoints can have multiple reasonable interpretations (e.g., in lower limb bones, should all adjacent leg keypoints be directly connected, or only "immediate neighbors"?). Experiments by Hirschorn & Avidan (2024) show that different graph definitions lead to significant performance variances, and it is difficult for humans to judge which is "optimal."
2.  **Fixed Edge Weights**: Even if the graph topology is correct, different edges should have different **contribution levels** to keypoint localization. For instance, when predicting the right elbow, both the right shoulder and right hand provide context, but their contribution intensities should differ. Traditional category-specific methods can implicitly learn these weights from abundant same-category samples, whereas CAPE cannot do so in category-unknown, data-scarce settings.
3.  **Inapplicability of Existing Methods**: Self-attention methods (e.g., Graph Attention Transformers) perform poorly in category-agnostic settings because they lack explicit structural priors; GCN variants assume fixed pose graphs and are unsuitable for dynamic CAPE scenarios.

**Key Challenge**

The tension between category-agnosticism and structural learning: the model must generalize to unseen object categories while implicitly understanding 3D object structure and anatomical common sense. Predicting the entire graph from scratch requires 3D prior knowledge; however, fixing the graph loses the flexibility to adapt to different objects.

**Goal**

Not to predict the entire pose graph from zero (which requires 3D prior knowledge), but to **start from a user-provided unweighted skeleton graph and learn: (1) edge weights; (2) addition/subtraction of new edges**. This preserves the guidance of manual priors while allowing the model to perform fine-grained adjustments based on specific objects and instances.

**Key Insight**

The authors' core observation: even without knowing the 3D structure of an object, a model can infer effective graph weights from its **image context**. The key is to design a **self-supervised** objective—making the graph prediction module learn to predict edge weights that maximize **occlusion handling capability**. When the graph predicted by the model allows the localization module to accurately locate query points even when support points are occluded, it indicates the graph has captured meaningful structure.

**Core Idea**

A combination of **residual edge prediction** + **self-supervised masking** allows the model to learn weighted skeleton graphs without weight annotations. **Markov Attention Bias** is employed to make the self-attention mechanism respect the predicted graph structural distances, thereby better modeling spatial dependencies across multi-hop keypoints.

## Method

### Overall Architecture

EdgeCAPE extends the graph-based CAPE framework of GraphCape. The core process is divided into two parallel stages:

1.  **Pose Graph Prediction Stage**: Receives the user-provided unweighted skeleton graph $A_{\text{prior}}$, support images, and keypoint features to predict a residual edge matrix $\Delta A$, subsequently generating the weighted skeleton graph $\tilde{A}$;
2.  **Enhanced Graph Transformer Localization Stage**: Utilizes the predicted weighted graph to precisely locate keypoints on query images through a self-attention mechanism integrated with Markov Attention Bias.

The entire framework allows the model to **adapt to instance-specific** structural variations (e.g., if a certain edge is unimportant for localization, the model can weaken its weight) while maintaining generalizability to new categories.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Support Image<br/>Support Keypoints<br/>Unweighted Graph A_prior"] --> B["Bidirectional Attention Graph Decoder<br/>Fusing Image & Keypoint Features"]
    B --> C["Residual Edge Prediction<br/>Calculating ΔA"]
    C --> D["Weighted Graph Fusion & Normalization<br/>Generating Ã"]
    D --> E["Query Image Feature Extraction"]
    E --> F["Enhanced Graph Transformer<br/>with Markov Attention Bias"]
    F --> G["Keypoint Localization Output"]
    
    B -.Self-supervised Chain.--> H["Masking Loss<br/>Maximizing Occlusion Robustness"]
    H -.Backpropagation.-> C
    D --> F
    
    style B fill:#fffacd
    style C fill:#fffacd
    style D fill:#fffacd
    style F fill:#ffe4b5
```

### Key Designs

**1. Bidirectional Attention Graph Decoder: Structural Refinement via Image-Keypoint Fusion**

The original GraphCape graph transformer decoder only interacted between keypoint features. EdgeCAPE adds a **cross-attention layer**, allowing support image features and keypoint features within the same image to enhance each other. The motivation for this improvement stems from a key observation: in category-agnostic settings, object orientation and appearance vary greatly; inferring skeleton relationships is difficult from local keypoint features alone, but combining global image context (e.g., object contours, keypoint spatial layout) enables a better understanding of structure.

Specifically, the module first processes keypoint features $F_s^k$ with a standard self-attention layer, then adds a cross-attention layer to update image features $F_s$, refining both into $F_{\text{refined}}^k$ and $F_{\text{refined}}^{\text{img}}$. This bidirectional interaction allows the model to learn richer structural representations, providing higher-quality features for subsequent edge weight prediction.

**2. Residual Edge Prediction and Adaptive Graph Fusion: Incremental Learning from Priors**

The core formula is:
$$\Delta A_{ij} = \langle F_i^{\text{refined}}, F_j^{\text{refined}} \rangle$$

Cosine similarity is used to calculate the "closeness" of all keypoint pairs, resulting in the residual matrix $\Delta A$. The subsequent fusion is clever: instead of directly adding $A_{\text{prior}} + \Delta A$ (which would cause instability in early training), a **learnable scaling factor $c$** is introduced, initialized to zero:

$$A' = \text{ReLU}(A_{\text{prior}} + c \cdot \Delta A)$$

In early training, $c \approx 0$, and the output graph is simply the prior graph, providing a stable foundation for the model. As training progresses, $c$ increases, and the model gradually "trusts" the learned residual adjustments. This warm-start strategy from zero is critical for training stability. Finally, symmetry and normalization are applied to ensure an effective random walk matrix:

$$A = \frac{A' + A'^T}{2}, \quad \tilde{A}_{ij} = \frac{A_{ij}}{\sum_j A_{ij}}$$

This design solves the "no labels for graph weights" problem: the model does not learn weights under a supervised target but automatically discovers useful structures in an **unsupervised occlusion handling task**.

**3. Self-supervised Masking Strategy: Learning Effective Graphs via Occlusion Robustness**

Key question: Without "ground truth optimal graph" labels, how can the model be guided to predict a good graph? The authors' answer: **Let the graph prove its value by handling random occlusions.**

Specifically: randomly mask a portion of the support keypoint features, replacing them with a learnable mask token $F_{\text{mask}}$. Use the corrupted keypoint features and the predicted graph $\tilde{A}$ to locate keypoints on the query image. The localization loss is:

$$L_{\text{adj}} = \sum_{i=1}^K |P_i^m - \hat{P}_i|$$

where $P_i^m$ is the predicted position under masking conditions and $\hat{P}_i$ is the ground truth position. During training, the localization decoder is frozen, and only $\tilde{A}$ is updated, forcing the graph prediction module to find edge weights that "compensate" for the occlusion. This is equivalent to saying: **if an edge weight is high, then when the corresponding keypoint is occluded, the graph will propagate useful information through other connected keypoints to recover the localization.** This approach is both elegant and practical—automatically encouraging the model to learn to handle common occlusion problems in real-world scenarios.

Final Loss:
$$L = L_{\text{offset}} + \lambda_{\text{adj}} L_{\text{adj}}$$

**4. Markov Attention Bias: Graph Distance-Driven Self-Attention Regulation**

The power of self-attention lies in global connectivity, which also means it tends to ignore local graph structures. The goal of Markov Attention Bias is to make self-attention respect the graph structure while maintaining its global expressive power.

Method: Treat the predicted weighted graph $\tilde{A}$ as a transition matrix for a random walk process. Calculate multi-hop adjacency matrices $\tilde{A}^k$ ($k$ from 1 to $K-1$), where $(\tilde{A}^k)_{ij}$ represents the probability of reaching $j$ from node $i$ in exactly $k$ steps. Assemble these into multi-hop features:

$$P_{ij} = [I, \tilde{A}, \tilde{A}^2, \ldots, \tilde{A}^{K-1}]_{ij} \in \mathbb{R}^K$$

Then, use an MLP to learn a scalar bias term from the multi-hop vector to add to the self-attention calculation:

$$a_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + \text{MLP}(P_{ij})$$

Intuitively, this allows the model to assign higher attention weights to keypoint pairs that are "close on the graph" while still permitting long-range interactions. Since the bias is learned, the model can flexibly decide which graph distance information is helpful for localization.

### Loss & Training

Training uses a weighted combination of two losses:

$$L = L_{\text{offset}} + \lambda_{\text{adj}} L_{\text{adj}}$$

$L_{\text{offset}}$ is the standard L1 localization loss (inherited from CapeFormer); $L_{\text{adj}}$ is the self-supervised adjacency loss. The key training strategy is to **freeze all weights of the localization decoder** when calculating the adjacency loss, allowing gradients to backpropagate only to $\tilde{A}$, forcing the graph prediction module to learn to encode information regarding "occlusion handling capability." This constraint prevents the graph prediction module from "free-riding" (adjusting decoder weights instead of the graph), ensuring that genuine structural learning occurs.

## Key Experimental Results

### Main Results

On the MP-100 benchmark (100 object categories, 20K+ images):

| Method | Graph Prior | mPCK@1-shot | mPCK@5-shot | PCK@0.2 (1-shot) |
|------|--------|------------|------------|-----------------|
| CapeFormer | None | 74.45 | 79.63 | 87.27 |
| GraphCape | Fixed Unweighted | 75.73 | 80.16 | 88.50 |
| EdgeCAPE (**Ours**) | Learned Weighted | **77.72** | **81.07** | **89.42** |
| **Gain** over GraphCape | - | **+1.99%** | **+0.91%** | **+0.92%** |
| **Gain** over CapeFormer | - | **+3.27%** | **+1.44%** | **+2.15%** |

(mPCK is the average of PCK at multiple thresholds, a stricter evaluation metric; PCK@0.2 is the standard metric but is reaching saturation).

Notably, when the evaluation threshold is lowered (e.g., PCK@0.05), the advantage over GraphCape is more pronounced: a **+4.0%** gain in the 1-shot scenario, indicating that EdgeCAPE's predictions are more precise, not just generally correct.

### Ablation Study

**Table 1: Contribution of Core Components**

| Graph Prediction Module | Markov Bias | mPCK (1-shot Split 1) |
|---------|----------|----------------------|
| ✗ | ✗ | 79.87 (GraphCape baseline) |
| ✓ | ✗ | 80.76 |
| ✗ | ✓ | 80.78 |
| ✓ | ✓ | **81.96** |

Both components yield gains (each ~+0.9%), with a synergistic effect (+2.09%) when combined, indicating that edge weight prediction and Markov bias enhance each other.

**Table 2: Robustness to Noisy Skeleton Graphs**

Randomly adding 0-16 fake edges to $A_{\text{prior}}$ resulted in the following performance drops:

| Number of Random Fake Edges | GraphCape mPCK | EdgeCAPE mPCK | Gain |
|----------|------------|----------|-----|
| 0 (Clean) | 80.16 | 81.96 | +1.80% |
| 4 | 77.12 | 79.68 | +2.56% |
| 8 | 72.89 | 76.42 | +3.53% |
| 16 | 63.45 | 70.21 | +6.76% |

EdgeCAPE's performance drops more slowly in the face of noise, proving that its graph prediction module has indeed learned to **denoise and correct** the user-provided prior, which is highly valuable in real applications where users struggle to define perfect graphs.

## Highlights & Insights

**Evolution of CAPE Research**

GraphCape's breakthrough was **introducing structural priors**, but it assumed graphs were fixed with equal edge weights. This assumption is acceptable in category-specific scenarios (where skeletal anatomy is constant) but has fundamental issues in category-agnostic settings: the same skeletal connection may have different "importance" on different objects. For example, in humans, the shoulder-elbow connection is always vital; but for certain animals or furniture, the "strongest" structural constraint might be elsewhere. A fixed graph cannot adapt.

**The Progression of Core Ideas**

*   GraphCape: "Use graph structure to encode keypoint relationships."
*   EdgeCAPE: "Use weighted graphs to encode the relative contribution intensity of keypoints."
*   Implicit next step: "Use dynamic graphs to encode instance-specific structural variations."

EdgeCAPE's core insight—**self-supervising graph quality through occlusion handling capability**—opens a new dimension: it no longer asks "what graph is mathematically optimal" but "what graph makes the model best at overcoming real-world challenges (occlusion, symmetry, ambiguity)."

## Limitations & Future Work

1.  **Diminished Advantage in High-Data Environments**: In 5-shot settings, EdgeCAPE's gain over GraphCape decreases from +1.99% (1-shot) to +0.91%; this is expected (graph-based methods naturally suit low-data regimes), but indicates that simple point-based methods might suffice when data is abundant.
2.  **Interpretability of Predicted Graphs**: Why does the model change edge weights the way it does? Beyond qualitative examples, there is a lack of systematic analysis. Do certain weight patterns generalize across all objects?
3.  **Computational Overhead**: Although the paper claims only a ~2ms increase in overhead, this was measured on specific hardware (A5000 GPU). Costs on mobile or edge devices need evaluation.
4.  **Sparse Graph Assumption**: The method assumes connections between keypoints are **sparse** (the user's prior graph is sparse). If handling fully connected or nearly fully connected graphs, computational complexity would surge.

## Related Work & Insights

**Comparison with SDPNet**

SDPNet (Ren et al., 2024) also attempts to predict adjacency matrices, but it does so directly based on keypoint self-attention, supervised by an auxiliary GCN and mask reconstruction task. The issue is that the GCN is only used during training and discarded during testing, which may cause structural information to be encoded in the GCN weights rather than the adjacency matrix itself. EdgeCAPE improves on this by: (1) predicting directly from graph and keypoint features rather than intermediate self-attention representations; (2) supervising with **feedback from the localization module** rather than auxiliary tasks, optimizing graph utility more directly.

**Comparison with AutoLink (He et al., 2022)**

AutoLink also explores weighted pose graph learning in category-specific scenarios but is limited by: (1) requiring fixed categories and abundant same-category training data; (2) constant skeletal topology. EdgeCAPE's novelty lies in breaking these two limitations for true category-agnostic, dynamic topology settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?](emoprefer_can_large_language_models_understand_human_emotion_preferences.md)
- [\[ICLR 2026\] Disentangled Hierarchical VAE for 3D Human-Human Interaction Generation](disentangled_hierarchical_vae_for_3d_human-human_interaction_generation.md)
- [\[ICLR 2026\] Sapiens2：面向人体视觉的高分辨率基础模型](sapiens2.md)
- [\[ICLR 2026\] Motion-Aligned Word Embeddings for Text-to-Motion Generation](motion-aligned_word_embeddings_for_text-to-motion_generation.md)
- [\[ICLR 2026\] SpeakerVid-5M: A Large-Scale High-Quality Dataset for Audio-Visual Dyadic Interactive Human Generation](speakervid-5m_a_large-scale_high-quality_dataset_for_audio-visual_dyadic_interac.md)

</div>

<!-- RELATED:END -->
