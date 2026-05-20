---
title: >-
  [Paper Note] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering
description: >-
  [NeurIPS 2025][Reinforcement Learning][interpretability] This paper proposes a Semantic Clustering Module (SCM) that combines a Feature Dimensionality Reduction (FDR) network with an adapted online VQ-VAE clustering mech…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "interpretability"
  - "semantic clustering"
  - "VQ-VAE"
  - "dimensionality reduction"
  - "Procgen"
date: 2026-05-08
content_hash: cc8dcf72b930a7d1
---

# Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2409.17411](https://arxiv.org/abs/2409.17411)  
**Code**: [github.com/ualiangzhang/semantic_rl](https://github.com/ualiangzhang/semantic_rl)  
**Area**: Reinforcement Learning
**Keywords**: interpretability, semantic clustering, VQ-VAE, dimensionality reduction, Procgen

## TL;DR
This paper proposes a Semantic Clustering Module (SCM) that combines a Feature Dimensionality Reduction (FDR) network with an adapted online VQ-VAE clustering mechanism, seamlessly integrated into the DRL training pipeline. The approach addresses the instability of t-SNE visualization and demonstrates that DRL inherently exhibits dynamic, semantics-based clustering behavior.

## Background & Motivation

**Background**: Deep reinforcement learning (DRL) has been widely applied in robotics, games, and related domains, yet its "black-box" decision-making process lacks interpretability. Semantic clustering has been thoroughly studied in NLP (Word2Vec, GloVe) and CV (image feature spaces), but remains underexplored in DRL.

**Limitations of Prior Work**: (a) Prior work (Mnih et al. 2015, Zahavy et al. 2016) analyzed DRL features using t-SNE only on fixed-scene Atari games, making it impossible to distinguish whether clusters arise from pixel similarity or semantic understanding; (b) t-SNE results are unstable—highly sensitive to initialization, random seeds, and sample size; (c) t-SNE lacks an automatic clustering mechanism and requires substantial manual annotation.

**Key Challenge**: Understanding the internal semantic organization of DRL models requires stable, automated clustering methods, yet existing visualization tools do not satisfy these requirements.

**Goal**: (a) Verify whether DRL inherently possesses semantic clustering capability, ruling out pixel-similarity confounds; (b) provide a stable alternative to t-SNE as an analytical tool; (c) develop policy analysis methods grounded in clustering results.

**Key Insight**: Procgen is used instead of Atari—its procedurally generated levels ensure scene diversity, so if clustering persists, it reflects genuine semantic understanding rather than pixel memorization. The discrete encoding mechanism of VQ-VAE is repurposed as an online clustering tool.

**Core Idea**: The codebook of an adapted VQ-VAE serves as online clustering centroids, trained end-to-end with the DRL agent to simultaneously improve clustering quality and maintain policy performance.

## Method

### Overall Architecture
A Semantic Clustering Module (SCM) is inserted after the feature extractor $f$ of a standard DRL agent (e.g., PPO). The SCM consists of two components: (1) an FDR network $g$ that projects high-dimensional features to 2D; (2) a VQ quantizer that assigns each 2D feature to the nearest codebook embedding (i.e., cluster). The VQ index $k$ is expanded in dimension and added back to the original feature, enabling conditioned policy training $\pi(a|s,k)$.

### Key Designs

1. **Feature Dimensionality Reduction (FDR) Network**:

    - **Function**: Learns a stable mapping from high-dimensional features to 2D space, replacing t-SNE.
    - **Mechanism**: The FDR network $g$ is trained to preserve pairwise distances from the high-dimensional space in the low-dimensional space. Pairwise similarities are computed using the Student's t-distribution:
    $p_{ij} = \frac{d(i,j)}{\sum_{k\neq l}d(k,l)}, \quad d(m,n) = \left(1+\frac{\|f(\mathbf{s}_m)-f(\mathbf{s}_n)\|^2}{\alpha}\right)^{-\frac{\alpha+1}{2}}$
    The similarity $q_{ij}$ for FDR features follows the same formula with $g \circ f$ replacing $f$. The FDR loss is the cross-entropy: $\mathcal{L}_{\text{FDR}} = -\sum_i\sum_j p_{ij}\log(q_{ij})$
    - **Design Motivation**: t-SNE is unstable because its non-convex objective is sensitive to initialization. The FDR network, once trained, yields a deterministic mapping invariant to random seeds or sample size.
    - **Key Difference from t-SNE**: The same degrees-of-freedom parameter $\alpha$ is used for both the high-dimensional and low-dimensional distributions, ensuring that original distance relationships are strictly preserved rather than only local neighborhoods.

2. **Adapted VQ-VAE Online Clustering**:

    - **Function**: Automatically assigns FDR features to discrete clusters.
    - **Mechanism**: Only the second term of the VQ-VAE loss (the embedding update term) is retained, allowing codebook embeddings to serve as online k-means centroids: $\mathcal{L}'_{\text{VQ-VAE}} = \|sg[g(f(\mathbf{s}))] - \mathbf{e}_k\|_2^2$, where $\mathbf{e}_k$ is the nearest codebook embedding.
    - **Design Motivation**: (a) The reconstruction term (observation reconstruction is unnecessary) and the commitment loss (replaced by $\mathcal{L}_{\text{FDR}}$) are discarded; (b) the stop-gradient operator prevents the clustering objective from directly pulling FDR features, instead allowing joint training via $\mathcal{L}_{\text{FDR}}$ to indirectly improve cluster compactness.

3. **Conditioned Policy Training**:

    - **Function**: Injects clustering information into the policy.
    - **Mechanism**: The VQ index $k$ is expanded to match the feature dimension as $\mathbf{k}^{\text{expand}} = \text{expand}(k, \dim(\mathbf{f}))$, then added element-wise to the original feature: $\mathbf{f}^{\text{fused}} = \mathbf{f} + \mathbf{k}^{\text{expand}}$. Both the policy and value function are conditioned on the fused feature.
    - **Design Motivation**: Enables the policy to leverage clustering information, laying the groundwork for hierarchical learning (e.g., using clusters as the basis for macro-action selection).

4. **Adaptive Control Factor $\lambda_{\text{ctrl}}$**:

    - **Function**: Reduces the SCM loss weight during early training.
    - **Mechanism**: $\lambda_{\text{ctrl}}$ is dynamically adjusted based on training performance; its value is decreased in early stages when semantic distributions are not yet clear, avoiding interference with policy learning.
    - **Design Motivation**: Effective semantic clustering depends on a well-formed semantic distribution, which has not yet emerged in early training.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DRL}} + \lambda_{\text{ctrl}}(w_{\text{FDR}}\mathcal{L}_{\text{FDR}} + w_{\text{VQ-VAE}}\mathcal{L}'_{\text{VQ-VAE}})$

PPO parameters $\theta$, FDR parameters $\phi$, and codebook embeddings $\{\mathbf{e}_k\}_{k=1}^K$ are jointly optimized. $K=8$ clusters are used.

## Key Experimental Results

### t-SNE vs. FDR Stability Comparison

| Comparison Dimension | t-SNE | FDR (Ours) |
|----------------------|-------|------------|
| Changing random seed | Visualization structure changes drastically | Mapping remains completely unchanged |
| Reducing samples by 50% | Distribution structure changes | Only density decreases; spatial distribution unchanged |
| Cluster boundaries | No clear boundaries; requires manual inspection | Clearly separated clusters |
| Automatic clustering | Not supported | VQ encoding assigns automatically |
| Analysis mode | Static visualization | Supports dynamic policy segmentation |

### Human Evaluation Results (5-point Likert Scale)

| Evaluation Statement | Jumper | FruitBot | Ninja |
|----------------------|--------|----------|-------|
| Each cluster consistently demonstrates the same skill | 4.24 (±0.15) | 4.10 (±0.11) | 4.30 (±0.15) |
| Clusters match the provided skill descriptions | 4.36 (±0.16) | 4.16 (±0.11) | 4.20 (±0.17) |
| Identified skills help understand the decision-making process | 4.50 (±0.22) | 4.10 (±0.18) | 4.20 (±0.20) |

All ratings exceed 4.0 (15 evaluators), indicating that human participants broadly agree that DRL exhibits meaningful semantic clustering properties.

### Cluster Semantic Analysis Example (Ninja, 8 Clusters)

| Cluster | Semantic Description |
|---------|----------------------|
| 0 | Walking on the first platform, then performing a high jump to a higher platform |
| 1 | Performing small jumps in the middle of the scene |
| 2 | Walking from the far left to the starting position / walking in preparation when no higher platform is present |
| 3 | Moving on a platform in preparation for jumping to a higher platform |
| 4 | After a high jump, losing sight of the platform below |
| 5 | Moving on a platform in preparation for jumping to a platform at the same or lower level |
| 6 | Mid-high jump while keeping the platform below visible |
| 7 | Moving to the right side of the scene and touching the mushroom |

### Key Findings
- **Semantic clustering is an intrinsic property of DRL**: Without external constraints such as bisimulation or contrastive learning, DRL models inherently organize the feature space along semantic lines.
- **Dynamic rather than static clustering**: Unlike semantic clustering in NLP/CV—which operates on individual inputs—DRL semantic clustering is temporal: consecutive state sequences are assigned to the same cluster and can be described in natural language (analogous to "skills").
- **Beyond pixel distance**: In Procgen's procedurally generated levels, visually dissimilar states are grouped together due to shared semantics, demonstrating that clustering is grounded in semantic understanding rather than pixel similarity.
- **Discovery of policy hierarchy**: Episode-level cluster segmentation reveals hierarchical structure in the policy (e.g., the switch from cluster 5 to cluster 7 is driven by right-wall detection rather than mushroom appearance).
- **Negligible impact on policy performance**: Integrating SCM leaves PPO game performance essentially unchanged.

## Highlights & Insights
- **Elegant repurposing of VQ-VAE**: The quantization mechanism of a generative model is reinterpreted as a clustering tool—the decoder and reconstruction loss are removed, retaining only the embedding update term. This "subtractive innovation" is both concise and effective.
- **Clusters as naturally discovered skills**: Each cluster corresponds to a human-interpretable behavioral phase ("walk to platform edge," "high jump," "touch mushroom"), suggesting that clusters can be directly used as options or macro-actions in hierarchical RL.
- **Practical utility of visualization tools**: An interactive hover tool and dynamic episode segmentation provide a new analytical paradigm for DRL debugging—observing cluster transition points can surface policy errors (e.g., the erroneous right-wall detection case in Figure 6c).

## Limitations & Future Work
- **Choice of cluster count**: A fixed $K=8$ is used; adaptive methods for determining the number of clusters (e.g., elbow method, silhouette score) are not explored.
- **Dependence on clear semantic distributions**: When the policy is far from optimal, semantic distributions are ambiguous and clustering becomes unstable.
- **Manual semantic labeling**: Semantic labels for clusters still require human annotation (approximately 15 minutes per environment); the authors plan to automate this using GPT-4V.
- **Evaluation limited to Procgen**: Validation in continuous control or more complex tasks (e.g., StarCraft) has not been conducted.
- **Choice of clustering affinity**: The current approach uses Student's t-distribution pairwise similarities; cosine similarity or bisimulation metrics could be explored as alternatives.

## Related Work & Insights
- **vs. Mnih et al. (2015, DQN t-SNE)**: That work identified feature clusters in fixed-scene Atari games but could not rule out the pixel-similarity hypothesis. This paper uses Procgen's procedural generation to eliminate this confound.
- **vs. Zahavy et al. (2016, Graying the Black Box)**: That approach required manually defined features for specific games, incurring high annotation cost. The end-to-end method proposed here accomplishes this automatically.
- **vs. PW-Net/DIGR and other interpretability methods**: These focus on single-frame decision explanations (prototypes/saliency maps), whereas this paper reveals temporal semantic structure—closer to a holistic human understanding of "policy."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The repurposing of VQ-VAE as a clustering tool is elegant and concise; the systematic study of semantic clustering in DRL fills a clear gap.
- **Experimental Thoroughness**: ⭐⭐⭐ Analysis across three Procgen games combined with human evaluation is reasonably sufficient, but continuous control and complex multi-agent scenarios are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Rich figures, excellent visualizations, and clear methodological exposition.
- **Value**: ⭐⭐⭐⭐ An analytical tool that reveals the intrinsic semantic organization of DRL, with implications for both interpretable RL and hierarchical RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning](stair_addressing_stage_misalignment_through_temporal-aligned_preference_reinforc.md)
- [\[NeurIPS 2025\] Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization](learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)

</div>

<!-- RELATED:END -->
