---
title: >-
  [Paper Note] Evolution of Concepts in Language Model Pre-Training
description: >-
  [ICLR 2026][Interpretability][Mechanistic interpretability] This paper is the first to apply crosscoders (cross-snapshot sparse dictionary learning) to track the emergence and evolution of features during language model…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Mechanistic interpretability"
  - "crosscoders"
  - "sparse autoencoders"
  - "training dynamics"
  - "feature evolution"
  - "pre-training"
  - "Pythia"
date: 2026-05-08
content_hash: 39ed24e24d5930aa
---

# Evolution of Concepts in Language Model Pre-Training

**Conference**: ICLR 2026
**arXiv**: [2509.17196](https://arxiv.org/abs/2509.17196)  
**Code**: [GitHub](https://github.com/OpenMOSS/Language-Model-SAEs)  
**Area**: Interpretability
**Keywords**: Mechanistic interpretability, crosscoders, sparse autoencoders, training dynamics, feature evolution, pre-training, Pythia

## TL;DR

This paper is the first to apply crosscoders (cross-snapshot sparse dictionary learning) to track the emergence and evolution of features during language model pre-training. It identifies a two-phase transition from "statistical learning → feature learning" and causally links micro-level feature evolution to macro-level downstream task metrics through attribution analysis.

## Background & Motivation

- **Pre-training remains a black box**: Although scaling laws reveal macroscopic relationships among compute, data, and loss, the internal reorganization of model parameters remains poorly understood.
- **Limitations of existing theoretical frameworks**: NTK, information bottleneck theory, and singular learning theory provide high-level explanations of generalization and grokking, but cannot answer how models develop their capabilities during pre-training.
- **Static limitations of SAEs**: Sparse autoencoders (SAEs) have been shown to extract interpretable features from fully trained models, yet nearly all analyses target post-training checkpoints; how features emerge and evolve remains unexplored.
- **New opportunity with crosscoders**: Crosscoders, originally proposed by Lindsey et al. for cross-layer feature alignment, are innovatively adapted here for cross-snapshot analysis.

## Method

### Overall Architecture

Crosscoders are applied to simultaneously process model activations from different pre-training checkpoints, aligning features into a unified feature space, thereby tracking the complete evolution of features from initialization to the end of training.

### Key Designs

**1. Cross-Snapshot Crosscoder Architecture**

Given a corpus $\mathcal{C}$ and a set of training checkpoints $\Theta$, the crosscoder encoding–decoding process is:

$$f(x) = \sigma\left(\sum_{\theta \in \Theta} W_{\text{enc}}^{\theta} a^{\theta}(x) + b_{\text{enc}}\right)$$

$$\hat{a}^{\theta}(x) = W_{\text{dec}}^{\theta} f(x) + b_{\text{dec}}^{\theta}$$

Key insight: the encoder aggregates cross-snapshot information to produce shared feature activations $f(x)$, while the decoder $W_{\text{dec}}^{\theta}$ is checkpoint-specific. When a feature exists only in a subset of checkpoints, the sparsity penalty naturally drives the decoder norms of irrelevant checkpoints toward zero.

**Core Observation**: The decoder norm $\|W_{\text{dec},i}^{\theta}\|$ directly reflects the intensity and presence of feature $i$ at checkpoint $\theta$, making it a natural proxy for feature evolution.

**2. Training Objective**

$$\mathcal{L}(x) = \underbrace{\sum_{\theta \in \Theta} \|a^{\theta}(x) - \hat{a}^{\theta}(x)\|^2}_{\text{reconstruction loss}} + \underbrace{\lambda_{\text{sparsity}} \sum_{\theta \in \Theta} \sum_{i=1}^{n_{\text{features}}} \Omega(f_i(x) \cdot \|W_{\text{dec},i}^{\theta}\|)}_{\text{sparsity loss}}$$

The sparsity regularizer $\Omega(\cdot)$ serves as a differentiable surrogate for $L_0$. Including the decoder norm $\|W_{\text{dec},i}^{\theta}\|$ in the regularization term prevents the degenerate solution in which activation values $f_i(x)$ are suppressed while decoder norms inflate under an imperfect $L_0$ approximation.

JumpReLU (with a learned threshold) is chosen as the activation function, outperforming the conventional ReLU + L1 combination.

**3. Experimental Setup**

- Models: Pythia-160M (Layer 6) and Pythia-6.9B (Layer 16)
- 32 checkpoints strategically selected from 154 public snapshots (all 20 snapshots within the first 10K steps + 12 uniformly sampled from later training)
- Feature counts: up to 98,304 (160M) and 32,768 (6.9B)
- Training corpus: SlimPajama

**4. Feature Attribution Analysis**

To connect micro-level features with macro-level behavior, attribution-based circuit tracing is used:

$$\text{attr}_i^{\theta}(x) = f_i(x) \cdot \frac{\partial m(a^{\theta}(x))}{\partial f_i(x)}$$

For tasks with clean/corrupted input pairs (e.g., subject-verb agreement), attribution patching is applied:

$$\text{attr}_i^{\theta}(x, \tilde{x}) = [f_i(x) - f_i(\tilde{x})] \cdot \frac{\partial m(a^{\theta}(x))}{\partial f_i(x)}$$

In practice, an integrated gradients (IG) variant is used to improve the accuracy of the linear approximation.

### Loss & Training

Reconstruction loss (MSE) + weighted sparsity regularization (JumpReLU activation + $L_0$ approximation incorporating decoder norms).

## Key Experimental Results

### Main Results

**Crosscoder Reconstruction Quality (Pythia-160M)**

| # Features | Explained Variance | L0 Norm |
|------------|-------------------|---------|
| 32,768 | ~92% | ~40 |
| 65,536 | ~95% | ~35 |
| 98,304 | ~97% | ~30 |

Increasing dictionary size yields Pareto improvements in both explained variance and sparsity. The Pareto frontier of crosscoders is even slightly superior to SAEs trained on the final checkpoint alone.

**Feature Evolution Patterns**

| Feature Type | Emergence Time | Persistence |
|--------------|---------------|-------------|
| Initialization features | Present at random initialization | Sharp drop at step 128, then recovery, followed by gradual decay |
| Emergent features (simple) | ~step 1,000 | Persist in 60%+ of checkpoints |
| Emergent features (complex) | Steps 10,000–100,000 | Persist in 60%+ of checkpoints |

**Feature Types and Emergence Timing (Pythia-6.9B)**

| Feature Type | Emergence Time Range |
|--------------|---------------------|
| Previous Token features | Steps 1,000–5,000 |
| Induction features | Steps 10,000–100,000 |
| Context-sensitive features | Steps 10,000–100,000 |

### Ablation Study

**Attribution Analysis on Subject-Verb Agreement (SVA Across-PP)**

Key contributing features ordered by emergence time:
1. Features 18341, 47045: capture plural nouns (47045 specializes in plural subjects)
2. Feature 68813: marks compound subjects and postpositive modifiers
3. Features 50159, 69636: identify the end of postpositive modifiers (69636 with higher precision)

Only tens of features are sufficient to consistently disrupt or restore model performance on downstream tasks across all training checkpoints.

### Key Findings

**1. Universal Directional Turning Point**

Nearly all features undergo dramatic directional changes at ~step 1,000, with pre- and post-transition directions nearly orthogonal. Features continue to rotate slowly thereafter, with the final checkpoint directions maintaining significant cosine similarity to early post-step-1,000 directions.

**2. Emergence Step Correlates with Complexity**

Feature complexity is scored (1–5) using an LLM (Claude Sonnet 4), revealing a moderate positive correlation between emergence time and complexity (Pearson $r = 0.309$, $p = 0.002$). More complex features tend to emerge later.

**3. Phase Transition from Statistical Learning to Feature Learning**

| Metric | Early Training | Post-Transition |
|--------|---------------|-----------------|
| Unigram/bigram KL divergence | Rapidly converges to low values | Already converged |
| Training loss | Approaches theoretical unigram/bigram entropy lower bound | Continues to decrease |
| Total feature dimensionality rate | First compresses | Then expands to ~70% |

Early training is almost entirely devoted to learning unigram and bigram distributions (Zipf's law); only afterward does the model enter the superposition learning phase for sparse features.

## Highlights & Insights

1. **Methodological innovation**: The first adaptation of crosscoders from cross-layer analysis to cross-training-snapshot analysis, enabling fine-grained tracking of feature evolution.
2. **Feature-level evidence for the two-phase learning hypothesis**: Changes in unigram/bigram KL divergence and feature dimensionality rate provide feature-level support for the "fitting → compression" two-phase dynamics predicted by information bottleneck theory.
3. **Hierarchical feature emergence**: The emergence order of Previous Token → Induction → context-sensitive features is consistent with causal dependencies.
4. **Micro-to-macro causal linkage**: Only tens of features suffice to explain downstream task performance, and attribution tracing reveals alternating feature dominance (i.e., the model iteratively evolves circuits through component alternation).
5. **Decoder norm as a proxy for feature strength**: This concise observation provides an efficient quantitative tool for tracking feature evolution.

## Limitations & Future Work

1. **Limited model scope**: Validation is restricted to the Pythia suite; although prior evidence suggests feature generality, generalization to different architectures and datasets remains to be confirmed.
2. **Relatively simple downstream tasks**: SVA, Induction, and IOI are basic tasks, constrained by Pythia's capabilities and the current state of circuit tracing methods.
3. **Discrete checkpoint limitation**: Crosscoder training requires activations from discrete checkpoints; memory and compute costs scale linearly with the number of checkpoints, limiting observation granularity.
4. **Moderate complexity correlation**: The Pearson correlation between feature emergence time and complexity is only 0.309, indicating that complexity is not the sole determinant of emergence timing.

## Related Work & Insights

- **Relation to SAE research**: Extends SAEs from static analysis to dynamic tracking; the unified feature space provided by crosscoders is the key enabling technology.
- **Resonance with information bottleneck theory**: The statistical learning → feature learning two-phase transition closely aligns with the experimental findings of Shwartz-Ziv 2017.
- **Relation to grokking research**: The abrupt changes in feature emergence suggest potential connections to phase transitions and grokking.
- **Implications for pre-training optimization**: Knowledge of when features emerge can inform learning rate scheduling, curriculum learning, and other pre-training strategies.
- **Implications for interpretability**: Crosscoders provide causal-level insight into why a model has learned a particular concept.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to achieve cross-training-snapshot feature evolution tracking; significant methodological contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated at multiple scales (160M/6.9B) with both quantitative and qualitative analyses.
- **Value**: ⭐⭐⭐½ Primarily oriented toward understanding and explanation; direct application value is limited but offers meaningful insights for pre-training optimization.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear paper structure, high-quality figures, and sufficient technical detail.
- **Overall**: ⭐⭐⭐⭐½ An excellent contribution to mechanistic interpretability, opening a feature-level observation window into pre-training dynamics for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)
- [\[ICLR 2026\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)

</div>

<!-- RELATED:END -->
