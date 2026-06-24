---
title: >-
  [Paper Note] Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment
description: >-
  [ICML 2025][Image Generation][Preference Modeling] Proposes Preference Embedding—embedding responses into a multi-dimensional latent space to capture complex preference structures (including intransitive preferences), achieving $O(K)$ query complexity (identical to Bradley-Terry models but with significantly higher expressiveness). Combined with General Preference Optimization (GPO), it outperforms Bradley-Terry reward models on RewardBench and AlpacaEval 2.0.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Preference Modeling"
  - "Bradley-Terry"
  - "Preference Embedding"
  - "Intransitive Preferences"
  - "Reward Models"
date: 2026-05-08
content_hash: bec635a436422139
---

# Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment

**Conference**: ICML 2025  
**arXiv**: [2410.02197](https://arxiv.org/abs/2410.02197)  
**Code**: [https://github.com/general-preference/general-preference-model](https://github.com/general-preference/general-preference-model)  
**Area**: Image Generation/LLM Alignment  
**Keywords**: Preference Modeling, Bradley-Terry, Preference Embedding, Intransitive Preferences, Reward Models

## TL;DR
Proposes Preference Embedding—embedding responses into a multi-dimensional latent space to capture complex preference structures (including intransitive preferences), achieving $O(K)$ query complexity (identical to Bradley-Terry models but with significantly higher expressiveness). Combined with General Preference Optimization (GPO), it outperforms Bradley-Terry reward models on RewardBench and AlpacaEval 2.0.

## Background & Motivation

**Background**: Modeling human preferences is core to foundation model alignment. The Bradley-Terry (BT) model represents preferences with scalar rewards and is the standard choice for RLHF. Supervised pairwise preference models (PairRM/PairPM) directly concatenate two responses to predict preferences.

**Limitations of Prior Work**:
   - The BT model assumes preferences can be determined by scalar rewards—unable to express intransitive preferences (e.g., $A>B$, $B>C$ but $C>A$, like rock-paper-scissors-style preference cycles).
   - Although PairRM/PairPM are highly expressive, they require $O(K^2)$ query complexity—evaluating all $K(K-1)/2$ pairs.
   - Human preferences are inherently multi-dimensional and context-dependent—scalar rewards oversimplify this.
   - During LLM test-time scaling, sorting a large number of candidate responses is required—$O(K^2)$ cost is prohibitive.

**Key Challenge**: The trade-off between expressiveness (BT: low, PairPM: high) and efficiency (BT: $O(K)$, PairPM: $O(K^2)$).

**Goal**: To design preference models that possess both high expressiveness and low query complexity.

**Key Insight**: Embedding each response into a $d$-dimensional latent space (instead of a 1-dimensional scalar reward), expressing preferences through the relationships between embedding vectors—the $d$-dimensional embedding is much richer than a scalar, while queries remain $O(K)$ (each response is embedded independently once).

**Core Idea**: Preference Embedding = high-dimensional reward representation, where preference relations are determined by an asymmetric function between embeddings (e.g., $P(A>B) = \sigma(\phi(e_A, e_B))$), maintaining $O(K)$ efficiency while breaking through the expressiveness bottleneck of scalar rewards.

## Method

### Overall Architecture
1. **General Preference Embedding Model (GPM)**: Embeds each response independently into a $d$-dimensional space.
2. **Preference Prediction**: Given two embeddings $e_A, e_B \in \mathbb{R}^d$, computes $P(A \succ B)$ using an asymmetric function.
3. **GPO (General Preference Optimization)**: Policy optimization based on preference scores (rather than scalar rewards).

### Key Designs

1. **Preference Embedding**:

    - **Function**: Maps each (prompt, response) pair to a $d$-dimensional vector.
    - **Mechanism**: Given prompt $x$ and response $y$, the embedder $E(x, y) = e \in \mathbb{R}^d$.
    - **Preference Calculation**: $P(y_1 \succ y_2 | x) = \sigma(e_1^T W e_2)$, where $W$ is a learnable skew-symmetric matrix (ensuring $P(A>B) + P(B>A) = 1$).
    - **Relationship with BT**: Degenerates to the BT model when $d=1$ ($P(y_1 \succ y_2) = \sigma(r_1 - r_2)$).
    - **Query Complexity**: $O(K)$—each response is embedded once, and the subsequent $O(K^2)$ preference calculations only involve simple matrix operations between embeddings without requiring model re-inference.
    - **Design Motivation**: Embeddings are "computed once, reused multiple times"—the real computational bottleneck is the model inference ($O(K)$ times), rather than the preference computation between embeddings.

2. **Modeling Intransitive Preferences**:

    - **Function**: Proves that GPM can precisely model cyclic preferences.
    - **Mechanism**: When $d \geq 2$, there can exist $e_A, e_B, e_C$ in the embedding space such that $A>B$, $B>C$, $C>A$—which is impossible when $d=1$ (BT).
    - **Theoretical Conclusion**: The accuracy of the BT model on cyclic preferences is equivalent to random guessing—mathematically proving the expressiveness limitations of BT.
    - **Design Motivation**: Cyclic phenomena truly exist in human preferences (e.g., A is superior to B in one dimension, but B is superior to A in another).

3. **GPO (General Preference Optimization)**:

    - **Function**: Optimizes LLM policies using preference scores (instead of scalar rewards).
    - **Mechanism**: Replaces scalar rewards in DPO/RLHF with preference embedding scores—$\mathcal{L}_{\text{GPO}} = -\log \sigma(s(y_w) - s(y_l))$, where $s(y) = e_y^T W \bar{e}$ ($\bar{e}$ is the reference direction).
    - **Design Motivation**: The preference signal of GPM is richer than BT rewards $\to$ providing more precise gradients when training LLMs.

### Loss & Training
- GPM Training: Binary cross-entropy loss on preference pairs $(y_w, y_l)$.
- GPO Training: DPO-like loss, but replacing scalar rewards with preference embedding scores.
- Embedding dimension $d$ is typically chosen from 32-128 (much higher than the 1-dimension of BT).
- Base Architecture: Based on LLaMA/Mistral with an added embedding projection head.

## Key Experimental Results

### Main Results
RewardBench Benchmark (Preference Model Evaluation):

| Model | Total Score ↑ | Chat | Safety | Reasoning | Cyclic Preference |
|------|-------|------|------|------|---------|
| BT-RM (Mistral-7B) | 81.2 | 96.3 | 84.2 | 63.1 | ~50% (Random) |
| PairRM | 82.5 | 95.8 | 86.1 | 65.5 | 78.6% |
| **GPM (d=64)** | **84.3** | **96.8** | **87.5** | **68.5** | **96.8%** |

### AlpacaEval 2.0 Downstream Evaluation (Post-GPO Training)

| Training Method | LC Win Rate ↑ |
|---------|-------------|
| DPO + BT-RM | 29.5% |
| DPO + PairRM | 31.2% |
| **GPO + GPM** | **33.8%** |

### Ablation Study

| Configuration | RewardBench Total Score | Cyclic Preference Accuracy |
|------|----------------|-------------|
| d=1 (degenerate to BT) | 81.2% | ~50% |
| d=4 | 82.5% | 82.3% |
| d=16 | 83.5% | 91.5% |
| d=64 | **84.3%** | **96.8%** |
| d=128 | 84.1% | 97.1% |

### Key Findings
- Embedding dimension $d>1$ is a necessary condition for modeling intransitive preferences—BT ($d=1$) is equivalent to random guessing on cyclic preferences (theoretically proven and experimentally verified).
- GPM outperforms BT on both standard (transitive) and cyclic preferences—high-dimensional embeddings not only handle cyclic preferences but are also more accurate on general preferences.
- $d=64$ is a solid choice in practice—scaling further yields diminishing returns.
- GPO outperforms DPO+BT on downstream tasks—richer preference signals $\to$ more precise policy optimization.
- Query efficiency is identical to BT ($O(K)$ inference) but the performance is significantly better.

## Highlights & Insights
- The intuition that **"preferences are multi-dimensional"** has finally been formalized mathematically and verified empirically—humans indeed trade off across multiple dimensions when judging two responses.
- The theoretical result that the BT model is equivalent to random guessing on cyclic preferences is an impressive negative result—revealing a fundamental limitation of a widely used model.
- $O(K)$ efficiency allows GPM to directly replace BT reward models—without any compromise in efficiency.
- Directly valuable for LLM test-time scaling (e.g., Best-of-N sampling)—better ranking $\to$ better final output.
- The idea of preference embedding can be generalized to other fields requiring preference modeling (such as recommendation systems, multi-objective optimization).

## Limitations & Future Work
- The choice and initialization of the skew-symmetric matrix $W$ affect performance.
- Real-world data for cyclic preferences is scarce—primarily validated on synthetic data.
- The training complexity of GPO is comparable to DPO, but requires training GPM first.
- Multi-dimensional embeddings are less interpretable than scalar rewards—making it difficult to directly explain to users why "A is preferred over B."
- Adaptive selection of the embedding dimension has not yet been explored.

## Related Work & Insights
- **vs BT Reward Model**: Scalar representation $\to$ fails on intransitive preferences; GPM multi-dimensional representation $\to$ succeeds.
- **vs PairRM/PairPM**: $O(K^2)$ efficiency $\to$ unscalable; GPM $O(K)$ $\to$ scalable.
- **vs DPO**: Optimizes policy using scalar rewards; GPO uses multi-dimensional preference scores $\to$ richer signals.
- **Insight**: Dimension selection in preference modeling is analogous to embedding dimension selection—1 dimension is too small, too many dimensions lead to overfitting, requiring a sweet spot to be found.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Preference embedding breaks through the fundamental limitations of BT models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ RewardBench + cyclic preferences + downstream AlpacaEval + ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison schematic of BT vs PairPM vs GPM is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Fundamentally advances preference modeling for LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond One-Hot Labels: Semantic Mixing for Model Calibration](beyond_one-hot_labels_semantic_mixing_for_model_calibration.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)
- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](../../CVPR2025/image_generation/diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)

</div>

<!-- RELATED:END -->
