---
title: >-
  [Paper Note] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals
description: >-
  [ICLR 2026][Video Understanding][RAG hallucination detection] This paper proposes the Lumina framework for detecting hallucinations in RAG systems via "context-knowledge signals": MMD is used to measure **external contex…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "RAG hallucination detection"
  - "external context utilization"
  - "internal knowledge utilization"
  - "maximum mean discrepancy"
  - "information processing rate"
date: 2026-05-08
content_hash: 2031a278b6f291da
---

# LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals

**Conference**: ICLR 2026
**arXiv**: [2509.21875](https://arxiv.org/abs/2509.21875)  
**Code**: [Available](https://github.com/deeplearning-wisc/LUMINA)  
**Area**: Video Understanding
**Keywords**: RAG hallucination detection, external context utilization, internal knowledge utilization, maximum mean discrepancy, information processing rate

## TL;DR

This paper proposes the Lumina framework for detecting hallucinations in RAG systems via "context-knowledge signals": MMD is used to measure **external context utilization**, while cross-layer token prediction evolution measures **internal knowledge utilization**, enabling hyperparameter-free generalization.

## Background & Motivation

RAG systems aim to reduce LLM hallucinations through retrieved external documents; however, hallucinations persist even when sufficient and relevant context is provided.

Root cause: an imbalance between internal parametric knowledge and external context—hallucinations arise when the model over-relies on its internal knowledge while neglecting the retrieved context.

Existing methods (e.g., ReDeEP, SEReDeEP) have validated the "internal-external knowledge utilization" direction, but suffer from two key limitations:

**Heavy hyperparameter dependence**: specific attention heads and transformer layers must be selected for score computation, requiring extensive tuning, with parameters varying across datasets and models.

**Lack of validation**: while correlations between scores and hallucinations are demonstrated, it is not verified whether the scores genuinely reflect the degree of external context / internal knowledge utilization.

## Method

### Overall Architecture

Lumina's core assumption (Conjecture 1): when $\mathcal{I}_{p_\theta}(a|q,d) \gg \mathcal{E}_{p_\theta}(a|q,d)$ (internal knowledge utilization far exceeds external context utilization), the response is more likely to be a hallucination.

The token-level hallucination score is defined as:

$$\mathcal{H}_t(a_t|q,d,a_{<t}) = \lambda \cdot \mathcal{I}_{p_\theta}(a_t|q,d,a_{<t}) - (1-\lambda) \cdot \mathcal{E}_{p_\theta}(a_t|q,d,a_{<t})$$

The response-level score is the mean of token-level scores: $\mathcal{H}_r(a|q,d) = \frac{1}{T}\sum_{t=1}^{T} \mathcal{H}_t$

### Key Designs

#### 1. External Context Utilization Measurement (MMD Method)

Core Idea: if the LLM effectively utilizes external context, replacing the relevant document with a random one should significantly alter the token probability distribution.

Two distributions are defined:
- $P(E_v) = p_\theta(v|q,d,a_{<t})$ (token probability distribution conditioned on retrieved document)
- $Q(E_v) = p_\theta(v|q,d',a_{<t})$ (token probability distribution conditioned on random document)

Maximum Mean Discrepancy (MMD) measures the distance between the two distributions:

$$\mathcal{E}_{p_\theta}(a_t|q,d,a_{<t}) = \text{MMD}_k^2(P, Q)$$

Expanded as a kernel function computation in token embedding space:

$$\mathcal{E} = \sum_{u,v} P(E_u)P(E_v)k(E_u,E_v) + \sum_{u,v} Q(E_u)Q(E_v)k(E_u,E_v) - 2\sum_{u,v} P(E_u)Q(E_v)k(E_u,E_v)$$

The cosine kernel $k_{\cos}(E_u, E_v) = \frac{1}{2}(1 + \frac{E_u^T E_v}{\|E_u\|_2 \|E_v\|_2})$ is adopted.

Advantage: non-parametric and LLM-agnostic, requiring no selection of specific attention heads or layers.

#### 2. Internal Knowledge Utilization Measurement (Information Processing Rate)

Core Idea: using logit lens to project each layer's hidden states into token probability space, the cross-layer evolution of predictions is tracked. If intermediate-layer predictions converge to the final output only in later layers, it indicates that the model "adds more information" across layers, i.e., relies more heavily on internal knowledge.

**Information Processing Rate** is defined as:

$$\mathcal{R}_{p_\theta}(x_{<t}) = \frac{\sum_{l=1}^{L-1}(1 - \min\{\frac{[f(h_{t,l})]_{x_{t,1}}}{p_\theta(x_{t,1}|x_{<t})}, 1\}) \cdot l}{\sum_{l'=1}^{L-1} \frac{l'}{H(f(h_{t,l'}))}}$$

where $f(\cdot) = \text{Softmax}(\text{LogitLens}(\cdot))$ and $H(\cdot)$ denotes the entropy function.

- **Numerator**: measures the degree of "non-convergence" of each layer toward the final prediction, weighted by layer depth (emphasizing later-layer processing)
- **Denominator**: adaptive normalization based on prediction entropy (assigning higher weight to layers with more confident predictions)

#### 3. Statistical Validation Framework

Four verifiable implications are proposed to validate the soundness of the measurements:
- **H1**: generation with retrieved documents should exhibit higher external context utilization than generation without documents
- **H2**: summarization tasks should exhibit higher external context utilization than QA tasks
- **H3**: generation without retrieved documents should require more internal knowledge than generation with documents
- **H4**: data-to-text generation should require more internal knowledge than summarization

All hypotheses pass the test at $p < 0.001$ across four LLMs.

### Loss & Training

Lumina is an **unsupervised method** requiring no training. Key hyperparameters:
- $\lambda = 0.5$ (balancing external and internal scores)
- Cosine kernel (no kernel parameter tuning required)

## Key Experimental Results

### Main Results

Datasets: RAGTruth (QA + summarization + data-to-text), HalluRAG (free-form QA). Models: Llama2-7B/13B, Llama3-8B, Mistral-7B.

| Category | Method | RAGTruth AUROC (Llama2-13B) | HalluRAG AUROC (Llama2-13B) |
|---------|------|:-:|:-:|
| Uncertainty | Perplexity | 0.454 | 0.255 |
| Uncertainty | LN-Entropy | 0.768 | 0.783 |
| Cross-sample Consistency | EigenScore | 0.633 | 0.786 |
| Verbalization | P(True) | 0.754 | 0.691 |
| Utilization Metric | ReDeEP | 0.806 | 0.765 |
| **Utilization Metric** | **Lumina** | **0.857** | **0.917** |

| LLM | RAGTruth AUROC | HalluRAG AUROC |
|-----|:-:|:-:|
| Llama2-7B | 0.765 | 0.915 |
| Llama2-13B | 0.857 | 0.917 |
| Mistral-7B | 0.769 | 0.990 |

Lumina achieves over 0.9 AUROC on HalluRAG across all models, improving over ReDeEP by up to +13%.

### Ablation Study

- **Kernel selection**: the cosine kernel performs comparably to the optimal RBF kernel while being parameter-free and more practical
- **Score combination**: combining external and internal scores outperforms either alone; on Llama2-13B, the joint score improves by >10% over individual scores
- **Robustness to context noise**: removing/adding 0–30% of sentences, performance remains stable for most LLMs
- **Cross-model detection**: Lumina using Llama2-7B to detect hallucinations generated by Llama3-8B achieves AUROC on par with or higher than Llama3-8B self-detection

### Key Findings

- Hallucinations are strongly correlated with "low external context score + high internal knowledge score" (verified via 2D KDE visualization)
- Same-model detection is not necessary—cross-model detection is equally effective or better
- Error analysis reveals that most false positives/negatives stem from dataset annotation quality and low-quality retrieved documents

## Highlights & Insights

1. **Layer-agnostic design**: eliminates the need to select specific attention heads or layers, resolving the primary portability bottleneck of prior methods
2. **Statistical validation framework**: the first work to rigorously validate "internal-external knowledge utilization scores" via hypothesis testing
3. **Unsupervised yet competitive with supervised methods**: achieves competitive performance against trained binary classifiers (SAPLMA), surpassing them in some settings
4. **Cross-model generalization**: enables small models to detect hallucinations in large models, substantially reducing deployment costs

## Limitations & Future Work

- Performance on Llama2-13B degrades by >0.1 under context noise, warranting further analysis
- The current approach assumes retrieved documents are relevant and sufficient; extremely low-quality retrieval scenarios are not thoroughly evaluated
- The logit lens projection in the information processing rate may require adaptation for newer architectures (e.g., MoE models)
- Validation on reasoning-intensive tasks (e.g., mathematical reasoning) has not been conducted

## Related Work & Insights

- The application of MMD as a distributional distance measure is elegant and extensible to other signal detection scenarios
- The information processing rate offers a new perspective for observing LLM internal states and may inspire new training objectives
- Cross-model detection results suggest that "knowledge utilization patterns" in LLMs may exhibit cross-model commonality
- The work has direct practical implications for reliability guarantees in RAG systems

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of MMD and information processing rate is novel and theoretically grounded
- Technical Depth: ⭐⭐⭐⭐⭐ — The statistical validation framework substantially enhances methodological credibility
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-model, multi-dataset evaluation with extensive ablations and robustness analysis
- Practical Value: ⭐⭐⭐⭐⭐ — Unsupervised, training-free, and cross-model generalizable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](../../AAAI2026/video_understanding/beyond_fact_retrieval_episodic_memory_for_rag_with_generative_semantic_workspace.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[ICML 2026\] Find, Fix, Reason: Context Repair for Video Reasoning](../../ICML2026/video_understanding/find_fix_reason_context_repair_for_video_reasoning.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](../../AAAI2026/video_understanding/kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)

</div>

<!-- RELATED:END -->
