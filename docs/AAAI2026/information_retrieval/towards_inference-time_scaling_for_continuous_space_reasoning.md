---
title: >-
  [Paper Note] Towards Inference-Time Scaling for Continuous Space Reasoning
description: >-
  [AAAI 2026][Information Retrieval & RAG][inference-time scaling] This work presents the first systematic investigation of whether inference-time scaling techniques from discrete text reasoning can transfer to continuous…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "inference-time scaling"
  - "continuous reasoning"
  - "COCONUT"
  - "geometric homogeneity"
  - "process reward model"
date: 2026-05-08
content_hash: 72a687c625c31ab1
---

# Towards Inference-Time Scaling for Continuous Space Reasoning

**Conference**: AAAI 2026
**arXiv**: [2510.12167](https://arxiv.org/abs/2510.12167)  
**Code**: To be released  
**Area**: Information Retrieval
**Keywords**: inference-time scaling, continuous reasoning, COCONUT, geometric homogeneity, process reward model

## TL;DR
This work presents the first systematic investigation of whether inference-time scaling techniques from discrete text reasoning can transfer to continuous latent-space reasoning models (COCONUT). It finds that dropout sampling can generate diverse reasoning paths (Pass@32 reaching 44.43%), but PRM/ORM yields less than 2.3% improvement, with the root cause being that continuous thought representations lack the geometric inductive bias needed to distinguish correct from incorrect reasoning.

## Background & Motivation
**Background**: Inference-time scaling is well established in text-based reasoning — generating multiple samples and reranking with PRM/ORM significantly boosts accuracy. Continuous space reasoning (e.g., COCONUT) is an emerging paradigm that performs multi-step reasoning directly in latent space, replacing explicit chain-of-thought text generation.

**Limitations of Prior Work**: Continuous reasoning models such as COCONUT operate deterministically, making it impossible to directly generate diverse reasoning paths. Moreover, the internal structure of their continuous thought vectors is opaque, and whether existing text-space PRM/ORM methodologies are applicable remains entirely unknown.

**Key Challenge**: Pass@N analysis reveals substantial scaling potential in continuous reasoning (~13% absolute headroom), yet existing reward model methodologies cannot effectively exploit this potential — a large gap exists between the theoretical upper bound and actual gains.

**Goal**: To answer two key questions: (1) Can diverse reasoning paths be generated for continuous reasoning? (2) Can effective PRM/ORM be trained to rerank these paths?

**Key Insight**: The first question is addressed by injecting stochasticity via dropout; the second by adapting the MATH-Shepherd annotation framework to train PRM/ORM. Failure is then analyzed in depth through geometric analysis, trajectory dynamics, and perturbation experiments.

**Core Idea**: Continuous reasoning holds significant inference-time scaling potential, but the geometric homogeneity of continuous thought representations constitutes the fundamental bottleneck preventing reward models from effective discrimination.

## Method

### Overall Architecture
Three stages: (1) generate diverse reasoning trajectories for COCONUT via dropout sampling; (2) adapt the MATH-Shepherd annotation framework to train PRM and ORM; (3) systematically analyze the geometric properties of the continuous reasoning space to explain reward model failure.

### Key Designs
1. **Dropout Sampling for Diverse Trajectory Generation**

    - Function: Selectively enable dropout during the continuous reasoning phase while disabling it during text generation, injecting controllable stochasticity into the otherwise deterministic COCONUT.
    - Mechanism: The COCONUT hidden state $\mathbf{s}_i = f_\theta(X, \mathbf{s}_{<i})$ is subject to dropout during the forward pass, so different samples yield different reasoning paths.
    - Design Motivation: Text LLMs obtain diversity through token sampling, but COCONUT's reasoning process is deterministic with no token distribution to sample from; dropout is the most natural alternative.

2. **MC Annotation and Reward Model Training in Continuous Space**

    - Function: Adapts the MATH-Shepherd framework to perform MC annotation on each continuous thought vector.
    - Mechanism: For each reasoning step $s_i$, $N$ completions are generated from that step; a hard estimate ($y^{HE}_{s_i}=1$ if any completion is correct) and a soft estimate (fraction correct $y^{SE}_{s_i}$) are computed.
    - Key Constraint: Continuous representations are model-specific — only the original COCONUT can interpret its own latent space, so PRM/ORM must use COCONUT itself as the backbone.

3. **Multi-Dimensional Analysis of the Continuous Reasoning Space**

    - **Geometric Analysis**: IsoScore★ (isotropy) and Hoyer (sparsity) are used to characterize the high-dimensional distribution of thought vectors.
    - **Trajectory Dynamics**: Four metrics are computed — compactness, curvature, local smoothness, and straightness.
    - **Perturbation Analysis**: Gaussian noise of varying intensity is injected into the latent space and the effect on reasoning performance is observed.
    - Design Motivation: If correct and incorrect reasoning paths are geometrically inseparable, reward models are in principle unable to learn effective discriminative features.

### Loss & Training
- **PRM Training**: Joint loss $\mathcal{L}_{PRM} = \mathcal{L}_{CE}(y^{HE}, \hat{y}^{HE}) + \mathcal{L}_{MSE}(y^{SE}, \hat{y}^{SE})$
- **ORM Training**: Cross-entropy loss $\mathcal{L}_{ORM} = \mathcal{L}_{CE}(r^{OUT}, \hat{r}^{OUT})$
- Positive/negative samples balanced at 1:1; PRM trained on 238k samples, ORM on 324k samples; 10 epochs; learning rate 1e-4.

## Key Experimental Results

### Main Results — Best-of-N Performance of Different Reranking Methods on GSM8k

| Method | N=1 | N=4 | N=8 | N=16 | N=32 |
|--------|-----|-----|-----|------|------|
| Pass@N (theoretical upper bound) | 31.08 | 38.67 | 41.02 | 42.61 | 44.43 |
| Confidence | 31.08 | 30.48 | 29.87 | 31.39 | 30.71 |
| Self-Consistency | 31.08 | 31.61 | 31.24 | 32.15 | 32.15 |
| PRM-HE | 31.08 | 32.45 | **33.06** | **33.36** | 32.83 |
| ORM | 31.08 | 32.15 | 31.46 | 32.37 | 31.39 |
| PRM-SE | 31.08 | 32.37 | 32.52 | 32.37 | **33.28** |

### Ablation Study — Geometric Properties of Continuous Thoughts

| Metric | Correct Thoughts | Incorrect Thoughts | Difference |
|--------|------------------|--------------------|------------|
| IsoScore★ (full set) | 0.0134 | 0.013 | Negligible |
| Hoyer (full set) | 0.21±0.01 | 0.22±0.01 | Negligible |
| Compactness (full set) | 19.81±2.53 | 19.39±2.48 | p=0.023, Cohen's d=0.17 |
| Local Smoothness (PRM+) | 0.39±0.09 | 0.48±0.10 | p=0.049, Cohen's d=-0.97 |

### Key Findings
- **Large Potential–Gain Gap**: The theoretical upper bound of Pass@32 is 44.43%, yet the best PRM/ORM achieves only 33.36% (+2.28%), far below comparable methods in discrete text space.
- **Geometric Homogeneity**: Correct and incorrect continuous thoughts show virtually no difference on IsoScore★, Hoyer, and related geometric metrics; t-SNE visualization confirms complete mixing.
- **High Robustness as a Problem**: In noise perturbation experiments, low noise ratios (0–0.2) barely affect performance; even full noise replacement (ratio=1.0) yields 12.59% Pass@5, suggesting COCONUT's reasoning does not fully depend on continuous thoughts.
- **Confidence Reranking Fails**: This indicates that COCONUT lacks effective probability calibration.
- **Aggregation Strategy Is Irrelevant**: PRM min/max/mean/last aggregation strategies show negligible differences (Table 2), confirming that the problem lies in the representations themselves rather than the scoring approach.

## Highlights & Insights
- **First Systematic Study**: The first work to apply inference-time scaling to continuous space reasoning, establishing a benchmark and analytical framework.
- **Simplicity of Dropout Sampling**: Stochasticity is injected via dropout that already exists during training, incurring zero additional training cost.
- **Analytical Depth**: The investigation proceeds through four independent lenses — classification performance, geometric properties, trajectory dynamics, and perturbation experiments — forming a rigorous analysis chain.
- **Key Insight**: Continuous reasoning training optimizes only final answer accuracy and introduces no inductive bias that would drive structural divergence between correct and incorrect reasoning paths — this is the root cause.
- **Constructive Directions**: Contrastive learning, isotropy constraints, and trajectory diversity are proposed as future directions.

## Limitations & Future Work
- Based solely on GPT-2 + COCONUT (a relatively small model); it remains unverified whether larger continuous reasoning models exhibit the same issue.
- Experiments are conducted only on GSM8k; other reasoning tasks (e.g., code generation, logical reasoning) are not evaluated.
- The work is primarily diagnostic — it identifies the problem but does not propose a concrete improved training method.
- Dropout sampling causes a slight drop at N=1 compared to deterministic inference (~31% vs. ~31.08%), motivating better diversity injection schemes.
- PRM/ORM exclusively uses COCONUT's own backbone; the possibility of external discriminative models is unexplored.
- The number of continuous reasoning steps is fixed at $T=6$ ($3 \times c$, $c=2$); the effect of varying reasoning length is not explored.
- The fact that ratio=1.0 noise still yields 12.59% accuracy suggests that some problems are solvable without continuous reasoning at all.

## Related Work & Insights
- **COCONUT** (Hao et al. 2024): The seminal work on continuous thought reasoning → this paper reveals fundamental limitations of its training paradigm.
- **MATH-Shepherd** (Wang et al. 2024): The standard MC annotation method in discrete space → direct transfer to continuous space yields limited effectiveness.
- **CODI/CCOT/CoT²**: Other continuous reasoning methods → likely to face the same geometric homogeneity issue.
- **Implications for Continuous Reasoning Training**: Future work must explicitly incorporate geometric discriminability constraints (e.g., contrastive loss) into training objectives, rather than optimizing answer accuracy alone.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first systematic study of inference-time scaling for continuous space reasoning; the research question itself constitutes an important contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The problem is analyzed from multiple independent perspectives (geometry, trajectories, perturbations), yielding comprehensive and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ — The logical chain is clear: identify potential → attempt exploitation → analyze failure → indicate future directions.
- Value: ⭐⭐⭐⭐⭐ — Provides foundational guidance for the future development of continuous reasoning, clearly identifying the direction in which the training paradigm must change.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](../../ACL2026/information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)
- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
