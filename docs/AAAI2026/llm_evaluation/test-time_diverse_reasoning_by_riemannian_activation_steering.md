---
title: >-
  [Paper Note] Test-time Diverse Reasoning by Riemannian Activation Steering
description: >-
  [AAAI 2026][LLM Evaluation][Activation Steering] This paper proposes SPREAD, an unsupervised test-time activation steering framework that maximizes the total volume spanned by hidden activations across multiple reasoning…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Activation Steering"
  - "Reasoning Diversity"
  - "Riemannian Optimization"
  - "Best-of-N Sampling"
  - "Language Model Reasoning"
  - "Manifold Optimization"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: a1a175885740a34f
---

# Test-time Diverse Reasoning by Riemannian Activation Steering

**Conference**: AAAI 2026
**arXiv**: [2511.08305](https://arxiv.org/abs/2511.08305)  
**Code**: [https://github.com/lythk88/SPREAD](https://github.com/lythk88/SPREAD)  
**Area**: LLM Evaluation
**Keywords**: Activation Steering, Reasoning Diversity, Riemannian Optimization, Best-of-N Sampling, Language Model Reasoning, Manifold Optimization, Mathematical Reasoning

## TL;DR

This paper proposes SPREAD, an unsupervised test-time activation steering framework that maximizes the total volume spanned by hidden activations across multiple reasoning paths by solving a Riemannian optimization problem on a product of spherical manifolds. SPREAD improves reasoning diversity and accuracy in Best-of-N sampling, outperforming temperature sampling baselines on mathematical reasoning benchmarks.

## Background & Motivation

The Best-of-N inference strategy improves LLM accuracy on complex tasks by sampling $N$ candidate answers and selecting the best one. However, its effectiveness is bounded by an **output diversity ceiling**: even under stochastic sampling, models tend to generate highly similar reasoning paths (diversity collapse), repeatedly making the same errors.

Existing approaches to improving diversity face three key challenges:

**Stochastic decoding methods** (temperature sampling, top-k, nucleus): introduce randomness at the token level, but the resulting reasoning paths often converge to nearly identical chains of thought.

**Search strategies** (contrastive search, diverse beam search): jointly consider trajectory distributions but incur high computational cost.

**Diversity measurement difficulties**: lexical/semantic diversity metrics are sensitive to text length and paraphrasing, and require additional neural network computation.

A core observation is that hidden activations constitute the model's internal "thinking space," and distinct activation clusters tend to correspond to distinct "reasoning circuits." Therefore, **promoting activation diversity can induce reasoning diversity**.

## Method

### Overall Architecture

SPREAD (SPherical intervention for REAsoning Diversity) operates during autoregressive generation. At designated synchronization anchor positions, it extracts the hidden state vectors of the last token from all $N$ generation paths: $H = [h_1, ..., h_N] \in \mathbb{R}^{p \times N}$. It then computes additive steering vectors $V = [v_1, ..., v_N]$ such that the intervened activations $H_{new} = H + V$ are geometrically as "spread out" as possible. The steering vectors remain active until the next synchronization anchor.

### Key Design 1: Volume Maximization Objective

SPREAD maximizes the squared sum of parallelotope volumes over all subsets:

$$\max_{\forall i: \|v_i\|_2^2 \leq \alpha_i} \sum_{\mathbb{I} \in 2^{[N]}} \text{Volume}(\mathcal{P}(\{h_i + v_i\}_{i \in \mathbb{I}}))^2$$

This is reformulated into a log-det minimization problem via Gram matrix determinant equivalence:

$$\min_{V} -\log\det[I + (H+V)^\top(H+V)]$$

subject to $\|v_i\|_2^2 = \alpha_i$ (the paper proves that the inequality constraint is always tight at the optimum). Setting $\alpha_i = C \|h_i\|_2 / p$ reduces the hyperparameter to a single relative scalar $C > 0$.

**The $2^N$ subsets are never enumerated explicitly**: leveraging the properties of Determinantal Point Processes (DPP), the sum of volumes over all subsets equals the determinant of a single $(N+1) \times (N+1)$ matrix.

### Key Design 2: Riemannian Block Coordinate Descent

The equality constraint $\|v_i\|_2^2 = \alpha_i$ confines each steering vector to a sphere of radius $\sqrt{\alpha_i}$, yielding the product manifold $\mathcal{M} = \mathcal{M}_1 \times ... \times \mathcal{M}_N$. The algorithm proceeds as follows:

1. **Initialization**: $v_i^{(0)} = \sqrt{\alpha_i} \cdot (h_i + \varepsilon_i - \bar{h}) / \|h_i + \varepsilon_i - \bar{h}\|_2$ (mean subtraction with small noise perturbation).
2. **Each iteration** updates $i = 1, ..., N$ sequentially:
    - Compute the Euclidean gradient $g_i$ and project it onto the tangent space of the sphere to obtain the Riemannian descent direction $d_i$.
    - Move along the geodesic via the exponential map: $v_i^{(k)} = \cos(\cdot) v_i^{(k-1)} + \sin(\cdot) \frac{d_i}{\|d_i\|_2} \sqrt{\alpha_i}$.

The step size $\eta_i = 1/L_i$ is determined by the block smoothness constant (with a closed-form expression provided in the paper), **requiring no tuning**.

### Key Design 3: Convergence Guarantee

The paper proves that the algorithm converges to a stationary point at rate $O(1/\sqrt{k})$:

$$\min_{s \leq k} \|\text{grad}\ \ell(V^{(s)})\|_F \leq \sqrt{\frac{\ell(V^{(0)}) - \ell^*}{Ck}}$$

Although the objective is non-convex (demonstrated by a counterexample in the paper), the properties of Riemannian block coordinate descent guarantee that every limit point is a stationary point.

### Loss & Training

SPREAD is a purely inference-time method requiring **no training or fine-tuning**. It does not modify model parameters; instead, it intervenes additively on the residual stream of a specified layer via $\tilde{x}^{(l+1)} = x^{(l+1)} + v^{(l+1)}$. Experiments apply steering at layer 28 (the final layer), with synchronization anchors at token positions $\tau \in \{100, 600, 1100, 1600\}$ and $K=20$ iterations.

## Key Experimental Results

### Main Results: Pass@N Accuracy (%)

| Model | Temp | AIME24 (SPREAD C=1) | AIME24 (Sampling) | MATH500 (SPREAD C=1) | MATH500 (Sampling) | OlympiadBench (SPREAD C=1) | OlympiadBench (Sampling) |
|------|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | 1.0 | **3.3** | 0.0 | **43.2** | 42.8 | **21.5** | 19.0 |
| Qwen2.5-1.5B | 0.6 | **10.0** | 3.3 | **55.0** | 53.4 | 28.4 | 29.3 |
| Math-1.5B-Inst | 1.0 | **26.7** | 20.0 | 83.8 | 84.6 | 47.6 | 48.3 |
| Math-1.5B-Inst | 0.6 | **26.7** | 20.0 | **85.4** | 84.6 | 50.4 | 50.8 |

SPREAD matches or outperforms temperature sampling in the vast majority of configurations.

### Diversity Metrics: Unique Solution Count

| Model | Temp | AIME24 (SPREAD C=1) | AIME24 (Sampling) | MATH500 (SPREAD C=1) | MATH500 (Sampling) |
|------|:----:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | 1.0 | **6.97** | 6.67 | **3.14** | 3.14 |
| Qwen2.5-1.5B | 0.8 | **6.83** | 3.60 | **3.03** | 2.97 |
| Math-1.5B-Inst | 1.0 | **6.63** | 3.67 | 1.92 | 1.93 |
| Math-1.5B-Inst | 0.8 | **6.63** | 3.47 | 1.87 | 1.89 |

SPREAD substantially increases the number of unique solutions on AIME24 (e.g., 6.97 vs. 6.67 or 3.67), demonstrating that it genuinely induces diverse reasoning paths.

### Key Findings

1. **Hypothesis validation**: Statistical hypothesis testing (logistic regression with cluster-robust standard errors) confirms a positive correlation between activation volume and answer diversity ($\hat{\beta}=0.88, p=0.001$); each unit increase in volume yields approximately 2.4× higher odds of producing a unique solution.
2. **Computational efficiency**: For $N=32, p=16384$ (corresponding to LLaMA-405B scale), the algorithm runs in under 1.8 seconds.
3. **Layer selection**: Applying steering at the final layer (layer 28) yields the best results; shallow-layer steering produces unstable effects.
4. **Minimal hyperparameters**: In practice, only a single parameter $C$ (the relative steering strength coefficient) needs to be set; $C=1$ is optimal in most cases.
5. **Pareto frontier advantage**: On accuracy–diversity bi-axial plots, SPREAD consistently dominates temperature sampling.

## Highlights & Insights

- **Rigorous mathematical foundation**: The derivation chain from volume maximization → log-det → spherical manifold optimization is theoretically sound and provides convergence guarantees.
- **Extremely lightweight**: No model weight modification, no auxiliary neural networks, and no positive/negative sample pairs are required; the method is purely geometric.
- **No contrastive samples needed**: Unlike conventional activation steering, which requires positive/negative behavior pairs, SPREAD is fully unsupervised.
- **First application of activation steering to mathematical reasoning**: Traditional activation steering has avoided mathematical reasoning due to the difficulty of defining positive/negative samples; SPREAD circumvents this limitation entirely.
- **Elegant connection between DPP and geometry**: The properties of Determinantal Point Processes are exploited to compress the exponential subset summation into a single determinant computation.

## Limitations & Future Work

1. There is no one-to-one correspondence between activation diversity and reasoning diversity — increasing activation diversity does not guarantee genuinely distinct reasoning paths.
2. Validation is conducted only on 1.5B-parameter models; models at the 7B+ scale remain untested.
3. Accuracy improvements in mathematical reasoning (a few percentage points) fall within the noise range under certain configurations.
4. The selection of synchronization anchor positions (100, 600, 1100, 1600) appears heuristic, with no adaptive strategy proposed.
5. Diversity evaluation relies on GPT-4.1-mini as a judge, which may introduce evaluation bias.

## Related Work & Insights

- **Activation Steering** (Turner et al., 2023): Contrastive steering — computes steering vectors from the mean difference between positive and negative behaviors, which is inapplicable to mathematical reasoning.
- **Inference-Time Intervention** (Li et al., 2023): Probe-based steering that requires training a classifier to identify target concepts.
- **Diverse Beam Search** (Vijayakumar et al., 2016): Introduces diversity at the beam level, incurring high computational overhead.
- **WLD-Reg** (Laakom et al., 2023): Intra-layer activation diversity regularization designed for training rather than inference.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Riemannian manifold optimization applied to reasoning diversity steering; a conceptually original and theoretically complete contribution)
- Experimental Thoroughness: ⭐⭐⭐ (Three benchmarks but only 1.5B-scale models; some improvements are marginal)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematical derivations are clear and rigorous; figures are intuitive)
- Value: ⭐⭐⭐⭐ (Offers a novel perspective on inference-time diversity enhancement; the lightweight design has practical potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](../../ICLR2026/llm_evaluation/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/llm_evaluation/test-time_adaptation_by_causal_trimming.md)

</div>

<!-- RELATED:END -->
