---
title: >-
  [Paper Note] Hyperbolic Fine-Tuning for Large Language Models
description: >-
  [NeurIPS 2025][LLM Evaluation][hyperbolic geometry] This work identifies that LLM token embeddings follow power-law distributions and exhibit tree-like hyperbolic structure, and proposes HypLoRA — performing low-rank adaptation directly on the Lorentz hyperbolic manifold (bypassing the cancellation effect of tangent space mappings) — achieving significant gains over standard LoRA on arithmetic and commonsense reasoning tasks (e.g., M.AVG +7.5% on Qwen2.5-7B).
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - hyperbolic geometry
  - LoRA
  - parameter-efficient fine-tuning
  - Lorentz model
  - LLM reasoning
date: 2026-05-08
content_hash: c9ec9d1ffa41fe50
---

# Hyperbolic Fine-Tuning for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2410.04010](https://arxiv.org/abs/2410.04010)
**Code**: [https://github.com/marlin-codes/HypLoRA](https://github.com/marlin-codes/HypLoRA)
**Area**: LLM Evaluation
**Keywords**: hyperbolic geometry, LoRA, parameter-efficient fine-tuning, Lorentz model, LLM reasoning

## TL;DR
This work identifies that LLM token embeddings follow power-law distributions and exhibit tree-like hyperbolic structure, and proposes HypLoRA — performing low-rank adaptation directly on the Lorentz hyperbolic manifold (bypassing the cancellation effect of tangent space mappings) — achieving significant gains over standard LoRA on arithmetic and commonsense reasoning tasks (e.g., M.AVG +7.5% on Qwen2.5-7B).

## Background & Motivation

**Background**: In LLM parameter-efficient fine-tuning (PEFT), LoRA has become the dominant approach due to its simplicity and effectiveness. However, all existing LoRA variants operate on weight matrices in Euclidean space.

**Limitations of Prior Work**: Linguistic concepts naturally exhibit hierarchical structure (e.g., "fruit" → "apple"/"banana"), which Euclidean space cannot efficiently represent. Hyperbolic space is better suited for such tree-like hierarchies due to its negative curvature and exponential volume growth, yet training hyperbolic LLMs from scratch is prohibitively expensive.

**Core Finding**: Through in-depth analysis of multiple open-source LLMs, the authors find: (a) token frequencies follow a power-law distribution ($\gamma \approx 1.9$), with high-frequency tokens concentrated near the origin and low-frequency tokens farther away; (b) the $\delta$-hyperbolicity of token embeddings is extremely low ($\delta_{\text{rel}} \approx 0.06$–$0.12$), indicating that the embedding space has strong tree-like structure.

**Key Challenge**: Conventional hyperbolic neural networks operate through the pipeline "tangent space → exponential map → logarithmic map → tangent space," causing the exponential and logarithmic maps to cancel each other ($\log \circ \exp = \text{id}$), thereby nullifying the benefits of hyperbolic geometry.

**Core Idea**: Design HypLoRA to perform low-rank transformations directly on the hyperbolic manifold (Lorentz model), bypassing tangent space mappings and preserving the modeling capacity of hyperbolic geometry.

## Method

### Overall Architecture
The pretrained LLM weights $W$ are frozen, and a HypLoRA adapter is added to each target Transformer layer. The input $\mathbf{x}^E$ passes through two paths: (1) the original frozen path $W\mathbf{x}^E$; and (2) the HypLoRA path — first projected to hyperbolic space via $\Pi_{\exp}^K$, then transformed by the low-rank Lorentz transformation LLR in hyperbolic space, and finally projected back to Euclidean space via $\Pi_{\log}^K$. The outputs of both paths are summed to produce the final output.

### Key Designs

1. **Hyperbolicity Analysis of LLM Embeddings**:

    - Function: Empirically verify that LLM embedding spaces possess genuine hyperbolic geometric properties.
    - Mechanism: (a) Compute global token frequency statistics → power-law distribution $P(k) \sim k^{-\gamma}$; (b) Analyze the relationship between token frequency and embedding norm → high-frequency tokens have smaller norms and low-frequency tokens have larger norms (consistent across LLaMA/Gemma/Qwen); (c) Compute $\delta$-hyperbolicity via the Gromov four-point condition → values near 0 (0.06–0.12), far below spherical space (0.99) and random graphs (0.62).
    - Design Motivation: Provide an empirical foundation for fine-tuning in hyperbolic space, demonstrating that this is not an arbitrary assumption.

2. **Direct Lorentz Low-Rank Transformation (LLR)**:

    - Function: Perform low-rank matrix transformations directly in hyperbolic space, avoiding the cancellation caused by tangent space mappings.
    - Mechanism: $\mathbf{LLR}(BA, \mathbf{x}^H) = (\sqrt{\|BA\mathbf{x}_s^H\|_2^2 + K},\ BA\mathbf{x}_s^H)$, where $\mathbf{x}_s^H$ is the spatial component of the hyperbolic point. The transformation is applied only to the spatial dimensions, while the temporal dimension is automatically recovered by the Lorentz constraint.
    - Design Motivation: Circumvent the cancellation problem of traditional "exponential map → logarithmic map" pipelines. The output remains on the Lorentz manifold ($\langle \mathbf{x}, \mathbf{x} \rangle_\mathcal{L} = -K$), equivalent to a pseudo-Lorentz rotation.

3. **Full HypLoRA Adaptation Formula**:

    - $\mathbf{z}^E = W\mathbf{x}^E + \Pi_{\log}^K(\mathbf{LLR}(BA,\ \Pi_{\exp}^K(\mathbf{x}^E)))$
    - where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{k \times r}$, and $r \ll \min(d, k)$.
    - The parameter count $(d+k) \cdot r$ is identical to standard LoRA, with only an $O(N)$ overhead from the hyperbolic projections.

### Loss & Training
- Only matrices $A$ and $B$ are trained during fine-tuning; the original $W$ is frozen.
- The curvature $K$ is treated as a learnable parameter, initialized to 0.5 or 1.0 (optimal values vary by model).
- Math10K and Commonsense170K datasets are used to fine-tune for arithmetic reasoning and commonsense reasoning tasks, respectively.

## Key Experimental Results

### Main Results — Arithmetic Reasoning

| Base Model | Method | Params (%) | MAWPS | SVAMP | GSM8K | AQuA | M.AVG |
|------------|--------|-----------|-------|-------|-------|------|-------|
| LLaMA3-8B | LoRA | 0.70 | 92.7 | 78.9 | 70.8 | 30.4 | 71.9 |
| LLaMA3-8B | HypLoRA | 0.70 | 91.6 | 80.5 | 74.0 | 34.2 | **74.2** |
| Gemma3-4B | LoRA | 1.04 | 90.8 | 77.3 | 72.3 | 50.8 | 73.7 |
| Gemma3-4B | HypLoRA | 1.04 | 88.2 | 83.9 | 76.1 | 53.2 | **77.8** |
| Qwen2.5-7B | LoRA | 0.71 | 90.8 | 84.4 | 78.6 | 68.1 | 80.8 |
| Qwen2.5-7B | HypLoRA | 0.71 | 91.2 | 92.2 | 87.9 | 71.6 | **88.3** |

### Main Results — Commonsense Reasoning

| Base Model | Method | BoolQ | PIQA | HellaSwag | ARC-c | OBQA | AVG |
|------------|--------|-------|------|-----------|-------|------|-----|
| LLaMA3-8B | LoRA | 70.8 | 85.2 | 91.7 | 71.2 | 79.0 | 80.8 |
| LLaMA3-8B | HypLoRA | 74.1 | 87.6 | 94.5 | 81.2 | 85.2 | **84.8** |
| Qwen2.5-7B | LoRA | 73.4 | 89.5 | 93.6 | 82.0 | 87.0 | 85.2 |
| Qwen2.5-7B | HypLoRA | 72.8 | 89.3 | 94.8 | 87.5 | 90.8 | **87.0** |

### Ablation Study — Effect of Curvature (Gemma3-4B)

| Curvature $K$ | MAWPS | SVAMP | GSM8K | AQuA | M.AVG |
|---------------|-------|-------|-------|------|-------|
| 0.5 | 88.2 | **83.9** | **76.1** | **53.5** | **77.8** |
| 1.0 | **91.9** | 80.3 | 73.8 | 52.7 | 75.8 |

### Key Findings
- HypLoRA yields the largest gains on complex reasoning datasets (GSM8K, AQuA, SVAMP), while LoRA sometimes outperforms it on simpler datasets (MAWPS) — suggesting that hyperbolic geometry is most advantageous in multi-step reasoning.
- The largest M.AVG improvement is observed on Qwen2.5-7B (+7.5%), possibly because Qwen's embedding distribution more closely conforms to a hyperbolic structure.
- Curvature $K=0.5$ is optimal for most models and tasks.
- Inference efficiency is comparable to LoRA, with negligible additional overhead.

## Highlights & Insights
- **Rigorous Geometric Analysis**: Rather than applying hyperbolic space directly, the paper first systematically verifies that LLM embeddings genuinely exhibit hyperbolic properties (power-law distribution + low $\delta$-hyperbolicity), providing a solid empirical foundation for the method design.
- **Elegant Bypass of the Cancellation Problem**: The traditional $\log \circ \exp$ pipeline in hyperbolic networks degenerates back to Euclidean space; LLR performs linear transformations directly on the manifold, preserving the benefits of hyperbolic geometry.
- **Plug-and-Play**: HypLoRA shares the same interface and parameter count as LoRA and can be orthogonally combined with other LoRA variants such as DoRA and AdaLoRA.
- **Theoretical Support**: Proposition 1 proves that HypLoRA introduces norm-dependent higher-order terms capable of capturing hierarchical relationships.

## Limitations & Future Work
- Validation is limited to arithmetic and commonsense reasoning; performance on code generation, translation, dialogue, and other tasks remains unknown.
- The initial value of the learnable curvature still requires manual selection (0.5 or 1.0), with optimal values varying across tasks.
- The $\delta$-hyperbolicity analysis is conducted at the prompt level; corpus-level analysis may yield different conclusions.
- Only the hyperbolic variant of LoRA is explored; whether other PEFT methods such as Adapter and Prefix Tuning can be similarly hyperbolicized remains an open question.

## Related Work & Insights
- **vs. LoRA**: LoRA performs low-rank adaptation in Euclidean space as $\Delta W = BA$; HypLoRA adds hyperbolic projection and LLR operations with the same parameter count, but introduces nonlinear higher-order terms.
- **vs. DoRA**: DoRA decomposes weights into direction and magnitude components, an improvement within Euclidean space; HypLoRA changes the geometric space itself, making the two approaches orthogonal and combinable.
- **vs. Hyperbolic Transformers**: Prior work (e.g., hyperbolic attention) trains models in hyperbolic space from scratch, which is prohibitively costly for LLMs; HypLoRA introduces hyperbolic geometry only during fine-tuning, making it far more practical.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce hyperbolic geometry into LLM fine-tuning, with analysis, method, and theory unified.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models and tasks with sufficient ablation; lacks validation on a broader range of task types.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from motivation → analysis → method → experiments is complete, with clear figures.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for hyperbolic PEFT with significant implications for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)

</div>

<!-- RELATED:END -->
