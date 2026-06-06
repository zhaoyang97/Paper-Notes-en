---
title: >-
  [Paper Note] Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention
description: >-
  [ACL 2026][LLM Efficiency][Attention Mechanism] TDA achieves sink-free, 99% precise sparse, and competitively performing long-context Transformer attention by combining length-adaptive thresholds with differential inhibi…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Attention Mechanism"
  - "Long Context"
  - "Sparse Attention"
  - "Differential Attention"
  - "Extreme Value Theory"
date: 2026-05-08
content_hash: 30e8013efb6d5ba5
---

# Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention

**Conference**: ACL 2026  
**arXiv**: [2601.12145](https://arxiv.org/abs/2601.12145)  
**Code**: https://github.com/snap-research/TDA  
**Area**: LLM Efficiency  
**Keywords**: Attention Mechanism, Long Context, Sparse Attention, Differential Attention, Extreme Value Theory

## TL;DR
TDA achieves sink-free, 99% precise sparse, and competitively performing long-context Transformer attention by combining length-adaptive thresholds with differential inhibitory views.

## Background & Motivation

**Background**: Self-attention has become the core of Transformers due to its differentiability and efficient vectorized implementation. However, Softmax attention faces fundamental structural limitations when processing long sequences, primarily manifested as two pathological phenomena.

**Limitations of Prior Work**: The sum-to-one constraint of Softmax forces the model to allocate non-zero probability mass to irrelevant tokens to satisfy normalization requirements, resulting in the attention sink phenomenon. Simultaneously, as sequence length increases, probability mass dilutes, leading to decreased focus on salient tokens. While projection-based sparse methods (e.g., Entmax) produce exact zeros, they are computationally expensive. Conversely, non-normalized rectified activations (e.g., ReLA) are efficient but suffer from performance degradation under long contexts due to noise accumulation.

**Key Challenge**: Existing methods cannot simultaneously achieve three objectives: (1) exact sparsity and computational efficiency, (2) sink-free attention, and (3) long-context robustness. Sparse methods typically still enforce the sum-to-one constraint, thus failing to fundamentally solve the sink problem. Rectified methods solve the sink problem but cannot control noise growth in long sequences with fixed thresholds.

**Goal**: Design a drop-in replacement for Softmax attention that satisfies sink-free, ultra-sparse, and long-context robustness requirements without exceeding the computational overhead of standard methods.

**Key Insight**: Starting from extreme value theory, it is observed that in high dimensions, the maximum value of dot products between irrelevant query-key pairs grows with sequence length (extreme value effect). Thus, an adaptive threshold related to context length can be employed to suppress these spurious matches. Furthermore, drawing on the idea of Differential Transformers, common-mode noise can be eliminated by calculating the difference between an inhibitory view and an excitatory view.

**Core Idea**: Filter extreme value noise with length-adaptive thresholds and cancel spurious matches with differential views to obtain sink-free sparse attention.

## Method

### Overall Architecture

TDA is constructed at two levels: first, starting from rectified attention, a length-aware threshold mechanism (TRA) is introduced; then, a differential structure is added, using the difference between two independent views to further suppress noise (TDA). The process consists of three stages: (1) Projection and normalization: normalize query and key vectors using the L2 norm; (2) Similarity calculation and threshold filtering: calculate the dot product of row-wise queries and all keys, subtract the length-adaptive threshold, retain components exceeding the threshold, and apply a non-linear transformation; (3) Value aggregation: accumulate selected value vectors and perform final normalization via RMSNorm.

### Key Designs

1.  **Length-adaptive Threshold**:

    - **Function**: Dynamically adjust the threshold based on context length to prevent extreme value noise from growing with the sequence.
    - **Mechanism**: Based on the sub-Gaussian assumption, the maximum value of spurious dot products theoretically satisfies $\tau_i \sim \sqrt{2\log(i/\kappa)/d}$. The authors define the row-level threshold as $\tau_i := \beta\sqrt{2\log((i+1)/\kappa)/d}$, where $i$ is the query position, $\beta>0$ is a learnable scalar, and $\kappa>0$ controls the expected number of spurious survivors. The resulting attention weights are $\mathbf{a}_{ij} = (\mathbf{s}_{ij} - \tau_i)_+^p$, where $(x)_+ = \max(x,0)$ and $p \geq 1$ is the power.
    - **Design Motivation**: Vershynin's extreme value theory shows that under sub-Gaussian noise, the probability decay of the maximum value is positively correlated with $\sqrt{\log i / d}$. Fixed thresholds fail in long sequences, whereas this threshold growing with $\log i$ keeps noise control stable as sequence length increases. It theoretically guarantees the expected number of spurious survivors per row is $O(1)$.

2.  **Differential View Construction**:

    - **Function**: Further suppress spurious matches appearing in both views by subtracting two independent thresholded views.
    - **Mechanism**: Maintain two independent sets of projection parameters $\{(\mathbf{q}^{(t)}, \mathbf{k}^{(t)})\}_{t \in \{1,2\}}$. Calculate similarity and apply length-adaptive thresholds for each view separately to get $\mathbf{a}_{ij}^{(t)} = (\mathbf{s}_{ij}^{(t)} - \tau_i)_+^p$. The final weights are $\Delta\mathbf{a}_{ij} = \mathbf{a}_{ij}^{(1)} - \lambda\mathbf{a}_{ij}^{(2)}$, where $\lambda \in (0,1)$ is a learnable suppression intensity parameter.
    - **Design Motivation**: Even if a single view controls spurious survivors to $O(1)$ via thresholds, occasional high-magnitude noise may occur. The differential construction is based on the observation that a large similarity value might be spuriously generated due to shared non-informative structures; the inhibitory view is trained to capture such non-selective excitations. The probability of exceeding thresholds in two independent views simultaneously drops to $O(1/(i+1))$ under the independence assumption, vanishing asymptotically. This endows TDA with signed attention weights, enhancing expressivity.

3.  **RMSNorm Value Aggregation**:

    - **Function**: Stabilize the value aggregation process for extremely sparse attention weights.
    - **Mechanism**: Calculate $\mathbf{o}_i := \mathrm{Norm}(\sum_{j=1}^{i}\Delta\mathbf{a}_{ij}\mathbf{v}_j)$, where Norm is RMSNorm, normalizing by the root mean square of activations. This replaces the row-stochastic normalization in standard Softmax.
    - **Design Motivation**: In extreme sparsity scenarios where 99% of weights are exactly zero, standard mean-variance normalization might be unstable due to small denominators. RMSNorm is more robust to changes in weight distribution by relying only on magnitude rather than mean and variance.

### Loss & Training

The paper pretrains a GPT-2-162M model from scratch on the FineWebEdu-10B dataset. Core hyperparameter settings: $\kappa=1$ (spurious survivor control), $\beta=1$ (threshold scaling), $p=2$ (power). A linear warmup + cosine decay learning rate schedule is used, with a max learning rate of $10^{-3}$, min $10^{-4}$, and weight decay of 0.1. NTK-aware RoPE scaling is used for long-context extension, along with an additional 500 steps of fine-tuning.

## Key Experimental Results

### Main Results

| Method | Val Loss | HellaSwag | ARC-Easy | ARC-Challenge | OpenBookQA | PIQA | Winogrande | Sparsity |
|------|---------|----------|----------|---------------|-----------|------|-----------|--------|
| Softmax | 3.1196 | 0.345 | 0.526 | 0.223 | 0.180 | 0.641 | 0.490 | 0% |
| Gated Softmax | 3.1489 | 0.330 | 0.474 | 0.194 | 0.162 | 0.620 | 0.500 | 0% |
| Entmax | 3.1941 | 0.342 | 0.508 | 0.194 | 0.198 | 0.632 | 0.523 | 43% |
| ReLA | 3.1657 | 0.329 | 0.512 | 0.226 | 0.194 | 0.634 | 0.509 | 94% |
| Diff Softmax | 3.1941 | 0.336 | 0.509 | 0.225 | 0.178 | 0.648 | 0.514 | 0% |
| Dex | 3.1349 | 0.339 | 0.492 | 0.215 | 0.172 | 0.640 | 0.519 | 0% |
| **Ours** | **3.1190** | 0.337 | 0.524 | 0.220 | 0.216 | 0.628 | 0.489 | **99%** |

Ours achieves the lowest validation loss (3.1190) while achieving 99% exact zero-weight sparsity, far exceeding other methods. Performance is comparable to or better than the Softmax baseline.

**Long-context SCROLLS Evaluation**

| Method | QMSum | SummScreenFD | GovReport | Qasper |
|------|-------|--------------|-----------|--------|
| Softmax | 10.29 | 7.25 | 3.78 | 8.82 |
| Entmax | 11.52 | 10.16 | 4.24 | 11.54 |
| ReLA | 11.20 | 9.14 | 4.42 | 10.77 |
| **Ours** | 11.46 | 9.13 | 5.24 | 11.41 |

Ours is competitive on the long-context SCROLLS benchmark, matching Entmax but avoiding the computational overhead of projection methods.

### Key Findings

- **Attention Sink Elimination**: The sink ratio of the first token $\mathrm{gSinkRatio}(1)$ remains at the uniform distribution baseline as sequence length grows, whereas Softmax rises sharply. The inhibitory behavior of the differential view broadly suppresses high-frequency functional words like "the" while preserving query-relevant selectivity for content words like "quick" or "brown".
- **Depth-dependent Sparsity Distribution**: Early and late layers are highly sparse (zero-weight rate near 100%), while middle layers maintain about 50% activity. This aligns with the understanding that middle layers generate stronger query-key alignment.
- **Hyperparameter Robustness**: $p=2$ is optimal; $p=1$ drops significantly due to the removal of non-linearity, and $p \geq 3$ increases gradient variance. $\beta=1.0$ yields optimal performance, remaining stable within the 0.5-1.0 range.
- **Passkey Retrieval**: At 4,000 tokens, TDA achieves 15% accuracy, surpassing Softmax's 6%. The advantage is more pronounced in multi-needle retrieval (2 and 4 needles).

## Highlights & Insights

- **Elegant combination of theory and practice**: The $\sqrt{\log i / d}$ threshold scaling derived from sub-Gaussian extreme value theory has a solid mathematical foundation and shows significant empirical effects. Theorem 4.3 guarantees the expected spurious survivors per row is $O(1)$ independent of sequence length, and Theorem 4.6 proves consensus spurious survivors decay to $O(1/(i+1))$.
- **Ingenious application of differential strategy**: Unlike other rectified methods, TDA cleverly reuses the idea from Differential Transformer but applies it to two independent thresholded views rather than Softmax views, avoiding the cost of dense Softmax while gaining the expressive advantage of signed weights.
- **Creative leap from extreme value theory to attention design**: Using standard techniques from extreme value statistics (logarithmic growth of maximums in high dimensions) to directly guide attention threshold parameterization is a cross-disciplinary insight rarely seen in attention design.

## Limitations & Future Work

**Limitations of Prior Work (as acknowledged by authors)**: Experiments were primarily conducted on small models (GPT-2-162M); performance at the multi-billion parameter scale remains to be verified. Extremely aggressive thresholds might lead to "dead heads," where a head has no survivors at all positions.

**Self-identified limitations**: (1) The sub-Gaussian assumption is empirically validated, but its tightness for highly non-linear Transformer latent states is not fully clear; (2) The independence assumption between views might be partially compromised during training (cross-view correlation rose from 0.0752 to 0.1231), with unknown long-term effects; (3) 15% absolute accuracy on 4,000-token Passkey retrieval still has room for improvement.

**Future Work**: (1) Explore layer-wise or head-wise adaptive threshold scheduling; (2) Validate TDA scalability on larger (billion-parameter) models; (3) Combine with other long-context methods (e.g., block-wise attention, memory mechanisms).

## Related Work & Insights

- **vs Rectified Attention (ReLA)**: ReLA naturally eliminates sinks by removing the sum-to-one constraint but suffers from noise accumulation due to a lack of length awareness; TDA retains the sparsity advantages of rectification but actively controls noise via $\sqrt{\log i / d}$ thresholds and differential views.
- **vs Projection Sparsity (Entmax)**: Entmax achieves sparsity via iterative projection but is computationally expensive (sorting cost) and still imposes the sum-to-one constraint; TDA achieves $O(1)$ spurious survivors via threshold truncation without normalization constraints.
- **vs Length-adaptive Softmax (SSMax)**: SSMax adapts to length by scaling dot products but still uses Softmax; TDA re-architects the attention mechanism structurally, fundamentally changing the nature of weight distribution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First combination of extreme value theory and attention design; length-adaptive threshold concept is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers standard LM, long context, Passkey, hyperparameter sensitivity, and efficiency; however, small models limit representativeness.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, smooth flow from problem statement to theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Directly addresses fundamental bottlenecks of long-context Transformers; 99% sparsity brings actual efficiency gains; open-source Triton kernel facilitates adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](../../ICML2026/llm_efficiency/stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)

</div>

<!-- RELATED:END -->
