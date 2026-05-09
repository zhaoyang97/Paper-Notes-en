---
title: >-
  [Paper Note] MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm
description: >-
  [AAAI 2026][LLM Efficiency][test-time adaptation] This paper proposes MoETTA, a test-time adaptation framework that reparameterizes LayerNorm into multiple structurally decoupled expert branches. A routing mechanism assigns samples from different domains to different experts, enabling multi-directional parameter updates and overcoming the limitations of a single adaptation path under mixed distribution shifts. The paper also introduces two more realistic evaluation benchmarks—potpourri and potpourri+—and achieves state-of-the-art performance across all settings.
tags:
  - AAAI 2026
  - LLM Efficiency
  - test-time adaptation
  - mixed distribution shifts
  - Mixture-of-Experts
  - LayerNorm
  - Vision Transformer
date: 2026-05-08
content_hash: 555427eec4319b40
---

# MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm

**Conference**: AAAI 2026
**arXiv**: [2511.13760](https://arxiv.org/abs/2511.13760)
**Code**: [GitHub](https://github.com/AnikiFan/MoETTA)
**Area**: Test-Time Adaptation / Domain Adaptation
**Keywords**: test-time adaptation, mixed distribution shifts, Mixture-of-Experts, LayerNorm, Vision Transformer

## TL;DR

This paper proposes MoETTA, a test-time adaptation framework that reparameterizes LayerNorm into multiple structurally decoupled expert branches. A routing mechanism assigns samples from different domains to different experts, enabling multi-directional parameter updates and overcoming the limitations of a single adaptation path under mixed distribution shifts. The paper also introduces two more realistic evaluation benchmarks—potpourri and potpourri+—and achieves state-of-the-art performance across all settings.

## Background & Motivation

Test-time adaptation (TTA) adjusts model parameters using unlabeled data at inference time to mitigate performance degradation caused by distribution shifts. Early methods (e.g., Tent, EATA) primarily target single-domain distribution shifts, where all test samples originate from the same target distribution. In real-world deployments, however, inference batches often contain samples from multiple heterogeneous domains—such as concurrent requests from edge devices—forming **mixed distribution shifts**.

The fundamental limitation of existing methods is that they force all test samples to share a **single adaptation path** (i.e., the same gradient update direction). Empirical analysis reveals that the cosine similarity between accumulated gradients across different domains is only 0.69 (over 15 domains of ImageNet-C), and approaches zero in extreme cases. Theoretical analysis further shows that when domain-specific parameters follow a Gaussian distribution, the expected cosine similarity between accumulated gradients converges to $0.5 + \mathcal{O}(1/d)$, indicating that gradient direction inconsistency is intrinsic and unavoidable.

The core idea of this paper: rather than enforcing a single adaptation direction, multiple experts are used to represent multiple adaptation paths within the model. Samples from different domains are routed to different experts, enabling decoupled, multi-directional parameter updates.

## Method

### Overall Architecture

MoETTA replaces the LayerNorm modules in ViT with MoE-LayerNorm modules. Each MoE-LayerNorm contains a set of structurally decoupled expert branches (LayerNorm parameters) and a router. At inference time, the router selects one expert per sample based on the mean of the input embedding. The selected expert's parameters are added to the frozen pretrained LayerNorm parameters to form sample-specific normalization parameters. The entire framework is jointly optimized with an entropy minimization loss and a load-balancing loss.

### Key Designs

1. **MoE-LayerNorm Module**:

    - Function: Replaces standard LayerNorm with a MoE structure containing multiple experts, where each expert is a set of LayerNorm affine parameters (initialized to zero).
    - Mechanism: For each input embedding, the mean is computed along the token dimension and fed into the router to obtain routing probabilities. The expert with the highest probability is selected. The effective LayerNorm parameters equal the selected expert's parameters plus the frozen pretrained parameters (shared expert). A top-1 routing strategy is adopted, activating only one expert per sample.
    - Design Motivation: (a) LayerNorm parameters as experts are extremely lightweight, satisfying the low-overhead requirements of TTA; (b) top-1 routing avoids parameter interference caused by merging outputs from multiple experts; (c) the shared expert provides domain-invariant knowledge, reducing redundancy among experts and encouraging complementary behavior.

2. **Routing Mechanism and Gradient Propagation**:

    - Function: Enables a trainable router to receive gradient signals under the entropy loss.
    - Mechanism: The router is a linear projection layer. During the forward pass, only the expert with the highest probability $p$ is activated, but the expert output is scaled by $p / p.\text{detach}()$ (which does not affect the forward value but allows gradients to flow to the router). A load-balancing loss $\mathcal{L}_\text{load balancing} = N \times \sum_{i=1}^{N} \bm{F}_i \times \bm{P}_i$ is added to encourage balanced expert utilization.
    - Design Motivation: Routing decisions must be jointly optimized with model predictions, enabling the router to learn to assign samples with similar features to the same expert, so that each expert adapts in a distinct direction.

3. **Sample Filtering and Entropy Reweighting**:

    - Function: Filters high-entropy (unreliable) samples and reweights reliable samples by entropy value.
    - Mechanism: A dynamic threshold $E_\text{max}^t$ is adaptively updated based on historical average entropy. The total loss is $\frac{1}{|\mathcal{S}_t|} \sum_{\bm{x} \in \mathcal{S}_t} \exp[E_0 - \text{Ent}(\bm{x})] \cdot \text{Ent}(\bm{x}) + \alpha_t \sum_{i=1}^{M} \mathcal{L}^i_\text{load balancing}$, where $\alpha_t$ dynamically balances the two loss terms.
    - Design Motivation: Unreliable samples (high entropy) in mixed distributions produce noisy gradients; filtering and reweighting improve gradient signal quality.

4. **Potpourri / Potpourri+ Benchmarks**:

    - Function: Two more realistic evaluation benchmarks for mixed distribution shifts.
    - Potpourri: Mixes ImageNet-C (synthetic corruptions), ImageNet-R (stylistic renditions), ImageNet-A (adversarial examples), and ImageNet-Sketch (semantic abstractions), covering noise, blur, weather, digital transformations, natural, artistic, and adversarial perturbations.
    - Potpourri+: Extends Potpourri by incorporating ImageNet validation set samples, evaluating catastrophic forgetting when methods handle both in-distribution (ID) and out-of-distribution (OOD) data simultaneously.
    - Design Motivation: Existing benchmarks (mixing only within ImageNet-C) fail to reflect the diverse distribution shift types encountered in real-world deployments.

### Loss & Training

The total loss comprises three components:
- **Entropy loss**: Computes the entropy of the predictive distribution for filtered samples, with exponential reweighting (lower-entropy samples receive higher weights).
- **Load-balancing loss**: Encourages balanced expert utilization and prevents routing collapse to a small number of experts.
- **Dynamic coefficient** $\alpha_t$: Adaptively adjusts the weight ratio between the two loss terms based on changes in average entropy.

Only the router and expert LayerNorm parameters are updated; all other model parameters remain frozen.

## Key Experimental Results

### Main Results

| Model | Setting | No Adapt. | Tent | EATA | SAR | DeYO | MGTTA | BECoTTA | MoETTA |
|-------|---------|-----------|------|------|-----|------|-------|---------|--------|
| ViT-B/16 | Classical | 55.52 | 63.20 | 64.28 | 60.76 | 63.97 | 66.20 | 61.57 | **67.20** |
| ViT-B/16 | Potpourri | 54.18 | 60.99 | 61.99 | 58.71 | 61.66 | 62.98 | 59.08 | **65.12** |
| ViT-B/16 | Potpourri+ | 55.92 | 62.28 | 63.17 | 59.99 | 62.90 | 64.35 | 58.87 | **66.15** |
| ConvNeXt-B | Classical | 54.81 | 58.88 | 64.50 | 61.67 | 64.32 | - | 50.16 | **67.40** |
| ConvNeXt-B | Potpourri | 53.91 | 58.23 | 62.69 | 61.16 | 62.46 | - | 28.28 | **65.70** |
| ConvNeXt-B | Potpourri+ | 55.69 | 59.69 | 63.94 | 62.72 | 63.57 | - | 48.92 | **66.68** |

MoETTA achieves the best performance across all 6 settings, outperforming the second-best method MGTTA by 2.14% on the more challenging Potpourri setting. Notably, MoETTA requires no additional pretraining samples, whereas MGTTA requires extra OOD and ID samples.

### Ablation Study

| Configuration | Classical | Potpourri | Potpourri+ | Average |
|---------------|-----------|-----------|------------|---------|
| Full method | 67.25 | 65.14 | 66.21 | 66.20 |
| w/o sample filtering | 67.04 | 64.01 | 57.61 | 62.89 |
| w/o entropy reweighting | 62.86 | 60.51 | 61.79 | 61.72 |
| w/o load-balancing loss | 26.27 | 16.27 | 21.29 | 21.28 |
| w/o router gradient | 65.17 | 62.80 | 63.92 | 63.96 |
| w/o sample-level routing | 28.69 | 28.60 | 24.96 | 27.42 |
| w/o MoE-LayerNorm | 22.38 | 17.94 | 26.93 | 22.42 |
| w/o layer-wise routing | 17.40 | 27.09 | 15.18 | 19.89 |

### Key Findings

- **Load-balancing loss is critical**: Removing it causes performance to collapse from 66.20% to 21.28%, with complete routing collapse.
- **Sample-level routing is essential**: Forcing all samples in a batch to use the same expert leads to performance collapse (27.42%), demonstrating the necessity of input-adaptive routing.
- **Expert parameters naturally diverge**: During adaptation, the cosine similarity between expert parameters within the same MoE-LayerNorm gradually decreases, particularly in shallower layers.
- **Shallow layers should be frozen**: Freezing the first 5 LayerNorm layers (preserving low-level domain-invariant features) yields the best performance.
- **Moderate computational overhead**: Runtime is 247% of the no-adaptation baseline, comparable to Tent (226%) and EATA (239%).

## Highlights & Insights

- **Solid motivational analysis**: Empirical analysis of gradient direction cosine similarity and theoretical derivations clearly reveal the intrinsic limitations of a single adaptation path under mixed distribution shifts.
- **Extremely lightweight design**: Only LayerNorm parameters serve as experts; per-sample activated parameters total only 0.23M, far fewer than CoTTA's 86.42M.
- **Well-designed Potpourri benchmark**: The combination of synthetic, natural, artistic, and adversarial shift types better approximates real-world deployment conditions compared to existing benchmarks.
- **Expert diversity emerges naturally** rather than being predefined, arising from the joint optimization of structural decoupling and the routing mechanism.

## Limitations & Future Work

- Validation is limited to classification tasks; performance on dense prediction tasks such as detection and segmentation remains unexplored.
- The number of experts (9 is optimal) and replacement strategy require re-tuning for different architectures.
- The router uses a simple linear projection; more sophisticated routing strategies may further improve performance.
- The continual learning scenario (where domains gradually evolve) is not considered; combining with methods such as BECoTTA warrants further investigation.
- Category spaces across datasets in the Potpourri benchmark are not fully aligned (e.g., ImageNet-A covers only 200 classes), which may introduce evaluation bias.

## Related Work & Insights

This paper introduces the MoE paradigm into the TTA domain, offering a contrast to BECoTTA (which uses LoRA as experts for continual TTA). MoETTA selects LayerNorm as the expert unit—lighter than LoRA and directly corresponding to the module most commonly updated in TTA (Tent also updates only BN/LN parameters). This strategy of "building MoE on top of existing adaptation targets" is worth extending to other adaptive learning problems. The load-balancing loss, borrowed from Switch Transformer in the large language model literature, proves equally critical in the TTA context.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](../../ICLR2026/llm_efficiency/xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)
- [\[AAAI 2026\] InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)
- [\[NeurIPS 2025\] The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](../../NeurIPS2025/llm_efficiency/the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)

<!-- RELATED:END -->
