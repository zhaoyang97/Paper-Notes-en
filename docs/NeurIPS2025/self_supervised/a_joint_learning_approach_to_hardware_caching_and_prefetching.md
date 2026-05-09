---
title: >-
  [Paper Note] A Joint Learning Approach to Hardware Caching and Prefetching
description: >-
  [NeurIPS 2025 (ML for Systems Workshop)][Self-Supervised Learning][Cache replacement] This paper proposes a joint training framework that unifies hardware cache replacement and prefetching policies. By constructing shared feature representations via a joint encoder and contrastive learning, the framework breaks the performance bottleneck imposed by independently trained policies.
tags:
  - NeurIPS 2025 (ML for Systems Workshop)
  - Self-Supervised Learning
  - Cache replacement
  - prefetching
  - joint learning
  - shared representations
  - contrastive learning
date: 2026-05-08
content_hash: f60e83f4d4a1ceb0
---

# A Joint Learning Approach to Hardware Caching and Prefetching

**Conference**: NeurIPS 2025 (ML for Systems Workshop)  
**arXiv**: [2510.10862](https://arxiv.org/abs/2510.10862)  
**Code**: None  
**Area**: ML for Systems / Hardware Caching  
**Keywords**: Cache replacement, prefetching, joint learning, shared representations, contrastive learning

## TL;DR

This paper proposes a joint training framework that unifies hardware cache replacement and prefetching policies. By constructing shared feature representations via a joint encoder and contrastive learning, the framework breaks the performance bottleneck imposed by independently trained policies.

## Background & Motivation

In modern computing systems, scheduling policies for various components—such as cache replacement and data prefetching—are increasingly transitioning from hand-crafted heuristics to learning-based approaches. These learned policies leverage feature extraction, historical trend analysis, and future behavior prediction to promise sustained high performance amid increasingly complex workloads and hardware evolution.

However, a critical limitation persists in existing methods: **policies are typically trained in isolation**. When multiple independently trained policies are combined at deployment, they may fail to achieve optimal joint performance. The authors argue that in the hardware caching domain, cache replacement and data prefetching exhibit **bidirectional mutual dependence**:

**Prefetching affects replacement**: The prefetching policy determines which data is preloaded into the cache, directly altering the cache content distribution that the replacement policy must handle.

**Replacement affects prefetching**: The replacement policy determines which cache lines are evicted, indirectly influencing the effectiveness evaluation and decision basis of the prefetching policy.

This bidirectional dependency makes independently trained policies prone to conflict or suboptimal behavior when composed.

## Method

### Overall Architecture

The authors propose a joint learning approach based on **shared representations**. The core idea is to have the cache replacement and prefetching policies share part of their feature encoding, enabling each to be aware of the other's existence and requirements during training.

The overall pipeline consists of:
1. Extracting raw features from memory access sequences (e.g., program counter PC, memory addresses, access patterns)
2. Encoding raw features into general-purpose embeddings via a shared representation module
3. Task-specific heads (Replacement Head / Prefetch Head) making decisions based on the shared embeddings
4. Joint optimization of losses from both tasks

### Key Designs

The authors propose two approaches for constructing shared representations:

**Approach 1: Joint Encoder**
- A single neural network encoder extracts features for both tasks simultaneously
- Encoder parameters are shared across gradient updates from both tasks
- Separate decision heads (MLPs) are appended at the top for replacement and prefetching respectively
- Advantage: Simple structure with natural parameter sharing for information exchange
- Disadvantage: Potential gradient conflicts between tasks

**Approach 2: Contrastive Learning**
- Separate encoders are trained for each task, but their embedding spaces are constrained via a contrastive learning objective
- The contrastive loss encourages both encoders to produce similar representations for the same memory access sequence
- Task-specific information is preserved in each encoder
- Advantage: More flexible, avoids direct parameter interference
- Disadvantage: Incurs additional training overhead for contrastive learning

### Loss & Training

The total loss for joint training is:

$$L_{total} = L_{replacement} + \lambda_1 L_{prefetch} + \lambda_2 L_{contrastive}$$

where:
- $L_{replacement}$: loss for the cache replacement task (e.g., predicting the optimal eviction candidate)
- $L_{prefetch}$: loss for the prefetching task (e.g., predicting the next required memory block)
- $L_{contrastive}$: contrastive regularization term (used only in Approach 2)
- $\lambda_1, \lambda_2$: balancing weights

Training is performed end-to-end, with memory access traces collected from a simulated cache environment.

## Key Experimental Results

### Main Results

The authors compare independently trained vs. jointly trained policies in a standard cache simulation environment:

| Method | Cache Hit Rate (%) | Prefetch Accuracy (%) | IPC Improvement (%) |
|--------|-------------------|----------------------|---------------------|
| LRU + No Prefetch | 62.3 | - | Baseline |
| Learned Replacement (independent) | 68.7 | - | +4.2 |
| Learned Prefetch (independent) | 62.3 | 71.5 | +6.8 |
| Independent Replacement + Independent Prefetch | 69.1 | 71.2 | +9.3 |
| Joint Encoder | 71.4 | 73.8 | +12.1 |
| Contrastive Learning | 70.9 | 74.1 | +11.8 |

### Ablation Study

Effect of varying degrees of sharing on performance:

| Shared Layers | Cache Hit Rate (%) | Prefetch Accuracy (%) | Training Time (relative) |
|---------------|-------------------|----------------------|--------------------------|
| 0 (fully independent) | 69.1 | 71.2 | 1.0x |
| 1 layer shared | 69.8 | 72.1 | 1.1x |
| 2 layers shared | 70.6 | 73.2 | 1.15x |
| All shared (Joint) | 71.4 | 73.8 | 1.2x |
| Contrastive Learning | 70.9 | 74.1 | 1.3x |

### Key Findings

1. **Joint training substantially outperforms naive combination**: The Joint Encoder improves cache hit rate by approximately 2.3 percentage points and IPC gain from 9.3% to 12.1% compared to simply combining independently trained policies.
2. **Both sharing strategies offer complementary advantages**: The Joint Encoder achieves slightly higher cache hit rate, while contrastive learning yields marginally better prefetch accuracy.
3. **Performance scales with sharing depth**: Greater numbers of shared layers generally lead to better performance, albeit with diminishing returns.
4. **Training overhead is manageable**: Joint training increases total training time by only 20–30%.

## Highlights & Insights

1. **Sharp problem identification**: The paper accurately identifies the bidirectional dependency between cache replacement and prefetching—a neglected yet important issue.
2. **Well-motivated design**: Two distinct shared representation schemes are proposed to accommodate different deployment scenarios.
3. **Systems-level thinking**: Introducing multi-task learning into system component design opens a new research direction.
4. **Strong generalizability**: The approach naturally extends to other system components with coupled policies.

## Limitations & Future Work

1. **Preliminary nature**: The paper explicitly acknowledges these as "promising preliminary results," with limited experimental scale and dataset coverage.
2. **Insufficient workload diversity**: Validation is restricted to a limited set of memory access patterns, lacking evaluation on diverse real-world workloads.
3. **Limited baseline comparisons**: No comparison against other multi-task learning methods such as MoE or task-specific adapters.
4. **Gradient conflict**: The Joint Encoder approach may suffer from conflicting gradient directions across tasks, a concern not thoroughly addressed in the paper.
5. **Deployment considerations**: Inference latency and area overhead of the joint model on real hardware are not discussed.

## Related Work & Insights

- **Learned cache replacement**: Works such as Glider and PARROT have demonstrated that learned policies can surpass classical methods like LRU and DRRIP.
- **Learned prefetching**: Works such as Voyager and TransFetch leverage sequence models for address prediction.
- **Multi-task learning**: The shared representation idea is grounded in established multi-task learning paradigms from NLP and CV.
- **Insights**: Coupled optimization of system components is a direction worthy of deeper exploration; similar ideas could extend to scheduling + memory management, compiler optimization + runtime adaptation, and related settings.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel problem framing, though methods are relatively standard)
- Technical Depth: ⭐⭐⭐ (Early-stage work with limited technical detail)
- Experimental Thoroughness: ⭐⭐⭐ (Workshop paper; limited scale but sufficient for proof of concept)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and well-organized structure)
- Overall: ⭐⭐⭐☆ (An interesting directional contribution, though overall completeness remains preliminary)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[NeurIPS 2025\] Long-Tailed Recognition via Information-Preservable Two-Stage Learning](long-tailed_recognition_via_information-preservable_two-stage_learning.md)

<!-- RELATED:END -->
