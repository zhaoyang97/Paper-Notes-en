---
title: >-
  [Paper Note] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs
description: >-
  [ACL2026][Model Compression][LoRA] This paper overturns the common assumption that "Multiple LoRAs should share the A matrix," demonstrating that the similarity of A primarily stems from identical initialization rather t…
tags:
  - "ACL2026"
  - "Model Compression"
  - "LoRA"
  - "Parameter Sharing"
  - "Multi-task Fine-Tuning"
  - "Federated Learning"
  - "Communication Compression"
date: 2026-05-08
content_hash: 10ccaa8f3829c1fc
---

# Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs

**Conference**: ACL2026 Findings  
**arXiv**: [2509.25414](https://arxiv.org/abs/2509.25414)  
**Code**: https://github.com/OptMN-Lab/ALoRA  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning / Federated Fine-Tuning  
**Keywords**: LoRA, Parameter Sharing, Multi-task Fine-Tuning, Federated Learning, Communication Compression

## TL;DR
This paper overturns the common assumption that "Multiple LoRAs should share the A matrix," demonstrating that the similarity of A primarily stems from identical initialization rather than shared knowledge. It proposes ALoRA / Fed-ALoRA, which shares the B matrix to balance performance, equilibrium, and communication efficiency in multi-task and federated fine-tuning.

## Background & Motivation
**Background**: LLM fine-tuning typically employs parameter-efficient methods like LoRA, which represent weight updates as a low-rank decomposition $\Delta W=BA$, freezing the original model to train only A and B. For multi-task, multi-domain, or multi-client data, a single LoRA often lacks capacity, leading to the emergence of structures like multiple LoRA experts, LoRA MoE, and federated LoRA aggregation.

**Limitations of Prior Work**: While multiple LoRAs enhance adaptation capacity, they introduce additional parameter, computation, and communication overheads. Existing parameter-sharing methods often observe similarity between A matrices of different LoRAs and thus share A while keeping individual B matrices, as seen in HydraLoRA and FedSA-LoRA.

**Key Challenge**: The similarity of A matrices does not necessarily imply that A carries shared knowledge. If this similarity originates solely from identical random initialization, sharing A might not only fail to facilitate knowledge transfer but also restrict tasks from exploring diverse feature subspaces. The parameters truly worth sharing might be B.

**Goal**: The authors first re-examine the similarity and evolution patterns of A and B matrices during training; then compare the knowledge transfer efficacy of sharing A versus sharing B; finally, they propose a shared-B architecture suitable for multi-task and federated scenarios.

**Key Insight**: Instead of starting with complex MoE or routers, the paper returns to the matrix decomposition of LoRA itself to ask a fundamental question: which matrix should be shared in multi-LoRA settings? This choice directly impacts model capacity allocation and federated communication costs.

**Core Idea**: Use multiple A matrices to handle different feature projections while using a shared B matrix to aggregate and transfer knowledge. In federated fine-tuning, only B is communicated, and matrix decomposition is employed to support heterogeneous ranks.

## Method

### Overall Architecture
The paper is divided into "Diagnosis" and "Methodology" sections. The diagnosis section controls the initialization seeds of A on models like LLaMA2-7B, discovering that high similarity of A only appears under identical initialization, vanishing when different initializations are used. Further analysis of matrix magnitude and direction changes before and after training shows that B undergoes more significant directional changes. Based on this, the methodology section proposes ALoRA, using multiple $A_i$ and one shared $B$ for multi-task fine-tuning, and Fed-ALoRA, which allows clients to update complete LoRAs locally but only upload and aggregate B-related parameters.

### Key Designs
1. **Reinterpreting A/B Specialization from Initialization and Training Dynamics**:

	- Function: To determine whether A or B is the more likely "carrier of shared knowledge."
	- Mechanism: The authors compare LoRA module similarity across same/different tasks and same/different initializations, using principal angle-based similarity to measure matrix subspaces. Results show that A is highly similar with the same initialization; however, with different initializations, A loses similarity even for the same task. Training dynamic analysis further shows A primarily preserves the projection structure, while B reflects more obvious changes in magnitude and direction.
	- Design Motivation: If the similarity of A is a residue of initialization, structural assumptions based on "sharing A to share knowledge" are unstable. The authors treat A as a feature projector and B as a domain/task knowledge aggregator.

2. **ALoRA: Asymmetric Multi-task LoRA with Multiple A and Single B**:

	- Function: To maintain task diversity in multi-task fine-tuning while facilitating transfer through shared parameters.
	- Mechanism: The forward pass of ALoRA is $y=W_0x+B\sum_i w_i A_i x$. Each $A_i$ serves as an expert to explore different low-rank feature subspaces, while the shared $B$ fuses these projected features into the output space. Routing weights are provided by a linear gate $w=softmax(W_gx)$, allowing adapters to be merged dynamically during inference.
	- Design Motivation: Sharing A causes tasks to compete for the same feature projection, leading to smaller gradients and more gradient conflicts; retaining multiple A matrices provides sufficient exploration space for different tasks, while sharing B aligns with its role in knowledge aggregation.

3. **Fed-ALoRA: Federated Fine-Tuning with B-only Aggregation and Heterogeneous Decomposition**:

	- Function: To reduce communication costs in federated LoRA fine-tuning while supporting different ranks across clients.
	- Mechanism: In homogeneous scenarios, each client trains $(A_i, B_i)$ locally but only uploads $B_i$. The server aggregates these into a global $B_0$ and broadcasts it back. Communication volume drops from $(d_{in}+d_{out})r$ for full-LoRA to $d_{out}r$. In heterogeneous scenarios where client ranks $r_i$ vary, $B_i$ cannot be averaged directly. The authors formulate the update as $(B_{i0}+B_{i2}B_{i1})M_iA_i$. The server reconstructs $B_{i2}B_{i1}$ with identical dimensions for aggregation and uses an accumulator to save historical global updates.
	- Design Motivation: FedSA-LoRA is only suitable for homogeneous ranks, and methods like ZeroPadding/FLoRA have high communication costs. Fed-ALoRA unifies knowledge transfer and communication compression through the shared-B structure.

### Loss & Training
ALoRA and Fed-ALoRA do not change downstream task losses; the core objective remains language modeling or supervised task loss. Multi-task ALoRA trains the router, multiple $A_i$, and the shared $B$. The homogeneous federated version optimizes local LoRA loss and uploads $B_i$ per round. The heterogeneous version optimizes local $(A_i, M_i, B_{i1}, B_{i2})$, with the server performing FedAvg-style aggregation on reconstructed B updates.

## Key Experimental Results

### Main Results
The authors first conducted a direct comparison: in federated fine-tuning across 8 FLAN task clients, sharing B significantly outperformed sharing A.

| Setting | Shared Parameters | Avg. ROUGE-1 | Relative Conclusion |
|------|----------|--------------|----------|
| Homogeneous rank | A | 44.30 | Weak knowledge transfer for shared A |
| Homogeneous rank | B | 66.32 | 49.71% higher than shared A |
| Heterogeneous rank | A | 40.76 | Harder to aggregate under heterogeneity |
| Heterogeneous rank | B | 50.30 | 23.41% higher than shared A |

In multi-task fine-tuning, ALoRA achieved the most balanced results in commonsense reasoning and cross-domain NLP. Lower $\Delta m\%$ is better, indicating smaller average performance loss relative to the single-task baseline.

| Benchmark | Model | Method | Avg. | $\Delta m\%$ | Key Comparison |
|-----------|------|------|------|--------------|----------|
| Commonsense | LLaMA3-8B | HydraLoRA | 84.57 | 0.32 | Strong baseline sharing A |
| Commonsense | LLaMA3-8B | ALoRA | 84.81 | 0.04 | Higher average with almost no task imbalance |
| Commonsense | Qwen2-7B | HydraLoRA | 86.09 | 1.32 | Sharing A is unstable on Qwen |
| Commonsense | Qwen2-7B | ALoRA | 86.47 | 0.91 | Better average and equilibrium |
| Cross-domain NLP | LLaMA2-7B | HydraLoRA | 66.45 | -6.39 | Strong multi-LoRA baseline |
| Cross-domain NLP | LLaMA2-7B | ALoRA | 67.13 | -8.33 | Avg. +0.68 with stronger equilibrium |
| Cross-domain NLP | Qwen2-7B | HydraLoRA | 80.03 | -7.14 | Shared A baseline |
| Cross-domain NLP | Qwen2-7B | ALoRA | 80.46 | -7.98 | Avg. +0.43 |

### Ablation Study
Federated fine-tuning experiments demonstrate the communication advantages of Fed-ALoRA. Params represents the average communication parameters per client per round in millions.

| Setting | Model | Method | Avg. | $\Delta m\%$ | Params | Description |
|------|------|------|------|--------------|--------|------|
| Homogeneous | LLaMA2-7B | FedIT | 82.47 | -3.92 | 8.39 | Full LoRA aggregation |
| Homogeneous | LLaMA2-7B | FedSA-LoRA | 81.15 | -2.21 | 4.19 | Sharing A, low comms but weak performance |
| Homogeneous | LLaMA2-7B | Fed-ALoRA | 82.51 | -4.29 | 4.19 | Performance parity with FedIT, half comms |
| Homogeneous | Qwen2-7B | FedIT | 82.80 | -0.05 | 6.42 | Full LoRA aggregation |
| Homogeneous | Qwen2-7B | Fed-ALoRA | 84.30 | -2.05 | 3.21 | Highest average, half comms |
| Heterogeneous | LLaMA2-7B | ZeroPadding | 82.29 | -0.91 | 49.28 | Requires padding to max rank |
| Heterogeneous | LLaMA2-7B | Fed-ALoRA | 82.50 | -1.07 | 12.12 | Communication reduced by ~75% |
| Heterogeneous | Qwen2-7B | ZeroPadding | 83.32 | 0.06 | 37.73 | Heterogeneous full aggregation baseline |
| Heterogeneous | Qwen2-7B | Fed-ALoRA | 84.13 | -1.02 | 9.23 | Superior performance and communication |

### Key Findings
- High similarity of the A matrix is not evidence of shared knowledge, but a byproduct of identical initialization; similarity of A disappears rapidly under different initializations.
- The B matrix undergoes more directional changes during training, behaving like the part that truly absorbs task and domain knowledge.
- Sharing A leads to smaller gradients and more gradient conflicts (referred to as "lazy learning"), while sharing B allows multiple A matrices to maintain feature subspace diversity.
- The value of Fed-ALoRA lies not just in performance, but in halving homogeneous communication and reducing heterogeneous communication by ~75% while improving average scores and task equilibrium.

## Highlights & Insights
- The core highlight of this paper is the "reinterpretation of existing observations." While many works share A upon seeing its similarity, this paper points out via initialization control experiments that this interpretation may be untenable, which is more enlightening than simply proposing a new structure.
- The design of ALoRA is elegant: multiple A, one B, and a lightweight router, without introducing complex expert transformations, yet delivering stable gains across multiple benchmarks.
- Heterogeneous Fed-ALoRA is highly practical. Real-world federated devices often have different computing power and rank budgets; the ability to aggregate LoRA updates of different sizes is closer to deployment needs than homogeneous-only settings.
- This work reminds the PEFT community not to look only at parameter counts but also at the functional roles of parameters within decomposed structures. The left and right factors of low-rank matrices are not symmetric.

## Limitations & Future Work
- The paper primarily discusses within the A/B factorization framework of LoRA; whether the conclusions transfer to DoRA, AdaLoRA, LoHa, or more complex adapter structures remains to be verified.
- Multi-task experiments cover commonsense, math, and FLAN-style NLP, but do not yet cover more complex LLM adaptation scenarios like long context, code, or tool calling.
- The router uses a simple linear gating mechanism. A more powerful router might further improve ALoRA but could introduce load imbalance and training instability.
- Federated experiments are based on offline benchmark construction; client dropouts, fluctuating non-IID levels, privacy budgets, and secure aggregation costs in real-world federated systems have not been systematically evaluated.

## Related Work & Insights
- **vs HydraLoRA**: HydraLoRA shares A and keeps multiple B, assuming A carries shared knowledge; ALoRA reverses this by sharing B and keeping multiple A, positing B is better for knowledge transfer, resulting in better average scores and equilibrium in experiments.
- **vs FedSA-LoRA**: FedSA-LoRA aggregates A in federated settings and only supports homogeneous ranks; Fed-ALoRA aggregates B and supports heterogeneous ranks through decomposition.
- **vs ZeroPadding / FLoRA**: These methods handle heterogeneous LoRAs but with high communication volume; Fed-ALoRA uses the shared-B structure to suppress communication costs while maintaining or improving performance.
- **Insight**: Future PEFT research could systematically study the functional roles of different parameter subspaces instead of treating all trainable parameters as interchangeable low-rank blocks.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not a completely new PEFT paradigm, but provides a powerful reversal of the multi-LoRA parameter sharing direction.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multi-task, cross-domain, federated homogeneous and heterogeneous settings; real-world large-scale federated deployment experiments could further strengthen it.
- Writing Quality: ⭐⭐⭐⭐☆ Natural connection between motivation, diagnosis, and methodology; mathematical notation is slightly dense but overall clear.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for multi-LoRA, federated PEFT, and communication-efficient fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](../../ICML2026/model_compression/compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)

</div>

<!-- RELATED:END -->
