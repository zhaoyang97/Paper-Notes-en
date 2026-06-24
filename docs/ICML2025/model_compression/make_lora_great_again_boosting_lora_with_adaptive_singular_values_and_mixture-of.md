---
title: >-
  [Paper Note] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment
description: >-
  [ICML 2025][Model Compression][LoRA] GOAT significantly enhances LoRA performance without altering the training algorithm or main architecture via "SVD-segmented initialized LoRA-MoE + theoretically derived scaling alignment", achieving SOTA on 25 tasks and substantially narrowing the gap with Full FT.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "LoRA"
  - "Mixture-of-Experts"
  - "SVD Initialization"
  - "Optimization Alignment"
  - "PEFT"
date: 2026-05-08
content_hash: b0f4ea2a4f322e65
---

# Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment

**Conference**: ICML 2025  
**arXiv**: [2502.16894](https://arxiv.org/abs/2502.16894)  
**Code**: [https://github.com/Facico/GOAT-PEFT](https://github.com/Facico/GOAT-PEFT)  
**Area**: model_compression (LLM Parameter-Efficient Fine-Tuning / LoRA)  
**Keywords**: LoRA, Mixture-of-Experts, SVD Initialization, Optimization Alignment, PEFT

## TL;DR
GOAT significantly enhances LoRA performance without altering the training algorithm or main architecture via "SVD-segmented initialized LoRA-MoE + theoretically derived scaling alignment", achieving SOTA on 25 tasks and substantially narrowing the gap with Full FT.

## Background & Motivation

### 1. Background
- LoRA compresses full parameter updates into low-rank updates via $W = W_0 + BA$, significantly reducing GPU memory and training costs.
- However, the typical initialization of LoRA (random $A$, zero $B$) provides a weak prior, lacking sufficient information at the starting point of optimization.
- Another direction incorporates LoRA into MoE, allowing different experts to learn different subspaces, but this introduces complexities in routing and gradient dynamics.

### 2. Limitations of Prior Work
- SVD initialization methods (e.g., selecting only head or tail singular values) typically rely on "static segment selection", which struggles to accommodate task differences.
- An optimization gap exists between LoRA and Full FT. This gap is further amplified in LoRA-MoE, where each expert possesses a lower rank.
- Directly inserting SVD priors into LoRA-MoE leads to weight alignment issues, which previous zero-initialization methods failed to address.

### 3. Key Challenge
- Retaining the structural prior of pre-trained weights while maintaining the low cost of LoRA;
- Utilizing MoE for "input-adaptive capability" while avoiding the weight and gradient mismatch caused by routing.

### 4. Goal
- Enabling the model to adaptively leverage different SVD segments based on the input, rather than fixing a single segment.
- Establishing an alignable optimization target between LoRA-MoE and Full FT MoE.
- Improving convergence and performance significantly using only lightweight modifications such as scaling.

### 5. Mechanism
- Initialization: Distribute different SVD segments to different experts, prompting the router to select the "more relevant pre-training prior" based on the input.
- Optimization: Perform equivalent weight alignment followed by equivalent gradient alignment, deriving the optimal scaling factor.

## Method

### Overall Architecture
GOAT can be viewed in two steps:
1. Construct LoRA-MoE expert initialization in a structured SVD manner (Adaptive Priors).
2. Address weight and gradient discrepancies between LoRA-MoE and Full FT MoE via theoretical alignment strategies (Optimization Alignment).

Basic LoRA formulation:
- $W = W_0 + sBA$, where $s$ is the scaling factor.

MoE output formulation:
- $\mathrm{MoE}(x)=\sum_{i=1}^{E} w^i(x)W^i(x)$.

LoRA-MoE formulation:
- $\mathrm{MoE}_{\mathrm{LoRA}}(x)=W(x)+\sum_{i=1}^{E}w^i(x)\big(sB^iA^i(x)\big)$.

### Key Designs

#### Key Design 1: SVD Segmentation and Input-Adaptive Priors

##### Function
- After performing SVD on the pre-trained weight $W_0$, rather than selecting only the head or tail segments, different singular value segments are distributed evenly across different experts.

##### Design Motivation
- The paper observes that different datasets prefer different optimal SVD segments, and the middle segments are equally critical under certain settings. This indicates that a "single static segment" is not a universally optimal prior.

##### Implementation
- Segmenting $W_0$:
	$$W_0=\sum_{i=0}^{l}U_i\Sigma_iV_i^\top$$
- Initializing each expert with a different segment:
	$$B_0^i=\sqrt{\frac{1}{s\rho}}U_i\Sigma_i^{1/2},\;A_0^i=\sqrt{\frac{1}{s\rho}}\Sigma_i^{1/2}V_i^\top$$
- The router activates experts under the top-$k$ mechanism based on the input, which is equivalent to "dynamically selecting SVD priors".

##### Differences from Prior Work
- PiSSA/MiLoRA/KaSA lean towards "single-segment or static-segment strategies", whereas GOAT utilizes "multi-segment parallelization + routing selection".

#### Key Design 2: Initialization Weight Alignment (Introducing $W_{res}$)

##### Function
- During the initialization of MoE with SVD priors, an extra residual constant $W_{res}$ is introduced to ensure the initial equivalent weight remains close to $W_0$.

##### Formula
- Initial alignment objective:
	$$\tilde W_0 = W_0 - W_{res} + \sum_{i=1}^{E} w^i(x)sB_0^iA_0^i \approx W_0$$
- The closed-form solution is acquired by minimizing the expected squared error:
	$$W_{res}^+ = \frac{s}{E}\sum_{i=1}^{E}B_0^iA_0^i$$

##### Intuition
- Instead of enforcing strict cancellation of all expert outputs for every sample, alignment is achieved in an "expectation sense" to reduce the offset caused by randomized routing.

#### Key Design 3: Gradient Alignment and Optimal Scaling

##### Definition
- Equivalent weight: $\tilde W \triangleq W + sBA$.
- Equivalent gradient: $\tilde g \triangleq \frac{\partial L}{\partial \tilde W}$.

##### Key Findings
- Under SVD initialization, even if the weight magnitude of $sBA$ can be made insensitive to $s$, the equivalent gradient remains proportional to $s$. Thus, scaling affects the convergence rate.
- For zero-initialized LoRA-MoE, the optimal scaling can be derived as:
	$$s = \sqrt{\frac{3n\eta}{r}}$$
- When $n \gg r$, $s>2$ often holds, explaining the empirical phenomenon where the "default smaller scaling is insufficient".

### Loss & Training
- The paper maintains the standard training pipeline of LoRA-MoE and the MoE balance loss without introducing additional complex training frameworks.
- The modifications are concentrated on the initialization strategy, $W_{res}$ alignment, and scaling strategies.

## Key Experimental Results

> *Note: This summary is strictly based on the locally cached text. The cached segments provide core trends and some specific values but omit full main table details. Therefore, the main results table is presented with "task ranges + conclusion-level results", while the ablation table contains legible numbers and qualitative trends from the cache.*

### Main Results

| Experimental Scope | Baselines | Ours (GOAT) Conclusion | Source of Evidence (Cache) |
|---|---|---|---|
| 25 datasets (NLU / Commonsense Reasoning / Image Classification / NLG) | Various LoRA and LoRA-MoE baselines | Achieved SOTA and narrowed the performance gap with Full FT | Abstract and Introduction conclusions |
| Fixed architecture and training algorithm | Modifications only to initialization and scaling | Significantly improved convergence and final performance without changing the main architecture | Method and Introduction statements |
| LoRA-MoE low-rank scenarios | Default small scaling | Larger and more reasonable scaling yields better gradient alignment and performance | Sec. 2.2 + Theorem 3.5 |

### Ablation Study

| Settings | Metrics / Phenomena | Value or Trend | Conclusion |
|---|---|---|---|
| Low rank ($r=1$), conventional scaling | Performance | 95.77 | Weak gradients in low-rank settings, leading to significantly lagging performance |
| Low rank ($r=1$), increased scaling (example $s=16$) | Performance | 97.70 | Enhancing gradients via scaling, significantly narrowing the gap |
| Higher rank ($r=64$) | Performance | 98.55 | Naturally closer to the upper bound when the rank is higher |
| SVD segment selection (different datasets) | Optimal segment location | Non-static: Head segments are sometimes superior, while tail or middle segments are better in other cases | Demands "input-adaptive segment selection" instead of static strategies |
| Introducing $W_{res}$ alignment | Initialization stability | Variance decrease (theoretical) | Mitigates the mismatch caused by SVD priors + random routing |

### Key Findings
- Finding 1: Different SVD segments carry task-relevant information, which cannot be simplified as "only the head is effective and the tail is ineffective."
- Finding 2: Scaling is not merely an empirical hyperparameter but can be theoretically justified from the perspective of equivalent gradients.
- Finding 3: The key challenge of LoRA-MoE lies in the coupling of "routing + low rank + priors." GOAT's alignment approach decouples this complexity.

## Highlights & Insights
- Highlight 1: Transforming "which singular value segment to select" from a static hyperparameter to a dynamic decision made by the router. This is the core innovation of integrating SVD priors into MoE.
- Highlight 2: The closed-form solution of $W_{res}$ is simple yet effective, introducing virtually zero extra engineering complexity while theoretically minimizing the initialization error.
- Highlight 3: The scaling factor shifts from an empirical rule to a derivable quantity, explaining why LoRA-MoE relies particularly on larger scaling under low ranks.
- Highlight 4: The method introduces no heavy new modules, aligning with the PEFT community's preference for "low invasiveness, reproducibility, and transferability."

## Limitations & Future Work
- Limitation 1: The cached segments do not contain the complete results table, making precise per-dataset review challenging.
- Limitation 2: Theoretical alignment is established mainly under approximations and expectations; discrete routing in practice may introduce deviations.
- Limitation 3: The method depends on the quality of SVD decomposition. If the pre-trained matrix noise structure is abnormal, the benefits of segment partitioning may diminish.
- Future Direction 1: Upgrade segment partitioning from equal-width slicing to data-driven slicing (e.g., based on spectral energy or Hessian information).
- Future Direction 2: Incorporate "segment coverage regularization" into the router to prevent experts from collapsing into a few segments over time.
- Future Direction 3: Transfer the alignment concept to QLoRA / DoRA / multimodal adaptation layers to verify generalizability.

## Related Work & Insights
- vs. PiSSA: PiSSA excels in rapid convergence along dominant singular value directions, whereas GOAT emphasizes that "different inputs should select different segments."
- vs. MiLoRA/KaSA: The latter retain or replace specific segments, remaining essentially static. GOAT dynamically combines them via MoE routing.
- vs. Standard LoRA-MoE: Standard LoRA-MoE has previously relied mostly on zero-initialization; GOAT addresses the new weight mismatch problem arising under SVD priors and provides an aligned solution.
- Insight 1: Future PEFT could explore the "prior library + routing selection" paradigm instead of being confined to a single low-rank subspace.
- Insight 2: In low-rank scenarios, prioritize analyzing gradient scales before deciding on the coupled configuration of rank / learning rate / scaling.

## Reproducibility & Practical Suggestions
- If you have an existing LoRA-MoE codebase: Verify GOAT gains quickly by replacing only the initialization and scaling rules without modifying the training loop.
- Hyperparameter Recommendations:
	- For small ranks, prioritize trying a larger $s$.
	- Verify whether SVD segments corresponding to different experts are all activated.
	- Monitor routing entropy and expert load to prevent unbalanced expert utilization.
- Diagnostic Sequence Recommendation: Examine initialization error first (whether $W_{res}$ is needed), then evaluate gradient norms (whether scaling is too small).

## Rating
- Novelty: ⭐⭐⭐⭐☆
	- Combines SVD segmented priors with MoE routing and provides alignment theory; the approach is clear and of practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐☆
	- The paper claims coverage of 25 tasks and SOTA results; however, the current cache segment lacks the complete detailed tables.
- Writing Quality: ⭐⭐⭐⭐☆
	- Problem decoupling (initialization vs. optimization) and mapping from theory to methodology are relatively complete.
- Value: ⭐⭐⭐⭐⭐
	- Offers gains without altering the main architecture or training algorithms, making it suitable for direct integration into existing PEFT engineering stacks.

## Personal Notes
- The value of this work goes beyond "recreating yet another LoRA variant"; it systematizes both prior selection and optimization alignment.
- For future experiments in LLM compression, GOAT can serve as a robust baseline with low adaptation costs.
- Focus replication efforts on three main scenarios: low rank, variations in MoE expert counts, and scaling sweeps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](../../CVPR2026/model_compression/tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)
- [\[ICLR 2026\] LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing](../../ICLR2026/model_compression/lora-mixer_coordinate_modular_lora_experts_through_serial_attention_routing.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)

</div>

<!-- RELATED:END -->
