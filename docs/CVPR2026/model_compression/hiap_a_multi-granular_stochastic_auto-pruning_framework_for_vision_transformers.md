---
title: >-
  [Paper Note] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers
description: >-
  [CVPR 2026][Model Compression][Vision Transformer pruning] HiAP formulates ViT pruning as an end-to-end budget-aware learning problem, applying stochastic differentiable gating simultaneously at two granularities—entire heads/blocks (macro) and intra-head value dimensions/FFN neurons (micro)—to automatically discover a compact dense subnetwork satisfying a compute budget within a single training run, eliminating the need for importance ranking, threshold search, and separate fine-tuning.
tags:
  - CVPR 2026
  - Model Compression
  - Vision Transformer pruning
  - multi-granular structured pruning
  - Gumbel-Sigmoid gating
  - budget-aware optimization
  - single-stage compression
date: 2026-05-08
content_hash: 40667d3f651be89c
---

# HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

**Conference**: CVPR 2026
**arXiv**: [2603.12222](https://arxiv.org/abs/2603.12222)
**Code**: None
**Area**: Model Compression
**Keywords**: Vision Transformer pruning, multi-granular structured pruning, Gumbel-Sigmoid gating, budget-aware optimization, single-stage compression

## TL;DR

HiAP formulates ViT pruning as an end-to-end budget-aware learning problem, applying stochastic differentiable gating simultaneously at two granularities—entire heads/blocks (macro) and intra-head value dimensions/FFN neurons (micro)—to automatically discover a compact dense subnetwork satisfying a compute budget within a single training run, eliminating the need for importance ranking, threshold search, and separate fine-tuning.

## Background & Motivation

Vision Transformers continue to achieve strong results across classification, detection, and generation, yet their deployment cost remains high. The challenge is not merely parameter count; both the attention and FFN components impose significant compute and memory bandwidth demands at inference time.

Existing structured pruning methods fall broadly into two categories.

The first operates at fine granularity, pruning intra-head dimensions or FFN neurons. Such methods can reduce theoretical FLOPs, but they typically retain the original network depth and most attention heads, meaning the hardware must still load large matrices layer by layer and construct attention maps—leaving memory access pressure largely unresolved.

The second operates at coarse granularity, removing entire attention heads or entire blocks. This is more hardware-friendly since eliminating a whole block directly bypasses the associated computation and memory traffic, but the approach is too aggressive and tends to discard still-useful representational capacity in large chunks, causing greater accuracy degradation.

The root cause the authors identify is as follows.

On one hand, achieving real speedup requires removing macroscopic structures that genuinely occupy memory access paths, not merely performing fine-grained pruning that appears sparse on paper.

On the other hand, preserving accuracy requires more than coarse block removal; it demands further width adjustment within the retained structures.

A further engineering pain point shared by many differentiable search and pruning methods is that, despite using continuous relaxations during training, they ultimately rely on post-hoc thresholds, importance ranking, heuristic rules, or a second fine-tuning stage—making the overall pipeline far from truly automatic.

The authors therefore re-frame the problem: can a model simultaneously decide *where* and *how much* to prune during training, and directly export a deployable dense subnetwork when training concludes?

The starting point is clear. The authors argue that redundancy in ViTs is not confined to a single granularity but is distributed across block, head, dimension, and neuron levels simultaneously. Incorporating all of these granularities into a unified hierarchical gating system, constrained by an explicit differentiable MACs budget, gives the network the opportunity to discover more heterogeneous, budget-conforming structural combinations on its own.

The core idea in one sentence:

HiAP uses "macro switches to decide whether to retain large structures, and micro switches to decide how wide those retained structures should be," combining Gumbel-Sigmoid gating, budget penalties, and feasibility constraints to turn ViT pruning into an automatic subnetwork discovery process completed within a single training run.

## Method

The key contribution of this work is not a novel importance scorer, but rather the embedding of structural selection itself as trainable variables within the ViT forward pass.

The authors place two levels of gating inside every Transformer block.

The first level is the **macro gate**, which decides whether an attention head or an FFN block should be retained.

The second level is the **micro gate**, which, within the structures that are retained, further prunes certain dimensions of the value projection and certain neurons in the FFN intermediate layer.

This hierarchical design encourages the network to first learn "whether to keep a block at all," and then learn "how wide the kept block needs to be."

From an optimization perspective, this is more natural than operating at a single granularity, because the two granularities correspond to qualitatively different cost types. Macro pruning primarily reduces the overhead associated with empty structures—especially memory access and the fixed cost of entire attention heads—while micro pruning reduces fine-grained computation within the active structures.

The authors further decompose these two cost components and include them separately in the loss, enabling the model to learn a coarse-before-fine pruning rhythm rather than shrinking all structures uniformly.

### Overall Architecture

HiAP can be understood through the following pipeline.

1. Starting from a standard dense ViT, learnable gate parameters are inserted around each attention and FFN sub-layer.
2. Each attention head at layer $l$, head $h$ is assigned a macro gate $g_{l,h}$; each FFN block at layer $l$ is assigned a macro gate $b_l$.
3. Each value-path dimension of a retained head is assigned a micro gate $d_{l,h,j}$; each FFN intermediate neuron is assigned a micro gate $c_{l,k}$.
4. During training, hard binary values are not sampled directly; instead, Gumbel-Sigmoid produces continuous approximations in $(0,1)$, allowing gradients to propagate through these otherwise discrete structural decisions.
5. The loss function jointly comprises a task loss, a macro cost penalty, a micro cost penalty, and structural feasibility constraints.
6. As the temperature anneals, soft gates progressively approach hard decisions, and the network topology transitions from "exploratory" to "essentially fixed."
7. After training, gates are hardened using a $0.5$ threshold; the corresponding matrices are physically pruned, yielding a truly smaller and faster ViT subnetwork.

An important point emphasized by the authors is that the final output is not a pseudo-sparse model with soft masks, but a physically truncated dense subnetwork. This means it does not depend on specialized sparse operators and is more amenable to real speedup on commodity hardware.

### Key Designs

1. **Macro Gates: Determining the Existence of Large Structures**

   - *Function*: Make on/off decisions for attention heads and FFN blocks.
   - *Mechanism*: If the gate $g_{l,h}=0$ for head $h$ at layer $l$, the entire head is bypassed; if $b_l=0$, the entire FFN sub-layer is removed.
   - *Mathematical form*: The attention output is $\text{AttnOut}_{l,h}(X)=g_{l,h}\cdot \text{Attention}(XW^Q_{l,h},XW^K_{l,h},XW^V_{l,h})$; the FFN output is $\text{FFNOut}_{l}(X)=b_l\cdot \text{FFN}(X)$.
   - *Design motivation*: Once a macro structure is switched off, all associated matrix computations and data movement become unnecessary. Macro gating therefore corresponds directly to pruning the actual inference path, not merely reducing FLOPs on paper.

2. **Micro Gates: Further Slimming Retained Structures**

   - *Function*: Perform fine-grained pruning of value dimensions within active heads and hidden neurons within active FFN blocks.
   - *Mechanism*: For surviving heads, the full $D_h$-dimensional value channel is no longer assumed to be necessary; a gate vector $d_{l,h}$ selects the genuinely useful dimensions. For FFN blocks, $c_l$ determines which intermediate neurons continue to participate in the two linear projections.
   - *Mathematical form*: $\text{Head}'_{l,h}(X)=g_{l,h}\left[\text{softmax}\left(\frac{Q_{l,h}K_{l,h}^{\top}}{\sqrt{D_h}}\right)(V_{l,h}\odot d_{l,h})\right]$; $\text{FFN}'_l(X)=b_l\left[(\phi(XW_{1,l})\odot c_l)W_{2,l}\right]$.
   - *Design motivation*: With only macro gates, the model can only make coarse retain/remove decisions, which severely limits the expressible structural space. Micro gates allow retained structures to develop heterogeneous widths per layer, per head, and per neuron, providing substantially more room for accuracy preservation.

3. **Analytic Differentiable Cost Modeling**

   - *Function*: Explicitly decompose prunable MACs into several categories of elementary costs, which are then used to penalize the gates.
   - *Mechanism*: The prunable cost is decomposed into three constants: $C_1$ corresponds to the macro overhead of a single head; $C_2$ corresponds to the micro cost of retaining one attention value dimension; $C_3$ corresponds to the cost of retaining one FFN neuron.
   - *Specific form*: $C_1=2ND(3D_h)+2N^2D_h$, $C_2=2ND+2N^2$, $C_3=4ND$.
   - *Expected total cost*: $\mathbb{E}[C(\mathcal{G})]=\sum_{l,h}\left(C_1\mathbb{E}[g_{l,h}]+C_2\sum_j\mathbb{E}[g_{l,h}d_{l,h,j}]\right)+\sum_{l,k}C_3\mathbb{E}[b_lc_{l,k}]$.
   - *Design motivation*: This decomposition enables the optimizer to clearly distinguish between the waste of keeping an empty head alive and the incremental benefit of retaining a head with fewer dimensions, thereby more naturally learning a strategy of first closing idle macro structures and then compressing their internal widths.

4. **Gumbel-Sigmoid Gating for Training**

   - *Function*: Enable binary structural selection to participate in gradient descent.
   - *Mechanism*: Each gate has a learnable logit $\alpha$; during the forward pass, Logistic noise $\epsilon$ is sampled and the continuous gate value is produced via $\hat{z}=\sigma((\alpha+\epsilon)/\tau)$; gradients are propagated in the backward pass using the Straight-Through Estimator.
   - *Design motivation*: Direct hard sampling leads to highly unstable training. The continuous relaxation allows the originally discrete, non-differentiable problem of structure removal to be handled by standard optimizers.

5. **Structural Feasibility Constraints to Prevent Layer Collapse**

   - *Function*: Prevent the model from prematurely emptying an entire layer or structural category in pursuit of budget reduction.
   - *Mechanism*: For example, the head-count constraint is written as $\mathcal{L}_{f,\text{head}}=\sum_l \text{ReLU}(k_{\min}-\sum_h g_{l,h})^2$, imposing a strong penalty whenever the number of active heads in a layer falls below a minimum; analogous constraints enforce minimum dimension retention within active heads and minimum neuron retention within FFN blocks.
   - *Design motivation*: A common failure mode in differentiable search is that optimization prematurely empties an entire layer—improving the budget term immediately while the network has no opportunity to reorganize its representations. These constraints essentially serve as structural safety guardrails for the search process.

6. **Single-Stage Search and Export**

   - *Function*: Integrate search, adaptation, and export into a single continuous training pipeline.
   - *Mechanism*: High temperature $\tau$ in early training causes gates to perturb structures continuously, akin to soft dropout; low temperature in late training bimodalizes the gates, progressively approaching 0/1; upon completion, gates are hardened and weight matrices are physically truncated.
   - *Design motivation*: This avoids the two-stage cost of "find mask first, then fine-tune separately," and eliminates additional engineering steps such as threshold tuning and structural backfilling.

### Loss & Training

The overall objective is:

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda_{\text{macro}}\mathcal{L}_{\text{macro}}+\lambda_{\text{micro}}\mathcal{L}_{\text{micro}}+\mathcal{L}_{\text{feasibility}}.
$$

Each of the four components serves a distinct role.

- $\mathcal{L}_{\text{task}}$ is the primary task loss, combining cross-entropy with knowledge distillation. The teacher is a pretrained dense DeiT-Small. The distillation is configured with $\alpha_{\text{KD}}=0.7$ and temperature $T=4.0$. Intuitively, this term instructs the model being pruned to maintain representational alignment with the dense teacher even as its structure continuously changes.

- $\mathcal{L}_{\text{macro}}$ penalizes the overhead attributable to macro structures, i.e., the aggregate cost of attention heads.
- $\mathcal{L}_{\text{micro}}$ penalizes the internal costs of retained value dimensions and FFN neurons.
- Decoupling these two terms rather than merging them into a single unified budget error is a critical design choice in the paper. It allows the authors to explicitly control whether the model preferentially prunes large blocks or shrinks widths first.

- $\mathcal{L}_{\text{feasibility}}$ comprises multiple constraints on head count, dimension ratios, and FFN width ratios. Its purpose is not to improve accuracy but to ensure the search trajectory does not lead to structural collapse.

Training details:

- On ImageNet, DeiT-Small serves as the backbone; training runs for 200 epochs using AdamW with learning rate $5\times10^{-5}$ and global batch size 256.
- On CIFAR-10, a 6-layer ViT-Tiny is used for controlled experiments, also trained for 200 epochs.
- The Gumbel-Sigmoid temperature is exponentially annealed from $2.0$ to $0.5$.
- After training, gates are hardened using the rule $\hat{z}>0.5$.
- The deactivated heads, FFN blocks, and their corresponding matrix rows/columns are physically removed to obtain the final subnetwork.

This training logic admits a concise interpretation. During the high-temperature phase, gates are relatively soft, effectively applying random structural perturbations that force the parameters to learn representations robust to varied substructure combinations. During the low-temperature phase, gates gradually harden and the network converges to a fixed topology. Because weights and structure co-adapt throughout, hardening does not produce the abrupt performance discontinuity typically observed in post-hoc pruning.

## Key Experimental Results

### Main Results

On ImageNet-1K, HiAP is compared against multiple ViT structured pruning methods using DeiT-Small as the backbone. The dense baseline achieves 22.1M parameters, 4.6G FLOPs, and 79.85% Top-1 accuracy.

| Method | Params (M) | FLOPs (G) | Top-1 Acc (%) | Change vs. Dense |
|--------|------------|-----------|---------------|------------------|
| Dense Baseline | 22.1 | 4.6 | 79.85 | — |
| WDPruning | 15.0 | 3.1 | 78.55 | −1.30 |
| WDPruning | 13.3 | 2.6 | 78.38 | −1.47 |
| S2ViT | 15.3 | 3.1 | 79.22 | −0.63 |
| S2ViT | 13.5 | 2.8 | 78.44 | −1.41 |
| ViT-Slim | 15.6 | 3.1 | 79.90 | +0.05 |
| ViT-Slim | 13.5 | 2.8 | 79.50 | −0.35 |
| GOHSP | 14.4 | 3.0 | 79.98 | +0.13 |
| GOHSP | 11.1 | 2.8 | 79.86 | +0.01 |
| **HiAP (Ours)** | **15.0** | **3.1** | **79.10** | **−0.75** |
| **HiAP (Ours)** | **12.3** | **2.5** | **77.95** | **−1.90** |

At the 3.1G operating point, HiAP achieves approximately 33% FLOPs reduction, though its absolute accuracy does not surpass GOHSP or ViT-Slim. The primary contribution is not single-point accuracy leadership but rather a substantially more streamlined train–search–export pipeline. In terms of engineering complexity, HiAP replaces graph optimization, importance ranking, post-processing thresholds, and multi-stage fine-tuning with a unified mechanism.

### Ablation Study

Controlled experiments on CIFAR-10 using a 6-layer ViT-Tiny compare HiAP against simple heuristic baselines.

| Method | MACs (M) | Compression | Accuracy (%) | Notes |
|--------|----------|-------------|--------------|-------|
| Dense Baseline | 174.0 | 0.0% | 90.50 | Original model |
| Uniform-Ratio | 116.6 | 33.0% | 86.63 | Uniform reduction per layer |
| $\ell_1$-Structured (FFN) | 116.5 | 33.0% | 87.15 | Heuristic importance ranking |
| **HiAP (Moderate)** | **116.3** | **33.1%** | **87.56** | Automatic multi-granular budget allocation |
| $\ell_1$-Structured (FFN) | 87.3 | 49.8% | 86.80 | More aggressive compression |
| **HiAP (Aggressive)** | **87.1** | **49.9%** | **87.25** | Maintains advantage under high compression |

Throughput and latency results are also reported. For the 33.1%-pruned model, single-GPU inference at batch size 1 over 50 runs shows latency reduced from 5.57 ms to 3.86 ms, corresponding to approximately $1.44\times$ real-world speedup—demonstrating that gains are not confined to theoretical MACs.

### Key Findings

- The search trajectory does not prune all structures simultaneously and uniformly. Instead, macro pruning precedes micro adjustment. Within the first 10 training epochs, the average number of active attention heads drops rapidly from 6 to approximately 2–4, indicating that under budget pressure the model prioritizes closing expensive but underutilized macro structures rather than uniformly narrowing every head.

- The FFN block in the final layer is consistently deactivated. This suggests that, at least under the given training settings and budget, certain deep FFN sub-layers in DeiT-Small exhibit persistent redundancy.

- Retained FFN widths exhibit a clearly heterogeneous distribution. Earlier layers tend to retain close to full capacity (approximately 1,400 out of 1,536 neurons active), while deeper layers are compressed to around 1,200 neurons.

- Intra-head dimensions of retained attention heads also vary. Many heads are reduced from 64 to 32 dimensions or fewer, confirming that micro gates are performing genuine fine-grained adjustment rather than serving as decorative components.

- Decoupling the cost terms is effective. Appendix experiments comparing different $\lambda_{\text{macro}}:\lambda_{\text{micro}}$ ratios show that a balanced 2:1 ratio performs best. Excessively strong macro penalties cause too many heads to be removed, while excessively strong micro penalties leave many blocks formally active but internally very narrow, resulting in a suboptimal budget allocation.

## Highlights & Insights

- The first highlight is the authors' explicit distinction between memory access pressure in real deployment and theoretical FLOPs reduction. This is more pragmatic than purely minimizing FLOPs, since the bottleneck in many ViT acceleration scenarios lies not in multiply-accumulate count but in the large-matrix memory traffic associated with attention.

- The second highlight is the hierarchical gating design, which closely mirrors the problem structure. Macro gates address "should this computational path be retained at all," while micro gates address "how much capacity should the retained path receive"—a clear and principled division of labor.

- The third highlight is the analytic cost modeling. Rather than simply regularizing FLOPs, the paper decomposes the marginal cost of different structural units into three categories $C_1$, $C_2$, $C_3$, enabling the optimizer to learn a more rational sparse allocation strategy.

- The fourth highlight is the complete single-stage training loop. Many pruning methods defer the most challenging steps to post-processing and retraining; HiAP integrates gate learning, structural contraction, and final export into a unified pipeline, which is of substantial engineering value.

- The fifth highlight is the physically exported subnetwork. The final output consists of ordinary dense matrices with reduced dimensions, not a pseudo-sparse structure with runtime masks, which is far more amenable to downstream deployment, compilation, and inference framework integration.

The most instructive takeaway from this paper is the reminder that model compression should not fix the pruning granularity as a prior assumption, but should treat granularity itself as a learnable design degree of freedom. More broadly, the genuinely valuable insight is not any particular importance score formula, but the principle of *decomposing penalty terms by cost type*—modeling structures that affect memory access separately from those that affect pure computation.

## Limitations & Future Work

- The most direct limitation is that the ImageNet results are not fully competitive. At the 3.1G operating point, HiAP's 79.10% lags noticeably behind GOHSP's 79.98% and ViT-Slim's 79.90%, positioning it as a pipeline-simpler competitor rather than a new absolute state of the art.

- The optimization target remains expected MACs rather than latency or energy calibrated on real hardware. The authors acknowledge that actual speedup is influenced by hardware, kernel implementations, and compiler behavior, meaning "budget-aware" is not yet fully equivalent to "platform-aware."

- Experimental coverage is relatively narrow. The paper primarily validates on DeiT-Small and a CIFAR-10 ViT-Tiny, with no evaluation on larger models or broader visual tasks—detection, segmentation, and multimodal settings are notably absent.

- Comparisons do not combine HiAP with orthogonal techniques such as token pruning, quantization, enhanced distillation, or compiler optimization. For HiAP to function as a complete deployment solution rather than a standalone pruning strategy, future work should demonstrate its composability with these methods.

- The theoretical analysis primarily argues for the reasonableness of the design rather than providing strong performance guarantees. For instance, the soft-to-hard budget alignment argument resembles an intuitive convergence explanation more than a rigorous achievable error bound.

Promising directions for improvement include:

- Upgrading the cost model from MACs to a platform-calibrated latency surrogate or energy surrogate.
- Incorporating hardware-aware rounding at the export stage to align the final structure with the preferences of efficient hardware kernels.
- Combining HiAP with token pruning to achieve joint compression of structural width and sequence length.
- Co-optimizing with quantization, so that macro path removal, micro width reduction, and low-bit representation occur simultaneously.
- Extending to DiT, VLMs, or video Transformers to assess whether hierarchical gating confers greater advantages under longer sequences and heavier attention.

## Related Work & Insights

- **vs. ViT-Slim**: ViT-Slim focuses more on searching internal dimensions and FFN channels in a continuous space and achieves strong fine-grained compression, but it primarily targets the width dimension. HiAP explicitly includes entire heads and entire blocks in the search space, placing greater emphasis on controlling memory access paths.

- **vs. GOHSP**: GOHSP jointly considers heads, intra-head dimensions, and FFN neurons, achieving stronger results, but its pipeline relies more heavily on graph-structural analysis and optimization steps. HiAP's advantage lies in unifying these structural selections into end-to-end gating learning, resulting in a shorter engineering pipeline.

- **vs. WDPruning / UPDP**: These methods emphasize coarser depth/head-level structural pruning that directly reduces large-block overhead. HiAP inherits their attention to macro structures while additionally retaining the ability to fine-tune internal widths.

- **vs. ProxylessNAS / AutoSlim**: Conceptually, HiAP resembles the application of budget-aware NAS principles to ViT structured pruning. The key difference is that the search space here is not a full architectural design but the set of removable structural units within a known architecture.

Two broader implications stand out. First, many Transformer compression works assume a fixed granularity prior; this paper demonstrates that granularity itself can be learned, and this learning is most effective when budget terms are decoupled by cost type. Second, for future compression of large vision or multimodal models, the most transferable insight is not any specific formula but the principle of *separately modeling structures that affect memory access and structures that affect pure computation*.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆
  Unifying macro and micro pruning into a single hierarchical, differentiable, directly exportable framework is a clear and creative combination, though the core tools—Gumbel-Sigmoid and budget regularization—are relatively mature.

- **Experimental Thoroughness**: ⭐⭐⭐☆☆
  The main validation on ImageNet and CIFAR-10 is complete and includes latency measurements and appendix sensitivity analyses, but the task coverage is narrow and ImageNet accuracy does not surpass the strongest baselines.

- **Writing Quality**: ⭐⭐⭐⭐☆
  Problem formulation, mathematical decomposition, training procedure, and theoretical motivation are all presented clearly; a reader can reconstruct the design logic with reasonable ease.

- **Value**: ⭐⭐⭐⭐☆
  HiAP offers a cleaner engineering path than multi-stage heuristic pruning. Although its accuracy ceiling has not yet been fully established, it provides a valuable reference for the direction of automatically discovering deployable subnetworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](batch_loss_score_for_dynamic_data_pruning.md)
- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)

</div>

<!-- RELATED:END -->
