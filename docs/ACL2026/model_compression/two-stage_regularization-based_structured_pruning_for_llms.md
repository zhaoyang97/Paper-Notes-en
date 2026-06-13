---
title: >-
  [Paper Note] Two-Stage Regularization-Based Structured Pruning for LLMs
description: >-
  [ACL2026][Model Compression][Structured Pruning] TRSP utilizes first-stage regularization to learn the importance of each Transformer layer and second-stage regularization to minimize the difference between the input and…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Structured Pruning"
  - "Layer Pruning"
  - "Two-Stage Regularization"
  - "LLM Acceleration"
  - "Retraining-Free Compression"
date: 2026-05-08
content_hash: e19e9faacc527087
---

# Two-Stage Regularization-Based Structured Pruning for LLMs

**Conference**: ACL2026  
**arXiv**: [2505.18232](https://arxiv.org/abs/2505.18232)  
**Code**: https://github.com/fmk345/TRSP  
**Area**: Model Compression / Structured Pruning / LLM Efficiency  
**Keywords**: Structured Pruning, Layer Pruning, Two-Stage Regularization, LLM Acceleration, Retraining-Free Compression

## TL;DR
TRSP utilizes first-stage regularization to learn the importance of each Transformer layer and second-stage regularization to minimize the difference between the input and output of the layers to be deleted. This transfers knowledge to the retained layers, achieving LLM layer-level structured pruning and practical inference acceleration without requiring retraining.

## Background & Motivation
**Background**: The bottlenecks in LLM deployment primarily stem from parameter scale, memory footprint, and inference latency. Structured pruning is more suitable for real-world acceleration than unstructured sparsity because it directly removes layers, heads, or channels, making it easier to achieve throughput gains on hardware.

**Limitations of Prior Work**: Most existing layer-wise pruning methods first identify "unimportant" layers using certain importance metrics and then delete them directly. The issue is that the deleted layers may still contain local knowledge, and direct removal leads to significant performance degradation. To compensate for this loss, many methods require additional LoRA or full-scale retraining, which increases compression costs.

**Key Challenge**: Layer pruning aims for "neat deletion" to gain speed, but knowledge distribution in LLMs is not neat. A layer with seemingly low importance does not mean its information can be discarded without loss. If the model is not adapted to the "disappearance of these layers" beforehand, the deletion operation acts like a sudden break in a computation chain.

**Goal**: To reduce knowledge loss after pruning while maintaining the hardware-friendly nature of layer-level structured pruning, all while avoiding expensive retraining.

**Key Insight**: The authors decouple "selecting which layers to delete" and "making these layers deletable" into two regularization stages. The first stage uses learnable layer weights to identify layers for removal, and the second stage regularizes these layers to approximate an identity mapping, reducing their unique contribution to the final output.

**Core Idea**: Instead of directly deleting low-importance layers, the model is first regularized so that these layers carry less knowledge before deletion is executed.

## Method
TRSP is a layer-wise structured pruning method. It only requires a small amount of calibration data, such as 128 sequences of length 2048 randomly sampled from the WikiText-2 training set. The process includes four steps: preparing data, learning layer weights, performing second-stage regularization on target layers, and deleting those layers. The paper emphasizes that TRSP is retraining-free: while baseline methods use 1,000 additional WikiText-2 samples for LoRA retraining after pruning, TRSP does not require this step.

### Overall Architecture
Given the original model $\mathbf{W}$, pruning ratio $p$, and number of layers $l$, TRSP first assigns a learnable scalar weight $S[i]$ to each Transformer layer, which scales the layer output during forward propagation. The first stage optimizes the language modeling loss plus an $\ell_1$ layer weight regularization to reduce the weights of less important layers. In each round, the layer with the smallest weight is added to the pruning set $P$ and masked; this repeats until the target pruning count is reached. After fixing the pruning set, the second stage regularizes the difference between the input and output $\mathbf{X}_{out}^i-\mathbf{X}_{in}^i$ of these layers to make them approximate an identity transformation. Finally, the layers in set $P$ are deleted.

### Key Designs
1.  **Iterative Layer Weight Learning**:
    - **Function**: Determines which Transformer layers are most suitable for deletion.
    - **Mechanism**: Assigns a learnable weight $S[i]$ to each layer and optimizes $\mathcal{L}_{learn}=\mathcal{L}(\mathbf{W},\mathbf{X})+\lambda_1\sum_i\|S[i]\|_1$. Only one layer with the minimum weight is selected per round to join the pruning set, followed by re-learning the weights of remaining layers.
    - **Design Motivation**: Selecting all low-weight layers at once often results in contiguous layer segments, which can disrupt deep network computation paths. An iterative greedy strategy re-evaluates the importance of remaining layers after each selection.

2.  **Second-Stage Input-Output Difference Regularization**:
    - **Function**: Reduces the unique contribution of the target layers to the model output before actual deletion.
    - **Mechanism**: Adds a regularization term $\mathcal{L}_{sum}=\mathcal{L}(\mathbf{W},\mathbf{X})+\lambda_2\sum_{i\in P}\|\mathbf{X}_{out}^i-\mathbf{X}_{in}^i\|$ (using $\ell_1$ or $\ell_2$ norms) for layers in the pruning set $P$. This forces the target layers to behave like identity mappings, pushing knowledge into the unregularized retained layers.
    - **Design Motivation**: The loss from direct layer removal occurs because those layers are still modifying representations. By making them "do less" before removal, the perturbation to the model is minimized when they are finally pruned.

3.  **Retraining-Free Structural Deletion**:
    - **Function**: Translates compression results into real throughput and latency gains.
    - **Mechanism**: TRSP deletes complete Transformer layers rather than zeroing out parameters. The resulting shallower model structure speeds up both prompt processing and token generation.
    - **Design Motivation**: Many sparsity methods compress parameters in terms of metrics but fail to provide speedups on standard hardware. While layer pruning is coarse-grained, its deployment benefits are more direct.

### Loss & Training
The first stage uses language modeling loss plus $\ell_1$ regularization on layer weights. Since $\ell_1$ is non-differentiable, the authors use an equivalent constraint form to transform $\|x\|_1$ into a backpropagation-compatible optimization problem with auxiliary variables. The second stage uses language modeling loss plus input-output difference regularization for target layers. $\lambda_1$ and $\lambda_2$ are determined via grid search; for LLaMA2-7B, the optimal combination is $\lambda_1=5\times10^{-3}$ and $\lambda_2=10^{-3}$. The experiments use a default pruning ratio of 25%, with $\ell_2$ second-stage regularization as the primary configuration.

## Key Experimental Results

### Main Results
The main experiments compare TRSP against SLEB, ShortGPT, LaCo, and Shortened LLaMA on models such as Phi-2, OPT, LLaMA2, and LLaMA3. Metrics include WikiText-2 perplexity and the average accuracy across five zero-shot benchmarks: PIQA, WinoGrande, HellaSwag, ARC-e, and ARC-c.

| Model | Method | Pruning Rate | PPL↓ | Avg_Acc↑ | Main Advantage over Strongest Baseline |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Phi-2 | ShortGPT | 25% | 7.15 | 54.49 | One of the strongest baselines |
| Phi-2 | TRSP-$\ell_2$ | 25% | 6.53 | 56.56 | Lower PPL, Avg_Acc +2.07 |
| OPT-2.7B | ShortGPT | 25% | 14.96 | 49.56 | One of the strongest baselines |
| OPT-2.7B | TRSP-$\ell_1$ | 25% | 13.12 | 51.27 | Best PPL and Avg_Acc |
| LLaMA2-7B | ShortGPT | 25% | 8.89 | 57.10 | Strongest baseline |
| LLaMA2-7B | TRSP-$\ell_2$ | 25% | 7.08 | 60.57 | PPL ~20% lower, Avg_Acc +3.47 |
| LLaMA3-8B | ShortGPT | 25% | 9.26 | 66.17 | Strongest baseline |
| LLaMA3-8B | TRSP-$\ell_2$ | 25% | 7.84 | 68.44 | Lower PPL, higher Avg_Acc |
| LLaMA2-13B | ShortGPT | 25% | 6.79 | 62.96 | Strongest baseline |
| LLaMA2-13B | TRSP-$\ell_1$ | 25% | 5.89 | 65.18 | Best PPL and Avg_Acc |

### Ablation Study
The analysis of TRSP primarily addresses three questions: whether it truly accelerates, the necessity of the two stages, and the superiority of iterative selection over one-shot.

| Exp | Config | PPL↓ | Avg_Acc↑ / Speed | Conclusion |
| :--- | :--- | ---: | ---: | :--- |
| Speed | OPT-13B Dense | 10.12 | 1029 tokens/s, 386.5 ms | Original model |
| Speed | OPT-13B TRSP 25% | 10.45 | 1348 tokens/s, 286.3 ms | Throughput 1.31×, Latency 1.35× |
| Speed | LLaMA2-13B TRSP 25% | 5.82 | 1386 tokens/s, 298.4 ms | Throughput 1.30×, Latency 1.33× |
| Ablation | LLaMA2-7B TRSP | 7.08 | 60.57 | Full method |
| Ablation | LLaMA2-7B w/o W | 9.26 | 56.19 | One-shot layer selection, PPL +2.18 |
| Ablation | LLaMA2-7B w/o R | 10.15 | 54.36 | Stage 2 reg removed, largest drop |
| Ablation | LLaMA2-13B TRSP | 5.82 | 65.11 | Full method |
| Ablation | LLaMA2-13B w/o R | 9.47 | 56.25 | Avg_Acc -8.86 after reg removal |

### Key Findings
- The second-stage regularization contributes the most. Removing it from LLaMA2-13B causes PPL to rise from 5.82 to 9.47 and Avg_Acc to drop from 65.11 to 56.25, indicating that "making layers deletable" is more critical than simply finding low-importance layers.
- Iterative layer selection is more stable than one-shot. One-shot selection tends to pick contiguous layers, leading to structural fragmentation; the iterative method re-evaluates importance after each deletion, resulting in a more dispersed pruning set.
- The difference between $\ell_1$ and $\ell_2$ in the second stage is minimal, suggesting that the key mechanism is the input-output difference regularization itself rather than the specific norm.
- The speed gains are real end-to-end benefits: 25% layer pruning on 13B models results in approximately 1.30× throughput improvement and 1.33-1.35× latency speedup.

## Highlights & Insights
- The core insight of this paper is straightforward: instead of asking "which layer is unimportant," the model is trained to make certain layers unimportant. This shifts pruning from passive measurement to active knowledge redistribution.
- The two-stage design effectively decouples selection and adaptation. The first stage solves "who to delete," and the second stage solves "how to minimize damage before deletion," which is more interpretable than merely modifying importance metrics.
- The method is deployment-friendly. Removing full layers means the model depth is reduced, translating gains into tokens/s and latency rather than just parameter sparsity ratios.
- The "low-cost" claim is well-supported by experiments. While baselines require LoRA retraining with 1,000 samples after pruning, TRSP achieves better results using only 128 calibration samples for regularization and pruning.

## Limitations & Future Work
- Currently validated primarily on autoregressive LLMs; the authors acknowledge it has not yet been proven to generalize to CNNs, encoder-only models, or multimodal models.
- Layer-wise pruning is coarse-grained. While acceleration is direct, the tuning space is limited; scenarios requiring fine-grained compression ratios might need integration with head or MLP channel pruning.
- The second-stage regularization requires accessing intermediate layer inputs/outputs and performing optimization, which, while cheaper than full retraining, still involves engineering costs for ultra-large models.
- Experiments focus on WikiText-2 perplexity and common zero-shot benchmarks, lacking systemic evaluation for long context, code, math, and instruction-following tasks.

## Related Work & Insights
- **vs SLEB**: SLEB also performs layer-wise pruning but relies more on direct deletion based on layer importance; TRSP adds a knowledge transfer step before deletion, leading to better performance retention.
- **vs ShortGPT**: ShortGPT uses block influence to determine redundant layers. TRSP uses learnable layer weights with regularization and iterative selection to avoid the contiguous deletion issues of one-shot methods.
- **vs LaCo**: Methods like LaCo depend on post-pruning recovery or calibration. TRSP emphasizes a retraining-free approach, suitable for scenarios where rapid deployment of shallower models is required.
- **Transferable Insights**: The concept of "regularizing target modules to approximate identity mappings before deletion" could be applied to MoE expert pruning, Vision Transformer block pruning, or adapter merging.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The two-stage regularization perspective is clear and effective, though it remains an improvement within the layer pruning framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, baselines, acceleration, pruning rates, data dependency, ablations, and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐☆ The method workflow is easy to follow, though minor inconsistencies in table numbering require careful reading of the ablation charts.
- Value: ⭐⭐⭐⭐☆ Highly relevant for real-world LLM deployment compression, especially for scenarios prioritizing structural acceleration over simple sparsity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[CVPR 2026\] F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](../../CVPR2026/model_compression/textf2texthdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_m.md)
- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](../../ICLR2026/model_compression/mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ACL 2026\] From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization](from_signal_degradation_to_computation_collapse_uncovering_the_two_failure_modes.md)

</div>

<!-- RELATED:END -->
