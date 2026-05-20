---
title: >-
  [Paper Note] Large Vision-Language Models Get Lost in Attention
description: >-
  [ICML 2026][Multimodal VLM][LVLM interpretability] This paper quantitatively diagnoses the residual stream of LVLMs using a geometric information-theoretic framework based on "information complexity (eRank) + subspace su…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "LVLM interpretability"
  - "attention redundancy"
  - "information theory"
  - "subspace analysis"
  - "attention replacement"
date: 2026-05-08
content_hash: bcf52e2c9f2280bf
---

# Large Vision-Language Models Get Lost in Attention

**Conference**: ICML 2026  
**arXiv**: [2605.05668](https://arxiv.org/abs/2605.05668)  
**Code**: Public  
**Area**: Multimodal VLM / Interpretability  
**Keywords**: LVLM interpretability, attention redundancy, information theory, subspace analysis, attention replacement

## TL;DR
This paper quantitatively diagnoses the residual stream of LVLMs using a geometric information-theoretic framework based on "information complexity (eRank) + subspace support." It finds that attention almost exclusively performs reconfiguration within the subspace, while the FFN injects new semantic dimensions. Even more strikingly, replacing learned attention weights with Gaussian noise leads to equal or improved performance on most vision tasks, revealing severe misalignment and redundancy in visual attention of contemporary LVLMs.

## Background & Motivation
**Background**: The decoder of LVLMs remains a Transformer with residual connections, where each module outputs an additive update $\Delta\mathbf{X}$ written back to the shared residual stream. The mainstream assumption is that attention is responsible for in-context reasoning (induction head, copy mechanism), while the FFN acts as key-value memory for storing facts. Recent empirical phenomena in LVLMs, such as visual attention sink and visual attention drift, suggest that models may not truly utilize visual evidence.

**Limitations of Prior Work**: Existing analyses mostly remain at the **statistical level**—visualizing attention maps, performing attention rollout, counting sparse heads, or conducting causal interventions. However, these tools: (i) lack a unified theoretical foundation, making it difficult to compare conclusions across modules and metrics; (ii) attention weights themselves have been shown by Jain & Wallace et al. to be unreliable attribution signals; (iii) lack a unified metric to quantify "what the residual update actually changes in the representation."

**Key Challenge**: To answer "what do attention and FFN actually do," a unified and comparable metric is needed. The LLM representation analysis community has used entropy, effective rank, and other geometric tools to characterize cross-layer quality, but module-level interpretation in LVLMs remains unexplored.

**Goal**: (i) Define "what information a representation matrix $\mathbf{X}$ contains"; (ii) Quantify "what an additive update $\Delta\mathbf{X}$ injects into $\mathbf{X}$"; (iii) Use these two metrics to diagnose the functional division of LVLM modules, especially to reveal whether visual attention is truly meaningful.

**Key Insight**: By viewing the representation matrix as lying on a fixed-rank matrix manifold and applying SVD, two geometric objects naturally emerge: "singular spectrum (complexity)" and "column/row subspace (semantic support)." The concept of innovation from least squares is then used to quantify the "energy outside the existing subspace" in the update. This turns the vague problem of "information change" into a computable subspace projection residual.

**Core Idea**: Decompose Transformer residual updates into two orthogonal dimensions: "innovation (RID)" vs "reconfiguration (MixIG)," and use this lens to re-examine LVLMs.

## Method

### Overall Architecture
For the residual stream $\mathbf{X}_{\text{new}} = \mathbf{X}_{\text{old}} + \Delta\mathbf{X}$, the authors define the representation information as $\mathcal{I}(\mathbf{X}) = (\mathcal{S}_\mathbf{X}, \mathcal{D}_\mathbf{X})$: $\mathcal{S}_\mathbf{X} = \mathrm{eRank}(\mathbf{X})$ describes spectral complexity, and $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$ describes column/row subspace support. **RID** measures "external innovation" (spectral change + subspace novelty), while **MixIG** measures "internal reconfiguration" (token-level mixing entropy change). Applying these metrics to Attention and FFN updates yields quantitative evidence for module-level functional division.

### Key Designs

1. **SVD Geometric Characterization of Representation Information (RQ1)**:

    - **Function**: Formalizes "what information is in $\mathbf{X}$."
    - **Mechanism**: On the matrix manifold $\mathcal{M}_r = \{\mathbf{X} : \mathrm{rank}(\mathbf{X}) = r\}$, SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ yields three geometric objects: left singular subspace $\mathcal{C}(\mathbf{X}) \in \mathrm{Gr}(r, S)$ (token association), right singular subspace $\mathcal{R}(\mathbf{X}) \in \mathrm{Gr}(r, H)$ (semantic direction), and singular spectrum $\mathbf{\Sigma}$ (energy distribution). Complexity is reduced to effective rank $\mathcal{S}_\mathbf{X} = \exp(-\sum_i p_i \log p_i)$ ($p_i = \sigma_i / \sum \sigma$), and support to projection operators.
    - **Design Motivation**: Directly examining $\|\mathbf{X}\|_F$ misses structure; SVD decouples "number of effective dimensions" from "directions covered," distinguishing "energy change" from "direction change."

2. **RID = Spectral Change + Subspace Innovation (RQ2)**:

    - **Function**: Quantifies whether $\Delta\mathbf{X}$ injects **external new information** (neither in the original spectrum nor in the original subspace).
    - **Mechanism**: Spectral change $\Delta\mathcal{S} = |\mathrm{eRank}(\mathbf{X}') - \mathrm{eRank}(\mathbf{X})| / \min(S, H)$; subspace innovation $\Delta\mathcal{D} = \frac{\|(\mathbf{I} - \mathbf{P}_{\mathcal{C}(\mathbf{X})})\mathbf{X}'\|_F + \|\mathbf{X}'(\mathbf{I} - \mathbf{P}_{\mathcal{R}(\mathbf{X})})\|_F}{2\|\mathbf{X}'\|_F}$, inspired by least squares innovation. Final $\mathrm{RID} = \Delta\mathcal{S} + \Delta\mathcal{D} \in [0, 2]$. Since position encodings like RoPE inherently make RID nonzero, a tolerance $\epsilon_{\text{RoPE}} = \mathrm{RID}(\mathbf{X}^{(\text{RoPE})} \mid \mathbf{X}^{(\text{no-RoPE})})$ is introduced as a baseline.
    - **Design Motivation**: Looking only at spectral change misses "direction change without dimension change"; looking only at subspace misses "dimension collapse." Both are needed to fully characterize external information injection.

3. **MixIG = Token Mixing Entropy Change + Attention Replacement Experiment (RQ3)**:

    - **Function**: Quantifies how much $\Delta\mathbf{X}$ reconfigures tokens **within the existing subspace**; controlled replacement experiments map functional division to performance impact.
    - **Mechanism**: Each token row is normalized to construct a token-to-token mixing distribution $P_{t,j} \propto \frac{\tilde{\mathbf{x}}_t^\top \tilde{\mathbf{x}}_j + 1}{2}$; average Shannon entropy yields TME; $\mathrm{MixIG} = \mathrm{TME}(\mathbf{X}') - \mathrm{TME}(\mathbf{X})$, where a positive value means the update increases token mixing. Diagnostic experiments on 15 open-source LVLMs: attention updates are replaced with two types of noise—Noise $\mathbf{\Delta}$ (directly replacing $\Delta\mathbf{X}_{\text{attn}}$ with Gaussian noise) and Noise $\mathbf{QKV}$ (replacing Q/K/V matrices with Gaussian weights)—to observe performance and geometric signal changes.
    - **Design Motivation**: MixIG captures "within-subspace rearrangement" invisible to RID; noise replacement experiments link theoretical metrics to downstream performance—if attention is truly important, noise replacement should break the model.

### Loss & Training
This is a diagnostic framework; no new models are trained. All metrics are geometric quantities computed in the forward pass. Experiments are conducted on 15 variants from the Qwen2.5-VL, LLaVA-1.5, and LLaVA-NeXT families, across 7 benchmarks: POPE, 3DSRBench, RealWorldQA, MMMU, VMCBench, MathVista, and HallusionBench, with 1000 samples per benchmark.

## Key Experimental Results

### Main Results
Module-level RID/MixIG aggregated across models (Table 1 in the paper):

| Module | RID | MixIG | Functional Feature |
|--------|-----|-------|-------------------|
| Noise $\mathbf{\Delta}$ | 0.61 | -0.80 | High RID + Negative MixIG (off-manifold perturbation) |
| Noise $\mathbf{QKV}$ | 0.44 | -0.50 | High RID + Negative MixIG |
| **Attention** | **0.06** | **0.61** | **Low RID + High MixIG** (subspace preservation + reconfiguration) |
| **FFN** | **0.21** | **0.02** | **High RID + Low MixIG** (subspace expansion + innovation) |

The separation across 15 models is highly stable: attention's RID is almost equal to $\epsilon_{\text{RoPE}} = 0.062$, indicating it **introduces almost no new support directions** and is purely mixing; FFN's RID is significantly higher than this baseline, making it the true source of innovation.

### Ablation Study
SAP (Stochastic Attention Probing) noise replacement experiment (excerpt from Table 2, Qwen-2.5-VL-3B):

| Configuration | POPE | RWQA | 3dSRBench |
|---------------|------|------|-----------|
| Vanilla | 86.13 | 59.35 | 53.46 |
| + Vis. Attn. (noise replacement) | 87.58 | 61.38 | — |

On most vision tasks, **replacing learned visual attention weights with Gaussian noise does not degrade, and even improves, performance**—the most dramatic finding of this paper.

### Key Findings
- The functions of Attention and FFN are geometrically orthogonal: Attention = subspace-preserving operator (reconfiguration); FFN = subspace-expanding operator (innovation). The previous hypothesis that "attention does in-context, FFN does memory" is substantiated by geometric evidence.
- Visual attention in LVLMs is highly redundant, with attention scores carrying little effective information; this corroborates the attention sink and attention drift phenomena.
- Since attention complexity is the main $O(S^2)$ bottleneck yet is redundant, this work provides strong theoretical and empirical support for approximate attention (sparse/predefined/low-rank) on visual tokens.

## Highlights & Insights
- "RID + MixIG" is an elegant pair of dual metrics, decomposing residual updates into "adding new bases" and "mixing within old bases." This language is more general than tool-level methods like attention rollout or tuned lens, and can be applied to any additive update module.
- The noise replacement experiment is more radical than any ablation—replacing "learned weights with pure randomness" yet not affecting vision tasks implies that LVLM training objectives may provide too weak a learning signal for visual attention. This directly motivates directions like "visual token pruning" and "attention-free visual fusion."
- The introduction of the RoPE baseline is pragmatic—subtracting the "spurious RID" introduced by position encoding avoids misjudgment, reflecting the authors' engineering awareness in making the metric a portable tool.

## Limitations & Future Work
- The framework only considers single-step additive updates; cumulative effects across layers (which layers together constitute true innovation) require further analysis.
- Noise replacement experiments focus on the visual side; for pure text tasks (MathVista), attention remains crucial, and the paper does not explain why text-visual dependency is asymmetric.
- All metrics are relative; an "absolute information quantity" baseline is lacking. The cross-model comparability of RID values needs more controlled experiments.

## Related Work & Insights
- **vs Tuned Lens / Linear Probes**: Probes only tell what is "contained" at a layer; this work tells what is "added" by a module, offering finer granularity.
- **vs Attention Sink / Drift Empirical Studies**: Those works observed phenomena; this work provides a geometric information-theoretic explanation—sink is essentially attention concentrating entropy on a few tokens during reconfiguration.
- **vs Sparse Attention / Attention-Free Models**: This work provides theoretical backing for these approaches—since attention scores are redundant, replacing them with linear or fixed patterns does not harm visual capability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces information geometry into LVLM module interpretation for the first time, with counterintuitive noise replacement findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 models × 7 benchmarks, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Three RQs progress logically, with clear definition-metric-diagnosis flow.
- Value: ⭐⭐⭐⭐⭐ Directly challenges design assumptions of LVLM visual pathways, offering guidance for architecture and efficiency research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2025/multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](../../CVPR2026/multimodal_vlm/can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)

</div>

<!-- RELATED:END -->
