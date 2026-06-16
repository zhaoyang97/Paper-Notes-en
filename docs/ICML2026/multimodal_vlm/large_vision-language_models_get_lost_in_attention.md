---
title: >-
  [Paper Note] Large Vision-Language Models Get Lost in Attention
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] This paper quantitatively diagnoses the residual streams of LVLMs using a geometric information theory framework of "Information Complexity (eRank) + Subspace Support." It finds that Attention almost exclusively performs intra-subspace reconfiguration while FFN injects new semantic dimensions. More surprisingly, replac
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 4bf46a974044f3b4
---
# Large Vision-Language Models Get Lost in Attention

**Conference**: ICML 2026  
**arXiv**: [2605.05668](https://arxiv.org/abs/2605.05668)  
**Code**: Available  
**Area**: Multimodal VLM / Interpretability  
**Keywords**: LVLM Interpretability, Attention Redundancy, Information Theory, Subspace Analysis, Attention Replacement

## TL;DR
This paper quantitatively diagnoses the residual streams of LVLMs using a geometric information theory framework of "Information Complexity (eRank) + Subspace Support." It finds that Attention almost exclusively performs intra-subspace reconfiguration while FFN injects new semantic dimensions. More surprisingly, replacing learned attention weights with Gaussian noise often results in maintained or even improved performance on visual tasks, revealing severe mismatch and redundancy in visual attention within modern LVLMs.

## Background & Motivation
**Background**: The decoders of LVLMs remain Transformers with residual connections, where each module outputs an additive update $\Delta\mathbf{X}$ to the shared residual stream. The mainstream hypothesis is that attention is responsible for in-context reasoning (induction heads, copy mechanisms), while FFNs act as key-value memories storing facts. Recent empirical phenomena such as visual attention sinks and visual attention drift in LVLMs suggest that these models might not be truly utilizing visual evidence.

**Limitations of Prior Work**: Existing analyses mostly remain at a **statistical level**—plotting attention maps, performing attention rollout, counting sparse heads, or conducting causal interventions. However, these tools: (i) lack a unified theoretical foundation, making it difficult to compare conclusions across different modules or metrics; (ii) attention weights themselves have been noted by Jain & Wallace and others as potentially unreliable attribution signals; (iii) there is no unified measure to quantify "exactly what information the residual update changes."

**Key Challenge**: To answer "what attention vs. FFN performs," a unified and comparable metric is required. The LLM representation analysis community has used geometric tools like entropy and effective rank to characterize quality across layers, but module-level interpretation in LVLMs remains unexplored.

**Goal**: (i) Define "what information a representation matrix $\mathbf{X}$ entails"; (ii) Quantify "what the additive update $\Delta\mathbf{X}$ injects into $\mathbf{X}$"; (iii) Use these two quantities to diagnose the functional division of LVLM modules, specifically revealing whether visual attention is performing meaningful work.

**Key Insight**: By viewing representation matrices on a manifold of fixed-rank matrices, SVD naturally yields two geometric objects: the "singular spectrum (complexity)" and the "column/row subspaces (semantic support)." The concept of "innovation" from least squares is then used to quantify the "energy in the update that exceeds the existing subspace." This transforms the vague problem of "information change" into computable subspace projection residuals.

**Core Idea**: Decompose the Transformer residual update into two orthogonal dimensions: "Innovation (RID) vs. Reconfiguration (MixIG)," and re-examine LVLMs through this lens.

## Method

### Overall Architecture
For the residual stream $\mathbf{X}_{\text{new}} = \mathbf{X}_{\text{old}} + \Delta\mathbf{X}$, the authors define representation information as $\mathcal{I}(\mathbf{X}) = (\mathcal{S}_\mathbf{X}, \mathcal{D}_\mathbf{X})$, where $\mathcal{S}_\mathbf{X} = \mathrm{eRank}(\mathbf{X})$ describes spectral complexity and $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$ describes column/row subspace support. Then, **RID** measures "external innovation" (spectral changes + subspace novelty), and **MixIG** measures "internal reconfiguration" (changes in token-level mixing entropy). Applying these metrics to Attention and FFN updates provides quantitative evidence of module-level functional division.

### Key Designs

**1. Characterizing "information in a representation matrix" via SVD geometry**

To determine what attention and FFNs do, "information" must first be defined—simply examining the Frobenius norm $\|\mathbf{X}\|_F$ only shows energy magnitude, not structure. This work places the representation matrix on a fixed-rank manifold $\mathcal{M}_r = \{\mathbf{X} : \mathrm{rank}(\mathbf{X}) = r\}$ and uses SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ to extract three types of geometric objects: the left singular subspace $\mathcal{C}(\mathbf{X})$ characterizing token correlations, the right singular subspace $\mathcal{R}(\mathbf{X})$ characterizing semantic directions, and the singular spectrum $\mathbf{\Sigma}$ characterizing energy distribution. Based on this, complexity is summarized as effective rank $\mathcal{S}_\mathbf{X} = \exp(-\sum_i p_i \log p_i)$ (where $p_i = \sigma_i / \sum \sigma$), and semantic support as a pair of projection operators for column/row subspaces $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$. Consequently, "how many effective dimensions were changed" and "which directions were covered" are decoupled, making it possible to distinguish whether an update changed energy or direction.

**2. RID: Quantifying whether the update injects external new information "outside the original spectrum and subspace"**

With representation information defined, one can ask what $\Delta\mathbf{X}$ contributes. RID decomposes this into two complementary quantities: spectral change $\Delta\mathcal{S} = |\mathrm{eRank}(\mathbf{X}') - \mathrm{eRank}(\mathbf{X})| / \min(S, H)$ to capture "dimension count changes," and subspace innovation $\Delta\mathcal{D} = \frac{\|(\mathbf{I} - \mathbf{P}_{\mathcal{C}(\mathbf{X})})\mathbf{X}'\|_F + \|\mathbf{X}'(\mathbf{I} - \mathbf{P}_{\mathcal{R}(\mathbf{X})})\|_F}{2\|\mathbf{X}'\|_F}$ to capture "energy falling outside the original subspace" via the least-squares innovation concept. Together, $\mathrm{RID} = \Delta\mathcal{S} + \Delta\mathcal{D} \in [0, 2]$. Both components are necessary: spectral change alone misses "direction changes without dimension changes," while subspace alone misses "dimensional collapse." A practical detail—positional encodings like RoPE inherently result in non-zero RID, so a tolerance $\epsilon_{\text{RoPE}} = \mathrm{RID}(\mathbf{X}^{(\text{RoPE})} \mid \mathbf{X}^{(\text{no-RoPE})})$ is introduced as a baseline to subtract "spurious innovation" caused by positional encoding.

**3. MixIG + Noise Replacement: Quantifying "intra-subspace token rearrangement" and linking metrics to real performance**

RID is blind to another type of update—those that do not introduce new directions but merely remix tokens within the existing subspace. MixIG fills this gap: it constructs a token-to-token mixing distribution $P_{t,j} \propto \frac{\tilde{\mathbf{x}}_t^\top \tilde{\mathbf{x}}_j + 1}{2}$ after normalizing each token row, and computes TME from the average Shannon entropy. $\mathrm{MixIG} = \mathrm{TME}(\mathbf{X}') - \mathrm{TME}(\mathbf{X})$, where a positive value implies the update mixed tokens more broadly. To link geometric signals to downstream performance, a controlled replacement experiment is designed: across 15 open-source LVLMs, attention updates are replaced with two types of noise—Noise $\mathbf{\Delta}$ replaces $\Delta\mathbf{X}_{\text{attn}}$ with Gaussian noise, and Noise $\mathbf{QKV}$ replaces Q/K/V matrices with Gaussian weights. The logic is direct: if attention performs meaningful work, replacing it with randomness should cause collapse. Instead, performance on most visual tasks improved, validating the RID/MixIG finding that attention primarily performs intra-subspace rearrangement without injecting new information.

### Loss & Training
This is a diagnostic framework; no new models were trained. All metrics are geometric quantities from the forward pass. Experiments were conducted on 15 variants across the Qwen2.5-VL, LLaVA-1.5, and LLaVA-NeXT families, evaluated on 7 benchmarks including POPE, 3DSRBench, RealWorldQA, MMMU, VMCBench, MathVista, and HallusionBench, using 1000 samples per category.

## Key Experimental Results

### Main Results
Module-level RID/MixIG aggregated across models (Table 1):

| Module | RID | MixIG | Functional Characteristics |
|------|-----|-------|---------|
| Noise $\mathbf{\Delta}$ | 0.61 | -0.80 | High RID + Negative MixIG (off-manifold perturbation) |
| Noise $\mathbf{QKV}$ | 0.44 | -0.50 | High RID + Negative MixIG |
| **Attention** | **0.06** | **0.61** | **Low RID + High MixIG** (Subspace-preserving + Reconfiguration) |
| **FFN** | **0.21** | **0.02** | **High RID + Low MixIG** (Subspace-expanding + Innovation) |

The separation across 15 models is highly stable: the RID of attention is nearly equal to $\epsilon_{\text{RoPE}} = 0.062$, indicating it **introduces almost no new support directions** and is purely focused on mixing. The FFN RID is significantly above this baseline, acting as the true source of innovation.

### Ablation Study
SAP (Stochastic Attention Probing) noise replacement experiments (partial data from Table 2, Qwen-2.5-VL-3B):

| Configuration | POPE | RWQA | 3dSRBench |
|------|------|------|-----------|
| Vanilla | 86.13 | 59.35 | 53.46 |
| + Vis. Attn. (Noise Replacement) | 87.58 | 61.38 | — |

In most visual tasks, **replacing learned visual attention weights with Gaussian noise results in maintained or improved performance**—the most dramatic finding of the paper.

### Key Findings
- The functions of Attention and FFN are geometrically orthogonal: Attention = subspace-preserving operator (reconfiguration); FFN = subspace-expanding operator (innovation). The prior hypothesis that "attention does in-context and FFN does memory" is solidified by geometric evidence.
- Visual attention in LVLMs is significantly redundant; attention scores carry very little effective information. This corroborates the phenomena of attention sink and attention drift.
- Since attention complexity is the primary $O(S^2)$ bottleneck yet is redundant, this work provides strong theoretical and empirical grounds for approximate attention (sparse/predefined/low-rank) on visual tokens.

## Highlights & Insights
- "RID + MixIG" provides an elegant pair of dual metrics, decomposing residual updates into "adding new bases" and "mixing within old bases." This language is more universal than tools like attention rollout or tuned lens and can be applied to any additive update module.
- The noise replacement experiment is more radical than any ablation—replacing learned weights with pure randomness without affecting visual tasks implies that LVLM training objectives may provide weak learning signals for visual attention. This provides direct motivation for visual token pruning and attention-free visual fusion.
- The inclusion of the RoPE baseline is practical—subtracting "fake RID" introduced by positional encoding avoids misinterpretation, demonstrating the authors' intent to create portable diagnostic tools.

## Limitations & Future Work
- The framework only examines single-step additive updates; cumulative effects across layers (which layer combinations constitute true innovation) require further analysis.
- Noise replacement experiments focused on the vision side; attention remains critical for pure text tasks (e.g., MathVista), and this work does not explain the asymmetrical text-visual dependency.
- Metrics are relative; there is a lack of an "absolute information" benchmark. Comparative RID values across different models require more controlled experiments.

## Related Work & Insights
- **vs. Tuned Lens / Linear Probes**: Probes tell you "what a layer contains," whereas this work tells you "what a module adds," providing finer granularity.
- **vs. Attention Sink / Drift Empirical Studies**: While those works identify phenomena, this paper provides a geometric information-theoretic explanation—sinks occur because attention concentrates entropy into a few tokens during reconfiguration.
- **vs. Sparse Attention / Attention-Free Models**: This work provides a theoretical backend for such efforts—since attention scores are redundant, replacing them with linear or fixed patterns without losing visual capabilities is justifiable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces information geometry to LVLM module interpretation for the first time, with counter-intuitive noise replacement findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 models × 7 benchmarks, covering a wide range.
- Writing Quality: ⭐⭐⭐⭐ Three RQs progress logically with clear definition-metric-diagnosis structure.
- Value: ⭐⭐⭐⭐⭐ Directly challenges design assumptions of LVLM visual paths, offering guidance for both architecture and efficiency research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](../../CVPR2026/multimodal_vlm/can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)

</div>

<!-- RELATED:END -->
