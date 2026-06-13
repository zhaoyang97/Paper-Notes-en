---
title: >-
  [Paper Note] Large Vision-Language Models Get Lost in Attention
description: >-
  [ICML 2026][Multimodal VLM][LVLM interpretability] This paper proposes a geometric information-theoretic framework using "Information Complexity (eRank) + Subspace Support" to quantitatively diagnose the residual flow of…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "LVLM interpretability"
  - "Attention redundancy"
  - "Information theory"
  - "Subspace analysis"
  - "Attention replacement"
date: 2026-05-08
content_hash: 625eaa649d6e5ac1
---

# Large Vision-Language Models Get Lost in Attention

**Conference**: ICML 2026  
**arXiv**: [2605.05668](https://arxiv.org/abs/2605.05668)  
**Code**: Public  
**Area**: Multimodal VLM / Interpretability  
**Keywords**: LVLM interpretability, Attention redundancy, Information theory, Subspace analysis, Attention replacement

## TL;DR
This paper proposes a geometric information-theoretic framework using "Information Complexity (eRank) + Subspace Support" to quantitatively diagnose the residual flow of LVLMs. It finds that Attention primarily performs intra-subspace reconfiguration while FFN injects new semantic dimensions. More surprisingly, replacing learned attention weights with Gaussian noise actually improves performance on most vision tasks, revealing severe mismatch and redundancy in visual attention within contemporary LVLMs.

## Background & Motivation
**Background**: The decoder of an LVLM remains a Transformer with residual connections, where each module outputs an additive update $\Delta\mathbf{X}$ back to the shared residual flow. The mainstream hypothesis is that attention handles in-context reasoning (induction heads, copy mechanisms), while FFN acts like key-value memory for facts. Recent empirical phenomena like visual attention sink and visual attention drift suggest tokens might not truly utilize visual evidence.

**Limitations of Prior Work**: Most analyses stay at the **statistical level**—plotting attention maps, attention rollouts, counting sparse heads, or causal interventions. These tools: (i) lack a unified theoretical basis, making it hard to compare different modules or metrics; (ii) attention weights themselves might not be reliable attribution signals; (iii) there is no unified metric to quantify "what the residual update actually changes in the representation."

**Key Challenge**: Answering "who does what" between attention and FFN requires a unified, comparable metric. While geometric tools like entropy and effective rank have been used in the LLM representation community, module-level interpretation for LVLMs remains unexplored.

**Goal**: (i) Define "what information a representation matrix $\mathbf{X}$ contains"; (ii) Quantize "what the additive update $\Delta\mathbf{X}$ injects into $\mathbf{X}$"; (iii) Use these to diagnose the functional division of LVLM modules, specifically revealing if visual attention performs meaningful work.

**Key Insight**: View the representation matrix on a fixed-rank matrix manifold. SVD naturally yields two geometric objects: the "singular spectrum (complexity)" and the "column/row subspace (semantic support)." The concept of "innovation" from least squares quantifies the "energy exceeding the existing subspace" in the update. This transforms the vague "information change" problem into computable subspace projection residuals.

**Core Idea**: Decompose Transformer residual updates into two orthogonal dimensions: "Innovation (RID)" and "Reconfiguration (MixIG)," then re-examine LVLMs through this lens.

## Method

### Overall Architecture
For residual flow $\mathbf{X}_{\text{new}} = \mathbf{X}_{\text{old}} + \Delta\mathbf{X}$, the authors define representation information as $\mathcal{I}(\mathbf{X}) = (\mathcal{S}_\mathbf{X}, \mathcal{D}_\mathbf{X})$: $\mathcal{S}_\mathbf{X} = \mathrm{eRank}(\mathbf{X})$ describes spectral complexity, and $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$ describes column/row subspace support. Then, **RID** is used to measure "external innovation" (spectrum change + subspace novelty), and **MixIG** measures "internal reconfiguration" (change in token-level mixing entropy). Applying these to Attention and FFN updates yields quantitative evidence of module-level functional roles.

### Key Designs

1.  **SVD Geometric Characterization of Representation Information (RQ1)**:
    - **Function**: Formalizes "what information is in $\mathbf{X}$."
    - **Mechanism**: Parameterize $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ on the matrix manifold $\mathcal{M}_r = \{\mathbf{X} : \mathrm{rank}(\mathbf{X}) = r\}$, obtaining three classes of geometric objects: left singular subspace $\mathcal{C}(\mathbf{X}) \in \mathrm{Gr}(r, S)$ (token correlations), right singular subspace $\mathcal{R}(\mathbf{X}) \in \mathrm{Gr}(r, H)$ (semantic directions), and singular spectrum $\mathbf{\Sigma}$ (energy distribution). Complexity is summarized by effective rank $\mathcal{S}_\mathbf{X} = \exp(-\sum_i p_i \log p_i)$ where $p_i = \sigma_i / \sum \sigma$, and support is summarized by projection operator pairs.
    - **Design Motivation**: Simply observing $\|\mathbf{X}\|_F$ misses structure. Decoupling "how many effective dimensions" from "which directions are covered" using SVD allows distinguishing between "changing energy" and "changing direction."

2.  **RID = Spectral Change + Subspace Innovation (RQ2)**:
    - **Function**: Quantifies whether $\Delta\mathbf{X}$ injects **external new information** (neither in the original spectrum nor the original subspace).
    - **Mechanism**: Spectral change $\Delta\mathcal{S} = |\mathrm{eRank}(\mathbf{X}') - \mathrm{eRank}(\mathbf{X})| / \min(S, H)$; subspace innovation $\Delta\mathcal{D} = \frac{\|(\mathbf{I} - \mathbf{P}_{\mathcal{C}(\mathbf{X})})\mathbf{X}'\|_F + \|\mathbf{X}'(\mathbf{I} - \mathbf{P}_{\mathcal{R}(\mathbf{X})})\|_F}{2\|\mathbf{X}'\|_F}$ inspired by least-squares innovation. Finally, $\mathrm{RID} = \Delta\mathcal{S} + \Delta\mathcal{D} \in [0, 2]$. Considering that positional encodings like RoPE inherently cause non-zero RID, a tolerance $\epsilon_{\text{RoPE}} = \mathrm{RID}(\mathbf{X}^{(\text{RoPE})} \mid \mathbf{X}^{(\text{no-RoPE})})$ is introduced as a baseline.
    - **Design Motivation**: Spectral change alone misses "direction change without dimension change" updates; subspace alone misses "dimension collapse." Combining both fully characterizes external information injection.

3.  **MixIG = Token Mixing Entropy Change + Attention Replacement Experiments (RQ3)**:
    - **Function**: Quantifies how much token reconfiguration $\Delta\mathbf{X}$ performs **within existing subspaces**; uses controlled replacement experiments to map functional division to performance.
    - **Mechanism**: Normalize each row to construct a token-to-token mixing distribution $P_{t,j} \propto \frac{\tilde{\mathbf{x}}_t^\top \tilde{\mathbf{x}}_j + 1}{2}$, taking the mean Shannon entropy as TME; $\mathrm{MixIG} = \mathrm{TME}(\mathbf{X}') - \mathrm{TME}(\mathbf{X})$, where positive values indicate the update mixes tokens more broadly. Diagnostic experiments on 15 LVLMs: replace attention updates with two types of noise—Noise $\mathbf{\Delta}$ (replacing $\Delta\mathbf{X}_{\text{attn}}$ directly with Gaussian noise) and Noise $\mathbf{QKV}$ (replacing Q/K/V weights with Gaussian noise) to see impacts on geometric signals and performance.
    - **Design Motivation**: MixIG captures "in-subspace reordering" invisible to RID. Noise replacement experiments link theoretical metrics to actual downstream performance—if attention were truly critical, performance should collapse.

### Loss & Training
This is a diagnostic framework; no new model is trained. All metrics are geometric quantities from the forward pass. Experiments are conducted across 15 variants from Qwen2.5-VL / LLaVA-1.5 / LLaVA-NeXT families, evaluated on 7 benchmarks including POPE, 3DSRBench, RealWorldQA, MMMU, VMCBench, MathVista, and HallusionBench (1000 samples per category).

## Key Experimental Results

### Main Results
Module-level RID/MixIG aggregated across models (Table 1):

| Module | RID | MixIG | Functional Features |
|------|-----|-------|---------|
| Noise $\mathbf{\Delta}$ | 0.61 | -0.80 | High RID + Negative MixIG (off-manifold perturbation) |
| Noise $\mathbf{QKV}$ | 0.44 | -0.50 | High RID + Negative MixIG |
| **Attention** | **0.06** | **0.61** | **Low RID + High MixIG** (Subspace-preserving + Reconfiguration) |
| **FFN** | **0.21** | **0.02** | **High RID + Low MixIG** (Subspace-expanding + Innovation) |

The separation is stable across 15 models: Attention's RID is almost equal to $\epsilon_{\text{RoPE}} = 0.062$, meaning it **injects almost no new support directions** and focuses entirely on mixing. FFN's RID is significantly higher, identifying it as the true source of innovation.

### Ablation Study
SAP (Stochastic Attention Probing) noise replacement experiments (partial data from Table 2, Qwen-2.5-VL-3B):

| Configuration | POPE | RWQA | 3dSRBench |
|------|------|------|-----------|
| Vanilla | 86.13 | 59.35 | 53.46 |
| + Vis. Attn. (Noise Replace) | 87.58 | 61.38 | — |

On most vision tasks, **replacing learned visual attention weights with Gaussian noise actually improves performance**—this is the most dramatic finding of the paper.

### Key Findings
- Attention and FFN functions are geometrically orthogonal: Attention = subspace-preserving operator (reconfiguration); FFN = subspace-expansion operator (innovation). The traditional hypothesis of "attention for in-context, FFN for memory" is solidified by geometric evidence.
- LVLM visual attention exhibits massive redundancy; effective information in attention scores is minimal, corroborating attention sink and attention drift phenomena.
- Since attention complexity is a $O(S^2)$ bottleneck yet redundant, this work provides strong theoretical and empirical grounds for approximate attention (sparse, predefined, low-rank) on visual tokens.

## Highlights & Insights
- "RID + MixIG" is an elegant dual-metric set decomposing residual updates into "adding new bases" and "mixing within old bases." This language is more general than tool-specific methods like attention rollout or tuned lens.
- Noise replacement experiments are more radical than typical ablations—replacing learned weights with random noise without affecting vision tasks suggests training objectives for LVLMs might not provide strong learning signals for visual attention. This motivates "visual token pruning" and "attention-free visual fusion."
- Introduction of the RoPE baseline is practical—subtracting "fake RID" prevents misinterpretation, reflecting engineering rigor in metric design.

## Limitations & Future Work
- The framework only examines single-step additive updates; cumulative multi-layer effects (identifying series of layers for innovation) require more analysis.
- Noise replacement results are concentrated on the vision side; attention remains critical for textual tasks (e.g., MathVista), and the paper doesn't explain the asymmetry.
- Metrics are relative; an "absolute information volume" benchmark is missing. Cross-model comparability of RID values needs more controlled experiments.

## Related Work & Insights
- **vs Tuned Lens / Linear Probes**: Probes tell you what a layer "contains," while this work tells you what a module "adds," offering finer granularity.
- **vs Attention Sink / Drift Empirical Studies**: Those works identify phenomena; this work provides a geometric information-theoretic explanation—sink is essentially attention concentrating entropy into a few tokens during reconfiguration.
- **vs Sparse Attention / Attention-Free Models**: This work provide theoretical backing—since attention scores are redundant, replacing them with linear or fixed patterns without losing visual capabilities is reasonable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces information geometry to LVLM module interpretation with counter-intuitive findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 models × 7 benchmarks, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Progressive structure, clear logical flow between definitions, metrics, and diagnosis.
- Value: ⭐⭐⭐⭐⭐ Directly challenges design assumptions of LVLM vision paths, guiding architecture and efficiency research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)

</div>

<!-- RELATED:END -->
