---
title: >-
  [Paper Note] Large Vision-Language Models Get Lost in Attention
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] This paper quantitatively diagnoses the residual streams of LVLMs using a geometric information theory framework of "Information Complexity (eRank) + Subspace Support." It finds that Attention primarily performs intra-subspace reconfiguration while FFN injects new semantic dimensions. More surprisingly, replacing learn
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: e77fd30d50cd101d
---
# Large Vision-Language Models Get Lost in Attention

**Conference**: ICML 2026  
**arXiv**: [2605.05668](https://arxiv.org/abs/2605.05668)  
**Code**: Public  
**Area**: Multimodal VLM / Interpretability  
**Keywords**: LVLM Interpretability, Attention Redundancy, Information Theory, Subspace Analysis, Attention Replacement

## TL;DR
This paper quantitatively diagnoses the residual streams of LVLMs using a geometric information theory framework of "Information Complexity (eRank) + Subspace Support." It finds that Attention primarily performs intra-subspace reconfiguration while FFN injects new semantic dimensions. More surprisingly, replacing learned attention weights with Gaussian noise maintains or even improves performance on most vision tasks, revealing severe mismatch and redundancy in contemporary LVLM visual attention.

## Background & Motivation
**Background**: LVLM decoders remain Transformers with residual connections, where each module outputs an additive update $\Delta\mathbf{X}$ to the shared residual stream. The prevailing assumption is that attention handles in-context reasoning (induction heads, copy mechanisms), while FFNs act as key-value memory for facts. Recent empirical phenomena in LVLMs, such as visual attention sinks and visual attention drift, suggest that models might not be effectively utilizing visual evidence.

**Limitations of Prior Work**: Most existing analyses remain at the **statistical level**—visualizing attention weight maps, performing attention rollout, counting sparse heads, or conducting causal interventions. However, these tools: (i) lack a unified theoretical foundation, making it difficult to compare conclusions across different modules or metrics; (ii) attention weights themselves have been identified by Jain & Wallace and others as potentially unreliable attribution signals; (iii) there is no unified measure to quantify "what the residual update actually represents."

**Key Challenge**: To answer "what attention vs. FFN does," a unified and comparable metric is required. While the LLM representation analysis community has used geometric tools like entropy and effective rank to characterize cross-layer quality, module-level interpretation for LVLMs remains unexplored.

**Goal**: (i) Define "what information a representation matrix $\mathbf{X}$ contains"; (ii) Quantify "what the additive update $\Delta\mathbf{X}$ injects into $\mathbf{X}$"; (iii) Use these metrics to diagnose the functional division of LVLM modules, particularly uncovering whether visual attention performs meaningful work.

**Key Insight**: By viewing the representation matrix on a fixed-rank matrix manifold, SVD naturally yields two geometric objects: the "singular spectrum (complexity)" and the "column/row subspaces (semantic support)." The concept of "innovation" from least squares can then be used to quantify the energy in the update that "exceeds the existing subspace." This transforms the vague problem of "information change" into computable subspace projection residuals.

**Core Idea**: Decompose Transformer residual updates into two orthogonal dimensions: "Innovation (RID)" vs. "Reconfiguration (MixIG)," and re-examine LVLMs through this lens.

## Method

### Overall Architecture
For the residual stream $\mathbf{X}_{\text{new}} = \mathbf{X}_{\text{old}} + \Delta\mathbf{X}$, the authors define representation information as $\mathcal{I}(\mathbf{X}) = (\mathcal{S}_\mathbf{X}, \mathcal{D}_\mathbf{X})$, where $\mathcal{S}_\mathbf{X} = \mathrm{eRank}(\mathbf{X})$ describes spectral complexity and $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$ describes column/row subspace support. **RID** is then used to measure "external innovation" (spectral change + subspace novelty), and **MixIG** measures "internal reconfiguration" (changes in token-level mixing entropy). Applying these metrics to Attention and FFN updates provides quantitative evidence of module-level functional division.

### Key Designs

**1. Geometric Characterization of "Information in a Representation Matrix" via SVD**

To determine the roles of attention and FFN, "information" must first be clearly defined. Simply looking at norms like $\|\mathbf{X}\|_F$ only captures energy, not structure. This paper places the representation matrix on a fixed-rank manifold $\mathcal{M}_r = \{\mathbf{X} : \mathrm{rank}(\mathbf{X}) = r\}$ and uses SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ to extract three types of geometric objects: the left-singular subspace $\mathcal{C}(\mathbf{X})$ characterizing token correlations, the right-singular subspace $\mathcal{R}(\mathbf{X})$ characterizing semantic directions, and the singular spectrum $\mathbf{\Sigma}$ characterizing energy distribution. Complexity is then defined as effective rank $\mathcal{S}_\mathbf{X} = \exp(-\sum_i p_i \log p_i)$ (where $p_i = \sigma_i / \sum \sigma$), and semantic support as the pair of subspace projection operators $\mathcal{D}_\mathbf{X} = (\mathbf{P}_{\mathcal{C}(\mathbf{X})}, \mathbf{P}_{\mathcal{R}(\mathbf{X})})$. This decouples "how many effective dimensions changed" from "which directions were covered," allowing differentiation between updates that change energy versus those that change direction.

**2. RID: Quantifying Injection of External New Information**

With the definition of representation information, one can measure exactly what $\Delta\mathbf{X}$ contributes. RID decomposes this into two complementary components: spectral change $\Delta\mathcal{S} = |\mathrm{eRank}(\mathbf{X}') - \mathrm{eRank}(\mathbf{X})| / \min(S, H)$ capturing dimension shifts, and subspace innovation $\Delta\mathcal{D} = \frac{\|(\mathbf{I} - \mathbf{P}_{\mathcal{C}(\mathbf{X})})\mathbf{X}'\|_F + \|\mathbf{X}'(\mathbf{I} - \mathbf{P}_{\mathcal{R}(\mathbf{X})})\|_F}{2\|\mathbf{X}'\|_F}$ capturing energy falling outside the original subspace (based on the "innovation" concept). Thus, $\mathrm{RID} = \Delta\mathcal{S} + \Delta\mathcal{D} \in [0, 2]$. Both components are necessary: spectral change alone misses "direction-swapping" updates, and subspace novelty alone misses "dimension collapse." A practical detail: since positional encodings like RoPE inherently result in non-zero RID, a tolerance baseline $\epsilon_{\text{RoPE}} = \mathrm{RID}(\mathbf{X}^{(\text{RoPE})} \mid \mathbf{X}^{(\text{no-RoPE})})$ is introduced to subtract "spurious innovation" caused by position encoding.

**3. MixIG + Noise Replacement: Quantifying "Intra-subspace Token Re-shuffling" and Linking Metrics to Performance**

RID cannot capture updates that don't introduce new directions but instead redistribute tokens within the existing subspace. MixIG fills this gap: it normalizes token rows and constructs a token-to-token mixing distribution $P_{t,j} \propto \frac{\tilde{\mathbf{x}}_t^\top \tilde{\mathbf{x}}_j + 1}{2}$, calculating the Token Mixing Entropy (TME). $\mathrm{MixIG} = \mathrm{TME}(\mathbf{X}') - \mathrm{TME}(\mathbf{X})$, where positive values indicate the update spreads token interactions more widely. To validate these geometric signals, the authors designed a controlled replacement experiment across 15 open-source LVLMs. Attention updates are replaced with two types of noise: Noise $\mathbf{\Delta}$ replaces $\Delta\mathbf{X}_{\text{attn}}$ with Gaussian noise, and Noise $\mathbf{QKV}$ replaces $Q/K/V$ matrices with Gaussian weights. The logic is straightforward: if attention performs meaningful work, randomization should degrade performance. The result—performance actually increases for most vision tasks—confirms the MixIG/RID finding that attention primarily performs intra-subspace reshuffling without injecting new information.

### Loss & Training
This work is a diagnostic framework and does not train new models. All metrics are geometric quantities computed during a forward pass. Experiments were conducted on 15 variants across the Qwen2.5-VL, LLaVA-1.5, and LLaVA-NeXT families using 7 benchmarks: POPE, 3DSRBench, RealWorldQA, MMMU, VMCBench, MathVista, and HallusionBench, using 1000 samples per category for statistics.

## Key Experimental Results

### Main Results
Module-level RID/MixIG aggregated across models (Table 1):

| Module | RID | MixIG | Functional Characteristics |
|------|-----|-------|---------|
| Noise $\mathbf{\Delta}$ | 0.61 | -0.80 | High RID + Negative MixIG (off-manifold perturbation) |
| Noise $\mathbf{QKV}$ | 0.44 | -0.50 | High RID + Negative MixIG |
| **Attention** | **0.06** | **0.61** | **Low RID + High MixIG** (Subspace-preserving + Reconfiguration) |
| **FFN** | **0.21** | **0.02** | **High RID + Low MixIG** (Subspace-expanding + Innovation) |

The separation remains highly consistent across 15 models: Attention's RID is nearly equal to $\epsilon_{\text{RoPE}} = 0.062$, indicating it **almost never introduces new support directions** and consists entirely of mixing. FFN's RID is significantly higher than this baseline, identifying it as the true source of innovation.

### Ablation Study
SAP (Stochastic Attention Probing) noise replacement experiments (Selection from Table 2, Qwen-2.5-VL-3B):

| Configuration | POPE | RWQA | 3dSRBench |
|------|------|------|-----------|
| Vanilla | 86.13 | 59.35 | 53.46 |
| + Vis. Attn. (Noise replacement) | 87.58 | 61.38 | — |

On most vision tasks, **replacing learned visual attention weights with Gaussian noise actually results in non-decreasing or improved performance**—the most dramatic finding of the paper.

### Key Findings
- Attention and FFN functions are geometrically orthogonal: Attention = subspace-preserving reconfiguration operator; FFN = subspace-expanding innovation operator. Prior hypotheses regarding "attention for in-context, FFN for memory" are substantiated by geometric evidence.
- Visual attention in LVLMs is heavily redundant, with attention scores carrying very little effective information; this corroborates the attention sink and attention drift phenomena.
- Given that attention complexity is the primary $O(S^2)$ bottleneck yet is redundant, this work provides strong theoretical and empirical motivation for approximate attention (sparse, predefined, or low-rank) on visual tokens.

## Highlights & Insights
- "RID + MixIG" constitutes an elegant set of dual metrics, decomposing residual updates into "adding new bases" and "shuffling within old bases." This language is more universal than tool-specific methods like attention rollout or tuned lens, as it can be applied to any additive update module.
- The noise replacement experiment is more aggressive than any ablation—replacing "learned weights with total randomness" without hurting vision tasks implies that the LVLM training objective may provide excessively weak learning signals for visual attention. This provides direct motivation for visual token pruning and attention-free visual fusion.
- The inclusion of the RoPE baseline is pragmatic—subtracting the "spurious RID" introduced by positional encoding avoids misinterpretation and demonstrates engineering awareness in developing portable tools.

## Limitations & Future Work
- The framework only examines single-step additive updates; cumulative cross-layer effects (identifying which sequences of layers constitute true innovation) require further analysis.
- Noise replacement experiments focused on the vision side; attention remains critical for pure text tasks (e.g., MathVista), and this paper does not explain why text-vision dependency is asymmetric.
- The metrics are relative; a baseline for "absolute information content" is lacking. Cross-model comparability of RID values requires more controlled experimentation.

## Related Work & Insights
- **vs. Tuned Lens / Linear Probes**: Probes tell you "what resides" in a layer; this work tells you "what was added" by a module, providing finer granularity.
- **vs. Attention Sink / Drift Empirical Studies**: While prior works identify phenomena, this paper provides a geometric information-theoretic explanation—sinks are essentially attention concentrating entropy into a few tokens during reconfiguration.
- **vs. Sparse Attention / Attention-Free Models**: This work provides theoretical backing for these directions—if attention scores are redundant, replacing them with linear or fixed patterns without losing visual capabilities is reasonable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introduces information geometry for LVLM module interpretation with counter-intuitive findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 15 models across 7 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear logic progressing from definition to metric to diagnosis across three Research Questions.
- Value: ⭐⭐⭐⭐⭐ Directly challenges design assumptions of the LVLM vision path, offering guidance for both architecture and efficiency research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)

</div>

<!-- RELATED:END -->
