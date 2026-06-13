---
title: >-
  [Paper Note] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness
description: >-
  [ICLR 2026][Interpretability][Language model planning] This paper proposes an information-theoretic framework based on VQ-VAE to analyze internal planning behavior in language models…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Language model planning"
  - "mutual information"
  - "VQ-VAE"
  - "horizon awareness"
  - "branch awareness"
date: 2026-05-08
content_hash: 5ebf3ccd8588c326
---

# Internal Planning in Language Models: Characterizing Horizon and Branch Awareness

**Conference**: ICLR 2026
**arXiv**: [2509.25260](https://arxiv.org/abs/2509.25260)  
**Code**: Available (with supplementary material)  
**Area**: Interpretability
**Keywords**: Language model planning, mutual information, VQ-VAE, horizon awareness, branch awareness

## TL;DR
This paper proposes an information-theoretic framework based on VQ-VAE to analyze internal planning behavior in language models, finding that planning horizon is task-dependent, that models implicitly retain information about unchosen correct paths, and that next-token decisions rely primarily on the most recent computations.

## Background & Motivation
LLMs exhibit remarkable capabilities, yet their training objective—next-token prediction—appears purely local, which seemingly contradicts the forward-looking nature of planning. This raises a core question: **to what extent are language models "horizon-aware" and "branch-aware"?**

Horizon awareness refers to the property that a good planner already accounts for long-term goals in its current decisions, analogous to model predictive control (MPC). Branch awareness refers to the property that a good planner maintains multiple possible futures simultaneously before committing to a decision, analogous to Tree-of-Thoughts.

Existing analysis methods have notable limitations: (1) circuit discovery requires substantial manual engineering; (2) linear probes may conflate the representations learned by the probe itself with the information actually encoded by the model (the probe confounding problem). An automated, confounding-free, and scalable analysis method is therefore needed.

Core Idea: VQ-VAE is used to compress the high-dimensional hidden states of a language model into discrete codes, after which mutual information (MI) between discrete codes is computed directly to measure information sharing across internal computations.

## Method

### Overall Architecture
The framework proceeds in three steps: Step 1 trains a VQ-VAE to compress hidden state blocks into discrete codes $Z_\mathcal{S}$; Step 2 uses the trained encoder to obtain the joint distribution of all discrete codes over a dataset; Step 3 computes mutual information $I(Z_A; Z_B)$ and normalized mutual information nMI to analyze information relationships between different computational blocks.

### Key Designs

1. **VQ-VAE Hidden State Compression**:

    - Function: Maps a variable-length set of high-dimensional hidden states $G_\mathcal{S} = \{h_t^\ell | (\ell,t) \in \mathcal{S}\}$ to a discrete code $Z_\mathcal{S} \in [K]$
    - Mechanism: A Transformer encoder maps the variable-length input to a fixed-dimensional latent vector $r_\mathcal{S}$, which is then quantized to a discrete code by finding the nearest neighbor in codebook $\{e_k\}_{k=1}^K$: $k^* = \arg\min_k \|r_\mathcal{S} - e_k\|_2^2$
    - Training objective: $\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_q \mathcal{L}_{\text{vq}} + \lambda_{\text{cos}} \mathcal{L}_{\text{cos}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$. Cosine similarity penalties and entropy regularization are additionally incorporated to ensure codebook diversity and full utilization.
    - Design Motivation: Discrete codes preserve the critical distinctions between different computations while filtering out fine-grained redundant details, thereby yielding more stable MI estimation.

2. **Horizon of the Plan Analysis**:

    - Function: Quantifies how much information about future tokens is contained in the prefix computation.
    - Mechanism: The nMI between the summary code $Z_{1:T}^{1:L-1}$ of all hidden state blocks in the prefix $H = \{h_t^\ell | t=1,...,T; \ell=1,...,L-1\}$ and the last-layer hidden state code $Z_{T+\tau}^L$ of the $\tau$-th generated token is computed. A slow decay of nMI with increasing $\tau$ indicates that the prefix encodes information over a long horizon.
    - Design Motivation: MI is more robust than probing—it does not introduce additional model capacity and directly measures information sharing.

3. **Branches in the Plan Analysis**:

    - Function: Tests whether the model also encodes information about alternative correct paths when generating a correct answer.
    - Mechanism: In the path-finding (PF) task, each sample is constructed to have 2 correct paths and 1 decoy path, with no shared nodes among the three. The MI between the prefix summary code and the alternative correct path code is compared to the MI with the decoy path code; a ratio $\mathcal{I}(Z_H; Z_{\text{alt}}) / \mathcal{I}(Z_H; Z_{\text{decoy}}) > 1$ indicates branch awareness.
    - Design Motivation: The no-shared-node constraint among the three paths eliminates trivial overlap as a confounding explanation.

### Experimental Setup
A GPT-3 Small architecture (with RoPE) is used, and analysis is conducted on three data types: (1) context-free grammars (CFG)—local syntactic rules; (2) path-finding (PF)—a graph task requiring multi-step reasoning; (3) natural language (OpenWebText). Differences between NTP and MTP training objectives are also examined.

## Key Experimental Results

### Planning Horizon (nMI Decay Patterns)

| Task | nMI Decay Rate | Interpretation |
|------|----------------|----------------|
| CFG (context-free grammar) | Rapid decay; drops to 1/5 of initial value at $\tau$=10 | Short horizon, local planning |
| PF-Short (4-node path) | nMI increases at $\tau$>1 | Non-myopic; prefix encodes subsequent nodes |
| PF-Long (6-node path) | nMI remains high at intermediate nodes | Long-horizon planning |

### Branch Awareness

| Model | PF-Short MI Ratio | PF-Short Accuracy | PF-Long MI Ratio | PF-Long Accuracy |
|-------|-------------------|-------------------|------------------|------------------|
| NTP | 7.60±0.78 | 0.92 | 1.45±0.01 | 0.60 |
| MTP | 6.29±0.17 | 0.88 | 1.82±0.27 | 0.85 |

### Key Findings
- **Planning horizon is task-dependent**: nMI decays rapidly on CFG (short-horizon planning) but remains high or even increases on PF (long-horizon planning).
- **On the PF task, nMI at the second intermediate node is higher than at the first**, potentially suggesting a "backward-from-goal" reasoning strategy.
- **Branch awareness is genuine**: MI ratios far exceed 1 (reaching 7.6 on PF-Short), confirming that the model retains information about unchosen correct paths.
- MTP training marginally reduces myopic behavior, but the difference between NTP and MTP is not substantial.
- Next-token decisions rely primarily on high-layer and recent computational blocks (recency effect).

## Highlights & Insights
- The VQ-VAE + MI analytical framework is broadly applicable—it avoids the probe confounding problem and the manual engineering required by circuit discovery.
- The finding that models internally retain information about alternative paths has important implications for understanding the robustness of language models.
- The higher nMI at the second node than the first in the PF task suggests implicit "backward planning," consistent with human problem-solving strategies.

## Limitations & Future Work
- VQ-VAE compression inevitably incurs information loss, making the absolute values of MI estimates unreliable; the authors acknowledge that only relative trends are analyzed.
- Experiments are conducted on GPT-3 Small (~125M parameters); planning behavior in larger models may differ.
- Analysis on natural language (OpenWebText) is limited to diagnosing computational history and does not extend to horizon or branch analysis.
- Inconsistencies in NTP/MTP differences may be related to model scale.

## Related Work & Insights
- **vs. Linear Probes**: Probes introduce additional expressive capacity that confounds results; the VQ-VAE + MI approach is immune to this issue.
- **vs. Circuit Discovery**: Circuit discovery requires extensive manual engineering and does not scale readily; the proposed framework is automated and general-purpose.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The VQ-VAE + MI analysis paradigm is entirely novel, and the three analytical dimensions are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three task types provide broad coverage, though model scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Framework is rigorous, equations are clear, and appendices are highly detailed.
- Value: ⭐⭐⭐⭐ Provides a new tool for LM interpretability, though practical application scenarios remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](../../ACL2026/interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)
- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](../../ICML2026/interpretability/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)

</div>

<!-- RELATED:END -->
