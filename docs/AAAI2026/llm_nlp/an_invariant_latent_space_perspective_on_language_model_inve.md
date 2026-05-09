---
title: >-
  [Paper Note] An Invariant Latent Space Perspective on Language Model Inversion
description: >-
  [AAAI 2026][LLM/NLP][language model inversion] This paper proposes the Invariant Latent Space Hypothesis (ILSH), which reframes the LLM inversion problem as reusing the LLM's own latent space. The Inv²A framework is designed to map outputs to denoised pseudo-representations via a lightweight inverse encoder, which are then decoded by a frozen LLM to recover hidden prompts. Inv²A achieves an average BLEU improvement of 4.77% across 9 datasets and attains comparable performance with only 20% of the training data.
tags:
  - AAAI 2026
  - LLM/NLP
  - language model inversion
  - privacy attack
  - invariant latent space
  - prompt recovery
  - contrastive learning
date: 2026-05-08
content_hash: 977de5d74eb74903
---

# An Invariant Latent Space Perspective on Language Model Inversion

**Conference**: AAAI 2026
**arXiv**: [2511.19569v1](https://arxiv.org/abs/2511.19569v1)
**Code**: [https://github.com/yyy01/Invariant_Attacker](https://github.com/yyy01/Invariant_Attacker)
**Area**: AI Security / LLM Privacy Attacks
**Keywords**: language model inversion, privacy attack, invariant latent space, prompt recovery, contrastive learning

## TL;DR
This paper proposes the Invariant Latent Space Hypothesis (ILSH), which reframes the LLM inversion problem as reusing the LLM's own latent space. The Inv²A framework is designed to map outputs to denoised pseudo-representations via a lightweight inverse encoder, which are then decoded by a frozen LLM to recover hidden prompts. Inv²A achieves an average BLEU improvement of 4.77% across 9 datasets and attains comparable performance with only 20% of the training data.

## Background & Motivation
Large language models (LLMs) have been widely deployed across diverse scenarios, with their outputs broadly created and disseminated. This gives rise to a new security threat: **Language Model Inversion (LMI)** — recovering hidden input prompts from model outputs. Prompts, as data assets, fall into two categories: **user prompts** (potentially containing private information) and **system prompts** (encoding proprietary capabilities and business logic).

Existing LMI methods (e.g., Logit2text, Output2prompt) adopt a brute-force paradigm: collecting large numbers of output–prompt pairs and training an external inverse model to learn the $\mathcal{Y} \to \mathcal{X}$ mapping. This approach has two major limitations: (1) heavy reliance on large-scale inverse data, making collection costly; and (2) assumption of stable OOD generalization, which often fails in practice.

## Core Problem
**Can the rich latent space already learned by the LLM itself be reused for efficient inversion, rather than training an entirely new inverse model from scratch?** The LLM itself realizes the forward mapping $\mathcal{X} \to \mathcal{Z} \to \mathcal{Y}$; if the latent space $\mathcal{Z}$ already encodes inverse mapping information, inversion can be achieved more efficiently and with less data.

## Method

### Overall Architecture
Inv²A (Invariant Inverse Attacker) adopts an encoder–decoder architecture:
- **Input**: one or more outputs $Y = \{y_i\}_{i=1}^N$ generated from a hidden prompt
- **Inverse encoder** (trainable): encodes outputs into denoised pseudo-representations $\mathbf{c}$
- **Invariant decoder** (frozen): directly reuses the original LLM $f$ to decode $\mathbf{c}$ into the recovered prompt $\hat{x}$
- **Output**: recovered prompt $\hat{x} = f(\mathbf{c})$

The core idea is **asymmetric round-trip decoding**: rather than naively feeding outputs back into the LLM (which yields only 4.75 BLEU), the inverse encoder first maps outputs to a "clean anchor" $\mathbf{c}$, eliminating noise introduced by sampling stochasticity, before LLM decoding.

### Key Designs

1. **Invariant Latent Space Hypothesis (ILSH)**: Two key properties are proposed:

    - **Source Invariance**: Different outputs generated from the same prompt should maintain consistent semantic representations in the latent space.
    - **Cyclic Invariance**: The forward mapping $\mathcal{X} \to \mathcal{Z} \to \mathcal{Y}$ and the inverse mapping $\mathcal{Y} \to \mathcal{Z} \to \mathcal{X}$ should be self-consistent within a shared latent space.

   The authors validate ILSH experimentally: when outputs are perturbed, the entropy, conditional probability, and round-trip fidelity of the inverse mapping all deteriorate sharply; when the forward mapping is enhanced, inverse mapping metrics improve in tandem. This demonstrates that the LLM latent space already encodes inverse mapping information.

2. **Semi-Sparse Encoder**: In the system prompt scenario, multiple outputs are available. Naive concatenation of all outputs with global attention incurs complexity $O(N^2 l^2)$. The authors find that cross-output attention contributes negligible gain to inversion, and thus adopt a semi-sparse mechanism — encoding each $y_i$ independently and concatenating at the representation level: $\mathbf{h} = \text{Enc}(y_1) \oplus \cdots \oplus \text{Enc}(y_N)$, reducing complexity to $O(Nl^2)$.

3. **Dynamic Filter (optional)**: To address failure cases caused by output-side bias, a training-free post-processing module is designed. It prompts the LLM to rewrite outputs to expand the neighborhood space, and selects the variant that most accurately reconstructs the original output as the optimal input. An iterative Monte Carlo search is used to extend the search range; this module is triggered only for low-confidence samples (~15%), with negligible time overhead.

### Loss & Training
Training proceeds in two stages, corresponding to the two invariance properties of ILSH:

**Stage 1: Alignment** — enhancing Source Invariance
- For each source prompt $x$, a set of outputs $\mathcal{D}_x$ is sampled as positive examples.
- Source-aware contrastive learning is performed using an InfoNCE loss, pulling closer representations of outputs from the same source and pushing apart those from different sources.
- Only the Enc is trained (excluding the Proj layer); 4 epochs.

**Stage 2: Reinforcement** — enhancing Cyclic Invariance
- Supervised learning is performed on $(Y, x)$ pairs, minimizing the loss between recovered and ground-truth prompts.
- Two sub-stages: first, the Proj layer is warmed up on 20% of the data (Enc frozen); then Enc+Proj are jointly fine-tuned on the remaining 80%.
- 1 epoch, learning rate $2 \times 10^{-4}$.

## Key Experimental Results

| Scenario | Metric | Inv²A | Output2prompt | Few-shot (4o) | Gain (vs O2p) |
|----------|--------|-------|---------------|---------------|---------------|
| User prompt (avg. over 8 datasets) | BLEU | 41.78 | 35.34 | 26.75 | +6.44 |
| User prompt | Token F1 | 65.89 | 60.20 | 51.66 | +5.69 |
| User prompt | CS | 82.11 | 77.05 | 75.34 | +5.06 |
| User prompt | GPT | 74.46 | 59.46 | 65.39 | +15.00 |
| System prompt (Synthetic GPTs) | BLEU | 24.34 | 21.25 | 11.00 | +3.09 |
| System prompt | GPT | 94.20 | 79.20 | 72.80 | +15.00 |

### Ablation Study
- **The inverse encoder is critical**: removing the encoder (w/o Enc) and inverting directly via the decoder causes BLEU to plummet from 35.20 to 1.31 (prompting) or 26.47 (LoRA fine-tuning).
- **The original LLM decoder outperforms alternatives**: replacing the LLaMA2 decoder with Qwen2 (w/o Raw $f$) drops BLEU to 33.38, indicating that the original LLM better adapts to its own output distribution.
- **Contrastive learning is effective**: removing contrastive learning (w/o CL) reduces BLEU to 33.91 with increased variance.
- **Dynamic filter provides additional gains**: adding 1 round of search improves BLEU to 35.97; 2 rounds yield 36.06, with diminishing marginal returns.
- **High data efficiency**: Inv²A achieves Output2prompt's full-data performance with only 20–30% of the training data.
- **Fewer trainable parameters**: Inv²A trains only 113M parameters (T5 encoder + projection), far fewer than the baseline's 222M (full T5).

## Highlights & Insights
- **Novel theoretical insight**: The ILSH reveals the coupling between forward and inverse mappings in the LLM latent space, validated through both sufficiency and necessity experiments. This constitutes not merely an attack method but also a contribution to understanding the internal representational structure of LLMs.
- **Simple and efficient design**: Freezing the LLM as the decoder and training only a lightweight encoder leverages the LLM's powerful generative capacity while substantially reducing training cost.
- **Engineering elegance of semi-sparse encoding**: Reducing multi-output attention complexity from $O(N^2 l^2)$ to $O(Nl^2)$ with negligible performance loss is a transferable trick applicable to other multi-input fusion tasks.
- **In-depth defense analysis**: Beyond demonstrating attack strength, the paper systematically analyzes the limitations of existing defenses (diversity sampling, layer-wise noise injection), highlighting that current defenses remain insufficient.

## Limitations & Future Work
- **White-box assumption**: Full access to model parameters is required, limiting applicability in strict black-box scenarios (though open-source models and distributed inference settings satisfy this assumption).
- **Difficulty with semantically ambiguous prompts**: Inversion accuracy degrades when prompts are highly abstract or when multiple prompts map to identical outputs (e.g., both "3-1" and "1+1" produce "2").
- **Preliminary defense exploration**: Layer-wise noise injection is more effective than sampling diversification but degrades forward performance (~8% BLEU drop); viable defenses remain limited.
- **Instability on long prompts**: Output2prompt occasionally surpasses Inv²A when prompt length is approximately 120 tokens.
- **Potential extensions**: Future work may consider extending the ILSH framework to multimodal model inversion and robust inversion combined with differential privacy.

## Related Work & Insights

| Method | Mechanism | Key Difference from Inv²A |
|--------|-----------|--------------------------|
| **Output2prompt** | Trains a full T5 as an external inverse model | Inv²A reuses the LLM itself as the decoder, improving data efficiency by 5× and halving parameter count |
| **Logit2text** | Inverts from next-token probability distributions | Requires logit access; performance is far below text-based methods |
| **Jailbreak strings** | Designs adversarial queries to induce prompt leakage | Relies on special assumptions about prompt placement in the input window; poor generalization |
| **DORY** | Uncertainty-based denoising | Performs poorly on long, complex prompts; Inv²A achieves more systematic denoising via the encoder |

The core advantage of Inv²A lies in activating the LLM's existing inverse mapping capability rather than learning the inverse mapping from scratch, yielding significantly better data efficiency and generalization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The ILSH is novel and experimentally validated, though the encoder–decoder architecture itself is not a breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 9 datasets, multi-model transfer, ablations, robustness analysis, defense analysis, and interpretability analysis; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Logic is clear; the narrative from hypothesis → validation → method → experiments flows smoothly.
- **Value**: ⭐⭐⭐⭐ — Makes a substantive contribution to LLM privacy security, particularly the finding of insufficient defenses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](../../CVPR2026/llm_nlp/sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)

</div>

<!-- RELATED:END -->
