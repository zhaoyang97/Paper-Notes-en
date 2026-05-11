---
title: >-
  [Paper Note] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?
description: >-
  [NeurIPS 2025][LLM Pretraining][sequence modeling] By introducing a "domain-restricted pre-training + OOD testing" evaluation framework…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "sequence modeling"
  - "base capabilities"
  - "Transformer"
  - "Mamba"
  - "RWKV"
  - "attention"
  - "Top-1 selection"
date: 2026-05-08
content_hash: 23329f75f12e948d
---

# How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?

**Conference**: NeurIPS 2025
**arXiv**: [2505.18522](https://arxiv.org/abs/2505.18522)
**Code**: [GitHub](https://github.com/luxinxyz/TSA)
**Area**: LLM Pre-training
**Keywords**: sequence modeling, base capabilities, Transformer, Mamba, RWKV, attention, Top-1 selection

## TL;DR
By introducing a "domain-restricted pre-training + OOD testing" evaluation framework, this paper reveals that stateful architectures such as Mamba and RWKV suffer from degraded base capabilities. It identifies the key design principle of "arbitrary selection over the full sequence" (full-sequence visibility + real relation calculation + non-uniform distribution), and validates this principle using a minimalist Top-1 Element/Chunk Selection architecture that recovers base capabilities to near-Transformer levels.

## Background & Motivation

**Background**: Stateful sequence modeling architectures such as Mamba, RWKV, and Gated DeltaNet replace the self-attention mechanism of Transformers with linear-complexity alternatives, achieving comparable or superior performance on language modeling and few-shot learning benchmarks while offering greater computational efficiency.

**Limitations of Prior Work**: Prior studies have identified deficiencies in these architectures on targeted tasks such as retrieval, copying, and associative recall, but their impact on "base capabilities" (OOD language modeling generalization) remains unclear—because the commonly used mixed-domain pre-training setting makes all architectures appear similarly capable.

**Key Challenge**: Mixed-domain pre-training is fundamentally an in-distribution evaluation, which fails to expose architectural differences. This creates the illusion that Mamba and Transformer have comparable base capabilities, whereas significant gaps may emerge under OOD conditions.

**Goal**: (1) Design an evaluation methodology capable of revealing architectural differences in base capabilities; (2) identify the key factors responsible for base capability degradation in stateful architectures; (3) propose architectural design principles that prevent such degradation.

**Key Insight**: Domain-restricted pre-training (training exclusively on cc+c4) combined with cross-domain testing (evaluating OOD performance on arxiv/github/stack) exposes architectural differences even at early stages of training.

**Core Idea**: A sequence modeling architecture must possess "arbitrary selection over the full sequence"—the ability to attend to the entire sequence, compute genuine relational scores, and produce non-uniform distributions—in order to maintain base capabilities without degradation.

## Method

### Overall Architecture
The paper proceeds in three stages: (1) propose the domain-restricted pre-training + OOD testing framework to reveal differences in base capabilities; (2) systematically ablate the Mamba family of architectures and general sequence modeling factors to identify critical components; (3) distill the critical factors into design principles and validate them using the minimalist Top-1 Selection architecture.

### Key Designs

1. **Domain-Restricted Pre-training Evaluation Framework**:

    - **Function**: Train on restricted-domain data and evaluate OOD generalization on unseen domains.
    - **Mechanism**: Training uses only the cc+c4 subsets of SlimPajama; evaluation is conducted on arxiv/github/stack. A scatter plot of "training loss vs. OOD test loss" is used to quantify differences in base capabilities across architectures at equivalent training loss levels.
    - **Design Motivation**: Mixed-domain training renders the test set in-distribution, masking architectural differences; domain-restricted training makes the test set OOD, thereby exposing the intrinsic generalization capacity of each architecture.

2. **Architectural Factor Analysis (Non-Critical Factors)**:

    - **Function**: Ablate Mamba's data-dependent decay, convolution, GroupNorm, and positional encoding.
    - **Key Finding**: These factors only affect convergence speed and have no impact on base capabilities. Removing data-dependent decay and convolution does not degrade—and may marginally improve—OOD performance.
    - **Design Motivation**: Eliminate confounding factors to isolate the truly critical architectural components.

3. **Architectural Factor Analysis (Critical Factors)**:

    - **Full-Sequence Visibility**: Larger sliding window sizes yield better base capabilities; a window of 256 causes significant degradation.
    - **Real Relation Calculation**: Replacing keys with random constants (eliminating genuine QK relation computation) leads to severe degradation of base capabilities.
    - **Non-Uniform Distribution**: Sharpening the attention distribution via lower softmax temperature (more non-uniform) improves base capabilities.

4. **Top-1 Element/Chunk Selection Architecture (Validation)**:

    - **Function**: Design a minimalist architecture satisfying all three criteria to validate the proposed principles.
    - **Mechanism**: Top-1 Element Selection directly selects the element with the highest probability in the attention distribution as the output (trained via the straight-through estimator). Top-1 Chunk Selection is its practical variant—the sequence is divided into chunks, and the top-1 element is selected within each chunk.
    - **Design Motivation**: If such an extremely simplified architecture (retaining only top-1 selection) can still achieve Transformer-level base capabilities, this provides strong evidence that "arbitrary selection over the full sequence" is the decisive factor.

### Loss & Training
- Two scales: 110M and 1.3B parameters, trained on 100B tokens.
- Sequence lengths of 2K (short) and 100K (long).
- Chunk size = 128.

## Key Experimental Results

### OOD Language Modeling (110M, 100B tokens)

| Architecture | Mixed Domain Test | OOD Test | Notes |
|---|---|---|---|
| Transformer++ | Best | Best | Baseline |
| Mamba-1 | ≈Transformer++ | Significant degradation | Mixed-domain masks the gap |
| Mamba-2 | ≈Transformer++ | Significant degradation | Same as above |
| RWKV-6/7 | ≈Transformer++ | Moderate degradation | Same as above |
| Top-1 Element Selection | Slightly worse | ≈Transformer++ | Validates principle |
| Top-1 Chunk Selection | ≈Transformer++ | ≈Transformer++ | Practical variant |

### Ablation: Impact of Architectural Factors

| Factor | Impact on Base Capabilities |
|---|---|
| Data-dependent decay | None (only accelerates convergence) |
| Convolution | None (only accelerates convergence) |
| Positional encoding | None (except ALiBi) |
| Full-sequence visibility | **Critical** — larger window yields better capabilities |
| Real relation calculation | **Critical** — removal causes severe degradation |
| Non-uniform distribution | **Critical** — sharper distribution yields better capabilities |

### Key Findings
- Mixed-domain pre-training combined with standard test perplexity or few-shot evaluation fails to differentiate architectural differences—this represents a fundamental limitation of existing benchmarks.
- The components responsible for Mamba's efficiency gains (data-dependent decay, convolution) provide no positive contribution to base capabilities.
- Top-1 Element Selection (extreme non-uniformity + full-sequence visibility + real relation computation) achieves Transformer-level OOD generalization with a minimalist design.

## Highlights & Insights
- **Methodological Contribution**: Domain-restricted pre-training + OOD testing is a simple yet effective tool for architectural analysis; future evaluations of novel architectures should incorporate this dimension.
- **Counter-Intuitive Finding**: Mamba's signature components (data-dependent decay, convolution) do not contribute to base capabilities—what truly matters are the three fundamental properties of the attention mechanism.
- **Minimalist Validation**: The fact that Top-1 selection (arguably the simplest instantiation of "arbitrary selection over the full sequence") suffices to recover base capabilities suggests that this principle functions as a necessary rather than merely sufficient condition.

## Limitations & Future Work
- **Scope Limited to Language Modeling**: Whether other base capabilities (e.g., reasoning, code generation) follow the same principles remains unverified.
- **Top-1 Selection is Less Efficient than Mamba**: Although Top-1 Chunk Selection is faster than full attention, it remains less efficient than Mamba.
- **Hybrid Architectures Not Explored**: Whether hybrid attention-SSM architectures (e.g., Jamba) can capture the advantages of both paradigms is not discussed.
- **Future Directions**: (1) Explore efficiency optimizations under the constraint of "arbitrary full-sequence selection" (e.g., sparse selection combined with linear recurrence); (2) extend validation of this principle to larger scales (7B+).

## Related Work & Insights
- **vs. Mamba/RWKV**: These architectures pursue efficiency at the cost of arbitrary full-sequence selection capability; this paper demonstrates that this trade-off is the root cause of base capability degradation.
- **vs. Linear Attention (Based)**: Linear attention also lacks non-uniform distributions and real relation computation, leading to analogous degradation in base capabilities.
- **vs. Gated DeltaNet**: The delta rule attempts to enhance expressiveness but remains constrained by the stateful framework and does not fully recover base capabilities.
- **Implications**: This work provides clear theoretical guidance for the efficiency–capability trade-off: future architecture designs should treat arbitrary full-sequence selection as a baseline requirement, with efficiency optimizations pursued on top of this foundation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic exposure of base capability degradation in stateful architectures, with actionable design principles distilled therefrom.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 architectures, two scales (110M–1.3B), systematic ablations, and Top-1 validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logical chain (problem identification → factor analysis → principle formulation → validation), with clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ Offers foundational guidance for sequence modeling architecture design; the evaluation methodology also carries independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](../../ACL2026/llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)

</div>

<!-- RELATED:END -->
