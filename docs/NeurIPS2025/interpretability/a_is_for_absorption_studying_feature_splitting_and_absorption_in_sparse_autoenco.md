---
title: >-
  [Paper Note] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders
description: >-
  [NeurIPS 2025][Sparse Autoencoder] This paper identifies and systematically studies the phenomenon of "feature absorption" in SAEs: apparently monosemantic SAE latents fail to activate on certain tokens because their feature directions are "absorbed" by more specific sub-latents. This is shown to be an inevitable consequence of hierarchical features combined with sparsity loss, posing a fundamental challenge to using SAEs for reliable LLM interpretation.
tags:
  - NeurIPS 2025
  - Sparse Autoencoder
  - Feature Absorption
  - Feature Splitting
  - Mechanistic Interpretability
  - LLM Internal Representations
date: 2026-05-08
content_hash: 9f96f1b0fad5fe7f
---

# A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders

**Conference**: NeurIPS 2025
**arXiv**: [2409.14507](https://arxiv.org/abs/2409.14507)
**Code**: [https://github.com/lasr-spelling/sae-spelling](https://github.com/lasr-spelling/sae-spelling)
**Area**: AI Safety / Interpretability / Mechanistic Interpretability
**Keywords**: Sparse Autoencoder, Feature Absorption, Feature Splitting, Mechanistic Interpretability, LLM Internal Representations

## TL;DR
This paper identifies and systematically studies the phenomenon of "feature absorption" in SAEs: apparently monosemantic SAE latents fail to activate on certain tokens because their feature directions are "absorbed" by more specific sub-latents. This is shown to be an inevitable consequence of hierarchical features combined with sparsity loss, posing a fundamental challenge to using SAEs for reliable LLM interpretation.

## Background & Motivation
**Background**: Sparse Autoencoders (SAEs) are a central tool in mechanistic interpretability, designed to decompose dense LLM activations into interpretable sparse feature directions.

**Limitations of Prior Work**:
- SAE latent precision/recall is unstable; some latents that appear to track a concept fail to activate when they should.
- Examining only max-activating examples can create a false illusion of interpretability.
- Prior work on feature splitting focused only on "benign" splits (e.g., capitalization splits), overlooking problematic ones.

**Key Challenge**: SAE sparsity loss encourages minimizing the number of simultaneously active latents. However, when features form hierarchical relationships (e.g., "starts with S" is a parent feature of "short"), SAEs absorb the parent feature direction into child latents to improve sparsity, causing the parent latent to fail to activate on those tokens.

**Goal**: To identify, define, quantify, and explain feature absorption in SAEs.

**Key Insight**: The study uses a spelling task (first-letter prediction) as a test bed, employing linear probes as ground truth to systematically compare SAE latent precision and recall.

**Core Idea**: Sparsity optimization over hierarchical features inevitably leads to feature absorption, rendering SAE latents unreliable classifiers.

## Method

### Overall Architecture
The test task is predicting the first letter of a token. Inputs are LLM residual stream activations. The pipeline is: (1) train a linear probe as a ground-truth classifier; (2) identify the SAE latent with the highest cosine similarity to the probe direction as the "first-letter latent"; (3) compare precision and recall between the two; (4) verify the causal effect of absorption via ablation experiments. The output is a quantitative metric for feature absorption rate.

### Key Designs

1. **Toy Model Proving the Inevitability of Absorption**:

   - **Function**: In a simple 4-feature setting where feature 1 only activates when feature 0 is active (hierarchical relationship), the learned latent 0 fails to activate when feature 1 is active, while latent 1's decoder absorbs the direction of feature 0.
   - **Mechanism**: With independent features, the SAE recovers perfectly. With hierarchical co-occurrence, the SAE encoder learns "¬feat1 ∧ feat0" rather than "feat0", since this requires activating only 1 latent instead of 2, yielding better sparsity.
   - **Design Motivation**: To provide an analytical proof that absorption genuinely reduces SAE loss in hierarchical settings.

2. **First-Letter Spelling Experiment Design**:

   - **Function**: ICL prompts are used to elicit first-letter predictions from the model; residual stream activations are then extracted.
   - **Mechanism**: Linear probes are trained for all 26 first letters and compared against individual SAE latents. $k$-sparse probing detects feature splitting; ablation experiments detect absorption.
   - **Design Motivation**: First-letter identity is a well-defined binary feature with unambiguous ground truth.

3. **Feature Absorption Rate Metric**:

   - **Function**: To quantify how frequently absorption occurs.
   - **Mechanism**: False-negative tokens — where all $k$ feature-split latents are inactive yet the probe correctly classifies — are identified. Integrated-gradients ablations are applied to these tokens. If the ablation effect of a non-first-letter latent is largest and its cosine similarity with the probe exceeds 0.025, the case is classified as absorption. $\text{absorption\_rate} = \text{num\_absorptions} / \text{probe\_true\_positives}$.
   - **Design Motivation**: To provide a conservative lower-bound estimate, counting only causally confirmed absorption cases.

### Loss & Training
- SAEs use standard $L_1$ loss or JumpReLU/TopK architectures.
- Linear probes use $L_1$-regularized logistic regression.
- $k$-sparse probing first selects top-$k$ latents by $L_1$ weights, then trains a standard probe.

## Key Experimental Results

### Main Results
Experiments are conducted on Gemma-2-2B using Gemma Scope SAEs (16k and 65k width), as well as self-trained SAEs on Qwen2 0.5B and Llama 3.2 1B.

| Configuration | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Linear Probe | ~0.98 | ~0.98 | ~0.98 | Ground truth baseline |
| SAE latent (best) | ~0.95 | ~0.70 | ~0.81 | Best single SAE latent |
| $k$=5 sparse probe | ~0.95 | ~0.90 | ~0.92 | Improvement from multi-latent combination |

Feature absorption is present in all tested SAEs; none match the performance of the linear probe.

### Ablation Study

| L0 Range | Absorption Rate | Notes |
|---|---|---|
| Low L0 (~25) | ~15–20% | High precision, low recall; severe absorption |
| Mid L0 (~50–100) | ~10–15% | Best F1 range |
| High L0 (~200) | ~5–8% | Low precision, high recall |

### Key Findings
- **Absorption is present in all tested SAEs**: including Gemma Scope, self-trained $L_1$ SAEs, and TopK SAEs, across Gemma, Qwen, and Llama.
- **Wider, sparser SAEs exhibit more absorption**: the 65k SAE shows higher absorption rates than the 16k SAE; lower L0 leads to more absorption.
- **Hyperparameter tuning cannot fundamentally resolve the issue**: varying SAE width or sparsity only partially mitigates absorption.
- **Case study**: The "starts with S" latent 6510 fails to activate on the token "short"; instead, token-aligned latent 1085 (encoding the meaning of "short") activates, and its decoder contains a small directional component corresponding to "starts with S."

## Highlights & Insights
- **Systematic exposure of interpretability illusions**: Apparently monosemantic latents harbor hidden false negatives; relying solely on max-activating examples is misleading. This is an important warning for the entire SAE-based interpretability paradigm.
- **Complete chain from toy model to large-scale validation**: Absorption is first proven analytically in a toy model, then validated across hundreds of real SAEs, yielding a methodologically rigorous argument.
- **Causal verification**: Beyond identifying absorbing latents, ablation experiments confirm that the absorbed probe-direction component causally mediates model behavior.
- **Absorption rate metric**: A reusable quantitative metric is introduced for evaluating future SAE improvements.

## Limitations & Future Work
- The absorption rate metric relies on ablation experiments and can only be applied to early layers (< layer 17) before attention has moved information; detection in later layers is infeasible.
- The metric is a conservative lower bound and does not capture cases where multiple latents jointly absorb a feature or where the primary latent activates weakly.
- Validation is limited to the first-letter spelling task; other types of hierarchical features (e.g., semantic hierarchies) require further study.
- No solution is proposed; the work is diagnostic. Potential directions include Meta-SAEs, group lasso, and hierarchical sparse coding.

## Related Work & Insights
- **vs. Anthropic SAE work [Bricken et al.]**: They first described feature splitting and treated it as a positive phenomenon; this paper identifies its pathological form — absorption.
- **vs. Meta-SAE [Bussman et al.]**: Meta-SAEs re-train an SAE on top of SAE decoders to decompose latents into sub-components, a potentially promising direction for addressing absorption.
- **vs. $k$-sparse probing [Gurnee et al.]**: This paper reuses their method to detect feature splitting, but is the first to apply it to detect absorption.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First systematic definition and study of feature absorption; raises fundamental questions about the SAE paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Hundreds of SAEs, three model families, toy model plus real-world validation, complete causal analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, compelling case studies, excellent figures.
- Value: ⭐⭐⭐⭐⭐ — Significant impact for AI Safety and mechanistic interpretability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] Distributional Autoencoders Know the Score](distributional_autoencoders_know_the_score.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)

<!-- RELATED:END -->
