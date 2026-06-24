---
title: >-
  [Paper Note] Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings
description: >-
  [ICML2026][LLM Pretraining][Positional Encoding] The authors point out that the mainstream positional encoding, RoPE, couples "content (what)" and "position (where)" into the same phase, leading to poor performance on tasks requiring "finding content by position" or "locating position by content." They propose PoPE, which uses softplus to separate magnitude (controlling what) and pure positional phase (controlling where). As a minor modification to RoPE…
tags:
  - "ICML2026"
  - "LLM Pretraining"
  - "Positional Encoding"
  - "RoPE"
  - "Polar Coordinates"
  - "Length Extrapolation"
  - "Autoregressive Sequence Modeling"
date: 2026-05-08
content_hash: 0dda66f9bdb4a6b5
---

# Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings

**Conference**: ICML2026  
**arXiv**: [2509.10534](https://arxiv.org/abs/2509.10534)  
**Code**: https://github.com/agopal42/pope  
**Area**: LLM Pre-training / Positional Encoding  
**Keywords**: Positional Encoding, RoPE, Polar Coordinates, Length Extrapolation, Autoregressive Sequence Modeling

## TL;DR
The authors point out that the mainstream positional encoding, RoPE, couples "content (what)" and "position (where)" into the same phase, leading to poor performance on tasks requiring "finding content by position" or "locating position by content." They propose PoPE, which uses softplus to separate magnitude (controlling what) and pure positional phase (controlling where). As a minor modification to RoPE, PoPE consistently outperforms it in diagnostic tasks, music/genomic/language modeling, and achieves **length extrapolation to 10x the training length without any fine-tuning**, surpassing YaRN which is specifically designed for extrapolation.

## Background & Motivation

**Background**: Transformer self-attention is inherently position-insensitive (permutation invariant + translation equivariant under causal masking); thus, positional information must be injected. Currently, almost all frontier large models (Llama 3, DeepSeek-v3, Gemma 3, Qwen3) adopt RoPE: splitting the $d$-dimensional query/key vectors into $d/2$ two-dimensional components, where each component rotates by an angle $t\theta_c$ in a 2D plane based on position. Consequently, when the query at position $t$ and key at position $s$ perform a dot product, the rotation leaves only the relative position $(s-t)\theta_c$, naturally achieving "relative position awareness."

**Limitations of Prior Work**: Upon rewriting the RoPE attention score from Cartesian to polar coordinates, the authors identified a long-overlooked structural defect. Each 2D component $\bm{q}_{tc}$ can be written as a magnitude $\mu_{q_{tc}}$ and an initial phase $\phi_{q_{tc}}$, resulting in the attention score:

$$a_{ts}^{\text{RoPE}}=\sum_{c=1}^{d/2}\mu_{q_{tc}}\mu_{k_{sc}}\cos\big((s-t)\theta_c+\phi_{k_{sc}}-\phi_{q_{tc}}\big).$$

**Key Challenge**: The phase term $\phi_{k_{sc}}-\phi_{q_{tc}}$ in the above equation is the problematic part—this phase is determined by the **content** of the query/key, yet it is added to the cosine argument of the **relative position**. In other words, "matching what content" dynamically shifts "at which relative position the response is strongest." What and where are entangled: if the model intends to index purely by position or locate purely by content, this cross-term interferes.

**Goal**: While retaining all advantages of RoPE (relative position, translation equivariance, efficient implementation), **eliminate the what-where cross-term $\phi_{k_{sc}}-\phi_{q_{tc}}$**, allowing key-query matching to be expressed as a conjunction of "a what-match ∧ a where-match."

**Core Idea**: Use a different polar coordinate representation—let the magnitude encode content purely (using softplus to ensure non-negativity, interpretable as amplitude) and let the phase be determined purely by position ($t\theta_c$), so that only $(s-t)\theta_c$ remains in the cosine of the dot product, cleanly decoupling content and position.

## Method

### Overall Architecture
The objective of PoPE (Polar Coordinate Positional Embedding) can be summarized in one sentence: remove the phase cross-term in the RoPE attention score where "content pollutes position." This is achieved by reinterpreting the $d$-dimensional real-valued query/key as $d$-dimensional **complex vectors** $\tilde{\bm{q}}_t, \tilde{\bm{k}}_s \in \mathbb{C}^d$: the **magnitude** of each complex element comes from the original real-valued feature passed through softplus (non-negative, carrying what), and the **phase** of each complex element is given purely by the sequence position ($t\theta_c$, carrying where). The attention score is the real part of the inner product of the two complex vectors, which naturally becomes:

$$a_{ts}^{\text{PoPE}}=\sum_{c=1}^{d}\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}\cos\big((s-t)\theta_c\big),$$

where the cosine term contains only the relative position. There are two differences compared to RoPE: ① the index $c$ now iterates over **individual** elements rather than 2D pairs, doubling the number of frequencies from $d/2$ to $d$; ② the content-to-phase cross-term is completely removed. Additionally, a **learnable but fixed** phase bias $\delta_c$ is added for each frequency as a "benign substitute" for the RoPE cross-term. This path introduces no new modules but hardcodes the "magnitude/phase each handle one task" inductive bias into the attention, making it a pure mechanism improvement without a multi-stage pipeline.

### Key Designs

**1. Polar Decoupling: Magnitude for content, Phase for position**

This is the foundation of the paper. The issue with RoPE is that once 2D components are treated as complex numbers, content information leaks into the relative position cosine parameter through the initial phase $\phi$. PoPE changes this: the magnitude of complex element $c$ is determined only by the original real-valued feature $\mu_{\tilde k_{sc}}=\sigma(k_{sc})$ and $\mu_{\tilde q_{tc}}=\sigma(q_{tc})$, where $\sigma(x)=\ln(1+e^x)$ is the softplus function; the phase of the complex element is determined only by position $\phi_{\tilde k_{sc}}=s\theta_c$ and $\phi_{\tilde q_{tc}}=t\theta_c$. Softplus is not a random choice—it ensures non-negative magnitudes so that the "modulus of the complex number" can be legitimately interpreted as content intensity (whether a feature "exists"). Thus, the real part of the conjugate inner product $\Re[\tilde{\bm q}_t^H\tilde{\bm k}_s]$ yields a phase difference exactly equal to the pure position difference $(s-t)\theta_c$, while content only enters the magnitude product $\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}$. This is precisely the conjunction of "what-match × where-match": magnitude handles content alignment, while the cosine handles relative position alignment, and they no longer shift each other.

**2. Frequency Definition and Doubling: Rotation per element instead of per pair**

RoPE splits the vector into $d/2$ 2D pairs, with each pair sharing a frequency $\theta_c=\theta^{-2(c-1)/d}$. Since PoPE defines the phase for **each** complex element individually, the frequencies are changed to $\theta_c=\theta^{(c-1)/d}$ for $c=1,\dots,d$, doubling the number of frequencies. This is more than a notation difference: frequency usage analysis shows that RoPE maintains high norms only on a sparse few low-frequency channels while high-frequency channels are compressed into near-noise; conversely, PoPE utilizes features more uniformly across almost all layers and the entire frequency range because the frequencies are denser and content no longer pollutes position. This denser frequency spectrum is one physical basis for PoPE's stable length extrapolation.

**3. Learnable Phase Bias $\delta_c$: A benign substitute for the removed cross-term**

Directly removing $\phi_{k}-\phi_{q}$ loses some flexibility—sometimes the "optimal relative shift" should indeed not be 0. The authors thus add a scalar bias to each frequency:

$$a_{ts}^{\text{PoPE}}=\sum_{c=1}^{d}\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}\cos\big((s-t)\theta_c+\delta_c\big),$$

where $\delta_c \in \mathbb{R}$ is learnable but independent of content, purely tuning an optimal relative shift for each frequency. It is constrained within $[-2\pi,0]$ (improving stability in tests), with two good initialization options: $\delta_c=0$ or $\delta_c\sim\text{Uniform}(-2\pi,0)$. A key observation is that **zero initialization is more critical for length extrapolation**, while uniform initialization performs slightly better in-distribution. The fundamental difference from the RoPE cross-term is that $\delta_c$ is a "fixed per-frequency bias" that does not change dynamically with query/key content, thereby restoring flexibility without dragging what back into where.

**4. Complex Flash Attention Efficiency: Nearly zero extra overhead**

PoPE seemingly replaces real-valued attention with complex-valued attention, but the authors wrote a Triton kernel based on Flash Attention 2 to perform complex multiplication within the kernel, avoiding explicit materialization of the $d\times d$ complex query-key matrix. By splitting $\tilde q_{tc}=\mu(\cos\phi+i\sin\phi)$ into real and imaginary parts, the real part of the conjugate inner product is $\sum_c x_{\tilde q}x_{\tilde k}+y_{\tilde q}y_{\tilde k}$, which requires only **one extra multiplication** compared to standard attention. The cost is 2x memory/bandwidth to access complex key/values, but this can be eliminated by "loading only magnitudes and performing rotation within the kernel," reducing overhead to just one extra multiplication. For ease of prototyping different PoPE variants, the paper uses a slower but more general version.

### Loss & Training
The method does not change the training objective: experiments use a decoder-only Transformer with causal masking for autoregressive modeling (standard cross-entropy / next-token prediction). LayerNorm is replaced with RMSNorm to align with contemporary frontier models. Two sets of comparison models have identical architecture and hyperparameters, **only differing in positional encoding** (RoPE vs. PoPE). In length extrapolation experiments, PoPE+ft is fine-tuned for only 500 steps on longer sequences without any frequency interpolation.

## Key Experimental Results

### Main Results

The diagnostic task **Indirect Indexing** (given a source string, source character, and relative offset, output the target character, requiring independent manipulation of content and position + pointer arithmetic) best exposes the what-where coupling issue:

| Task | Positional Encoding | Accuracy (3 seeds) |
|------|----------|------------------|
| Indirect Indexing | RoPE | 11.16 ± 2.45 |
| Indirect Indexing | PoPE | **94.82 ± 2.91** |

RoPE fails to learn (11% ≈ random), while PoPE solves it almost perfectly (95%), directly validating the "decoupling" hypothesis.

Sequence modeling (NLL, lower is better) and language modeling perplexity (PPL, lower is better):

| Dataset / Scale | Metric | RoPE | PoPE |
|---------------|------|------|------|
| JSB (Music) | NLL | 0.5081 | **0.4889** |
| MAESTRO (Music) | NLL | 1.501 | **1.486** |
| Human Ref. Genome | NLL | 4.217 | **4.152** |
| OpenWebText 124M | PPL | 21.55 | **21.33** |
| OpenWebText 253M | PPL | 18.88 | **18.55** |
| OpenWebText 774M | PPL | 15.85 | **15.45** |

In language modeling, PoPE is consistently lower across 124M to 774M scales, with the performance gap remaining stable or slightly increasing with scale. For zero-shot downstream tasks (average accuracy across 6 benchmarks): 124M 45.33→46.19, 253M 48.76→48.78, 774M 51.80→**52.46**, winning in all three tiers.

### Ablation Study

Removing PoPE components on OpenWebText (PPL, lower is better):

| Configuration | 124M | 253M | Description |
|------|------|------|------|
| PoPE w/o $\sigma()$ (softplus) | 21.57 | 18.93 | Remove non-negative constraint; worse than RoPE |
| PoPE using ReLU instead of $\sigma()$ | 21.55 | 18.90 | Degrades to ≈ RoPE level |
| PoPE w/o $\bm{\delta}$ | 21.42 | 18.57 | Remove learnable phase bias; still better than RoPE |
| **Full PoPE** | **21.33** | **18.55** | Complete model |

### Key Findings
- **softplus is a key pillar**: Removing it (or replacing it with ReLU) causes perplexity to revert to or perform worse than RoPE, indicating that "magnitude must be non-negative and interpretable as content intensity" is a prerequisite for decoupling, not just a decoration.
- **Zero-shot length extrapolation is a highlight**: Trained on a 1024 context and tested on PG-19 up to 10240 (10x) tokens. RoPE performance collapses sharply with length; YaRN only works within its fine-tuned length (4096) and collapses beyond; **PoPE extrapolates 10x out-of-the-box without fine-tuning or frequency interpolation, surpassing YaRN**.
- **Extrapolation contrast with scale**: RoPE's extrapolation ability **worsens** as models grow larger, while PoPE remains stable. The authors attribute this to RoPE's what-where cross-term dynamically shifting position tuning, which becomes destructive on low-frequency components as the context window expands.
- **More uniform frequency usage**: RoPE maintains high norms only in sparse low-frequency channels (suppressing high-frequency noise), whereas PoPE maintains high norms across almost all layers and the entire frequency range (except the first layer), using features more distributively—consistent with frequency doubling and decoupling.

## Highlights & Insights
- **Revealing the issue by changing the coordinate system**: The authors did not invent a completely new mechanism but rewrote RoPE from Cartesian to polar coordinates, immediately exposing the content-polluting-position cross-term $\phi_k-\phi_q$. "Clarifying structural defects through representation change" is a brilliant analysis.
- **Minimal change + strong inductive bias**: PoPE is a minor tweak of RoPE (softplus for magnitude, pure position for phase, frequency doubling, adding $\delta_c$), yet it introduces the "what ∧ where conjunction match" prior, improving data efficiency, asymptotic accuracy, and length generalization.
- **Convincingly designed diagnostic task**: Indirect Indexing pushes "retrieving content by position / locating position by content" to the limit. The 11% vs 95% contrast makes the "decoupling" hypothesis undeniable and serves as a probe to evaluate what-where entanglement in any positional encoding.
- **Deployable**: Complex Flash Attention requires only one extra multiplication and can eliminate memory overhead, meaning PoPE is not just a theoretical method but a viable replacement for RoPE in frontier LLMs.

## Limitations & Future Work
- **Small gains in the language domain**: The authors admit that music/genome were chosen because they explicitly require separation of position and content + precise positioning; it is unclear if human language truly possesses these properties. The PPL improvement on OpenWebText is only 0.2~0.4, far less dramatic than in diagnostic tasks.
- **Memory trade-off remains an issue**: The general implementation requires 2x memory/bandwidth for complex key/values. The paper did not enable the memory-saving "in-kernel rotation" version for prototyping convenience, which would be necessary for large-scale deployment.
- **$\delta_c$ initialization is a trade-off**: Zero initialization favors extrapolation, while uniform initialization favors in-distribution performance. The inability to achieve both suggests a unified optimum hasn't been found, leaving a hyperparameter tuning burden.
- **Relatively single baseline**: The main tables primarily compare against RoPE/YaRN. The text acknowledges that broader comparisons with sinusoidal / learnable / ALiBi / other RoPE variants are in the appendix, focusing primary evidence on pairwise comparisons with RoPE.

## Related Work & Insights
- **vs RoPE**: Both use rotation to encode relative position and both are translation equivariant. However, RoPE allows content to shift position tuning via the initial phase (what-where coupling). PoPE decouples them using softplus magnitude + pure positional phase and doubles the frequency count; the cost is a complex representation, which is mitigated by Flash Attention.
- **vs YaRN**: YaRN is an extrapolation patch for RoPE, relying on frequency interpolation for low frequencies + fine-tuning on long sequences, but it still fails beyond the fine-tuned length. PoPE extrapolates 10x without interpolation or fine-tuning, solving the problem at the root (removing cross-terms) rather than patching it.
- **vs Length Extrapolation Series (Chen 2023 / ALiBi / Sun 2023 / Wang 2024)**: These methods either change rotation frequencies, add ALiBi-style decay and block masks, or round wavelengths to eliminate wrap-around shifts—mostly repairs within the RoPE framework. PoPE provides a more fundamental perspective by attributing extrapolation collapse to the what-where cross-term.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting in polar coordinates reveals RoPE's structural defects and eliminates them with minimal changes; perspective is fresh and explanation is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Diagnostics, music, genome, language, downstream, extrapolation, and frequency analysis are comprehensive; however, language gains are small, and broader baselines are in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless derivation from RoPE to PoPE; formulas and motivation are tightly coupled.
- Value: ⭐⭐⭐⭐⭐ If replication is robust, it offers a near-zero cost replacement for RoPE with significant improvements to length extrapolation, which is highly attractive for LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](../../ICLR2026/llm_pretraining/deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[ICLR 2026\] What Scales in Cross-Entropy Scaling Law?](../../ICLR2026/llm_pretraining/what_scales_in_cross-entropy_scaling_law.md)
- [\[ICLR 2026\] FoNE: Precise Single-Token Number Embeddings via Fourier Features](../../ICLR2026/llm_pretraining/fone_precise_single-token_number_embeddings_via_fourier_features.md)
- [\[ICLR 2026\] Not All Documents Are What You Need for Extracting Instruction Tuning Data](../../ICLR2026/llm_pretraining/not_all_documents_are_what_you_need_for_extracting_instruction_tuning_data.md)
- [\[CVPR 2025\] The Scene Language: Representing Scenes with Programs, Words, and Embeddings](../../CVPR2025/llm_pretraining/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)

</div>

<!-- RELATED:END -->
