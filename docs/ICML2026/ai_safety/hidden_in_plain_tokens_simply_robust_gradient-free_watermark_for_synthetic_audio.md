---
title: >-
  [Paper Note] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio
description: >-
  [ICML 2026][AI Safety][Autoregressive Audio Watermarking] Addressing the issue where KGW-style token watermarks in autoregressive audio models suffer exponential decay due to "decoding $\to$ re-encoding non-idempotency…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Autoregressive Audio Watermarking"
  - "KGW"
  - "Re-encoding Robustness"
  - "Vocabulary Community Detection"
  - "Gradient-Free"
date: 2026-05-08
content_hash: aaba30ccbb56290f
---

# Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio

**Conference**: ICML 2026  
**arXiv**: [2605.25967](https://arxiv.org/abs/2605.25967)  
**Code**: https://g-milis.github.io/projects/nograd-audio-wm.html (Project Page + Partial Code)  
**Area**: AI Safety / Content Attribution / Audio Watermarking  
**Keywords**: Autoregressive Audio Watermarking, KGW, Re-encoding Robustness, Vocabulary Community Detection, Gradient-Free

## TL;DR
Addressing the issue where KGW-style token watermarks in autoregressive audio models suffer exponential decay due to "decoding $\to$ re-encoding non-idempotency," the authors apply Leiden community detection to the codec's confusion matrix to derive a contracted "cluster vocabulary." By defining green/red sets on clusters rather than tokens, the method raises the exponential base of the $z$-score from $r$ to $r_{cl} > r$ under fully gradient-free, black-box access. Detectability is improved by several orders of magnitude compared to baselines and WMAR (which requires fine-tuning), and it is naturally robust against perturbations such as MP3 compression, denoising, and cropping.

## Background & Motivation

**Background**: The mainstream watermark for LLM text generation is KGW (Kirchenbauer et al.), which pseudo-randomly partitions the vocabulary into green/red sets and adds a $\delta$ logit bias to green tokens. Detection only requires binomial testing of the green token proportion. This "injection during sampling, detection during statistics" paradigm is training-free and incurs almost zero cost for autoregressive models. Consequently, efforts have been made to adapt it to autoregressive audio models (e.g., Moshi/Mimi, MusicGen/EnCodec) to solve the attribution problem of synthetic audio.

**Limitations of Prior Work**: Directly applying KGW to audio fails immediately—audio codec encoder-decoders are not idempotent. Re-encoding the decoded waveform $x_{1:N}$ yields $y_{1:N}$, which is inconsistent with the original tokens; the token match rate $r$ is much less than 1 (near 1 for text, but roughly 0.4 for audio/images). Since the detector uses $y$ for statistics, the green token count is severely discounted, and the signal decays faster as the hash context length $h$ increases. Existing solutions (WMAR, Jovanović et al. for images) choose to fine-tune the codec for idempotency, but they sacrifice the training-free and black-box nature, requiring expensive training and white-box access. Distortion-free schemes based on k-means (Wu et al.) sacrifice detectability and perform worse than the KGW baseline.

**Key Challenge**: Watermark detectability highly depends on the token-level match rate $r$, whereas continuous-modality codecs inherently do not satisfy $r \approx 1$. Preserving token-level matching requires modifying the codec, while maintaining "training-free" status forces one to accept exponential signal decay.

**Goal**: To elevate the actual detection signal of audio KGW to reach or even exceed that of WMAR (which requires fine-tuning), without modifying codec parameters and using only black-box encoder/decoder queries.

**Key Insight**: The authors observe that re-encoding errors are not uniformly random but structured—a token is typically confused only with a small set of "semantic neighbors." By grouping these neighbors into a cluster and establishing green/red rules on "whether the cluster falls in the green set" rather than the token, the watermark hits as long as the re-encoded token remains within the same cluster. This raises the exponential base from the token match rate $r$ to the cluster match rate $r_{cl} > r$.

**Core Idea**: Construct a token graph using confusion counts from the codec on a dataset, and use community detection to obtain a "semantic cluster vocabulary." The partition and context hashing of KGW are then performed at the cluster level. This can be implemented via a lookup table and is fully compatible with existing autoregressive inference.

## Method

### Overall Architecture
The method consists of two decoupled stages. **Off-line Distillation Stage**: Select an audio dataset (labels not required), perform $E \to D \to E$ encoding passes, and record which tokens $j$ replace the original token $i$. Accumulate these into a confusion matrix $M \in \mathbb{N}^{|V| \times |V|}$. Use $M$ as the adjacency matrix for a weighted directed graph and run Leiden community detection to obtain a many-to-one mapping $\mathcal{C}$ from token $\to$ cluster. This is done independently for each RVQ channel, allowing multi-scale watermarking via different resolution parameters $\rho$. **On-line Watermarking Stage**: Move the two KGW steps—pseudo-random green/red partitioning and adding $\delta$ logit bias—to the cluster level. Partitions are applied to the set of clusters, and the bias is added simultaneously to all tokens belonging to green clusters. The hash context uses cluster indices of the previous $h$ tokens instead of token indices, ensuring stability against re-encoding. During detection, the received waveform is re-encoded into tokens, mapped to clusters, and the $z$-score is calculated based on cluster occurrences in the green set.

Theoretical analysis provides the core quantitative result: under a noisy channel approximation with conditional independence, the expected $z$-score of the KGW baseline for an $h$-gram context is $\mathbb{E}[z|H_1] = \sqrt{N} \frac{g-\gamma}{\sqrt{\gamma(1-\gamma)}} r^{h+1}$. Ours replaces $r$ with $r_{cl}$, resulting in $\mathbb{E}[z|H_1] = \sqrt{N} \frac{g-\gamma}{\sqrt{\gamma(1-\gamma)}} r_{cl}^{h+1}$. Since $r_{cl} > r$ and it appears to the power of $h+1$, the gain is exponentially amplified. This is the algorithmic basis for the title "hidden in plain tokens"—there is no need to modify the codec; one simply acknowledges the redundant structure hidden in the vocabulary.

### Key Designs

1.  **Community Detection-based Vocabulary Distillation from Confusion Matrix**:
    - **Function**: Compresses a token vocabulary of size $|V|$ into a cluster vocabulary of size $c|V|$ ($c \in (0, 1)$), such that the probability $r_{cl}$ of a re-encoded token staying in the same cluster is much higher than the probability $r$ of it staying the same token.
    - **Mechanism**: Treats $M_{ij}$ (count of $i$ being confused with $j$) as a directed weighted graph. The Leiden algorithm maximizes modularity—high intra-cluster weights and low inter-cluster weights correspond exactly to errors occurring mainly within clusters. The resolution parameter $\rho$ controls cluster granularity. Leiden is chosen over Louvain because it handles edge direction, ensures cluster connectivity, and converges faster while being more favorable for detectability.
    - **Design Motivation**: While k-means or semantic embedding clustering are common, they do not encode behavior-specific information about "what a codec actually confuses." Using the codec’s own confusion counts as the adjacency matrix aligns the clustering objective with the detection objective, which is key to ensuring a "gradient-free + black-box" approach.

2.  **Cluster-level KGW: Synchronized Elevation of Partitions and Hash Context**:
    - **Function**: Modifies the two positions where KGW depends on token identity—green/red partitioning and $h$-gram hash keys—to depend on cluster identity, making the pipeline invariant to re-encoding.
    - **Mechanism**: At each timestep, calculate the hash using the cluster indices of the previous $h$ tokens to define the green cluster set $G_i$. Apply bit $\delta$ to all tokens in $c \in G_i$. The detector maps tokens to clusters and performs the binomial test. Being confused with a neighbor in the same cluster is indistinguishable from perfect restoration to the detector. This requires only a single cluster lookup during inference, adding zero overhead.
    - **Design Motivation**: This is the minimal intrusive modification to bring the off-line structure into on-line KGW—preserving statistical rigor while lifting the exponential decay base.

3.  **Explicit Entropy-Key Space Trade-off and Multi-channel Resolution**:
    - **Function**: Characterizes the cost of losing generation entropy and potential hash collisions when compressing the vocabulary, and mitigates this through channel-wise resolution.
    - **Mechanism**: Compressing from $|V|$ to $c|V|$ reduces the $h$-gram key space from $|V|^h$ to $(c|V|)^h$. The authors require $(c|V|)^h \geq K_{\min}$ to avoid collisions, with unwatermarked sampling deferral as a fallback. Since RVQ channels are clustered independently, some channels can use fine clusters (preserving entropy) while others use coarse clusters (ensuring robustness).
    - **Design Motivation**: While extremely small cluster sets maximize $r_{cl}$, they collapse the key space, making watermarks forgeable. Characterizing this trade-off provides the engineering "knobs" necessary for deployment on multi-channel RVQ models like Moshi or MusicGen.

### Loss & Training
The method is entirely training-free. Off-line, it runs community detection once to compute $\mathcal{C}$. On-line, it adds a constant bias $\delta$ to logits without gradients. Hyperparameters include KGW's $\gamma$, $\delta$, $h$, and Leiden's per-channel resolution $\rho$.

## Key Experimental Results

### Main Results
The authors sampled 500 audio clips each from Moshi (Mimi codec, speech) and MusicGen (EnCodec, music), comparing Base (Vanilla KGW), WMAR, WMAR (aug) (two fine-tuning schemes), and Ours. Quality was measured via FAD (VGGish/CLAP) and MOS (NISQA/DNSMOS).

| Dataset | $h$ | Metric | None | Base | WMAR | WMAR(aug) | Ours |
|---|---|---|---|---|---|---|---|
| Moshi/Dialogue | 0 | FAD-VGGish ↓ | 0.080 | 0.128 | 0.407 | 0.267 | **0.133** |
| Moshi/Dialogue | 1 | FAD-VGGish ↓ | 0.080 | 0.068 | 0.357 | 0.218 | **0.051** |
| Moshi/LibriSpeech | 0 | FAD-VGGish ↓ | 1.921 | 1.858 | 2.195 | 2.153 | **1.670** |
| Moshi/Dialogue | 1 | NISQA MOS ↑ | 3.54 | 3.56 | 3.37 | 3.54 | **3.58** |
| Moshi/LibriSpeech | 1 | NISQA MOS ↑ | 3.15 | 3.23 | 3.12 | 3.19 | 3.22 |

In terms of quality, Ours is comparable to Base and sometimes better than "None" (unwatermarked), suggesting cluster-level bias does not harm codec decodability. WMAR variants show significant FAD degradation due to modified codec weights.

### Robustness / Detection Strength (Moshi)

| Attack Type | Transformation | Base $-\log p$ | WMAR | WMAR(aug) | **Ours** |
|---|---|---|---|---|---|
| Baseline | Identity | 8.51 | 17.44 | 13.72 | **42.47** |
| Signal | Lowpass | 5.82 | 9.23 | 10.52 | **41.51** |
| Signal | Smooth | 1.99 | 1.61 | 3.73 | **32.68** |
| Signal | Noise | 2.23 | 0.61 | 8.01 | **20.59** |
| Compression | MP3 | 7.47 | 15.31 | 12.66 | **41.26** |
| Compression | EnCodec | 2.59 | 2.82 | 2.78 | **32.64** |
| Time-domain | Crop | 1.51 | 1.27 | 1.51 | **16.48** |
| Time-domain | Speedup | 1.52 | 1.20 | 1.35 | **26.49** |

Without attacks, Ours achieves a $-\log p$ roughly $3 \times$ that of WMAR(aug) and $5 \times$ that of Base. Under all 12 attacks, it outperforms the runner-up by several-fold to an order of magnitude, with particularly large gaps in time-domain attacks like cropping and speedup. In Figure 3-5, at very low FPR ($10^{-6}$), Ours is the only scheme maintaining high TPR.

### Key Findings
- The improvement stems entirely from replacing $r$ with $r_{cl}$: the gap between Ours and Base widens as $h$ increases (e.g., $h=2$), as the $r^{h+1}$ vs $r_{cl}^{h+1}$ difference is exponentially amplified, consistent with theory.
- Multi-channel differentiated resolution is essential: token-level KGW barely works on RVQ channels, while multi-scale clusters allow different channels to handle "high entropy" and "high robustness" roles respectively.
- WMAR collapses under time-domain attacks despite fine-tuning, showing that "making a codec idempotent" does not solve alignment-breaking attacks like cropping. Cluster-level hashing ensures the same key is produced even if tokens are replaced by neighbors, fundamentally aligning the clustering geometry with attack behavior.

## Highlights & Insights
- Quantizing the empirical observation that "re-encoding errors are structured" into a Confusion Matrix $\to$ Graph $\to$ Leiden Community pipeline, and then mapping this to cluster-level KGW, is a beautiful "identification of the right abstraction."
- Fully gradient-free and black-box nature means this is friendly for closed-source codecs or API deployments, allowing attribution for commercial services where retraining is not permitted.
- The multi-channel resolution strategy is transferable to any RVQ-based multimodal generation (VQ-VAE images, video tokens, SEED-LLaMA): distributing "robustness budget" and "entropy budget" across codebooks is a universal trick.

## Limitations & Future Work
- The paper does not fully characterize the Pareto front of the "$c$ (compression) - $h$ (context) - $\delta$ (bias)" space, relying on data-driven heuristics.
- The conditional independence assumption holds for non-sliding window codecs, but Mimi/EnCodec have overlapping convolutions; detection estimates under extreme attacks might be slightly optimistic.
- Cluster vocabularies are dataset-dependent—shifting to a significantly different distribution (e.g., extreme music genres) might require re-running community detection.
- As part of the KGW family, it remains vulnerable to "rewrite attacks" where another LM paraphrases or regenerates the content, though it effectively cleans up "noise in the signal channel."

## Related Work & Insights
- **vs WMAR (Wu et al., 2025a,b)**: WMAR fine-tunes the codec so $r \to 1$, while Ours keeps the codec and ensures $r_{cl} \to 1$. Ours is cheaper, avoids quality loss, and shows better detection.
- **vs k-means Distort-free Audio Watermark (Wu et al., 2025a)**: k-means uses semantic embedding distance, which may mismatch actual codec confusion behavior; Ours uses the confusion graph itself to better align objectives.
- **vs Image Token Watermarks (Tong 2025 / Jovanović 2025)**: These works face similar re-encoding errors but choose fine-tuning; the "community detection + cluster KGW" approach is modality-agnostic and could potentially be applied to image/video VQ tokens.
- **vs Post-processing Audio Watermarks (San Roman 2024, Liu 2024)**: Post-processing embeds payloads in waves, but modern codecs can wipe these out; Ours embeds in the token distribution and utilizes the codec's confusion structure rather than fighting it.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The dual relationship between codec confusion graphs and cluster-level KGW is clean, original, and obvious-in-hindsight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid grid of 2 codecs $\times$ 3 $h$-values $\times$ 12 attacks $\times$ 4 baselines, though OOD cluster transferability is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation from $r^{h+1}$ to $r_{cl}^{h+1}$; diagrams and tables are well-structured.
- Value: ⭐⭐⭐⭐⭐ Achieves a new SOTA without training costs, offering high practical value for real-world attribution systems across RVQ/VQ modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] Flatness-Aware Stochastic Gradient Langevin Dynamics](flatness-aware_stochastic_gradient_langevin_dynamics.md)
- [\[ICML 2026\] The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)

</div>

<!-- RELATED:END -->
