---
title: >-
  [Paper Note] Proxy Compression for Language Modeling
description: >-
  [ICML 2026][Model Compression][byte-level LM] The authors propose "proxy compression": during training, 90% of data is fed as short sequences produced by a tokenizer or neural compressor, and 10% as raw UTF-8 bytes…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "byte-level LM"
  - "tokenizer-free inference"
  - "mixed-representation training"
  - "arithmetic coding"
  - "neural compressor"
date: 2026-05-08
content_hash: 6692bcf31f5c295c
---

# Proxy Compression for Language Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.04289](https://arxiv.org/abs/2602.04289)  
**Code**: https://github.com/LZhengisme/proxy-compression (available)  
**Area**: LLM Efficiency / Byte-level Modeling / Tokenizer Replacement  
**Keywords**: byte-level LM, tokenizer-free inference, mixed-representation training, arithmetic coding, neural compressor

## TL;DR
The authors propose "proxy compression": during training, 90% of data is fed as short sequences produced by a tokenizer or neural compressor, and 10% as raw UTF-8 bytes, combined with sentinel tokens and a brief in-context translation warm-up. At inference, all compressors are discarded and the model sees only raw bytes, yet it significantly outperforms pure byte models under fixed compute, and at scale matches or surpasses tokenizer baselines.

## Background & Motivation
**Background**: Modern LMs are almost universally built on "external fixed tokenizers"—BPE/SentencePiece compress UTF-8 bytes into tokens, making sequence length manageable; arithmetic coding with a small byte LM is a similar form of compression. Tokenizers maximize training efficiency, but tokens become permanently welded into the model interface.

**Limitations of Prior Work**: Hardwiring tokenizers introduces many well-documented side effects—prompt-boundary issues, retokenization drift, glitch tokens ("SolidGoldMagikarp"), low-resource language bias, poor adversarial robustness, etc. More fundamentally, models only learn statistics in token space, not true end-to-end byte modeling. Pure byte training solves these issues, but sequence lengths increase by several times, reducing data seen under the same compute budget and leading to much slower convergence than tokenizer models.

**Key Challenge**: Training efficiency (short sequences) ↔ inference flexibility (byte-level interface) ↔ robustness—existing approaches can only achieve two out of three. Tokenizer models get the first two, pure byte models get the latter two; no approach achieves all three.

**Goal**: Retain the "compressed short sequence" efficiency advantage during training, while enabling inference entirely on raw UTF-8, without any architectural changes (no variants, no tokenizer changes, no attention changes), and with benefits that scale with model size.

**Key Insight**: Treat external compressors as "training-time proxies" rather than permanent interfaces—the same model learns both representations during training and automatically builds internal mappings; at inference, the compressor is discarded, leaving only bytes. The key observation is that large models can encode this cross-representation alignment in their weights.

**Core Idea**: Use a shared vocabulary, add `<comp>/<raw>` sentinels, perform mixed-representation next-token prediction, and conduct in-context translation pairing warm-up for the first 10k steps; inference is on pure bytes.

## Method

### Overall Architecture
Training pipeline: (1) For each sample $x_{\text{raw}}$, with Bernoulli probability $r$ (default 0.9), replace with $x_{\text{comp}}=f(x_{\text{raw}})$, otherwise keep raw; (2) Each segment is wrapped with sentinels as $[\langle\text{raw}\rangle x_{\text{raw}}\langle/\text{raw}\rangle]$ or $[\langle\text{comp}\rangle x_{\text{comp}}\langle/\text{comp}\rangle]$; (3) During warm-up (first 10k steps), enable in-context pairing—concatenate both views of the same sample in the same context, and linearly increase $r$ from 0.4 to 0.9; (4) After warm-up, disable pairing and fix $r=0.9$; (5) At inference, only feed raw bytes, discarding all compressors. Vocabulary sharing: first 64 indices for sentinels, next 256 for UTF-8 bytes, remainder for compressed symbols (tokenizer uses OpenCoder 96,640 vocab; neural uses 16-bit pack for 65,536 symbols; gzip uses 256 bytes).

### Key Designs

1. **Tokenizer-based proxy (§2.2)**:

    - **Function**: Use an off-the-shelf tokenizer to offline compress raw byte streams into token index sequences as $x_{\text{comp}}$, making "training with tokenizer efficiency, inference without tokenizer" the simplest instantiation.
    - **Mechanism**: Directly apply OpenCoder BPE, with an average compression rate of about $2.9\times$. Tokens are input as vocabulary indices, with the only difference from standard tokenizer models being their appearance in `<comp>`-tagged sequences, and a 10% chance of using raw bytes during training. The paper also tried re-encoding token ids as fixed-length byte sequences, but direct id representation performed better.
    - **Design Motivation**: Tokenizer outputs are highly stable (Levenshtein distance barely changes with 10% character deletion), making it easy for the LM to learn the "comp ↔ raw" mapping; also, all preprocessing can be done offline, with no extra training cost.

2. **Neural proxy + entropy segmentation parallelism (§2.3)**:

    - **Function**: Use a 40M byte-level LM + arithmetic coding to optimally entropy-encode byte streams, achieving a compression rate of $\sim 2.6\times$ for a "fuzzy" compressed stream.
    - **Mechanism**: First train a small byte LM to provide $p(\cdot|\text{ctx})$ at each position, then perform arithmetic coding over equal-information windows; to avoid the explosion in serial encoding speed, introduce "entropy segmentation"—use the LM to compute per-byte entropy, treat high-entropy positions as segment boundaries, and compress each segment independently in parallel. Each 16 bits is packed as one symbol. Note: this mapping is a deterministic injection from raw to comp, but not surjective in reverse—different raw bytes may map to the same comp segment ("fuzzy"), but over 90% of colliding raw chunks share LCP $\geq 0.8$, differing only in low-entropy tails like whitespace/newline/indent.
    - **Design Motivation**: Tokenizers are hand-crafted BPE products; neural compressors are theoretically superior. Entropy segmentation is key to making this approach "engineering feasible"—without parallelism, 3.3 TB of data would be intractable. Structured fuzziness helps the model abstract away formatting noise, improving robustness.

3. **In-context translation pairing + sentinel + high $r$ warm-up (§2.1, 2.5)**:

    - **Function**: Enable the model to align comp ↔ raw in its weights, without requiring comp to be present at inference.
    - **Mechanism**: (a) Use `<raw>/<comp>` sentinels to explicitly indicate the representation type of each segment, allowing next-token prediction to be conditioned on representation; (b) During warm-up, concatenate $[\langle\text{raw}\rangle x_{\text{raw}}\langle/\text{raw}\rangle\langle\text{comp}\rangle x_{\text{comp}}\langle/\text{comp}\rangle]$ (order randomized) in the same context, forcing the model to see both views; (c) After warm-up, immediately disable pairing to avoid the model developing a dependency on comp being present at inference. $r$ is ramped from 0.4 to 0.9 to prevent insufficient raw exposure early in training, which would hinder representation alignment.
    - **Design Motivation**: Without pairing, oracle-translation pass@1 only reaches 30–46%; always-on pairing achieves 95%+ but the model becomes dependent on pairing, and downstream raw-byte pass@1 slightly drops; warm-up-only ensures early alignment without fostering dependency, which is empirically optimal (see Table 3).

### Loss & Training
The only loss is standard next-token cross-entropy, treating raw and comp segments equally. Architecture uses EvaByte (efficient byte-level multi-byte prediction), trained for 50K steps / batch of 2M symbols, covering 0.5B / 1.5B / 4B / 7B / 14B model sizes.

## Key Experimental Results

### Main Results
With a fixed 100B symbol training budget (compute roughly matched), pass@1 on HumanEval-Plus / MBPP-Plus:

| Task | Model | 0.5B | 1.5B | 4B | 7B | 14B |
|------|------|------|------|----|----|-----|
| HumanEval-Plus | Tokenizer | 17.7 | 18.3 | 28.0 | 28.7 | 29.3 |
|  | Byte-level | 15.9 | 18.3 | 22.0 | 23.8 | 24.4 |
|  | Proxy (Neural) | 13.4 | 18.3 | 22.6 | 26.8 | 29.9 |
|  | Proxy (Tokenizer) | 12.2 | 20.7 | 24.4 | 26.2 | **30.5** |
| MBPP-Plus | Tokenizer | 29.4 | 41.0 | 46.3 | 45.2 | 48.1 |
|  | Byte-level | 25.9 | 33.6 | 41.8 | 41.3 | 42.1 |
|  | Proxy (Neural) | 22.0 | 29.6 | 41.8 | 41.8 | **49.2** |
|  | Proxy (Tokenizer) | 25.4 | 38.4 | 44.4 | 45.5 | **49.5** |

Proxy surpasses pure byte models at ≥1.5B, and at 14B surpasses the tokenizer baseline—transfer improves with scale.

### Ablation Study

| Configuration | HumanEval-Plus pass@1 (1.5B) | Notes |
|------|-------------------------------|------|
| Always-on pairing | 17.0 | oracle-translation 96%, but ordinary lower |
| Warmup-only (default) | **20.7** | ensures alignment without dependency |
| No pairs | 17.0 | no explicit cross-representation signal |
| Gzip proxy (any ratio) | < pure byte | unstable stream, no transfer |
| Tokenizer / Neural proxy | significantly > byte | stable + structured |

### Key Findings
- Proxy gain is strongly correlated with model size: weak or negative transfer at 0.5B, but at 14B outperforms both byte and tokenizer baselines.
- "Compressor stability" is key for transfer: tokenizer has lowest Levenshtein distance, neural is intermediate, gzip is highest; only the first two transfer successfully, gzip fails completely.
- Robustness inherits byte model advantages: on ReCode perturbations (function rewrite / format / syntax / docstring), 7B proxy model achieves Robust Pass@1 of 19.1 (neural) vs tokenizer baseline 14.9 vs byte baseline 18.7, with almost no degradation on format/docstring.
- Always-on pairing boosts oracle-translation to 95%+, but ordinary pass@1 slightly drops—indicating that "in-context translation" and "internalization in weights" are distinct paths, with the latter determining downstream byte performance.

## Highlights & Insights
- "External compressors as training proxies, discarded at inference" is an elegant decoupling—this approach can generalize to latent diffusion VAEs, audio codecs, and all "external encoder + internal modeling" structures.
- Using sentinel tokens to explicitly mark different representations allows a single model to perform next-token prediction across multiple representations, much simpler than multi-branch/multi-decoder designs, and offers methodological insights for future multi-modal training.
- "Structured fuzziness" is a counterintuitive finding: the non-invertibility of the neural compressor acts as regularization, removing formatting noise and yielding robustness that even exceeds lossless tokenizers.

## Limitations & Future Work
- Experiments are mainly on code corpora (RefineCode), with natural language only validated at 1.5B; whether transfer continues to scale for multilingual and low-resource languages remains to be seen.
- The total vocabulary includes bytes (256) + tokenizer (96K) + sentinels, slightly increasing embedding table memory; neural proxy also requires training and maintaining a 40M byte LM.
- No inference speed comparison is provided: pure byte inference drops the tokenizer but increases sequence length ($\sim 2.9\times$); how much EvaByte's multi-byte prediction offsets this needs finer quantification.
- The total number of bytes seen during training is actually less than in pure byte models—using fixed FLOPs for comparison is fair, but in real deployments where both data and FLOPs are limited, whether proxy still leads remains to be tested.

## Related Work & Insights
- **vs Lester 2024 (neural arithmetic coding LM)**: They use the neural compressed stream as the final representation for both training and inference; this work, in contrast, uses it only as a training proxy, with inference on raw bytes.
- **vs EvaByte / ByT5 / MegaByte**: Extensions in the pure byte direction; this work achieves byte-level inference with orders-of-magnitude higher training efficiency.
- **vs Tokenizer + occasional raw mix**: Superficially similar to token-byte mixed training, but sentinel + warm-up pairing + high $r$ are key differences—removing any of them significantly hurts performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The "compressor as training proxy only" perspective is fresh and practically useful
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 scales + 3 proxy types + multiple robustness/in-context probes
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, intuitive scaling curves
- Value: ⭐⭐⭐⭐ Opens a practical path for "byte-level inference," a direction long suppressed by efficiency constraints

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](../../ICCV2025/model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->
