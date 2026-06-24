---
title: >-
  [Paper Note] Byte Latent Transformer: Patches Scale Better Than Tokens
description: >-
  [ACL 2025][LLM Pretraining][byte-level LLM] Proposes Byte Latent Transformer (BLT), a tokenizer-free byte-level LLM architecture that aggregates bytes into variable-length patches via entropy-based dynamic grouping. It matches the performance of token-based models at the 8B scale for the first time, while unlocking a new scaling dimension of improving inference efficiency by simultaneously scaling both patch and model sizes.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "byte-level LLM"
  - "tokenizer-free"
  - "dynamic patching"
  - "entropy-based segmentation"
  - "scaling laws"
date: 2026-05-08
content_hash: fdd0f39501090eaa
---

# Byte Latent Transformer: Patches Scale Better Than Tokens

**Conference**: ACL 2025  
**arXiv**: [2412.09871](https://arxiv.org/abs/2412.09871)  
**Code**: [https://github.com/facebookresearch/blt](https://github.com/facebookresearch/blt)  
**Area**: Others  
**Keywords**: byte-level LLM, tokenizer-free, dynamic patching, entropy-based segmentation, scaling laws

## TL;DR

Proposes Byte Latent Transformer (BLT), a tokenizer-free byte-level LLM architecture that aggregates bytes into variable-length patches via entropy-based dynamic grouping. It matches the performance of token-based models at the 8B scale for the first time, while unlocking a new scaling dimension of improving inference efficiency by simultaneously scaling both patch and model sizes.

## Background & Motivation

**Background**: Almost all modern LLMs use tokenizers like BPE to convert byte sequences into tokens from a fixed vocabulary. Tokenization is the only non-end-to-end, heuristic preprocessing step in the entire training pipeline, having become the "default choice."

**Limitations of Prior Work**: Tokenization with a fixed vocabulary introduces multiple defects: (1) the same word can be split into different tokens across different contexts, causing inconsistency; (2) extreme sensitivity to input noise (typos, casing variations); (3) lack of orthographic knowledge (not knowing what characters make up a word); (4) multilingual inequity (longer tokens and higher computational costs for low-resource languages); and (5) domain/modality sensitivity. Prior byte-level models (such as MegaByte) suffered from a computational explosion due to excessively long sequences, making them uncompetitive at scale.

**Key Challenge**: While byte-level modeling eliminates all problems associated with tokenization, the computational cost of running a Transformer directly on bytes is dominated by large FFN layers (rather than attention), where sequence length expansion leads to a linear cost increase. A key insight is that most byte predictions are simple (e.g., bytes following or within a word) and do not require the full computational power of a large Transformer.

**Goal**: How to efficiently train LLMs at the byte level—retaining the advantages of byte-level modeling while matching token-based models in both efficiency and performance.

**Key Insight**: Dynamically allocate computation based on the entropy of the next-byte prediction—allocating more computation where information density is high and less where it is low. This is more principled than BPE's compression heuristics.

**Core Idea**: Use the entropy estimates of a small byte-level language model to dynamically segment bytes into variable-length patches. The large Transformer operates only at the patch level, while lightweight models handle intra-patch bytes.

## Method

### Overall Architecture

BLT consists of three components: (1) **Local Encoder**—a lightweight Transformer (layers $l_\mathcal{E} \ll l_\mathcal{G}$) that encodes input bytes into patch representations by pooling bytes into patches using cross-attention; (2) **Latent Global Transformer**—a large autoregressive Transformer that operates on patch representations and consumes the vast majority of FLOPs; (3) **Local Decoder**—a lightweight Transformer that decodes patch representations back into byte sequences. Pipeline: Input bytes $\rightarrow$ dynamic grouping via entropy patching $\rightarrow$ Local Encoder encoding to patches $\rightarrow$ Global Transformer processing $\rightarrow$ Local Decoder decoding to bytes.

### Key Designs

1. **Entropy Patching (Entropy-based Dynamic Grouping)**:

    - **Function**: Dynamically allocate computational resources according to data complexity.
    - **Mechanism**: Train a 100M parameter byte-level language model to compute the next-byte entropy at each byte position: $H(x_i) = \sum_{v \in \mathcal{V}} p_e(x_i=v|\mathbf{x}_{<i}) \log p_e(x_i=v|\mathbf{x}_{<i})$. A new patch is started when the entropy exceeds a global threshold $\theta_g$. For instance, in "George R.R. Martin", "G" has high entropy (uncertain next character) and thus becomes the start of a new patch, while "eorge" has low entropy and is grouped into the same patch. An approximate monotonicity constraint is also applied: segment when $H(x_t) - H(x_{t-1}) > \theta_r$. Average patch size can be arbitrarily controlled by adjusting these thresholds.
    - **Design Motivation**: BPE compresses based on frequency statistics, which is not necessarily aligned with prediction difficulty. Entropy patching allows the model to dedicate full Transformer computation to difficult-to-predict locations (e.g., the beginning of a new sentence) while bypassing easy areas (e.g., inside words) at almost zero cost.

2. **Hash N-gram Embeddings**:

    - **Function**: Inject local contextual information into byte positions.
    - **Mechanism**: For each byte position $i$, construct 3-gram to 8-gram byte spans and map them to a 500K-sized embedding table using a polynomial rolling hash, adding them to the byte embedding: $e_i = x_i + \sum_{n=3}^{8} E_n^{hash}(\text{Hash}(g_{i,n}))$.
    - **Design Motivation**: A single byte (0-255) contains very little information. N-gram embeddings allow each position to "see" patterns of the preceding bytes (common prefixes, suffixes), compensating for the lack of subword information in byte-level models.

3. **Encoder-Decoder Cross-Attention**:

    - **Function**: Efficiently transfer information between byte and patch representations.
    - **Mechanism**: In the Encoder, patches serve as queries and bytes as keys/values (Perceiver-style), pooling byte information into patches. In the Decoder, this is reversed: bytes act as queries and patches as keys/values. Queries are initialized via max-pooling. Each patch query only attends to bytes within its corresponding patch. The patch dimension $h_\mathcal{G}$ is formed by concatenating multiple heads of dimension $h_\mathcal{E}$.
    - **Design Motivation**: Cross-attention is more efficient than global self-attention and naturally fits the byte-to-patch scale transformation. Masking strategies ensure causal compliance.

### Loss & Training

Standard byte-level autoregressive cross-entropy loss is employed. The Local Decoder outputs 256-dimensional logits (corresponding to the byte vocabulary size). The model is optimized using AdamW ($\beta_1=0.9, \beta_2=0.95$), a learning rate of 4e-4 with cosine decay to 0, a 2000-step warmup, and weight decay of 0.1. Scaling laws are studied on Llama 2 data (2T tokens), and full training is conducted on the BLT-1T high-quality dataset. The batch size is held at 16M bytes/batch, avoiding padding by packing patches.

## Key Experimental Results

### Main Results (Downstream Task Evaluation of 8B Model)

| Model | Arc-E | Arc-C | HellaSwag | PIQA | MMLU | MBPP | HumanEval | Average |
|------|-------|-------|-----------|------|------|------|-----------|------|
| Llama 3 (1T tokens) | 77.6 | 53.3 | 79.1 | 80.7 | 58.1 | 40.2 | 31.1 | 60.0 |
| BLT-Space (6T bytes) | 75.4 | 49.8 | 79.6 | 81.1 | 54.8 | 37.6 | 27.4 | 58.0 |
| BLT-Entropy (4.5T bytes) | **79.6** | 52.1 | **80.6** | 80.6 | 57.4 | **41.8** | **35.4** | **61.1** |

(FLOP-matched. BLT-Entropy is roughly equivalent to or outperforms Llama 3 on 7 out of 7 tasks.)

### Robustness and Character-level Tasks

| Task | Llama 3 (1T) | Llama 3.1 (16T) | BLT (1T) |
|------|-------------|----------------|----------|
| HellaSwag Noise Avg | 56.9 | 64.3 | **64.3** |
| CUTE Character Understanding | 27.5 | 20.0 | **54.1** |
| Spelling | 1.1 | - | **99.9** |
| Spelling Inverse | 30.1 | 3.6 | **99.9** |
| Contains Char | 0.0 | 0.0 | **55.9** |
| Substitute Char | 0.4 | 1.2 | **48.7** |

### Key Findings

- BLT-Entropy outperforms Llama 3 in average performance given equivalent training FLOPs (61.1 vs 60.0), with significant improvements in coding tasks (HumanEval 35.4 vs 31.1).
- **Character-level understanding vastly outperforms token models**: Spelling accuracy achieves 99.9% vs 1.1%, and character inclusion checks reach 55.9% vs 0.0%. Token-based models fundamentally lack access to individual characters inside tokens.
- Noise robustness improves by 8 percentage points, matching Llama 3.1 trained on 16x more data—indicating that byte-level awareness is not something "more data can easily compensate for."
- **Scaling under fixed inference FLOPs**: BLT can simultaneously increase patch size and model parameters while maintaining a constant inference cost. The Patch size 8 model outperforms the BPE model after approximately 2.5x compute-optimal training data.
- Significant improvements in low-resource language translation: Armenian 1.7 $\rightarrow$ 6.3, Georgian 1.7 $\rightarrow$ 7.4, Bengali 4.7 $\rightarrow$ 12.7 (BLEU), showing an especially pronounced advantage for non-Latin alphabets.
- BLT-Space (patch size 6) performs slightly below Llama 3 but saves approximately 30% in inference FLOPs, offering a flexible trade-off between performance and efficiency.

## Highlights & Insights

- **Entropy patching elegantly solves computation allocation**: It uses the "uncertainty" of a small model to guide the computational allocation of a large model, letting simple bytes pass at near-zero cost while difficult bytes receive full processing. This is a perfect reflection of "spending computing resources where they matter most."
- **A new scaling dimension**: Scaling the vocabulary of token-based models is limited by embedding layer growth. BLT's patch size can be increased arbitrarily without affecting the parameter count, unlocking a unique path of "larger model + larger patch = better performance + constant inference cost." As the model size scales up, the FLOP ratio of the Local Encoder/Decoder gradually shrinks, making the advantages of larger patch sizes even more prominent.
- **The "free lunch" of byte-level modeling**: Capabilities like spelling, character manipulation, noise robustness, and multilingual processing—which token models require massive amounts of data to partially recover—are naturally built-in with BLT.

## Limitations & Future Work

- The entropy model introduces preprocessing overhead (though this can be optimized using smaller models or lookup tables).
- Inference requires step-by-step determination of patch boundaries (incremental patching), presenting higher engineering complexity than BPE.
- It has only been validated up to the 8B scale; whether the trend persists at 70B+ sizes requires verification.
- BLT-Space performs worse than Llama 3 under equivalent FLOPs, suggesting that excessively large patch sizes lead to information loss—the optimal patch size needs to be adjusted in accordance with scale.
- Exploration in instruction tuning and RLHF scenarios is currently lacking.

## Related Work & Insights

- **vs MegaByte (Yu et al., 2023)**: MegaByte uses fixed-stride grouping without considering complexity and lacks n-gram embeddings and cross-attention. Each innovation in BLT directly addresses a specific limitation of MegaByte.
- **vs SpaceByte (Slagle, 2024)**: Grouping by space is better than a fixed stride but cannot handle non-space-segmented languages (such as Chinese or Japanese), nor can it adjust the patch size. BLT's entropy patching serves as a universally applicable, adjustable alternative.
- **vs Llama 3**: BLT matches performance while providing extra leverage for inference efficiency and character-level capabilities. It exhibits a superior long-term scaling trend—the most compelling evidence of its potential.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Demonstrates for the first time at scale that byte-level models can match token-based models. Entropy patching and patch scaling are highly original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Provides a comprehensive evaluation with full scaling laws from 400M to 8B, downstream tasks, robustness, multilingual capabilities, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear, with transparent and reproducible FLOP calculations and excellent visualization design.
- Value: ⭐⭐⭐⭐⭐ Carries the potential to shift the preprocessing paradigm of LLMs, fundamentally resolving several long-standing pain points of tokenization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ICML 2026\] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](../../ICML2026/llm_pretraining/the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure.md)
- [\[NeurIPS 2025\] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](../../NeurIPS2025/llm_pretraining/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [\[ICML 2025\] Language Models over Canonical Byte-Pair Encodings](../../ICML2025/llm_pretraining/language_models_over_canonical_byte-pair_encodings.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)

</div>

<!-- RELATED:END -->
