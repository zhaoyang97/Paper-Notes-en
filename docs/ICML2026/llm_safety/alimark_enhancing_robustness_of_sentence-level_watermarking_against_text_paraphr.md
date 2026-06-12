---
title: >-
  [Paper Note] AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing
description: >-
  [ICML 2026][LLM Safety][Sentence-level watermarking] AliMark reframes sentence-level text watermarking from "sentence-by-sentence detection conditioned on prefixes" to "encoding and alignment of a global secret bit seque…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Sentence-level watermarking"
  - "robustness to paraphrasing"
  - "sequence alignment"
  - "structural perturbation"
  - "watermark detection"
date: 2026-05-08
content_hash: 6ceebd03e590b9cd
---

# AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing

**Conference**: ICML 2026  
**arXiv**: [2605.29434](https://arxiv.org/abs/2605.29434)  
**Code**: https://github.com/imethanlee/AliMark  
**Area**: LLM Security / Text Watermarking  
**Keywords**: Sentence-level watermarking, robustness to paraphrasing, sequence alignment, structural perturbation, watermark detection  

## TL;DR
AliMark reframes sentence-level text watermarking from "sentence-by-sentence detection conditioned on prefixes" to "encoding and alignment of a global secret bit sequence." By reconstructing candidate texts and utilizing adaptive block edit distance, it significantly enhances detection robustness against strong paraphrasing attacks from DIPPER, GPT-3.5, and others.

## Background & Motivation
**Background**: LLM text watermarking is generally categorized into token-level and sentence-level methods. Token-level methods bias the sampling distribution during decoding and analyze signals statistically; sentence-level methods anchor watermarks in a semantic embedding space, aiming to retain detection signals after synonymous paraphrasing.

**Limitations of Prior Work**: Token-level watermarks are easily disrupted by synonym substitution and rewriting. While sentence-level watermarks are more resistant to surface-level changes, many inherit the KGW-style prefix design: the watermark signal of a sentence depends on the previous sentence or context. When a paraphraser splits, merges, or reorders sentences, the "prefixes" for subsequent sentences become misaligned, causing a cascading loss of detection signals.

**Key Challenge**: Sentence-level watermarking relies on semantic stability, but prefix conditioning binds the detection of each sentence to a local structure. Strong paraphrasers primarily destroy sentence boundaries and context order rather than semantics; thus, local prefix hashing amplifies structural perturbations into multi-sentence signal failures.

**Goal**: This paper aims to address three specific problems: how to embed sentence-level signals during generation without local prefix dependency, how to tolerate sentence splitting and merging during detection, and how to maintain low false-positive detection capabilities after strong paraphrasing without significantly sacrificing text quality.

**Key Insight**: The authors observe that GPT-3.5 frequently changes sentence counts when paraphrasing C4 text, suggesting that "sentence boundary changes" are standard behavior for strong paraphrasers rather than edge attacks. Consequently, they borrow sequence alignment concepts from token watermarking for handling insertions and deletions, treating the entire text as a sequence of bit blocks to be matched against a secret sequence.

**Core Idea**: Use a global secret bit sequence to replace prefix-dependent pseudo-random relationships between sentences, and then apply text reconstruction and block-level edit distance alignment to absorb offsets caused by sentence splitting, merging, insertion, and deletion.

## Method
The core of AliMark is not a more complex local hash, but shifting the detection paradigm from "whether each sentence hits a green zone" to "how well the entire text matches a secret bit sequence." This changes the interface at both ends: during generation, each sentence carries a fixed-length bit block; during detection, block-level alignment allows for insertions, deletions, and substitutions.

### Overall Architecture
In the generation phase, given a prompt and context, the LLM generates $Q$ candidate next sentences. AliMark maps each candidate to a semantic vector using a sentence embedder and computes inner products with a set of orthogonal secret vectors; the sign of the $m$-th inner product determines the $m$-th watermark bit. The global secret sequence is divided into blocks of length $M$, and the $n$-th sentence aims to match the $n$-th secret block. If multiple candidates match, one is selected randomly; otherwise, the one with the highest bit-match count is chosen.

In the detection phase, the input text is segmented into sentences and passes through a two-stage robustness pipeline. The **Re-Structurer** creates candidate texts by trial merges of adjacent sentences and trial splits of single sentences. The **Adaptive Bit Sequence Alignment** module then extracts bit block sequences from these candidates and aligns them with secret sequence candidates of varying lengths using dynamic programming. The maximum detection score among all candidates is selected.

### Key Designs
1. **Prefix-Independent Bit Block Embedding**:
	- **Function**: Allows each sentence to carry a bit block directly from the secret sequence, eliminating dependency on previous sentences.
	- **Mechanism**: Calculates the sign of the inner product between the sentence embedding $e$ and secret vector $v_m$. If $\langle e,v_m\rangle<0$, the bit is 0, otherwise 1. The resulting length-$M$ vector is the watermark block.
	- **Design Motivation**: Sentence semantics remain relatively stable after paraphrasing, but boundaries do not. Binding signals to the current sentence's semantic block prevents a single prefix change from ruining all subsequent detections.

2. **Text-Level Re-Structurer**:
	- **Function**: Actively attempts to recover sentence boundaries altered by paraphrasers before detection.
	- **Mechanism**: For a text with $N$ sentences, it enumerates $N-1$ single-step merge candidates and $N$ single-step split candidates, passing these to the alignment module.
	- **Design Motivation**: Strong paraphrasers like DIPPER and GPT-3.5 often introduce structural perturbations through single splits or merges. Single-step enumeration covers common errors at a lower computational cost than exhaustive multi-step combinations.

3. **Adaptive Block-Level Sequence Alignment**:
	- **Function**: Tolerates bit block misalignments caused by sentence insertions, deletions, splits, or merges.
	- **Mechanism**: Calculates a Block Edit Rate (BER) between extracted blocks and candidate secret sequences within a length range $[\alpha N', \beta N']$. Insertion and deletion costs are $M$, while substitution cost is the Hamming distance. The minimum cost is converted to a $z$-score.
	- **Design Motivation**: Structural perturbations affect entire blocks. Standard bit-level Levenshtein distance underestimates these block-level offsets; BER aligns better with the error granularity of sentence-level watermarking.

### Loss & Training
AliMark is not an end-to-end trained model; it utilizes a frozen LLM, frozen sentence embedder, and random secret vectors. Generation hyperparameters include bit block size $M$ and candidate budget $Q$; detection hyperparameters include reconstruction candidates and the secret sequence length range. The authors default to all-mpnet-base-v2 as the embedder and use vLLM to minimize KV-cache overhead during candidate generation.

## Key Experimental Results

### Main Results
The study sampled 500 instances from Booksum and C4 using OPT-1.3B and Qwen3-1.7B backbones, with Pegasus, Parrot, DIPPER, and GPT-3.5 as attackers. Below are the TPR@5% results for OPT-1.3B, highlighting robustness to structural perturbations.

| Dataset | Attack | AliMark TPR@5% | Prev. SOTA TPR@5% (PMark) | Gain |
|--------|------|----------------|------------------|----------|
| Booksum | DIPPER | 61.6 | 30.4 | +31.2 |
| Booksum | GPT-3.5 | 66.6 | 33.0 | +33.6 |
| C4 | DIPPER | 49.8 | 29.6 | +20.2 |
| C4 | GPT-3.5 | 51.6 | 28.2 | +23.4 |
| Booksum | Pegasus | 95.6 | 86.0 | +9.6 |
| C4 | Parrot | 91.2 | 89.4 | +1.8 |

### Ablation Study
The table below shows the impact of embedders, budgets, and detection modules.

| Configuration | Key Metric | Description |
|------|---------|------|
| all-mpnet-base-v2 | Booksum/GPT-3.5 TPR@5% 66.6 | Default embedder, most stable overall |
| all-distilroberta-v1 | Booksum/GPT-3.5 TPR@5% 56.8 | Usable but significantly lower under strong paraphrasing |
| multi-qa-mpnet-base-dot-v1 | Booksum/GPT-3.5 TPR@5% 55.2 | Semantic space less suited for watermark blocks |
| $Q=8$ | Booksum/GPT-3.5 TPR@5% 29.6 | Insufficient candidates to match secret blocks |
| $Q=64$ | Booksum/GPT-3.5 TPR@5% 66.6 | High budget significantly improves embeddability |
| AliMark Detection | 128 sentences, 0.34s | RS and adaptive alignment add acceptable overhead |
| w/o RS | 128 sentences, 0.07s | Faster, but detection rate drops sharply under structural changes |
| w/o Ada | 128 sentences, 0.27s | Weakened resilience to deletions and insertions |

### Key Findings
- AliMark’s performance gains are most pronounced against strong paraphrasers like DIPPER and GPT-3.5 that modify sentence structures, addressing the core vulnerability of previous sentence-level methods.
- The Re-Structurer is more critical than adaptive length alignment; omitting RS leads to the largest performance drops because splits/merges destroy sentence boundaries first.
- Text quality remains high. PPL for OPT-1.3B and Qwen3-1.7B stays close to unwatermarked baselines, although excessively large $M$ values can overly constrain the candidate semantic space.

## Highlights & Insights
- Framing sentence-level watermarking as a sequence alignment problem is a powerful abstraction. It accepts that sentence boundaries drift and treats that drift as a solvable sequence offset.
- The BER design is highly tailored to the task. Sentence splits or merges are not independent bit errors but block-level offsets; modeling them as such is more natural than using standard edit distance.
- Single-step reconstruction is a pragmatic compromise, covering most common structural perturbations while keeping detection costs within deployable limits.

## Limitations & Future Work
- The Re-Structurer is limited to single-step operations, struggling with complex restructuring, semantic reordering, or paragraph-level paraphrasing.
- Generation requires a large candidate budget $Q$; although vLLM mitigates this, it remains a burden for low-latency applications.
- Detection still depends on initial sentence segmentation and embedder quality; stability across languages, code-mixed text, or very short snippets requires further validation.
- Future directions include training a lightweight structural restorer to probabilistically identify split or merge points instead of enumerating candidates.

## Related Work & Insights
- **vs KGW / SynthID**: Token-level methods are efficient but sensitive to distribution shifts; AliMark uses semantic blocks for paraphrase-robust detection.
- **vs SemStamp / k-SemStamp**: These methods rely on prefix relationships; structural changes cause cascading failures. AliMark avoids this using global sequences and alignment.
- **vs PMark / SimMark**: Strong under weak paraphrasing but vulnerable to sentence structural changes; AliMark demonstrates that robustness requires explicit structural modeling rather than just semantic similarity.
- **Insight**: Any task partitioning long text into local units for verification or consistency can adopt the "local signal + global sequence alignment" framework to mitigate insertion and deletion errors.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Reframing detection as block-level alignment is a clear innovation targeting real-world attack patterns.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers diverse datasets, backbones, and paraphrasers, though human-paraphrased and cross-lingual scenarios are less explored.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation, well-structured method, and thorough ablation.
- **Value**: ⭐⭐⭐⭐☆ Highly relevant for practical watermark deployment, especially in scenarios requiring resistance to automated paraphrasing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](../../ACL2026/llm_safety/subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](../../NeurIPS2025/llm_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)
- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](../../NeurIPS2025/llm_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[ICLR 2026\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](../../ICLR2026/llm_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)

</div>

<!-- RELATED:END -->
