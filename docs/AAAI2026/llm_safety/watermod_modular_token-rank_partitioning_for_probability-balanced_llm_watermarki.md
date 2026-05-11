---
title: >-
  [Paper Note] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking
description: >-
  [AAAI 2026][LLM Safety][LLM watermarking] This paper proposes WaterMod, an LLM text watermarking method based on modular arithmetic ($\text{rank} \bmod k$) that partitions the vocabulary into modular residue classes afte…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "LLM watermarking"
  - "text watermarking"
  - "modular arithmetic"
  - "zero-bit/multi-bit watermarking"
  - "probability balancing"
date: 2026-05-08
content_hash: da977e978b783790
---

# WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking

**Conference**: AAAI 2026
**arXiv**: [2511.07863](https://arxiv.org/abs/2511.07863)
**Code**: [github](https://github.com/Shinwoo-Park/WaterMod)
**Area**: AI Safety
**Keywords**: LLM watermarking, text watermarking, modular arithmetic, zero-bit/multi-bit watermarking, probability balancing

## TL;DR

This paper proposes WaterMod, an LLM text watermarking method based on modular arithmetic ($\text{rank} \bmod k$) that partitions the vocabulary into modular residue classes after sorting tokens by probability. Under both zero-bit ($k=2$) and multi-bit ($k>2$) watermarking settings, WaterMod achieves high detection rates and low quality degradation within a unified framework, requiring no external thesaurus or hashing tricks.

## Background & Motivation

LLMs can now produce text at near-human fluency for news articles, legal analyses, and code, yet this capability introduces challenges in provenance tracking and downstream risks such as misinformation, plagiarism, and data poisoning. The EU AI Act mandates machine-verifiable source markers for AI-generated content, making watermarking a recommended compliance mechanism. Both OpenAI and Google DeepMind are actively developing text watermarking systems.

**Limitations of Prior Work**:

**KGW (random green/red list partitioning)**: A pioneering work in logit-based watermarking, but random partitioning frequently assigns the most contextually appropriate tokens to the red list, degrading fluency.

**WatMe (synonym clustering)**: Constructs synonym sets via WordNet or LLM prompting to ensure at least one suitable synonym falls in the green list. However, it depends on external synonym resources and is constrained by dictionary coverage, polysemy, and prompt sensitivity.

**LSH (locality-sensitive hashing)**: Generates green lists via semantic hashing of token embeddings, but is susceptible to hyperplane sensitivity, leading to collision errors and semantic drift.

**Most methods support only zero-bit watermarking**: They can only indicate "AI-generated or not," without embedding richer provenance information such as model instance IDs or user IDs.

**Core Idea**: After sorting the vocabulary in descending order of model probability, **tokens at adjacent ranks are treated by the model as semantically interchangeable**. Applying modular arithmetic $\text{rank} \bmod k$ assigns adjacent-rank tokens to different color classes, naturally guaranteeing that every color class contains at least one high-probability token—enabling watermark embedding without sacrificing fluency.

## Method

### Overall Architecture

The core of WaterMod is a unified modular residue class partitioning rule:

1. At each decoding step, sort the vocabulary in descending order of model probability.
2. Partition the resulting ranks into $k$ color classes by $\text{rank} \bmod k$.
3. Apply a logit bias $\delta$ to the selected color class.
4. $k=2$ corresponds to zero-bit watermarking; $k>2$ to multi-bit watermarking.

### Key Designs

#### 1. **Parity-Based Partitioning over Probability-Sorted Ranks (Zero-Bit, $k=2$)**

Given logits $\boldsymbol{\ell}_t$ at time step $t$, probabilities are computed and sorted in descending order as $\pi = \text{argsort}(\boldsymbol{\ell}_t; \downarrow)$. Tokens with $r \bmod 2 = 0$ are assigned to the even class; those with $r \bmod 2 = 1$ to the odd class.

**Key Property**: The highest-probability token (rank 0) and the second-highest (rank 1) are always assigned to different parity classes. Regardless of which class is designated the green list, at least one high-probability token is guaranteed to be available for sampling—a guarantee that random partitioning methods such as KGW cannot provide.

**Entropy-Adaptive Gating**: Shannon entropy is used to determine the probability of designating the odd class as the green list:

$$p_{odd} = \left(\frac{H_t}{H_{max}}\right)^{H_{scale}}$$

- Low entropy (sharp distribution, probability concentrated on few tokens) → low $p_{odd}$ → even class preferred as green list → rank 0 (most probable token) is protected.
- High entropy (flat distribution, many tokens with similar probability) → high $p_{odd}$ → odd class preferred → rank 1 is also a high-probability candidate, so watermark embedding does not harm quality.

The actual green list is determined by a pseudorandom variable $u$ derived from a secret key and the threshold $p_{odd}$: $g = \mathbf{1}[u < p_{odd}]$.

**Design Motivation**: When $H_{scale} > 1$, the mapping is steeper—the odd class is only permitted to serve as the green list when the distribution is genuinely flat, maximizing fluency protection in low-entropy (deterministic) contexts. When $H_{scale} < 1$, watermark embedding is more aggressive.

#### 2. **Modular Residue Class Partitioning and Payload Embedding (Multi-Bit, $k>2$)**

The parity partition is generalized to a $k$-color partition:

$$\mathcal{C}_d = \{\pi[r] \mid r \bmod k = d\}, \quad d \in \{0, \ldots, k-1\}$$

A $b$-bit message is converted into a base-$k$ vector $\mathbf{m} \in \{0, \ldots, k-1\}^{\tilde{b}}$ (where $\tilde{b} = \lceil b / \log_2 k \rceil$).

At each decoding step:
1. A pseudorandom position $p$ is selected via key-based hashing.
2. The current digit $d = \mathbf{m}[p]$ is read.
3. Logits for tokens satisfying $r \bmod k = d$ are increased by $\delta$.

Each generated token thus carries one base-$k$ digit, encoding $\log_2 k$ bits of payload.

**Design Motivation**: Probability mass is distributed nearly uniformly across the $k$ color classes (each class contains tokens ranked $d, d+k, d+2k, \ldots$), so biasing one class does not substantially alter the overall distribution. Multi-bit embedding inherits the fluency guarantees of the zero-bit case.

#### 3. **Detection and Payload Recovery**

**Zero-Bit Detection**: The parity of the green list at each step is reconstructed, the number of green tokens $G$ is counted, and a z-score is computed:

$$z = \frac{G - N/2}{\sqrt{N/4}}$$

A watermarked text is declared when $z > \tau$.

**Multi-Bit Detection**: A count table $C[p][d]$ is constructed, and majority voting over each digit position recovers the payload $\hat{\mathbf{m}}$. The z-score uses the null probability $p_0 = 1/k$:

$$z = \frac{G - Tp_0}{\sqrt{Tp_0(1-p_0)}}$$

The same count table simultaneously supports watermark detection (via z-score) and message recovery (via majority voting), accomplishing both tasks in a single decoding pass.

### Loss & Training

WaterMod is an **inference-time watermark** requiring no additional training. Key configurations:

- Zero-bit: $\delta = 1.0$, $H_{scale} = 1.2$, green list ratio 0.5.
- Multi-bit: $\delta = 2.5$ (higher bias compensates for the smaller target class probability $1/k$), $k=4$, 16-bit payload.
- Decoding strategy: deterministic (argmax), eliminating stochastic variation to facilitate analysis.
- Hardware: single NVIDIA RTX 3090 (24 GB).

## Key Experimental Results

### Main Results

**Zero-Bit Watermarking: 6 Methods × 3 Tasks**

| Method | C4 PPL↓ | C4 AUROC↑ | GSM8K Acc↑ | GSM8K AUROC↑ | MBPP+ Pass@1↑ | MBPP+ AUROC↑ |
|--------|---------|-----------|------------|-------------|---------------|-------------|
| EXPEdit | 36.35 | 36.90 | 10.84 | 37.37 | 22.80 | 34.38 |
| ITSEdit | 31.03 | 11.29 | 11.75 | 35.44 | 20.10 | 27.40 |
| KGW | 21.96 | 80.83 | 51.78 | 44.38 | 29.90 | 72.43 |
| LSH | 26.19 | 88.03 | 53.07 | 52.63 | 41.30 | 30.72 |
| SynthID-Text | 12.77 | **94.36** | 47.61 | 97.65 | 27.80 | 66.90 |
| **WaterMod** | **12.58** | 87.09 | **53.83** | **100** | 36.80 | **82.66** |

- On C4, WaterMod achieves the **lowest perplexity** (12.58 vs. 12.77) while ranking third in detection rate.
- On GSM8K, it achieves **perfect detection** (AUROC = 100) and the **highest accuracy** (53.83%), outperforming SynthID-Text by 13%.
- On MBPP+, its AUROC exceeds the second-best method KGW by 14%, with Pass@1 ranking second.

**Multi-Bit Watermarking (16-bit payload): WaterMod vs. MPAC**

| Method | C4 PPL↓ | C4 AUROC↑ | GSM8K Acc↑ | GSM8K AUROC↑ | MBPP+ Pass@1↑ | MBPP+ AUROC↑ |
|--------|---------|-----------|------------|-------------|---------------|-------------|
| MPAC | 10.88 | 97.78 | 31.77 | 95.05 | 20.60 | 48.40 |
| **WaterMod** | **10.87** | **98.02** | **40.33** | **96.94** | **26.20** | **98.29** |

- WaterMod outperforms MPAC on all metrics.
- On MBPP+, AUROC improves from 48.40 to 98.29, a 103% relative gain.
- GSM8K Accuracy improves by 27%.

### Ablation Study

**Robustness: ChatGPT Paraphrase Attack**

| Source | Mean z-score | AUROC |
|--------|-------------|-------|
| Human writing | 0.09 | — |
| WaterMod (no attack) | 14.89 | 100.00 |
| WaterMod (ChatGPT paraphrase) | 9.95 | 99.95 |

After paraphrasing, the z-score drops from 14.89 to 9.95 but remains far above the human baseline of 0.09; AUROC decreases by only 0.05%. In mathematical reasoning tasks, paraphrasers must preserve mathematical correctness, so many high-rank tokens are not substituted.

**Shannon vs. Spike Entropy**: Spike entropy yields superior detection performance, while Shannon entropy better preserves task quality. WaterMod is robust to both entropy definitions, allowing flexible selection based on application priorities.

### Key Findings

1. **Significant advantage in low-entropy settings (code/math)**: WaterMod achieves 100% AUROC on GSM8K and 82.66% AUROC on MBPP+, far exceeding LSH's 30.72%. Low-entropy settings are a critical weakness for other methods, whereas WaterMod's modular arithmetic guarantees that at least one high-probability token belongs to the biased class.
2. **Unified zero/multi-bit framework**: Switching from binary membership to arbitrary payload capacity requires only changing $k$, with no architectural modifications.
3. **No external dependencies**: No WordNet, embedding hashes, or LLM prompts are required—the method relies entirely on the model's own probability rankings.
4. **Robustness to paraphrasing**: AUROC drops by only 0.05% after ChatGPT paraphrasing.

## Highlights & Insights

- **Remarkably simple and elegant design**: The core idea reduces to a single operation—$\text{rank} \bmod k$—yet addresses the fundamental problem of green lists excluding high-probability tokens.
- **Probability ranking as natural semantic clustering**: Tokens with adjacent model probabilities are typically near-equivalent candidates; modular arithmetic interleaves them across different classes, ensuring each class contains high-quality candidates.
- **Sophisticated entropy-adaptive gating**: Fluency is protected in deterministic contexts (low entropy) while watermark embedding is aggressive in uncertain contexts (high entropy), adapting automatically to different text domains.
- **Natural multi-bit extension**: Scaling from $k=2$ to $k>2$ requires no architectural changes, only a single hyperparameter adjustment.
- **Adherence to Kerckhoffs's principle**: Security depends on the secret key $K$, not on algorithm secrecy.

## Limitations & Future Work

1. **Detection rate on C4 is not the highest**: AUROC of 87.09 is below SynthID-Text's 94.36; there remains room for improvement in high-entropy natural language settings.
2. **Evaluation limited to Qwen-2.5-1.5B**: Although the method is claimed to be model-agnostic, it has not been validated on larger models (7B, 70B).
3. **Deterministic decoding constraint**: All experiments use argmax decoding; performance under temperature sampling or top-p sampling has not been verified.
4. **Manual tuning of $H_{scale}$**: Currently fixed at 1.2; the authors acknowledge that automatic optimization for different domains is possible.
5. **Limited adversarial robustness evaluation**: Only ChatGPT paraphrasing was assessed; stronger watermark attacks (e.g., DIPPER, repeated paraphrasing, translation–back-translation) were not tested.
6. **Payload capacity vs. text length trade-off**: A 16-bit payload may result in recovery errors for short texts due to insufficient observations.

## Related Work & Insights

- The probability-ranking idea in WaterMod motivates a form of "model-intrinsic semantic clustering" that requires no external semantic resources.
- **Relationship to KGW**: KGW can be viewed as a special case of random partitioning, while WaterMod represents an upgrade through probability-sorted modular partitioning.
- **Complementarity with SynthID-Text**: SynthID-Text modifies the sampling strategy while WaterMod modifies logit biases; the two approaches may be combinable.
- Multi-bit watermarking combined with traceability enables fine-grained LLM output provenance (user-level or instance-level identification).
- Watermarking in low-entropy settings (code, mathematics) is a critical open problem; WaterMod offers a viable solution.

## Rating

- Novelty: ⭐⭐⭐⭐ — The modular arithmetic partitioning concept is concise and novel, though the underlying mechanism remains a variant of logit biasing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three domains and multiple baselines, but lacks diversity in model scale and decoding strategies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear, with complete algorithmic pseudocode, intuitive figures, and a thorough appendix.
- Value: ⭐⭐⭐⭐ — Provides a practical, concise, and unified watermarking framework with real-world relevance for compliance and provenance tracking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](../../ACL2026/llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](perturb_your_data_paraphrase-guided_training_data_watermarking.md)
- [\[ICLR 2026\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](../../ICLR2026/llm_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](../../ACL2026/llm_safety/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

</div>

<!-- RELATED:END -->
