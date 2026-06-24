---
title: >-
  [Paper Note] RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding
description: >-
  [ICML 2025 Spotlight][Information Retrieval & RAG][Speculative Decoding] This paper proposes RAPID, a framework combining RAG with speculative decoding. It utilizes a RAG drafter (an LLM running on compressed retrieval contexts) to generate candidate tokens for a long-context target LLM, and enhances the target distribution through test-time knowledge distillation. This simultaneously delivers a >2× speedup and improved generation quality in long-context inference.
tags:
  - "ICML 2025 Spotlight"
  - "Information Retrieval & RAG"
  - "Speculative Decoding"
  - "Long-Context Inference"
  - "Retrieval-Augmented Generation"
  - "Knowledge Distillation"
  - "KV Cache"
date: 2026-05-08
content_hash: b7aab4d395891ed1
---

# RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2502.20330](https://arxiv.org/abs/2502.20330)  
**Code**: [https://github.com/NUS-TRAIL/RAPID](https://github.com/NUS-TRAIL/RAPID)  
**Area**: Information Retrieval  
**Keywords**: Speculative Decoding, Long-Context Inference, Retrieval-Augmented Generation, Knowledge Distillation, KV Cache

## TL;DR

This paper proposes RAPID, a framework combining RAG with speculative decoding. It utilizes a RAG drafter (an LLM running on compressed retrieval contexts) to generate candidate tokens for a long-context target LLM, and enhances the target distribution through test-time knowledge distillation. This simultaneously delivers a >2× speedup and improved generation quality in long-context inference.

## Background & Motivation

Traditional speculative decoding (SD) accelerates inference via a draft-and-verify paradigm using a small draft model and a large target model, but **its performance drops significantly in long-context scenarios**. The core reasons include:

**KV Cache Bottleneck**: In long-context scenarios, KV Cache access becomes memory-bound, which drastically narrows the speed advantage of small models relative to large models. For instance, the throughput advantage of LLaMA-3.1-8B over 70B drops from 23.6× at a 1K context to 9.4× at 128K.

**Trade-offs between RAG and Long Context**: RAG excels at retrieving relevant passages (e.g., in multiple-choice questions) but falls short compared to long-context LLMs in tasks requiring global understanding (e.g., long-form question answering). While complementary, there was previously no effective method to mathematically unify them during inference.

**Strict Rejection in SD**: Traditional SD uses the target LLM distribution as the "ground truth" for rejection sampling. When the RAG drafter actually produces better generation quality, high-quality candidates may be unnecessarily rejected, leading to wasted computation.

**Key Observation**: At a 128K context, LLaMA-3.1-8B using RAG on compressed 4K~16K retrieval contexts can recover most of its long-context performance, indicating that a RAG-based drafter can efficiently generate high-quality candidate tokens.

## Method

### Overall Architecture

RAPID consists of two core components:

1. **RAG Drafter**: Uses a retrieval-augmented, short-context LLM instead of a traditional smaller model as the draft model to generate candidate tokens on a compressed retrieval context $\mathcal{C}^S$.
2. **Retrieval-Augmented Target Distribution**: Transfers knowledge from the RAG drafter to the target distribution via test-time knowledge distillation, thereby increasing the acceptance rate of candidate tokens.

Workflow: The RAG drafter quickly generates γ candidate tokens on the short context $\rightarrow$ the target LLM validates them in parallel on the full long context $\rightarrow$ rejection sampling is conducted based on the augmented target distribution.

RAPID supports two settings:

- **Self-speculation**: The target LLM and the RAG drafter are of equal capacity (e.g., 8B-8B).
- **Upward-speculation**: The RAG drafter is larger than the target LLM (e.g., 70B drafter $\rightarrow$ 8B target). Because the RAG drafter processes short contexts, its computational overhead remains manageable.

### Key Designs

#### 1. RAG Drafter Construction

The long context $\mathcal{C}$ is split into 512-token chunks, which are encoded into vectors using BGE-M3. The top-$k$ relevant passages are retrieved based on cosine similarity (with a similarity threshold of 0.3) to construct the compressed context $\mathcal{C}^S$. The compression ratio is strictly controlled to ensure $|\mathcal{C}^S| \leq |\mathcal{C}|/\lambda$ ($\lambda \gg 1$), with the retrieved length constrained between 4,096 tokens and 1/24 of the original length.

The draft distribution is defined as:

$$q(x_i) = q_{\psi}(x_i | [\mathcal{C}^S; x_{<i}])$$

Core advantages: (1) Eliminates the KV Cache memory bottleneck of long contexts; (2) Potentially yields higher-quality candidates by focusing on relevant information.

#### 2. Retrieval-Augmented Target Distribution

Traditional SD performs rejection sampling directly using the target distribution $p(x_i)$, but the RAG drafter can exhibit better quality in certain scenarios. RAPID frames the RAG drafter as a "teacher" to enhance the target distribution via a one-step test-time knowledge distillation:

$$\hat{z}(x_i) = z(x_i) + \eta T(q(x_i) - p(x_i))$$

Where $z(x_i)$ represents the raw logits of the target LLM, $\eta$ controls the intensity of knowledge transfer, and $T$ is the temperature. The augmented target distribution is:

$$\hat{p}(x_i) = \text{softmax}(\hat{z}(x_i) / T)$$

**Mathematical Derivation**: This is mathematically equivalent to taking a single gradient descent step on the KL divergence distillation loss $\mathcal{L} = T^2 \cdot \text{KL}(q \| p)$. The gradient is $\partial\mathcal{L}/\partial z = T(p - q)$, yielding $\hat{z} = z - \eta \cdot \partial\mathcal{L}/\partial z = z + \eta T(q - p)$.

#### 3. Tail-Distribution Protection

To prevent the knowledge transfer from distorting the long-tail distribution, tokens with probabilities lower than 10% of the peak value retain their original target distribution:

$$\hat{w}_k = w_k, \quad \forall k: \hat{w}_k < 0.1 \cdot \max_j \hat{w}_j$$

#### 4. Adjusted Residual Sampling

Upon rejection, sampling is performed from an adjusted residual distribution, which remains mathematically equivalent to sampling directly from the original target distribution:

$$x_i \sim \text{norm}(\max(p(x_i) - \hat{p}(x_i), p(x_i) - q(x_i), 0))$$

## Loss & Training

RAPID **requires no training** and is a purely training-free, drop-in decoding method. Hyperparameter settings:

- Candidate length per step γ=10
- Self-speculation: η ∈ {5, 10, 20}
- Upward-speculation: η ∈ {40, 50}

## Key Experimental Results

### Main Results

Evaluated on $\infty$Bench and LongBench v2, featuring LLaMA-3.1 and Qwen2.5 model families:

| Configuration (Target → Draft) | ∞Bench AVG | LongBench v2 CoT | Speedup |
|---|---|---|---|
| LLaMA-3.1-8B LC (baseline) | 39.33 | 30.4% | 1.00× |
| LLaMA-3.1-8B RAG | 40.40 | 33.4% | 3.35× |
| LLaMA-3.1-8B + SD | 39.64 | 31.0% | 1.63× |
| LLaMA-3.1-8B + MagicDec | 37.35 | 30.6% | 0.71× |
| **RAPID self (8B→8B)** | **42.83** | **34.2%** | **2.10×** |
| **RAPID upward (8B→70B)** | **49.98** | **40.2%** | 1.14× |
| LLaMA-3.1-70B LC | 45.07 | 36.2% | 1.00× |
| **RAPID self (70B→70B)** | **50.62** | **40.2%** | **2.69×** |
| Qwen2.5-7B LC | 38.12 | 33.2% | 1.00× |
| **RAPID self (7B→7B)** | **42.48** | **35.4%** | **2.65×** |
| **RAPID upward (7B→72B)** | **48.72** | **41.2%** | 0.93× |

### Ablation Study

**Robustness to Parameter η** (stress test utilizing irrelevant retrieved contexts):

| η | Self-spec ΔAcc | Self-spec Speedup | Upward-spec ΔAcc | Upward-spec Speedup |
|---|---|---|---|---|
| 0 | +1.20 | 1.62× | -1.30 | 0.67× |
| 5 | +2.80 | 1.75× | +0.40 | 0.69× |
| 10 | +1.60 | 1.77× | +1.20 | 0.72× |
| 20 | +1.20 | 1.78× | +4.40 | 0.75× |
| 40 | -2.60 | 2.08× | +6.60 | 0.84× |
| 50 | -6.30 | 2.10× | +6.00 | 0.87× |

**Multi-turn Dialogue Experiments** (MT-Bench-101, 122K context):

| Method | Quality Score (1-10) | Acceptance Rate | Throughput (tok/s) |
|---|---|---|---|
| Target LLM | 2.82 | - | 10.64 |
| RAG Drafter | 3.95 | - | 40.49 |
| SD | 2.94 | 56.34% | 14.07 |
| **RAPID** | **4.21** | **76.94%** | **18.18** |

### Key Findings

1. **Self-speculation consistently outperforms both long-context (LC) and RAG**: On $\infty$Bench, RAPID boosts LLaMA-3.1-8B from 39.33 to 42.83, while delivering a 2.10× speedup.
2. **Remarkable Performance of Upward-speculation**: Combining an 8B target with a 70B RAG drafter achieves 49.98 on $\infty$Bench, which even exceeds the performance of the full 70B LC baseline (45.07).
3. **Emergent Capability**: RAPID correctly answers questions where **both the target LLM and the RAG drafter fail individually**, demonstrating a collaborative effect where the sum is greater than the parts ($1+1>2$).
4. **Robustness**: Even when using completely irrelevant retrieved contexts, RAPID retains positive performance gains under an appropriately configured η.
5. **32K Context Inflection Point**: RAPID achieves latency speedup once context length exceeds 32K, whereas traditional SD typically requires contexts over 64K to show gains.

## Highlights & Insights

1. **Paradigm Innovation**: First to propose using RAG as a speculative decoding drafter, elegantly marrying the efficiency benefits of RAG with the quality guarantees of speculative decoding.
2. **Upward-speculation Breaks Convention**: While conventional SD requires a smaller model as the drafter, RAPID enables a larger model to act as the drafter (due to its execution on compressed short contexts), creating a novel decoding paradigm.
3. **Test-Time Knowledge Distillation**: Requires no specialized training; active knowledge transfer is achieved simply through a single-step gradient update in the logit space, making the approach elegant and highly efficient.
4. **Theoretical Guarantees**: Proves that the adjusted residual sampling remains mathematically equivalent to sampling directly from the original target distribution, thus preserving the lossless nature of speculative decoding.
5. **High Practical Utility**: As a training-free, drop-in method with no modifications to model weights, it can be directly integrated into existing long-context LLM deployment frameworks.

## Limitations & Future Work

1. **Upward-speculation Requires Extra GPU Resources**: Although the large RAG drafter is highly effective, deploying the larger model demands extra GPU memory and resources.
2. **Dependency on Retriever Quality**: Despite showing robust optimization under poor retrieval quality, extremely low-quality retrieval still degrades final performance.
3. **Sensitivity of Parameter η**: The optimal range for η differs between the self-speculation and upward-speculation settings, requiring task-specific tuning.
4. **No Acceleration in the Prefill Stage**: RAPID mostly accelerates the generation (decoding) phase; the prefill latency remains largely unchanged as the target LLM still needs to process the entire long context.
5. **Opportunity for Dynamic η**: The current implementation treats η as a static hyperparameter. Future work could explore dynamically adjusting η based on retrieval confidence or step-wise token position.

## Related Work & Insights

- **TriForce / MagicDec**: Perform long-context SD by compressing the KV Cache. However, compression degrades the quality of the draft model. RAPID bypassed this with a superior RAG-based drafter design.
- **REST (He et al., 2024)**: Retrieves candidate continuations directly from a datastore, which is complementary to RAPID's design of combining "retrieved context + model-based generation."
- **Speculative RAG (Wang et al., 2024)**: Implements speculative decoding concepts to improve RAG quality, whereas RAPID does the reverse—leveraging RAG to boost the speed of speculative decoding.
- **Inspiration**: This paradigm of performing "model fusion via logit manipulation during inference" could be extended to other context settings, such as multi-model collaboration and MoE (Mixture-of-Experts) inference.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Speculative decoding coupled with RAG is an elegant alignment, and the upward-speculation paradigm is highly refreshing.
- **Technical Depth**: ⭐⭐⭐⭐ — Test-time knowledge distillation is supported by solid mathematical derivation and theoretical guarantees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 models × 2 settings × 2 benchmarks, with thorough ablations on robustness, context, and retrieval lengths.
- **Value**: ⭐⭐⭐⭐⭐ — A training-free, drop-in method that is directly ready for accelerating long-context inference in practical deployments.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly structured, clear motivations, and insightful takeaways.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](../../ICLR2026/information_retrieval/bridging_draft_policy_misalignment_group_tree_optimization_for_speculative_decod.md)
- [\[ACL 2025\] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](../../ACL2025/information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](../../ACL2025/information_retrieval/hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](../../ACL2025/information_retrieval/gor_rag_long_context_summary.md)
- [\[ACL 2025\] RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](../../ACL2025/information_retrieval/raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)

</div>

<!-- RELATED:END -->
