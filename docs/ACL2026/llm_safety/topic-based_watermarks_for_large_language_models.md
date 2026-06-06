---
title: >-
  [Paper Note] Topic-Based Watermarks for Large Language Models
description: >-
  [ACL 2026][LLM Safety][Text Watermarking] This paper proposes TBW, a lightweight topic-based watermarking scheme that clusters the vocabulary into "green lists" based on semantic topics rather than random partitioning. B…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Text Watermarking"
  - "Topic Alignment"
  - "Semantic Partitioning"
  - "Paraphrase Robustness"
  - "Lightweight Detection"
date: 2026-05-08
content_hash: 877dfc5a7096ee8d
---

# Topic-Based Watermarks for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2404.02138](https://arxiv.org/abs/2404.02138)  
**Code**: [GitHub](https://github.com/ANCP2021/Topic-Based-Watermarks)  
**Area**: AI Security / Text Watermarking  
**Keywords**: Text Watermarking, Topic Alignment, Semantic Partitioning, Paraphrase Robustness, Lightweight Detection

## TL;DR

This paper proposes TBW, a lightweight topic-based watermarking scheme that clusters the vocabulary into "green lists" based on semantic topics rather than random partitioning. By selecting a semantically aligned topic list for logit biasing based on the input prompt, TBW maintains perplexity comparable to unwatermarked text while significantly enhancing robustness against paraphrase and lexical perturbation attacks.

## Background & Motivation

**Background**: LLM-generated text is nearly indistinguishable from human writing, posing risks such as misinformation spread, copyright infringement, and model collapse (AI training on AI). Watermarking techniques identify AI-generated text by embedding detectable signatures during the generation process. The mainstream KGW method randomly partitions the vocabulary into "green" and "red" lists, biasing sampling toward green tokens.

**Limitations of Prior Work**: (1) **Vulnerability of random partitioning**: KGW's random splits mean green tokens may be irrelevant to the current semantic context, allowing attackers to significantly reduce the green token ratio through paraphrasing; (2) **Quality-robustness trade-off**: Computationally intensive methods (EXP-Edit, ITS-Edit) improve robustness via multiple decodings but severely increase latency; lightweight methods like SynthID are efficient but weak against paraphrasing; (3) **Deployment hurdles for semantic watermarking**: Methods like SIR that introduce semantic information require decoder modifications or prompt access, hindering deployment in large-scale commercial LLMs.

**Key Challenge**: Existing methods struggle to balance robustness, text quality, and computational efficiency—lightweight methods are weak against attacks, while robust methods are expensive and degrade text quality.

**Goal**: Design a lightweight semantic-aware watermarking scheme that enhances both robustness and text quality without adding significant computational overhead.

**Key Insight**: Integrate semantic information into vocabulary partitioning—instead of random green/red list assignment, cluster tokens semantically based on predefined topics. Tokens substituted during paraphrasing likely remain within the same topic list, making the watermark signal harder to destroy.

**Core Idea**: Topic-aligned vocabulary partitioning naturally possesses "semantic cohesion"—tokens under the same topic are synonyms or near-synonyms. Lexical replacements during paraphrase attacks are highly likely to fall within the same green list, thereby preserving the watermark signal.

## Method

### Overall Architecture

TBW consists of three phases: (1) **Offline vocabulary partitioning**—allocating all tokens into $K$ topic lists based on semantic similarity; (2) **Online watermark embedding**—extracting topics from the input prompt to select the corresponding green list and applying a logit bias $\delta$ during generation; (3) **Watermark detection**—using a $z$-score statistical test to determine if the text is watermarked, supporting three detection schemes.

### Key Designs

1.  **Token-to-Topic Mapping**:
    *   Function: Assigns all tokens in the vocabulary to semantically consistent topic lists.
    *   Mechanism: Defines $K$ high-level topics (e.g., {animals, technology, sports, medicine}). A sentence embedding model (all-MiniLM-L6-v2) computes the cosine similarity $\text{sim}(v, t_i) = e_v \cdot e_{t_i} / (\|e_v\| \|e_{t_i}\|)$ for each token $v$ and topic $t_i$. If the maximum similarity exceeds threshold $\tau$, the token is assigned to the corresponding topic list $G_{t_i}$; the remaining tokens are distributed across all lists via round-robin to ensure full coverage. $K=4$ corresponds to an effective green list ratio of approximately 0.25.
    *   Design Motivation: Compared to KGW's random partitioning, topic partitioning ensures tokens within the same list are semantically related—synonyms used in paraphrasing are likely to stay in the same green list, making the watermark harder to break.

2.  **Topic-Based Watermark Embedding**:
    *   Function: Embeds a topic-aligned watermark signal during text generation.
    *   Mechanism: Given an input prompt $x^{\text{prompt}}$, KeyBERT extracts key topics. If the extracted topic matches the predefined set, the corresponding list $G_{t^*}$ is selected; otherwise, $k$-means clustering is applied to the extracted topic embeddings to select the most similar predefined topic. During generation, a bias $\delta$ is added to the logits of $v \in G_{t^*}$ at each step, followed by standard softmax sampling. This process requires only one topic extraction and step-wise biasing, involving no extra decoding or re-ranking.
    *   Design Motivation: Semantically aligned green lists make the biased sampling distribution closer to the natural distribution—the model is already inclined to select topic-related tokens. The additional bias has less impact, resulting in lower perplexity.

3.  **Three-level Detection**:
    *   Function: Performs detection with varying trade-offs between robustness and accuracy across different scenarios.
    *   Mechanism: All schemes share the $z$-score statistical test $z = (g - \gamma \cdot n) / \sqrt{n \cdot \gamma \cdot (1-\gamma)}$, where $g$ is the count of green tokens and $n$ is the total token count. (1) **Strict Topic Matching**: Extracts topics from the candidate text to select the green list for $z$-score calculation; (2) **Sliding Window Detection**: Partitions text into windows, performs independent topic extraction, and uses majority voting for the global topic; (3) **Max $z$-score Detection**: Calculates the $z$-score for every predefined topic list and takes the maximum $t^* = \arg\max_{t_i} z_i$—completely independent of topic extraction.
    *   Design Motivation: The Max $z$-score scheme eliminates the risk of topic extraction failure, achieving nearly perfect performance (99.6%-100%) in practice, making it the most viable deployment option.

### Loss & Training

TBW requires no training and only performs logit biasing at inference time. Primary hyperparameters: $K=4$ (number of topics), $\delta=2.0$ (bias strength, unified for comparison with KGW), $\tau=0.7$ (similarity threshold).

## Key Experimental Results

### Main Results — Paraphrase Attack Robustness (ROC-AUC)

| Model | Attack | TBW | KGW | DiP | Unigram | SynthID | SIR |
|------|------|-----|-----|-----|---------|---------|-----|
| OPT-6.7B | No Attack | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 | 0.995 |
| OPT-6.7B | PEGASUS | **0.990** | 0.975 | 0.824 | 0.987 | 0.910 | 0.971 |
| OPT-6.7B | DIPPER | 0.945 | 0.826 | 0.576 | **0.955** | 0.650 | 0.891 |
| Gemma-7B | PEGASUS | 0.981 | 0.983 | 0.836 | **0.985** | 0.912 | 0.952 |
| Gemma-7B | DIPPER | 0.871 | 0.825 | 0.546 | **0.911** | 0.656 | 0.822 |

### Detection Scheme Comparison (OPT-6.7B)

| Detection Scheme | Detection Rate | Avg z-score | Topic Accuracy |
|---------|--------|-------------|----------|
| Strict K-means | 54.0% | 6.32±10.80 | 54.2% |
| Strict Embedding | 57.4% | 7.05±10.68 | 62.4% |
| Sliding Window | 56.6% | 6.91±10.67 | 60.2% |
| **Max z-score** | **99.6%** | **15.88±3.03** | **100%** |

### Key Findings

- **Text Quality**: TBW perplexity is close to the unwatermarked baseline, representing an improvement of approximately 42% (OPT-6.7B) and 48% (Gemma-7B) over Unigram.
- **Paraphrase Robustness**: Under PEGASUS attacks, TPR@1%FPR reaches 91.0% (OPT-6.7B), significantly outperforming KGW's 57.8%.
- **Lexical Perturbation**: TBW maintains high detection scores under both random and targeted perturbations. While Unigram is robust against paraphrasing, it is vulnerable to simple lexical perturbations.
- **Max z-score Detection**: This scheme is nearly perfect (99.6%/100%) and removes the need for topic extraction steps.
- **Computational Efficiency**: TBW generation time is nearly identical to the unwatermarked baseline, whereas EXP-Edit and SIR significantly increase latency.
- **Topic Scalability**: Increasing $K$ from 4 to 32 results in a graceful decline of $z$-scores from ~11 to ~7, remaining competitive.

## Highlights & Insights

- The design of the Max $z$-score detection scheme is ingenious: it entirely bypasses the unreliable topic extraction step, allowing the watermark signal to "self-select" the correct topic list. This "try all possibilities and pick the best" strategy is simple yet effective, boosting detection rates from 57.4% to 99.6%.
- Semantic cohesion is the linchpin of TBW's robustness: tokens replaced by synonyms remain within the same topic list, a property that random partitioning schemes lack. This insight is transferable to any watermarking scenario requiring edit-resistance.
- TBW has an extremely low deployment barrier: it requires no modifications to model architecture, no multiple decodings, and no access to internal decoder parameters—only a logit-level bias.

## Limitations & Future Work

- Using only four broad topics (animals, technology, sports, medicine) limits topic matching precision for domain-specific texts.
- The use of secret random seeds when assigning residual tokens via round-robin increases security but adds a key management burden.
- Robustness against stronger semantic attacks, such as meticulous human rewriting, remains untested.
- Detection requires knowledge of parameters like bias strength $\delta$ and topic configurations, limiting cross-provider interoperability.
- While topic drift in long texts is mitigated by the Max $z$-score scheme, more fine-grained paragraph-level detection is worth exploring.

## Related Work & Insights

- **vs KGW**: KGW uses random partitioning, forcing semantically unrelated tokens into the same list. TBW clusters by semantics, ensuring natural correlation within green lists; this yields a TPR@1%FPR of 91.0% under PEGASUS attack compared to KGW's 57.8%.
- **vs SynthID-Text**: SynthID uses tournament sampling for efficiency but has weak paraphrase resistance (ROC-AUC 0.650 under DIPPER); TBW is equally lightweight but achieves a ROC-AUC of 0.945.
- **vs Unigram**: Unigram's paraphrase resistance is comparable to TBW, but it is more vulnerable to simple lexical perturbations. TBW performs well under both classes of attack.
- **vs SIR**: SIR uses user context to enhance robustness but require specific decoder modifications and prompt access; TBW requires no model changes.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing semantic topics into watermark partitioning is a natural but effective improvement; the Max $z$-score detection is particularly clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across text quality, paraphrase/perturbation robustness, detection comparisons, efficiency, and scalability.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-defined threat models and detection hierarchies, though some content is repetitive.
- Value: ⭐⭐⭐⭐ Low deployment threshold provides a practical solution for AI text provenance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ACL 2026\] Multi-component Causal Tracing in Large Language Models](multi-component_causal_tracing_in_large_language_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)

</div>

<!-- RELATED:END -->
