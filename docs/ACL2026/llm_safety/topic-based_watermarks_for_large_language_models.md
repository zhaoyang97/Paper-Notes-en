---
title: >-
  [Paper Note] Topic-Based Watermarks for Large Language Models
description: >-
  [ACL 2026][LLM Safety][Text Watermarking] This paper proposes TBW, a lightweight topic-based watermarking scheme that clusters the vocabulary into semantically coherent "green lists" via predefined topics (rather than ra…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Text Watermarking"
  - "Topic Alignment"
  - "Semantic Partitioning"
  - "Paraphrase Robustness"
  - "Lightweight Detection"
date: 2026-05-08
content_hash: 10438a6ce42b2908
---

# Topic-Based Watermarks for Large Language Models

**Conference**: ACL 2026
**arXiv**: [2404.02138](https://arxiv.org/abs/2404.02138)
**Code**: [GitHub](https://github.com/ANCP2021/Topic-Based-Watermarks)
**Area**: AI Safety / Text Watermarking
**Keywords**: Text Watermarking, Topic Alignment, Semantic Partitioning, Paraphrase Robustness, Lightweight Detection

## TL;DR

This paper proposes TBW, a lightweight topic-based watermarking scheme that clusters the vocabulary into semantically coherent "green lists" via predefined topics (rather than random partitioning), selects the topic list most aligned with the input prompt for logit bias injection, and achieves text quality comparable to unwatermarked outputs while significantly improving robustness against paraphrase and lexical perturbation attacks.

## Background & Motivation

**Background**: LLM-generated text has become nearly indistinguishable from human writing, giving rise to risks including misinformation propagation, copyright infringement, and model collapse (AI training on AI-generated data). Watermarking addresses this by embedding detectable signatures during generation. The dominant approach, KGW, randomly partitions the vocabulary into "green"/"red" lists and biases sampling toward green tokens.

**Limitations of Prior Work**: (1) **Fragility of random partitioning**: KGW's random partition makes green-list tokens semantically unrelated to the current context, allowing attackers to substantially reduce the green token ratio through paraphrasing. (2) **Quality–robustness trade-off**: Computationally intensive methods (EXP-Edit, ITS-Edit) improve robustness via multiple decoding passes but incur significant latency; lightweight methods such as SynthID are efficient but weak against paraphrasing. (3) **Deployment barriers for semantic watermarking**: Methods that incorporate semantic information, such as SIR, require decoder modifications or prompt access, hindering deployment in large-scale commercial LLMs.

**Key Challenge**: Existing methods struggle to simultaneously achieve robustness, text quality, and computational efficiency — lightweight methods are vulnerable to attacks, while robust methods are computationally expensive and degrade text quality.

**Goal**: To design a lightweight, semantically aware watermarking scheme that improves both robustness and text quality without introducing significant computational overhead.

**Key Insight**: Introducing semantic information into vocabulary partitioning — replacing random green/red list assignment with semantic clustering by predefined topics. Synonyms substituted during paraphrasing are likely to remain in the same topic list, making the watermark signal harder to destroy.

**Core Idea**: Topic-aligned vocabulary partitioning exhibits natural "semantic cohesion" — tokens within the same topic are synonyms or near-synonyms, so lexical substitutions under paraphrase attacks are likely to remain within the same green list, thereby preserving the watermark signal.

## Method

### Overall Architecture

TBW consists of three stages: (1) **Offline vocabulary partitioning** — all tokens are assigned to $K$ topic lists based on semantic similarity; (2) **Online watermark embedding** — the topic is extracted from the input prompt, the corresponding green list is selected, and a logit bias $\delta$ is applied to green tokens during generation; (3) **Watermark detection** — a $z$-score statistical test determines whether a text is watermarked, supported by three detection variants.

### Key Designs

1. **Topic-Aligned Vocabulary Partitioning (Token-to-Topic Mapping)**:

    - **Function**: Assigns all tokens in the vocabulary to semantically coherent topic lists.
    - **Mechanism**: $K$ high-level topics are predefined (e.g., {animals, technology, sports, medicine}). A sentence embedding model (all-MiniLM-L6-v2) computes the cosine similarity $\text{sim}(v, t_i) = e_v \cdot e_{t_i} / (\|e_v\| \|e_{t_i}\|)$ between each token $v$ and each topic $t_i$. If the maximum similarity exceeds threshold $\tau$, the token is assigned to the corresponding topic list $G_{t_i}$; tokens below the threshold are distributed to all lists via round-robin to ensure full vocabulary coverage. With $K=4$, the effective green list ratio is approximately 0.25.
    - **Design Motivation**: Compared to KGW's random partitioning, topic-based partitioning ensures that tokens within the same list are semantically related — synonym substitutions under paraphrase attacks are likely to remain within the same green list, making the watermark signal harder to corrupt.

2. **Topic-Based Watermark Embedding**:

    - **Function**: Embeds a topic-aligned watermark signal during text generation.
    - **Mechanism**: Given input prompt $x^{\text{prompt}}$, keywords are extracted using KeyBERT. If the extracted topic directly matches a predefined topic, the corresponding list $G_{t^*}$ is selected; otherwise, $k$-means clustering is applied to the extracted topic embeddings and the predefined topic most similar to the centroid is chosen. During generation, a logit bias $\delta$ is added to all $v \in G_{t^*}$ at each step, followed by standard softmax sampling. The entire process requires only one topic extraction step and per-step logit biasing, with no additional decoding passes or reranking.
    - **Design Motivation**: Semantically aligned green lists make the biased sampling distribution closer to the natural distribution — the model already tends to select tokens relevant to the topic, so the additional bias has a smaller perturbative effect, resulting in lower perplexity.

3. **Three-Tier Watermark Detection Scheme**:

    - **Function**: Supports detection under different robustness/accuracy trade-offs across deployment scenarios.
    - **Mechanism**: All variants share the $z$-score statistical test $z = (g - \gamma \cdot n) / \sqrt{n \cdot \gamma \cdot (1-\gamma)}$, where $g$ is the number of green tokens and $n$ is the total token count. (1) **Strict topic matching**: The topic is extracted from the candidate text, matched to a predefined topic to select the green list, and the $z$-score is computed. (2) **Sliding window detection**: The text is divided into windows; the topic is independently extracted per window, and a majority vote determines the global topic. (3) **Maximum $z$-score detection**: The $z$-score is computed separately for each predefined topic list, and the maximum is taken: $t^* = \arg\max_{t_i} z_i$ — entirely independent of topic extraction.
    - **Design Motivation**: The maximum $z$-score scheme eliminates the risk of topic extraction failure and achieves near-perfect detection in practice (99.6%–100%), making it the most practical deployment option.

### Loss & Training

TBW requires no training and applies logit biasing only at inference time. Key hyperparameters: $K=4$ (number of topics), $\delta=2.0$ (bias strength, unified with KGW for fair comparison), $\tau=0.7$ (similarity threshold).

## Key Experimental Results

### Main Results — Robustness Against Paraphrase Attacks (ROC-AUC)

| Model | Attack | TBW | KGW | DiP | Unigram | SynthID | SIR |
|------|------|-----|-----|-----|---------|---------|-----|
| OPT-6.7B | None | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 | 0.995 |
| OPT-6.7B | PEGASUS | **0.990** | 0.975 | 0.824 | 0.987 | 0.910 | 0.971 |
| OPT-6.7B | DIPPER | 0.945 | 0.826 | 0.576 | **0.955** | 0.650 | 0.891 |
| Gemma-7B | PEGASUS | 0.981 | 0.983 | 0.836 | **0.985** | 0.912 | 0.952 |
| Gemma-7B | DIPPER | 0.871 | 0.825 | 0.546 | **0.911** | 0.656 | 0.822 |

### Detection Scheme Comparison (OPT-6.7B)

| Detection Scheme | Detection Rate | Mean z-score | Topic Accuracy |
|---------|--------|-------------|----------|
| Strict K-means | 54.0% | 6.32±10.80 | 54.2% |
| Strict Embedding | 57.4% | 7.05±10.68 | 62.4% |
| Sliding Window Embedding | 56.6% | 6.91±10.67 | 60.2% |
| **Maximum z-score** | **99.6%** | **15.88±3.03** | **100%** |

### Key Findings

- **Text quality**: TBW perplexity closely matches the unwatermarked baseline, improving approximately 42% (OPT-6.7B) and 48% (Gemma-7B) over Unigram.
- **Paraphrase robustness**: TPR@1%FPR under PEGASUS attack reaches 91.0% (OPT-6.7B), substantially outperforming KGW at 57.8%.
- **Lexical perturbation**: TBW maintains high detection scores under both random and targeted perturbations; Unigram, despite its paraphrase robustness, is paradoxically more vulnerable to simple perturbations.
- The maximum $z$-score detection scheme achieves near-perfect performance (99.6%/100%) without requiring any topic extraction.
- **Computational efficiency**: TBW generation time is nearly identical to the unwatermarked baseline, whereas EXP-Edit and SIR incur significant additional latency.
- **Topic count scalability**: Increasing $K$ from 4 to 32, the $z$-score degrades gracefully from approximately 11 to approximately 7, remaining competitive.

## Highlights & Insights

- The maximum $z$-score detection scheme is particularly elegant: it entirely bypasses the unreliable topic extraction step, allowing the watermark signal itself to "automatically select" the correct topic list. This "try all possibilities and take the best" strategy is simple yet highly effective, boosting detection rate from 57.4% to 99.6%.
- Semantic cohesion is the key to TBW's robustness: synonym substitutions under paraphrasing are likely to remain within the same topic list — a property unattainable by random partitioning schemes. This insight transfers to other watermarking scenarios requiring robustness to editing.
- TBW has an exceptionally low deployment barrier: no model architecture modifications, no multiple decoding passes, and no access to decoder parameters are required — only logit-level biasing.

## Limitations & Future Work

- Only four very broad topics (animals, technology, sports, medicine) are used, limiting topic matching precision for domain-specific texts.
- Round-robin assignment of residual tokens introduces a random seed as a private parameter, increasing security but also key management overhead.
- Robustness against stronger semantic attacks (e.g., carefully crafted manual rewriting) is not evaluated.
- Detection requires knowledge of parameters such as bias strength $\delta$ and topic configuration, limiting cross-provider interoperability.
- Topic drift in long texts is partially mitigated by the maximum $z$-score scheme, but finer-grained paragraph-level detection warrants further exploration.

## Related Work & Insights

- **vs. KGW**: KGW uses random partitioning, forcing semantically unrelated tokens into the same list; TBW uses semantic clustering, so tokens within the green list are naturally related and synonym substitutions under paraphrasing likely remain in the green list, yielding stronger robustness. KGW's TPR@1%FPR under PEGASUS attack is only 57.8%, compared to TBW's 91.0%.
- **vs. SynthID-Text**: SynthID uses tournament sampling for lightweight efficiency but is extremely weak against paraphrasing (ROC-AUC of only 0.650 under DIPPER); TBW is equally lightweight but achieves a paraphrase-robust ROC-AUC of 0.945.
- **vs. Unigram**: Unigram assigns tokens based on unigram statistics with comparable paraphrase robustness to TBW, but is paradoxically more vulnerable to simple lexical perturbations — TBW performs well under both attack types.
- **vs. SIR**: SIR improves robustness by incorporating user context but requires decoder modification and prompt access, complicating deployment; TBW requires no model modifications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing semantic topics into watermark partitioning is a natural yet effective improvement; the maximum $z$-score detection scheme is particularly ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers text quality, paraphrase/perturbation robustness, detection scheme comparison, efficiency, and scalability — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-organized threat model and detection scheme hierarchy; some content is repetitive.
- **Value**: ⭐⭐⭐⭐ Low practical deployment barrier; provides a practical solution for AI-generated text provenance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](../../AAAI2026/llm_safety/sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](../../AAAI2026/llm_safety/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)

</div>

<!-- RELATED:END -->
