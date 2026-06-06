---
title: >-
  [Paper Note] STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability
description: >-
  [ACL 2026][LLM Safety][Watermarking] STELA utilizes "linguistic indeterminacy" $\lambda(c_t)$ estimated by POS n-grams as a watermark strength modulation signal. It weakens the watermark at positions with high syntactic…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Watermarking"
  - "POS n-gram"
  - "Linguistic Indeterminacy"
  - "Publicly Verifiable"
  - "Cross-lingual"
date: 2026-05-08
content_hash: c87a1d842d0a37bd
---

# STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability

**Conference**: ACL 2026  
**arXiv**: [2510.13829](https://arxiv.org/abs/2510.13829)  
**Code**: https://github.com/Shinwoo-Park/stela_watermark  
**Area**: LLM Security / Watermarking / Publicly Verifiable Detection  
**Keywords**: Watermarking, POS n-gram, Linguistic Indeterminacy, Publicly Verifiable, Cross-lingual

## TL;DR
STELA utilizes "linguistic indeterminacy" $\lambda(c_t)$ estimated by POS n-grams as a watermark strength modulation signal. It weakens the watermark at positions with high syntactic constraints (preserving quality) and strengthens it at syntactically free positions (improving detectability). Similar to KGW, it remains publicly verifiable using only a POS tagger, without requiring access to model logits.

## Background & Motivation
**Background**: The foundational LLM watermarking scheme, KGW, uses a hash to partition the vocabulary into green/red lists and adds a bias $\delta$ to green list logits to embed statistical signals. Detection only requires recalculating the hash and performing a z-test without internal model access, allowing for public verification. However, in positions with "low token entropy" (e.g., following proper nouns or mandatory functional words), adding a bias fails to change the most likely token, and forcing change results in degraded generation quality.

**Limitations of Prior Work**: To address the low-entropy issue, schemes like SWEET (selecting positions via token entropy thresholds) and EWD (weighting z-scores by token entropy) were developed. While effective, **both require access to LLM logits for detection**, compromising "public verifiability," a core advantage of KGW. MorphMark adapts at the embedding stage but still relies on output probabilities, leaving its flexibility constrained.

**Key Challenge**: There is a long-standing trade-off between "adaptive strength" and "model-free public detectability": the former typically requires token-level entropy, while the latter usually necessitates static rules.

**Goal**: To identify a signal that is "model-independent yet capable of modulating watermark strength," enabling both insertion and detection to be adaptive without relying on LLM internals.

**Key Insight**: The authors decompose token-level entropy into two causes: "semantic fixing" (e.g., proper nouns) and "syntactic necessity" (e.g., Korean particles). The latter is determined by the syntactic structure of the language and is independent of the specific model. By modeling "syntactic predictability" using POS n-gram conditional entropy, a truly model-free indeterminacy signal can be obtained.

**Core Idea**: A factor $\lambda(c_t) \in [0, 1]$ is calculated using "POS n-gram conditional entropy + language-specific $K$ normalization" as a watermark modulator. On the embedding side, $\delta'_t = \delta \cdot \lambda(c_t)$, and on the detection side, the z-score is weighted by $\lambda$. The entire pipeline requires only a POS tagger.

## Method

### Overall Architecture
Offline Pre-calculation: A POS tagger is run on large human-written corpora (Wikipedia / OpenWebText2 / C4 / KOREAN-WEBTEXT) to compute the conditional probability distribution $P(\pi_t \mid c_t)$ for each POS context $c_t$ of length $k-1$, resulting in a $\lambda$ lookup table.

Online Generation: At each step $t$, the POS context $c_t$ of the preceding $k-1$ tokens is extracted. $\lambda(c_t)$ is retrieved from the table, and the fixed bias $\delta$ of KGW is modified to $\delta'_t = \delta \cdot \lambda(c_t)$ before softmax sampling.

Online Detection: The green list and $\lambda$ weights are recalculated for each position, and detection is determined via a weighted z-score. The entire pipeline requires only a POS tagger and a hash function; the detector is completely model-free.

### Key Designs

1. **Linguistic Indeterminacy $\lambda(c_t)$ — POS n-gram conditional entropy as a model-free modulation signal**:

    - **Function**: Uses a scalar between 0 and 1 to quantify the "syntactic freedom of the next POS tag" given the POS context $c_t$, serving as a multiplier for watermark strength.
    - **Mechanism**: Defines conditional Shannon entropy $H(P(\pi_t \mid c_t)) = -\sum_{\pi'} P(\pi' \mid c_t) \log P(\pi' \mid c_t)$, which is then normalized by the number of distinct tags $K_{c_t}$ that actually appear after context $c_t$: $\lambda(c_t) = H(P(\pi_t \mid c_t)) / \log K_{c_t}$. $\lambda \to 1$ indicates "almost any POS is possible," while $\lambda \to 0$ indicates the "next POS is syntactically fixed." $k$ is a context window hyperparameter, with $k=2$ for English and $k=4$ for Chinese/Korean (based on linguistic typology).
    - **Design Motivation**: Syntax is a property of the language itself, independent of the model. By replacing token entropy with POS conditional entropy, the "modulation signal" is moved from the model space to the linguistic space, decoupling "adaptivity" from "requirement for logits." Normalization ensures comparability across different languages and contexts.

2. **Adaptive Insertion: $\delta'_t = \delta \cdot \lambda(c_t)$**:

    - **Function**: Dynamically adjusts the green-list bias strength during generation based on syntactic freedom. It applies minimal bias in constrained positions (preserving quality) and enhanced bias in free positions (improving detectability).
    - **Mechanism**: Modifies the fixed KGW bias to a linear scaling $l'_{t, i} = l_{t, i} + \delta'_t \cdot \mathbb{I}[i \in \mathcal{V}_G]$. In syntactically constrained positions (e.g., where a Korean nominative particle must appear), $\lambda \approx 0$, resulting in negligible interference. In free positions, $\lambda \approx 1$, and the watermark is embedded at full strength.
    - **Design Motivation**: This allows the watermark to "follow the flow"—interfering where language allows choice and yielding where it does not. This aligns with the statistical properties of human text, keeping perplexity nearly unchanged.

3. **Adaptive Detection: weighted z-score $z'$**:

    - **Function**: Applies the same $\lambda$ weight at the detection stage, allowing green tokens in high indeterminacy positions to contribute more to the detection signal while weakening contributions from low indeterminacy positions.
    - **Mechanism**: Defines a weight $w_t = \lambda(c_t)$ for each token and calculates the weighted statistic $W_G = \sum_t w_t \cdot \mathbb{I}(x_t \in \mathcal{V}_{G, t})$. Under the null hypothesis $H_0$, $\mathbb{E}[W_G] = \gamma \sum_t w_t$ and $\text{Var}(W_G) = \gamma(1-\gamma) \sum_t w_t^2$. The z-score is $z' = \frac{W_G - \gamma \sum_t w_t}{\sqrt{\gamma(1-\gamma) \sum_t w_t^2}}$.
    - **Design Motivation**: This aligns the detection stage with the "per-position modulation" used in generation, giving higher weight to more reliable positions without ever reading LLM logits. It preserves the public verification property of KGW.

### Loss & Training
STELA is a training-free method with no loss or gradients, requiring only offline statistics and online interpolation. During generation, the temperature is fixed at 0.7, the green list ratio $\gamma = 0.5$, and $\delta = 2.0 / \mathbb{E}[\lambda(c_t)]$ (calibrated by language: 0.575 for English, 0.523 for Chinese, 0.475 for Korean) to ensure a fair comparison of average embedding strength with KGW/EWD/SWEET.

## Key Experimental Results

### Main Results: Detection performance across three models and three languages (TPR@5%FPR / Best F1)

| LLM | Method | English TPR / F1 | Chinese TPR / F1 | Korean TPR / F1 |
|-----|--------|------------------|------------------|-----------------|
| Llama-3.2 | KGW | 0.950 / 0.963 | 0.962 / 0.963 | 0.906 / 0.932 |
| Llama-3.2 | SWEET | 0.850 / 0.906 | 0.872 / 0.910 | 0.862 / 0.912 |
| Llama-3.2 | EWD | 0.870 / 0.916 | 0.850 / 0.902 | 0.896 / 0.928 |
| Llama-3.2 | MorphMark | 0.926 / 0.943 | 0.936 / 0.945 | 0.826 / 0.893 |
| Llama-3.2 | **STELA** | 0.938 / 0.953 | **0.976 / 0.972** | **0.950 / 0.954** |
| Qwen-3 | STELA | 0.978 / 0.966 | **0.996 / 0.994** | **0.950 / 0.950** |
| HyperCLOVA | STELA | **0.988 / 0.975** | **0.932 / 0.942** | **0.960 / 0.960** |

Across 9 (model, language) combinations, STELA achieves the highest average F1; on HyperCLOVA, STELA ranks first across all three languages.

### Ablation Study: POS context length $k$ and tagset granularity

| Language | Optimal $k$ | Universal UD tagset TPR | Language-specific tagset TPR | Gain |
|----------|-------------|-------------------------|------------------------------|------|
| English | 2 | 0.948 / 0.972 / 0.984 (3 LLM) | 0.938 / 0.978 / 0.988 | Negligible |
| Chinese | 4 | 0.976 / 0.998 / 0.930 | 0.976 / 0.996 / 0.932 | Negligible |
| Korean | 4 | 0.928 / 0.932 / 0.950 | 0.950 / 0.950 / 0.960 | **+1–2 pts** |

Adversarial Robustness (English Llama-3.2, Dipper rewriting attack): Baseline F1 0.953. F1 remains **0.825** even under heavy rewriting (L=50), and F1 > 0.85 with 50% synonym substitution.

### Key Findings
- STELA's advantage grows with the syntactic complexity of the language: the lead is more pronounced in Chinese and Korean compared to English, confirming that STELA's strength comes from capturing syntactic constraints.
- Korean shows a significant gain (+1.6 TPR) from fine-grained tagsets because the universal UD merges all particles, whereas STELA requires finer distinctions (e.g., nominative JKS vs. accusative JKO) to pinpoint constrained positions.
- Strong model independence: Rankings are consistent across Llama, Qwen, and HyperCLOVA, proving the signal originates from the language rather than the model.
- Exceptional robustness to structural rewriting (Dipper): Since the watermark signal is deeply embedded in the syntactic structure, word substitution or sentence restructuring find it difficult to remove the signal systematically.
- Word class contributions align with linguistic typology: In English, content and functional words each contribute ~43% of the z-score; in Chinese and Korean, content words contribute 67% and 74%, respectively—STELA automatically places watermarks where the language allows "high freedom."

## Highlights & Insights
- **Conceptual shift from "model-specific" to "language-universal" signals**: Previous adaptive watermarks assumed token entropy was an irreplaceable measure. This work replaces it with a model-independent equivalent (POS n-gram entropy), recovering the public verifiability of KGW. This approach of "switching signal sources" can be applied to other scenarios requiring model-free properties (e.g., AI text detection, alignment auditing).
- **Valuable cross-lingual typology validation**: By including analytical (English), isolating (Chinese), and agglutinative (Korean) languages, the study proves the method is not a product of English bias. Future NLP universal tools can reference this typology-aware evaluation.
- **Syntactic-based watermarking inherently resists rewriting**: Syntactic constraints are difficult for rewriting attacks to neutralize (Dipper must still follow grammar rules). Consequently, STELA remains stable under heavy attack, opening a path for robust watermarking based on physical linguistic intuition rather than encryption.
- **Clean coupling of hyperparameter $k$ with language type**: A single hyperparameter $k=2/4$ successfully explains performance differences across three languages, indicating a principled design rather than exhaustive tuning.

## Limitations & Future Work
- Strong dependence on POS tagger accuracy; languages with low resources or missing POS tools may find this method unusable. Performance degrades even with UD tagset fallbacks.
- Text quality evaluation is limited to perplexity and simple LLM-as-judge A/B testing, failing to cover finer dimensions like semantic coherence and stylistic naturalness.
- $\lambda$ is estimated from reference corpora and may be inaccurate if the target domain differs significantly (specialized fields like medicine or law may require domain-specific $\lambda$ tables).
- The three language types are not exhaustive: fusional (deeply inflected Latin/Slavic languages) and polysynthetic languages are missing, along with low-resource languages.
- Lack of adversarial analysis against attackers aware of STELA: If an attacker also calculates $\lambda$, can they evade detection by targeting green tokens in low indeterminacy positions?

## Related Work & Insights
- **vs KGW**: KGW uses static bias; STELA performs adaptive scaling by multiplying bias $\delta$ by $\lambda(c_t)$, achieving significantly higher accuracy while remaining publicly verifiable.
- **vs SWEET / EWD (model-dependent adaptive)**: These use token entropy for adaptivity but sacrifice public verifiability; STELA is a strict improvement by using POS entropy to retain both.
- **vs MorphMark**: MorphMark adapts at insertion (using green token cumulative probability) but remains uniform at detection; STELA is adaptive on both ends, resulting in denser detection signals.
- **vs Semantic-aware watermark (Guo et al. 2024)**: The latter uses LSH for clustering semantically similar tokens for rewriting robustness; STELA achieves similar goals through syntactic invariance at a lower cost (POS only).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Replacing token entropy with POS n-gram entropy" is a clean conceptual substitution, introducing syntactic predictability to watermarking with high originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 models × 3 languages × multiple baselines + synonym/Dipper attacks + word class contribution + tagset ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations, fair comparisons (calibrating $\delta$ via $\mathbb{E}[\lambda]$), and insightful interpretability analysis via typology and word-class decomposition.
- Value: ⭐⭐⭐⭐⭐ Under regulatory contexts like the EU AI Act, "publicly verifiable + adaptive strength" watermarking has direct regulatory value; the open-source code and low migration cost further enhance its utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](../../ICML2026/llm_safety/watermarking_llm_agent_trajectories.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)

</div>

<!-- RELATED:END -->
