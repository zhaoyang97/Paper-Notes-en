---
title: >-
  [Paper Note] Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG
description: >-
  [ICML 2026][Information Retrieval & RAG][RAG Poisoning] This paper argues that existing RAG poisoning attacks, while capable of manipulating LLM outputs with a few malicious passages…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "RAG Poisoning"
  - "Attention Analysis"
  - "Stealth Game"
  - "Poisoning Detection"
  - "Adaptive Attacks"
date: 2026-05-08
content_hash: 638c3a4f2ffaeddb
---

# Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG

**Conference**: ICML 2026  
**arXiv**: [2506.04390](https://arxiv.org/abs/2506.04390)  
**Code**: https://github.com/sarthak-choudhary/Stealthy_Attacks_Against_RAG (Available)  
**Area**: Information Retrieval / RAG Security / Retrieval Poisoning Defense  
**Keywords**: RAG Poisoning, Attention Analysis, Stealth Game, Poisoning Detection, Adaptive Attacks

## TL;DR
This paper argues that existing RAG poisoning attacks, while capable of manipulating LLM outputs with a few malicious passages, are **not truly stealthy**. Successful low-budget attacks inevitably cause the model to focus excessively on malicious passages. Consequently, the authors introduce the Normalized Passage Attention Score (NPAS) and the AV Filter based on its variance to filter out anomalous passages. Evaluated across 4 datasets × 5 LLMs × 5 attacks, this method improves RACC by up to 20% compared to Certified Robust RAG.

## Background & Motivation

**Background**: RAG compensates for LLMs' outdated knowledge and hallucinations by prepending the top-$k$ retrieved passages to the prompt. It serves as the foundation for systems like Google AI Overview, Bing, and Perplexity. However, the external knowledge base represents an open attack surface; an attacker can inject a few "malicious passages" into sources like Wikipedia or social media to be retrieved and manipulate generation. Works like PoisonedRAG demonstrate that corrupting just 1 out of 10 passages can force GPT-4 to output a specific target answer.

**Limitations of Prior Work**: Existing defenses fall into two categories: (1) **Passage-isolated** filtering (e.g., perplexity, vigilant prompting, reranking), which is largely ineffective against semantically fluent LLM-generated poisoning; (2) Certified Robust RAG (Xiang et al., 2024), which uses isolate-then-aggregate to provide an empirical upper bound but incurs a high cost in clean accuracy (ACC drops ~20% vs. Vanilla). Both fail to exploit the internal signal that "malicious passages are dominating the generation."

**Key Challenge**: Under a low corruption budget $\epsilon < 0.5$, for an attacker to make a few passages override the majority of benign ones, they **must** ensure these passages exert significantly more influence on LLM reasoning than benign ones—this is inherently at odds with "stealth." However, prior attacks have neither formalized "stealthiness" nor systematically detected it via internal model signals.

**Goal**: (i) Formally define a "stealthiness" metric for RAG poisoning to falsify existing claims; (ii) Design a lightweight, plug-and-play detection and filtering defense that does not require additional forward passes; (iii) Explore the robust lower bound of this signal through adaptive attacks.

**Key Insight**: During Transformer inference, **attention weights serve as a free proxy for token influence** (Vig & Belinkov 2019). If an attack successfully induces a target response $s'$, the tokens in $s'$ must allocate substantial attention to the malicious tokens containing or implying $s'$. This manifests at the passage level as a high-variance anomaly where a few passages "steal" excessive attention.

**Core Idea**: Treat the Normalized Passage Attention Score (NPAS) of each passage as a "proxy for passage influence on the response," use the **variance of NPAS across $k$ passages** as a statistical signature of poisoning, and implement the AV Filter to iteratively strip away the highest-scoring passages.

## Method

### Overall Architecture
The pipeline operates during the RAG generation phase (Step II):

Input: query $q$, retrieved top-$k$ passages $z^{(k)}$, LLM $\text{LLM}_\theta$, corruption budget $\epsilon$, variance threshold $\delta$.

Mechanism: (1) Perform a standard LLM forward pass and **reuse the attention matrices** (no extra compute); (2) Average multi-layer multi-head attention into a single matrix $A \in \mathbb{R}^{l \times T}$, where $l$ is the number of response tokens and $T$ is the number of input tokens; (3) For each passage $z_t$, calculate the sum of attention for its top-$\alpha$ attended tokens and normalize across passages to derive the NPAS; (4) If the NPAS variance $> \delta$, remove the passage with the highest NPAS and **restart from step (1)** until the variance falls below the threshold or $\lfloor \epsilon k \rfloor$ passages have been removed.

Output: The purified passage set $\tilde z$ is fed back to the LLM for final generation. The same NPAS can be used as a "discriminator" $\mathcal{D}_{\text{AV}}$ in the SADG game.

### Key Designs

1. **SADG (Stealth Attack Distinguishability Game) — Formalizing "Stealthiness" in a Cryptographic Style**:
    - **Function**: Provides a mathematical, adversarial game definition to quantify the advantage of any (attack, defense) combination.
    - **Core Idea**: An arbiter samples $q$ and constructs a benign set $z^{(k)}_{\text{benign}}$ and a poisoned set $z^{(k)}_{\text{corrupt}}$. These are **randomly shuffled** and sent to the defender to guess which is poisoned. The defender's advantage is defined as $\mathsf{Adv} = |\Pr[\text{win}] - 1/2|$. An attack is considered $\tau$-stealthy if $\mathsf{Adv} \le \tau$ for all PPT defenders.
    - **Design Motivation**: Previous papers used subjective criteria like "whether a human can spot malicious passages." By upgrading this to a game falsifiable by any detector, one can use defenses to determine the upper bound of an attack's stealthiness—the logical basis for the conclusion that "existing attacks are not stealthy."

2. **NPAS (Normalized Passage Attention Score) — Passage-level Influence Proxy**:
    - **Function**: Aggregates token-level attention to the passage level, creating an influence score that is invariant to passage length and comparable across queries/models.
    - **Core Idea**: Average all decoding layers and attention heads to get $A$. For passage $z_t$, the total attention of its top-$\alpha$ tokens ($\alpha \in \{5, 10, \infty\}$) in $A$ is taken as the raw score: $\mathsf{Score}_\alpha(z_t, A) = \sum_i \sum_{x_j \in \text{Top}_\alpha(z_t)} A[i,j]$. This is then normalized: $\mathsf{NormScore}_\alpha(z_t) = \mathsf{Score}_\alpha(z_t) / \sum_{i=1}^k \mathsf{Score}_\alpha(z_i)$.
    - **Design Motivation**: Raw token attention is too noisy; using top-$\alpha$ captures "Heavy Hitters" (often keywords containing the target answer) and mitigates length bias. Cross-passage normalization makes thresholds transferable. Variance serves as a robust global statistic—benign distributions are nearly uniform, while poisoned ones are right-skewed.

3. **AV Filter (Attention-Variance Filter) — Iterative Stripping + Reordering to Mitigate Positional Bias**:
    - **Function**: Filters suspicious passages within a budget of $\lfloor \epsilon k \rfloor$ removals without knowing which passage is malicious.
    - **Core Idea**: Passages are first reordered by NPAS to **eliminate recency bias** (where passages closer to the generation position naturally attract more attention). A `while` loop calculates the NPAS variance $\sigma^2$; if $\sigma^2 \le \delta$, it terminates early. Otherwise, it removes $\arg\max \mathsf{NormScore}$ and re-runs the forward pass. $\delta = 26.2$ is estimated from the RQA + Llama-2 clean set (mean + 1 std), **prioritizing low false negatives**.
    - **Design Motivation**: A single-pass score might be masked by the "second largest" passage; hence, iteration is necessary. Reordering counters positional biases observed in recent literature. Since the process reuses existing attention matrices, the inference overhead is minimal.

### Loss & Training
This is a **purely inference-time defense** requiring no LLM training. The threshold $\delta$ is estimated once on a single dataset/model and transferred. Adaptive attacks utilize GCG-style optimization to minimize the NPAS gap between malicious and benign passages.

## Key Experimental Results

### Main Results
Evaluation: 4 Datasets × 5 LLMs × 5 Attacks (Poison, MA, Paradox, CorruptRAG, PIA), $k=10$, $\epsilon=0.1$.

| Setting | Metric | Vanilla | Keyword (CR-RAG) | Decoding (CR-RAG) | AV Filter (Ours $\alpha=10$) |
|------|------|---------|------------------|-------------------|------------------------------|
| Mistral-7B / RQA-MC / Clean ACC | ↑ | 81.0 | 58.0 | 57.0 | **74.0** |
| Llama2-C / RQA-MC / Clean ACC | ↑ | 79.0 | 56.0 | 44.0 | **75.0** |
| Mistral-7B / RQA-MC / PIA | RACC↑ / ASR↓ | 59.6 / 31.0 | 57.0 / 7.0 | 55.0 / 5.0 | **77.2 / 6.0** |
| Llama2-C / RQA-MC / PIA | RACC↑ / ASR↓ | 33.4 / 63.0 | 54.0 / 6.0 | 38.0 / 12.0 | **(↑ ~20% vs Prev. SOTA)** |
| Avg SADG Win Rate (CIR) | ↑ | — | — | — | **0.78** |

**Key Findings**: AV Filter maintains **clean utility** effectively (avg. drop $\le 5\%$ vs. Vanilla), whereas CR-RAG drops 15-20%. Under PIA/Poison attacks, RACC is up to 20% higher than baselines.

### Ablation Study
| Configuration | Key Finding | Description |
|------|---------|------|
| $\alpha = 5 / 10 / \infty$ | Performance is similar; $\alpha=10$ is slightly better. | $\alpha$ should match the number of Heavy Hitters; too small misses signals, too large adds noise. |
| No Reordering vs. Reordering | Recency bias increases false removals at positions 9-10 without reordering. | Validates positional bias as a real issue; reordering is a necessary engineering step. |
| Single NPAS vs. Iterative AV Filter | Iteration is significantly more stable for multiple poisoned passages ($\epsilon=0.2$). | Single scoring can be masked by the "runner-up" anomalous passage. |
| Black-box (GPT-4o + Mistral aux) | SADG advantage drops from 0.78 → ~0.65 but remains significantly above 0.5. | Attention signals remain discriminative even on auxiliary models. |
| Adaptive Attack vs. AV Filter | Post-optimization ASR reaches 35% (still < Vanilla). | Requires $\sim 10^3\times$ baseline time and knowledge of benign passages, making it impractical. |

### Key Findings
- **NPAS is nearly uniform on clean sets but spikes to 30%+ for poisoned passages**, showing a clear right-shift in variance distribution.
- The threshold $\delta$ estimated on RQA + Llama-2 transfers well across settings, indicating that normalization successfully handles scale consistency.
- **Black-box availability**: Even for closed models like GPT-4o, using Mistral-7B as a sidecar to calculate NPAS allows AV Filter to function effectively.

## Highlights & Insights
- **Upgrading stealthiness from intuition to a cryptographic game**: SADG allows any attack's stealth claim to be falsified by any detector.
- **Attention as a free defense signal**: Reusing existing LLM forward pass attention matrices requires no additional training and has near-zero deployment cost.
- **Honest presentation of adaptive attacks**: Rather than hiding weaknesses, the paper quantifies the $10^3\times$ cost required for adaptive attacks to reach 35% ASR, clarifying the boundaries of the security "arms race."

## Limitations & Future Work
- **Reliance on benign-majority + redundancy**: Assumption 3.2 requires at least 2 benign passages to support the correct answer. The defense fails if the knowledge base has poor coverage.
- **Information-theoretic constraint $\epsilon < 0.5$**: The authors state that cases with majority corruption are theoretically unsolvable, the same boundary as Certified Robust RAG.
- **Incomplete Signal**: 35% ASR under adaptive attacks shows NPAS isn't the final answer. Future work might explore attention rollout or multi-signal integration (hidden states).

## Related Work & Insights
- **vs. Certified Robust RAG (Xiang et al., 2024)**: They isolate each passage for independent generation, providing an empirical bound but losing ~20% clean ACC. Ours performs **joint attention analysis**, maintaining clean ACC while boosting RACC by 20%.
- **vs. Perplexity Filter (Jain et al., 2023)**: Perplexity is an isolated score, ineffective against fluent poisoning. NPAS is a "passage-response joint score" that captures high-influence malicious passages.

## Rating
- Novelty: ⭐⭐⭐⭐ SADG formalization and using attention variance as a poisoning signature are systematic firsts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across datasets, models, and adaptive scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions and algorithms; symbols are slightly dense.
- Value: ⭐⭐⭐⭐⭐ High engineering value for production RAG systems due to zero-training and black-box compatibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](../../ACL2026/information_retrieval/disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](../../ACL2026/information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[ICML 2026\] Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)
- [\[ICLR 2026\] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](../../ICLR2026/information_retrieval/bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)

</div>

<!-- RELATED:END -->
