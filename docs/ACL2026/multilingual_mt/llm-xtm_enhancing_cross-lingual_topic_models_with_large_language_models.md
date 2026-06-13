---
title: >-
  [Paper Note] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Cross-Lingual Topic Model] A two-stage enhancement module consisting of "LLM refinement + self-consistency voting + MMD word distribution alignment + QA-style document seman…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Cross-Lingual Topic Model"
  - "LLM Refinement"
  - "Self-Consistency"
  - "MMD Alignment"
  - "QA-style Document Alignment"
date: 2026-05-08
content_hash: dff537d025409592
---

# LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2605.03299](https://arxiv.org/abs/2605.03299)  
**Code**: https://github.com/tienphat140205/LLM-XTM (Available)  
**Area**: Multilingual / Topic Modeling  
**Keywords**: Cross-Lingual Topic Model, LLM Refinement, Self-Consistency, MMD Alignment, QA-style Document Alignment

## TL;DR
A two-stage enhancement module consisting of "LLM refinement + self-consistency voting + MMD word distribution alignment + QA-style document semantic alignment" is wrapped around pre-trained cross-lingual topic models. It serves as a plugin for various backbones such as NMTM, InfoCTM, and XTRA. Across three bilingual corpora (EC News, Amazon Review, Rakuten Amazon), it improves CNPMI by 9%–51% and TQ by 6%–44%, while reducing LLM calls to "once every $f$ epochs."

## Background & Motivation

**Background**: The goal of Cross-Lingual Topic Modeling (CLTM) is to extract "semantically corresponding" topic pairs from multilingual corpora, where high-frequency words for the same topic across different languages (e.g., English/Chinese/Japanese) are semantically consistent. Prevailing methods (MCTA, MTAnchor, NMTM, InfoCTM, XTRA, etc.) largely rely on external bilingual resources: parallel corpora, seed dictionaries, bilingual embeddings, or anchor words.

**Limitations of Prior Work**: Low bilingual dictionary coverage and noise in parallel corpora (mistranslations, domain shifts, lexical ambiguity) lead to "nominally aligned" topics drifting semantically. Table 1 provides a startling example where InfoCTM pairs the English words `rating/gauge/height/mile/shoe` with Chinese financial terms `investors/finance/funds/stock market/index`, which are completely unrelated.

**Key Challenge**: Shallow corpus-driven signals cannot characterize deep cross-lingual semantic consistency, whereas LLMs possess deep semantic priors from massive multilingual pre-training. Existing LLM-based approaches suffer from three issues: (1) treating LLM outputs as ground-truth for independent document calls, ignoring global structure and incurring high costs; (2) LLM instability and hallucinations; (3) requirements for token probabilities in white-box solutions like LLM-in-the-Loop, which are unavailable for closed-source models (e.g., Gemini/Claude).

**Goal**: Inject LLM semantic knowledge into both the **topic-word distribution** $\beta$ and **document-topic distribution** $\theta$ with minimal LLM calls, while ensuring (a) black-box availability, (b) robustness to hallucinations, and (c) preservation of the backbone's existing corpus-driven signals.

**Key Insight**: Drawing from the "self-consistency as uncertainty measure" idea in SelfCheckGPT, the authors sample LLM outputs multiple times, retaining high-consistency words and discarding low-consistency ones to filter hallucinations via voting. Simultaneously, the task of "assigning a document to a topic" is re-interpreted as a QA-style matching where the document is the "question" and the refined topic word set is the "candidate answer," using a multilingual encoder (BGE-M3) to calculate cosine similarity.

**Core Idea**: Wrap LLM refinement as a "periodic, self-consistent voting" black-box process. The output is aligned with the original $\beta$ via MMD and pulls $\theta$ toward a semantic target $\hat{\theta}$ calculated by BGE-M3 using KL divergence. This "softly guides" the backbone toward more coherent and cross-lingually aligned solutions without disrupting its original loss functions.

## Method

### Overall Architecture
LLM-XTM is a **two-phase post-processing enhancement**:
- **Phase 1**: Run an off-the-shelf VAE-based CLTM backbone (NMTM / InfoCTM / XTRA) using its original loss $\mathcal{L}_{\text{Phase 1}}$ until convergence to obtain $\beta^{(\ell)}$ and $\theta_d$.
- **Phase 2**: Train the converged model for an additional 30 epochs. Every $f$ epochs, trigger LLM refinement and inject the results into the backbone via two external losses:

The total objective is $\mathcal{J}(\phi, \psi) = \mathcal{L}_{\text{Phase 1}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}} + \lambda_{\text{qa}} \mathcal{L}_{\text{doc-align}}$. The workflow within an epoch involves: sampling top-15 bilingual words $\rightarrow$ calling LLM for voting to get $\bar{w}_k$ $\rightarrow$ calculating $\mathcal{L}_{\text{MMD}}$ $\rightarrow$ encoding documents/topics with BGE-M3 $\rightarrow$ calculating KL-based $\mathcal{L}_{\text{doc-align}}$ $\rightarrow$ updating the backbone. The LLM module remains a black box to the backbone, requiring no token probabilities.

### Key Designs

1.  **Self-Consistent Cross-Lingual Topic Word Refinement**:
    - **Function**: Merges top-15 English and top-15 Chinese words from the backbone into a candidate pool $C_k = w_k^{(\text{en})} \cup w_k^{(\text{zh})}$. The LLM removes noise, fills gaps, and retains common topic words, outputting a refined set $\bar{w}_k$ (15 words per language).
    - **Mechanism**: Due to LLM output variance, the same prompt is run $R$ times to obtain $\tilde{w}_k^{(1)}, \dots, \tilde{w}_k^{(R)}$. The hit frequency for each word is calculated as $f_k(v) = \frac{1}{R}\sum_{r=1}^R \mathbf{1}\{v \in \tilde{w}_k^{(r)}\}$, and the top-$M$ high-frequency words form the final $\bar{w}_k$. A "refinement frequency" hyperparameter $f$ is introduced to call the LLM only every $f$ epochs (rather than every step), reducing costs by an order of magnitude.
    - **Design Motivation**: Consistency across multiple samplings is a strong signal for detecting hallucinations. High-consistency words are likely core topic words, whereas low-consistency words are often fabricated. This replicates uncertainty estimation without requiring logit access.

2.  **MMD Topic-Word Distribution Alignment (MMD Refinement Loss)**:
    - **Function**: Pulls the backbone's original $\beta_k^{(\text{raw})}$ toward the LLM-derived target distribution $\beta_k^{(\text{refined})}$, without losing corpus-driven signals.
    - **Mechanism**: For each topic $k$, two distributions are constructed: the "raw" distribution from decoder top-$N$ word probabilities and the "refined" distribution from $R$-round voting counts. Both are language-balanced and normalized. Squared MMD is calculated in the BGE-M3 embedding space using a Gaussian kernel (with median heuristic bandwidth): $\mathcal{L}_{\text{MMD}} = \frac{1}{K}\sum_{k=1}^{K} \text{MMD}^2(\beta_k^{\text{(raw)}}, \beta_k^{\text{(refined)}})$.
    - **Design Motivation**: MMD significantly outperformed Optimal Transport (OT) (CNPMI 0.016 vs. 0.013). Kernel methods naturally treat synonym replacement as a match via embedding similarity, whereas OT is more sensitive to word identity. MMD provides "soft alignment" that is gentler than hard replacement and does not disrupt the reconstruction loss.

3.  **QA-style Document-Topic Alignment (Document-Topic Alignment via QA)**:
    - **Function**: Ensures documents with the same semantics in different languages have similar document-topic distributions $\theta_d$.
    - **Mechanism**: Documents are treated as questions and refined topics as candidate answers. BGE-M3 encodes documents into $h_d$ and $\bar{w}_k$ into topic vectors $t_k = \text{Enc}(\bar{w}_k)$. Cosine similarity $s_{d,k}$ is computed, followed by a softmax with temperature $\tau$ to obtain target $\hat{\theta}_{d,k} = \frac{\exp(s_{d,k}/\tau)}{\sum_j \exp(s_{d,j}/\tau)}$. The backbone's $\theta_d$ is pulled toward this target using KL divergence: $\mathcal{L}_{\text{doc-align}} = \sum_{d=1}^D \text{KL}(\theta_d \| \hat{\theta}_d)$.
    - **Design Motivation**: BoW is naturally incomparable across languages. However, multilingual sentence embeddings bridge this gap. Using semantic similarity as external supervision forces the backbone to learn cross-lingually consistent document representations.

### Loss & Training
The total Phase 2 objective is $\mathcal{J} = \mathcal{L}_{\text{Phase 1}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{MMD}} + \lambda_{\text{qa}} \mathcal{L}_{\text{doc-align}}$. Experiments used $\lambda_{\text{mmd}} = 20,000$, $\lambda_{\text{qa}} \in \{100, 200, 300\}$, $f \in \{8, 10\}$, and $R = 5$. The LLM components use the Gemini API, and Phase 2 completes in 30 epochs on a single NVIDIA P100.

## Key Experimental Results

### Main Results
Evaluated on EC News (EN-ZH), Amazon Review (EN-ZH), and Rakuten Amazon (JA-EN) benchmarks using CNPMI (cross-lingual coherence), TU (topic uniqueness), and TQ = max(0, CNPMI) × TU:

| Backbone | Dataset | CNPMI (base $\rightarrow$ +LLM-XTM) | TQ Gain | Notes |
|----------|---------|------------------------------------|---------|-------|
| XTRA | EC News | 0.078 $\rightarrow$ 0.088 | +10.5% | TU slightly down 2.5% |
| XTRA | Amazon Review | 0.053 $\rightarrow$ 0.072 | +32.7% | CNPMI +35.8% |
| InfoCTM | EC News | 0.041 $\rightarrow$ 0.062 | +43.6% | CNPMI +51.2% |
| InfoCTM | Amazon Review | 0.037 $\rightarrow$ 0.050 | +38.2% | TU up 0.3% |
| NMTM | Amazon Review | 0.043 $\rightarrow$ 0.056 | +34.6% | CNPMI +30.2% |
| NMTM | Rakuten Amazon | 0.012 $\rightarrow$ 0.016 | +37.5% | CNPMI +33.3% |

Across 9 combinations, CNPMI gains were consistently positive while TU volatility remained within ±5%, proving LLM-XTM is a **universal enhancement layer**. On long-document benchmarks (Airiti Thesis), TQ increased by up to +121.3%. In downstream document classification, cross-lingual accuracy (-C) improved significantly (e.g., 0.734 $\rightarrow$ 0.788 for InfoCTM on Rakuten Amazon).

### Ablation Study
On NMTM + Rakuten Amazon (50 topics):

| Configuration | CNPMI | TU | EN-C | JA-C | Description |
|---------------|-------|----|------|------|-------------|
| NMTM (base) | 0.012 | 0.633 | 0.610 | 0.681 | Backbone |
| Full LLM-XTM | 0.016 | 0.666 | 0.621 | 0.728 | All components |
| w/o $\mathcal{L}_{\text{doc-align}}$ | 0.012 | 0.679 | 0.611 | 0.723 | CNPMI drops to base |
| w/o $\mathcal{L}_{\text{MMD}}$ | 0.012 | 0.641 | 0.621 | 0.723 | TU drops by 0.025 |
| w/o self-consistency | 0.011 | 0.654 | 0.619 | 0.720 | CNPMI falls below base |
| MMD $\rightarrow$ OT | 0.013 | 0.664 | 0.620 | 0.720 | OT is 18.7% worse in CNPMI |

### Key Findings
- **$\mathcal{L}_{\text{doc-align}}$ is crucial for alignment**: Removing it causes CNPMI to drop to backbone levels, showing topic-word alignment alone cannot constrain document-level consistency.
- **Self-consistency is indispensable**: Without voting, CNPMI falls below the base (0.011 < 0.012), as single-call hallucinations contaminate the backbone.
- **MMD Over OT**: Kernel methods are more tolerant of synonym replacements in embedding space.
- **Hyperparameter Sensitivity**: $R \in [5, 7]$ is the sweet spot for voting rounds. Frequent refinement ($f$ small) increases CNPMI but can lower TU.

## Highlights & Insights
- **Dual control of cost and hallucinations**: Periodic calls ($f$) combined with self-consistent voting ($R$) transform the LLM from a per-document oracle into a periodic expert consultant.
- **Soft guidance via MMD**: Using MMD with Gaussian kernels in embedding space recognizes synonymy, allowing the backbone to maintain its reconstruction signals while being guided toward semantic coherence.
- **QA Paradigm for Alignment**: Reframing document-topic distribution as a retrieval task allows multilingual sentence encoders like BGE-M3 to act as a bridge, bypassing the incomparability of BoW across languages.
- **True Plugin Architecture**: Consistent gains across multiple backbones and datasets indicate this post-processing paradigm is more economical than designing new backbones from scratch.

## Limitations & Future Work
- **Backbone Dependency**: LLM-XTM is a refinement tool and cannot create reasonable topics from a completely failed initialization.
- **API Cost and Latency**: Despite periodic calls, $R=5$ voting still involves multiple calls per topic, which may be expensive for large-scale corpora or many topics.
- **Language Scope**: Validated only on EN-ZH and EN-JA pairs. Performance on typologically distant or low-resource pairs depends heavily on the quality of the multilingual encoder.
- **Encoder Dependency**: Results may vary with different encoders; future work could investigate distilling BGE-M3 into smaller, more efficient models.

## Related Work & Insights
- **vs. LLM-ITL**: LLM-ITL requires white-box token probabilities for uncertainty estimation; LLM-XTM uses black-box sampling/voting.
- **vs. TopicGPT**: TopicGPT calls LLMs per document (unscalable); LLM-XTM calls LLMs per topic, decoupling cost from corpus size.
- **vs. XTRA**: XTRA uses contrastive learning for alignment; LLM-XTM is complementary, providing further gains (+10.5% TQ) when stacked on top.
- **Insight**: The MMD + QA paradigm can be generalized to refine any latent distribution in generative models using LLMs, such as item embeddings in recommendation systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of self-consistency, MMD, and QA into a black-box plugin is innovative; the QA-style $\theta$ alignment is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across backbones, datasets, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly explained with convincing qualitative examples.
- Value: ⭐⭐⭐⭐ Provides a highly effective, ready-to-use plugin for the CLTM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)
- [\[ACL 2026\] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)

</div>

<!-- RELATED:END -->
