---
title: >-
  [Paper Note] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages
description: >-
  [ICML 2026][Multilingual & Machine Translation][Low-resource languages] LiRA inserts a lightweight fine-tuning module featuring "anchoring + consistency regularization" between a frozen multilingual encoder and an Englis…
tags:
  - "ICML 2026"
  - "Multilingual & Machine Translation"
  - "Low-resource languages"
  - "cross-lingual alignment"
  - "anchored representation"
  - "translation noise"
  - "LLM adaptation"
date: 2026-05-08
content_hash: 90d4bd4745ab446f
---

# Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages

**Conference**: ICML 2026  
**arXiv**: [2510.14466](https://arxiv.org/abs/2510.14466)  
**Code**: Code is promised to be open-sourced; repository is currently pending release.  
**Area**: Multilingual Machine Translation / Cross-lingual Retrieval & Reasoning  
**Keywords**: Low-resource languages, cross-lingual alignment, anchored representation, translation noise, LLM adaptation

## TL;DR
LiRA inserts a lightweight fine-tuning module featuring "anchoring + consistency regularization" between a frozen multilingual encoder and an English LLM. It constrains the sentence vectors of low-resource languages into a shared English semantic space based on two theoretically controllable quantities: $\epsilon_1$ (anchoring error) and $\epsilon_2$ (translation KL divergence), achieving consistent improvements across retrieval, ranking, and reasoning tasks.

## Background & Motivation
**Background**: LLM capabilities are highly concentrated in high-resource languages like English and Chinese, but performance severely degrades for low-resource languages (e.g., Bengali, Indonesian, Burmese, Pashto, Thai, Filipino, Vietnamese). Current mainstream approaches follow two paths: the MT pipeline ("Translation → English Model Reasoning → Back-translation") or training multilingual encoders (mBERT, XLM-R, E5-Mistral) for language-agnostic alignment.

**Limitations of Prior Work**: MT pipelines accumulate translation noise and cause semantic drift, especially in multi-step reasoning queries. Although multilingual encoders are inherently cross-lingual, they lack the powerful reasoning heads of English LLMs and suffer from scarce training samples in low-resource domains. Recent works like MindMerger and LUSIFER attempt to bridge encoders with English LLMs, but the former depends on parallel corpora (inheriting translation noise), while the latter rarely sees low-resource languages during training, leading to unstable cross-lingual alignment during transfer.

**Key Challenge**: Errors in cross-lingual systems stem from two categories: representation space mapping errors (inconsistent endpoints of different encoding paths) and translation-induced semantic shifts. Existing methods neither systematically model how these errors propagate nor provide formal bounds driven by optimization objectives, resulting in robustness that relies on "empirical tuning."

**Goal**: (i) Establish a theoretical framework for cross-lingual adaptation that defines an upper bound for representation deviation, making both error types explicitly optimizable; (ii) Design a lightweight plugin compatible with different backbones that supports retrieval, ranking, and reasoning; (iii) Provide a multilingual evaluation set reflective of real-world low-resource scenarios (e.g., e-commerce product retrieval).

**Key Insight**: Visualize low-resource sentences $x$ and their English translations $y = T(x)$ as "two paths leading to the same English semantic space"—the multilingual anchoring path $g(x)$ and the English encoding path $h(y)$. Using the "ideal English representation" $\mathbf{z}^\star = [h(y^\star); h(y^\star)]$ as a mathematical reference frame, it can be proven that as long as the anchoring error $\epsilon_1$ and translation KL divergence $\epsilon_2$ decrease simultaneously, the pipeline output is bounded by the Lipschitz constant $L^{\text{loc}}$.

**Core Idea**: LiRA = Arca (utilizing critic-actor reinforcement learning to compress $\epsilon_1, \epsilon_2$) + LaSR (utilizing language-dependent heads for cross-lingual consistency regularization), translating theoretical upper bounds into two jointly optimizable losses.

## Method

### Overall Architecture
Input consists of a low-resource language sentence $x \in \mathcal{X}$. LiRA processes it through two parallel paths: Path 1 uses a multilingual encoder $g: \mathcal{X} \to \mathbb{R}^d$ to obtain the "anchored representation" $g(x)$; Path 2 first uses a lightweight LLM translator $T$ to obtain English $y = T(x)$, then uses an English encoder $h: \mathcal{Y} \to \mathbb{R}^d$ to obtain $h(y)$. The two representations are concatenated into $\mathbf{z}(x) = [g(x); h(y)] \in \mathbb{R}^{2d}$ and fed to a downstream LLM scorer $f_{\text{LLM}}$ for retrieval, ranking, or reasoning.

Theoretically, $\mathbf{z}$ is expected to approach the "ideal reference" $\mathbf{z}^\star = [h(y^\star); h(y^\star)]$ (where both paths consistently land at the ideal English translation $y^\star$). The paper proves that under the anchoring hypothesis $\|g(x) - h(y^\star)\|_2 \le \epsilon_1$ and translation fidelity hypothesis $D_{\text{KL}}(p(s|x) \| p(s|T(x))) \le \epsilon_2$, we have:
$$\|\mathbf{z} - \mathbf{z}^\star\|_2 \le \epsilon_1 + C\sqrt{2\epsilon_2},$$
consequently, $\|f_{\text{LLM}}(\mathbf{z}) - f_{\text{LLM}}(\mathbf{z}^\star)\|_2 \le L^{\text{loc}}(y;\delta)(\epsilon_1 + C\sqrt{2\epsilon_2})$. Arca and LaSR are the technical implementations pushing these $\epsilon$ values toward zero.

### Key Designs

1. **Arca — Anchored Representation Composition Architecture**:

    - **Function**: Synergistically reduces anchoring error $\epsilon_1$ and translation distortion $\epsilon_2$ via a "translation critic + embedding alignment critic + REINFORCE actor" setup.
    - **Mechanism**: Multilingual token streams $g_{\text{tok}}(x) \in \mathbb{R}^{L_x \times d_g}$ and English token streams $h_{\text{tok}}(y) \in \mathbb{R}^{L_y \times d_h}$ are first aligned to equal lengths via $S_{\text{feat}}$-bin temporal pooling, then mapped to a unified $d$-dimensional space via a shared Adaptor $A(\cdot)$ to obtain sentence vectors $E_{lr}$ and $E_{en}$. The anchoring loss is defined as cosine distance $\mathcal{L}_{\text{anchor}} = 1 - \cos(E_{lr}, E_{en})$, corresponding to $\epsilon_1$. Simultaneously, a lightweight LLM assigns a 3D score $(s_k, e_k, p_k) \in [1, 10]$ (semantic faithfulness, emotional consistency, pragmatic appropriateness) to $K$ candidate translations. These are combined with Adaptor similarity $\text{sim}_k$ into policy features $\mathbf{c}_k = [s_k, e_k, p_k, \text{sim}_k]^\top$, fed to an MLP to form policy $\pi_\phi(k | \mathbf{c}_{1:K}) = \text{softmax}(g_\phi(\mathbf{c}_{1:K}))$, and optimized via REINFORCE using composite reward $R_k = 0.1(\alpha s_k + \beta e_k + \gamma p_k) + \delta\, \text{sim}_k$. The objective $\mathcal{L} = \mathcal{L}_{\text{RL}} + \eta\, \mathcal{L}_{\text{anchor}}$ suppresses both $\epsilon_1$ and $\epsilon_2$.
    - **Design Motivation**: Explicitly decomposes the theoretical upper bound into differentiable/RL-optimizable terms; the critic-actor format enables feedback between translation selection and representation alignment, preventing supervised learning from being biased by translation noise.

2. **LaSR — Language-coupled Semantic Reasoner**:

    - **Function**: Serves as the downstream reasoning/retrieval head, enforcing convergence of "multilingual path output" and "English path output" in semantic space through consistency regularization, while supporting queue-based contrastive learning for stable large-batch retrieval.
    - **Mechanism**: Merges Arca outputs $(E_{lr}, E_{en})$ using a lightweight language-aware head to obtain a unified multilingual embedding. Contrastive loss (queue-based InfoNCE style) pulls multilingual versions and English versions of the same query closer while pushing negative samples away, effectively applying another level of regularization as $\mathcal{L}_{\text{consistency}}$. This embedding is shared across retrieval, ranking, QA, and reasoning.
    - **Design Motivation**: While Arca addresses the "mapping" problem, LaSR addresses "downstream consistency"—even with Arca, downstream LLM sensitivity might amplify residual errors. LaSR suppresses $L^{\text{loc}}(y;\delta)$, tightening the total upper bound defined in Corollary 2.

3. **Theory-driven two-stage training**:

    - **Function**: Decouples "representation deviation compression" and "downstream task adaptation" into two pluggable steps.
    - **Mechanism**: In Stage 1, multilingual and English encoder backbones are frozen, and only the Adaptor + critic + actor are trained to minimize $\mathcal{L}_{\text{anchor}} + \mathcal{L}_{\text{RL}}$. In Stage 2, LaSR is attached and fine-tuned for specific tasks (retrieval/ranking/reasoning) using contrastive and consistency losses. The module is plug-and-play, compatible with backbones like Qwen3-E, E5-Mistral, or BGE.
    - **Design Motivation**: Optimizing theoretical objectives ($\epsilon_1, \epsilon_2$) and engineering metrics (task metrics) in stages leverages pre-trained encoder capabilities while avoiding interference between objectives during end-to-end training.

### Loss & Training
- **Arca Stage**: $\mathcal{L} = \mathcal{L}_{\text{RL}} + \eta\, \mathcal{L}_{\text{anchor}}$, where $\mathcal{L}_{\text{RL}} \approx -\log \pi_\phi(a | \mathbf{c}_{1:K}) \cdot R_a$. Reward weights $(\alpha, \beta, \gamma, \delta)$ control the relative importance of 3D translation quality and embedding similarity.
- **LaSR Stage**: Queue-based contrastive loss + multilingual consistency regularization. Key hyperparameters include the number of pooling bins $S_{\text{feat}}$, neighborhood radius $\delta$, and Lipschitz quantile $q$ (default $q=0.95$, corresponding to $L^{(0.95)} \approx 0.034$).

## Key Experimental Results

### Main Results
The authors released LazRetrieval (a real-world e-commerce product retrieval set for 5 SEA + 2 South Asian languages) and compared various sentence encoders across 7 languages. All codes except LiRA-Large are public baselines; metrics represent average retrieval scores (higher is better).

| Method | Parameters | Bd | Id | My | Pk | Th | Ph | Vn | Avg |
|------|--------|------|------|------|------|------|------|------|------|
| Sentence-T5-XXL | 4.8B | 34.11 | 71.77 | 49.19 | 27.84 | 23.58 | 84.20 | 28.61 | 44.56 |
| GTR-XXL | 4.8B | 34.85 | 75.92 | 49.36 | 30.39 | 22.94 | 84.94 | 46.15 | 48.17 |
| Contriever | 110M | 39.95 | 74.95 | 48.90 | 35.71 | 15.43 | 83.74 | 64.75 | 51.00 |
| BGE-en-v1.5 | 335M | 41.06 | 78.78 | 53.28 | 37.52 | 18.35 | 88.18 | 68.72 | 54.09 |
| E5-Mistral-7B | 7.24B | 48.27 | 75.43 | 71.01 | 53.62 | 61.75 | 83.18 | 65.44 | 64.51 |
| Qwen3-E-0.6B | 0.6B | 38.36 | 63.95 | 62.37 | 40.73 | 55.28 | 74.38 | 59.46 | 56.31 |
| **LiRA-Large (Ours)** | 8.5B | **48.60** | 74.43 | **71.26** | 49.84 | **66.39** | 83.90 | **70.67** | **66.44** |

LiRA-Large outperforms E5-Mistral-7B by ~1.9 points on average and achieves SOTA on the four scarcest languages (Bd, My, Th, Vn).

### Ablation Study
Based on trends reported in the paper:

| Configuration | Retrieval Avg | Reasoning Task | Description |
|------|----------|----------|------|
| Full LiRA | 66.4 | best | Complete Arca + LaSR |
| w/o $\mathcal{L}_{\text{anchor}}$ | ↓ | ↓ | Anchoring loss removed; $\epsilon_1$ rebounds, cross-lingual alignment weakens |
| w/o RL critic | ↓↓ | ↓↓ | No translation quality driver; $\epsilon_2$ increases, low-resource performance drops significantly |
| w/o LaSR | ↓ | ↓↓ | Multilingual embeddings lack consistency regularization; reasoning suffers more than retrieval |
| Training data: No low-resource | ↓↓ | ↓↓ | Validates limitations of LUSIFER-style methods that omit low-resource samples |

### Key Findings
- The empirical measure of $\|\mathbf{z} - \mathbf{z}^\star\|$ monotonically decreases during training, aligning with the theoretical forecast of $\epsilon_1 + C\sqrt{2\epsilon_2}$—proving that abstract theoretical hypotheses are effectively implemented via engineering objectives.
- Under a token-edit neighborhood radius of $\delta = 1$, the 95th percentile Lipschitz constant $L^{(0.95)} \approx 0.034$, indicating the downstream LLM rarely amplifies errors from small representation perturbations.
- Relative gains of LiRA in long-tail languages like Burmese, Thai, and Vietnamese are much larger than in relatively high-resource languages like Indonesian, indicating that anchoring + consistency regularization provides maximum benefit in sample-sparse regions.
- Module swapping experiments show that LiRA consistently improves performance when combined with different backbones (Qwen3-E-0.6B / 4B), verifying its plug-and-play nature.

## Highlights & Insights
- Translating an engineering problem ("how to align low-resource languages") into two optimizable scalars $(\epsilon_1, \epsilon_2)$, and linking translation KL divergence with representation deviation via RKHS/KME, provides a "theoretical bound ↔ loss function" mapping applicable to any dual-stream alignment scenario (e.g., multi-modal alignment).
- Using an LLM as a translation scorer in a critic-actor framework essentially outsources multi-dimensional human quality assessment to a small LLM. This REINFORCE-based candidate selection is a reusable "unsupervised data augmentation" trick for distilling rewards.
- Concatenating multilingual and English paths rather than fusing them preserves distinct semantic perspectives, avoiding the loss of information caused by "forced averaging"—mirroring the philosophy of dual-stream multi-modal architectures.

## Limitations & Future Work
- The translator $T$ remains an external LLM; its capability determines the lower bound of $\epsilon_2$. If the target language is outside the translator's training distribution, the framework degrades.
- Theoretical assumptions require $f_{\text{LLM}}$ to be locally Lipschitz under representation perturbations, which might not hold for discrete decision scenarios like explicit tool calls or long-chain reasoning.
- LazRetrieval focuses primarily on the e-commerce product domain; cross-domain (legal, medical, government) validation is needed.
- $\epsilon_1$ and $\epsilon_2$ are upper bounds; a minimum learnable lower bound is not provided, suggesting performance may be approaching a bottleneck for certain language pairs.

## Related Work & Insights
- **vs MindMerger**: Both bridge multilingual encoders with English LLMs, but MindMerger relies heavily on parallel translation corpora (effectively minimizing empirical translation loss). LiRA uses critic-actor RL for candidate selection, bypassing the requirement for "clean" parallel data and explicitly modeling $\epsilon_2$.
- **vs LUSIFER**: LUSIFER connects encoders to LLM-based embeddings but lacks low-resource samples during training, making inference-time alignment fragile. LiRA includes low-resource samples during its two-stage training and explicitly optimizes cross-lingual mapping via anchoring loss.
- **vs XRAG / CCPR**: These work on end-to-end cross-lingual RAG evaluation or phrase-level retrieval; LiRA functions as a more foundational representation alignment layer that can serve as a front-end embedder for such tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizing cross-lingual alignment through RKHS + Lipschitz bounds and translating it into critic-actor RL is a rare "theory + engineering" closed loop.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 low-resource languages across multiple backbones and tasks, introducing the new LazRetrieval benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theory and method, with consistent cross-referencing between formulas and text.
- Value: ⭐⭐⭐⭐ Provides a robust front-end for low-resource NLP systems that is directly applicable to real-world scenarios like SEA/South Asian e-commerce.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](../../ACL2026/multilingual_mt/efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](../../ACL2026/multilingual_mt/reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](../../ACL2026/multilingual_mt/why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)
- [\[AAAI 2026\] ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation](../../AAAI2026/multilingual_mt/vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](../../ACL2026/multilingual_mt/mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)

</div>

<!-- RELATED:END -->
