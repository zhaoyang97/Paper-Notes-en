---
title: >-
  [Paper Note] Optimizing Language Models for Crosslingual Knowledge Consistency
description: >-
  [ICML 2026][Multilingual & Machine Translation][Crosslingual Consistency] This paper addresses the issue where multilingual LLMs provide conflicting answers to the same question in different languages. It proposes an RL…
tags:
  - "ICML 2026"
  - "Multilingual & Machine Translation"
  - "Crosslingual Consistency"
  - "DCO"
  - "DPO Variant"
  - "Product of Experts"
  - "RankC"
date: 2026-05-08
content_hash: aad4c14f6e3d3f33
---

# Optimizing Language Models for Crosslingual Knowledge Consistency

**Conference**: ICML 2026  
**arXiv**: [2603.04678](https://arxiv.org/abs/2603.04678)  
**Code**: [github.com/Betswish/ConsistencyRL](https://github.com/Betswish/ConsistencyRL)  
**Area**: Reinforcement Learning / Multilingual LLM / Preference Alignment  
**Keywords**: Crosslingual Consistency, DCO, DPO Variant, Product of Experts, RankC

## TL;DR
This paper addresses the issue where multilingual LLMs provide conflicting answers to the same question in different languages. It proposes an RL objective that uses the "log-likelihood of the answer in another language" as the reward, proves that the optimal policy takes a product-of-experts form and guarantees crosslingual preference consistency when $\gamma_1\gamma_2=\beta^2$. Based on this, it derives the **Direct Consistency Optimization (DCO)** algorithm, which requires neither a reward model nor online sampling. DCO improves both crosslingual consistency (RankC) and answer accuracy across 9 LLMs, 3 multilingual QA benchmarks, and 26 languages.

## Background & Motivation

**Background**: Modern LLMs (Llama, Qwen, Aya, Gemma, etc.) claim multilingual capabilities, but **often provide conflicting answers to the same question when asked in different languages**. After qi-etal-2023-cross introduced the RankC metric, crosslingual knowledge inconsistency (Crosslingual Consistency, CLC) has become a standard evaluation for multilingual LLMs.

**Limitations of Prior Work**: (1) Interpretability-based interventions (vector editing, representation alignment) are only validated on small data/specific models and are hard to scale. (2) CALM (wang-etal-2025) applies DPO to the "winner" selected by multilingual majority voting, but requires >2 languages and fails in bilingual scenarios; adding low-resource languages distorts majority voting. (3) There is no objective function that theoretically guarantees "optimal strategies are always consistent".

**Key Challenge**: DPO's Bradley-Terry preference modeling is inherently "winner vs loser in a single language", and **there is no direct way to express the second-order constraint that "winner/loser ranking should be consistent across languages"**; forcing preference pairs can also harm original accuracy in post-train languages.

**Goal**: 1) Provide a formal definition of CLC (preference rank invariance across languages); 2) Design a reward that directly drives RL to converge to a consistent policy; 3) Derive an efficient algorithm requiring no online sampling or reward model; 4) Validate across multiple models and benchmarks.

**Key Insight**: The authors move away from the "find a crosslingual winner" paradigm and instead use "reward in language A = log-likelihood of the answer translated to language B under the original model"—this "scoring with the partner language's likelihood" design **naturally embeds crosslingual symmetry in the dual form of the optimal policy**.

**Core Idea**: By defining the reward $r_{\text{align}}$ using the other language's $\log\pi_{\text{ref}}(\tau(\mathbf y)|\tau(\mathbf x))$, the optimal policy of KL-regularized RL is a **crosslingual product of experts**; as long as the hyperparameter constraint $\gamma_1\gamma_2=\beta^2$ is satisfied, the optimal policy maintains consistent preference ranking between the two languages and can be solved DPO-style without a reward model.

## Method

### Overall Architecture
The method consists of three layers: (1) **CLC Formalization (Def 1)**: Model $\pi^\star$ is consistent on $L_1, L_2$ ⟺ for any pair of translation-equivalent answers $(\mathbf y_w^1, \mathbf y_l^1) \sim (\mathbf y_w^2, \mathbf y_l^2)$, the preference order is consistent across the two languages. (2) **Structured Reward and Optimal Policy**: Define a piecewise reward (Eq.8), solve KL-regularized RL, obtain the product-of-experts form of the optimal policy (Eq.9), and prove that $\gamma_1\gamma_2 = \beta^2$ is a sufficient condition for consistency (Lemma 1). (3) **DCO Algorithm**: Transform reward matching (Eq.10) into a DPO-style difference objective, avoiding the need for a reward model and online sampling, and directly train on parallel prompt/response dataset $\mathcal D_\|$.

### Key Designs

1. **Structured Reward and Crosslingual Duality (Eqs. 7-9)**:

    - Function: Expresses "the preference for answers in this language should be consistent with that in the other language" using a single reward.
    - Mechanism: Define the piecewise reward
      $r_{\text{align}}(\mathbf x, \mathbf y) = \gamma_i \log\pi_{\text{ref}}(\tau^j(\mathbf y)|\tau^j(\mathbf x))$ when $\mathbf x, \mathbf y \in L_i$ ($j\ne i$).
      According to Rafailov's KL-regularized RL optimal policy formula,
      $\pi^\star(\mathbf y^1|\mathbf x^1) \propto \pi_{\text{ref}}(\mathbf y^1|\mathbf x^1) \cdot \pi_{\text{ref}}^{\gamma_1/\beta}(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$, which is a **product of experts**—multiplying the original likelihood in the current language with the translated likelihood in the other language.
      By the rearrangement inequality, maximizing the reward is equivalent to monotonic alignment between $\{\pi_\theta(\mathbf y|\mathbf x)\}_y$ and $\{r_{\text{align}}(\mathbf x, \mathbf y)\}_y$, which matches Def 1's consistency.
    - Design Motivation: The authors seek a **formal guarantee**: "the optimal solution of this reward is always consistent", not just a heuristically plausible approach. The product-of-experts form preserves base model knowledge (avoiding performance drop) while enforcing crosslingual dual constraints.

2. **Hyperparameter Constraint $\gamma_1\gamma_2 = \beta^2$ and NN Language Extension (Lemma 1)**:

    - Function: Selects the unique subset of $\beta, \gamma_1, \gamma_2$ combinations that guarantee consistency.
    - Mechanism: Taking both sides of Eq.9a to the $\beta/\gamma_1$ power, it can be rewritten as $(\pi^\star(\mathbf y^1|\mathbf x^1))^{\beta/\gamma_1} \propto \pi^\star(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$; since $x \mapsto cx^{\beta/\gamma_1}$ is monotonic, preference order is consistent across languages. $\gamma_1, \gamma_2$ control the deviation from $\pi_{\text{ref}}$ in each language (smaller $\gamma$ = closer to the original model), $\beta$ controls overall KL deviation. For $N$ languages, introduce $N^2 - N$ $\gamma_{ij}$ to control pairwise alignment strength, with corresponding constraints for consistency (see Appendix E).
    - Design Motivation: In practice, it is desirable to "tune" which language is more tightly aligned (e.g., low-resource languages may want to stay closer to the original model to avoid being "dragged" by high-resource languages); the $\gamma_{ij}$ design provides a controllable knob. The $\gamma_1\gamma_2=\beta^2$ constraint also simplifies implementation (just pick a valid set, e.g., $\gamma_1=\gamma_2=\beta$).

3. **DCO Algorithm: No Reward Model, No Online Sampling (Eq. 10)**:

    - Function: Converts the above RL objective into an offline objective for direct gradient descent on model parameters $\theta$.
    - Mechanism: Following DPO, use $\hat r_\theta(\mathbf x, \mathbf y) = \beta\log\frac{\pi_\theta(\mathbf y|\mathbf x)}{\pi_{\text{ref}}(\mathbf y|\mathbf x)}$ to reparameterize the reward, and match the difference of $\hat r_\theta$ to that of $r_{\text{align}}$:
      $L(\theta) = \mathbb E\big[\|d_\theta^1 - \gamma_1\log\frac{\pi_{\text{ref}}(\mathbf y_w^2|\mathbf x^2)}{\pi_{\text{ref}}(\mathbf y_l^2|\mathbf x^2)}\| + \|d_\theta^2 - \gamma_2\log\frac{\pi_{\text{ref}}(\mathbf y_w^1|\mathbf x^1)}{\pi_{\text{ref}}(\mathbf y_l^1|\mathbf x^1)}\|\big]$,
      where $d_\theta^i = \hat r_\theta(\mathbf x^i, \mathbf y_w^i) - \hat r_\theta(\mathbf x^i, \mathbf y_l^i)$; thus: (a) winner/loser labels **do not require ground truth**, random pairing suffices; (b) no need to train a reward model; (c) fully offline, only runs on parallel prompt-response data $\mathcal D_\|$, each sample is a translation pair $(\mathbf x^1, \mathbf y^1, \mathbf x^2, \mathbf y^2)$. Lemma 2 proves that the optimal $\hat r_\theta^\star$ converges to $r_{\text{align}}$ plus a constant $c(\mathbf x)$ independent of $\mathbf y$ (which does not affect the policy).
    - Design Motivation: Like DPO, uses the "difference form to eliminate the partition function $Z(\mathbf x)$" trick, but the objective shifts from "matching human preferences" to "matching crosslingual consistency reward"; this makes the training pipeline fully compatible with existing DPO frameworks and lightweight in engineering.

### Loss & Training
Nine LLMs (Qwen2.5-7B/14B, Qwen3-8B/14B, Aya-Expanse-8B, Llama3.1-8B, Llama3.2-3B, Gemma3-4B/12B), three parallel QA datasets (MMMLU 14 languages, XCSQA 16 languages, BMLAMA 17 languages), totaling 26 languages. DCO loss (Eq. 10) is used for training on parallel prompt-response pairs.

## Key Experimental Results

### Main Results
MMMLU multilingual joint training (clc_all = average RankC over all language pairs; a_en / a_¬en = English/non-English accuracy), relative to the base model:

| Model | Method | $\Delta$clc_all | $\Delta$a_en | $\Delta$a_¬en |
|-------|--------|-----------------|--------------|---------------|
| Qwen2.5-14B | Base = 68.6 / 72.5 / 58.1 | — | — | — |
| Qwen2.5-14B | + SFT* | +0.6 | +1.5 | +6.7 |
| Qwen3-14B | + SFT* | -0.2 | +0.1 | +0.5 |
| Aya-Expanse-8B | + SFT* | +3.5 | +0.7 | — |
| Llama3.1-8B | + SFT* | — | — | — |

(The full Table 1 in the paper also includes +DPO, +CALM, and **+DCO** rows—DCO consistently outperforms other methods on clc_all, with accuracy maintained or slightly improved; see the original Table 1 for detailed numbers.)

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|---------------|---------------|-------------|
| DCO vs SFT* | DCO significantly higher on RankC | SFT only optimizes gold answer, does not address crosslingual ranking |
| DCO vs DPO* | DCO achieves higher crosslingual consistency, does not rely on gold label | DCO uses parallel pairs instead of preference pairs |
| DCO vs CALM | CALM degrades when low-resource languages are added, DCO remains stable | Majority voting is not the key |
| DCO + DPO combination | When gold label is available, DCO and DPO are complementary | Different objectives solve different subproblems |
| Bilingual training | DCO is effective, CALM fails | DCO does not require ≥3 languages |
| OOD generalization | RankC improves even on unseen domains | The learned structure is consistency, not specific knowledge |
| $\gamma_1 \ne \gamma_2$ controls language bias | Can bias toward a language to preserve original performance | Engineering controllability |

### Key Findings
- **DCO improves consistency without harming single-language accuracy**—this is the key advantage over DPO: DPO often sacrifices performance in post-train languages for alignment.
- Asymmetric $\gamma_1/\gamma_2$ enables "directional alignment": high-resource languages can more tightly pull low-resource languages while preserving their own performance.
- Good cross-domain generalization: consistency patterns trained on MMMLU transfer to XCSQA and BMLAMA.

## Highlights & Insights
- Reformulating CLC consistency as "likelihood in another language as reward" yields a product-of-experts form with mathematical elegance (rearrangement inequality directly proves consistency) and engineering advantages (compatible with DPO pipeline).
- **No gold label required for training**: random winner/loser pairing still works, as the difference form only cares about "whether reward differences are consistent across languages", reducing data requirements to just parallel translations.
- The elegant algebraic constraint $\gamma_1\gamma_2 = \beta^2$ clarifies which hyperparameter combinations guarantee consistency—a rare case where theory provides hyperparameter selection guidance.

## Limitations & Future Work
- Evaluation depends on the **existence of a translation mapping $\tau$**—the paper is limited to factual QA tasks with "finite, objectively translatable answers"; for open-ended generation (creative writing, summarization), the definition of "consistency" is itself ambiguous, and DCO is not directly applicable.
- The quality of translations in parallel datasets (MMMLU/XCSQA/BMLAMA) affects training; translation noise in low-resource languages may distort rewards.
- Lemma 1's $\gamma_1\gamma_2=\beta^2$ is a sufficient but not necessary condition; whether a looser valid region exists is unexplored.
- Interaction with chain-of-thought reasoning is not discussed; crosslingual consistency is harder in CoT as intermediate steps must also align.
- Computational cost: each sample requires forward passes in both languages, doubling the cost compared to monolingual DPO.

## Related Work & Insights
- **vs DPO (Rafailov et al. 2023)**: DCO replaces DPO's "matching human preferences" with "matching crosslingual consistency reward", retains the difference trick to eliminate the partition function, but the objectives are fundamentally different—one aligns preferences, the other aligns across languages.
- **vs CALM (wang-etal-2025)**: CALM requires ≥3 languages for majority voting to find the "winner" before DPO; DCO trains directly on parallel pairs, works for bilingual, and does not degrade with low-resource languages.
- **vs Representation Intervention Methods (Lu, Wang, Liu)**: DCO does not require white-box access to hidden states, uses only likelihood signals, and is easy to scale.
- **vs RankC Evaluation (qi-etal-2023-cross)**: This is among the first works to use RankC as an RL training objective rather than just for evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating crosslingual consistency as an RL reward and deriving a DPO-style offline algorithm is a genuinely new construction.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 models × 3 datasets × 26 languages, with OOD/bilingual/control experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivations, complete proofs for Lemma 1/2, complex notation but clear logic.
- Value: ⭐⭐⭐⭐ Directly valuable for multilingual LLM deployment, and complementary to DPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](../../ACL2026/multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](../../ACL2026/multilingual_mt/language_models_entangle_language_and_culture.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](../../ACL2026/multilingual_mt/multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](../../ACL2026/multilingual_mt/efficient_training_for_cross-lingual_speech_language_models.md)

</div>

<!-- RELATED:END -->
