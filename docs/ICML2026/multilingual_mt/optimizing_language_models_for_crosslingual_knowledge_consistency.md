---
title: >-
  [Paper Note] Optimizing Language Models for Crosslingual Knowledge Consistency
description: >-
  [ICML 2026][Multilingual & Machine Translation][Crosslingual Consistency] This paper addresses the issue of multilingual LLMs providing conflicting answers to the same question across different languages. It proposes an…
tags:
  - "ICML 2026"
  - "Multilingual & Machine Translation"
  - "Crosslingual Consistency"
  - "DCO"
  - "DPO variants"
  - "product of experts"
  - "RankC"
date: 2026-05-08
content_hash: 7d690388c7f34bb1
---

# Optimizing Language Models for Crosslingual Knowledge Consistency

**Conference**: ICML 2026  
**arXiv**: [2603.04678](https://arxiv.org/abs/2603.04678)  
**Code**: [github.com/Betswish/ConsistencyRL](https://github.com/Betswish/ConsistencyRL)  
**Area**: Reinforcement Learning / Multilingual LLM / Preference Alignment  
**Keywords**: Crosslingual Consistency, DCO, DPO variants, product of experts, RankC

## TL;DR
This paper addresses the issue of multilingual LLMs providing conflicting answers to the same question across different languages. It proposes an **RL objective using the "log-likelihood of the answer in another language" as a reward**. The authors prove that the optimal policy follows a product-of-experts form and guarantees crosslingual preference consistency when $\gamma_1\gamma_2=\beta^2$. Based on this, they derive **DCO (Direct Consistency Optimization)**, an algorithm that requires neither a reward model nor online sampling. It improves both crosslingual consistency (RankC) and accuracy across 9 LLMs, 3 multilingual QA benchmarks, and 26 languages.

## Background & Motivation

**Background**: Although modern LLMs (Llama, Qwen, Aya, Gemma, etc.) are multilingual, **asking the same question in different languages often yields conflicting answers**. Since the introduction of the RankC metric by qi-etal-2023-cross, Crosslingual Consistency (CLC) has become a standard for evaluating multilingual LLMs.

**Limitations of Prior Work**: (1) Interpretability-based interventions (vector editing, representation alignment) are only validated on small datasets or specific models and are difficult to scale. (2) CALM (wang-etal-2025) applies DPO to "winners" selected via multilingual majority voting, but it requires $>2$ languages and fails in bilingual scenarios; furthermore, majority voting becomes unreliable when low-resource languages are included. (3) There is no objective function that theoretically guarantees an "optimal consistent policy."

**Key Challenge**: The Bradley-Terry preference modeling in DPO is inherently designed for "winner vs. loser in a single language." **There is no direct way to express the second-order constraint that "winner/loser rankings should be consistent across languages."** Forcing preference pairs can also degrade the original accuracy in post-trained languages.

**Goal**: 1) Provide a formal definition of CLC (preference rank invariance across languages); 2) Design a reward that directly drives RL convergence to a consistent policy; 3) Derive an efficient algorithm without online sampling or reward models; 4) Validate across multiple models and benchmarks.

**Key Insight**: Instead of finding a "crosslingual winner," the authors set the "reward of an answer in language A = the log-likelihood of the translated answer in language B under the reference model." This design, which **uses the partner language's likelihood to score the current language**, ensures that the dual form of the optimal policy naturally contains crosslingual symmetry.

**Core Idea**: By defining $r_{\text{align}}$ using $\log\pi_{\text{ref}}(\tau(\mathbf y)|\tau(\mathbf x))$ of the counterpart language, the optimal policy for KL-regularized RL becomes a **crosslingual product of experts**. As long as the hyperparameter constraint $\gamma_1\gamma_2 = \beta^2$ is met, the optimal policy necessarily maintains consistent preference rankings between two languages and can be solved in a DPO style without a reward model.

## Method

### Overall Architecture
The method consists of three layers: (1) **CLC Formalization (Def 1)**: A model $\pi^\star$ is consistent in $L_1, L_2$ $\iff$ for any translation-equivalent answer pair $(\mathbf y_w^1, \mathbf y_l^1) \sim (\mathbf y_w^2, \mathbf y_l^2)$, the preference order is identical in both languages. (2) **Structured Reward and Optimal Policy**: A piecewise reward is defined in Eq. 8. Solving the KL-regularized RL yields an optimal policy in a product-of-experts form (Eq. 9), where $\gamma_1\gamma_2 = \beta^2$ is proven to be a sufficient condition for consistency (Lemma 1). (3) **DCO Algorithm**: Reward matching (Eq. 10) is rewritten as a DPO-style differential objective, avoiding reward models and online sampling, and trained directly on parallel prompt/response datasets $\mathcal D_\|$.

### Key Designs

1. **Structured Reward and Crosslingual Duality (Eqs. 7-9)**:
    - **Function**: Expresses the desire for the current language's preferences to match the partner language's preferences through a single reward.
    - **Mechanism**: Defines a piecewise reward:
      $r_{\text{align}}(\mathbf x, \mathbf y) = \gamma_i \log\pi_{\text{ref}}(\tau^j(\mathbf y)|\tau^j(\mathbf x))$ when $\mathbf x, \mathbf y \in L_i$ ($j\ne i$).
      Following Rafailov's KL-regularized RL derivation, the optimal policy is:
      $\pi^\star(\mathbf y^1|\mathbf x^1) \propto \pi_{\text{ref}}(\mathbf y^1|\mathbf x^1) \cdot \pi_{\text{ref}}^{\gamma_1/\beta}(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$.
      This is a **product of experts**—the product of the original likelihood and the translated likelihood. By the rearrangement inequality, maximizing this reward is equivalent to aligning $\{\pi_\theta(\mathbf y|\mathbf x)\}_y$ monotonically with $\{r_{\text{align}}(\mathbf x, \mathbf y)\}_y$, corresponding to Def 1.
    - **Design Motivation**: The authors sought a **formal guarantee** that the optimal solution of this reward is inherently consistent, rather than an empirical heuristic. The product-of-experts form preserves base model knowledge while enforcing crosslingual constraints.

2. **Hyperparameter Constraint $\gamma_1\gamma_2 = \beta^2$ and Multilingual Generalization (Lemma 1)**:
    - **Function**: Selects the subset of $\beta, \gamma_1, \gamma_2$ combinations that guarantee consistency.
    - **Mechanism**: Raising Eq. 9a to the power of $\beta/\gamma_1$ yields $(\pi^\star(\mathbf y^1|\mathbf x^1))^{\beta/\gamma_1} \propto \pi^\star(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$. Since $x \mapsto cx^{\beta/\gamma_1}$ is a monotonic increasing function, the ranking is consistent. $\gamma_i$ controls the deviation strength from $\pi_{\text{ref}}$, while $\beta$ controls overall KL divergence. For $N$ languages, $N^2 - N$ parameters $\gamma_{ij}$ are introduced to control pairwise alignment.
    - **Design Motivation**: In practice, it is necessary to "tune" which language is aligned more strictly (e.g., low-resource languages should stay closer to the original model). This design provides controllable knobs while the $\gamma_1\gamma_2=\beta^2$ constraint simplifies implementation.

3. **DCO Algorithm: No Reward Model, No Online Sampling (Eq. 10)**:
    - **Function**: Converts the RL objective into an offline objective for direct gradient descent on $\theta$.
    - **Mechanism**: Similar to DPO, the reward is reparameterized as $\hat r_\theta(\mathbf x, \mathbf y) = \beta\log\frac{\pi_\theta(\mathbf y|\mathbf x)}{\pi_{\text{ref}}(\mathbf y|\mathbf x)}$. The differences in $\hat r_\theta$ are matched to differences in $r_{\text{align}}$:
      $L(\theta) = \mathbb E\big[\|d_\theta^1 - \gamma_1\log\frac{\pi_{\text{ref}}(\mathbf y_w^2|\mathbf x^2)}{\pi_{\text{ref}}(\mathbf y_l^2|\mathbf x^2)}\| + \|d_\theta^2 - \gamma_2\log\frac{\pi_{\text{ref}}(\mathbf y_w^1|\mathbf x^1)}{\pi_{\text{ref}}(\mathbf y_l^1|\mathbf x^1)}\|\big]$,
      where $d_\theta^i = \hat r_\theta(\mathbf x^i, \mathbf y_w^i) - \hat r_\theta(\mathbf x^i, \mathbf y_l^i)$. Advantages: (a) **No ground truth** winner/loser labels are needed; (b) No reward model training; (c) Fully offline using parallel prompt-response pairs $(\mathbf x^1, \mathbf y^1, \mathbf x^2, \mathbf y^2)$.
    - **Design Motivation**: Uses the partition function cancellation trick from DPO but replaces "matching human preference" with "matching crosslingual consistency." This makes the pipeline compatible with existing DPO frameworks.

### Loss & Training
The study utilizes 9 LLMs (Qwen2.5, Qwen3, Aya-Expanse, Llama3.1, Llama3.2, Gemma3) and 3 parallel QA datasets (MMMLU, XCSQA, BMLAMA) covering 26 languages. Training is performed using the DCO loss (Eq. 10) on parallel prompt-response pairs.

## Key Experimental Results

### Main Results
Multilingual joint training on MMMLU (clc_all = average RankC; a_en / a_¬en = English/Non-English accuracy), incremental changes relative to the base model:

| Model | Method | $\Delta$clc_all | $\Delta$a_en | $\Delta$a_¬en |
|------|------|-----------------|--------------|----------------|
| Qwen2.5-14B | Base = 68.6 / 72.5 / 58.1 | — | — | — |
| Qwen2.5-14B | + SFT* | +0.6 | +1.5 | +6.7 |
| Qwen3-14B | + SFT* | -0.2 | +0.1 | +0.5 |
| Aya-Expanse-8B | + SFT* | +3.5 | +0.7 | — |

(DCO consistently outperforms other methods across all models in RankC without accuracy degradation; specific values are in Table 1 of the original paper.)

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| DCO vs SFT* | DCO is significantly higher in RankC | SFT only optimizes gold answers; it does not solve the ranking problem |
| DCO vs DPO* | DCO has higher consistency and doesn't rely on gold labels | DCO uses parallel pairs instead of preference pairs |
| DCO vs CALM | CALM degrades with low-resource languages; DCO is stable | Majority voting is not the critical factor |
| DCO + DPO combination | DCO and DPO are complementary when gold labels are available | Different objectives address different sub-problems |
| Bilingual training | DCO remains effective where CALM fails | DCO does not require $\ge 3$ languages |
| OOD generalization | RankC improves even on unseen domains | Learns consistency structures, not specific knowledge |

### Key Findings
- **DCO improves consistency without destroying monolingual accuracy**—the most critical advantage over DPO, which often sacrifices performance in post-trained languages to align preferences.
- Asymmetric settings for $\gamma_1/\gamma_2$ enable "directional alignment," allowing high-resource languages to pull low-resource ones while maintaining their own performance.
- Excellent cross-domain generalization: consistency patterns learned on MMMLU transfer to XCSQA and BMLAMA.

## Highlights & Insights
- Reformulating CLC as "likelihood of another language as a reward" results in a product-of-experts form that is both mathematically elegant and engineering-friendly.
- **No gold labels required**: Randomly paired responses work because the differential form only cares whether the reward gap is consistent across languages.
- The $\gamma_1\gamma_2 = \beta^2$ algebraic constraint provides a clear guide for hyperparameter selection, which is rare in RL research.

## Limitations & Future Work
- Evaluation relies on the **existence of translation mapping $\tau$**—currently limited to factual QA with finite, objective answer spaces. Its applicability to open-ended generation (summarization, creative writing) is less clear.
- Parallel dataset translation quality affects training; noise in low-resource translations can distort the reward.
- $\gamma_1\gamma_2=\beta^2$ is a sufficient but not necessarily a necessary condition.
- Interaction with Chain-of-Thought (CoT) reasoning is not discussed; CoT consistency is harder as intermediate steps must also align.
- Computational overhead: Each sample requires forwards in two languages, doubling the cost compared to monolingual DPO.

## Related Work & Insights
- **vs DPO (Rafailov et al. 2023)**: DCO swaps human preference matching for crosslingual consistency matching. It retains the partition function cancellation trick but targets a different alignment goal.
- **vs CALM (wang-etal-2025)**: Unlike CALM, DCO doesn't need majority voting or $\ge 3$ languages, making it more robust for bilingual and low-resource scenarios.
- **vs Representation Interventions**: DCO is a black-box method using likelihood signals rather than hidden states, facilitating better scalability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reformulating crosslingual consistency into an RL reward and deriving a DPO-style algorithm is a unique contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Extensive coverage across 9 models, 3 datasets, and 26 languages, including OOD and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous derivation with full proofs for Lemmas 1 and 2; clear logic despite complex notation.
- **Value**: ⭐⭐⭐⭐ Directly applicable to multilingual LLM deployment and complementary to standard preference alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](../../ACL2026/multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ICML 2026\] Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](../../ACL2026/multilingual_mt/dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](../../ACL2026/multilingual_mt/language_models_entangle_language_and_culture.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)

</div>

<!-- RELATED:END -->
