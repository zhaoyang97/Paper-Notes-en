---
title: >-
  [Paper Note] Optimizing Language Models for Crosslingual Knowledge Consistency
description: >-
  [ICML 2026][Multilingual & Translation][DCO] This paper addresses the issue of multilingual LLMs providing conflicting answers to the same question across different languages. It designs an **RL objective using the "log-likelihood of the answer in another language" as a reward**, proving that the optimal policy follows a product-of-experts form and guarantees cro
tags:
  - ICML 2026
  - Multilingual & Translation
  - DCO
  - product of experts
  - RankC
date: 2026-05-08
content_hash: 5b4744872a06c716
---
# Optimizing Language Models for Crosslingual Knowledge Consistency

**Conference**: ICML 2026  
**arXiv**: [2603.04678](https://arxiv.org/abs/2603.04678)  
**Code**: [github.com/Betswish/ConsistencyRL](https://github.com/Betswish/ConsistencyRL)  
**Area**: Reinforcement Learning / Multilingual LLM / Preference Alignment  
**Keywords**: Crosslingual consistency, DCO, DPO variants, product of experts, RankC

## TL;DR
This paper addresses the issue of multilingual LLMs providing conflicting answers to the same question across different languages. It designs an **RL objective using the "log-likelihood of the answer in another language" as a reward**, proving that the optimal policy follows a product-of-experts form and guarantees crosslingual preference consistency when $\gamma_1\gamma_2=\beta^2$. Based on this, the authors derive **DCO (Direct Consistency Optimization)**, a reward-model-free and online-sampling-free algorithm. Experiments across 9 LLMs, 3 multilingual QA benchmarks, and 26 languages demonstrate simultaneous improvements in crosslingual consistency (RankC) and response accuracy.

## Background & Motivation

**Background**: Modern LLMs (Llama, Qwen, Aya, Gemma, etc.) claim to be multilingual, but **answering the same question in different languages often yields conflicting results**. Since the introduction of the RankC metric by qi-etal-2023-cross, crosslingual consistency (CLC) has become a standard for evaluating multilingual LLMs.

**Limitations of Prior Work**: (1) Interpretability-based intervention methods (vector editing, representation alignment) are only validated on small datasets or specific models and are difficult to scale. (2) CALM (wang-etal-2025) applies DPO to "winners" selected via multilingual majority voting, but it requires $>2$ languages and fails in bilingual scenarios; furthermore, majority voting can become distorted when low-resource languages are included. (3) There is no objective function that theoretically guarantees that the "optimal policy must be consistent."

**Key Challenge**: Bradley-Terry preference modeling in DPO is inherently designed for "winner vs. loser within a single language." **There is no direct way to express the second-order constraint that "winner/loser rankings should be consistent across languages."** Forcing preference pairs can also easily damage the original accuracy in post-trained languages.

**Goal**: 1) Provide a formal definition of CLC (preference rank remains invariant across languages); 2) Design a reward capable of directly driving RL to converge to a consistent strategy; 3) Derive an efficient algorithm without online sampling or reward models; 4) Validate across multiple models and benchmarks.

**Key Insight**: Moving away from the paradigm of "finding a crosslingual winner," the authors use "the reward of a response in language A = the log-likelihood of the translated response in language B under the original model." This design, which **uses the partner language's likelihood to score the current language**, ensures that the dual form of the optimal policy inherently contains crosslingual symmetry.

**Core Idea**: By defining a reward $r_{\text{align}}$ using the partner language's $\log\pi_{\text{ref}}(\tau(\mathbf y)|\tau(\mathbf x))$, the optimal policy of KL-regularized RL becomes a **crosslingual product of experts**. As long as the hyperparameter constraint $\gamma_1\gamma_2=\beta^2$ is met, the optimal policy necessarily maintains consistent preference rankings between two languages and can be solved in a DPO style without a reward model.

## Method

### Overall Architecture
The method consists of three layers: (1) **CLC Formalization (Def 1)**: A model $\pi^\star$ is consistent across $L_1, L_2$ $\iff$ for any translation-equivalent response pairs $(\mathbf y_w^1, \mathbf y_l^1) \sim (\mathbf y_w^2, \mathbf y_l^2)$, the preference order is consistent between languages. (2) **Structured Reward and Optimal Policy**: A piecewise reward Eq.8 is defined, and the KL-regularized RL is solved to obtain the product-of-experts optimal policy Eq.9. It is proven that $\gamma_1\gamma_2 = \beta^2$ is a sufficient condition for consistency (Lemma 1). (3) **DCO Algorithm**: The reward matching (Eq.10) is rewritten as a DPO-style differential objective, avoiding reward models and online sampling, and trained directly using a parallel prompt/response dataset $\mathcal D_\|$.

### Key Designs

**1. Structured Reward and Crosslingual Duality: Native Product-of-Experts Optimal Policy**

The Bradley-Terry preference in DPO can only express "winner vs. loser in a single language" and cannot directly represent the second-order constraint that "preference rankings in two languages should be consistent." The authors take a different approach: let the reward of a response in language $L_i$ be its log-likelihood under the reference model after translation to the partner language $L_j$. This gives the piecewise reward $r_{\text{align}}(\mathbf x,\mathbf y) = \gamma_i \log\pi_{\text{ref}}(\tau^j(\mathbf y)|\tau^j(\mathbf x))$ (where $\mathbf x,\mathbf y\in L_i, j\ne i$). Substituting this into Rafailov's KL-regularized RL optimal policy formula yields $\pi^\star(\mathbf y^1|\mathbf x^1) \propto \pi_{\text{ref}}(\mathbf y^1|\mathbf x^1)\cdot\pi_{\text{ref}}^{\gamma_1/\beta}(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$—a product of experts: the original likelihood of the current language multiplied by the translated likelihood of the partner language.

Why does this guarantee consistency? According to the rearrangement inequality, maximizing this reward is equivalent to monotonically aligning the rankings of $\{\pi_\theta(\mathbf y|\mathbf x)\}_y$ and $\{r_{\text{align}}(\mathbf x,\mathbf y)\}_y$. Since the latter is crosslingually symmetric, it perfectly matches the consistency definition in Def 1. The authors seek a formal guarantee that the optimal solution of this reward is necessarily consistent, rather than just an "empirically reasonable" heuristic. The product-of-experts form also preserves the knowledge of the base model, preventing bilingual performance degradation during alignment.

**2. Hyperparameter Constraint $\gamma_1\gamma_2=\beta^2$: Selecting the Unique Consistency-Guaranteed Family**

A product-of-experts is only a necessary structure; one must also identify which values of $\beta, \gamma_1, \gamma_2$ truly ensure consistent ranking. By raising both sides of the optimal policy equation to the power of $\beta/\gamma_1$, it can be rewritten as $(\pi^\star(\mathbf y^1|\mathbf x^1))^{\beta/\gamma_1} \propto \pi^\star(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$. Because $x\mapsto cx^{\beta/\gamma_1}$ is monotonically increasing, the preference rankings in both languages must be identical—Lemma 1 thus provides the sufficient condition $\gamma_1\gamma_2=\beta^2$.

Here, $\gamma_1$ and $\gamma_2$ control the deviation intensity from $\pi_{\text{ref}}$ for the two languages (smaller $\gamma$ stays closer to the original model), while $\beta$ controls the overall KL deviation. This set of "knobs" is practical: low-resource languages can have their $\gamma$ tuned smaller to remain close to the original model and avoid being "distorted" by high-resource languages. When generalized to $N$ languages, $N^2-N$ variables $\gamma_{ij}$ are introduced to control pairwise alignment (constraints in Appendix E). The condition $\gamma_1\gamma_2=\beta^2$ also simplifies implementation—one can simply choose valid values like $\gamma_1=\gamma_2=\beta$.

**3. DCO Algorithm: Offline Objective Without Reward Models or Online Sampling**

The RL objective must be trainable to be useful. Similar to DPO, the authors reparameterize the reward as $\hat r_\theta(\mathbf x,\mathbf y) = \beta\log\frac{\pi_\theta(\mathbf y|\mathbf x)}{\pi_{\text{ref}}(\mathbf y|\mathbf x)}$, allowing the difference in $\hat r_\theta$ to match the difference in $r_{\text{align}}$:

$$L(\theta) = \mathbb E\Big[\big\|d_\theta^1 - \gamma_1\log\tfrac{\pi_{\text{ref}}(\mathbf y_w^2|\mathbf x^2)}{\pi_{\text{ref}}(\mathbf y_l^2|\mathbf x^2)}\big\| + \big\|d_\theta^2 - \gamma_2\log\tfrac{\pi_{\text{ref}}(\mathbf y_w^1|\mathbf x^1)}{\pi_{\text{ref}}(\mathbf y_l^1|\mathbf x^1)}\big\|\Big],$$

where $d_\theta^i = \hat r_\theta(\mathbf x^i,\mathbf y_w^i) - \hat r_\theta(\mathbf x^i,\mathbf y_l^i)$. The differential form adopts the DPO trick to eliminate the partition function $Z(\mathbf x)$, resulting in: (a) winner/loser labels do not require ground truth and can be randomly paired; (b) no reward model training is needed; (c) the process is entirely offline, using only parallel prompt-response data $\mathcal D_\|$, where each sample is a translation pair $(\mathbf x^1,\mathbf y^1, \mathbf x^2, \mathbf y^2)$. Lemma 2 proves that the optimal $\hat r_\theta^\star$ converges to $r_{\text{align}}$ plus a constant $c(\mathbf x)$ independent of $\mathbf y$, which does not affect the policy.

The only difference from DPO is shifting the objective from "matching human preference" to "matching crosslingual consistent reward." Thus, the training pipeline is fully compatible with existing DPO frameworks, making it extremely easy to implement—allowing reproduction across 9 models.

### Loss & Training
The study utilizes 9 LLMs (Qwen2.5-7B/14B, Qwen3-8B/14B, Aya-Expanse-8B, Llama3.1-8B, Llama3.2-3B, Gemma3-4B/12B) and 3 parallel QA datasets (MMMLU in 14 languages, XCSQA in 16 languages, BMLAMA in 17 languages), covering 26 languages. Training is performed using the DCO loss (Eq. 10) on parallel prompt-response pairs.

## Key Experimental Results

### Main Results
Multilingual joint training on MMMLU (clc_all = average RankC across all language pairs; a_en / a_¬en = English / non-English accuracy), incremental changes relative to base models:

| Model | Method | $\Delta$clc_all | $\Delta$a_en | $\Delta$a_¬en |
|------|------|-----------------|--------------|----------------|
| Qwen2.5-14B | Base = 68.6 / 72.5 / 58.1 | — | — | — |
| Qwen2.5-14B | + SFT* | +0.6 | +1.5 | +6.7 |
| Qwen3-14B | + SFT* | -0.2 | +0.1 | +0.5 |
| Aya-Expanse-8B | + SFT* | +3.5 | +0.7 | — |
| Llama3.1-8B | + SFT* | — | — | — |

(The full version of Table 1 in the paper also includes +DPO, +CALM, and **+DCO** rows—DCO consistently outperforms other methods on clc_all, with accuracy remaining stable or slightly increasing; see the original Table 1 for specific values.)

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| DCO vs SFT* | DCO is significantly higher in RankC | SFT only optimizes the gold answer and does not solve the crosslingual ranking problem |
| DCO vs DPO* | DCO has higher crosslingual consistency and does not rely on gold labels | DCO uses parallel pairs instead of preference pairs |
| DCO vs CALM | CALM degrades after adding low-resource languages, DCO remains stable | Majority voting is not the key factor |
| DCO + DPO combination | DCO and DPO are complementary when gold labels are available | Different objectives solve different sub-problems |
| Bilingual Training | DCO is equally effective, CALM fails | DCO does not require $\ge$3 languages |
| OOD generalization | RankC still improves on unseen domains | The model learns the consistency structure, not specific knowledge |
| $\gamma_1 \ne \gamma_2$ to control language bias | Directional bias towards specific languages can maintain original performance | Engineering controllable |

### Key Findings
- **DCO improves consistency without destroying single-language accuracy**—this is a critical advantage over DPO, which tends to sacrifice performance on post-trained languages to align preferences.
- Asymmetric settings for $\gamma_1/\gamma_2$ allow for "directional alignment": high-resource languages can pull low-resource languages closer while maintaining their original performance.
- Strong cross-domain generalization: consistency patterns trained on MMMLU can transfer to XCSQA and BMLAMA.

## Highlights & Insights
- Formulating CLC consistency as "using the likelihood of another language as a reward" results in a product-of-experts form that is both mathematically elegant (rearrangement inequality directly proves consistency) and engineering-friendly (compatible with DPO pipelines).
- **Training without gold labels**: Random pairing of winner/loser remains effective because the differential form only cares about whether "the difference in rewards for two responses is consistent across languages," reducing data requirements to simple parallel translations.
- The elegant algebraic constraint $\gamma_1\gamma_2 = \beta^2$ clears the confusion regarding hyperparameter selection, representing a rare case where theory provides a direct guide for hyperparameter tuning.

## Limitations & Future Work
- Evaluation relies on the **existence of a translation mapping $\tau$**—the paper is limited to factual QA scenarios where answer spaces are finite and objectively translatable. For open-ended generation (creative writing, summarization) where answer spaces are ill-defined, the definition of "consistency" is ambiguous, and DCO is not directly applicable.
- Training is affected by the translation quality of parallel datasets (MMMLU/XCSQA/BMLAMA); translation noise in low-resource languages may distort the reward.
- Lemma 1's $\gamma_1\gamma_2=\beta^2$ is a sufficient but not a necessary condition; whether broader valid regions exist remains unexplored.
- Interaction with Chain-of-Thought (CoT) reasoning is not discussed; crosslingual consistency in CoT is harder because intermediate steps also require alignment.
- Computational overhead: Each sample requires forward passes in two languages, doubling the cost compared to monolingual DPO.

## Related Work & Insights
- **vs DPO (Rafailov et al. 2023)**: DCO replaces DPO's "matching human preference" with "matching crosslingual consistency reward," utilizing the differential trick to eliminate the partition function, but the objectives are entirely different—aligning preferences vs. aligning languages.
- **vs CALM (wang-etal-2025)**: CALM requires $\ge$3 languages for majority voting to find a "winner" for DPO. DCO uses parallel pairs directly, works in bilingual settings, and does not degrade with low-resource languages.
- **vs Representation Intervention (Lu, Wang, Liu)**: DCO does not require white-box access to hidden states and relies purely on likelihood signals, making it easier to scale.
- **vs RankC Evaluation (qi-etal-2023-cross)**: This paper is among the first to treat RankC as an RL training objective rather than just an evaluation metric.

## Rating
- Novelty: ⭐⭐⭐⭐ Converting crosslingual consistency into an RL reward and deriving a DPO-style offline algorithm is a genuine innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 9 models × 3 datasets × 26 languages, including OOD, bilingual, and controlled experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivation with full proofs for Lemma 1/2; complex notation but clear logic.
- Value: ⭐⭐⭐⭐ Direct value for multilingual LLM deployment and complementary to DPO.

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
