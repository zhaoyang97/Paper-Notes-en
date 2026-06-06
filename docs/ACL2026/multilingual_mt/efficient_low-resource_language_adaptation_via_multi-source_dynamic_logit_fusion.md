---
title: >-
  [Paper Note] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion
description: >-
  [ACL 2026][Multilingual & Machine Translation][Low-resource languages] TriMix decomposes LRL (low-resource language) adaptation into three logit benefit vectors: "language capability + task capability + scaling bonus." I…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Low-resource languages"
  - "Proxy Tuning"
  - "Logit Fusion"
  - "Dynamic weights"
  - "Continued Pre-training"
date: 2026-05-08
content_hash: 86a64e349d519117
---

# Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion

**Conference**: ACL 2026  
**arXiv**: [2604.18106](https://arxiv.org/abs/2604.18106)  
**Code**: https://github.com/luciusssss/TriMix  
**Area**: Multilingual / Low-Resource  
**Keywords**: Low-resource languages, Proxy Tuning, Logit Fusion, Dynamic weights, Continued Pre-training

## TL;DR
TriMix decomposes LRL (low-resource language) adaptation into three logit benefit vectors: "language capability + task capability + scaling bonus." It only requires continued pre-training (CPT) on small models. During inference, weights are dynamically determined based on perplexity. It consistently outperforms single-model baselines and Proxy Tuning across 4 model families and 8 LRLs. A core empirical finding is that "the weight of the small CPT model should be higher than the large instruction model," directly challenging the "large-model dominant" assumption of Proxy Tuning.

## Background & Motivation

**Background**: Transferring Large Language Models (LLMs) dominated by High-Resource Languages (HRLs like English) to LRLs (Tibetan, Uyghur, Kazakh, Bengali, etc.) remains a significant challenge in multilingual NLP. Two main paradigms exist: (1) Model merging (e.g., TIES): parameter-level fusion of an "instruction model with task capabilities in HRL" and a "base model CPT-ed on LRL," which requires the two models to have the **same architecture and size**; (2) Proxy Tuning (Liu et al. 2024): injecting logits from a small specialized model into large model logits to avoid large model CPT, which has succeeded in the code domain.

**Limitations of Prior Work**: Proxy Tuning implicitly assumes "the large model serves as the main signal, while the small model provides the delta." However, in LRL scenarios, the large model itself is **also a weak performer** on the target LRL. Treating it as the main signal "suppresses" the strong LRL capabilities of the small CPT model and may even disrupt basic LRL generation (examples provided in Appendix B.4). In other words, capabilities from different sources are not equivalent in logit fusion.

**Key Challenge**: LRL tasks lack three elements simultaneously: LRL language data, task annotations, and LLM compute power. Existing methods typically solve only one or two and default to "large model logits as the backbone," which conflicts with the fact that "large models are weak in LRL."

**Goal**: Design a framework that (i) requires no LRL task annotations, (ii) bypasses large model CPT, (iii) correctly balances the three capability sources, and (iv) is universal across multiple model families.

**Key Insight**: Categorize models into base, ins, and cpt variants. Represent "task," "LRL," and "scaling" as delta vectors between these variants in the logit space. Use perplexity, an unsupervised measure of input distribution fit, to automatically select weights.

**Core Idea**: Perform "three-source linear decomposition + dynamic weights" on logits, intentionally coupling the scaling coefficient with the task coefficient ($\gamma=\alpha$). This eliminates the large base model, requiring only "large ins + small base + small cpt" for inference.

## Method

### Overall Architecture
TriMix is a **purely test-time** framework (no training required except for small model CPT). Given an LRL input prompt: (1) feed it simultaneously to three models: a large instruction model (large-ins), a small base model (small-base), and a small model CPT-ed on the LRL (small-cpt); (2) perform linear fusion of their next-token logits: $L=\alpha L_{large\text{-}ins}+\beta L_{small\text{-}cpt}+(1-\alpha-\beta)L_{small\text{-}base}$; (3) select $\alpha, \beta$ **online** from a small grid using perplexity-guided (default) or entropy-guided methods; (4) sample the next token after softmax and repeat. This workflow requires only one CPT session on raw LRL text for the small model, **requires no LRL task annotations**, and **never updates** the large model.

### Key Designs

1.  **Three-source Linear Decomposition (Task / Language / Scaling benefit vectors)**:
    - **Function**: Expresses the "ideal logit" $L$ as a linear combination of the small-base logit and three independent gain vectors, making capability sources explicitly adjustable.
    - **Mechanism**: Defines the task gain $\delta_T=L_{large\text{-}ins}-L_{large\text{-}base}$ (extracting task capability from large models due to their higher learning capacity), the LRL gain $\delta_L=L_{small\text{-}cpt}-L_{small\text{-}base}$ (as only small models can undergo CPT, LRL capability is derived from the small model delta), and the scaling gain $\delta_S=L_{large\text{-}base}-L_{small\text{-}base}$ (comparing base models to avoid interference from instruction tuning). The final logit is $L=L_{small\text{-}base}+\alpha\delta_T+\beta\delta_L+\gamma\delta_S$.
    - **Design Motivation**: Explicitly disentangling capabilities allows independent weight adjustment for each source. Using base-to-base for $\delta_S$ ensures "instruction style" is not mistaken for "scaling dividends."

2.  **$\gamma=\alpha$ Coupling + Automatic Elimination of Large Base**:
    - **Function**: Reduces the number of models loaded during inference from 4 to 3, saving VRAM and bandwidth costs associated with the large base model.
    - **Mechanism**: By setting $\gamma=\alpha$, the terms $\alpha(L_{large\text{-}ins}-L_{large\text{-}base})$ and $\alpha(L_{large\text{-}base}-L_{small\text{-}base})$ are combined, canceling out the $L_{large\text{-}base}$ term. The formula simplifies to $L=\alpha L_{large\text{-}ins}+\beta L_{small\text{-}cpt}+(1-\alpha-\beta)L_{small\text{-}base}$. This requires only large-ins, small-base, and small-cpt for the forward pass.
    - **Design Motivation**: Practical LRL deployment often lacks the VRAM for an 80B large-base model. This coupling trades flexibility for engineering efficiency.

3.  **Perplexity-guided Dynamic Weight Selection**:
    - **Function**: Selects $(\alpha, \beta)$ **per sample** without LRL task annotations, preventing the failure of traditional hyperparameter searches.
    - **Mechanism**: For each prompt (including in-context examples), the perplexity is calculated using the fused model. The $(\alpha, \beta)$ pair that yields the lowest PPL from a small grid is selected. This follows the logic that "the weight set that best explains the input distribution is used." An alternative "ENT" strategy selects the configuration with the lowest entropy for the first generated token. Both are unsupervised.
    - **Design Motivation**: LRLs lack validation sets. PPL is a robust unsupervised proxy highly correlated with generation quality. Experiments show PPL-selected weights are very close to the empirical Upper Bound.

### Loss & Training
The **only training** required is the CPT of the small model on raw LRL corpora to obtain the small-cpt. All other stages (task capability transfer, scaling gain, weight selection) occur at test time with **zero task annotations and zero gradient updates** for the large model. CPT details vary by family: Qwen2.5, Llama3.2, and Gemma3 utilize custom CPT, while Llama2 reuses checkpoints from Tao et al. 2024.

## Key Experimental Results

### Main Results
Comparison within the Qwen2.5 family across different large model sizes (averages for 4 minority languages: Tibetan bod, Uyghur uig, Kazakh kaz, Mongolian mvf). $\Delta$ denotes gain over the best single-model baseline:

| Setting | Method | #Param Train | #Param Test | MC | ENG-G | LRL-G | Avg | $\Delta$ |
|------|------|--------------|-------------|-----|-------|-------|------|----------|
| 1.5B+3B | Qwen2.5-3B-ins | 0 | 3B | 42.4 | 12.2 | 10.8 | 24.8 | – |
| 1.5B+3B | Proxy Tuning | 1.5B | 6B | 45.4 | 14.1 | 14.1 | 28.5 | -7.2% |
| 1.5B+3B | **TriMix (PPL)** | 1.5B | 6B | 48.7 | 19.5 | 16.3 | **31.1** | **+1.3%** |
| 1.5B+3B | TriMix (Upper Bound) | 1.5B | 6B | 52.4 | 21.3 | 17.6 | 33.6 | +9.4% |
| 1.5B+7B | Qwen2.5-7B-ins | 0 | 7B | 49.7 | 20.0 | 12.5 | 30.6 | – |
| 1.5B+7B | Proxy Tuning | 1.5B | 10B | 50.5 | 16.3 | 13.3 | 30.0 | -2.3% |
| 1.5B+7B | **TriMix (PPL)** | 1.5B | 10B | **53.4** | 19.8 | 15.7 | **33.0** | **+7.5%** |
| 1.5B+14B | Qwen2.5-14B-ins | 0 | 14B | 57.1 | 21.0 | 13.8 | 34.4 | – |
| 1.5B+14B | Proxy Tuning | 1.5B | 17B | 57.7 | 15.4 | 16.8 | 33.9 | -1.5% |
| 1.5B+14B | **TriMix (PPL)** | 1.5B | 17B | **59.5** | 20.5 | 16.8 | **36.1** | **+4.9%** |

Key observations: (1) Proxy Tuning degrades performance on most Qwen2.5 configurations (up to -7.2%), confirming that "large-model dominance" fails in LRL; (2) TriMix-PPL consistently provides positive gains across all sizes; (3) The gap between PPL and Upper Bound shrinks as the large model size increases; (4) The ENT strategy is inferior to PPL.

### Ablation Study
Empirical weight distribution findings (based on Upper Bound):

| Setting | Optimal $\alpha$ (large-ins) | Optimal $\beta$ (small-cpt) | $\beta/\alpha$ |
|------|-------------------------------|------------------------------|----------------|
| Proxy Tuning Assumption | ≈1.0 | ≈0.x | <1 (Large dominant) |
| TriMix Upper Bound | Smaller | **Significantly larger** | **>1 (Small-cpt dominant)** |

The **optimal strategy is the inverse of the Proxy Tuning assumption**: in LRL scenarios, the small CPT model should dominate the logits, with the large ins model acting as a secondary signal for task and scaling.

### Key Findings
- **Large-model dominance causes Proxy Tuning to fail in LRL**: The default assumption is disproven by empirical weight distributions. Future logit-fusion should follow the principle of "dominance by the strongest in the target domain."
- **Perplexity is a robust unsupervised proxy**: PPL-chosen weights closely match the Upper Bound and outperform ENT.
- **High leverage of small model CPT**: Utilizing a 1.5B CPT model with a 14B off-the-shelf ins model yields a 4.9% gain over the base 14B model, effectively saving the compute needed to CPT the 14B model.
- **Task sensitivity**: TriMix shows the largest gains in generative tasks (LRL-G, ENG-G) compared to Multiple Choice (MC) tasks.
- **Divergence-from-base explains weight needs**: As the small CPT model diverges more from its base (indicating deeper LRL adaptation), the optimal $\beta$ increases.

## Highlights & Insights
- **Three-source Decomposition + Coupling**: Upgrades logit fusion into a controllable combination of three capabilities. The $\gamma=\alpha$ coupling elegantly eliminates the large base model.
- **Challenging Proxy Tuning**: Provides strong evidence against the intuition that "the large model must be the main signal."
- **Plug-and-play without Labels**: Extremely friendly to LRL communities (Tibetan, Uyghur) by reducing the barrier for task-specific data.
- **PPL-based Paradigm**: Establishing "PPL on prompt" as a free proxy for weight selection is highly transferable to other inference-time fusion works.

## Limitations & Future Work
- The $\gamma=\alpha$ coupling is a compromise for engineering utility; theoretically, decoupled coefficients could achieve higher performance.
- CPT remains necessary, making the method inapplicable to ultra-low resource languages with no raw text.
- Experiments focused on Llama, Qwen, and Gemma; MoE model logit fusion behavior remains unexplored.
- Evaluation relies on MiLiC-Eval and Belebele; fine-grained human evaluation for LRL generation quality is still needed.
- Continuous optimization (e.g., token-level dynamic gating) could further refine the weight selection grid.

## Related Work & Insights
- **vs Proxy Tuning (Liu 2024)**: TriMix reverses the dominance hierarchy and adds independent "task + scaling" channels.
- **vs Model Merging (Tao 2024, TIES)**: TriMix does not require identical model sizes and enjoys scaling benefits without large model CPT.
- **vs Contrastive Decoding (Li 2023)**: TriMix explicitly incorporates large models for task and scaling signals and introduces dynamic weighting.
- **Insight**: Any scenario where "small models understand specific X, large models are generalists" (medicine, law, specialized agents) can benefit from TriMix by mapping specific capabilities to $\delta_L$ and general task capability to $\delta_T$.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)

</div>

<!-- RELATED:END -->
