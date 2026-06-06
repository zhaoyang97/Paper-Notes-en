---
title: >-
  [Paper Note] Membership Inference Attacks on In-Context Learning Recommendation
description: >-
  [ACL 2026][LLM Safety][Membership Inference Attacks] This paper presents the first systematic study of Membership Inference Attacks (MIA) on LLM-based ICL recommendation systems. It designs four attacks—Similarity…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Membership Inference Attacks"
  - "ICL-RecSys"
  - "LLM Privacy"
  - "Prompt Injection"
  - "Memorization"
date: 2026-05-08
content_hash: 01f8a5df65ad75f7
---

# Membership Inference Attacks on In-Context Learning Recommendation

**Conference**: ACL 2026  
**arXiv**: [2508.18665](https://arxiv.org/abs/2508.18665)  
**Code**: To be confirmed  
**Area**: LLM Security / MIA / Recommendation Systems  
**Keywords**: Membership Inference Attacks, ICL-RecSys, LLM Privacy, Prompt Injection, Memorization

## TL;DR
This paper presents the first systematic study of Membership Inference Attacks (MIA) on LLM-based ICL recommendation systems. It designs four attacks—Similarity, Memorization, Inquiry, and Poisoning—discovering that the Memorization attack, which exploits the LLM's intrinsic **memorization**, achieves an attack advantage $\ge 82\%$ on MovieLens-1M. Furthermore, existing prompt-based defenses (including those against Poisoning) prove largely ineffective.

## Background & Motivation
**Background**: With the rise of emergent abilities in LLMs, industrial players (e.g., Amazon, Google) have begun utilizing In-Context Learning (ICL) for cross-domain recommendations. By directly prepending several "history-user-interaction-recommendation" examples into the system prompt, LLMs can perform zero/few-shot recommendation. This eliminates the fine-tuning costs associated with models like P5, M6-Rec, or TALLRec, while achieving comparable or superior performance.

**Limitations of Prior Work**: User interaction histories are inserted into the prompt in their original form, essentially exposing "sensitive behavior logs" to the model. If an attacker can determine whether a target user's interactions appeared in the prompt, it constitutes a classic Membership Inference Attack (MIA)—a severe privacy breach for recommendation systems (revealing shopping history, movie tastes, or medical preferences). However:
1. **Traditional RecSys MIA is incompatible**: Previous MIAs (Zhang et al. 2021 / Wang et al. 2022 / Zhong et al. 2024) rely on **item embeddings** derived from matrix factorization to measure similarity, requiring large-scale interaction data for training. LLM-RecSys only includes a few demos in the prompt, providing no basis for training shadow models.
2. **Shift in LLM output format**: Traditional MIAs use model confidence/loss, whereas LLM-RecSys outputs natural language lists without associated probabilities.
3. **New LLM characteristics**: Behaviors such as memorization (Carlini 2023) and reasoning, absent in traditional models, may give rise to new types of attacks.

**Key Challenge**: To enable personalized recommendations in LLMs, user history must be included in the prompt; however, this creates an extreme "training sample as input" scenario where samples in the prompt are naturally and strongly memorized.

**Goal**: (i) Systematically design MIAs specifically for ICL-LLM-RecSys; (ii) ensure attacks function in a black-box setting without probability outputs or shadow models; (iii) evaluate the vulnerability of existing prompt-based defenses to locate real-world risks.

**Key Insight**: The authors align traditional MIA item similarity with the inherent memorization of LLMs and their susceptibility to prompt injection, deriving an attack strategy from each.

**Core Idea**: Rather than indirectly calculating embedding similarity, the authors exploit the "photographic memory" of LLMs regarding their prompts—inducing the model to repeat remembered recommendations.

## Method

### Overall Architecture
The attacker possesses: the target user $u$'s historical interactions $I_u$ and their (known) recommendations $R_u$, the target LLM (black-box, no tokens/logits/tokenizer), and a third-party semantic embedding model (e.g., Sentence-BERT, not necessarily the target LLM's own). The attacker **does not know** if $u$ was selected by the RecSys maintainer to be included in the $k$-shot demos of the system prompt. Under this setting, the authors design four attacks: (1) Similarity (replicating traditional MIA), (2) Memorization (exploiting LLMs to directly output remembered items), (3) Inquiry (directly asking the LLM "Have you seen this user?"), and (4) Poisoning (using perturbed prompts to detect "model stance shifts"). Each attack use a threshold $\tau$ to map a scalar signal to member/non-member status.

### Key Designs

1. **Memorization Attack (Most stable and powerful)**:
    - **Function**: Determines if the recommendations for target user $u$ have leaked via the system prompt.
    - **Mechanism**: The attacker submits a prompt: "The user has watched the following movies: $I_u$. Based on this watch history, please recommend the top 10 movies...". The LLM outputs a recommendation set $R_u^{\text{new}}$. The attacker compares $|R_u^{\text{new}} \cap R_u^{\text{historical}}|$. If the overlap exceeds $\tau_m \in [6, 10]$ items, $u$ is classified as a member. The principle is that when $u$ is in the $k$-shot demos, the LLM's "next-token prediction + contextual recency" strongly leans towards repeating items from the prompt; non-members lack this bias.
    - **Design Motivation**: It was observed that the Similarity attack fails in LLM-RecSys because general semantic embeddings are incompatible with interaction-matrix embeddings (T-SNE visualizations show that collaborative filtering neighbors of "Animal House" rarely overlap with semantic neighbors). However, the **distribution of duplicate item counts** differs significantly between members and non-members, providing a robust discriminative signal that bypasses embedding compatibility issues.

2. **Poisoning Attack (Hardest to defend)**:
    - **Function**: Detects the presence of $u$ while attempting to evade prompt-protection mechanisms.
    - **Mechanism**: The attacker submits a **perturbed** prompt, replacing several items $i_k$ in $I_u = (i_1, \ldots, i_n)$ with semantically distant ones $i_k' = \arg\min_{j \in I} \text{sim}(i_k, j)$ (e.g., replacing "Star Wars" with "Cooking Recipe"). The LLM generates $R_u'$ based on this poisoned history. The attacker then calculates $\text{Sim}(R_u, R_u') = \text{sim}(\text{con}(R_u), \text{con}(R_u'))$. **Intuition**: If the model has seen $u$'s true recommendations in the system prompt, it will remain "stubborn" due to memory, keeping $R_u'$ close to $R_u$ (high similarity = member). If not, the model follows the perturbation, causing $R_u'$ to deviate (low similarity = non-member). Threshold $\tau_p \in [0.6, 0.85]$.
    - **Design Motivation**: While Memorization and Inquiry are direct questions that a model might detect and refuse, Poisoning acts as an indirect probe of the model's "persistence." It appears like a normal recommendation request, making it difficult for rule-based prompt-injection defenses to identify. Interestingly, **increased poisoning can decrease attack effectiveness**: beyond a certain threshold, the model switches from being "memory-dominated" to "new context-dominated."

3. **Inquiry Attack (Simplest, but easily blocked)**:
    - **Function**: Directly queries the LLM about a specific user.
    - **Mechanism**: A direct prompt: "Have you seen a user interacted with the item set $I_u$? Only answer Yes or No." The output is used as the classification result.
    - **Design Motivation**: This "lazy" strategy tests the degree of LLM self-disclosure regarding its prompt history. While models like GPT-OSS-120b still leak information (advantage $\ge 78\%$), many newer models have jailbreak preventions that fluctuate in effectiveness.

### Loss & Training
**Completely training-free.** All attacks are zero-shot black-box queries. Each core attack uses a threshold $\tau \in \{\tau_s, \tau_m, \tau_p\}$ and a single scalar signal. Sentence-BERT acts as the text encoder. Prompt demos utilize LightGCN-generated ground-truth recommendations, with 1/5/10 shots randomly selected for the prompt. Each experiment involves 100 paired evaluations (50 members, 50 non-members).

## Key Experimental Results

### Main Results

**Attack advantage (= 2 × (Acc − 0.5)) on MovieLens-1M / Amazon Book / Amazon Beauty**, showing best settings for Llama4-109B, Mistral-7B, and GPT-OSS-120B:

| Attack | Movie (Llama4 / Mistral / GPT-OSS:120b) | Book | Beauty |
|------|----------------------------------------|------|--------|
| Similarity | ~0.05 / 0.34 / 0.42 | ~0 / 0.34 / 0.34 | ~0.04 / 0.21 / 0.34 |
| **Memorization** | **0.95 / 0.99 / 1.00** | 0.84 / 1.00 / 0.95 | 0.02 / 0.71 / 0.85 |
| Inquiry | 0.82 / 0.48 / 0.92 | 0.83 / 0.48 / 1.00 | 0.52 / 0.44 / 0.98 |
| Poisoning | 0.92 / 0.91 / 1.00 | 0.77 / 0.97 / 0.88 | 0.44 / 0.73 / 0.80 |

### Ablation Study

| Factor | Memorization | Inquiry | Poisoning |
|------|--------------|---------|-----------|
| Shots 1 → 10 | Minimal impact | Significant drop | Moderately sensitive |
| Shot Position | Stable across positions | Unstable for small models | Stable across positions |
| Poisoned items 1 → 10 | — | — | **Monotonic decrease** |
| Instruction-based Defense | -0.5 on GPT-OSS; increased for Mistral | Variable | **Largely ineffective; sometimes worsens results** |

**Control for Pre-training Memory**: To verify that attacks aren't succeeding because LLMs simply "know" MovieLens, models were asked to complete histories. Recall rates for Llama3 / Mistral / GPT-OSS:120b were only 0.03% / 0.18% / 0.22%. **Conclusion**: Memorization signals originate from the prompt, not pre-training.

### Key Findings
- **Newer LLMs are more vulnerable**: Llama4 and GPT-OSS-120B are more susceptible than Llama3 or Gemma3, suggesting that stronger in-context memory creates a trade-off with privacy.
- **Total failure of Similarity attacks**: The geometric inconsistency between semantic and collaborative filtering embeddings renders traditional RecSys MIAs obsolete in the LLM era.
- **Defenses are a double-edged sword**: Instructing Mistral "not to mention prompt examples" actually increased attack advantage, as explicit safety prompts can unintentionally focus the model's attention on protected content.

## Highlights & Insights
- **Behavioral attacks as a new paradigm**: While previous MIAs were statistical games, this study transforms LLM behavioral traits—memorization, reasoning, and jailbreak resistance—into quantifiable attack tools.
- **Black-box effectiveness**: Achieving F1 scores $\ge 0.9$ without logits or shadow models is highly practical, as attackers only need API access.
- **Poisoning as a "stubbornness probe"**: This can be extended beyond MIA to detect training data bias or behavioral shifts after RLHF.

## Limitations & Future Work
- **Limitations**: Testing was restricted to 6 open-source models; closed-source flags like GPT-4 or Claude were not evaluated. The balanced 50/50 sampling may overstate effectiveness compared to real-world scenarios where non-members dominate.
- **Future Directions**: (1) Designing prompt-level Differential Privacy (DP-ICL); (2) using secret-sharer style canaries to measure leakage rates; (3) exploring privacy when items are stored in external KV stores rather than prompts.

## Related Work & Insights
- **vs. Traditional RecSys MIA**: Those methods rely on matrix embeddings; this work proves such mechanisms fail for LLMs and provides the "non-embedding" alternative.
- **vs. Wen et al. 2024**: While previous ICL MIAs focused on classification and relied on logits, this is the first study targeting **generative recommendation**.
- **Inspiration**: Any system placing sensitive data in a prompt should assume that content can be inferred. The "defense-induced exposure" phenomenon indicates that safety prompts represent an attack surface themselves.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic ICL-LLM-RecSys MIA; the Memorization and Poisoning designs are simple yet powerful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive testing across models, datasets, and configurations, including defensive evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear intuition and convincing visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant as industries adopt LLM-RecSys; functions as a timely warning for system designers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](../../NeurIPS2025/llm_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[ACL 2026\] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)

</div>

<!-- RELATED:END -->
