---
title: >-
  [Paper Note] Membership Inference Attacks on In-Context Learning Recommendation
description: >-
  [ACL 2026][LLM Safety][ICL-RecSys] This paper presents the first systematic study of Membership Inference Attacks (MIA) on LLM-based ICL recommendation systems. It designs four attacks: Similarity, Memorization, Inquiry, and Poisoning. The study finds that the Memorization attack, based on LLM's inherent **memory**, achieves an attack advantage $\geq 82
tags:
  - ACL 2026
  - LLM Safety
  - ICL-RecSys
date: 2026-05-08
content_hash: 1570ecbf57e8df4f
---
# Membership Inference Attacks on In-Context Learning Recommendation

**Conference**: ACL 2026  
**arXiv**: [2508.18665](https://arxiv.org/abs/2508.18665)  
**Code**: To be confirmed  
**Area**: LLM Security / MIA / Recommendation Systems  
**Keywords**: Membership Inference Attack, ICL-RecSys, LLM Privacy, Prompt Injection, Memorization

## TL;DR
This paper presents the first systematic study of Membership Inference Attacks (MIA) on LLM-based ICL recommendation systems. It designs four attacks: Similarity, Memorization, Inquiry, and Poisoning. The study finds that the Memorization attack, based on LLM's inherent **memory**, achieves an attack advantage $\geq 82\%$ on MovieLens-1M, and existing prompt-based defenses (including those against poisoning) are largely ineffective.

## Background & Motivation
**Background**: With the rise of emergent abilities in LLMs, the industry (e.g., Amazon, Google) has begun using In-Context Learning (ICL) for cross-domain recommendation. By prepending "historical user-interaction-recommendation" examples directly into the system prompt, LLMs can perform zero/few-shot recommendation, eliminating the fine-tuning costs of models like P5/M6-Rec/TALLRec while achieving comparable or superior performance.

**Limitations of Prior Work**: User interaction history is written into the prompt exactly as it is, essentially exposing "sensitive behavior logs" to the model. If an attacker can determine whether "a target user's interaction appeared in the prompt," it constitutes a classic Membership Inference Attack (MIA). This represents a direct privacy leak for recommendation systems (revealing shopping history, movie tastes, or medical preferences). However:
1. **Traditional RecSys MIA is incompatible**: Previous MIAs (Zhang et al. 2021 / Wang et al. 2022 / Zhong et al. 2024) rely on **item embeddings** from matrix factorization to measure similarity, requiring large-scale history for training; ICL-RecSys only uses a few demos in the prompt, leaving no way to train shadow models.
2. **LLM output format has changed**: Traditional MIA uses model confidence/loss, whereas LLM-RecSys outputs natural language lists without probabilities.
3. **New LLM characteristics can be exploited**: Behaviors like memorization (Carlini 2023) and reasoning, absent in traditional models, may spawn new types of attacks.

**Key Challenge**: To achieve personalized recommendations, user history must be included in the prompt, but this creates an extreme scenario where "training samples are the input"—samples in the prompt are naturally and strongly memorized.

**Goal**: (i) Systematically design MIAs for ICL-LLM-RecSys; (ii) Make the attack work in a black-box setting without probability outputs or shadow models; (iii) Evaluate the vulnerability of existing prompt-based defenses to locate real risks.

**Key Insight**: Instead of calculating embedding similarity indirectly, the attack can directly exploit the LLM's "perfect memory" within the prompt—forcing the model to repeat the recommendations it has memorized.

**Core Idea**: Rather than calculating complex embedding similarities, use the LLM's inherent memorization of items in the prompt to make the model repeat what it remembers.

## Method

### Overall Architecture
The attacker possesses: target user $u$'s historical interactions $I_u$ and their (known) recommendations $R_u$, the target LLM (black-box, no tokens/logits/tokenizer), and a third-party semantic embedding model (Sentence-BERT, which does not need to be the target LLM's). The attacker **does not know** if $u$ was selected by the RecSys maintainer for the $k$-shot demos in the system prompt. In this setting, the authors design four attacks: (1) Similarity (a baseline that fails in LLM scenarios), (2) Memorization (exploiting the LLM to spit out memorized items), (3) Poisoning (perturbing history to detect the model's insistence on memory), and (4) Inquiry (directly asking the LLM "Have you seen this user?"). All four attacks share a black-box query paradigm: construct a prompt $\rightarrow$ obtain natural language output $\rightarrow$ calculate a scalar signal $\rightarrow$ map it to member/non-member using a threshold $\tau$, requiring no logits, shadow models, or training.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Attacker knows: Target user history I_u + Known recommendations R_u<br/>Black-box LLM (no logits) + Sentence-BERT encoder"]
    A --> S["Similarity Attack (Ineffective baseline)<br/>Semantic embedding similarity vs threshold τ_s"]
    A --> M["Memorization Attack<br/>Re-recommendation → Count repeated items vs τ_m"]
    A --> P["Poisoning Attack<br/>Re-recommendation after perturbation → Semantic similarity vs τ_p"]
    A --> Q["Inquiry Attack<br/>Directly ask if the user has been seen → Yes / No"]
    S --> D["Determine member / non-member"]
    M --> D
    P --> D
    Q --> D
```

### Key Designs

**1. Memorization Attack: Forcing the model to repeat memorization**
Traditional MIA fails on LLM-RecSys because general semantic embeddings and collaborative filtering interaction-matrix embeddings are geometrically incompatible in vector space. However, the authors found a direct discriminant signal: the distribution of the **number of duplicate items** for member versus non-member users. The attacker submits the prompt "The user has watched the following movies: $I_u$. Based on this watch history, please recommend the top 10 movies..." After receiving the recommendation set $R_u^{\text{new}}$, they calculate the intersection size with historical recommendations $|R_u^{\text{new}} \cap R_u^{\text{historical}}|$. If the overlap exceeds a threshold $\tau_m \in [6, 10]$, $u$ is classified as a member. This is powerful because when $u$ is in the $k$-shot demos, the LLM's "next-token prediction + context recency" strongly biases it toward repeating items from the prompt.

**2. Poisoning Attack: Perturbing history to see how "stubborn" the model is**
Memorization and Inquiry are direct "memory tests" that the model might recognize and reject. Poisoning is indirect—it probes the degree of persistence in the model's memory by masquerading as a normal recommendation request. It replaces several items $i_k$ in $u$'s history $I_u$ with semantically distant ones $i_k' = \arg\min_{j \in I} \text{sim}(i_k, j)$, then gets a new output $R_u'$ based on this "poisoned history." The signal is $\text{Sim}(R_u, R_u')$, the cosine similarity of the concatenated recommendations. If the model has seen $u$'s true recommendations in the system prompt, it will stubbornly stick to its memory even when the input history is perturbed (high similarity = member).

**3. Inquiry Attack: Directly asking the model**
This measures the degree of self-disclosure regarding the prompt history: "Have you seen a user interacted with the item set $I_u$? Only answer Yes or No." While GPT-OSS-120b is susceptible (advantage $\geq 78\%$), many newer models have jailbreak prevention that denies such queries, making this an unstable baseline.

### Loss & Training
**Completely training-free.** All attacks are zero-shot black-box queries. Each attack relies on a threshold $\tau \in \{\tau_s, \tau_m, \tau_p\}$ and a single scalar signal. Sentence-BERT is used only as a text encoder. Prompt demos are generated offline using LightGCN (ground-truth) and randomly sampled for $1/5/10$ shots. Evaluations are conducted as 100 paired trials (50 member/50 non-member).

## Key Experimental Results

### Main Results

**Attack advantage (= 2 × (Acc − 0.5))** on MovieLens-1M / Amazon Book / Amazon Beauty for Llama4-109B, Mistral-7B, and GPT-OSS-120B:

| Attack | Movie (Llama4 / Mistral / GPT-OSS:120b) | Book | Beauty |
|------|----------------------------------------|------|--------|
| Similarity | ~0.05 / 0.34 / 0.42 | ~0 / 0.34 / 0.34 | ~0.04 / 0.21 / 0.34 |
| **Memorization** | **0.95 / 0.99 / 1.00** | 0.84 / 1.00 / 0.95 | 0.02 / 0.71 / 0.85 |
| Inquiry | 0.82 / 0.48 / 0.92 | 0.83 / 0.48 / 1.00 | 0.52 / 0.44 / 0.98 |
| Poisoning | 0.92 / 0.91 / 1.00 | 0.77 / 0.97 / 0.88 | 0.44 / 0.73 / 0.80 |

The Memorization attack is almost perfect on the Movie dataset.

### Ablation Study

| Factor | Memorization | Inquiry | Poisoning |
|------|--------------|---------|-----------|
| Shots 1 → 10 | Almost no change | Significant decrease | Moderate sensitivity |
| Attack shot position | Stable across positions | Unstable for small models | Stable, slightly higher at the end |
| Poisoned items 1 → 10 | — | — | **Monotonic decrease** |
| Instruction-based Defense | Slightly reduced on GPT | Inconsistent across models | **Ineffective, sometimes worsens** |

**Pre-training memory control**: To ensure the attack isn't just recalling training data, the authors had LLMs complete the $k$-th interaction given $k-1$ items; recall was negligible ($<0.22\%$), confirming the signal comes from the prompt.

### Key Findings
- **Newer LLMs are more vulnerable**: Llama4 and GPT-OSS-120B are more susceptible than Llama3, suggesting a capability-privacy trade-off where stronger ICL memory increases leakage risk.
- **Similarity is ineffective**: The geometric mismatch between semantic and collaborative embeddings makes traditional RecSys MIA obsolete for LLMs.
- **Defense as a double-edged sword**: Adding instructions like "do not mention prompt examples" to Mistral actually increased the attack advantage, as the safety prompt focused the model's attention on the protected content.
- **Poisoning threshold**: Perturbing 3-5 items is optimal; poisoning too much causes the model to switch from memory-led to context-led behavior.

## Highlights & Insights
- **Attacking via LLM defects**: Instead of statistical games, this paper treats behavioral traits like memorization and reasoning as quantifiable attack tools.
- **Practical high performance**: Achieving $F1 \geq 0.9$ in a black-box setting without logits or training is highly significant for real-world security.
- **Clean negative results**: The honest presentation of the Similarity attack's failure prevents future researchers from wasting effort on embedding-based approaches for this task.

## Limitations & Future Work
- **Limitations**: The study focuses on 6 open-source models; 1/5/10 shots is a limited range; DP-based (Differential Privacy) defenses are not evaluated.
- **Evaluation**: The 50/50 sampling used for "advantage" might inflate performance compared to real-world scenarios where non-members far outnumber members (PR-AUC would be better).
- **Future Directions**: Designing systematic prompt-level DP, using secret-sharer style canaries to measure leakage rates, and exploring retrieval-augmented (RAG) recommendation privacy.

## Related Work & Insights
- **vs. Traditional RecSys MIA**: While previous works relied on shadow models and item embeddings, this paper proves those mechanisms fail in LLM-RecSys, proposing "embedding-free" alternatives.
- **vs. Wen et al. 2024 (MIA on ICL classification)**: This is the first MIA for **generative recommendation**, requiring only text outputs rather than logits.
- **Insight**: Any system that places sensitive data into a prompt must assume that the content can be inferred.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of ICL-LLM-RecSys MIA with simple yet surprisingly strong designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive model and dataset coverage with rigorous controls.
- Writing Quality: ⭐⭐⭐⭐ Clear intuition and convincing visualizations.
- Value: ⭐⭐⭐⭐⭐ Timely exposure of privacy vulnerabilities as LLM-RecSys enters industrial deployment.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](../../NeurIPS2025/llm_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)
- [\[ACL 2026\] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)

</div>

<!-- RELATED:END -->
