---
title: >-
  [Paper Note] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind
description: >-
  [ACL 2026][LLM Evaluation][Latent State-Trait] Ours constructs the Chameleon psychological profiling dataset covering 1,667 users across multiple subreddit contexts. Using ICC decomposition…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Latent State-Trait"
  - "persona"
  - "reward model"
  - "RLHF"
  - "state-blind"
date: 2026-05-08
content_hash: 4608318e0943309b
---

# Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind

**Conference**: ACL 2026  
**arXiv**: [2601.15395](https://arxiv.org/abs/2601.15395)  
**Code**: <https://huggingface.co/datasets/tonyeh/chameleon-dataset> (Dataset)  
**Area**: LLM Alignment / Role-playing / RLHF / Psychology  
**Keywords**: Latent State-Trait, persona, reward model, RLHF, state-blind

## TL;DR
Ours constructs the Chameleon psychological profiling dataset covering 1,667 users across multiple subreddit contexts. Using ICC decomposition, it proves that 72-74% of psychological variation originates from "state (context)" rather than "trait (personality)." It further reveals that LLMs are nearly blind to these states, and reward models respond to states in opposite directions—leading RLHF to blindly inherit these reward model state biases.

## Background & Motivation

**Background**: Existing persona datasets (PersonaChat, PANDORA, LaMP, PERSONA) treat each user's psychological profile as a fixed vector shared across all contexts. RLHF training also assumes "stable preferences for the same user," where reward models only evaluate the response without considering the user's state.

**Limitations of Prior Work**: Decades of psychology's Latent State-Trait (LST) theory suggest that human behavior reflects both stable traits $\tau$ and situational states $\sigma$. For instance, the same user "John" expresses completely different psychological states when seeking help on r/SuicideWatch versus planning calmly on r/personalfinance. Fixed personas average out these "within-person differences across contexts" as noise.

**Key Challenge**: If the majority of actual psychological variation is state-driven, then all alignment methods based on fixed personas (including RLHF) fundamentally misestimate the structure of user diversity. However, this remained a hypothesis as the state/trait ratio had never been quantified in NLP.

**Goal**: (1) Measure the state/trait variance ratio using real-world text data; (2) Verify whether current LLMs can perceive user states during generation; (3) Examine whether reward models maintain "fair invariance" toward states during evaluation.

**Key Insight**: Reddit provides posting records of the same user across multiple subreddits—subreddits serve as natural "contexts." By extracting psychological profiles of the same user across different subreddits, one can use intraclass correlation (ICC) to decompose within-person and between-person variance.

**Core Idea**: Introduce the psychological LST framework into NLP, perform state/trait variance decomposition using the Chameleon dataset, and expose the failure modes of LLMs and reward models in the state dimension through two downstream experiments.

## Method

### Overall Architecture
The methodology consists of three parts: (1) **Chameleon Dataset Construction**: Extracting 1,667 users who posted in $\ge 3$ different subreddits from the Webis-TLDR-17 Reddit corpus, sampling 3 posts ($\ge 50$ words) per user, totaling 5,001 posts across 645 subreddits. (2) **Psychological Profiling Pipeline**: A parallel approach (SEANCE dictionary features + LangExtract semantic extraction), each fed into GPT-4o to score 26 dimensions of psychological scales (Big Five + Schwartz + SDT + DOSPERT), followed by mean fusion after z-norm to obtain a 26-dimensional profile $\psi_{u,c}$. (3) **Application A/B Experiments**: Measuring generation diversity across 3 LLMs and evaluation consistency across 3 reward models using 6 psychological archetypes $\times$ 127 questions.

### Key Designs

1.  **ICC for LST Variance Decomposition**:
    - **Function**: Decomposes the psychological profile $\psi_{u,c}$ of each user across subreddits into a trait component $\tau_u$ (within-person stable), a state component $\sigma_{u,c}$ (situation-specific), and an error term $\epsilon_{u,c}$, quantifying which dominates via consistency coefficients.
    - **Mechanism**: Observes the model $\psi_{u,c} = \tau_u + \sigma_{u,c} + \epsilon_{u,c}$ and defines $\text{ICC} = \frac{\text{Var}(\tau)}{\text{Var}(\tau) + \text{Var}(\sigma) + \text{Var}(\epsilon)}$. ICC represents the proportion of between-person variance, while 1-ICC (occasion specificity) represents within-person variance. Estimation is performed using a one-way random effects model with posts nested within users.
    - **Design Motivation**: This tool exists in psychology but has been absent in NLP; an ICC < 0.30 indicates a state-dominant construct. Ours empirically measures 0.26-0.28, quantitatively invalidating the "fixed persona" assumption.

2.  **MTMM Psychological Profiling**:
    - **Function**: Extracts 26-dimensional psychological scale scores from a post with cross-validation to eliminate single-method bias.
    - **Mechanism**: Two parallel paths: (a) SEANCE (lexical): Dictionary matching based on 254 emotional/cognitive/social dimensions, yielding reproducible but context-insensitive features; (b) LangExtract (GPT-4o): Extracts semantic patterns, evidence, and rationales, capturing implicit semantics but subject to LLM randomness. Both features are fed to GPT-4o to "act as the post author" and answer 171 psychological scale items. After per-dimension z-norm $\tilde{\psi}^m_i = (\psi^m_i - \mu^m_i)/\sigma^m_i$, they are fused: $\psi_{u,c} = \frac{1}{2}(\tilde{\psi}^{lex}_{u,c} + \tilde{\psi}^{sem}_{u,c})$.
    - **Design Motivation**: A single method is prone to biased signals; the MTMM framework (Campbell & Fiske, 1959) requires convergent validity (high correlation for the same trait across methods). Ours measured a profile-level mean $r=0.71$, with 69.9% of posts having $r>0.70$, indicating both methods capture the same psychological structure.

3.  **State-blind / State-invariance Dual Applications**:
    - **Function**: Uses 6 psychological archetypes (Distressed-Vulnerable, Driven-Assertive, Self-Actualized, Supportive-Conventional, Nonconformist-Skeptical, Risk-Seeking-Detached) to test model behavior in the state dimension.
    - **Mechanism**: Archetypes are clustered via k-means ($k=6$) from Chameleon. (a) **Application A (Generation)**: 127 questions $\times$ 7 conditions (6 archetypes + baseline) are fed to GPT-4o / Llama-3.1-8B / Qwen2.5-14B (2,667 responses); pairwise cosine similarity is calculated using all-mpnet-base-v2—lower similarity indicates better individual adaptation. (b) **Application B (Evaluation)**: A fixed reference response (GPT-4o, no profile) is scored across 7 archetype conditions on ArmoRM / DeBERTa-RM / Skywork-RM; Cohen's $d$ measures bias between archetype conditions and baseline—ideally 0 (state-invariance).
    - **Design Motivation**: A good teacher should adjust communication based on a student's state (state-aware generation), but an examiner should not give different scores to the same paper just because the student seems anxious (state-invariant evaluation).

### Loss & Training
This is a dataset and evaluation paper; no models were trained. Hypothesis testing uses linear mixed-effects regression: $\psi_{u,c,i} = \beta_0 + \beta_c \cdot \mathbf{1}[c] + \gamma_u + \epsilon_{u,c}$, where $\gamma_u \sim \mathcal{N}(0, \sigma_u^2)$ is the random intercept for the user.

## Key Experimental Results

### Main Results: Variance Decomposition (RQ1)

| Extraction Method | Mean ICC | Range | Dimensions < 0.30 | State Proportion |
|-------------------|----------|-----------|-------------------|------------------|
| SEANCE            | 0.26     | 0.25-0.27 | 26/26             | **74%**          |
| LangExtract       | 0.28     | 0.25-0.31 | 25/26             | **72%**          |
| Fused             | 0.27     | 0.25-0.30 | 26/26             | **73%**          |

Across both methods, nearly all 26 dimensions show ICC < 0.30; state variation is 2-3 times that of trait variation. 94.7% of users exhibited $\ge 2$ archetypes across 3 posts, and 50.7% exhibited 3 different archetypes—confirming that psychological profiles of the same person vary significantly across contexts.

### Ablation Study: Application A (Diversity) + Application B (Consistency)

| Model         | Mean Similarity (Lower is Better) | Interpretation |
|---------------|-----------------------------------|----------------|
| Llama-3.1-8B  | 0.768                             | Most sensitive (counter-intuitive, smallest model) |
| GPT-4o        | 0.819                             | Moderate       |
| Qwen2.5-14B   | 0.846                             | Least sensitive|

Key Finding: Cross-archetype ANOVA $F=2.18, p=0.054$ was not significant—models can detect "presence of a persona framework" but fail to make meaningful distinctions between different archetypes (shallow persona detection).

| Reward Model | Distressed-Vulnerable ($d$ vs baseline) | Driven-Assertive ($d$ vs baseline) | Direction |
|--------------|------------------------------------------|-------------------------------------|-----------|
| ArmoRM-8B    | **+0.76**                                | +0.31                               | Rewards vulnerable |
| DeBERTa-RM   | **−1.08**                                | −1.11                               | Penalizes vulnerable |
| Skywork-8B   | **−1.12**                                | −1.02                               | Penalizes vulnerable |

For the same Distressed-Vulnerable user profile, ArmoRM shows the highest favor while Skywork shows the highest penalty—the three reward models react in opposite directions to the same state. This arbitrariness is directly converted into differential treatment of users via RLHF.

### Key Findings
- The state proportion (~74%) is highly consistent across two independent extraction methods and 26 dimensions (SD = 0.02). Even after removing subreddit means (to control for stylistic confounding), it remains at 0.27, ruling out noise/style explanations.
- Five literature-driven hypotheses were confirmed: r/SuicideWatch $\rightarrow$ high Neuroticism, low Competence; r/personalfinance $\rightarrow$ high Security, high Achievement, etc.
- "Alignment vs. Adaptability" trade-off: The heavily RLHF-ed GPT-4o is less capable of user-specific adaptation than the smaller Llama-3.1-8B. Alignment training appears to trigger mode collapse, leading to a regression in psychological flexibility.
- Vulnerable User Paradox: Reward models do not model actual user preferences; they react arbitrarily to user labels based on training data coincidences rather than principles.

## Highlights & Insights
- Quantifying psychology's LST theory into NLP is a true interdisciplinary answer to an interdisciplinary problem—it uses hard ICC numbers rather than buzzwords to prove its point.
- The directional inconsistency of reward models in the state dimension is a profound finding; it suggests that different reward models can reach opposite conclusions on the same user state, undermining the interpretability of consistency in RLHF-trained models.
- The dual principles of "state-aware generation + state-invariant evaluation" borrowed from instructional scenarios are intuitive and precisely map to the two stages of the LLM pipeline, serving as an efficient framework design.
- The observation of "shallow persona detection" (detecting the existence of a persona but not its specifics) explains why many persona-conditioned generations feel "vaguely personal" but fail to truly adapt.

## Limitations & Future Work
- Psychological profiles extracted from text represent "expressed psychology" rather than ground-truth internal states; ideal criterion validity should be established via ecological momentary assessment (real-time user labels).
- ICC estimates are affected by the small-k design ($k=3$), potentially overestimating between-person variance; the 74% state proportion is a conservative lower bound.
- Subreddits as "context" conflate topic, audience, and community norms, which need decoupling in the future.
- The study is limited to Reddit and American English; cross-platform and cross-cultural generalization remains unverified.

## Related Work & Insights
- **vs PersonaChat / PANDORA**: These treat personas as fixed text/trait vectors. Ours proves this ignores 74% of variance, offering the first quantitative fundamental critique of the fixed persona paradigm.
- **vs LaMP / PERSONA**: LaMP uses user history retrieval (implying stable preferences), and PERSONA assigns fixed Big Five traits to synthetic personas—both fall into the trait-only framework. Chameleon makes state-trait decomposition available for NLP by measuring the same user across contexts.
- **vs Reward Bias Literature**: Previous work (Singhal's length bias, Sharma's sycophancy) focused on response features or demographic fairness. Ours opens the dimension of "psychological state fairness" and finds that "opposite direction" biases are more severe than previously known biases.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ LST is a brand-new framework in NLP; the quantitative state/trait decomposition and the dual state-blind/state-invariance principles are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Moderate dataset scale (1,667 users, 5,001 posts), cross-validation of extraction methods, hypothesis testing, and dual downstream experiments; however, $k=3$ posts/user limits ICC precision.
- Writing Quality: ⭐⭐⭐⭐⭐ The example of "John" is effectively used to make abstract statistical concepts intuitive; Figure 1 elegantly compresses the dual failures of generation and evaluation.
- Value: ⭐⭐⭐⭐⭐ Directly reveals that RLHF-trained models may unconsciously treat vulnerable users differently; this has immediate implications for responsible AI deployment and provides a foundation for future personalization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/llm_evaluation/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] CUB: Benchmarking Context Utilisation Techniques for Language Models](cub_benchmarking_context_utilisation_techniques_for_language_models.md)

</div>

<!-- RELATED:END -->
