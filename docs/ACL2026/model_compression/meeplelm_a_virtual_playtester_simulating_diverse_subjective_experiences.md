---
title: >-
  [Paper Note] MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences
description: >-
  [ACL 2026][Model Compression][Virtual Playtesting] MeepleLM serves as a "virtual playtester" for board game designers. By providing official rulebooks and five distinct player personas to a fine-tuned Qwen3-8B…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Virtual Playtesting"
  - "MDA Reasoning"
  - "Persona Simulation"
  - "Board Game"
  - "Persona-conditional Fine-tuning"
date: 2026-05-08
content_hash: bcfcf6fa96f49a4a
---

# MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences

**Conference**: ACL 2026  
**arXiv**: [2601.07251](https://arxiv.org/abs/2601.07251)  
**Code**: https://github.com/leroy9472/MeepleLM  
**Area**: Human-AI Collaboration / LLM Simulation / Board Game Design  
**Keywords**: Virtual Playtesting, MDA Reasoning, Persona Simulation, Board Game, Persona-conditional Fine-tuning  

## TL;DR
MeepleLM serves as a "virtual playtester" for board game designers. By providing official rulebooks and five distinct player personas to a fine-tuned Qwen3-8B, the system generates ratings and reviews following a Mechanics→Dynamics→Aesthetics (MDA) three-step reasoning process. Evaluated on 207 games, it outperforms GPT-5.1 and Gemini3-Pro in community distribution alignment (Wasserstein 0.22 vs. GPT-5.1's 0.95), content diversity (Div 4.34 vs. 4.26), and Opinion Recovery (69.77 vs. 63.44), while achieving over 70% user preference in blind A/B testing.

## Background & Motivation
**Background**: LLMs have been utilized for various board game tasks, including chess agents, role-playing, social simulation, mechanism generation, and rule-code synthesis (Code World Models). However, as a "co-designer," no existing system provides feedback based on **authentic player experiences**. Current approaches are limited to rule validity checks or balancing through player LLM self-play (e.g., RuleSmith).

**Limitations of Prior Work**: (1) Feeding rulebooks to general LLMs (GPT-5.1, Gemini3) for reviews results in severe **central tendency bias**, where games receive average scores of 7-8, failing to capture the polarization found in player communities. (2) General LLM reviews often sound like marketing copy (e.g., using empty phrases like "social WD-40") and lack actual community jargon (e.g., "alpha gamer," "variant rules," "roll-and-move hell"). (3) Existing player simulations either use failed classifiers to learn objective features (e.g., DeBERTa misclassifying "house rules" as System Purist instead of Thrill Seeker) or rely on forward-model playtesting, which requires custom game engines for every title and cannot scale.

**Key Challenge**: (a) **Static Rules ↔ Emergent Experience**: Rulebooks are "code," but gameplay enjoyment is a causal chain that emerges at runtime. LLMs lack game engines to execute rules. (b) **Average Consensus ↔ Subjective Heterogeneity**: The same mechanic can evoke opposite reactions in different players (e.g., high randomness excites a Socializer but frustrates a Strategist). "One-size-fits-all" evaluations are useless for design.

**Goal**: (1) Construct a dataset of 1,727 rulebooks and 150K high-quality reviews covering both breadth and quality. (2) Explicitly model the "Rules → Experience" causal chain as a CoT using the classic Mechanics-Dynamics-Aesthetics (MDA) game design theory. (3) Cluster five player personas from 1.8M raw reviews in a data-driven manner. (4) Internalize persona-specific reasoning in Qwen3-8B via persona-conditional instruction tuning.

**Key Insight**: The MDA framework (Hunicke 2004) is inherently a causal model of "how mechanics trigger dynamics and how dynamics produce aesthetic experiences." Treating it as a CoT template externalizes the LLM's implicit reasoning into three verifiable steps.

**Core Idea**: `[Rule, Persona] -- MDA CoT --> Critique`, where the persona is not a label but a full semantic profile embedded in the system prompt. This forces the model to use the persona as a contextual prior to modulate the Dynamics→Aesthetics transition.

## Method

### Overall Architecture
A four-stage pipeline: (1) **Data Construction**: Hierarchical sampling of 1,727 BGG board games → PDF parsing with Mineru + structuring with Qwen3-235B + GPT-5.1 proofreading to obtain standard rulebooks; scraping 1.8M raw reviews → multi-task filtering by Qwen3-235B (hard filtering + MDA scoring + facet labeling) → stratified coverage-max sampling to retain 150K reviews (8% retention). (2) **Persona Discovery**: Embedding reviews using Qwen3-Embedding-8B (concatenated text + logic score + facets) → K-Means with $K=15$ → profiling centroids with GPT-5.1 → expert merging into 5 personas (System Purist, Efficiency Essentialist, Narrative Architect, Social Lubricator, Thrill Seeker) → GPT-5.1 majority voting (3 iterations) to label all data. (3) **MDA CoT Synthesis**: Using Qwen3-235B as a teacher to generate implicit reasoning chains $Z$ for each (rule, persona, review) triplet: "What mechanics are mentioned → What interactions did the dynamics trigger → What emotions did the aesthetics produce." A GPT-5.1 verifier ensures $Z$ is consistent with the ground-truth rating. (4) **Training**: LoRA fine-tuning of Qwen3-8B to maximize the joint likelihood of $P([Z; Y] | R, P_{profile})$.

### Key Designs

1. **MDA-Guided Reasoning as a Causal Mediator**:
    - **Function**: Explicitly decomposes the semantic gap between "rule text" and "player experience" into three verifiable CoT steps, enabling the LLM to simulate runtime experience before generating a critique.
    - **Mechanism**: While the original MDA is an analytical framework, this work uses it as a generative constraint: Step 1 (Mechanics) forces the model to ground its response in actual rules mentioned in the review; Step 2 (Dynamics) infers system behaviors or player interactions triggered by those rules; Step 3 (Aesthetics) synthesizes emotional reactions, modulated by the persona $P$. This is formalized as $[R, P] \xrightarrow{Z_{MDA}} Y$. A verifier checks for sentiment consistency between $Z$ and the rating.
    - **Design Motivation**: Direct $R \to Y$ mapping leads to shallow sentiment generation (central tendency bias). The three-step process forces the model to ground the rules, derive causality, and then express emotion, with each step being independently verifiable.

2. **Data-Driven Player Personas and Semantic Profile Injection**:
    - **Function**: Captures the subjective heterogeneity of the player community, avoiding "average player" evaluations.
    - **Mechanism**: (a) Cluster-then-Refine: K-Means with $K=15$ on review embeddings allows the data to define existing groups, followed by GPT-5.1 profiling and expert merging into 5 stable personas. (b) **Key Decision**: During training, $P$ is not a simple label token; instead, a full profile (core values, interaction preferences, pain points) is included in the system instruction. (c) At inference, $N=100$ samples are drawn based on the ground-truth persona distribution to aggregate a rating distribution.
    - **Design Motivation**: Experiments with a DeBERTa-v3-large classifier showed misclassifications of "house rules" as System Purist, proving persona is a product of subtle cognitive attribution that symbolic labels cannot capture. Semantic profiles allow the LLM to interpolate across persona dimensions using commonsense.

3. **Multi-dimensional Quality Filtering and Distribution Fidelity**:
    - **Function**: Extracts 150K high-quality critiques useful for designers from 1.8M raw reviews while preserving the original rating distribution.
    - **Mechanism**: (a) Hard filtering to remove noise (too short, off-topic, sentiment mismatch). (b) **MDA Scoring**: Three 1-5 dimensions (mechanism_anchoring, causal_attribution, constructiveness). (c) Facet Identification mapping reviews to 8 semantic topics (Rule Clarity, Cognitive Load, etc.). (d) Coverage-max stratified sampling to maintain rating distribution (Pearson $r=0.92$) and facet diversity.
    - **Design Motivation**: Filtered reviews were 1.24× longer than unfiltered ones with extreme ratings, proving increased information density.

### Loss & Training
LoRA on all linear layers via LLaMA-Factory. The objective is to maximize:
$$L = -\sum_{t=1}^{|S|} \log P(s_t | s_{<t}, R, P_{profile})$$
where $S = [Z; Y]$ is the sequence of the MDA reasoning chain followed by the critique (rating + review). Teacher: Qwen3-235B; Student: Qwen3-8B. Inference involves sampling $N=100$ times according to actual persona distributions.

## Key Experimental Results

### Main Results: 207 Held-out Games (Excerpts from Table 2)

| Model | MAE ↓ | Wasserstein ↓ | Kendall τ ↑ | Fact. ↑ | Dist-2 ↑ | Div. ↑ | Op-Rec ↑ |
|---|---|---|---|---|---|---|---|
| GPT-5.1 | 0.987 | 0.950 | 0.256 | **99.46** | 0.693 | 4.26 | 63.44 |
| Gemini3-Pro | 1.428 | 0.509 | 0.247 | 98.28 | 0.648 | 3.98 | 57.74 |
| Qwen3-235B | 1.229 | 0.635 | 0.145 | 98.95 | 0.657 | 3.56 | 54.27 |
| Qwen3-8B (backbone) | 0.891 | 1.012 | 0.049 | 97.88 | 0.594 | 1.58 | 11.39 |
| **MeepleLM (Ours)** | **0.658** | **0.221** | **0.282** | 98.86 | **0.712** | **4.34** | **69.77** |

**Key Finding**: The most significant comparison is Wasserstein 0.22 vs. GPT-5.1's 0.95 (4.3× improvement). GPT-5.1 suffers from mode collapse (safe scores of 7-8), while MeepleLM accurately reproduces the polarized community distribution.

### Ablation Study

| Configuration | MAE ↓ | WD ↓ | τ ↑ | Fact. ↑ | Div. ↑ | Op-Rec ↑ |
|---|---|---|---|---|---|---|
| **Full MeepleLM** | 0.658 | 0.221 | 0.282 | 98.86 | 4.34 | 69.77 |
| w/o MDA (No CoT) | 0.740 | 0.415 | 0.227 | 91.56 | 3.70 | 55.35 |
| w/o Persona (Generic) | 0.789 | 0.363 | 0.135 | 92.13 | 3.56 | 53.84 |
| w/o Rulebook (Blind) | 0.704 | 0.550 | 0.003 | **59.87** | 3.30 | 9.99 |

### Key Findings
- **Rulebooks are essential for factual grounding**: Removing the rulebook causes Fact. to drop from 98.86 to 59.87. Interestingly, MAE only slightly degrades, suggesting that "guessing the average" can lower MAE without providing design value.
- **Persona is key to Kendall τ**: Removing persona causes τ to drop from 0.282 to 0.135 (52% decrease). Without persona heterogeneity, the relative ranking of games across different groups is flattened into a mean estimation.
- **MDA is key to Op-Rec**: Removing MDA CoT drops Op-Rec from 69.77 to 55.35 (21% decrease), indicating that explicit causal reasoning allows the model to surface distinct viewpoints.

## Highlights & Insights
- **MDA as a CoT Template**: Adapting classic domain theories (Mechanics-Dynamics-Aesthetics) into LLM inference templates is a powerful "domain-theory-as-prompt" paradigm applicable to any "rules → experience" domain.
- **Central Tendency Bias in LLM Judges**: The usage of Wasserstein distance to quantify mode collapse serves as a warning for "LLM-as-judge" research; MAE alone significantly overestimates evaluation capabilities.
- **Data-Driven Persona Discovery**: The "Cluster-then-Refine" pipeline avoids the pitfalls of both purely algorithmic and purely expert-defined personas. Using semantic profiles instead of categorical labels allows for better generalization.

## Limitations & Future Work
- **Limitations**: (1) Relies solely on text, ignoring visual signals like card art or board iconography. (2) The 5 personas are coarse-grained and lack individual nuances.
- **Future Work**: Integration of multimodal grounding (visual encoders for board game components) and individual-level historical data for more granular simulation.

## Related Work & Insights
- **vs. General LLM-as-judge (G-Eval)**: Unlike static text evaluation, this work handles emergent experiences of interactive systems via MDA causal mediation.
- **vs. Forward-Model Playtesting**: While RL agents find balance bugs in game engines, MeepleLM derives subjective experiences directly from rulebooks without requiring an engine.
- **vs. Persona Modeling (Generative Agents)**: Instead of dialog history, this work uses review behavioral data for grounding, reducing stereotyping risks.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Systematic study of LLM-based board game testing; MDA-CoT is a clever integration of domain theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on 207 games with diverse metrics and significant testing.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear narrative and intuitive case studies.
- **Value**: ⭐⭐⭐⭐☆ Directly applicable to the board game industry; the methodology is transferable to HCI/UX and educational assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](../../ICML2026/model_compression/speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)
- [\[ACL 2026\] UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text](ukp_psycontrol_at_semeval-2026_task_2_modeling_valence_and_arousal_dynamics_from.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ACL 2026\] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)
- [\[ACL 2026\] GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts](glimprouter_efficient_collaborative_inference_by_glimpsing_one_token_of_thoughts.md)

</div>

<!-- RELATED:END -->
