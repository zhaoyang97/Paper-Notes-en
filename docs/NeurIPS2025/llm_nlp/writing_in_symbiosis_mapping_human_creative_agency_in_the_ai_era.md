---
title: >-
  [Paper Note] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era
description: >-
  [NeurIPS 2025][LLM/NLP][human-AI coevolution] Through longitudinal corpus analysis of 50,000+ documents, this paper proposes the "Dual-Track Evolution" hypothesis — that in the LLM era…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "human-AI coevolution"
  - "creative writing"
  - "stylometric analysis"
  - "authorial archetypes"
  - "LLM influence detection"
date: 2026-05-08
content_hash: ccf38708f7e8537c
---

# Writing in Symbiosis: Mapping Human Creative Agency in the AI Era

**Conference**: NeurIPS 2025
**arXiv**: [2512.13697](https://arxiv.org/abs/2512.13697)
**Code**: Not yet open-sourced (paper states code available on request)
**Area**: LLM/NLP
**Keywords**: human-AI coevolution, creative writing, stylometric analysis, authorial archetypes, LLM influence detection

## TL;DR

Through longitudinal corpus analysis of 50,000+ documents, this paper proposes the "Dual-Track Evolution" hypothesis — that in the LLM era, human writing exhibits thematic convergence alongside structural stylistic differentiation — and identifies three authorial adaptation archetypes: Adopters, Resistors, and Pragmatists.

## Background & Motivation

### State of the Field
The widespread adoption of large language models raises a fundamental question: as "creative writing" increasingly becomes a human–machine collaborative act, is the distinctiveness of human authorship disappearing?

### Limitations of Prior Work

**Oversimplified homogenization narrative**: Existing work primarily focuses on LLM-driven "stylistic homogenization" of internet and academic text, framing it as a unidirectional influence from AI to humans.

**Lack of individual-level resolution**: Most studies examine only aggregate trends, overlooking differential adaptation strategies at the individual author level.

**Conflation of topic and style**: Prior research frequently conflates thematic shifts with stylistic changes, lacking effective methods to disentangle the two.

**Absence of longitudinal controls**: No systematic within-author comparisons across the pre- and post-LLM boundary exist.

### Core Hypothesis

**Dual-Track Evolution Hypothesis**:
- **Track 1**: Universal thematic convergence toward AI-related topics.
- **Track 2**: Structural stylistic differentiation — rather than simple homogenization — at the level of writing style.

## Method

### Overall Architecture

The research design proceeds in three stages: corpus construction → feature engineering → multi-perspective analysis.

**Temporal boundary**: November 30, 2022 (the release date of ChatGPT) divides the timeline into pre-LLM and post-LLM periods.

### Corpus Construction

- **Scale**: 50,000+ documents (derived from 823k+ messages and papers)
- **Time span**: January 2021 – December 2024
- **Two genres**:
    - **Formal text**: arXiv computer science preprints (permissive license)
    - **Informal text**: Discord Unveiled dataset (CC BY 4.0), anonymized communications from public servers
- **AI reference corpus**: ShareGPT-90k (Apache-2.0) + Dolly-15k (CC BY-SA 3.0)
- **Sampling strategy**:
    - Discord: topic-controlled stratified sampling, balancing pre/post-LLM quotas across categories
    - arXiv: monthly sampling to ensure temporal continuity

### Key Designs: Perplexity-Gap Analysis

The core methodological contribution is the use of a **Perplexity Gap** to quantify stylistic temporal evolution:

$$\Delta_{ppl} = \frac{-\ln(p_{GPT2}(x))}{|x|_{chars}} - \frac{-\ln(p_{Llama}(x))}{|x|_{chars}}$$

- **Pre-LLM Judge**: GPT-2 Medium (355M), trained exclusively on pre-2022 data (847M tokens, ~41.7 hours on A100)
- **Current Baseline**: Llama-3-8B-base (frozen modern model)
- **Interpretation**: Text that is "easy" for the current model but "difficult" for the older model exhibits linguistic characteristics of the LLM era.

**AI-likeness index** (within-author z-score normalized):

$$AI_{likeness} = \frac{\Delta_{ppl} - \mu_{author}}{\sigma_{author}}$$

### Δ-Feature Vector

Each author's stylistic change is represented as a 7-dimensional normalized vector:

| Feature | Description |
|---------|-------------|
| $\Delta_{ppl}$ | Perplexity gap |
| $\Delta_{TTR}$ | Lexical diversity (type-token ratio) |
| $\Delta_{FKGL}$ | Readability grade (Flesch-Kincaid) |
| $\Delta_{passive\%}$ | Passive voice ratio |
| $\Delta_{1p\%}$ | First-person pronoun frequency |
| $\Delta_{punct}$ | Punctuation density |
| $\Delta_{sent\_len}$ | Mean sentence length |

All features are z-score normalized within each author across the temporal boundary.

### Statistical Controls

Fixed-effects model:

$$y_{it} = \beta \cdot PostLLM_t + \gamma \cdot len_{it} + \delta_c + \alpha_i + \epsilon_{it}$$

where $\alpha_i$ denotes author fixed effects, $\delta_c$ denotes server-category fixed effects, with HC3 robust standard errors and Holm-Bonferroni correction.

### Clustering Method

**HDBSCAN** is applied to the Δ-feature vectors (min_cluster_size=15, min_samples=5, metric=euclidean). Rationale for selecting HDBSCAN:
- Identifies clusters of varying shapes and densities
- Handles noise points automatically (without forced assignment)
- Allows genuine authorial archetypes to emerge organically from the data

## Key Experimental Results

### Main Results: Three Authorial Archetypes

Clustering analysis of 2,100 social-corpus authors reveals three behavioral patterns:

| Archetype | Count | Share | Core Characteristics |
|-----------|-------|-------|----------------------|
| **Resistors** | 442 | 21% | Low/negative perplexity gap; maintain pre-LLM linguistic complexity |
| **Adopters** | 370 | 18% | Highest perplexity gap; writing converges toward LLM style |
| **Pragmatists** | 866 | 41% | Moderate stylistic change + high engagement with AI topics |

**Clustering quality metrics**:
- Silhouette: 0.426 (95% CI: 0.419–0.433)
- ARI robustness: 0.891 (95% CI: 0.884–0.898)
- Bootstrap consistency: 89%
- Predictive AUC: 0.813

### Macro-Trend Validation

| Finding | Evidence |
|---------|----------|
| AI topic convergence | Both genres show significant increase in AI-related content after Nov 2022 |
| Thematic structural breakpoint | Q1 2023 (detected via PELT algorithm) |
| Stylistic complexity breakpoint | Q2 2023 (lagging behind thematic shift) |
| Social corpus perplexity gap increase | +23% (early 2023) |
| Formal corpus perplexity gap increase | +15% (early 2023) |

### Ablation Study: Dynamic Arc of Stylistic Adaptation

The most important finding is a **two-phase pattern**:

| Phase | Period | Social Corpus | Formal Corpus |
|-------|--------|---------------|---------------|
| Convergence phase | Early 2023 | Perplexity gap ↑23% | ↑15% |
| Avoidance phase | Late 2023–2024 | Perplexity gap ↓18% (from peak) | ↓12% |

This suggests that once LLM-like stylistic features become stigmatized, authors actively avoid them — particularly in formal contexts.

### Key Findings

- Cross-validation accuracy: 89.3% (95% CI: 87.1–91.5)
- Held-out arXiv data: 89.1% (86.8–91.4)
- Null model comparison: silhouette 0.31 vs. 0.43 (p<0.001)
- Temporal boundary robustness: 84% archetype assignment consistency (91% for extreme cases)
- AI-likeness remains significant after controlling for FKGL/TTR/sentence length: partial correlation r=0.34, p<0.001

## Highlights & Insights

1. The **Dual-Track Evolution framework** is an elegant theoretical contribution — unifying the seemingly contradictory phenomena of "convergence" and "differentiation" within a single model.
2. The **Perplexity-Gap method** ingeniously leverages the capability gap between language models from different eras to quantify stylistic change, avoiding circular reasoning.
3. The **two-phase dynamic arc** (initial convergence followed by avoidance) reveals how social pressure modulates writing style — awareness of AI detection and academic peer-review pressure both drive stylistic retreat.
4. **Important implications for AI detection**: the three archetypes imply that a simple human-vs-machine binary detection framework is insufficient — Adopters' texts are statistically closer to AI output than those of Resistors.
5. The majority of authors (Resistors + Pragmatists, totaling 62%) maintain non-AI stylistic signatures, suggesting that distinctive human expression remains both valued and actively preserved.

## Limitations & Future Work

1. **Observational rather than causal**: It cannot be confirmed that AI tool usage directly causes stylistic change; other confounding factors may be present.
2. **English only**: Findings may be biased toward English-speaking writing communities; different languages and cultures may exhibit different responses.
3. **Lack of direct human validation**: The archetypes are statistically constructed and have not been validated through participant studies (e.g., surveys of authors' actual AI usage behavior).
4. **Social corpus bias**: Discord users are not representative of all internet users.
5. **Potential for misuse**: The archetype framework could be exploited for author identity surveillance or unjust stylistic discrimination.
6. **Future directions**: Extension to multilingual settings, incorporation of causal inference designs, and integration of user surveys.

## Related Work & Insights

- **Relationship to Geng & Trotta (2025)**: The latter focuses on human–LLM co-evolution in academic writing; this paper extends that scope to social corpora and proposes individual-level analysis.
- **Tension with AI detection research**: Detectors assume a human/machine binary, but this paper demonstrates that assumption is no longer valid in a co-evolution context.
- **Scaffolded collaboration (Dhillon et al. 2024)**: Different strategies of scaffolded collaboration align with the archetypes identified in this paper.
- **Language simplification trends (Di Marco et al. 2024)**: Broader language simplification trends on social media may confound the assessment of AI influence.

**Insight**: When designing AI writing tools, the differential needs of each archetype should be considered — Resistors require tools that preserve distinctive voice; Adopters benefit from deep collaboration tools; Pragmatists need tools that support content exploration while protecting stylistic identity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "Dual-Track Evolution" hypothesis and individual-level archetype framework are relatively novel contributions to the field; the Perplexity-Gap method is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Large-scale corpus, multiple statistical controls, and thorough clustering robustness validation; however, human validation and multilingual experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and coherent narrative logic, progressing systematically from macro to micro levels.
- **Value**: ⭐⭐⭐⭐ — Significant implications for creative writing research in the AI era, AI detection, and human–computer interaction; practical application pathways remain to be explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](../../ACL2026/llm_nlp/xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[NeurIPS 2025\] Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companions](systematizing_llm_persona_design_a_four-quadrant_technical_taxonomy_for_ai_compa.md)
- [\[ICLR 2026\] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](../../ICLR2026/llm_nlp/optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](../../AAAI2026/llm_nlp/irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)

</div>

<!-- RELATED:END -->
