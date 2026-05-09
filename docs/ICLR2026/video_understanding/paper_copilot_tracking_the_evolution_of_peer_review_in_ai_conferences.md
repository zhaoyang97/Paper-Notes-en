---
title: >-
  [Paper Note] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences
description: >-
  [ICLR 2026][Video Understanding][peer review] Paper Copilot is constructed as a persistent digital archive and analysis platform for peer reviews spanning dozens of AI/ML venues. It adopts a tri-source hybrid data collection strategy—OpenReview API, web scraping, and community contributions—to archive real-time score snapshots capturing pre- and post-rebuttal dynamics. The platform reveals a structural anomaly in ICLR 2025: a counterintuitive decline in decision entropy, signaling a shift from probabilistic tiering to near-deterministic score-driven decision-making. LLM-driven author–affiliation metadata extraction further supports talent trajectory tracking.
tags:
  - ICLR 2026
  - Video Understanding
  - peer review
  - score dynamics
  - decision entropy
  - conference statistics
  - dataset
  - LLM metadata extraction
date: 2026-05-08
content_hash: bd2ce84d33cef910
---

# Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences

**Conference**: ICLR 2026
**arXiv**: [2510.13201](https://arxiv.org/abs/2510.13201)
**Code**: [Project Page](https://papercopilot.com)
**Area**: Scientometrics / Review Analysis
**Keywords**: peer review, score dynamics, decision entropy, conference statistics, dataset, LLM metadata extraction

## TL;DR

Paper Copilot is constructed as a persistent digital archive and analysis platform for peer reviews spanning dozens of AI/ML venues. It adopts a tri-source hybrid data collection strategy—OpenReview API, web scraping, and community contributions—to archive real-time score snapshots capturing pre- and post-rebuttal dynamics. The platform reveals a structural anomaly in ICLR 2025: a counterintuitive decline in decision entropy, signaling a shift from probabilistic tiering to near-deterministic score-driven decision-making. LLM-driven author–affiliation metadata extraction further supports talent trajectory tracking.

## Background & Motivation

**State of the Field**: Submission volumes at top AI/ML venues now exceed 10,000 per year (ICLR 2025: 11,672), placing unprecedented pressure on peer review. Some venues (ICLR/NeurIPS) adopt open review via OpenReview, while most (CVPR/AAAI/ICCV) remain closed. Review dimensions have also expanded from a single score to multi-dimensional assessments covering soundness, correctness, novelty, and contribution.

**Limitations of Prior Work**: (1) Review data is fragmented across social platforms such as Twitter, Reddit, Zhihu, and Xiaohongshu; (2) OpenReview overwrites earlier review versions—score histories during the rebuttal period constitute an irrecoverable information loss; (3) cross-venue and longitudinal comparisons of review standards lack a unified data source and toolset; (4) during the rebuttal window (only 1–2 weeks), authors lack statistical reference points to assess their score levels and the potential value of rebuttal.

**Root Cause**: The review process is central to research transparency, yet existing infrastructure cannot support systematic tracking of review dynamics or longitudinal analysis.

**Paper Goals**: To construct a unified platform for review data collection, archiving, and analysis that supports cross-venue longitudinal studies and real-time score dynamics tracking.

**Starting Point**: A tri-source hybrid data strategy maximizes coverage, while real-time snapshot archiving preserves otherwise irrecoverable historical data.

**Core Idea**: Consolidate scattered, ephemeral AI conference review information into a persistent, structured, and analyzable digital archive—establishing a "meta-scientific infrastructure" for the peer review process.

## Method

### Overall Architecture

Paper Copilot is a modular system comprising: a venue configuration layer → a multi-source data collection pipeline (multi-source assigners + worker pool + parallel bots) → cleaning and normalization → versioned datasets (JSON format, 30+ fields per paper) → backend storage/API (LAMP/MySQL) → frontend visualization and analysis (WordPress + custom JS). The system supports rapid onboarding of new venues with minimal configuration.

### Key Designs

1. **Tri-Source Hybrid Data Collection Pipeline**:
    - Function: Unified collection of review data from heterogeneous sources to maximize coverage.
    - Mechanism: (1) OpenReview API—scheduled scripts pull scores, confidence levels, and comments for open venues (ICLR/NeurIPS), storing timestamped snapshots to track pre- and post-rebuttal changes; (2) Web scraping—targeted extraction of accepted papers, authors, and metadata for venues without APIs (CVPR/AAAI); (3) Community opt-in contributions—authors from closed-review venues voluntarily submit their reviews (accumulating 6,584 valid records), with approximately 60% consenting to public release of anonymized scores.
    - Design Motivation: No single data source can cover all venues. For closed-review venues, community contribution is the only viable data channel.

2. **Review Dynamics Temporal Snapshot Archiving**:
    - Function: Real-time archiving of the complete evolution of review scores throughout the discussion and rebuttal phases.
    - Mechanism: Daily crawling of review snapshots for ICLR 2024/2025, recording all dimensional scores (rating, confidence, soundness, contribution, presentation) from each reviewer at each time point. OpenReview's official platform retains only the final version; earlier versions are overwritten and irrecoverable. Paper Copilot is the only publicly available archive on the internet preserving complete review time-series data. Score footprint visualizations trace the scoring trajectory of individual papers across multiple dimensions and reviewers.
    - Design Motivation: The review *process*—how scores change and how consensus forms—is as important as the final outcome, yet it had not previously been systematically preserved.

3. **LLM-Driven Author–Affiliation Metadata Extraction**:
    - Function: Large-scale automated extraction of structured author tuples $(a_i, \mathcal{A}_i, e_i)$ (name, affiliation set, email) from papers.
    - Mechanism: GLM-series models are used to extract metadata from camera-ready PDFs. A mismatch indicator $\mathbf{1}(x,y) = 1$ if $|x| \neq |y|$ is defined to evaluate structural consistency. The evaluation metric is Success Rate $= 1 - \frac{1}{|\mathcal{D}|} \sum_{i} (\delta_{\text{aff}}^i \lor \delta_{\text{email}}^i \lor \delta_{\text{parse}}^i)$. glm-4-plus achieves an 86.82% success rate on ~70K papers ($\delta_{\text{aff}} = 5.01\%$, $\delta_{\text{email}} = 4.94\%$, $\delta_{\text{parse}} = 0.81\%$).
    - Design Motivation: The vast majority of venues do not provide structured author–affiliation mappings, which are a prerequisite for institutional and country-level analyses.

### Analysis Methods

**Decision Entropy Analysis**: Quantifies the certainty of area chair (AC) decisions. For year $t$ and score bin $b$, the entropy is defined as $H_{t,b} = -\sum_{s \in \{\text{Reject, Poster, ...}\}} p_{t,b,s} \log p_{t,b,s}$, with the weighted average $\bar{H}_t = \sum_b w_{t,b} H_{t,b}$. Entropy typically grows logarithmically with submission volume as $\bar{H}_t \approx a \log X_t + b$; however, a strong negative residual in 2025 indicates anomalously high decision sensitivity $\kappa_{2025}$—ACs rely more heavily on average scores for deterministic tiering.

## Key Experimental Results

### Analysis of ICLR Review Evolution (2017–2025)

| Metric | Finding | Quantitative Evidence |
|------|------|----------|
| Submission growth | 490 → 11,672 (24×) | AC count increased from 31 to 823 |
| Decision entropy trend | Typically grows logarithmically with submissions | $\bar{H}_t \approx a\log X_t + b$ |
| 2025 structural shift | Decision entropy anomalously declines | $\text{resid}_{2025}$ deviates strongly negative from the fitted line |
| Rebuttal score changes | 54.8% of papers exhibit overall rating changes | soundness and other dimensions change only ~10–13% |
| Consensus evolution | Disagreement initially increases then converges after discussion begins | Oral converges fastest; Reject maintains high divergence |
| Boundary asymmetry | High-score, low-variance papers benefit for acceptance | Low-score, high-variance papers paradoxically benefit for acceptance |

### Community Transparency Survey (1,860 Responses Across 4 Venues)

| Venue | Responses | Agreed to Release Anonymized Reviews | Proportion |
|------|:------:|:------:|:------:|
| CVPR 2025 | 357 | 191 | 53.5% |
| ICML 2025 | 1,034 | 628 | 60.7% |
| ICCV 2025 | 254 | 151 | 59.4% |
| ACL 2025 | 215 | 145 | 67.4% |
| **Total** | **1,860** | **1,115** | **59.9%** |

### LLM Metadata Extraction Accuracy

| Model | $\delta_{\text{aff}}$ | $\delta_{\text{email}}$ | $\delta_{\text{parse}}$ | Success Rate |
|------|:------:|:------:|:------:|:------:|
| glm-4-plus | 5.01% | 4.94% | 0.81% | **86.82%** |
| glm-4-air | 49.98% | 17.11% | 0.51% | 44.73% |
| glm-4-flash | 76.39% | 43.27% | 0.62% | 18.52% |
| glm-3-turbo | 76.07% | 32.34% | 1.34% | 20.90% |

### Key Findings

- **Structural turning point in 2025**: Despite the largest-ever submission volume, decision entropy declined—ACs relied more heavily on average scores for acceptance decisions, shifting from probabilistic tiering to near-deterministic mapping.
- **Dual role of rebuttal**: Amplifies score changes for borderline papers while driving consensus formation for strong papers.
- **Spotlight converging toward Oral**: Average scores between tiers are increasingly separated, with Spotlight trending closer to Oral year over year.
- **Dimensional divergence in score changes**: Overall rating is the most frequently changed dimension during rebuttal; soundness and related dimensions change far less.

## Highlights & Insights

- **The only review time-series archive**: OpenReview overwrites earlier versions during discussion; Paper Copilot's real-time archiving preserves the only complete record of review dynamics available on the internet—an irreplaceable historical resource.
- **Decision entropy analysis framework**: An ordered-logit model combined with decision entropy provides a quantitative characterization of the evolution of the review system, elevating scattered community intuitions to rigorous meta-scientific analysis.
- **Integrity of ethical design**: The paper provides thorough discussion of data source compliance, privacy protection, re-identification risks, and dual-use safeguards, reflecting best practices in research ethics.

## Limitations & Future Work

- Closed-review venues rely on voluntary community submissions, introducing self-selection bias (authors with high or low scores may submit at different rates).
- Non-zero error rates in LLM-based affiliation extraction may affect the reliability of institutional ranking analyses.
- Author trajectory analysis could be repurposed for high-stakes assessments such as recruitment, posing dual-use risks.
- As a continuously updated live platform, exact reproduction of the system's state at any given point in time is not feasible.

## Related Work & Insights

- **vs. PeerRead (Kang et al., 2018)**: Covers 14.7K papers but is limited to specific venues and time snapshots; does not support longitudinal dynamic tracking.
- **vs. MOPRD (Lin et al., 2023)**: Multidisciplinary in scope but does not cover rebuttal dynamics or score time-series for AI venues.
- **vs. CSRankings**: Focuses on institutional rankings but updates slowly, lacks data source transparency, and contains no review data whatsoever.

## Rating

- Novelty: ⭐⭐⭐⭐ Distinctive tri-source data strategy and review time-series archiving; the decision entropy analysis framework is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale longitudinal analysis spanning ICLR 2017–2025 with well-grounded quantitative findings.
- Writing Quality: ⭐⭐⭐⭐ System description is clear; ethical discussion is comprehensive and detailed.
- Value: ⭐⭐⭐⭐⭐ Represents an infrastructure-level contribution to the AI research community; the combined value of the dataset and platform far exceeds that of a standalone paper.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[ICLR 2026\] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](../../CVPR2026/video_understanding/storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
