---
title: >-
  [Paper Note] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation
description: >-
  [ACL 2026][Social Computing][Community Notes] An empirical analysis of 30.8K health-related Community Notes on platform X reveals systemic slow-response issues (median delay of 17.6 hours for the first helpful verdict…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Community Notes"
  - "Misinformation Governance"
  - "Retrieval Augmentation"
  - "Hierarchical Evaluation"
  - "LLM-as-judge"
date: 2026-05-08
content_hash: 8639d29a72f065af
---

# Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation

**Conference**: ACL 2026  
**arXiv**: [2510.11423](https://arxiv.org/abs/2510.11423)  
**Code**: https://github.com/jiayingwu19/CrowdNotesPlus (Available)  
**Area**: Social Computing / Health Misinformation Governance / LLM Applications  
**Keywords**: Community Notes, Misinformation Governance, Retrieval Augmentation, Hierarchical Evaluation, LLM-as-judge

## TL;DR
An empirical analysis of 30.8K health-related Community Notes on platform X reveals systemic slow-response issues (median delay of 17.6 hours for the first helpful verdict; 87.9% of notes remain unrated). To address this, the authors propose CrowdNotes+, a framework featuring (1) **evidence-grounded augmentation** and (2) **utility-guided automation** for LLM note generation, paired with a "Relevance → Correctness → Helpfulness" hierarchical evaluation. Across 15 LLMs on a new benchmark, HealthNotes, the highest-performing model (o3) achieved 81.15% helpfulness, significantly outperforming the human baseline of 73.19%.

## Background & Motivation

**Background**: X’s Community Notes is currently the most influential **crowdsourced fact-checking system**, where users flag suspicious posts, write supplementary notes, and vote. Notes are only publicly displayed if they reach a "Currently Rated Helpful" status. Prior research focuses on voting dynamics, consensus formation, and polarization, generally assuming "the note already exists."

**Limitations of Prior Work**: A temporal analysis of 30,791 health notes over four years reveals two systemic bottlenecks: (1) **Slow note appearance**: The median delay from a misleading post to the first note is 10.4 hours, with another 7.2 hours to reach a verdict. (2) **Lack of ratings**: 87.9% of notes remain in "Needs More Ratings" and are never shown. Health misinformation (vaccines, outbreaks) often spreads within a window of a few hours.

**Key Challenge**: The "breadth" (anyone can write) and "depth" (sufficient voting) of the crowdsourcing model are incompatible in fast-evolving health misinformation scenarios. Manual fact-checking (e.g., FactCheck.org) is high-quality but slow; pure LLM alternatives (Singh et al. 2025) lack web search capabilities for emerging rumors; and recent methods (De et al. 2025) still require multiple human notes as input.

**Goal**: (1) Accelerate note creation via LLM-assisted or fully automated generation; (2) Improve evaluation accuracy to prevent voters from equating "fluency" with "correctness."

**Key Insight**: The authors identified a "loophole" in the current voting mechanism: 11.7% of notes labeled "Helpful" by humans contained irrelevant citations, and 14.0% misinterpreted their citations. This suggests voters often vote "Helpful" simply because a note is well-written. Breaking evaluation into three gated steps—Relevance, Correctness, and Helpfulness—can mitigate this.

**Core Idea**: **Use LLM agents to accelerate the entire Community Notes pipeline (writing + evaluation) and decompose evaluation into hierarchical conditional probabilities to prevent fluency from masking poor helpfulness**.

## Method

### Overall Architecture
The framework, **CrowdNotes+**, takes a misleading post $p$ as input and outputs a concise note (≤280 characters) with citation URLs. It features two generation modes and a tiered evaluation:

1. **Evidence-Grounded Note Augmentation**: Given a set of human-provided evidence URLs $\mathcal{E}_h$, the LLM follows a RETRIEVE → MATCH → GENERATE pipeline to synthesize note $n_h$.
2. **Utility-Guided Note Automation**: The LLM generates diverse queries $\mathcal{Q}$, retrieves a candidate pool $\mathcal{P}$ from the web, uses a **LLM-based utility module** to select top-$\tau$ evidence $\mathcal{E}_m$, and then proceeds through RETRIEVE/MATCH/GENERATE.
3. **Hierarchical Evaluation**: An LLM-judge sequentially verifies Relevance → Correctness → Helpfulness; a step is only reached if the previous gate is passed.

### Key Designs

1. **Utility-Guided Evidence Automation (Query Diversity + Utility Judgment)**:
    - **Function**: Enables the LLM to autonomously determine what to search, what to select, and what to write without human intervention.
    - **Mechanism**: Based on the finding that diverse queries provide complementary retrieval, the system generates a semantically diverse query set $\mathcal{Q}$. Candidates are merged and deduplicated: $\mathcal{P}=\text{dedup}\left(\bigcup_{q\in\mathcal{Q}}\text{TopK}(q)\right)$. A utility judge module then iteratively selects the highest-utility evidence snippets (prioritizing health authorities over general news) to form $\mathcal{E}_m$.
    - **Design Motivation**: Pure RAG often retrieves redundant or low-quality sources. Query diversification broadens recall, while the utility module refines the source quality. Removing either component leads to a 7–11 percentage point drop in helpfulness.

2. **Hierarchical Evaluation (Relevance/Correctness/Helpfulness)**:
    - **Function**: Closes the "fluency ≠ accuracy" loophole by splitting helpfulness into interpretable, auditable steps.
    - **Mechanism**: Uses three binary indicators ($R/C/H$) with conditional dependencies. The joint probability is $P(R=1, C=1, H=1) = P(H=1 \mid C=1, R=1) \cdot P(C=1 \mid R=1) \cdot P(R=1)$. The first two gates use GPT-4.1, while the final gate uses **HealthJudge** (fine-tuned Lingshu-7B).
    - **Design Motivation**: Standard Community Notes voting mixes these dimensions. By enforcing these gates, CrowdNotes+ identifies "false positives" where notes are fluent but inaccurate.

3. **HealthNotes Benchmark + HealthJudge Evaluator**:
    - **Function**: Provides a reproducible, domain-specific infrastructure for health note evaluation.
    - **Mechanism**: 1,268 post-note pairs covering seven categories (diseases, vaccines, etc.) were filtered from 30K notes. HealthJudge is an LLM fine-tuned on expert-annotated data to be a more reliable judge of medical guidelines than general-purpose LLMs.

### Loss & Training
The study does not train the generation models but performs Supervised Fine-Tuning (SFT) on HealthJudge (Lingshu-7B) using expert data. All notes are strictly truncated to 280 characters to match platform constraints. Retrieval quotas and time windows in the Automation mode are strictly aligned with original human note metadata to ensure fairness.

## Key Experimental Results

### Main Results: 15 LLMs vs. Human Baseline on HealthNotes (C=Correctness, H=Helpfulness, R=Relevance, in %)

| Model Group | Model | Aug. C (Helpful) | Aug. H (Helpful) | Auto. H (Helpful) | Auto. H (Not Helpful) | Overall H |
|---|---|---|---|---|---|---|
| — | **Human baseline** | 75.24 | 73.19 | 73.19 | 5.52 | 39.36 |
| G1 LRM | o3† | 87.70 | 86.91 ↑ | 92.11 ↑ | 70.19 ↑ | **81.15** ↑ |
| G1 LRM | Gemini-2.5-pro† | 88.64 | 85.65 ↑ | 91.17 ↑ | 69.24 ↑ | 80.21 ↑ |
| G2 LLM | GPT-4.1 | 87.85 | 85.80 ↑ | 88.49 ↑ | 69.87 ↑ | 79.18 ↑ |
| G4 medical | MedGemma-27B | 84.38 | 79.02 ↑ | 79.81 ↑ | 58.68 ↑ | **69.25** ↑ |

→ **Models >14B generally outperform humans in helpfulness**. Closed-source LRMs (o3) in automation mode score 64.7pp higher than humans on the "Not Helpful" subset, proving LLMs are particularly effective for "difficult cases."

### Ablation Study: Contribution of Utility-Guided Automation Components (H, %)

| Configuration | Helpful | Not Helpful | Overall |
|---|---|---|---|
| **CrowdNotes+ (o3) Full** | **92.11** | **70.19** | **81.15** |
| — Query Diversity | 79.50 | 69.09 | 74.30 (-6.85) |
| — Utility Judgment | 79.02 | 64.83 | 71.93 (-9.22) |

### Key Findings
- **Utility judgment is more critical than query diversity**: Removing the utility module caused a larger drop (~9-11pp) than removing query diversity. "Selecting the right evidence" is harder than "finding evidence."
- **Reasoning models > General LLMs > Medical-tuned LLMs**: o3 was the only model to break 80% overall helpfulness, as explicit reasoning traces help in selecting from multi-source evidence.
- **Human blind tests favor CrowdNotes+**: In a blind test of 100 pairs, o3-generated notes had an 87% win rate against human notes.
- **89 Human "Helpful" notes were actually erroneous**: Hierarchical evaluation identified errors centered on lack of evidence support, misinterpretation of sources, and overgeneralization.

## Highlights & Insights
- **The hierarchical decomposition $P(R) \cdot P(C|R) \cdot P(H|C,R)$ is elegant**: It maps directly to "Is the source real? → Is the interpretation correct? → Is it helpful?", making every step auditable and rejectable.
- **Quantifying the "fluency bias" in crowdsourcing**: By comparing human scores with the hierarchical judge, the authors provide empirical evidence of a vulnerability in existing governance systems.
- **Utility judgment module is highly reusable**: Using LLMs as rerankers for evidence selection is more robust than simple cosine similarity and can be applied to general RAG systems.

## Limitations & Future Work
- The authors acknowledge that the study is limited to English X and that HealthJudge’s data scale restricts its ceiling for professional medical judgment.
- **Self-evaluation bias**: Since GPT-4.1 serves as both a baseline and part of the hierarchical judge, results may be slightly inflated; a purely independent panel of physicians might yield different absolute numbers.
- **Truncation**: The 280-character limit might unfairly penalize notes that were coherent before being truncated.
- **Future Directions**: Implementing multi-turn dialogues to allow LLM notes to respond to rebuttals, and weights for source trust levels (e.g., Journals > Media).

## Related Work & Insights
- **vs. De et al. 2025**: Their work requires human notes as input for aggregation; Ours supports full automation.
- **vs. Singh et al. 2025**: They rely on internal LLM knowledge; Ours uses utility-guided web retrieval to handle emerging rumors.
- **Insight**: The hierarchical conditional probability evaluation paradigm can be transferred to RAG citation quality assessment or uncertainty grading in high-stakes agent workflows.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of dual-mode generation and hierarchical evaluation in the Community Notes context is systematic and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Testing 15 LLMs across multiple modes and dimensions, including human review and error analysis, is extensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The structure from empirical analysis (§3) to framework (§4) and benchmark (§5) is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses social governance infrastructure; the open-sourcing of HealthNotes and HealthJudge is of high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[AAAI 2026\] T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](../../AAAI2026/social_computing/t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ACL 2026\] Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation](estimating_the_black-box_llm_uncertainty_with_distribution-aligned_adversarial_d.md)

</div>

<!-- RELATED:END -->
