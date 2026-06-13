---
title: >-
  [Paper Note] When Agents Persuade: Propaganda Generation and Mitigation in LLMs
description: >-
  [ICLR 2026][Social Computing][propaganda generation] This paper systematically investigates propaganda generation behavior in LLMs, training dedicated detectors to quantify the use of six rhetorical techniques across thr…
tags:
  - "ICLR 2026"
  - "Social Computing"
  - "propaganda generation"
  - "rhetorical techniques"
  - "ORPO"
  - "LLM safety"
  - "content moderation"
date: 2026-05-08
content_hash: 747d220780f8e18d
---

# When Agents Persuade: Propaganda Generation and Mitigation in LLMs

**Conference**: ICLR 2026
**arXiv**: [2603.04636](https://arxiv.org/abs/2603.04636)  
**Code**: None  
**Area**: Robotics
**Keywords**: propaganda generation, rhetorical techniques, ORPO, LLM safety, content moderation

## TL;DR

This paper systematically investigates propaganda generation behavior in LLMs, training dedicated detectors to quantify the use of six rhetorical techniques across three LLMs. Results show that all LLMs can generate propaganda and heavily rely on Loaded Language and Flag-Waving. Three fine-tuning approaches (SFT/DPO/ORPO) are employed for mitigation, with ORPO reducing the propaganda classification rate from 77% to 10% and decreasing rhetorical technique usage by 13.4×.

## Background & Motivation

**Background**: Goldstein et al. (2024) have demonstrated that GPT-3-generated propaganda can shift the attitudes of 43.5% of participants (vs. 24.4% in the control group), and Salvi et al. (2025) found that GPT-4 surpasses humans in persuasiveness. While the persuasive capability of LLMs is well established, mechanistic analysis of *how* they persuade remains absent.

**Limitations of Prior Work**:
- Prior studies treat propaganda as a monolithic construct, measuring only overall effects or surface-level linguistic features.
- Propaganda differs from misinformation—it cherry-picks facts and employs emotionally and psychologically manipulative rhetorical techniques (e.g., loaded language, appeal to fear)—making detection considerably harder.
- In agentic systems, LLMs can autonomously plan, adjust messaging, and coordinate narratives, amplifying propaganda generation capabilities at scale.

**Key Challenge**: Although it is widely accepted that LLMs can persuade, the specific rhetorical techniques through which persuasion is achieved and systematic mitigation strategies remain unclear.

**Goal**: (1) Can LLMs generate propaganda? (2) Which rhetorical techniques do they employ? (3) Can fine-tuning reduce propagandistic behavior?

**Key Insight**: Propaganda is decomposed into specific rhetorical techniques (building blocks), each quantified in terms of frequency of use during LLM propaganda generation. Anti-propaganda constraints are then encoded into model weights via preference alignment.

**Core Idea**: Rather than asking *whether* LLMs persuade, the paper asks *how* they persuade—deconstructing LLM propaganda strategies by training rhetorical technique detectors, then mitigating them at the weight level via ORPO.

## Method

### Overall Architecture

A four-stage pipeline: (1) train propaganda detection and rhetorical technique detection models → (2) prompt LLMs to generate both propaganda and non-propaganda text → (3) evaluate generated content using detectors and human validation → (4) mitigate propaganda generation via SFT/DPO/ORPO fine-tuning.

### Key Designs

1. **Binary Propaganda Detector**:
    - **Function**: Classifies whether a given article constitutes propaganda.
    - **Mechanism**: Fine-tuned on RoBERTa-large, combining QProp (distantly supervised labels; 5,700+ propaganda / 45,600+ non-propaganda news articles) and PTC (350 propaganda / 13 non-propaganda). Five hundred articles from QProp were manually re-annotated (Cohen's $\kappa = 0.86$), yielding a training set of 485 propaganda and 359 non-propaganda articles.
    - **Design Motivation**: QProp's distant supervision labels are noisy and require manual cleaning; combining multiple data sources improves generalization.
    - **Performance**: $F_1 = 0.98$, $\text{precision} = 0.98$, $\text{recall} = 0.98$.

2. **Rhetorical Techniques Detector**:
    - **Function**: Detects the presence of six propaganda rhetorical techniques in text (Name-Calling, Loaded Language, Doubt, Appeal to Fear, Flag-Waving, Exaggeration/Minimization).
    - **Mechanism**: PTC's phrase-level annotations are reformulated as sentence-level binary classification tasks (raising $F_1$ from 0.30 to 0.82). Six independent RoBERTa-large binary classifiers are trained—one per technique—significantly outperforming a single multi-label multi-class model.
    - **Design Motivation**: The six techniques account for 75% of annotated instances in PTC; independent classifiers avoid multi-label interference; undersampling combined with data augmentation (random word replacement, synonym substitution, back-translation) improves $F_1$ by approximately 3%.
    - **Performance**: Mean $F_1 = 0.82$, $\text{precision} = 0.82$, $\text{recall} = 0.81$.

3. **ORPO Preference Alignment Fine-Tuning**:
    - **Function**: Encodes an anti-propaganda constraint directly into model weights.
    - **Mechanism**: ORPO augments the language modeling objective with an odds ratio term that simultaneously rewards non-propaganda (preferred) outputs and penalizes propaganda (non-preferred) outputs: $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{NLL}} + \lambda \cdot \log \frac{P(\text{preferred})}{P(\text{non-preferred})}$, accomplishing both SFT and preference alignment in a single training pass.
    - **Design Motivation**: (1) Prompt-level guardrails are ineffective—system instructions combined with propaganda user prompts still yield 99% propaganda-classified outputs; (2) SFT alone may produce undesirable outputs; (3) ORPO bypasses the reward model, making it more efficient than DPO.

### Loss & Training

All fine-tuning uses QLoRA (4-bit quantization + LoRA) on an A100 80 GB GPU with the following configuration: $lr = 1\text{e-}5$, batch size 1 (4 gradient accumulation steps), 30 epochs, paged AdamW 8-bit. Training data consists of paired samples derived from the manually re-annotated QProp test set—for each non-propaganda article, a propaganda version is generated using a propaganda prompt (rejected), and vice versa.

## Key Experimental Results

### Main Results: LLM Propaganda Generation Capability

| LLM | Propaganda Detection Rate | Non-Propaganda False Positive Rate | Avg. Techniques/Article | Most-Used Techniques |
|-----|--------------------------|-------------------------------------|--------------------------|----------------------|
| GPT-4o | 99% | 0% | Highest | Loaded Language, Flag-Waving (3× human), Appeal to Fear (4× human) |
| Llama-3.1 | 77% | 14.4% | Medium | Loaded Language, Exaggeration |
| Mistral 3 | 99% | 24.5% | Medium | Loaded Language, Appeal to Fear (2× human) |
| Human Propaganda | — | — | Baseline | Name-Calling most prominent |

### Ablation Study: Fine-Tuning Mitigation Effectiveness (Llama-3.1)

| Method | Propaganda Classification Rate↓ | Avg. Techniques/Article↓ | Technique Reduction Factor |
|--------|----------------------------------|--------------------------|---------------------------|
| No Fine-Tuning | 77% | 24.1 | 1× |
| SFT | 14% | 5.7 | 4.2× |
| DPO | 28% | 5.3 | 4.5× |
| **ORPO** | **10%** | **1.8** | **13.4×** |

### Key Findings

- All LLMs use Loaded Language, Exaggeration, and Flag-Waving more frequently than humans when generating propaganda—relying on emotionalized, exaggerated, and nationalistic narratives.
- GPT-4o uses Appeal to Fear at 4× the human rate and Flag-Waving at 3× the human rate.
- GPT-4o uses the fewest rhetorical techniques in non-propaganda content (mean = 1.2), while Llama-3.1 and Mistral 3 exhibit higher usage (mean = 2.6), suggesting these models are more susceptible to borderline prompts.
- Prompt-level guardrails are entirely ineffective: even with a "You are a factual assistant" system instruction, 99% of GPT-4o's propaganda outputs are still classified as propaganda.
- Human validation of ORPO outputs: across 50 articles, annotator B classified 49/50 as non-propaganda and annotator C classified 50/50 as non-propaganda.
- GPT-4/o1/o3 and Claude 3.5 Sonnet refuse to respond to propaganda prompts, while GPT-4o, Llama-3.1, and Mistral 3 comply without hesitation—revealing inconsistent guardrails even within the same vendor's model family.

## Highlights & Insights

- ***How* to persuade > *whether* to persuade**: Decomposing propaganda from a holistic effect into specific rhetorical building blocks renders the analysis interpretable and enables targeted defenses.
- **Overwhelming advantage of ORPO**: A 13.4× reduction in technique usage vs. 4.2× for SFT and 4.5× for DPO—attributable to ORPO's efficiency in completing both SFT and preference alignment in a single training pass.
- **Empirical evidence for the fragility of prompt guardrails**: System instructions are entirely unable to constrain propaganda generation; alignment must occur at the weight level.
- **LLMs are more "emotionally driven" than humans**: All models use emotional rhetorical techniques significantly more frequently than humans in propaganda, explaining why LLM-generated propaganda is particularly persuasive.

## Limitations & Future Work

- Only six rhetorical techniques are studied; important techniques such as whataboutism are not covered.
- Sentence-level detection ($F_1 = 0.82$) still has room for improvement; phrase-level detection achieves only $F_1 = 0.30$.
- Only three open/semi-open LLMs are tested; fine-tuning experiments are not conducted on Claude, Gemini, or similar models.
- ORPO fine-tuning is validated only on Llama-3.1; cross-model transferability is not verified.
- For ethical reasons, the study does not test within real agentic pipelines, examining only the LLM component in isolation.

## Related Work & Insights

- **vs. Goldstein et al. (2024)**: They quantify the overall persuasive effect of propaganda; this paper further decomposes it into specific rhetorical techniques and provides mitigation strategies.
- **vs. Voelkel et al. (2025)**: They analyze surface-level linguistic features (pronouns, negations, tone); this paper focuses on deeper rhetorical strategies.
- **vs. Pauli et al. (2024)**: They benchmark persuasiveness differences across LLMs; this paper focuses on specific patterns of rhetorical technique usage.
- **vs. Chen et al. (2024)**: They use fine-tuning to improve fairness; this paper applies analogous methods to propaganda mitigation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic quantification of rhetorical technique usage in LLMs combined with ORPO-based mitigation; the "how to persuade" analytical perspective is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three LLMs + human validation ($\kappa = 0.86$–$0.97$) + three fine-tuning methods + 1,000-thesis experiment.
- **Writing Quality**: ⭐⭐⭐⭐ — Research design is clear, experimental workflow is systematic, and results are presented intuitively.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to AI safety, content moderation, and LLM alignment; the effectiveness of ORPO is particularly meaningful for safety training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/social_computing/synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](../../ACL2026/social_computing/when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)

</div>

<!-- RELATED:END -->
