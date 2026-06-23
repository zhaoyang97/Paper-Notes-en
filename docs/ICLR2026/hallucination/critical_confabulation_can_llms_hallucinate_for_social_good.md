---
title: >-
  [Paper Note] Critical Confabulation: Can LLMs Hallucinate for Social Good?
description: >-
  [ICLR 2026][Hallucination Detection][Paper Note] This paper reframes "hallucination" as a viable resource: it proposes **critical confabulation**, where LLMs "fill in" structural gaps in historical archives under evidentiary constraints. By evaluating 19 models on a "narrative cloze" task using unpublished Black history corpora, the authors demonstrate that controlle
tags:
  - ICLR 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: c59e13ecb5914aa3
---
# Critical Confabulations: Can LLMs Hallucinate for Social Good?

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wGFD7ITicm](https://openreview.net/forum?id=wGFD7ITicm)  
**Code**: To be confirmed (released post-camera-ready; BWTC data requires ARTFL authorization)  
**Area**: Hallucinations / Computational Humanities / Narrative Understanding  
**Keywords**: Critical confabulation, controlled hallucination, narrative cloze, data contamination audit, digital humanities  

## TL;DR
This paper reframes "hallucination" as a viable resource: it proposes **critical confabulation**, where LLMs "fill in" structural gaps in historical archives under evidentiary constraints. By evaluating 19 models on a "narrative cloze" task using unpublished Black history corpora, the authors demonstrate that controlled, well-defined hallucinations can serve knowledge production without collapsing into falsehood.

## Background & Motivation
**Background**: LLM hallucinations are typically treated as failure modes to be eliminated. However, recent work suggests that a class of behavior called **confabulation**—using self-consistent stories to "fill" missing information while maintaining high verisimilitude—possesses narrative value. This has shown social utility in computational creativity, narrative exposure therapy, and digital storytelling for cultural heritage.

**Limitations of Prior Work**: In the humanities, **critical fabulation** (Hartman 2008) is a mature methodology that uses speculative narrative to repair historical archival injustices, specifically giving voice to "hidden figures" who never received recording privileges due to systemic oppression. However, it relies heavily on scholars' close reading of dense documents, making it labor-intensive and unscalable for vast archives.

**Key Challenge**: Strict factuality vs. narrative completion. Treating "whether it was archived" as a proxy for truth essentially overfits to the biased standard of "what survived," further silencing hidden figures. Conversely, allowing unconstrained LLM hallucinations collapses speculation into falsehood, losing historical fidelity.

**Goal**: To operationalize existing LLM confabulation behaviors into a scalable critical confabulation workflow within strict evidentiary boundaries. This workflow aims to identify potential gaps in archives and provide multiple evidence-constrained possibilities (rather than asserting a single truth) to assist humanists in expanding historical knowledge.

**Core Idea**: **[Controlled Hallucination as a Resource]** Formalize critical fabulation as an **open-ended narrative cloze** task. Given an event timeline of a hidden figure, one event is masked. The model is required to reconstruct the masked event under known contextual constraints. Narrative embedding similarity is used to judge if the output is "close enough," flipping "hallucination as a defect" into "hallucination as an optimizable capability."

## Method

### Overall Architecture
The workflow addresses two levels of objectives: **known unknowns (gap reconstruction)** and the harder **unknown unknowns (gap detection)**; this paper focuses on the former. The system uses the unpublished Black history archive BWTC as the "unseen" ground truth. After a double data contamination audit to filter out seen corpora, timelines of hidden figures are extracted. Models are then evaluated via a mask-reconstruct cloze task coupled with various "hallucination-inducing" prompts.

```mermaid
flowchart LR
    A[BWTC Archive Corpus B] --> B1[Double Data Contamination Audit]
    B1 -->|String Search + Behavioral Probes| C[Exclude SEEN Documents<br/>Keep only Bunseen]
    C --> D[Hidden Figure Mining<br/>NER + Aho-Corasick Long-tail Filtering]
    D -->|156 Hidden Figures| E[GPT-o3 Extraction<br/>Event Timeline T_n + Event Types]
    E --> F[Mask one event → C n,m]
    F --> G[19 LLMs reconstruct ê_m under controlled prompts]
    G --> H[story-emb Cosine Similarity ≥ ε*<br/>Judge as correct]
```

### Key Designs

**1. Task Formalization: Turning critical fabulation into a measurable narrative cloze.** For each hidden figure $n$, an ordered event timeline $T(n)=\langle(t_1,e_1),\dots,(t_{m(n)},e_{m(n)})\rangle$ is constructed from relevant archives, where each element is a timestamp $t_i$ and a one-sentence event $e_i$. Historical gaps are simulated by replacing the $m$-th event with a `[MASK]` literal, resulting in $C(n,m)=\langle(t_1,e_1),\dots,(t_m,\text{[MASK]}),\dots\rangle$. The model $f_\theta$ must reconstruct $e_m$ given the remaining timeline fragments and fixed instructions. A reconstruction is marked correct if $\text{sim}_{\text{emb}}(\hat e_m, e_m) \ge \epsilon$. This design converts vague "speculative narrative" into reproducible, comparable, and optimizable metrics while deliberately masking only one event to maintain sufficient evidentiary constraint.

**2. Two-Stage Data Contamination Audit: Ensuring "unseen history" is not contaminated by memory.** A key premise is that the model has not seen these archives; otherwise, constrained confabulation degrades into memorization. The authors use the OLMO-2 fully open-data model for the primary audit: first, a Boyer–Moore substring match compares every sentence in BWTC with the OLMO-2 training set. Documents with $\text{matches}(d) = \sum_{x \in O} \sum_{s \in S(d)} \text{BM}(x, s) \ge 100$ are labeled SEEN (21% total). Next, a behavioral probe cross-validates this: OLMO-2 continues a text given the first 20 sentences. If the labels are credible, SEEN documents should have continuations closer to the ground truth, i.e., $\text{mean}_{d \in B_{\text{seen}}}[\text{sim}_i(d)] > \text{mean}_{d \in B_{\text{unseen}}}[\text{sim}_i(d)]$. Results confirmed higher similarity for SEEN docs, with the advantage decaying as the continuation position moves from $p_1$ to $p_5$, consistent with memory advantages being strongest immediately after the observed context. Only $B_{\text{unseen}}$ is used for final analysis.

**3. Hidden Figure Mining + Evidence-Constrained Truth Extraction.** Even after excluding SEEN documents, parametric knowledge might contain priors for specific names. An Aho–Corasick multi-pattern match is performed: up to 10,000 PERSON names are extracted from $B_{\text{unseen}}$ using NLTK, keeping only long-tail names (frequency $<51$) appearing in at least 3 documents. The trainer set is scanned for counts $c(n)$, and names with $c(n) \ge 100$ are marked SEEN-IN-O. After manual filtering of incidental mentions, 156 clean hidden figures remain. GPT-o3 extracts chronological timelines with explicit citations under strict "source-constrained" instructions. Each event is an active-voice sentence ($\le 30$ words) labeled with one of five types: `{AGENTIVE, RELATIONAL, OBSERVATIONAL, COGNITIVE, ROLE}`.

**4. Narrative-Specific Evaluation + Controlled Inducement Prompts.** Evaluation uses **story-emb**, a model emphasizing storyline structure over generic semantic embeddings (to avoid topic-similarity bias). An operational threshold $\epsilon^\star = 73.13$ (macro-F1 = 0.805) was selected via a labeled validation set. On the prompting side, six system/instruction templates designed to "induce increased hallucination/creativity" (e.g., Null-Shot, HaluEval, LLM-Discussion, Eccentric Automatic Prompts) are layered over a unified baseline, occasionally providing the masked event type as structural supervision.

## Key Experimental Results

### Main Results (Narrative Cloze Accuracy, Abridged)

| Model | Baseline (No Type) | Best (No Type) | Baseline (With Type) | Peak (With Type) |
|-------|-------|-------|-------|-------|
| GPT-5-chat | 51.0 | 57.4 | 55.5 | **59.7** |
| Qwen3-4B-0725 | 47.1 | 50.9 | 50.2 | 53.8 |
| OLMo-2-7B | 33.8 | 49.1 | 38.5 | **58.8** |
| OLMo-2-32B | 44.9 | 46.0 | 48.4 | 50.8 |
| Qwen3-4B | 44.6 | 55.0 | 46.5 | 56.8 |
| GPT-4o | 42.6 | 45.5 | 45.4 | 49.1 |
| gemma-2-27b | 40.1 | 40.1 | 42.1 | 46.1 |

- GPT-5-chat is the overall leader and the only model exceeding 50% across most prompts, peaking at 59.7%. 
- Small models like OLMo-2-7B and Qwen3-4B punch above their weight, outperforming larger baselines.
- Providing **EVENT_TYPE prompts yields a stable +2~10 point gain** across nearly all (model, prompt) pairs.

### Ablation Study (Sampling Temperature Randomness, Average Change Relative to Deterministic Baseline)

| Temperature Setting | Avg Accuracy Change |
|------|------|
| Deterministic (T=0) | Baseline |
| Low (T=0.2) | −0.3 |
| Mid (T=0.7) | −0.8 |
| High (T=1.2) | −2.3 |

- Performance is robust to sampling randomness; model rankings remain consistent. Larger OLMo-2 variants show steeper declines at high temperatures.

### Key Findings
- **Task is feasible but difficult**: Most models stall below 50%, with peaks near 60% under strong prompts.
- **No significant memorization advantage in audit**: No significant difference between OLMo-2 and un-audited peers (avg $p=0.354$), suggesting memory did not provide a shortcut.
- **Event Type Gradients**: Models perform best on "role" (44.8%, biographical) → "relational" → "agentive" → "observational," and worst on "cognitive" (24.9%, where internal states lack observable anchors).
- **Structural Sensitivity**: Accuracy rises slightly with longer event descriptions ($\rho=0.09$) but drops with longer timelines ($\rho=-0.173$). Events at the start of a timeline are easiest to reconstruct (0.45) compared to the end (0.337).
- **Highly Clustered Errors**: Only 2.2% of events were solved by all 19 models; 59.1% were failed by at least 10 models. Jaccard overlap of error sets reached 0.6–0.9, indicating models fail on the same difficult events rather than randomly.

## Highlights & Insights
- **Paradigm Shift**: Reconceptualizes hallucination from a "defect to be eliminated" to an "optimizable resource," rigorously operationalizing humanities theory (Hartman’s critical fabulation) into a measurable NLP task.
- **Rigorous Auditing**: Using fully open-source training data (OLMO-2) for double-contamination auditing (substring + behavioral probes) creates a robust precedent for "unseen data" evaluation.
- **Tailored Evaluation**: Employs narrative-specific embeddings (`story-emb`) rather than general semantic ones, with human-calibrated thresholds, demonstrating nuanced understanding that narrative verisimilitude $\neq$ semantic proximity.
- **Social Value Orientation**: Explicitly includes a Humanities Mission Statement and Ethics Statement, emphasizing that the goal is not to invent fake history but to serve as a "recovery technology" for archival silences.

## Limitations & Future Work
- **High Sensitivity**: Performance is highly dependent on prompts and input structure, limiting the robustness of conclusions.
- **Monolingual/Monocultural**: Experiments are limited to English and Black history; generalizability to other traditions is unknown.
- **Lack of Training/Inference Optimization**: Current work is zero-shot; no methods have been designed yet to explicitly optimize "well-defined, evidence-constrained confabulation."
- **Ethical Risks**: Authors acknowledge the need for provenance tracking and ethical guardrails to prevent reconstruction from inadvertently "compounding archival violence."

## Related Work & Insights
- **Useful Hallucinations**: Follows the lineage of Jiang et al. 2024 and Sui et al. 2024 regarding the value of confabulation, but provides the first systematic evidence-constrained evaluation framework.
- **AI Text Restoration**: Extends domain-specific restoration (e.g., AlphaGeometry for ancient texts, Assael et al. 2022/2025) into the more open semantic space of cultural/historical narrative.
- **Data Contamination Detection**: Echoes behavioral probe logic (Oren et al. 2024) and notes the unreliability of MIA in OOD settings (Maini/Duan et al. 2024), prompting the use of open-data models for auditing—a useful reminder for all "unseen data" evaluations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High originality in operationalizing "critical fabulation" as an NLP task and reframing hallucination as an asset.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid analysis across 19 models, multiple prompts, and various metadata; rigorous auditing. Restricted by single-culture/language and lack of fine-tuning methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear and powerful interdisciplinary motivation; research questions (R1-R3) are well-organized. 
- **Value**: ⭐⭐⭐⭐ Opens a quantifiable research direction for "beneficial hallucinations" in digital humanities; significant social and methodological merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Honest Lying: Understanding Memory Confabulation in Reflexive Agents](../../ICML2026/hallucination/honest_lying_understanding_memory_confabulation_in_reflexive_agents.md)
- [\[ICLR 2026\] HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs](halluguard_demystifying_data-driven_and_reasoning-driven_hallucinations_in_llms.md)
- [\[ACL 2026\] Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination](../../ACL2026/hallucination/vocabulary_hijacking_in_lvlms_unveiling_critical_attention_heads_by_excluding_in.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](../../CVPR2026/hallucination/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [\[ICLR 2026\] GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs](ghost_hallucination-inducing_image_generation_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
