---
title: >-
  [Paper Note] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content
description: >-
  [NEURIPS2025 (MusIML Workshop)][AIGC Detection][LLM Evaluation] This paper proposes a dual-agent (quantitative + qualitative) evaluation framework that systematically assesses the faithfulness of GPT-4o, Ansari AI, and Fanar on Islamic content generation tasks across three dimensions—theological accuracy, citation integrity, and stylistic appropriateness—finding that even the best-performing model exhibits significant deficiencies in citation reliability.
tags:
  - NEURIPS2025 (MusIML Workshop)
  - AIGC Detection
  - LLM Evaluation
  - Islamic Content Generation
  - Dual-Agent Framework
  - Citation Verification
  - High-Stakes Domain Generation
date: 2026-05-08
content_hash: 48579e8d1a67cc88
---

# Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content

**Conference**: NEURIPS2025 (MusIML Workshop)
**arXiv**: [2510.24438](https://arxiv.org/abs/2510.24438)
**Code**: To be confirmed
**Area**: AIGC Detection
**Keywords**: LLM Evaluation, Islamic Content Generation, Dual-Agent Framework, Citation Verification, High-Stakes Domain Generation

## TL;DR

This paper proposes a dual-agent (quantitative + qualitative) evaluation framework that systematically assesses the faithfulness of GPT-4o, Ansari AI, and Fanar on Islamic content generation tasks across three dimensions—theological accuracy, citation integrity, and stylistic appropriateness—finding that even the best-performing model exhibits significant deficiencies in citation reliability.

## Background & Motivation

**Special requirements of high-stakes domains**: Islamic content generation demands exceptionally high standards of theological accuracy, citation attribution, and tonal appropriateness; subtle errors—such as misquoting Qur'anic verses or misattributing hadith—can propagate misinformation and cause spiritual harm.

**Limitations of conventional metrics**: Surface-overlap metrics such as BLEU and ROUGE are incapable of measuring doctrinal fidelity, citation integrity, or theological correctness.

**Evaluation gap**: Specialized evaluation pipelines have been developed for high-stakes domains such as medicine and law, yet the religious domain remains nearly unaddressed; existing Islamic chatbots (Ansari AI, Fanar) have only been assessed on general Arabic benchmarks without any theological-level evaluation.

**Infrastructure deficiencies**: A substantial portion of classical Islamic texts remains in unstructured PDF or scanned-image formats, impeding computational utilization.

**Cross-domain precedents**: The legal domain (Mata v. Avianca) has exposed fabricated citations; 50–90% of medical responses lack adequate citation support; 41 of 77 AI-generated articles by CNET required correction—the religious domain faces analogous, if not more severe, risks.

**Core research question**: Can current LLMs generate Islamic content that is theologically accurate, correctly cited, and tonally appropriate? And how can such generation be systematically evaluated?

## Method

### Overall Architecture

The paper proposes a **Dual-Agent Evaluation Framework** consisting of a quantitative evaluation agent and a qualitative comparative agent. Both agents share a citation verification toolchain and assess LLM outputs from complementary perspectives.

### Three Core Design Modules

**1. Prompt Collection and Response Acquisition**

- Fifty prompts are collected from five authoritative Islamic blog platforms (The Thinking Muslim, IslamOnline, Yaqeen Institute, etc.), comprising article titles authored by prominent Islamic scholars.
- The prompts span five domains: Fiqh (Islamic jurisprudence), Tafsir (Qur'anic exegesis), Ulum al-Hadith (hadith sciences), Aqidah (theology), and Adab (spiritual conduct).
- Each prompt is submitted to GPT-4o, Ansari AI, and Fanar, yielding 150 responses in total.

**2. Quantitative Evaluation Agent**

- Built on the OpenAI o3 reasoning model, equipped with three verification tools: Qur'an Ayah (verse retrieval), Internet Search, and Internet Extract.
- Each article is segmented into introduction, body, and conclusion; scoring is performed on **6 dimensions** (1–5 scale):
    - **Style & Structure** (4 sub-dimensions): Structure, Theme, Clarity, Originality
    - **Islamic Content** (2 sub-dimensions): Islamic Accuracy, Citation & Source Usage
- Detected citations are automatically retrieved and verified, returning one of four labels: confirmed / partially confirmed / unverified / refuted.
- Scores are penalized for citations that are not fully confirmed.

**3. Qualitative Comparative Agent**

- Processes responses from all three models simultaneously (delimited by XML tags `<R1>/<R2>/<R3>`) and performs side-by-side comparison.
- Evaluates five dimensions: Clarity & Structure, Islamic Accuracy, Tone & Appropriateness, Depth & Originality, and Comparative Reflection.
- Identifies the strongest and weakest response per dimension, supported by specific textual excerpts.
- Employs the same verification toolchain as the quantitative agent to ensure consistency.

### Loss & Training

- The quantitative dimensions adopt a 1–5 scale; citation verification outcomes directly affect the Islamic Accuracy and Citation scores.
- The qualitative dimensions adopt a Best/Worst voting scheme, yielding binary judgments across three models per dimension per prompt.
- Alignment between the two evaluation modes provides evidence of convergent validity.

## Key Experimental Results

### Main Results

| Model | Overall Mean | Std | Structure | Theme | Clarity | Originality | Islamic Accuracy | Citation |
|-------|-------------|-----|-----------|-------|---------|-------------|------------------|----------|
| GPT-4o | **3.90** | 0.589 | 4.16 | **4.43** | 4.10 | 3.10 | **3.93** | **3.38** |
| Ansari AI | 3.79 | — | — | — | — | — | 3.68 | 3.32 |
| Fanar | 3.04 | 0.923 | — | — | — | 2.73 | 2.76 | 1.82 |

### Qualitative Comparative Results (Best/Worst Votes, max 200 each)

| Model | Total Best | Total Worst | Strongest Dimension |
|-------|-----------|------------|---------------------|
| Ansari AI | **116** | 3 | Clarity & Structure (41), Islamic Accuracy (42) |
| GPT-4o | 84 | 4 | Tone & Appropriateness (48) |
| Fanar | 0 | 193 | Weakest across all dimensions |

### Key Findings

1. **GPT-4o leads quantitatively**: Achieving an overall mean of 3.90/5 with the lowest variance (std = 0.589), GPT-4o demonstrates consistent superiority in structure, theme, and Islamic accuracy.
2. **Ansari AI leads qualitatively**: With 116/200 Best votes, Ansari AI excels in clarity and religious fidelity, reflecting the advantages of domain adaptation.
3. **Fanar lags overall but shows architectural innovation**: Its 9B parameters and 4,096-token context window constrain reasoning capacity; nevertheless, its morphological tokenizer, region-specific datasets, and Islamic RAG pipeline represent valuable contributions.
4. **Citation deficiencies are universal**: Even the best-performing model (GPT-4o, Citation = 3.38/5) exhibits the most pronounced weakness in citation accuracy—a core requirement for faith-sensitive writing.
5. **Model scale has a substantial impact**: The performance gap between GPT-4o (128K context) and Fanar (4,096 context) is strongly correlated with differences in model scale and context length.

## Highlights & Insights

- **First systematic faithfulness evaluation for Islamic content**: This work fills a critical gap in LLM evaluation for the religious domain; the framework design is transferable to other high-stakes domains such as medicine and law.
- **Elegant dual-agent complementarity**: The quantitative agent provides comparable numerical scores while the qualitative agent captures nuanced differences in tone and rhetoric; shared toolchains ensure methodological consistency.
- **Practical citation verification toolchain**: Automatic retrieval of Qur'anic verses and hadith with four-level labeling (confirmed / partially confirmed / unverified / refuted) yields directly applicable output.
- **Rigorous experimental design**: Fifty prompts covering five Islamic knowledge domains, a blind-review protocol to reduce bias, and human inspection as a sanity check collectively strengthen the evaluation's credibility.

## Limitations & Future Work

1. **Evaluator bias**: Both the quantitative and qualitative agents are based on OpenAI models, introducing a risk of same-family bias; future work should incorporate heterogeneous evaluators such as Claude and Gemini for cross-validation.
2. **Limited scale**: Only 50 prompts are used, leaving different legal schools (*madhahib*), edge cases, and contemporary jurisprudential issues uncovered.
3. **Single-language evaluation**: Only English responses are assessed; Fanar's primary language (Arabic) is not evaluated, which may disadvantage Fanar unfairly.
4. **Lack of multi-expert validation**: Only one human reviewer is involved, falling short of the 3–5 scholar consensus panel recommended for theological review.
5. **Imprecise domain categorization**: Some prompts may span multiple domains, introducing ambiguity in domain attribution.

## Related Work & Insights

- **High-stakes domain evaluation**: Hallucination and citation verification research exists for the legal (LEGAL-BERT, LegalBench), medical (SourceCheckup), and journalism domains; this paper extends those paradigms to the religious domain.
- **Islamic NLP**: AraBERT and Qur'anQA have advanced Arabic language understanding; Ansari AI and Fanar are representative Islamic chatbots, but their evaluation has been limited to general benchmarks.
- **Agent-based evaluation**: Tool-augmented approaches combining RAG, chain-of-thought reasoning, and multi-agent collaboration (LangChain / CrewAI / CamelAI) have improved verifiability on general tasks; this paper is the first to apply such methods to theological verification.
- **Data infrastructure**: Platforms such as Usul.ai, SHARIAsource, Shamela, and OpenITI provide machine-readable Islamic legal data, but these resources have not yet been systematically integrated into LLM evaluation pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic faithfulness evaluation framework for Islamic content; the dual-agent design is inventive.
- **Experimental Thoroughness**: ⭐⭐⭐ — The 50-prompt scale is modest; single-language evaluation and a single human reviewer limit the strength of conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated problem statement, and strong connections to cross-domain related work.
- **Value**: ⭐⭐⭐⭐ — The framework is transferable to other high-stakes domains; the problem formulation and evaluation dimension design offer useful references.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](../../ACL2026/aigc_detection/flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](../../ICLR2026/aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](../../ACL2026/aigc_detection/reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)

<!-- RELATED:END -->
