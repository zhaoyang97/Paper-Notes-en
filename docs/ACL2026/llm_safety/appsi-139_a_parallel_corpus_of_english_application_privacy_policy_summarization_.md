---
title: >-
  [Paper Note] APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation
description: >-
  [ACL 2026][LLM Safety][Privacy Policy] APPSI-139 is the first English application privacy policy summarization and interpretation parallel corpus meticulously annotated by legal experts (139 policies / 36…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Privacy Policy"
  - "Parallel Corpus"
  - "Multi-task Learning"
  - "Summarization and Interpretation"
  - "Legal NLP"
date: 2026-05-08
content_hash: d506131d6a477ca7
---

# APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation

**Conference**: ACL 2026  
**arXiv**: [2604.27550](https://arxiv.org/abs/2604.27550)  
**Code**: https://github.com/EnlightenedAI/APPSI-139  
**Area**: LLM Security / Privacy / Text Summarization  
**Keywords**: Privacy Policy, Parallel Corpus, Multi-task Learning, Summarization and Interpretation, Legal NLP

## TL;DR
APPSI-139 is the first English application privacy policy summarization and interpretation parallel corpus meticulously annotated by legal experts (139 policies / 36,351 annotations / 15,692 paraphrase pairs). The accompanying TCSI-pp-V2 framework utilizes a shared encoder with 5 alternately trained expert heads to implement sub-tasks for "Importance / Risk / Sensitivity / Topic / Paraphrase." Compared to TCSI-pp v1, encoding time is reduced by 73% and memory consumption drops from 7.3GB to 2.7GB, while subjective readability votes outperform GPT-4o and Llama3-70b.

## Background & Motivation

**Background**: Privacy policies are the legal foundation for user authorization of personal data processing by applications. However, they are generally lengthy, filled with legalese and technobabble, and compounded by "rational ignorance" and "dark patterns," leading most users to click "agree" without reading, which results in the silent use of sensitive data. While standardized labels like "Privacy Nutrition Labels," "LPL," and "TILT" have been attempted, their implementation depends entirely on the service provider's integrity.

**Limitations of Prior Work**: Automatic summarization is seen as a solution, but existing privacy policy corpora are almost entirely oriented toward English **Information Extraction** (OPP-115 / APP-350 / PI-Extract / Optoutchoice / PrivacyQA / PolicyQA), which solves the "length" issue but not the "comprehensibility" issue. The only corpus with paraphrased interpretations, CAPP-130, is in Chinese, and machine translation into English loses legal precision.

**Key Challenge**: Summarizing privacy policies requires being **concise**, **easy to understand**, and **legally precise**—three goals that are naturally in conflict. Without a parallel corpus, mapping "original text → easy-to-understand paraphrase" cannot be learned; using general LLMs leads to degraded legal semantics; using small models suffers from a lack of professional data.

**Goal**: ① Construct an English parallel corpus featuring "sentence-level multi-labeling + paraphrasing"; ② Design an efficient framework that unifies five tasks—identifying important/risk/sensitive clauses, topic classification, and paraphrasing—under a shared encoder, avoiding the redundancy of running one encoder per sub-task in v1.

**Key Insight**: Instead of machine translation, 5 LL.M. holders and 1 Law Professor were recruited to manually annotate the English version using the CAPP-130 schema. On the model side, five tasks are integrated into one shared encoder with five parallel expert heads and trained using an alternating strategy.

**Core Idea**: By utilizing human-annotated English parallel corpora, a multi-task shared encoder, and an alternating training strategy, the triangle of "computational efficiency + comprehensibility + trustworthiness" is addressed simultaneously.

## Method

### Overall Architecture
The methodology follows two tracks. **Data Track**: Latest privacy policies (as of 2023-10) were collected from Top-100 English apps on Google Play and the App Store. After deduplication, 139 policies were retained and split into 30,877 sentences. Five LL.M. holders each specialized in one sub-task (Importance / Risk / Sensitivity / Topic / Paraphrase). A pilot annotation yielded Cohen’s Kappa = 0.892, followed by formal annotation with a senior reviewer for final arbitration of ambiguous terms. **Model Track**: TCSI-pp-V2 consists of a shared bottom encoder + 4 classification heads (Importance / Topic / Risk / Sensitivity) + 1 paraphrase decoder. The inference pipeline involves Importance filtering, Topic filtering based on user selection, Risk/Sensitivity highlighting, and finally, Rewrite generation for readability.

### Key Designs

1.  **APPSI-139 Corpus Schema: 11 Data Practices + 3 Special Labels + Paraphrase Pairs**:
    *   **Function**: Slices privacy policies along legal risk dimensions and provides human-written paraphrases for high-risk clauses.
    *   **Mechanism**: Each sentence can have multiple labels. The 11 data practices cover First-Party Collection / Permission Acquisition / Third Party Sharing / Usage / Data Retention / Data Security / Edit-Control / Specific Audiences / Contact / Policy Change / Cease Operation. The 3 special labels are Importance (critical clauses), Risk (vague compliance language violating GDPR/CCPA), and Sensitivity (biometric, precise location, financial accounts, aligned with GDPR + NIST SP 800-122 + GB/T 35273-2020). The paraphrase set contains 15,692 pairs, reducing average sentence length from 27 words to 20 words (-26%).
    *   **Design Motivation**: Topic classification alone solves "length" but not "readability." Introducing orthogonal labels for Importance/Risk/Sensitivity allows the model to identify "what must be read" and "what is dangerous/sensitive," providing a complete legal perspective for trustworthy decision-making. Risk labels account for only 1.9% but hold the highest value, reflecting legal blind spots.

2.  **TCSI-pp-V2 Shared Encoder + 5 Expert Heads Multi-task Architecture**:
    *   **Function**: Compresses the v1 architecture from one encoder per sub-task to "one shared encoder + 5 parallel experts," reducing encoding redundancy by 5x.
    *   **Mechanism**: Sentence-level embeddings $E=\{e_1,\dots,e_n\}$ are passed through a shared bottom $F_f(e_j,\theta_f)$ to obtain $\{f_1,\dots,f_n\}$. Five expert heads perform $F_i$ (Importance binary), $F_t$ (11-class Topic), $F_r$ (Risk binary), $F_s$ (Sensitivity binary), and $F_{rewrite}$ (Autoregressive rewriting $P(z_t\mid f_j;z_{1:t-1})$).
    *   **Design Motivation**: Sub-tasks share high-level features as they make judgments on the same legal text. A shared encoder learns more general legal semantic representations and slashes inference memory from 7.3GB to 2.7GB.

3.  **Alternating Training Strategy + Single-Expert/Single-Task Labor Division**:
    *   **Function**: Uses alternating updates for expert heads instead of weighted joint loss, and assigns one annotator per sub-task to avoid cognitive confusion.
    *   **Mechanism**: For annotation, individual workers focus on one task to reduce cognitive bias between labels. For training, only one expert head backpropagates per step, allowing the shared bottom to transfer stably across tasks.
    *   **Design Motivation**: In joint training, strong tasks (e.g., Sensitivity) dominate gradients, drowning out weak tasks (e.g., Cease Operation at 0.04%). Alternating training ensures fair updates for all tasks.

### Loss & Training
Classification heads use standard Cross-Entropy; the rewriting head uses teacher-forcing language modeling. During alternating training, each mini-batch backpropagates only one expert head. Backbones include mT5-small / Bert2GPT / XLNet2GPT / Electra2GPT. Data is split 80:10:10. Hardware: NVIDIA V100. Cease/Permission classes were excluded from binary evaluation due to extreme sparsity.

## Key Experimental Results

### Main Results (Comparison with LLMs on APPSI-139 Test Set)

| Task | Metric | Qwen3-8B | Llama3-8B | GPT-4o-mini | Gemini-2.5 | **TCSI-pp-V2** |
|---|---|---|---|---|---|---|
| Topic | Micro-F1 | 47.85 | 30.77 | 50.36 | 65.38 | **78.18** |
| Topic | Macro-F1 | 44.44 | 24.65 | 32.83 | 6.42 | **77.12** |
| Important | Micro-F1 | 63.50 | 52.53 | 51.00 | 73.33 | **73.93** |
| Risk | Micro-F1 | 85.53 | 96.00 | 89.34 | 85.05 | **95.60** |
| Sensitive | Micro-F1 | 45.37 | 28.01 | 38.64 | 23.33 | **96.96** |
| Rewritten | ROUGE-L | 0.4541 | 0.4156 | 0.4286 | 0.4776 | **0.6903** |
| Rewritten | BERTScore | 0.8970 | 0.8520 | 0.8950 | 0.9070 | **0.9430** |
| Rewritten | BARTScore | -2.76 | -3.03 | -2.89 | -2.78 | **-1.68** |

Subjective Readability Voting (53 participants, 10 multiple-choice questions): TCSI-pp-V2 **39.06%** > Llama3-70b 25.28% > GPT-4o 24.52% > Kimi 11.13%; the proposed model won 7 out of 10 questions.

### Ablation Study 1: V1 vs V2 Memory and Latency (mT5-small back-end, 1,893 sentences)

| Metric | TCSI-pp (V1) | TCSI-pp-V2 | Improvement |
|---|---|---|---|
| Encoding Time | 92.66 s | 24.72 s | **-73%** |
| Total Time | 191.94 s | 123.26 s | -36% |
| Avg. Time / Sent | 0.101 s | 0.065 s | -36% |
| Inference VRAM | 7,343 MB | **2,766 MB** | -62% |

### Ablation Study 2: Input Length Robustness (Longest 100 vs Shortest 100 vs All)

| Task | Metric | Longest | Shortest | All |
|---|---|---|---|---|
| Topic | Micro-F1 | 79.41 | 76.23 | 78.18 |
| Risk | Micro-F1 | 94.74 | 94.34 | 95.60 |
| Sensitive | Micro-F1 | 96.83 | 96.18 | 96.96 |
| Rewritten | ROUGE-L | 0.6979 | 0.6542 | 0.6903 |

### Key Findings
-   **Specialized Data + Small Model > Large Model + Prompt Engineering**: mT5-small (300M parameters) outperforms GPT-4o-mini / Llama3-8B / Gemini-2.5 across all tasks. Particularly on the Sensitive task, TCSI-pp-V2 (96.96) crushes Gemini-2.5 (23.33)—general LLMs struggle to identify specific legal sensitive semantics like "biometrics."
-   **Shared Encoder Gains**: Differences between V2 and V1 single-task models are <0.02, but VRAM is reduced by 62% and encoding time by 73%. ROUGE/BERTScore for rewriting improved slightly, indicating generalization benefits from multi-task regularization.
-   **Readability Wins Over GPT-4o**: In subjective voting, GPT-4o often "almost verbatim restates the original text," leading to verbosity. TCSI-pp-V2's multi-level bullet structure + mandatory paraphrasing scored highest.
-   **Severe Data Imbalance**: Important (51.04%), Risk (1.9%), Cease (0.04%). The lower macro-F1 for Risk (~60) compared to micro (~96) indicates the need for more minority class data or resampling.

## Highlights & Insights
-   **Orthogonal Legal Labels (Importance / Risk / Sensitivity)**: Unlike prior corpora that only label "topics," this work treats "must-read / dangerous / sensitive" as independent labels, enabling both "extractive filtering" and "risk highlighting."
-   **Global Standard Alignment (GDPR + NIST + GB/T)**: Aligning sensitivity labels with cross-regional standards ensures global transferability.
-   **Shared Encoder + Alternating Training**: A practical paradigm for the small-model era, applying "expert routing" logic at the 300M parameter scale for edge-friendly deployment.
-   **Expert Single-Task Annotation**: Cohen's Kappa of 0.892 proves the labor division strategy is effective and transferable to other multi-label legal corpora.

## Limitations & Future Work
-   Limited to English; multi-lingual coverage is missing.
-   Policies are from 2023-10 and may become outdated as regulations evolve.
-   Generative rewriting still carries hallucination risks—it may omit critical clauses or distort legal meaning.
-   Subjective evaluation samples demographic bias (students/young adults).
-   No comparison with latest reasoning models (e.g., o1 / DeepSeek-R1), which might close the gap via chain-of-thought.

## Related Work & Insights
-   **vs OPP-115 / PolicyQA**: Those focus on classification/QA; APPSI-139 is the first English **paraphrase** parallel corpus.
-   **vs CAPP-130 (NeurIPS 2023)**: CAPP-130 is the Chinese predecessor; APPSI-139 is a ground-up English reconstruction with extended sensitivity standards.
-   **vs GPT Prompting**: Pure prompting fails on niche legal semantics (Sensitivity); this work confirms that "high-quality data + fine-tuning" still beats zero-shot LLMs in legal NLP.

## Rating
-   Novelty: ⭐⭐⭐⭐ First English paraphrase corpus + orthogonal legal labels are practical breakthroughs.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison with 13 backbones and 4 LLMs; latency and robustness tests included.
-   Writing Quality: ⭐⭐⭐ Generally clear, with some inconsistencies in tense usage.
-   Value: ⭐⭐⭐⭐⭐ Dataset + Model + Evaluation suite provides a complete path for consumer protection products.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[CVPR 2026\] Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation](../../CVPR2026/llm_safety/select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
