---
title: >-
  [Paper Note] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models
description: >-
  [ACL 2026][LLM Safety][incomplete learning] This paper presents the first systematic study of the *Incomplete Learning Phenomenon* (ILP) in SFT — i.e.…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "incomplete learning"
  - "SFT diagnostics"
  - "knowledge conflict"
  - "forgetting"
  - "fine-tuning failure modes"
date: 2026-05-08
content_hash: 6e0be553da6bfb76
---

# Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.10079](https://arxiv.org/abs/2604.10079)
**Code**: None
**Area**: LLM Safety
**Keywords**: incomplete learning, SFT diagnostics, knowledge conflict, forgetting, fine-tuning failure modes

## TL;DR

This paper presents the first systematic study of the *Incomplete Learning Phenomenon* (ILP) in SFT — i.e., the model's inability to correctly reproduce a subset of training samples even after convergence. Five recurring causes are identified (knowledge absence, knowledge conflict, intra-dataset contradiction, left-side forgetting, and insufficient optimization), along with a diagnostic framework and targeted mitigation strategies.

## Background & Motivation

**Background**: SFT is the standard paradigm for adapting LLMs to downstream tasks and is broadly regarded as a reliable and efficient specialization mechanism.

**Limitations of Prior Work**: (1) Even when training loss fully converges, models frequently fail on a subset of training samples — this is not an overfitting or generalization issue but a failure on the training set itself; (2) unlearned samples tend not to be random but correspond to rare cases, combinatorial patterns, or knowledge-intensive instances; (3) improvements in aggregate metrics may mask persistently unlearned subsets.

**Key Challenge**: SFT datasets — particularly in specialized domains such as law and medicine — are costly to construct, yet $15.3\%\pm2.1\%$ of samples remain unlearned after training, directly reducing data utilization efficiency.

**Goal**: Rather than proposing a new fine-tuning algorithm, this work aims to systematically characterize, diagnose, and validate the sources of incomplete learning in SFT.

**Key Insight**: Unlearned samples are treated as diagnostic signals rather than noise — analyzing *why* specific samples are not learned reveals fundamental limitations of SFT.

**Core Idea**: Each of the five ILP sources requires a distinct mitigation strategy; there is no one-size-fits-all solution, and fine-grained sample-level diagnosis is necessary.

## Method

### Overall Architecture

A three-stage framework: (1) fine-tune the model with SFT until convergence; (2) convert training data into multiple-choice format and detect unlearned samples via pass@N and Best-of-N (BoN) sampling; (3) attribute the cause of each unlearned sample to one of five categories via diagnostic tests and apply targeted interventions.

### Key Designs

1. **Unlearned Sample Detection (BoN-5 Sampling)**

    - **Function**: Reliably identify samples that consistently fail after training.
    - **Mechanism**: SFT samples are reformatted as multiple-choice questions and subjected to $N$ independent inference runs; the pass@N ratio is computed. Samples with pass@5 $< 0.2$ that remain stable across random seeds are flagged as unlearned. The top-$K=1000$ most severe cases are selected for in-depth analysis.
    - **Design Motivation**: To distinguish genuine learning failure from stochastic decoding noise — repeated failure across multiple samples indicates that the knowledge has truly not been internalized.

2. **Diagnosis and Intervention for Five ILP Sources**

    - **Function**: Attribute unlearned samples to specific causes and validate causal relationships.
    - **Mechanism**:
      - *Pretrained Knowledge Absence*: Factual triples are extracted via OpenIE; the base model is probed with BoN → knowledge augmentation + continued pre-training (CPT).
      - *Knowledge Conflict*: Detects high-confidence incorrect answers from the base model that contradict SFT labels → external knowledge correction + CPT.
      - *Intra-SFT Data Contradiction*: Label inconsistency among semantically similar samples → GPT-based evaluation + bucketed training to avoid mini-batch conflicts.
      - *Left-Side Forgetting*: Early samples overwritten by later ones during sequential dataset processing → random shuffling + dynamic resampling.
      - *Insufficient Optimization*: Inadequate training signal for rare or complex patterns → increased training or re-weighting.
    - **Design Motivation**: Mitigation strategies serve as causal interventions rather than general solutions — the effectiveness of strategy $X$ validates cause $Y$.

3. **Knowledge State Probing (JSD Diagnosis)**

    - **Function**: Distinguish knowledge absence from knowledge conflict.
    - **Mechanism**: The Jensen–Shannon divergence between the prediction distributions of the base model and the fine-tuned model is computed. High JSD + base model error = knowledge conflict; low JSD + fine-tuned model still incorrect = knowledge absence.
    - **Design Motivation**: Inspecting final accuracy alone cannot differentiate "does not know" from "knows but incorrectly" — distribution-level signals are required.

### Loss & Training

Standard SFT cross-entropy loss. Experiments are conducted on Qwen, LLaMA, and OLMo2. CPT employs a mixed corpus $\mathcal{C}_{\text{mix}} = 0.8\mathcal{C}_{\text{general}} + 0.2\mathcal{C}_{\text{aug}}$.

## Key Experimental Results

### Main Results

**ILP Prevalence (averaged over 10 benchmark SFT datasets)**

| Metric | Value |
|--------|-------|
| Average unlearned sample ratio | $15.3\%\pm2.1\%$ |
| Cross-model consistency | Observed on Qwen / LLaMA / OLMo2 |
| Cross-domain consistency | Present in medical / legal / financial domains |

### Ablation Study

**CPT Intervention Effect (Knowledge Absence + Conflict Categories)**

| Domain | SFT only Acc | +CPT Acc | Gain |
|--------|-------------|---------|------|
| Medical (MedQA) | baseline | significant improvement | Validates knowledge absence hypothesis |
| Legal (LegalBench) | baseline | significant improvement | Validates knowledge conflict hypothesis |
| Financial (FinanceBench) | baseline | significant improvement | — |

### Key Findings

- ILP is pervasive and heterogeneous — no single intervention resolves all failure modes.
- Knowledge absence and knowledge conflict are the two most prevalent sources; CPT is effective for both.
- Left-side forgetting is particularly severe in multi-task SFT — simply shuffling the data order mitigates most of this effect.
- ILP caused by intra-SFT annotation inconsistency can be partially addressed through bucketed training.
- Improvements in aggregate metrics can conceal persistently unlearned subsets — sample-level monitoring is necessary.

## Highlights & Insights

- The conceptualization of "ILP" itself constitutes a significant contribution — it formalizes a widely observed but previously unsystematized phenomenon.
- The five-source taxonomy offers direct practical guidance for SFT practitioners, providing a checklist for diagnosing data and model issues.
- The framework adopts a *diagnosis-before-treatment* philosophy — understanding the cause of failure precedes designing targeted remedies.

## Limitations & Future Work

- Sample-level evaluation via multiple-choice reformatting may introduce evaluation bias.
- The computational costs of mitigation strategies (CPT, bucketed training, etc.) are not reported in detail.
- The five-source taxonomy may be incomplete — additional, unidentified ILP causes may exist.
- It remains unexplored whether subsequent training stages such as RLHF or DPO exacerbate or alleviate ILP.

## Related Work & Insights

- **vs. Catastrophic Forgetting**: The latter concerns the loss of previously acquired capabilities, whereas ILP concerns the failure to acquire new knowledge — the two phenomena are directionally opposite.
- **vs. Data Quality Research**: Prior work on data quality typically targets overall performance improvement, while ILP focuses on why specific samples are not learned.
- **vs. Curriculum Learning (Bengio et al., 2009)**: Curriculum learning orders training samples by complexity; ILP diagnostics indicate that ordering alone is insufficient — it is necessary to identify and address five distinct failure modes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of incomplete learning in SFT; both the concept and the taxonomy are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model × multi-domain evaluation + causal intervention validation + 10 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; the diagnostic framework is logically rigorous.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both SFT practice and theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](../../AAAI2026/llm_safety/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

</div>

<!-- RELATED:END -->
