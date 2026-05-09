---
title: >-
  [Paper Note] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning
description: >-
  [ACL 2026][LLM Evaluation][Machine Unlearning] This paper formally defines two granularities of LLM unlearning—domain-level and instance-level—and proposes the BiForget framework. Rather than relying on external strong models, BiForget leverages the target model itself to construct high-quality forget datasets via two stages: seed-guided synthesis and adversarial probing. On the Harry Potter domain, it improves relevance by ~20 and diversity by ~0.05 while halving the data volume.
tags:
  - ACL 2026
  - LLM Evaluation
  - Machine Unlearning
  - Forget Set Synthesis
  - Domain-Level Unlearning
  - Instance-Level Unlearning
  - Adversarial Probing
date: 2026-05-08
content_hash: 8f8fdb56c8f14502
---

# From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning

**Conference**: ACL 2026  
**arXiv**: [2601.04278](https://arxiv.org/abs/2601.04278)  
**Code**: [GitHub](https://github.com/XiaoyuXU1/Biforget)  
**Area**: LLM Evaluation  
**Keywords**: Machine Unlearning, Forget Set Synthesis, Domain-Level Unlearning, Instance-Level Unlearning, Adversarial Probing

## TL;DR

This paper formally defines two granularities of LLM unlearning—domain-level and instance-level—and proposes the BiForget framework. Rather than relying on external strong models, BiForget leverages the target model itself to construct high-quality forget datasets via two stages: seed-guided synthesis and adversarial probing. On the Harry Potter domain, it improves relevance by ~20 and diversity by ~0.05 while halving the data volume.

## Background & Motivation

**Background**: LLMs trained on massive corpora tend to memorize private, harmful, or copyrighted content. Machine unlearning mitigates this through fine-tuning methods (gradient ascent, NPO, etc.) optimized over a defined forget set and retain set, steering model behavior toward that of a model never trained on the target data.

**Limitations of Prior Work**: (1) Forget sets in existing unlearning benchmarks often fail to accurately reflect the model's true internal knowledge, leading to over- or under-estimation of unlearning effectiveness; (2) Benchmark construction relies heavily on manual curation (e.g., WMDP requires hand-collected domain texts), limiting scalability; (3) Existing work (e.g., TOFU) uses templated QA pairs, allowing models to "pass" unlearning evaluations by suppressing surface patterns while recovering target knowledge upon rephrasing; (4) Relying on external strong models (e.g., GPT-4o-mini) to generate unlearning data causes a mismatch between the synthetic data and the target model's knowledge boundary.

**Key Challenge**: Effective unlearning must target underlying information rather than surface form—semantically equivalent variants (paraphrases, reorderings) can still leak knowledge even after verbatim samples are removed. Yet existing forget sets cover only the original training text $D^{real}_f$, without extending to the ideal forget set $D^{ideal}_f$.

**Goal**: (1) Formally define domain-level and instance-level unlearning granularities; (2) Design an automated framework that generates high-quality forget datasets aligned with the target model's internal knowledge distribution; (3) Propose a unified quality evaluation suite.

**Key Insight**: Have the target model generate its own unlearning data—synthetic data produced this way is naturally aligned with the model's knowledge boundary, avoiding the distributional mismatch introduced by external models.

**Core Idea**: Through a two-stage strategy of "seed-guided synthesis (broad coverage) + adversarial probing (deep knowledge elicitation)," the target model is prompted to expose its own memorized knowledge, yielding a forget dataset that more faithfully reflects the model's true knowledge distribution.

## Method

### Overall Architecture

BiForget supports both domain-level and instance-level unlearning granularities. Domain-level unlearning employs a two-stage design: Stage I performs seed-guided synthesis (instantiating prompt templates with model-generated domain key points to elicit diverse domain content), and Stage II performs adversarial probing (jailbreaking and membership inference attacks to surface deep memorized content). Instance-level unlearning employs an information paraphrasing strategy (generating diverse semantically equivalent variants of target sentences). The input is a domain name or target sentence; the output is a high-quality synthetic forget set $\Omega_f$.

### Key Designs

1. **Domain-Level Synthesis — Seed Guidance + Adversarial Probing**:

    - Function: Generate forget data covering a broad semantic space over the target domain.
    - Mechanism: In the preprocessing stage, the target model enumerates domain-relevant seed key points (concepts, characters, etc.). Stage I instantiates seeds using QA-style and information-synthesis templates to elicit diverse domain content from the target model; temperature variation promotes diversity, and SimCSE monitors semantic convergence, terminating when incremental gain falls below threshold $\epsilon=0.001$. Stage II uses jailbreak prompts to elicit safety-sensitive responses and applies membership inference (Min-k% token probability exceeding threshold $\tau$) to identify deeply memorized content.
    - Design Motivation: Heuristic prompts frequently miss implicit knowledge and stylistic variants; adversarial probing exposes deeply encoded knowledge inaccessible to standard prompts. Ablation experiments confirm that removing either component significantly increases privacy leakage.

2. **Instance-Level Synthesis — Information Paraphrasing**:

    - Function: Generate diverse semantically equivalent variants for a specific target sentence.
    - Mechanism: The target sentence serves as the seed; the model is prompted to generate paraphrase variants $x^* \sim q_{inst}$ from different perspectives, structures, or styles. Semantic drift introduced by paraphrasing is small, so convergence is fast (typically within one round); a larger diversity batch $d_{inst}$ delays convergence checking to ensure sufficient coverage.
    - Design Motivation: The templated format of benchmarks such as TOFU causes models to suppress only surface patterns. Paraphrase variants force unlearning to target semantic content rather than surface form.

3. **Unified Quality Evaluation Suite**:

    - Function: Comprehensively evaluate the quality of synthesized data.
    - Mechanism: Relevance (t-SNE distance to the domain keyword centroid computed over 1,000 sampled instances; lower is better), Diversity (remote-clique metric capturing semantic variation, superior to Self-BLEU which only measures surface n-gram overlap), and Efficiency (number of 128-token chunks; lower is better).
    - Design Motivation: Prior evaluations relying on LLM-based relevance judgments introduce bias and overlook generation efficiency. The unified suite provides objective, reference-free evaluation criteria that do not require an ideal forget set.

### Loss & Training

BiForget is a data synthesis framework and does not involve model training. The synthesized data is used for fine-tuning in downstream unlearning algorithms (GA, NPO, OBLIVIATE, etc.). Static prompt templates are generated offline once by GPT-5; all synthetic data is produced by the target model itself.

## Key Experimental Results

### Main Results

**Harry Potter Domain Data Quality Comparison**

| Dataset | Relevance (Centroid Dist.↓) | Diversity (Remote-Clique↑) | Efficiency (#Chunks↓) |
|-------|------------------------|----------------------|----------------|
| HP book | 36.44 | 0.5277 | 8,401 |
| Textbook | 48.11 | 0.5324 | 20,806 |
| **BiForget** | **14.94** | **0.5824** | **4,122** |

**WMDP-bio Unlearning Performance Comparison (RMU algorithm)**

| Dataset | WMDP-bio↓ | MMLU↑ | GSM8K↑ |
|-------|----------|-------|--------|
| Official | 28.42(↓60.0%) | 59.09(↓7.3%) | 72.59(↓0.7%) |
| Textbook | 32.99(↓53.6%) | 45.03(↓29.4%) | 71.49(↓2.2%) |
| **BiForget** | **26.54(↓62.7%)** | **62.70(↓1.7%)** | **72.58(↓0.7%)** |

### Ablation Study

**BiForget Component Ablation (Harry Potter, GA, PrivLeak)**

| Configuration | PrivLeak (∈[-5%,5%]) | Δ vs BiForget |
|------|---------------------|---------------|
| w/o Jailbreaking | -22.66 | -7.58 |
| w/o MI | -21.67 | -6.59 |
| w/o Both | -24.46 | -9.38 |
| **BiForget (Full)** | **-15.08** | **0.00** |

### Key Findings

- BiForget improves relevance by ~20 on Harry Potter (14.94 vs. 36.44), diversity by 0.05 (0.5824 vs. 0.5277), and halves the data volume (4,122 vs. 8,401 chunks).
- On WMDP-bio, BiForget + RMU achieves the strongest unlearning (↓62.7%) while retaining the most general capability (MMLU only ↓1.7%), compared to Textbook's MMLU ↓29.4%.
- On TOFU, OBLIVIATE + BiForget achieves the best forget-utility balance (F.Q.=0.92, M.U.=0.65), far surpassing Official (F.Q.=0.08).
- Ablation of adversarial components confirms both are important: removing jailbreaking increases privacy leakage by 7.58; removing membership inference increases it by 6.59.
- Performance is weaker in the cybersecurity domain, as the target model has limited domain knowledge there, constraining synthesis quality.

## Highlights & Insights

- The idea of "having the model expose its own knowledge" is particularly elegant—target-model-guided synthesis naturally resolves the distribution alignment problem, eliminating the knowledge boundary mismatch caused by external models.
- The formal distinction between domain-level and instance-level granularities provides a clear problem framework for unlearning research, where these two granularities have often been conflated.
- Integrating adversarial probing directly into the data construction pipeline not only improves unlearning robustness but also offers a transferable paradigm for other data synthesis scenarios.

## Limitations & Future Work

- Synthesis quality is bounded by the target model's domain knowledge—performance degrades in domains where the model's knowledge is weak (e.g., cybersecurity).
- The framework currently targets single unlearning requests and has not been extended to continual or multi-domain dynamic unlearning.
- Prompt quality and sampling stochasticity may introduce semantic drift or uneven domain coverage.
- Safety-critical domains may require stronger gold-standard references (e.g., retrained models) to validate synthesis quality.

## Related Work & Insights

- **vs. Textbook-style (Zhu et al.)**: Textbook relies on external generators (GPT-4o-mini); BiForget uses the target model itself, avoiding distributional mismatch and halving the data volume.
- **vs. TOFU**: TOFU uses templated QA pairs, allowing knowledge to be recovered through rephrasing after unlearning. BiForget's paraphrase variants compel unlearning to genuinely target semantic content.
- **vs. MUSE/HP Book**: Official datasets have high relevance but do not cover semantically equivalent variants; BiForget extends the scope of unlearning accordingly.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First formal dual-granularity unlearning framework + target-model-guided synthesis + integrated adversarial probing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three domains (HP/WMDP/TOFU) + five unlearning algorithms + component ablation + adversarial robustness evaluation.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and experiments are comprehensive, though some tables are dense.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic solution to the problem of forget set data quality; directly applicable.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_evaluation/llm_unlearning_with_llm_beliefs.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_evaluation/revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ICLR 2026\] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](../../ICLR2026/llm_evaluation/rethinking_benign_relearning_syntax_as_the_hidden_driver_of_unlearning_failures.md)

<!-- RELATED:END -->
