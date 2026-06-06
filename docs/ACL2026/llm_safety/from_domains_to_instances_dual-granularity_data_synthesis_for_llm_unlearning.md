---
title: >-
  [Paper Note] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning
description: >-
  [ACL 2026][LLM Safety][Machine Unlearning] This paper formally defines two unlearning granularities, domain-level and instance-level…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Unlearning Dataset Synthesis"
  - "Domain-level Unlearning"
  - "Instance-level Unlearning"
  - "Adversarial Probing"
date: 2026-05-08
content_hash: 7040abe4c46ae5a5
---

# From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning

**Conference**: ACL 2026  
**arXiv**: [2601.04278](https://arxiv.org/abs/2601.04278)  
**Code**: [GitHub](https://github.com/XiaoyuXU1/Biforget)  
**Area**: LLM Evaluation  
**Keywords**: Machine Unlearning, Unlearning Dataset Synthesis, Domain-level Unlearning, Instance-level Unlearning, Adversarial Probing

## TL;DR

This paper formally defines two unlearning granularities, domain-level and instance-level, and proposes the BiForget framework. It utilizes the target model itself (rather than external strong models) to generate high-quality unlearning datasets through two stages: seed-guided synthesis and adversarial probing. In the Harry Potter domain, it improves relevance by approximately 20 and diversity by approximately 0.05 while halving the data volume.

## Background & Motivation

**Background**: LLMs trained on massive corpora prone to memorizing private, harmful, or copyrighted content. Machine unlearning uses fine-tuning methods (gradient ascent, NPO, etc.) on defined forget and retain sets to optimize model behavior such that it approximates a state of never having seen the target data.

**Limitations of Prior Work**: (1) Existing unlearning benchmarks often fail to accurately reflect the true internal knowledge of the model—potentially overestimating or underestimating unlearning effects; (2) Benchmark construction depends heavily on manual curation (e.g., WMDP requires manual collection of domain text), which is hard to scale; (3) Existing work (e.g., TOFU) uses templated QA pairs, allowing the model to "pass" unlearning evaluations by merely suppressing surface patterns, while target knowledge can be recovered via rephrasing; (4) Dependence on external strong models (e.g., GPT-4o-mini) results in a mismatch between synthesized data and the target model’s knowledge boundaries.

**Key Challenge**: Effective unlearning must target underlying information rather than surface forms—semantic equivalent variants (rewriting, reordering) may still leak even after verbatim samples are removed. However, existing forget sets only cover original texts $D^{real}_f$ from the training corpus, failing to expand to the ideal forget set $D^{ideal}_f$.

**Goal**: (1) Formally define domain-level and instance-level unlearning granularities; (2) Design an automated framework to generate high-quality unlearning datasets aligned with the target model's internal knowledge distribution; (3) Propose a unified quality evaluation suite.

**Key Insight**: Leveraging the target model to generate unlearning data itself—ensuring synthesized data naturally aligns with the model’s knowledge boundaries and avoiding distribution mismatch issues introduced by external models.

**Core Idea**: Through a two-stage strategy of "seed-guided synthesis (broad coverage) + adversarial probing (excavating deep knowledge)," the target model is induced to expose its memorized knowledge, constructing an unlearning dataset more faithful to the model's true knowledge distribution.

## Method

### Overall Architecture

BiForget supports both domain-level and instance-level unlearning granularities. Domain-level unlearning employs a two-stage design: Stage I Seed-guided Synthesis (instantiating prompt templates with model-generated domain points to elicit diverse domain content) + Stage II Adversarial Probing (utilizing jailbreaking and membership inference attacks to excavate deep memory content). Instance-level unlearning adopts an information rewriting strategy (generating diverse semantic equivalent variants of target sentences). The input is a domain name or a target sentence, and the output is a high-quality synthesized forget set $\Omega_f$.

### Key Designs

1.  **Domain-level Synthesis—Seed-guided + Adversarial Probing**:

    - **Function**: Generate forget data covering a broad semantic space of the target domain.
    - **Mechanism**: In the pre-processing stage, the target model enumerates domain-related key point seeds (concepts, characters, etc.). Stage I uses QA-style and information synthesis templates to instantiate seeds and elicit diverse domain content. Diversity is promoted via temperature variations, and semantic convergence is monitored using SimCSE, terminating when incremental gain falls below the threshold $\epsilon=0.001$. Stage II uses jailbreak prompts to elicit safety-sensitive responses and membership inference (Min-k% token probability exceeding threshold $\tau$) to identify deeply memorized content.
    - **Design Motivation**: Heuristic prompts often miss implicit knowledge and stylistic variants. Adversarial probing exposes deep encoded knowledge unreachable by standard prompts—removing either component leads to a significant increase in privacy leakage in ablation experiments.

2.  **Instance-level Synthesis—Information Rewriting**:

    - **Function**: Generate diverse semantic equivalent variants for specific target sentences.
    - **Mechanism**: The target sentence serves as a seed, prompting the model to generate rewriting variants $x^* \sim q_{inst}$ from different perspectives, structures, or styles. Rewriting introduces minimal semantic shift and converges quickly (usually within one round). Convergence checks are delayed by setting a larger diversity batch $d_{inst}$ to ensure sufficient coverage.
    - **Design Motivation**: Templated formats in benchmarks like TOFU lead models to suppress only surface patterns—rewriting variants force unlearning to target semantic content rather than surface forms.

3.  **Unified Quality Evaluation Suite**:

    - **Function**: Comprehensively evaluate the quality of synthesized data.
    - **Mechanism**: Relevance (sampling 1,000 instances to calculate t-SNE distance from domain keyword centroids, smaller is better), Diversity (remote-clique metric to capture semantic variation, superior to Self-BLEU which only considers n-gram overlap), Efficiency (number of 128-token chunks, fewer is better).
    - **Design Motivation**: Prior evaluations relied on LLM judgments for relevance, which introduced bias and ignored generation efficiency. The unified suite provides objective evaluation criteria without requiring an ideal forget set.

### Loss & Training

BiForget is a data synthesis framework and does not involve model training. Synthesized data is used for fine-tuning in downstream unlearning algorithms (GA, NPO, OBLIVIATE, etc.). Static prompt templates are generated once offline by GPT-5; all synthesized data is produced by the target model itself.

## Key Experimental Results

### Main Results

**Data Quality Comparison in Harry Potter Domain**

| Dataset | Relevance (Centroid Dist.↓) | Diversity (Remote-Clique↑) | Efficiency (#Chunks↓) |
| :--- | :--- | :--- | :--- |
| HP book | 36.44 | 0.5277 | 8,401 |
| Textbook | 48.11 | 0.5324 | 20,806 |
| **Ours** | **14.94** | **0.5824** | **4,122** |

**Unlearning Performance Comparison on WMDP-bio (RMU Algorithm)**

| Dataset | WMDP-bio↓ | MMLU↑ | GSM8K↑ |
| :--- | :--- | :--- | :--- |
| Official | 28.42(↓60.0%) | 59.09(↓7.3%) | 72.59(↓0.7%) |
| Textbook | 32.99(↓53.6%) | 45.03(↓29.4%) | 71.49(↓2.2%) |
| **Ours** | **26.54(↓62.7%)** | **62.70(↓1.7%)** | **72.58(↓0.7%)** |

### Ablation Study

**Ablation of BiForget Components (Harry Potter, GA, PrivLeak)**

| Configuration | PrivLeak ($\in[-5\%,5\%]$) | $\Delta$ vs Ours |
| :--- | :--- | :--- |
| w/o Jailbreaking | -22.66 | -7.58 |
| w/o MI | -21.67 | -6.59 |
| w/o Both | -24.46 | -9.38 |
| **Ours (Full)** | **-15.08** | **0.00** |

### Key Findings

- BiForget improves relevance by ~20 (14.94 vs 36.44) on Harry Potter, increases diversity by 0.05 (0.5824 vs 0.5277), and halves the data volume (4,122 vs 8,401 chunks).
- On WMDP-bio, BiForget + RMU achieves the strongest unlearning (↓62.7%) while retaining the most general capability (MMLU only ↓1.7%)—compared to MMLU ↓29.4% for Textbook.
- On TOFU, OBLIVIATE + BiForget achieves the optimal unlearning-utility balance (F.Q.=0.92, M.U.=0.65), significantly outperforming the Official data's F.Q.=0.08.
- Ablation of adversarial components confirms both are essential: removing jailbreaking increases privacy leakage by 7.58, and removing membership inference increases it by 6.59.
- Performance in the Cybersecurity domain is weaker—synthesis quality is limited because the target model has relatively thin knowledge in this area.

## Highlights & Insights

- The approach of "letting the model expose its own knowledge" is ingenious—target-model-guided synthesis naturally solves the distribution alignment problem and avoids knowledge boundary mismatches from external models.
- The formal distinction between domain-level and instance-level granularities provides a clear problem framework for unlearning research—these two granularities were often conflated in previous discussions.
- The design of the adversarial probing stage directly integrates red-teaming techniques into the data construction pipeline—not only improving unlearning robustness but also providing insights for other data synthesis scenarios.

## Limitations & Future Work

- Synthesis quality is limited by the target model’s domain knowledge—effectiveness is compromised in domains where the model's knowledge is weak (e.g., Cybersecurity).
- Currently focuses on single unlearning requests and has not yet expanded to continual or multi-domain dynamic unlearning.
- Prompt quality and sampling randomness may result in semantic drift or uneven domain coverage.
- Safety-critical domains may require stronger gold-standard references (such as retrained models) to verify synthesis quality.

## Related Work & Insights

- **vs Textbook-style (Zhu et al.)**: Textbook relies on an external generator (GPT-4o-mini), whereas BiForget uses the target model itself, avoids distribution mismatch, and halves data volume.
- **vs TOFU**: TOFU uses templated QA pairs, allowing knowledge restoration via rephrasing after unlearning. BiForget’s rewriting variants force unlearning to target semantic content.
- **vs MUSE/HP Book**: Official datasets have high relevance but fail to cover semantic equivalent variants; BiForget expands the scope of unlearning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to formalize dual-granularity unlearning + target-model-guided synthesis + adversarial probing integration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three domains (HP/WMDP/TOFU) + five unlearning algorithms + component ablation + adversarial robustness.
- **Writing Quality**: ⭐⭐⭐⭐ Formal definitions are clear, experiments are comprehensive, though some tables are high-density.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic solution for unlearning data quality issues; directly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](../../ICML2026/llm_safety/differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_safety/revisiting_the_past_data_unlearning_with_model_state_history.md)

</div>

<!-- RELATED:END -->
