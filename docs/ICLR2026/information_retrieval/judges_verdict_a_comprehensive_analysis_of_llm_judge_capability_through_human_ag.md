---
title: >-
  [Paper Note] Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement
description: >-
  [ICLR 2026][Information Retrieval & RAG][LLM-as-a-Judge] This paper proposes the Judge's Verdict Benchmark—a two-stage evaluation framework based on relevance filtering followed by a Cohen's Kappa human-similarity test—t…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "LLM-as-a-Judge"
  - "Cohen's Kappa"
  - "human agreement"
  - "benchmark"
  - "RAG evaluation"
  - "Turing Test for judges"
date: 2026-05-08
content_hash: 7004cf18c20bbbfa
---

# Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement

**Conference**: ICLR 2026
**arXiv**: [2510.09738](https://arxiv.org/abs/2510.09738)  
**Code**: [nvidia/judges-verdict](https://github.com/nvidia/judges-verdict)  
**Area**: Information Retrieval
**Keywords**: LLM-as-a-Judge, Cohen's Kappa, human agreement, benchmark, RAG evaluation, Turing Test for judges

## TL;DR

This paper proposes the Judge's Verdict Benchmark—a two-stage evaluation framework based on relevance filtering followed by a Cohen's Kappa human-similarity test—to systematically assess 54 LLM judges. The framework identifies 27 Tier 1 judges (23 human-like and 4 super-consistent). The central finding is that high correlation does not imply high agreement; Kappa combined with z-score is necessary to properly measure LLM judge quality.

## Background & Motivation

LLM-as-a-Judge has become the dominant paradigm for evaluating output quality in RAG and agent pipelines, yet existing evaluation methods suffer from fundamental flaws. **First**, the vast majority of prior work relies solely on Pearson correlation to validate judge quality. However, correlation measures linear relationships rather than absolute agreement—an LLM judge that is systematically stricter (or more lenient) can achieve $r = 0.95$ while having a Cohen's Kappa of only 0.45, indicating that it captures scoring trends but fails to align with human judgments at the absolute level. **Second**, different studies use different datasets and varying numbers of annotators, resulting in a lack of standardized benchmarks. **Third**, existing methods entirely overlook distinctions in agreement patterns—some LLM judges exhibit "natural human-like variation," while others display agreement levels that exceed inter-human consistency. These two patterns carry fundamentally different implications for downstream applications.

This paper seeks to answer a core question: how can a rigorous methodology distinguish LLM judges that "genuinely reach human-level performance" from those that are merely correlated but inconsistent with human judgments?

## Method

### Overall Architecture: Two-Stage Evaluation

The methodology consists of two progressive stages:

**Step 1 — Correlation Filtering**: The Pearson $r$ between each LLM judge's Answer Accuracy scores and the human consensus scores (mean of three annotators) is computed, with a threshold of $r \geq 0.80$ ("very strong"). This step serves as a necessary but not sufficient condition—filtering out models that fail to align even at the level of linear trend.

**Step 2 — Cohen's Kappa + Human Similarity Test**: Two complementary analyses are applied to models passing Step 1:

- **Static Baseline Comparison**: The mean Cohen's Kappa $\bar{\kappa}_{LLM}$ between the LLM and each of the three human annotators is computed and compared against the inter-human baseline $\kappa = 0.801$.
- **Dynamic Group Analysis (Judge Turing Test)**: The LLM is inserted into a group with three human annotators to form a four-person panel. Cohen's Kappa is computed for all pairwise combinations, and a z-score is used to determine whether the LLM can "blend in" among the human annotators: $z = (\kappa_{LLM} - \mu_{human}) / \sigma_{human}$.

### Key Designs

**Tiered Classification Scheme**: Based on the two-stage results, judges are categorized into three tiers—Tier 1A Human-Like (r ≥ 0.80 and |z| < 1, indicating agreement patterns consistent with natural human variation), Tier 1B Super-Consistent (r ≥ 0.80 and z > 1, exhibiting agreement exceeding inter-human levels), and Tier 2 and below (failing either stage).

**Answer Accuracy Scoring Design**: A dual-prompt strategy from the RAGAS framework is adopted. Two independent LLM judge prompts receive the question, generated answer, and reference answer in different orders, each producing a discrete score $S \in \{0, 2, 4\}$, which are normalized and averaged. The dual-prompt design substantially mitigates position bias.

**Dataset Construction**: Six heterogeneous benchmarks totaling 1,994 samples are used—SQuAD v2.0 (346), HotPotQA (342, multi-hop reasoning), Coral (318, conversational QA), TechQA (295, technical QA), DC767 (347, PDF documents with tables and figures), and EKRAG (346, enterprise knowledge RAG with earnings reports and SEC filings). Three expert annotators from North America independently labeled each sample (5,982 annotations in total), yielding Fleiss $\kappa = 0.79$ and Krippendorff $\alpha = 0.79$, indicating high annotation quality.

## Key Experimental Results

### Two-Stage Filtering Results

Of 54 LLM judges (43 open-source and 11 proprietary, ranging from 1B to 405B parameters), 36 passed Step 1 ($r \geq 0.80$), and 27 were identified as Tier 1 judges after Step 2. The top-ranked models among Tier 1 are as follows:

| Rank | Model | Pearson r | Cohen's κ | z-score | Category |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | Mixtral-8x22B-Instruct | 0.879 | 0.813 | 1.45 | Super-Consistent |
| 2 | Llama-3-70B-Instruct | 0.880 | 0.811 | 1.43 | Super-Consistent |
| 3 | Gemma-3-27B-IT | 0.879 | 0.812 | 1.34 | Super-Consistent |
| 4 | Bagel-34B-v0.2 | 0.872 | 0.804 | 1.01 | Super-Consistent |
| 5 | GPT-4.5 | 0.874 | 0.806 | 0.90 | Human-Like |
| 6 | Llama-3.1-70B | 0.868 | 0.798 | 0.61 | Human-Like |
| 7 | GPT-4.1 | 0.862 | 0.792 | 0.41 | Human-Like |
| 12 | Qwen3-30B-A3B | 0.846 | 0.780 | -0.04 | Human-Like |
| 17 | Claude Sonnet 4 | 0.847 | 0.768 | -0.44 | Human-Like |
| 27 | Qwen2.5-32B | 0.831 | 0.753 | -0.96 | Human-Like |

Notably, all 27 Tier 1 models cluster within the Kappa range of 0.753–0.813, spanning the boundary between "substantial" and "almost perfect" on the Landis & Koch scale. The model closest to natural human judgment is Qwen3-30B-A3B ($|z| = 0.04$), suggesting that 30B-scale models with appropriate training can achieve human-level judge capability.

### Representative Cases Failing Filtering

A substantial number of high-correlation models are eliminated in Step 2, corroborating the paper's central thesis that correlation does not imply agreement:

| Model | Pearson r | Cohen's κ | z-score | Tier 1? |
|:---|:---:|:---:|:---:|:---:|
| GPT-4o | 0.818 | 0.728 | -1.55 | ✗ |
| GPT-4 | 0.811 | 0.723 | -1.73 | ✗ |
| GPT-5-chat | 0.809 | 0.720 | -1.85 | ✗ |
| GPT-4o-mini | 0.804 | 0.709 | -2.20 | ✗ |
| Llama-3.1-8B | 0.800 | 0.730 | -2.73 | ✗ |
| Llama-3.2-1B | 0.020 | 0.005 | -54.74 | ✗ |

GPT-4o achieves $r = 0.818$ (passing Step 1) but yields $\kappa = 0.728$ and $z = -1.55$, indicating that while it captures scoring trends, it is systematically more lenient than humans at the absolute judgment level. GPT-5-chat similarly passes Step 1 but is eliminated in Step 2. The most extreme case is Llama-3.2-1B, with $r \approx 0$ and $\kappa \approx 0$, reflecting essentially no judge capability.

### Z-Score Threshold Sensitivity Analysis

The authors examine how classification outcomes change under different $|z|$ thresholds (with $r \geq 0.80$ held fixed): $|z| < 0.5$ yields 18 Tier 1 models (12 human-like + 6 super-consistent); $|z| < 1.0$ yields 27 (23 + 4); $|z| < 1.5$ yields 29 (29 + 0, with the super-consistent category disappearing); and $|z| < 1.96$ yields 33 (33 + 0). The choice of $|z| < 1$ is justified because it identifies a sufficient number of human-like models while retaining the ability to detect super-consistent behavior—relaxing the threshold to 1.5 subsumes the super-consistent category, eliminating the framework's core capacity to distinguish the two patterns.

### Relationship Between Model Size and Judge Quality

Judge quality does not scale linearly with parameter count. Mistral-Nemo-12B (12B) achieves Tier 1 ($\kappa = 0.774$), as does Phi-3.5-MoE (MoE architecture, $\kappa = 0.775$), whereas Llama-3.1-8B (8B) and GPT-4o-mini pass Step 1 but are eliminated in Step 2. On the other hand, Llama-3.1-405B (405B) ranks in the middle of Tier 1. This suggests that architectural design and training strategies—particularly alignment training—exert a far greater influence on judge capability than raw parameter scale alone.

## Highlights & Insights

- **Paradigm Shift from Correlation to Agreement**: The paper's most significant contribution is its empirical demonstration of why Pearson $r$ is insufficient for evaluating LLM judges—addressing a methodological blind spot that has been widely overlooked in the field. A model with $r = 0.95$ that is systematically biased is entirely unsuitable as a human judge surrogate, yet prior literature would have deemed it an excellent judge.
- **Judge Turing Test**: The design of embedding LLMs within human annotator groups and using z-scores to detect deviations is elegant—it directly answers the practically relevant question of whether an LLM's judgment pattern is statistically indistinguishable from that of human annotators.
- **Dual Interpretation of Super-Consistent Models**: The four models with $z > 1$ exhibit higher consistency than inter-human agreement. The paper honestly acknowledges that this may reflect "more reliable judgment" or "oversimplification of complex assessments," and explicitly states that the current methodology cannot distinguish between the two interpretations, leaving the choice to the user—a degree of rigor more valuable than simply claiming "superhuman performance."
- **Practical Model Selection Guide**: The paper implicitly provides actionable recommendations—scenarios requiring preserved judgment diversity (content moderation, creative evaluation) should favor human-like models, while scenarios demanding maximum reproducibility (compliance checking, standardized testing) may prefer super-consistent models.

## Limitations & Future Work

1. **Coarse Scoring Granularity**: Only three score levels (0/0.5/1) are used, limiting assessment of fine-grained judgment capability; more continuous rating scales might reveal different agreement patterns.
2. **Narrow Task Coverage**: Evaluation is restricted to RAG answer accuracy and does not cover creative generation, code review, safety judgment, or other task types.
3. **Insufficient Annotator Diversity**: Only three North American native English speakers are included; the cultural and linguistic homogeneity of annotators may introduce systematic bias.
4. **Absence of Temporal Stability Analysis**: The consistency of a given model's judgments across different time points or random seeds is not examined.
5. **Unexplained Super-Consistent Behavior**: No sample-level analysis is conducted to characterize where super-consistent and human-like models diverge (e.g., whether they differ on boundary cases).

## Related Work & Insights

| Work | Core Method | Difference from Ours |
|:---|:---|:---|
| MT-Bench (Zheng et al.) | Pairwise preference judgment + Chatbot Arena | Does not use Cohen's Kappa; primarily relies on Elo ratings |
| G-Eval (Liu et al.) | CoT + form-guided scoring | Validates with correlation only; does not consider agreement |
| Prometheus (Kim et al.) | Fine-tuned dedicated 13B judge model | Focuses on Pearson $r = 0.897$; does not analyze agreement patterns |
| JudgeBench (Lin et al.) | Hard-sample pairwise discrimination test | Uses preference accuracy as primary metric; no statistical testing |
| JUDGE-BENCH (Bavaresco et al.) | Large-scale evaluation across 20 datasets | Finds high inter-model variance but proposes no tiered framework |

The core advancement of this paper is the shift from asking "is the LLM judge sufficiently correlated with humans?" to "are the LLM judge's decisions statistically indistinguishable from those of human annotators?"—a stricter and more practically meaningful question.

## Rating

| Dimension | Score |
|:---:|:---:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

> A methodology-driven empirical study offering an important corrective to the LLM-as-a-Judge community. The experimental scale is large and the conclusions are clear. While technically straightforward, the insights are sharp and the work provides direct practical reference for RAG evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](../../ICML2026/information_retrieval/understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[ACL 2026\] Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers](../../ACL2026/information_retrieval/code-switching_information_retrieval_benchmarks_analysis_and_the_limits_of_curre.md)

</div>

<!-- RELATED:END -->
