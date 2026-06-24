---
title: >-
  [Paper Note] Rethinking Table Instruction Tuning
description: >-
  [ACL 2025][LLM Alignment][table understanding] This work systematically ablates overlooked hyperparameter choices (learning rate, data scale, and epoch count) in table instruction tuning. It reveals that existing table LLMs suffer from severe degradation of general capabilities (MMLU drops by 14 points, AI2ARC by 21 points) due to excessively large learning rates ($2 \times 10^{-5}$). To address this, the authors propose TAMA, which is constructed by tuning LLaMA 3.1 8B Instr…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "table understanding"
  - "instruction tuning"
  - "hyperparameter analysis"
  - "data efficiency"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 49c56e9304ebaaae
---

# Rethinking Table Instruction Tuning

**Conference**: ACL 2025  
**arXiv**: [2501.14693](https://arxiv.org/abs/2501.14693)  
**Code**: [MichiganNLP/TAMA](https://github.com/MichiganNLP/TAMA)  
**Area**: Alignment RLHF / Instruction Tuning  
**Keywords**: table understanding, instruction tuning, hyperparameter analysis, data efficiency, catastrophic forgetting

## TL;DR

This work systematically ablates overlooked hyperparameter choices (learning rate, data scale, and epoch count) in table instruction tuning. It reveals that existing table LLMs suffer from severe degradation of general capabilities (MMLU drops by 14 points, AI2ARC by 21 points) due to excessively large learning rates ($2 \times 10^{-5}$). To address this, the authors propose TAMA, which is constructed by tuning LLaMA 3.1 8B Instruct on only 2,600 samples (200 samples from each of 13 datasets) with a learning rate of $1 \times 10^{-6}$ and for 2 epochs. TAMA matches or outperforms GPT-3.5 and GPT-4 across 13 table tasks while fully preserving its general capabilities.

## Background & Motivation

**Background**: Table understanding is an important direction in NLP. Recently, several works like TableLLaMA (2 million samples), TableLLM (309k samples), and TableBenchLLM (20k samples) have emerged to enhance the tabular capabilities of LLMs via large-scale instruction tuning. Influenced by the "big data training" paradigm of closed-source models, these works tend to collect massive tabular data for full-parameter fine-tuning, uniformly using a "default" learning rate of $lr = 2 \times 10^{-5}$.

**Limitations of Prior Work**: (1) Severe degradation of general capabilities: TableLLaMA's MMLU score drops by 13.95 points ($44.22 \rightarrow 30.27$), and TableBenchLLM's AI2ARC score drops by 20.90 points ($74.40 \rightarrow 53.50$). (2) Poor out-of-domain generalization: All existing table LLMs perform worse on the unseen Table-Syn dataset than their base models. (3) Loss of instruction-following capability: TableLLM's IFEval score drops from 48.32 to 30.46, failing to return JSON format as requested by users.

**Key Challenge**: Existing works focus entirely on the data scale while neglecting the critical role of hyperparameter selection—a learning rate of $2 \times 10^{-5}$ is too large for models that have already undergone instruction tuning, which is the fundamental cause of catastrophic forgetting.

**Goal**: Through systematic ablation, this work aims to find the optimal strategy in table instruction tuning to "preserve general capabilities while enhancing tabular performance", demonstrating that a smaller learning rate and a small amount of data can achieve both goals simultaneously.

**Key Insight**: Instead of striving for more data, this study systematically explores the impact of hyperparameters such as learning rate, data quantity, epoch count, and multi-task synergy on both tabular performance and general capabilities, validating the generalizability of the findings across 5 different foundation models.

**Core Idea**: In table instruction tuning, an excessively large learning rate is the root cause of general capability degradation and poor out-of-domain generalization; a careful selection of hyperparameters ($lr = 1 \times 10^{-6}$, 2,600 samples, 2 epochs) is far superior to brute-force training on massive datasets.

## Method

### Overall Architecture

This work does not propose a new model architecture, but rather a systematic empirical study and practical guide. The overall pipeline consists of three phases:
1. **Diagnostic Phase** (Section 2): Evaluating the level of degradation of TableLLaMA, TableLLM, and TableBenchLLM on out-of-domain tabular tasks (Table-Syn) and general benchmarks (MMLU, IFEval, AI2ARC) to quantify the severity of the issue.
2. **Ablation Phase** (Section 3): Systematically exploring the effects of learning rate (five levels from $1 \times 10^{-7}$ to $1 \times 10^{-5}$) $\times$ data quantity (50 to 1,500 samples) $\times$ epoch count $\times$ multi-task combinations on LLaMA 3.1 8B Instruct, using three representative datasets: TabFact, FeTaQA, and HiTab.
3. **Construction Phase** (Section 4): Constructing TAMA based on the ablation findings—sampling 200 items from each of the 13 tabular datasets (2,600 items in total) and performing full-parameter fine-tuning with $lr = 1 \times 10^{-6}$ for 2 epochs.

### Key Design 1: Learning Rate is the Decisive Factor

- **Function**: Performing a grid search across 5 learning rates ($1 \times 10^{-7}$, $5 \times 10^{-7}$, $1 \times 10^{-6}$, $5 \times 10^{-6}$, $1 \times 10^{-5}$) $\times$ multiple data quantities, while monitoring the performance of both tabular and general tasks.
- **Key Findings**: $1 \times 10^{-6}$ / $5 \times 10^{-7}$ is the optimal range. Taking TabFact as an example, $lr = 1 \times 10^{-6}$ + 1,500 samples achieves 73.10 (best). At $lr = 1 \times 10^{-5}$, MMLU and IFEval crash significantly. While $lr = 1 \times 10^{-7}$ preserves general capabilities, its tabular improvement is insufficient (FEVEROUS 66.86 vs. 74.63 of $5 \times 10^{-6}$). Further verification on LLaMA 2 7B, Qwen 2.5 7B, Mistral v0.3 7B, and Phi 3 7B shows that the optimal learning rate universally falls within $1 \times 10^{-6}$ to $5 \times 10^{-7}$ (Table 5).
- **Design Motivation**: Existing table LLMs uniformly adopt $2 \times 10^{-5}$ as the learning rate, which is excessively large for already instruction-tuned models. A lower learning rate achieves a Pareto optimum between table and general capabilities.

### Key Design 2: Small Data Volume is Sufficient to Saturate

- **Function**: Fixing the learning rate and observing the performance curves as the data quantity increases from 50 to 1,500.
- **Key Findings**: The first 200 samples provide a rapid boost (the model quickly learns tabular reasoning patterns), after which marginal utility diminishes. On HiTab, $lr = 1 \times 10^{-6}$ + 1,500 samples achieves 66.29, outperforming TableLLaMA's 64.71 (which utilized the entire 2 million training samples of HiTab). On FEVEROUS, 1,500 samples achieve 74.63, outperforming TableLLaMA's 73.77.
- **Design Motivation**: LLMs already possess fundamental table understanding capabilities during the pre-training stage. The role of instruction tuning is primarily to "activate" rather than to "learn from scratch"; thus, a small amount of high-quality data is sufficient. The final configuration selects 200 samples per dataset.

### Key Design 3: Multi-task Synergy and Epoch Control

- **Multi-task Synergy**: Positive transfer effects (synergy) exist across different tabular tasks. Thus, 200 samples are sampled from each of 13 diverse datasets (covering 4 main categories: QA, fact verification, dialogue generation, and data-to-text), totaling 2,600 samples. Multi-task training shows better out-of-domain generalization compared to single-task training.
- **Epoch Selection**: Increasing the number of epochs does not yield significant improvements; 2 to 3 epochs is the optimal range. 2 epochs is chosen to prevent overfitting.

### Loss & Training

Full-parameter fine-tuning of LLaMA 3.1 8B Instruct is conducted with $lr = 1 \times 10^{-6}$ for 2 epochs, using 2,600 training samples in total. The paper also explores the optimal learning rates under LoRA and QLoRA settings, providing a comprehensive list of recommendations (Appendix D.5).

## Key Experimental Results

### Main Tests: TAMA vs. Baselines (Selected from Tables 7 & 9)

| Task | LLaMA 3.1 Base | TAMA | GPT-3.5 | GPT-4 |
|------|---------------|------|---------|-------|
| HiTab (Acc) | 32.83 | **63.51** | 43.62 | 48.40 |
| WikiSQL (Acc) | 20.43 | **68.31** | 41.91 | 47.60 |
| HybridQA (Acc) | 32.83 | **60.86** | 40.22 | 58.60 |
| TabFact (Acc) | 58.44 | **73.82** | 67.41 | 74.40 |
| FEVEROUS (Acc) | 66.37 | **77.39** | 60.79 | 71.60 |
| InfoTabs (Acc) | 48.39 | **64.54** | 56.00 | 58.60 |
| AIT-QA (Acc) | 82.54 | **89.21** | 84.13 | 88.57 |
| OOD Table-Syn S1 | 53.60 | **64.93** | 54.80 | 80.20 |

### General Capability Preservation (Table 9)

| Benchmark | LLaMA 3.1 Base | TAMA | Difference |
|-----------|---------------|------|------|
| MMLU | 66.04 | 66.99 | +0.95 |
| IFEval | 79.62 | 74.70 | -4.92 |
| AI2ARC | 80.89 | 81.23 | +0.34 |
| MMLUPro | 22.10 | 31.84 | +9.74 |
| GPQA | 32.14 | 31.92 | -0.22 |

### Degradation Comparison of Existing Table LLMs (Table 4)

| Model | MMLU Change | AI2ARC Change | IFEval Change | Table-Syn Change |
|------|----------|------------|------------|---------------|
| TableLLaMA | -13.95 | -11.35 | -5.63 | -2.40 |
| TableLLM | -8.79 | -13.91 | -17.86 | -15.00 |
| TableBenchLLM | -9.41 | **-20.90** | +0.72 | -4.40 |
| **TAMA** | **+0.95** | **+0.34** | -4.92 | **+11.33** |

### Ablation Study: Impact of Learning Rate (Figure 2 Data)

| Learning Rate | TabFact (1,500 samples) | MMLU | IFEval |
|--------|-----------------|------|--------|
| $1 \times 10^{-7}$ | Suboptimal | Preserved | Preserved |
| $5 \times 10^{-7}$ | One of the best | Preserved | Preserved |
| **$1 \times 10^{-6}$** | **73.10** | **Preserved** | **Mostly Preserved** |
| $5 \times 10^{-6}$ | Good | Slight Decrease | Decrease |
| $1 \times 10^{-5}$ | Suboptimal | **Significant Collapse** | **Significant Collapse** |

### Key Findings
- **Learning rate is the most critical hyperparameter**: $1 \times 10^{-6}$ achieves a Pareto optimum between tabular and general capabilities, consistently holding true across 5 different foundation models.
- **200 samples achieve over 80% of final performance**: The marginal utility of data scales diminishes rapidly. Just 2,600 samples can match the performance of TableLLaMA, which was trained on 2 million samples.
- **TAMA even shows improvements in STEM-related MMLU categories**: Since tabular tasks involve data analysis and mathematical reasoning, the STEM subset score rises from 56.03 to 58.25 after fine-tuning, explaining the substantial improvement on MMLUPro.
- **Instruction-following is a critically overlooked capability**: Existing table LLMs collapse entirely on IFEval (scoring between 25 and 32), while TAMA maintains a score of 74.70, close to GPT-3.5.

## Highlights & Insights

- **Core Insight: "Less is More"**: 2,600 samples + correct learning rate > 2 million samples + incorrect learning rate. This offers crucial inspiration for any domain-specific fine-tuning work—when continuing to tune an already instruction-tuned model, the learning rate should be dramatically reduced (e.g., from $2 \times 10^{-5}$ to $1 \times 10^{-6}$); otherwise, the gained domain capability does not justify the loss of general capabilities.
- **"Diagnosis-First" Research Paradigm**: First quantifying the degradation of existing methods, then pinpointing the cause through ablation, and finally constructing the model based on the ablation findings—this "diagnosis-ablation-construction" pipeline is highly exemplary.
- **Cross-Model Consistency Validation**: The universality of the learning rate conclusions is verified across five different architectures (LLaMA 2/3.1, Qwen 2.5, Mistral, and Phi 3), which greatly strengthens the credibility of the findings.
- **Counter-intuitive Improvements on STEM in MMLU**: Tabular fine-tuning unexpectedly enhances STEM reasoning, hinting at an implicit transfer of capabilities between these domains.

## Limitations & Future Work

- **Only Evaluating 7B/8B Scale Models**: The findings are not validated on larger models such as 13B or 70B, which might be more robust to learning rate variations.
- **Predominantly Full-Parameter Fine-Tuning**: Although the appendix provides LoRA/QLoRA results, the main experiments and TAMA both utilize full-parameter fine-tuning, which incurs high costs in real-world deployment.
- **Tables Represented Solely as Text**: This work does not explore scenarios where tables are processed as images (multimodal table understanding), which is prevalent in practical applications.
- **Uniform Data Selection Strategy**: 200 samples are uniformly sampled from each dataset without exploring smarter selection strategies such as active learning or difficulty stratification.

## Related Work & Insights

- **vs. TableLLaMA (Zhang et al., 2024a)**: TableLLaMA uses 2 million samples + $lr = 2 \times 10^{-5}$ + 6 epochs to achieve 64.71 on HiTab; TAMA uses 2,600 samples + $lr = 1 \times 10^{-6}$ + 2 epochs to achieve a comparable 63.51, while fully preserving general capabilities.
- **vs. TableBenchLLM (Wu et al., 2024)**: TableBenchLLM fine-tunes LLaMA 3.1 using 20k synthetic samples, suffering a 20.9 point drop on AI2ARC; TAMA fine-tunes the identical base model with 2,600 real samples, with AI2ARC instead increasing by 0.34 points.
- **vs. LIMA (Zhou et al., 2024)**: The "less is more" conclusion of this work aligns with the findings in LIMA—LLMs acquire fundamental capabilities during pre-training, and instruction tuning only needs a gentle nudge to showcase/activate them.

## Rating
- Novelty: ⭐⭐⭐⭐ Rather than proposing a new method, it systematically exposes overlooked key factors, directly challenging the prevailing consensus of "the more data, the better."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Grid searches over 5 learning rates $\times$ multiple data quantities $\times$ multiple epochs $\times$ 5 base models $\times$ 13 datasets make the ablation study exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ Clear paradigm of diagnosis-ablation-construction, with intuitive data presentations.
- Value: ⭐⭐⭐⭐⭐ Holds direct practical value for all domain fine-tuning efforts; both the TAMA model and dataset are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)

</div>

<!-- RELATED:END -->
