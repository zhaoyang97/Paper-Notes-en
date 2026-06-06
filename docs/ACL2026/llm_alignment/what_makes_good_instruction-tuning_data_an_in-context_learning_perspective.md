---
title: >-
  [Paper Note] What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective
description: >-
  [ACL2026][LLM Alignment][Instruction tuning] This paper proposes weighted In-Context Influence (wICI), which measures the value of instruction data by whether using a candidate sample as a one-shot demonstration can redu…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Instruction tuning"
  - "Data selection"
  - "In-Context Learning"
  - "Sample influence"
  - "Diversity constraints"
date: 2026-05-08
content_hash: 0af829279b469663
---

# What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective

**Conference**: ACL2026  
**arXiv**: [2604.25132](https://arxiv.org/abs/2604.25132)  
**Code**: https://github.com/trust-nlp/SyntheticData-Curator  
**Area**: llm_alignment  
**Keywords**: Instruction tuning, Data selection, In-Context Learning, Sample influence, Diversity constraints

## TL;DR
This paper proposes weighted In-Context Influence (wICI), which measures the value of instruction data by whether using a candidate sample as a one-shot demonstration can reduce the instruction-following difficulty of related difficult probes. Under a 10% data budget, it outperforms or matches selection methods such as IFD, DEITA, NUGGETS, and SelectIT.

## Background & Motivation
**Background**: Instruction tuning typically relies on large-scale instruction-response datasets, such as Alpaca-GPT4 and WizardLM. Extensive research has found redundancy, noise, and uneven quality in these datasets. Consequently, training models that approach or even exceed full-data performance using a small number of high-value samples has become a critical problem for efficient alignment and low-cost fine-tuning.

**Limitations of Prior Work**: Existing data selection methods have different focuses. IFD/Superfiltering measures sample difficulty using perplexity or instruction-following difficulty; DEITA combines complexity, quality, and diversity rewards; NUGGETS treats candidate samples as one-shot demonstrations and measures improvement on a fixed anchor set. However, a fixed global anchor set ignores semantic relevance, binary scoring fails to reflect the magnitude of improvement, and these methods often incur high computational costs.

**Key Challenge**: A "difficult" sample is not necessarily a "good teacher." Difficult samples might simply be ones the model is inherently poor at or those with complex labeling, which do not necessarily serve as effective examples for related tasks. Conversely, the value of a good demonstration lies in its ability to make it easier for the model to complete semantically similar but not identical probes. Existing methods do not sufficiently distinguish between "self-difficulty" and "teaching influence on peer samples."

**Goal**: The authors explore three questions: what kind of data is suitable for instruction tuning from an ICL perspective; whether high-IFD difficult samples are also strong demonstrations; and whether samples with high ICL influence lead to better instruction-following performance after actual fine-tuning.

**Key Insight**: The paper reinterprets instruction-tuning data selection as "finding examples that can help related difficult tasks in-context." If a sample, when used as a one-shot demonstration, significantly reduces the generation difficulty of multiple semantically related probes, it is not only a good ICL example but likely a good fine-tuning sample as well.

**Core Idea**: For each candidate sample, a diverse and difficult probe set of semantically related samples is constructed. The reduction in IFD for these probes when the candidate is provided as a demonstration is measured, then aggregated into a wICI score weighted by semantic distance. Finally, a coreset is selected using diversity constraints.

## Method
The method consists of four steps: finding probes for each candidate, calculating in-context influence on these probes, ranking by wICI with diversity constraints for selection, and finally performing standard SFT with the selected subset. The framework requires no reward model training and does not rely on external knowledge bases.

### Overall Architecture
The input is an instruction dataset $D=\{(x_i,y_i)\}_{i=1}^n$ and a budget $k$, and the output is a training subset $Q$ of size $k$. Each candidate sample $a_i=(x_i,y_i)$ is tested as a one-shot demonstration: whether it can reduce the instruction-following difficulty of a related probe $b=(x_b,y_b)$. If it significantly reduces difficulty, the sample is considered to have a "teaching effect" on neighboring tasks.

The authors first define IFD as a sample difficulty metric: $IFD(y|x)=PPL(y|x)/PPL(y)$, where a larger value indicates the model benefits less from the instruction and finds generation more difficult. Then, ICI is defined as: $ICI_{i\rightarrow b}=IFD(y_b|x_b)-IFD(y_b|a_i,x_b)$. If the probe's IFD decreases after adding the candidate sample, ICI is positive.

### Key Designs
1. **Diverse and Difficult Probe Set Construction**:

	- **Function**: To find a set of probes for each candidate that can truly test its value as a demonstration.
	- **Mechanism**: First, $N=32$ nearest neighbors are retrieved in the embedding space to ensure semantic relevance. Then, $K=5$ k-means clusters are formed among these neighbors to prevent probes from concentrating on a single semantic pattern. Finally, the sample with the highest complexity (via DEITA complexity scorer) is selected from each cluster to ensure probes are not too simple.
	- **Design Motivation**: Random probes are noisy, nearest-neighbor probes are redundant, and simple probes fail to demonstrate the effect of a demonstration. This three-stage retrieval ensures the influence assessment is relevant, diverse, and challenging.

2. **Weighted In-Context Influence Scoring**:

	- **Function**: To quantify the assistance a candidate sample provides as a one-shot demonstration for related tasks.
	- **Mechanism**: For each probe, the difference in IFD before and after adding the candidate is calculated (ICI). These are then aggregated using normalized cosine distance weights: $wICI(a_i)=\sum_{b\in B_i}(1-cos(f(x_i),f(x_b)))/(2|B_i|)\cdot ICI_{i\rightarrow b}$.
	- **Design Motivation**: If only average influence is considered, the model might favor near-duplicate neighbors. Distance weighting encourages the selection of instructions with transferable teaching effects that help slightly more distant but related probes.

3. **Greedy Selection with Diversity Constraints**:

	- **Function**: To prevent the final training set from being filled with high-scoring but similar samples.
	- **Mechanism**: Candidates are sorted by wICI in descending order and added greedily. A sample is accepted only if its cosine similarity to any sample already in the selected set is less than a threshold $\tau=0.9$, until the budget $k$ is met. The selected subset is used directly for standard SFT without extra weighting.
	- **Design Motivation**: High-influence samples may cluster in a few task patterns. Fine-tuning data needs to cover various instruction structures to ensure balanced performance across different benchmarks.

### Loss & Training
The selection phase uses IFD, ICI, and wICI for scoring without backpropagation. The training phase is standard supervised fine-tuning. Experiments use LlamaFactory for full-parameter fine-tuning of Llama3.1-8B and Mistral-7B-v0.3, with DeepSpeed ZeRO-3, bf16, sequence length of 2048, 3 epochs, AdamW optimizer with a learning rate of $1\times10^{-5}$, and a total batch size of 64.

## Key Experimental Results

### Main Results
Main experiments were conducted on Alpaca-GPT4 and WizardLM, with all methods selecting 10% of the data. Pairwise evaluation used GPT-4o-mini as a judge to compare the subset-tuned models against the full-data baseline; scores > 1 indicate outperforming the full-data baseline.

| Dataset | Method | Llama3.1-8B | Mistral-7B-v0.3 |
|--------|------|-------------|-----------------|
| Alpaca-GPT4 | Full | 1.000 | 1.000 |
| Alpaca-GPT4 | IFD | 1.198 | 1.248 |
| Alpaca-GPT4 | DEITA | 1.076 | 1.099 |
| Alpaca-GPT4 | NUGGETS | 1.133 | 1.201 |
| Alpaca-GPT4 | SelectIT | 1.146 | 1.227 |
| **Ours** | **wICI** | **1.215** | **1.261** |
| WizardLM | Full | 1.000 | 1.000 |
| WizardLM | IFD | 1.186 | 1.294 |
| WizardLM | DEITA | 1.114 | 1.140 |
| WizardLM | NUGGETS | 1.133 | 1.249 |
| WizardLM | SelectIT | 1.176 | 1.281 |
| **Ours** | **wICI** | 1.169 | **1.308** |

It is observed that 10% high-quality data often outperforms the full dataset, indicating significant redundancy and noise in the original instruction corpora. The proposed method performs best on both models for Alpaca-GPT4, and best for Mistral on WizardLM, while Llama3.1-8B is slightly below IFD but still stronger than the full data.

| Model / Data | Method | ARC-C | HellaSwag | MMLU | GSM8K | MT-Bench | AlpacaEval LC |
|-------------|------|-------|-----------|------|-------|----------|---------------|
| Llama3.1 / Alpaca-GPT4 | Full | 52.99 | 79.78 | 61.81 | 47.46 | 4.30 | 13.19 |
| Llama3.1 / Alpaca-GPT4 | Ours | 58.98 | 81.52 | 63.45 | 55.17 | 4.88 | 14.42 |
| Llama3.1 / WizardLM | Full | 54.61 | 78.36 | 61.32 | 55.42 | 4.75 | 14.75 |
| Llama3.1 / WizardLM | Ours | 57.79 | 81.02 | 64.90 | 52.84 | 5.28 | 13.13 |
| Mistral / Alpaca-GPT4 | Full | 44.03 | 73.01 | 51.40 | 18.73 | 3.80 | 13.19 |
| Mistral / Alpaca-GPT4 | Ours | 49.43 | 81.14 | 54.73 | 28.53 | 4.18 | 11.35 |
| Mistral / WizardLM | Full | 46.25 | 73.57 | 51.15 | 32.37 | 3.97 | 10.77 |
| Mistral / WizardLM | Ours | 51.27 | 78.51 | 56.31 | 29.44 | 4.40 | 11.36 |

### Ablation Study
Ablations focus on two diversity modules: w/o DA (removing semantic clustering in probe construction) and w/o DS (removing the cosine-similarity diversity constraint during final selection).

| Dataset | Configuration | Llama3.1-8B | Mistral-7B-v0.3 | Note |
|--------|------|-------------|-----------------|------|
| Alpaca-GPT4 | w/o DA | 1.140 | 1.181 | Probes not diverse enough, influence estimation narrows |
| Alpaca-GPT4 | w/o DS | 1.155 | 1.198 | Training set prone to redundant clusters |
| Alpaca-GPT4 | Ours | 1.215 | 1.261 | Both diversity layers preserved |
| WizardLM | w/o DA | 1.132 | 1.204 | Still better than Full, but lower than complete method |
| WizardLM | w/o DS | 1.154 | 1.239 | Demonstration quality is useful, but coverage is insufficient |
| WizardLM | Ours | 1.169 | 1.308 | Complete method is the most stable |

The authors also analyzed whether "difficult samples" and "high-ICI samples" are consistent, showing only partial overlap.

| Dataset | Top 10% overlap | Top 30% overlap | Top 50% overlap | Spearman |
|--------|------------------|------------------|------------------|----------|
| Alpaca-GPT4 | 0.1006 | 0.3874 | 0.6476 | 0.3947 |
| WizardLM | 0.1442 | 0.3650 | 0.5942 | 0.2568 |

### Key Findings
- Difficult samples are not necessarily good demonstrations. The overlap between Top 10% IFD and Top 10% ICI is only 10%-14%, indicating that "what the model finds hard" and "what can teach the model related tasks" are different signals.
- Good ICL demonstrations indeed translate to good instruction-tuning data. Even without diversity modules, wICI variants generally outperform the full-data baseline; adding probe diversity and selection diversity yields the best results.
- Data selection assists knowledge- and quality-based benchmarks more significantly than strict instruction-following benchmarks like IFEval. In the appendix, full data often performs best on IFEval, suggesting that format following may rely more on coverage scale.
- Medical domain transfer experiments show cross-domain capability. When training with 30% MedQuAD, Ours on Mistral achieved 37.05, 39.54, and 50.00 on MedMCQA, MedQA, and MMLU-med respectively, generally outperforming random selection and matching or exceeding full data on some metrics.

## Highlights & Insights
- The paper shifts the perspective of data selection from "sample quality" to "sample teaching capability for related samples." This is an insightful angle as fine-tuning essentially requires transferable training signals rather than isolated complex riddles.
- The three-stage probe set construction is solid: relevance, diversity, and complexity each address a source of bias, avoiding the inefficiency and mismatch of NUGGETS-style fixed anchor sets.
- Using semantic distance weighting in wICI is clever. It does not reward samples that only help near-neighbors (duplicates) but rewards demonstrations that generalize to slightly distant semantic regions.
- The results serve as a reminder that there is no single universal metric for data selection. IFD, DEITA, NUGGETS, and wICI favor different capabilities; as benchmark dimensions vary, the optimal method may change.

## Limitations & Future Work
- The experiments only cover 7B/8B scale models and do not evaluate larger models like Llama3-70B or larger corpora like Tulu3. Whether wICI maintains marginal gains on larger models requires further verification.
- The method focuses on supervised instruction tuning and has not been tested on DPO, PPO, or other preference optimization stages. Whether ICL influence predicts sample value for preference optimization remains an open question.
- Each sample requires approximately 16 forward passes. While much lower than the 2,000 passes of NUGGETS, this still presents cost pressures for selection on million-scale datasets.
- The effectiveness of wICI depends on the quality of embedding neighbors, complexity scorers, and IFD estimation. If embeddings are insensitive to domain semantics, the probe set might deviate from truly related tasks.

## Related Work & Insights
- **vs IFD / Superfiltering**: IFD focuses on the sample's own difficulty. This paper proves that difficulty and teaching influence are only moderately correlated, hence screening by difficulty alone misses samples with true transfer value.
- **vs DEITA**: DEITA ranks using complexity, quality, and diversity rewards. This paper borrows the complexity scorer, but complexity is used to select probes rather than directly defining data value.
- **vs NUGGETS**: NUGGETS is the most similar to this work in treating instructions as one-shot demonstrations. The difference is that NUGGETS uses fixed global anchors and coarser scoring, whereas wICI uses local semantically related probes, improvement magnitude, and distance weighting, while being more computationally efficient.
- **vs SelectIT**: SelectIT relies on uncertainty and multi-round self-reflection. This paper does not require a teacher LLM or complex multi-prompt evaluation, instead defining influence by the change in IFD.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explaining instruction-tuning data quality through ICL influence is a clear angle with significant progress over NUGGETS.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main experiments, ablation, difficulty consistency, budget, and medical transfer; however, large models and preference optimization are missing.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological formulas and research questions are well-organized; despite many tables, the narrative thread is clear.
- Value: ⭐⭐⭐⭐☆ Highly practical for low-budget SFT data filtering and provides an operational metric for the "ICL vs. Fine-tuning" relationship.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[ACL 2026\] SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](sftmix_elevating_language_model_instruction_tuning_with_mixup_recipe.md)
- [\[ICML 2026\] GIST: Targeted Data Selection for Instruction Tuning via Gradient Subspace Projection](../../ICML2026/llm_alignment/gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)

</div>

<!-- RELATED:END -->
