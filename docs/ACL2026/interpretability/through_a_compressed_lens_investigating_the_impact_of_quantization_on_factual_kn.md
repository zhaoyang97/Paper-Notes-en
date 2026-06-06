---
title: >-
  [Paper Note] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall
description: >-
  [ACL 2026][Interpretability][Quantization] This paper systematically evaluates the impact of weight quantization (such as GPTQ, AWQ…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Quantization"
  - "Factual Knowledge Recall"
  - "Knowledge Neurons"
  - "Implicit Multi-hop Reasoning"
  - "Model Compression"
date: 2026-05-08
content_hash: bee58c1340a2f8bf
---

# Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall

**Conference**: ACL 2026  
**arXiv**: [2505.13963](https://arxiv.org/abs/2505.13963)  
**Code**: No public code / To be confirmed  
**Area**: Interpretability / Model Compression / Factual Knowledge Recall  
**Keywords**: Quantization, Factual Knowledge Recall, Knowledge Neurons, Implicit Multi-hop Reasoning, Model Compression  

## TL;DR
This paper systematically evaluates the impact of weight quantization (such as GPTQ, AWQ, and BitsAndBytes) on the factual knowledge recall of LLMs. It finds that quantization generally leads to information loss and weakens knowledge retrieval, particularly harming smaller models and non-saturated relations. However, 8-bit/BitsAndBytes often preserve capabilities well, and in certain cases, quantization can even enhance multi-hop factual recall.

## Background & Motivation
**Background**: Quantization is one of the most commonly used compression techniques in LLM deployment. By compressing weights or activations from high-precision floating point to 8-bit, 4-bit, or lower, models can reduce memory footprint, accelerate inference, and deploy more easily on limited hardware.

**Limitations of Prior Work**: Existing research has investigated the impact of quantization on multilingualism, bias, fairness, calibration, alignment, and in-context learning, but analysis regarding factual knowledge recall remains insufficient. Factual knowledge recall is not merely downstream accuracy; it concerns whether a model can retrieve entities, relations, and composite facts from its parametric memory, serving as a fundamental capability for QA, reasoning, and knowledge-intensive tasks.

**Key Challenge**: While quantization appears to be a mere reduction in numerical precision, the factual knowledge of LLMs may be stored in scattered neurons, inter-layer representations, and implicit reasoning paths. If the contribution scores of certain key neurons are suppressed, the retrieval chain for specific facts might be damaged even if surface perplexity or general benchmarks do not show significant degradation.

**Goal**: The authors aim to answer three questions: how many facts the model forgets due to quantization; in which internal layers and neurons this loss occurs; and which step quantization disrupts in scenarios requiring the recall of a bridge entity followed by two-hop reasoning.

**Key Insight**: The paper shifts from black-box performance comparison to decomposing factual knowledge recall into two complementary perspectives: first-hop factual memory and knowledge neuron attribution on LRE, and latent multi-hop reasoning on TwoHop-Fact to observe bridge entity recall, output distribution consistency, and final answer accuracy.

**Core Idea**: Utilizing a three-layer evidence chain comprising "capability metrics + interpretability attribution + internal multi-hop paths" to determine if quantization truly damages the factual knowledge recall of LLMs, rather than solely relying on overall post-compression accuracy.

## Method
The paper investigates post-training weight-only quantization (PTQ) without retraining the model or introducing new compression algorithms. It uses the full-precision model as a reference and subjects models with different quantization methods and bit-widths to the same diagnostic pipeline for factual knowledge recall.

### Overall Architecture
The experiments follow two primary tracks. The first is knowledge memorization analysis: testing whether models can directly recall object entities in single-hop factual queries on the LRE dataset (e.g., generating the correct object given a subject-relation). Subsequently, knowledge neuron attribution methods are used to trace which neurons contribute most to the log-probability of the correct answer token and observe how these contributions change after quantization.

The second track is latent multi-hop reasoning analysis: constructing two-hop factual chains on TwoHop-Fact, such as $e_1 \xrightarrow{r_1} e_2$ and $e_2 \xrightarrow{r_2} e_3$. The model must recall the bridge entity $e_2$ and follow the second relation to obtain $e_3$. The authors compare the differences between full-precision and quantized models across three types of accuracy ($r_1(e_1)$, $r_2(e_2)$, and $r_2(r_1(e_1))$) while analyzing the impact of quantization using internal representation metrics like EntRec and CnstScore.

The evaluated models include Llama3-8B, Qwen2.5-7B, and Qwen2.5-14B; quantization methods include 4-bit and 8-bit versions of GPTQ, AWQ, and BitsAndBytes. Due to varying availability of public checkpoints, not every model has all quantization configurations.

### Key Designs
1. **Joint Analysis of Factual Memory and Knowledge Neurons**:
	- **Function**: To determine if quantization causes the model to "forget" parametric knowledge of single-hop facts and to trace the internal neuronal changes corresponding to this forgetting.
	- **Mechanism**: Factual recall accuracy is first calculated on LRE for both full and quantized models. Then, a contribution score is computed for each neuron, representing the increase in log-probability of the correct answer token attributed to that neuron. The authors set a threshold $\tau$ based on the minimum contribution of the top-300 feed-forward neurons in the full-precision model and count how many neurons in the quantized model still exceed this threshold.
	- **Design Motivation**: Accuracy alone makes it difficult to discern if quantization causes global minor perturbations or specifically weakens key neurons carrying factual knowledge. The distribution of top neurons bridges macro performance degradation and internal representation loss.

2. **Hierarchical Attribution for Locating Information Loss**:
	- **Function**: To identify which layers and sub-modules are most significantly impacted by quantization.
	- **Mechanism**: The authors compare the aggregate contribution score drop in attention sublayers versus feed-forward sublayers. On Qwen2.5-7B, the drop is most pronounced in the last two layers; on Llama3-8B, it occurs more in the middle-to-late layers, with the final layer potentially showing an increase. This suggests that different architectures have distinct hierarchical patterns for storing factual knowledge.
	- **Design Motivation**: Quantization is usually applied uniformly by matrix or layer, but factual knowledge is not distributed evenly. Hierarchical attribution can indicate which layers are more sensitive to lower bits and explain why the same quantization method performs inconsistently across different model families.

3. **Deconstruction of Implicit Multi-hop Reasoning**:
	- **Function**: To distinguish whether quantization damages the first-hop bridge entity, the second-hop factual relation, or the final composite answer.
	- **Mechanism**: In TwoHop-Fact, $r_1(e_1)$ measures bridge entity recall, $r_2(e_2)$ measures the second-hop fact, and $r_2(r_1(e_1))$ measures the complete two-hop composition. EntRec evaluates whether hidden representations recall the bridge entity, while CnstScore measures whether the output distributions of the two-hop prompt and the corresponding one-hop prompt are consistent.
	- **Design Motivation**: Failures in multi-hop reasoning are often vaguely attributed to "poor reasoning." This deconstruction clarify whether quantization primarily hurts first-hop factual recall or the subsequent compositional path.

### Loss & Training
This study did not train new models; all experiments were conducted on public full-precision and quantized checkpoints. The quantization methods belong to the PTQ category, where GPTQ uses Hessian approximation for second-order error compensation, AWQ protects low-bit weights by handling activation outliers, and BitsAndBytes provides efficient integer quantization kernels. Experiments were conducted using A100/H100 GPUs; the authors report that neuron-level and layer-level attribution can be completed within 10 hours, while LMHR experiments take approximately 30 hours on average.

## Key Experimental Results

### Main Results
LRE single-hop factual memory results show that quantization overall reduces factual knowledge recall, though the magnitude of the drop depends on model size and quantization method. Qwen2.5-14B with GPTQ4 represents the most extreme failure case, with accuracy dropping from 73.08% to 25.20%; in contrast, bib8 and GPTQ8 largely match full precision. Llama3-8B's 4-bit/8-bit quantization also shows losses ranging from 0.67 to 6.23 percentage points.

| Model | Full | bib4 | bib8 | GPTQ4 | GPTQ8 | AWQ | Main Observation |
|------|------|------|------|-------|-------|-----|----------|
| Qwen2.5-7B | 63.25 | 60.72 | 63.01 | 60.10 | 63.22 | 60.60 | 4-bit and AWQ/GPTQ4 show stable drops; 8-bit is close to full |
| Qwen2.5-14B | 73.08 | 70.33 | 73.06 | 25.20 | 73.03 | 70.61 | GPTQ4 collapses severely; bib8/GPTQ8 are nearly lossless |
| Llama3-8B | 77.62 | 72.19 | 76.95 | - | 71.39 | 71.83 | Llama3 shows significant but non-catastrophic drops across quantizations |

Knowledge neuron analysis provides a mechanistic explanation. In Qwen2.5-7B and Llama3-8B, the number of neurons exceeding the full-precision top-300 threshold decreases after quantization, and the contribution scores of attention/FFN in the last two layers of Qwen2.5-7B drop significantly. The authors also found that relations not yet saturated in the full model are more likely to drop after quantization, indicating that "fragile knowledge" is more susceptible to numerical perturbations than stable knowledge.

| Analysis Level | Qwen2.5-7B Phenomenon | Llama3-8B Phenomenon | Implication |
|----------|-----------------|----------------|------|
| Top-300 neuron | High-contribution neurons exceeding the threshold decrease | Also decrease, but layer distribution differs | Contribution of key factual neurons is suppressed |
| Layer-wise drop | Last two layers show the most significant drop | Middle-to-late layers drop; final layer may show compensatory rise | Factual storage locations vary by architecture |
| Relation sensitivity | Non-saturated relations show more severe drops | Similar trend | Weakly mastered facts are easily perturbed by quantization |
| Method Differences | GPTQ4/AWQ/bib4 more damaging; bib8/GPTQ8 more stable | Smaller differences but universal decline | Bit-width is not the only factor; algorithm matters |

### Ablation Study
TwoHop-Fact results demonstrate that quantization has the greatest impact on the first-hop bridge entity. The main text reports that the first hop can drop by up to 30.08%, while the second hop shows a mean degradation of only 4.25%. The degradation of final two-hop accuracy is highly correlated with the ability to predict the bridge entity (Spearman correlation of 0.93). Notably, quantization is not always detrimental: multi-hop metrics for Llama3-8B under GPTQ8 and AWQ were significantly higher than those of the full model.

| Model / Method | $r_1(e_1)$ ↑ | $r_2(e_2)$ ↑ | $r_2(r_1(e_1))$ ↑ | Key Conclusion |
|-------------|--------------|--------------|--------------------|----------|
| Qwen2.5-7B full | 25.03 | 39.07 | 20.61 | Baseline |
| Qwen2.5-7B bib4 / bib8 | 25.01 | 39.02 | 20.61 | Almost fully preserves two-hop performance |
| Qwen2.5-7B AWQ | 22.07 | 38.89 | 18.05 | First-hop drop drives final answer decline |
| Qwen2.5-14B full | 35.23 | 40.45 | 24.72 | Larger Qwen model has a stronger baseline |
| Qwen2.5-14B bib4 / bib8 | 35.16 | 40.56 | 24.76 | Final two-hop slightly higher than full |
| Qwen2.5-14B AWQ | 24.69 | 35.61 | 21.80 | Substantial loss in the first hop |
| Llama3-8B full | 7.79 | 21.39 | 4.45 | Full model is very weak in this setting |
| Llama3-8B GPTQ8 | 23.62 | 40.73 | 20.94 | Quantization significantly improves multi-hop accuracy |
| Llama3-8B AWQ | 22.35 | 37.79 | 19.56 | Similar quantization-induced improvement |

### Key Findings
- Quantization generally causes information loss in factual knowledge, but the drop is not linearly "worse as bits go lower." bib8/GPTQ8 are often nearly lossless, and bib4 remains stable in multi-hop reasoning for Qwen.
- Smaller models are more fragile. Within the Qwen2.5 family, the 7B model shows more obvious quantization loss; the 14B model is more stable with bib8/GPTQ8, but GPTQ4 fails catastrophically.
- The first-hop bridge entity is the bottleneck for multi-hop FKR. The high correlation between final two-hop accuracy and $r_1(e_1)$ suggests many errors stem from failing to recall the intermediate entity rather than failing the second step of reasoning.
- Quantization effects are highly heterogeneous. Different models, layers, relations, and methods exhibit distinct patterns; Llama3-8B even showed higher EntRec/CnstScore in deep layers after quantization compared to the full model.
- BitsAndBytes is the most stable practical choice in the study. The authors conclude that the bib series overall outperforms GPTQ4/AWQ in preserving FKR, particularly on Qwen2.5.

## Highlights & Insights
- The paper refines "quantization impacts capability" into "how quantization changes internal storage and retrieval of factual knowledge." This diagnostic analysis is more valuable than just looking at task accuracy.
- The discovery of the first-hop bottleneck is practical. Many multi-hop tasks appear to be reasoning problems but are actually decided by whether the first step of factual recall can successfully retrieve the intermediate path.
- The occasional improvement of FKR post-quantization is a phenomenon worth further investigation. The authors speculate this may arise from regularization effects or quantization noise, suggesting compression does not just destroy but can also alter the model's internal path selection.
- The results offer a moderate but important caution for deployment: while common PTQ methods generally do not cause FKR to collapse entirely, general benchmarks alone cannot determine if quantization is safe for knowledge-intensive tasks.

## Limitations & Future Work
- The experiments used only English datasets, so it remains unclear if multilingual factual knowledge is equally robust. Quantization is known to affect multilingual capabilities, making cross-lingual FKR potentially more fragile.
- Model scope is limited to Llama3-8B, Qwen2.5-7B, and Qwen2.5-14B. Larger models, MoE architectures, or other families like DeepSeek/Mistral may have different knowledge storage structures.
- The authors only compared 4-bit and 8-bit quantization, excluding more aggressive compression like 1-bit or 2-bit, and did not systematically study weight-activation quantization, KV cache compression, or QAT.
- Knowledge neurons and contribution scores are interpretability hypotheses and do not constitute a complete mechanistic explanation. Representations may redistribute post-quantization, and the decline of top neurons might not cover all compensatory paths.
- The phenomenon where Llama3-8B full is weak while the quantized version is stronger in TwoHop-Fact results requires more controlled experiments to confirm whether it is due to quantization noise, checkpoint differences, or an artifact of the evaluation setup.

## Related Work & Insights
- **vs. Namburi et al. on compression cost**: While early work focused on the impact of compression on parametric knowledge, this paper goes further by adding neuron/layer attribution and latent multi-hop reasoning to localize exactly where losses occur.
- **vs. Singh and Sajjad's quantization interpretability**: Where the latter focuses on how quantization changes internal model behavior, this paper centers on factual knowledge recall, tying interpretability analysis to specific knowledge tasks.
- **vs. Yang et al. on latent multi-hop reasoning**: Yang et al. proposed methods to check if LLMs perform multi-hop reasoning internally; this paper applies those methods to compare reasoning path differences between full and quantized models.
- **Insights for follow-up research**: "FKR regression tests" could be established for knowledge-intensive applications before and after quantization, specifically monitoring non-saturated relations, bridge entity recall, and contribution scores in the final layers instead of relying solely on general QA sets.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Rather than proposing new quantization algorithms, it combines FKR, knowledge neurons, and multi-hop internal paths for diagnosis, defining the problem precisely.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers three model categories, three PTQ methods, two types of data, and multiple layers of explanatory analysis, though model families and language scope remain limited.
- **Writing Quality**: ⭐⭐⭐⭐☆ The main narrative is clear and conclusions are restrained; some tables and figures are scattered in the appendix, requiring readers to cross-reference frequently.
- **Value**: ⭐⭐⭐⭐☆ Highly relevant for teams deploying knowledge-intensive LLMs with quantization, serving as a reminder not to rely solely on average downstream scores.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective](investigating_more_explainable_and_partition-free_compositionality_estimation_fo.md)
- [\[ACL 2026\] The Impact of Off-Policy Training Data on Probe Generalisation](the_impact_of_off-policy_training_data_on_probe_generalisation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)

</div>

<!-- RELATED:END -->
