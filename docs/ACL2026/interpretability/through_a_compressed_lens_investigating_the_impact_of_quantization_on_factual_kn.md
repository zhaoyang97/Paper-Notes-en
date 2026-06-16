---
title: >-
  [Paper Note] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall
description: >-
  [ACL 2026][Interpretability][Quantization] This paper systematically evaluates the impact of weight quantization (e.g., GPTQ, AWQ, BitsAndBytes) on the factual knowledge recall of LLMs. It finds that quantization generally causes information loss and weakens knowledge retrieval, particularly harming smaller models and unsaturated relations; however, 8-bit/BitsA
tags:
  - ACL 2026
  - Interpretability
  - Quantization
  - Model Compression
date: 2026-05-08
content_hash: eea81964919f2253
---
# Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall

**Conference**: ACL 2026  
**arXiv**: [2505.13963](https://arxiv.org/abs/2505.13963)  
**Code**: No public code / To be confirmed  
**Area**: Interpretability / Model Compression / Factual Knowledge Recall  
**Keywords**: Quantization, Factual Knowledge Recall, Knowledge Neurons, Implicit Multi-hop Reasoning, Model Compression  

## TL;DR
This paper systematically evaluates the impact of weight quantization (e.g., GPTQ, AWQ, BitsAndBytes) on the factual knowledge recall of LLMs. It finds that quantization generally causes information loss and weakens knowledge retrieval, particularly harming smaller models and unsaturated relations; however, 8-bit/BitsAndBytes tend to preserve capabilities well, and some quantizations even enhance multi-hop factual recall.

## Background & Motivation
**Background**: Quantization is one of the most commonly used compression techniques for LLM deployment. By compressing weights or activations from high-precision floating point to 8-bit, 4-bit, or lower, models can save VRAM, accelerate inference, and be deployed more easily on limited hardware.

**Limitations of Prior Work**: Existing research has investigated the impact of quantization on multilingualism, bias, fairness, calibration, alignment, and in-context learning, but analysis of factual knowledge recall remains insufficient. Factual knowledge recall is not merely downstream accuracy; it concerns whether a model can retrieve entities, relations, and composite facts from its parametric memory, serving as a fundamental capability for QA, reasoning, and knowledge-intensive tasks.

**Key Challenge**: Quantization may seem like a simple reduction in numerical precision, but the factual knowledge of LLMs might be stored in dispersed neurons, interlayer representations, and implicit reasoning paths. If the contribution scores of certain critical neurons are suppressed, specific factual retrieval chains might be damaged even if surface perplexity or general benchmarks do not show significant degradation.

**Goal**: The authors aim to answer three questions: how much factual knowledge the model forgets due to quantization; which internal layers and neurons this loss occurs in; and which step quantization disrupts in scenarios requiring the recall of a bridge entity followed by two-hop reasoning.

**Key Insight**: The paper does not rely solely on black-box performance comparisons. Instead, it decomposes factual knowledge recall into two complementary perspectives: first, one-hop factual memory and knowledge neuron attribution on LRE; and second, latent multi-hop reasoning on TwoHop-Fact, observing bridge entity recall, output distribution consistency, and final answer accuracy.

**Core Idea**: Use a three-layer evidence framework—"capability metrics + interpretability attribution + multi-hop internal paths"—to determine whether quantization truly damages the factual knowledge recall of LLMs, rather than merely looking at aggregate post-compression accuracy.

## Method
The paper investigates post-training weight-only quantization (PTQ), neither retraining the model nor introducing new compression algorithms. It uses the full-precision model as a reference and places models with different quantization methods and bit-widths into the same diagnostic pipeline for factual knowledge recall.

### Overall Architecture
The experiments follow two main tracks. The first is knowledge memorization analysis: using one-hop factual queries on the LRE dataset to test whether the model can directly recall object entities (e.g., generating the correct object given a subject-relation); subsequently, knowledge neuron attribution methods are used to track which neurons contribute most to the log-probability of the correct answer token and observe how these contributions change after quantization.

The second track is latent multi-hop reasoning analysis: constructing two-hop factual chains on TwoHop-Fact, such as $e_1 \xrightarrow{r_1} e_2$ and $e_2 \xrightarrow{r_2} e_3$. The model must recall the bridge entity $e_2$ and follow the second relation to obtain $e_3$. The authors compare the differences between full-precision and quantized models regarding accuracy for $r_1(e_1)$, $r_2(e_2)$, and $r_2(r_1(e_1))$, combined with internal representation indicators like EntRec and CnstScore to analyze the impact of quantization.

Evaluated models include Llama3-8B, Qwen2.5-7B, and Qwen2.5-14B; quantization methods include 4-bit/8-bit versions of GPTQ, AWQ, and BitsAndBytes. Due to the varying availability of public checkpoints, not every model has all quantization configurations.

### Key Designs

**1. Joint Analysis of Factual Memory and Knowledge Neurons: Linking Macro Performance Drops to Specific Neuron Suppression**

Simply observing factual recall accuracy on LRE cannot distinguish whether quantization causes a slight global perturbation or directly weakens critical neurons carrying facts. The paper first calculates recall accuracy on one-hop queries for both full and quantized models, then computes a contribution score for each neuron—the increment provided by that neuron to the log-probability of the correct answer token. A key step is taking the lowest contribution among the top-300 feed-forward neurons in the full-precision model as a threshold $\tau$, and counting how many neurons in the quantized model still exceed $\tau$. A decrease in the number of neurons exceeding the threshold directly links "macro accuracy decline" with "internal high-contribution neuron suppression," allowing macro performance and internal representation loss to validate each other.

**2. Hierarchical Attribution to Locate Information Loss: Identifying Which Layers are Specifically Damaged by Quantization**

Quantization is usually applied uniformly across matrices or layers, but factual knowledge is not evenly distributed across all layers. The paper compares the aggregate contribution score drop of attention sublayers and feed-forward sublayers. Different architectures show significant variation: on Qwen2.5-7B, the decline in the last two layers is most significant, whereas for Llama3-8B, it falls more into the middle-to-late layers, with the final layers potentially showing a compensatory increase. This hierarchical profiling suggests which layers are more sensitive to low bits and explains why the same quantization algorithm performs inconsistently across different model families—the locations of factual storage themselves vary by architecture.

**3. Decomposition of Implicit Multi-hop Reasoning: Clarifying Whether Quantization Fails at the Bridge Entity, the Second-hop Relation, or the Final Composition**

Multi-hop failure is often vaguely attributed to "poor reasoning capability," but TwoHop-Fact decomposes it into three separately measurable segments: $r_1(e_1)$ measures the recall of the bridge entity $e_2$, $r_2(e_2)$ measures the second-hop fact, and $r_2(r_1(e_1))$ measures the complete two-hop composition. Coupled with two internal indicators—EntRec, which measures whether the hidden representation truly recalls the bridge entity, and CnstScore, which measures the consistency between the output distributions of the two-hop prompt and the corresponding one-hop prompt—it becomes clear whether quantization primarily damages first-hop factual recall or the subsequent compositional path.

### Loss & Training
Ours does not involve training new models; all experiments are conducted on public full-precision and quantized checkpoints. The quantization methods belong to the PTQ category, where GPTQ uses Hessian approximation for second-order error compensation, AWQ protects low-bit weights by handling activation outliers, and BitsAndBytes provides efficient integer quantization kernels. Experiments were conducted using A100/H100 GPUs; authors report that neuron-level and layer-level attribution can be completed within 10 hours, while LMHR experiments take approximately 30 hours on average.

## Key Experimental Results

### Main Results
The LRE one-hop factual memory results show that quantization overall reduces factual knowledge recall, but the magnitude of the drop depends on model size and quantization method. Qwen2.5-14B with GPTQ4 represents the most extreme failure case, with accuracy dropping from 73.08% to 25.20%; relatively, bib8 and GPTQ8 stay close to full precision. Llama3-8B 4-bit/8-bit quantization also sees losses ranging from 0.67 to 6.23 percentage points.

| Model | Full | bib4 | bib8 | GPTQ4 | GPTQ8 | AWQ | Main Observations |
|------|------|------|------|-------|-------|-----|----------|
| Qwen2.5-7B | 63.25 | 60.72 | 63.01 | 60.10 | 63.22 | 60.60 | 4-bit and AWQ/GPTQ4 show stable drops; 8-bit is close to full |
| Qwen2.5-14B | 73.08 | 70.33 | 73.06 | 25.20 | 73.03 | 70.61 | GPTQ4 collapses severely; bib8/GPTQ8 show almost no loss |
| Llama3-8B | 77.62 | 72.19 | 76.95 | - | 71.39 | 71.83 | Llama3 shows significant but non-catastrophic declines across methods |

Knowledge neuron analysis provides mechanistic explanations. In Qwen2.5-7B and Llama3-8B, the number of neurons exceeding the full-precision top-300 threshold decreases after quantization, and the contribution score drop in the attention/FFN layers of the final two layers of Qwen2.5-7B is particularly pronounced. The authors also find that relations not yet saturated in the full model are more likely to drop after quantization, indicating that "fragile knowledge" is more susceptible to numerical perturbation than stable, well-mastered knowledge.

| Analysis Level | Qwen2.5-7B Phenomenon | Llama3-8B Phenomenon | Implication |
|----------|-----------------|----------------|------|
| Top-300 neurons | High-contribution neurons exceeding threshold decrease | Similar decrease, but different layer distribution | Contribution of key factual neurons is suppressed |
| Layer-wise drop | Most significant drop in the last two layers | Drops in middle-to-late layers; final layer may compensate | Factual storage locations vary by architecture |
| Relation sensitivity | Unsaturated relations drop more severely | Similar trend | Weakly mastered facts are more easily disturbed |
| Method differences | GPTQ4/AWQ/bib4 more damaging; bib8/GPTQ8 more stable | Smaller differences but universal decline | Bit-width is not the only factor; the algorithm also matters |

### Ablation Study
TwoHop-Fact results demonstrate that quantization has the greatest impact on the first-hop bridge entity; the main text reports a maximum first-hop drop of 30.08%, whereas the average second-hop degradation is only 4.25%. The degradation of final two-hop accuracy is highly correlated with the ability to predict the bridge entity, with a Spearman correlation coefficient of 0.93. Notably, quantization is not always detrimental: the multi-hop indicators for Llama3-8B under GPTQ8 and AWQ are significantly higher than the full-precision baseline.

| Model / Method | $r_1(e_1)$ ↑ | $r_2(e_2)$ ↑ | $r_2(r_1(e_1))$ ↑ | Key Conclusion |
|-------------|--------------|--------------|--------------------|----------|
| Qwen2.5-7B full | 25.03 | 39.07 | 20.61 | Baseline |
| Qwen2.5-7B bib4 / bib8 | 25.01 | 39.02 | 20.61 | Almost complete preservation of two-hop performance |
| Qwen2.5-7B AWQ | 22.07 | 38.89 | 18.05 | First-hop drop drives final answer decline |
| Qwen2.5-14B full | 35.23 | 40.45 | 24.72 | Larger Qwen model baseline is stronger |
| Qwen2.5-14B bib4 / bib8 | 35.16 | 40.56 | 24.76 | Final two-hop slightly higher than full |
| Qwen2.5-14B AWQ | 24.69 | 35.61 | 21.80 | Heavy loss in the first hop |
| Llama3-8B full | 7.79 | 21.39 | 4.45 | Full model is very weak in this setting |
| Llama3-8B GPTQ8 | 23.62 | 40.73 | 20.94 | Quantization significantly improves multi-hop accuracy |
| Llama3-8B AWQ | 22.35 | 37.79 | 19.56 | Similar quantization-induced improvement |

### Key Findings
- Quantization generally causes information loss in factual knowledge, but the effect is not a linear "lower bits mean worse results." bib8/GPTQ8 are often almost lossless, and bib4 is also stable in Qwen's two-hop reasoning.
- Smaller models are more fragile. Within the Qwen2.5 family, the quantization loss for 7B is more significant; 14B is more stable with bib8/GPTQ8, but GPTQ4 fails catastrophically.
- The first-hop bridge entity is the bottleneck for multi-hop FKR. The high correlation between final two-hop accuracy and $r_1(e_1)$ suggests that many errors are not due to second-step reasoning failure but a failure to recall the intermediate entity in the first step.
- The effects of quantization are highly heterogeneous. Different models, layers, relations, and methods exhibit different patterns; Llama3-8B even shows higher EntRec/CnstScore in deep layers after quantization compared to the full model.
- BitsAndBytes is the most stable practical choice. The authors conclude that the bib series is overall superior to GPTQ4/AWQ in preserving FKR, particularly maintaining stability on Qwen2.5.

## Highlights & Insights
- The paper further refines "quantization impacts capability" into "how quantization changes the internal storage and retrieval of factual knowledge." This analysis has more diagnostic value than merely looking at task accuracy.
- The discovery of the first-hop bottleneck is practical. Many multi-hop tasks appear to be reasoning problems on the surface, but the first-step factual recall actually determines whether subsequent paths are accessible.
- The occasional improvement in FKR due to quantization is a phenomenon worth further investigation. The authors speculate it may arise from regularization effects or quantization noise; it suggests that compression does not just destroy, but may also alter internal path selection.
- The results provide a moderate but important caution for deployment: common PTQs usually do not cause FKR to collapse entirely, but in knowledge-intensive tasks, general benchmarks should not be the sole basis for judging quantization safety.

## Limitations & Future Work
- The experiments use only English datasets, failing to show whether multilingual factual knowledge remains equally robust. Quantization is known to affect multilingual capabilities, and cross-lingual FKR might be more fragile.
- The model range is limited to Llama3-8B, Qwen2.5-7B, and Qwen2.5-14B. Larger models, MoEs, and families like DeepSeek or Mistral might have different knowledge storage structures.
- The authors only compare 4-bit and 8-bit, failing to cover more aggressive compression like 1-bit or 2-bit, and did not systematically study weight-activation quantization, KV cache compression, or QAT.
- Knowledge neurons and contribution scores are interpretability hypotheses, not complete mechanistic explanations. Representations might redistribute after quantization, and a drop in top neurons might not capture all compensatory paths.
- For TwoHop-Fact, where Llama3 full was weak and quantized became stronger, more controlled experiments are needed to confirm if this is due to quantization noise, checkpoint differences, or accidental gains from the evaluation setup.

## Related Work & Insights
- **vs. Compression cost by Namburi et al.**: Early work focused on the impact of compression on parametric knowledge; Ours goes a step further by adding neuron/layer attribution and latent multi-hop reasoning to locate where the loss originates.
- **vs. Quantization interpretability by Singh and Sajjad**: The latter focuses on how quantization changes internal model behavior; Ours focuses specifically on factual knowledge recall, binding interpretability analysis to concrete knowledge tasks.
- **vs. Latent multi-hop reasoning by Yang et al.**: Yang et al. proposed methods to check whether LLMs execute multi-hop reasoning internally; Ours applies this method to compare reasoning path differences between full and quantized models.
- **Inspiration for future research**: "FKR regression testing" can be established for knowledge-intensive applications before and after quantization, specifically detecting unsaturated relations, bridge entity recall, and contribution scores in the final layers, rather than just running general QA sets.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not proposing a new quantization algorithm, but combining FKR, knowledge neurons, and multi-hop internal paths for diagnosis is a precise framing of the problem.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three model classes, three PTQ methods, two types of data, and multi-layered interpretability analysis, though the model families and language scope remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear narrative and restrained conclusions; some tables and figures are dispersed in the appendix, requiring readers to cross-reference.
- Value: ⭐⭐⭐⭐☆ Highly relevant for teams deploying knowledge-intensive LLMs with quantization, specifically cautioning against relying solely on average downstream scores.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ACL 2026\] Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective](investigating_more_explainable_and_partition-free_compositionality_estimation_fo.md)

</div>

<!-- RELATED:END -->
