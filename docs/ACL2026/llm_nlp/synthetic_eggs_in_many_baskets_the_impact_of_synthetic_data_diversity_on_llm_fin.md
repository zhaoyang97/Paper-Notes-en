---
title: >-
  [Paper Note] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning
description: >-
  [ACL 2026 Findings][LLM/NLP][Synthetic Data Diversity] This paper systematically compares the effects of single-source, multi-source, and human-generated data for Supervised Fine-Tuning (SFT) on Llama models. It finds th…
tags:
  - "ACL 2026 Findings"
  - "LLM/NLP"
  - "Synthetic Data Diversity"
  - "Model Collapse"
  - "LoRA Fine-tuning"
  - "Jailbreak Robustness"
  - "Self-preference bias"
date: 2026-05-08
content_hash: 8e4aa29d3a32a177
---

# Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning

**Conference**: ACL 2026 Findings  
**arXiv**: [2511.01490](https://arxiv.org/abs/2511.01490)  
**Code**: https://github.com/maxschaffelder/synthetic_data_diversity  
**Area**: LLM / Synthetic Data / Model Safety  
**Keywords**: Synthetic Data Diversity, Model Collapse, LoRA Fine-tuning, Jailbreak Robustness, Self-preference bias

## TL;DR
This paper systematically compares the effects of single-source, multi-source, and human-generated data for Supervised Fine-Tuning (SFT) on Llama models. It finds that multi-source synthetic data alleviates distributional collapse and self-preference, yet synthetic data may weaken safety guardrails while maintaining output quality, with risks varying in complex ways based on source model scale and mixing strategy.

## Background & Motivation
**Background**: As high-quality human-generated text becomes increasingly scarce, synthetic data has been integrated into pre-training, instruction fine-tuning, and alignment pipelines. many studies focus on whether synthetic data improves benchmark scores, but analyses of how it alters model output distribution, safety robustness, and LLM-as-Judge bias remain insufficient.

**Limitations of Prior Work**: Training on synthetic data can lead to "model collapse," where model outputs increasingly resemble existing models, resulting in decreased lexical and syntactic diversity and a weakened ability to model human text. Simultaneously, fine-tuning on seemingly harmless data can compromise original refusal and safety policies; if generated harmful responses remain fluent and actionable, the risk is higher than with low-quality outputs.

**Key Challenge**: While synthetic data is a practical choice for low-cost training set expansion, it may narrow the target model's distribution and transfer the preferences and safety flaws of source models. The core problem is not whether synthetic data can be used, but whether it should originate from a single model or multiple models, and how the scale of source models alters the consequences.

**Goal**: The authors aim to isolate the impact of synthetic data source diversity, examining its effects on three areas: whether output distributions collapse, whether models become more vulnerable to jailbreak attacks, and whether they exhibit stronger self-preference or pro-synthetic biases when acting as judge models.

**Key Insight**: The paper selects Llama-3.1 8B and 70B as target models and uses Dolly-15K as the base task set to construct single-source, multi-source, and human-source fine-tuning data. This allows for a direct comparison of behavioral differences resulting from different data source compositions under identical SFT conditions.

**Core Idea**: Treat synthetic data diversity as a controllable experimental variable to observe its cascading effects on distribution, diversity, safety, and evaluation bias, rather than focusing solely on downstream task accuracy.

## Method
The methodology employs a carefully designed controlled experiment. Synthetic responses for Dolly-15K are generated using LLMs of different scales and families, followed by fine-tuning the target Llama models. Finally, three sets of evaluations are conducted to observe model behavior: distributional collapse metrics, jailbreak safety metrics, and LLM-as-Judge bias metrics.

### Overall Architecture
The input consists of the Databricks-Dolly-15K human-response dataset and three levels of source models: Small (approx. 5-15B), Medium (approx. 50-150B), and Large (400B+ or closed-source models). The output is a set of Llama-small / Llama-medium models fine-tuned via LoRA SFT, each corresponding to different data source conditions.

The experiment is divided into three main lines. The first measures the impact of synthetic data on output distribution using perplexity, inverted Self-BLEU, semantic distance, and Heaps' Law for vocabulary growth. The second measures safety using RefusalBench and RefusalBench+Jailbreak to evaluate harmfulness and response quality. The third measures judgment bias by having the fine-tuned Llama act as a judge for pairwise ranking of CNN/DailyMail summaries, calculating self-preference bias and pro-synthetic bias.

### Key Designs
1. **Parallel Comparison of Single-source, Multi-source, and Human Data**:
	- **Function**: Isolates the impact of synthetic data source diversity on post-fine-tuning model behavior.
	- **Mechanism**: The single-source condition uses one Llama source model to generate responses for the entire Dolly dataset; the multi-source condition divides the data into multiple subsets generated by different non-target models within the same scale tier, while retaining one subset generated by the target model; the human-source condition uses the original Dolly human responses as a control group. All generations use temperature=0.7, top_p=0.9, and max_output_tokens=1024.
	- **Design Motivation**: If only "synthetic vs. human" is compared, it is impossible to know if issues arise from synthetic data itself or from stylistic narrowing caused by a single source model. Isolating single-source and multi-source allows for direct observation of whether multi-model mixing alleviates distributional collapse.

2. **Combination of Distributional Collapse and Diversity Metrics**:
	- **Function**: Characterizes whether model outputs are narrowing from multiple perspectives rather than relying on task scores.
	- **Mechanism**: Perplexity is calculated for different text sources using the target model; lexical diversity is measured by $100 \cdot (1-\text{Self-BLEU})$; semantic diversity is measured by average pairwise cosine distance of SentenceBERT embeddings; and Heaps' Law $V(n)=K\cdot n^\beta$ fits vocabulary growth speed. Human text has an average lexical diversity of 89.15, higher than the 79.60 of synthetic text; human text also shows significantly faster vocabulary growth.
	- **Design Motivation**: Model collapse may not immediately appear as complete semantic repetition; it may first manifest as slower vocabulary growth and higher perplexity on human text. Multiple metrics prevent over-interpretation of perplexity alone.

3. **Downstream Behavioral Measurement of Safety and Judge Bias**:
	- **Function**: Examines whether distributional changes translate into actual risks.
	- **Mechanism**: The safety experiment combines RefusalBench with jailbreak prompts (RB+J), using Llama-3.1-70B-Instruct as a judge to score harmfulness and quality. The "danger zone" is defined as outputs where both harmfulness and quality scores $\ge 4$. The bias experiment requires fine-tuned models to compare summaries, calculating $\text{SPB}=S_{target}-\overline{S}_{other}$ and $\text{PSB}=\overline{S}_{synthetic}-S_{human}$.
	- **Design Motivation**: A model output that is harmful but nonsensical carries a different risk level than one that is harmful, high-quality, and actionable. Combining quality and harmfulness is more relevant for safety deployment than refusal rates alone.

### Loss & Training
This study utilizes LoRA Supervised Fine-Tuning without proposing a new loss function. Target Llama-3.1 8B and 70B models are trained on H100 GPUs in 16-bit precision with LoRA rank 16, $\alpha=32$, and a maximum of 1024 tokens per sample. Training runs for 3 epochs with a learning rate of 5e-5, AdamW optimizer, 3% warmup, cosine decay, and weight decay of 0.01. The focus is on keeping training settings fixed to ensure data source is the primary variable.

## Key Experimental Results

### Main Results
The first set of results shows that multi-source synthetic data generally alleviates distributional collapse. In the table, perplexity represents the mean value on the Dolly test set, with arrows indicating significant changes relative to the vanilla baseline.

| Target Model | Fine-tuning Data | Source Model Scale | Perplexity | Interpretation |
|--------|---------|----------|------------|------|
| Llama-small | Vanilla | - | 1.30 | Un-tuned baseline |
| Llama-small | Single-source | Small | 1.26 | Narrower distribution, shifts towards internal style |
| Llama-small | Multi-source | Small | 1.38 | Higher than single-source, retains more distributional width |
| Llama-small | Human-source | - | 2.68 | Closest to human distribution, but deviates most from its own output |
| Llama-medium | Vanilla | - | 1.20 | Un-tuned baseline |
| Llama-medium | Single-source | Medium | 1.15 | Medium single-source SFT significantly reduces perplexity |
| Llama-medium | Multi-source | Large | 1.42 | Multi-source large model data broadens output distribution |
| Llama-medium | Human-source | - | 2.41 | Human data still results in the largest distributional shift |

### Ablation Study
The second set of results comes from safety and judge bias analysis. Rather than traditional module ablation, this swaps data source conditions to observe changes in downstream behavior.

| Configuration / Metric | Key Figure | Description |
|------|---------|------|
| RB+J danger zone / Llama-small | 39.4% | Significant outputs have both high harmfulness and high quality under jailbreaks |
| Llama-small single-source small | 36.3% in danger zone | Small model single-source synthetic data is prone to high-risk outputs |
| Llama-medium single-source small | 44.2% in danger zone | 70B target models can also have safety weakened by small single-source models |
| Llama-small SPB vanilla | 0.258 | Original judge strongly prefers its own summaries |
| Llama-small SPB human-source | -0.013 | Human data nearly eliminates self-preference |
| Llama-small SPB multi-source | 0.159 | Multi-source synthetic data reduces bias better than single-source (0.193) |
| Llama-small PSB vanilla | 0.558 | Original judge significantly prefers synthetic summaries |
| Llama-small PSB human-source | -0.013 | Human fine-tuning nearly eliminates pro-synthetic bias |

### Key Findings
- Human responses average 78.5 tokens, whereas synthetic responses average 243.2, indicating that synthetic data provides more than just "more text"—it carries a verbose model style.
- Human lexical diversity is 89.15 compared to 79.60 for synthetic data; the semantic diversity gap is smaller (0.9713 vs. 0.9507), suggesting narrowing is more pronounced at the lexical and stylistic levels.
- Multi-source synthetic data is more human-like in distribution and reduces target model perplexity on human test sets; however, regarding safety, higher diversity does not automatically ensure higher safety. Larger source models in a multi-source mix can introduce conflicting safety strategies.
- Fine-tuning generally reduces self-preference and pro-synthetic bias, with human data being most effective, followed by multi-source synthetic data, and single-source being the weakest.

## Highlights & Insights
- The paper transforms the intuitive concept of "synthetic data diversity" into an experimental variable. The comparison between single-source and multi-source is crucial for demonstrating that "synthetic eggs in different baskets" indeed shifts the degree of collapse.
- The safety analysis provides deep insight: synthetic fine-tuning may compromise refusal strategies while maintaining fluency and actionability, which is more dangerous than low-quality harmful outputs. The "danger zone" definition is simple yet captures the actual risk in safety assessments.
- This work serves as a reminder that synthetic data quality depends not only on answer accuracy but also on the distribution, preferences, and safety strategies it injects into the target model. In the open-source ecosystem, the source of fine-tuning data may become a new supply chain risk.
- Significant for LLM-as-Judge: Judge preferences are reshaped by the fine-tuning source. Using the same type of synthetic data to fine-tuned a judge model may lead to misidentifying stylistic preferences as quality advantages.

## Limitations & Future Work
- Target models only include Llama-3.1 8B and 70B; conclusions may not directly transfer to Qwen, Mistral, Gemma, or larger 405B models.
- The fine-tuning method is limited to LoRA SFT, without testing full-parameter fine-tuning, DPO, or RLHF/RLAIF; different optimization methods might change the intensity of synthetic data's impact.
- Dolly-15K consists of single-turn English Q&A and cannot account for the consequences of synthetic data diversity in multi-turn dialogues, tool use, or long-context agent scenarios.
- Harmfulness and quality metrics rely on LLM-as-Judge; while scalable, judge bias itself is a concern of this paper, necessitating larger-scale human verification in the future.

## Related Work & Insights
- **vs. Model Collapse studies**: While works like Shumailov et al. demonstrate that recursive training leads to distribution degradation, this paper investigates whether multi-source data alleviates this and extends the analysis to 70B models.
- **vs. Performance-improvement synthetic data works**: Unlike Chen et al. who found diversity improves benchmark performance, this work focuses on output distribution, safety, and judge bias.
- **vs. LLM-as-Judge bias research**: Works by Panickssery, Xu, and Wataoka studied self-preference; this paper links such bias to fine-tuning data sources, proving that judge bias is a malleable property reshaped by post-training data.
- **Insights for Data Governance**: Synthetic data pipelines must record source models, scales, sampling parameters, and mixing ratios; simply labeling data as "synthetic" is insufficient.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The focus on the linked impact of source diversity on distribution, safety, and judge bias is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Metrics have broad coverage with detailed tables, though limited to one model family and language.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with well-explained safety and bias sections; some results require consulting the appendix for full context.
- Value: ⭐⭐⭐⭐⭐ Direct relevance for synthetic data fine-tuning, open-source model releases, safety evaluation, and judge model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Valid Inference with Imperfect Synthetic Data](../../NeurIPS2025/llm_nlp/valid_inference_with_imperfect_synthetic_data.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/llm_nlp/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)

</div>

<!-- RELATED:END -->
