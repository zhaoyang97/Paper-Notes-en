---
title: >-
  [Paper Note] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?
description: >-
  [ICLR2026][Reinforcement Learning][Reinforcement Post-Training (RPT)] Through an observational study (18 open-source RPT models) and an interventional study (single-domain GRPO training)…
tags:
  - "ICLR2026"
  - "Reinforcement Learning"
  - "Reinforcement Post-Training (RPT)"
  - "RLVR"
  - "Cross-Domain Generalization"
  - "Structured Reasoning"
  - "Unstructured Reasoning"
  - "LLM"
date: 2026-05-08
content_hash: fead83f434632ca5
---

# Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?

**Conference**: ICLR2026
**arXiv**: [2506.19733](https://arxiv.org/abs/2506.19733)
**Code**: To be confirmed
**Area**: Reinforcement Learning
**Keywords**: Reinforcement Post-Training (RPT), RLVR, Cross-Domain Generalization, Structured Reasoning, Unstructured Reasoning, LLM

## TL;DR
Through an observational study (18 open-source RPT models) and an interventional study (single-domain GRPO training), this paper systematically reveals the generalization limitations of Reinforcement Post-Training (RPT/RLVR): RPT yields substantial within-domain gains, but cross-domain generalization is inconsistent — structured domains (math ↔ code) exhibit mutual transfer, whereas gains do not generalize to unstructured domains (law/finance/medicine). This finding holds consistently across algorithms, model scales, and training steps.

## Background & Motivation

**Background**: RPT (particularly RLVR) has recently achieved remarkable progress in mathematical and code reasoning — Gemini 3 Pro reaches 100% on AIME 2025, Claude Opus 4.5 reaches 81% on SWE-bench, and GPT-5.2-Pro reaches 93.2% on GPQA Diamond. These models are typically post-trained on mixed multi-domain data.

**Core Problem**: Do the reasoning gains brought by RPT generalize broadly across domains, as pretraining does? Existing work almost exclusively evaluates RPT models within their training domains, lacking systematic cross-domain analysis.

**Research Challenges**: (1) Existing RPT models employ different algorithms, hyperparameters, and multi-domain data, making it difficult to isolate the effect of RPT itself; (2) Coverage must span both structured (math, code) and unstructured (law, finance, medicine) reasoning tasks.

**Core Design**: A two-phase "observational + interventional" research paradigm is adopted — broad evaluation of existing models to identify trends, followed by controlled experiments to establish causal relationships.

## Method

### Research Design

**Four Research Questions**:
- RQ1: Do RPT gains transfer to domains outside the training distribution?
- RQ2: How does the structural similarity of reasoning affect generalization?
- RQ3: How well does RPT generalize across sub-domains within the same domain?
- RQ4: Does generalization vary with hyperparameters (algorithm, model scale, training steps)?

### Domain Taxonomy

Reasoning tasks are categorized into three major domains:
- **Math** (structured): GSM8K, MATH-500, AIME 2024, AMC 2023
- **Code** (structured): MBPP, HumanEval, BigCodeBench, LiveCodeBench, USACO, Codeforces, Polyglot
- **Knowledge-Intensive Reasoning** (unstructured): PubMedQA, MedQA, TabFact, LegalBench, FinBench

Key definition: Structured reasoning follows deterministic logical steps and precise syntax, whereas unstructured reasoning requires flexible, context-sensitive inference, world knowledge, and handling of ambiguity.

### Observational Study

- Open-source RPT models are systematically filtered from Hugging Face: 466 → 31 → 18
- Selection criteria: publicly available RPT data, model size 1.5B–14B parameters, base model not purely pretrained
- Each model is compared against its base model across 16 benchmarks
- Coverage includes math, code, legal, financial, and medical domains

### Interventional Study

- Base model: DeepSeek-R1-Distill-Qwen-1.5B (unified)
- RPT is applied separately on three disjoint 40K-sample datasets: math, code, and knowledge-intensive reasoning
- Algorithm: GRPO (Group Relative Policy Optimization) with identical hyperparameters throughout
- Additional validation: DAPO algorithm, 2-epoch training, and Llama-3.2-3B-Instruct as the backbone

### Evaluation Metrics

- **Aggregate accuracy gain** $\Delta_{i,j}^{(\mathcal{D})}$: weighted-average pass@1 improvement
- **CMH statistical test**: Cochran-Mantel-Haenszel test, computing the common odds ratio $\hat{\theta}$; significance marked at $p < 0.05$
- Small benchmarks (AMC/AIME) are repeated 16 times and averaged; others are run once

## Key Experimental Results

### RQ1: RPT Does Not Generalize to Arbitrary Unseen Domains

| Metric | In-Domain (ID) Avg. | Out-of-Domain (OOD) Avg. |
|--------|---------------------|--------------------------|
| $\Delta$ (pass@1 %) | +2.87 | **-3.19** |
| Odds ratio $\hat{\theta}$ | 3.10 | 1.32 |

- Typical case: DeepScaleR-1.5B gains +5.1% on math but only +1.7% elsewhere (3× drop)
- Extreme case: AZR-Coder-7B (near-zero-data post-training) gains +30.12% on code but **-23.31%** OOD
- In the single-domain interventional experiments, none of the math, code, or knowledge RPT conditions yield statistically significant OOD improvements

### RQ2: Structured Domains Transfer Mutually; Cross-Type Transfer Fails

| Training Domain → Test Domain | Math | Code | Knowledge-Intensive |
|-------------------------------|------|------|---------------------|
| **Math RPT** | +2.18% | +4.77% | -0.27% |
| **Code RPT** | +15.44% | +9.49% | -0.27% |
| **Knowledge RPT** | +21.40%* | +12.16%* | Decline |

- Math → Code and Code → Math transfer are both effective bidirectionally; Math → Code transfer is stronger, reflecting math as a more foundational form of structured reasoning
- Structured → Knowledge-Intensive: no statistically significant improvement, and occasional decline
- **Counterintuitive finding**: Knowledge-Intensive → Structured domains shows significant positive transfer, suggesting that unstructured reasoning is in some sense a "superset" of structured reasoning capabilities

### RQ3: Within-Domain Sub-Domain Generalization Depends on Structural Similarity

- Within-domain generalization is consistent for structured domains (uniform gains across math sub-tasks and across code sub-tasks)
- Within-domain generalization is poor for unstructured domains: Fino1-8B (finance RPT) shows -2% on PubMedQA, -1.6% on LegalBench, and **-15.8%** on TabFact
- In the interventional study, Knowledge-RPT also exhibits overall decline on the knowledge domain, indicating a lack of shared logical templates across unstructured tasks

### RQ4: Generalization Limitations Are Consistent Across Configurations

| Base + Algorithm | $\Delta^{(ID)}$ | $\Delta^{(OOD)}$ | Gap |
|-----------------|-----------------|-------------------|-----|
| DS-Qwen-1.5B + GRPO | +3.13 | **-1.81** | 4.94 |
| Llama-3.2-3B + GRPO | +6.47 | +1.41 | 5.06 |
| DS-Qwen-1.5B + DAPO | +3.96 | **-1.27** | 5.23 |

- The same pattern holds across different RL algorithms (GRPO vs. DAPO), different base models, and additional training steps
- The ID–OOD gap increases with training steps and eventually stabilizes
- Larger model size leads ID gains to outgrow OOD gains by an additional 16.5%, exacerbating overfitting

## Highlights & Insights

- **"Observational + Interventional" experimental design paradigm**: Trends are first identified via broad evaluation of 18 models, then causal relationships are validated through controlled single-domain training — a rigorous research design
- **Counterintuitive positive transfer from unstructured to structured domains**: Knowledge RPT models exhibit statistically significant gains on math, suggesting that broad knowledge reasoning subsumes structured reasoning to some degree
- **Hierarchy of structured reasoning**: Math → Code transfer is stronger than Code → Math, reflecting mathematical reasoning as a more foundational capability
- **Even minimal training data can cause OOD harm**: AZR-Coder causes -23% OOD degradation with almost no training data, indicating that RPT itself — not overfitting per se — is the root cause of generalization failure
- **Use of CMH statistical testing**: The introduction of a standardized hypothesis testing framework, rather than relying solely on point estimates, strengthens the credibility of the conclusions

## Limitations & Future Work

- Interventional experiments use only 1.5B models; single-domain RPT experiments on larger models (7B+) are absent
- Training data for knowledge-intensive reasoning is filtered by o3-mini to remove math/code content; filtering quality may affect conclusions
- Joint SFT + RPT training scenarios are not explored; the study focuses exclusively on pure RLVR
- The number of benchmarks for unstructured domains is limited (only 5), and whether some (e.g., TabFact) truly represent "unstructured reasoning" is debatable
- Changes in internal model representations during RPT (e.g., probing analysis) are not analyzed, leaving the mechanistic explanation incomplete
- All experiments are based on open-source models; the generalization behavior of closed-source RPT models (e.g., o1, the full DeepSeek-R1) may differ

## Related Work & Insights

- **vs. DeepSeek-R1/o1 series**: These models are post-trained on multi-domain mixed data and appear to improve broadly, but because ID and OOD are interleaved, genuine generalization cannot be distinguished. This paper reveals generalization limitations through single-domain experiments
- **vs. RPT limitation studies (Yue et al., Ma et al.)**: Prior work questions RPT in terms of reasoning quality and computational efficiency; this paper is the first systematic study of RPT generalization across data domains
- **vs. pretraining generalization**: Pretraining achieves broad generalization through massive, diverse data; RPT generalization is far more limited, indicating that "reasoning learning" during post-training is fundamentally domain-specific pattern reinforcement
- **Practical implications**: RPT deployment should use training data tailored to the target domain; expecting math/code RPT to yield legal/medical reasoning capabilities is unrealistic

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of cross-domain generalization in RPT, with a well-designed experimental framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models × 16 benchmarks, dual validation via observational + interventional studies, ablation across algorithms/models/steps
- Writing Quality: ⭐⭐⭐⭐ Clear structure, RQ-driven organization, rigorous statistical testing
- Value: ⭐⭐⭐⭐⭐ An important cautionary contribution to the RPT community, exposing a widely overlooked generalization bottleneck

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/reinforcement_learning/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](../../NeurIPS2025/reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)

</div>

<!-- RELATED:END -->
