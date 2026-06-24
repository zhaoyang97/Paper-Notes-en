---
title: >-
  [Paper Note] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning
description: >-
  [ACL 2025][LLM (Other)][Moral reasoning] A two-stage framework is proposed: first aggregating continuous moral ratings from multiple LLMs into collective consensus probabilities using a truncated normal distribution EM algorithm, and then optimizing token-level embeddings representing ethical theories of outlier models to align them with the collective consensus, achieving coherent moral reasoning across multiple LLMs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Moral reasoning"
  - "LLM alignment"
  - "Multi-model aggregation"
  - "Truncated Normal EM"
  - "Embedding optimization"
  - "Collective consensus"
date: 2026-05-08
content_hash: 04117df8f9e59e39
---

# Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning

**Conference**: ACL 2025  
**arXiv**: [2506.14625](https://arxiv.org/abs/2506.14625)  
**Authors**: Chenchen Yuan, Zheyu Zhang, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci (TU Munich)  
**Code**: [GitHub](https://github.com/yuanchencn/Collective-Moral-Reasoning)  
**Area**: LLM/NLP  
**Keywords**: Moral reasoning, LLM alignment, Multi-model aggregation, Truncated Normal EM, Embedding optimization, Collective consensus  

## TL;DR

A two-stage framework is proposed: first aggregating continuous moral ratings from multiple LLMs into collective consensus probabilities using a truncated normal distribution EM algorithm, and then optimizing token-level embeddings representing ethical theories of outlier models to align them with the collective consensus, achieving coherent moral reasoning across multiple LLMs.

## Background & Motivation

### Background
LLMs have demonstrated certain capabilities in moral reasoning; however, when facing complex social moral dilemmas, the judgments provided by different models often vary significantly. Existing alignment paradigms (such as Constitutional AI and RLHF) primarily focus on calibrating a single model, failing to address scenarios where multiple LLMs need to converge to a unified moral understanding.

### Limitations of Prior Work
- **Limitations of Binary Labels**: Traditional methods simplify moral judgments into binary "moral/immoral" classifications, failing to depict the continuous spectrum of moral acceptability—where answers to many dilemmas lie in a gray area.
- **Single-Model Perspective Bias**: Relying solely on a single model or views from highly narrow sources is prone to introducing systemic biases or incomplete moral representations.
- **Inapplicability of Classic Aggregation Methods**: Crowdsourcing aggregation methods such as Dawid-Skene are designed based on discrete labels and cannot naturally extend to continuous moral ratings in $[0,1]$.
- **Lack of Cross-Model Consensus Mechanisms**: There is no mature method to aggregate continuous moral judgments from multiple LLMs, and to automatically identify and rectify models that deviate from the consensus.

### Design Motivation
To design a complete framework capable of both aggregating continuous moral judgments from multiple LLMs to form a collective consensus and target-correcting models that deviate from the consensus, pursuing "coherence" rather than "correctness" in moral reasoning.

## Method

### Overall Architecture
The framework consists of two core stages:
1. **Probabilistic Aggregation Stage**: Multiple LLMs provide continuous ratings in $[0,1]$ for each moral scenario-theory pair, which are aggregated into a collective consensus probability $\gamma_{j,i}$ using a truncated normal EM algorithm.
2. **Embedding Optimization Stage**: Models that deviate significantly from the consensus, along with their weak moral theory dimensions, are identified, and only the token embeddings corresponding to that theory are adjusted to align with the consensus.

### Key Design 1: Truncated Normal EM Aggregation
The rating $a_{m,j,i} \in [0,1]$ of each model $m$ for scenario $i$ under theory $j$ is modeled as a truncated normal distribution:

$$a_{m,j,i} \sim \text{TND}(\mu_{\phi_{j,i}}(m), \sigma^2_{\phi_{j,i}}(m), 0, 1)$$

where $\phi_{j,i} \in \{0, 1\}$ is the latent moral acceptability label. Each model is characterized by four reliability parameters: $\mu_1(m)$ and $\sigma_1(m)$ (mean and variance when the latent class is positive/morally acceptable), and $\mu_0(m)$ and $\sigma_0(m)$ (mean and variance when the latent class is negative/immoral).

The **E-step** computes the posterior probability:

$$\gamma_{j,i} = \frac{P(\phi_{j,i}=1)\prod_m f_{tn}^{(\phi_{j,i}=1)}(m)}{\sum_{\phi \in \{0,1\}} P(\phi)\prod_m f_{tn}^{(\phi)}(m)}$$

The **M-step** updates the model reliability parameters weighted by the posterior probabilities. Upon iterative convergence, $\gamma_{j,i}$ is obtained as the collective consensus probability.

Compared to simple averaging or GMM, this method naturally handles bounded data in $[0,1]$, explicitly models annotator reliability, and is robust to outliers.

### Key Design 2: Moral Theory Token Embedding Optimization
For models that deviate severely from the consensus (e.g., in a theory dimension where the model achieves a low F1 score), only the $N_t$ token embeddings corresponding to that theory are fine-tuned (e.g., the three tokens "_de", "ont", "ology" corresponding to "deontology").

The loss function consists of two parts:
- **JS Divergence Loss**: Aligns the model's predicted moral acceptability distribution with the consensus distribution: $\text{loss}_{JS} = \text{JS}(P^{\text{pre}}_{\tilde{j}}, P^{\text{tgt}}_{\tilde{j}})$
- **Cosine Distance Regularization**: Constraints the embeddings from drifting too far from the original semantics: $\text{loss}_{CS} = \frac{1}{N_t}\sum_{k=1}^{N_t}\text{cos-dist}(e^{\text{ud}}_k, e^{\text{og}}_k)$

During training, all layers of the model are frozen, and only the target theory token embeddings and the parameters of a newly introduced feed-forward layer are updated. This highly localized intervention strategy ensures that the general capabilities of the model are preserved.

### Key Design 3: Diagnostic Value of Alignment Failure
An ingenious aspect of the framework's design is that if alignment does not improve even after embedding optimization, this is in itself a valuable signal—potentially indicating that the collective consensus itself is of insufficient quality (e.g., the participating models possess weak overall moral reasoning capabilities), or that there are deeper issues in the model's understanding of that theory. Experiments on four Llama variants precisely substantiate this point.

## Key Experimental Results

### Experiment 1: Four-Model Base Setup (Llama2-13B, GPT-3.5, GPT-4omini, Claude)

| Model | $\mu_1$ | $\sigma_1$ | $\mu_0$ | $\sigma_0$ |
|------|---------|-----------|---------|-----------|
| GPT-4omini | 0.658 | 0.129 | 0.418 | 0.140 |
| Claude | 0.571 | 0.143 | 0.373 | 0.127 |
| GPT-3.5 | 0.546 | 0.147 | 0.274 | 0.159 |
| Llama2-13B | 0.529 | 0.158 | 0.401 | 0.135 |
| Llama2-13B* | 0.552 | 0.154 | 0.420 | 0.138 |

GPT-4omini exhibits the highest reliability (highest $\mu_1$, lowest $\sigma_1$), whereas Llama2-13B is the weakest. After optimization, the $\mu_1$ of Llama2-13B rises from 0.529 to 0.552.

### Experiment 2: F1 Alignment Scores (Four-Model Setup)

| Model | Justice | Virtue | Deontology | Utilitarianism | Commonsense |
|------|---------|--------|-----------|---------------|-------------|
| GPT-4omini | 88.73 | 83.01 | 78.57 | 78.02 | 81.81 |
| Claude | 75.78 | 67.56 | 74.52 | 78.20 | 60.40 |
| GPT-3.5 | 74.05 | 77.13 | 56.49 | 65.86 | 68.29 |
| Llama2-13B (Before → After) | 75.25 | 63.37 | 37.68→**58.96** | 41.55→**49.76** | 45.06 |

Llama2-13B improves by **+21.28%** on Deontology and **+8.21%** on Utilitarianism, representing a highly significant improvement.

### Experiment 3: Specific Experiment on Four Llama Variants

| Model | Deontology (Before/After) | Utilitarianism (Before/After) |
|------|-------------------|----------------------|
| Llama3-8B | 69.24 / 69.51 | 78.87 / 80.04 |
| Llama3-3B | 41.86 / 41.59 | 61.06 / 60.62 |
| Llama2-13B | 56.82 / 56.96 | 48.32 / 48.21 |
| Llama2-7B | 39.82 / **37.89**↓ | 43.08 / **39.05**↓ |

When the participating models for aggregation suffer from overall weak moral reasoning capabilities, the consensus signal itself becomes unreliable, and embedding optimization instead leads to performance degradation. This validates the sensitivity of the framework to cognitive uncertainty.

## Key Findings

- **Inter-Theory Correlation**: Justice and Virtue theories show the highest correlation (Pearson $\approx$ 0.83), while Deontology and Utilitarianism exhibit the lowest correlation ($\approx$ 0.55), which aligns with the classic tension between the two in ethical philosophy.
- **Robustness Validation**: Upon introducing a random 0/1 annotator (Random01), the simple averaging method is severely disrupted, whereas the truncated normal EM method preserves the consensus among the four base models almost unaffected.
- **Semantic Preservation of Embedding Optimization**: t-SNE visualization shows that the optimized theory tokens still form tight clusters with the original tokens and semantically related tokens, demonstrating that semantic integrity is preserved during the optimization process.

## Highlights & Insights

- **Continuous Ratings over Binary Labels**: Modeling moral judgments as continuous probabilities in $[0,1]$ instead of binary classifications better captures the ambiguity and gradations of moral dilemmas.
- **Natural Reliability-Weighted Mechanism**: Models with smaller variance and well-calibrated means hold greater weight in the consensus, eliminating the need for manual model weight assignment.
- **Extremely Precise Intervention**: Modifying the embeddings of merely 3–4 tokens significantly improves alignment on specific theoretical dimensions, achieving extreme parameter efficiency.
- **Failure as Information**: The framework does not force a consensus where none exists—when there is wide disagreement within the model ensemble, optimization failure serves as a diagnostic signal.
- **Philosophical Stance of "Coherence over Correctness"**: Recognizing that there is no objective correct answer to moral dilemmas, the framework pursues a reasonable reference consensus rather than normative truths.

## Limitations & Future Work

- **Limited Model Diversity**: Only 4–5 LLMs were tested, without covering architecture families like PaLM or T5, meaning generalizability remains to be verified.
- **Insufficient Evaluation of Side Effects of Local Interventions**: The impact of embedding optimization on out-of-domain tasks lacks systematic testing.
- **Circularity in Consensus Quality Evaluation**: In the absence of a ground truth, measuring the "degree of alignment" using F1 scores is inherently self-referential.
- **Neglect of Cultural Differences**: The framework treats consensus as a unified metric, failing to account for reasonable disagreements in moral judgments across diverse cultural backgrounds.
- **High Computational Overhead**: Annotating all samples for Llama2-13B takes $N \times 3$ minutes, and embedding optimization requires 57GB of VRAM and about 4 hours per epoch.
- **Testing Limited to Five Western Ethical Theories**: Justice, virtue, deontology, utilitarianism, and commonsense ethics are evaluated, without covering diverse frameworks such as care ethics or Confucian ethics.

## Related Work & Insights

- **Dawid & Skene (1979)**: Classic multi-annotator aggregation method; this paper extends it from discrete labels to continuous ratings and introduces the truncated normal distribution.
- **ROME/MEMIT**: Knowledge editing methods that correct factual errors by modifying internal model weights; this work borrows the idea of localized editing but focuses specifically on moral theory tokens.
- **ClarifyDelphi (Pyatkin et al. 2023)**: Refines moral judgments through clarifying questions, which is complementary to this paper's pathway of aggregating multi-model perspectives.
- **ETHICS (Hendrycks et al. 2021)**: Provides five ethical theory frameworks; this work builds on them by requiring LLMs to output fine-grained continuous ratings.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of truncated normal EM aggregation for continuous moral ratings and token-level embedding optimization for theories is novel.
- Experimental Thoroughness: ⭐⭐⭐ — The 42K dataset size is substantial, but the variety of models is limited and it lacks human evaluation benchmarks.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition, complete methodological derivation, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐ — Provides a systematic framework for multi-LLM moral alignment; the concept of "failure as diagnostics" is highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[ACL 2025\] Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)
- [\[ACL 2025\] Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](moral_values_western.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
