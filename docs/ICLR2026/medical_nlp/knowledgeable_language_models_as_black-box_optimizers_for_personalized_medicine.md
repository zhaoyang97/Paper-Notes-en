---
title: >-
  [Paper Note] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine
description: >-
  [ICLR 2026][Medical NLP][LLM-based optimization] This paper proposes LEON (LLM-based Entropy-guided Optimization with kNowledgeable priors)…
tags:
  - "ICLR 2026"
  - "Medical NLP"
  - "LLM-based optimization"
  - "personalized medicine"
  - "black-box optimization"
  - "distribution shift"
  - "prior knowledge"
date: 2026-05-08
content_hash: 47360727d9fbdc0f
---

# Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine

**Conference**: ICLR 2026
**arXiv**: [2509.20975](https://arxiv.org/abs/2509.20975)  
**Code**: [Code](https://code.roche.com/braid/projects/leon)  
**Area**: Medical Imaging / Personalized Medicine
**Keywords**: LLM-based optimization, personalized medicine, black-box optimization, distribution shift, prior knowledge

## TL;DR

This paper proposes LEON (LLM-based Entropy-guided Optimization with kNowledgeable priors), a mathematically rigorous framework that models personalized treatment design as a constrained conditional black-box optimization problem. Through entropy constraints and an adversarial source critic, LEON guides an LLM to serve as a zero-shot optimizer that proposes personalized treatment plans without any fine-tuning.

## Background & Motivation

- **Background**: Personalized medicine aims to discover optimal treatment strategies based on a patient's genetic and environmental factors. Recent work has demonstrated the potential of LLMs as black-box optimizers in domains such as mathematics and code.
- **Limitations of Prior Work**: (1) Evaluating real treatment outcomes is extremely costly, and surrogate models (digital twins, ML models) are typically used as proxies; (2) surrogate models yield unreliable predictions under distribution shift (e.g., patients from a new hospital), producing "spurious" plans that appear promising but perform poorly in practice; (3) certain patient subpopulations are systematically underrepresented in clinical studies.
- **Key Challenge**: Naively substituting a surrogate $\hat{f}$ for the true objective $f$ leads to out-of-distribution extrapolation and poor real-world treatment outcomes. Improving the surrogate is further constrained by data availability and privacy concerns.
- **Goal**: To design personalized treatment plans for patients under distribution shift when the surrogate model is unreliable and the true objective function is inaccessible.
- **Key Insight**: Leveraging the domain prior knowledge internalized by LLMs (medical textbooks, knowledge graphs) as a complementary signal, and using constrained optimization to simultaneously control surrogate extrapolation and the entropy of LLM proposals.
- **Core Idea**: Two constraints—a Wasserstein distance constraint to limit distribution shift and an entropy constraint to promote LLM certainty—regularize LLM-driven conditional black-box optimization, with prior knowledge improving the quality of the LLM as a stochastic treatment recommendation engine.

## Method

### Overall Architecture

The LEON optimization loop proceeds as follows: (1) **Sampling**—the LLM generates a batch of treatment plans conditioned on task description, patient information, prior knowledge, and a history of proposal–score pairs; (2) **Clustering**—plans are assigned to equivalence classes; (3) **Certainty Estimation**—LLM certainty parameter $\mu$ and source critic parameter $\lambda$ are estimated; (4) **Scoring**—each plan is scored by $\mu[\hat{f}(x;z) + \lambda c^*(x)]$ and stored for subsequent prompting.

### Key Designs

#### 1. Constrained Conditional Optimization Problem

- **Function**: Models personalized medicine as a constrained conditional black-box optimization problem.
- **Mechanism**:
$$\arg\max_{q(x)} \mathbb{E}_{x \sim q(x)}[\hat{f}(x;z)] \quad \text{s.t.} \quad W_1(p_{\text{src}}, q) \leq W_0, \quad \mathcal{H}_\sim(q(x)) \leq H_0$$
  - The first constraint limits the deviation of the proposed plan distribution from the historical distribution via the 1-Wasserstein distance (implemented through an adversarial source critic $c^*$), preventing surrogate extrapolation.
  - The second constraint bounds the coarse-grained entropy of LLM proposals, encouraging higher-certainty plans.
- **Design Motivation**: The two constraints independently address surrogate unreliability and LLM stochasticity.

#### 2. Lagrangian Dual Solution

- **Function**: Derives a tractable solution to the constrained optimization problem.
- **Mechanism**:
    - **Lemma 4.2 (Intra-class Collapse)**: The optimal distribution $q^*$ concentrates within each equivalence class on the best design $x_i^* = \arg\max_{x \in [x]_i} (\hat{f}(x;z) + \lambda c^*(x))$.
    - **Lemma 4.3 (Probabilistic Sampling)**: Equivalence class probabilities satisfy $\bar{q}_i \propto \exp[\mu(\hat{f}(x_i^*;z) + \lambda c^*(x_i^*))]$.
    - The two Lagrange multipliers $\lambda$ (source critic certainty) and $\mu$ (LLM certainty) control the respective constraints.

#### 3. Dynamic Certainty Parameter Estimation

- **LLM Certainty $\mu$**: Equivalence class occupancy $\hat{q}_i$ is estimated from LLM batch samples; linear regression on $(\hat{f}(x_i^*;z) + \lambda c^*(x_i^*), \log \hat{q}_i)$ yields $\hat{\mu}$. Intuitively, high entropy (low certainty) gives $\hat{\mu} \approx 0$, reducing reward amplification; high certainty gives $\hat{\mu} > 0$, strengthening the reward signal.
- **Source Critic Parameter $\lambda$**: Updated via dual function gradient descent: $\lambda_{t+1} = \lambda_t - \eta_\lambda [W_0 - W_1(\text{estimated})]$. When proposals are in-distribution, $\lambda$ decreases to allow broader exploration; when proposals deviate, $\lambda$ increases to constrain extrapolation.

#### 4. Prior Knowledge Generation

- **Function**: Supplies domain priors to the LLM from external knowledge sources.
- **Toolset**: Medical textbook corpora, MedGemma 27B, HetioNet/PrimeKG knowledge graphs, Cellosaurus cell line data, COSMIC cancer mutation data, GDSC drug sensitivity data, and DepMap cancer cell dependency data.
- **Pipeline**: The LLM acts as a tool-calling agent that autonomously selects relevant knowledge sources and synthesizes natural-language prior knowledge statements.
- **Design Motivation**: Prior knowledge helps the LLM overcome the statistical randomness inherent in next-token generation, improving plan quality and $\mu$ values.

### Loss & Training

LEON requires no LLM training. The source critic $c^*$ is trained via the Wasserstein dual (Eq. 1), with Lipschitz constraints enforced through weight clipping.

## Key Experimental Results

### Main Results

Five real-world personalized medicine optimization tasks (distribution shift setting, 100 test patients per task):

| Method | Warfarin RMSE↓ | HIV Viral Load↓ | Breast TTNTD↑ | Lung TTNTD↑ | ADR NLL↓ | Avg. Rank |
|---|---|---|---|---|---|---|
| Human (actual treatment) | 2.68 | 4.55 | 29.65 | 21.10 | — | 8.5 |
| Gradient Ascent | 1.37 | 4.52 | 65.23 | 24.09 | 23.7 | 5.2 |
| BO-qEI | 1.36 | 4.53 | 67.05 | 27.97 | 23.2 | 3.4 |
| OPRO | 1.40 | 4.55 | 55.68 | 24.35 | 23.8 | 7.0 |
| Eureka | 1.54 | 4.58 | 63.48 | 25.10 | 21.3 | 6.8 |
| **LEON** | **1.36** | **4.50** | **72.43** | **32.71** | **12.4** | **1.2** |

### Ablation Study

- Removing prior knowledge: significant performance degradation, indicating sensitivity to knowledge quality.
- Removing the source critic constraint ($\lambda = 0$): surrogate extrapolation leads to degraded ground-truth performance.
- Removing the entropy constraint ($\mu = 0$): LLM proposals become overly diffuse, requiring more iterations to converge.
- LLM choice: gpt-4o-mini achieves the best cost-effectiveness ratio.

### Key Findings

1. LEON achieves an average rank of 1.2 across all five tasks, significantly outperforming both traditional optimization methods and other LLM-based optimizers.
2. LEON's proposed plans **outperform the treatments patients actually received** (Human baseline), demonstrating tangible clinical value.
3. The two constraints (Wasserstein + entropy) exhibit clear synergistic effects; removing either one individually degrades performance.
4. Prior knowledge is especially important under distribution shift, as it enables the LLM to compensate for the surrogate model's blind spots in OOD regions.

## Highlights & Insights

1. **Mathematical Rigor Combined with Practical Value**: The derivation from constrained optimization through Lagrangian duality to the lemmas is rigorous, while experiments on five real clinical tasks are highly compelling.
2. **Elegant Design of the Two Certainty Parameters**: $\mu$ quantifies the LLM's "degree of consensus" and $\lambda$ quantifies the "degree of in-distribution-ness" of a design; together they dynamically balance exploration and exploitation.
3. **Zero-Shot Optimization**: The LLM requires no fine-tuning and surpasses dedicated optimization algorithms purely through prompting and external knowledge.
4. **Privacy Preservation**: The source critic $c^*$ requires only treatment design data $\mathcal{D}_{\text{src}} \subseteq \mathcal{X}$, with no patient information needed.

## Limitations & Future Work

1. **Sensitivity to Prior Knowledge**: Incorrect or outdated domain knowledge propagates directly into optimization outputs, necessitating knowledge quality assurance mechanisms.
2. **Limitations of Simulated Benchmarks**: Although real data are used, the evaluation framework still relies on a learned ground-truth function $f$, which cannot fully capture the heterogeneity of real patient responses.
3. **Equivalence Class Definition**: Equivalence classes are defined via k-means clustering; alternative clustering approaches may affect results.
4. **Computational Cost**: Each patient requires 2,048 surrogate model queries and substantial LLM API calls.
5. **Fairness Not Fully Validated**: Although gender and racial fairness are discussed in the appendix, social biases in LLMs may influence treatment recommendations.

## Related Work & Insights

- **LLM Optimization**: OPRO (Yang et al., 2024a) optimizes via prompting but lacks constraints; Eureka (Ma et al., 2024b) adds reflection but provides no distribution control.
- **Optimization under Distribution Shift**: Trabucco et al. (2021) assume a controllable surrogate in conservative model-based optimization, whereas LEON handles black-box surrogates.
- **Discriminative Models in Optimization**: The Wasserstein distance constraint draws from the MBO and biological sequence design literature (Yao et al., 2024, 2025b).
- **Insights**: The paradigm of LLMs as conditional optimizers can generalize to other domains requiring personalized decision-making (e.g., education, finance); the key challenge lies in injecting domain knowledge while calibrating confidence.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ Unifies LLM optimization, distribution-shift robustness, and prior knowledge injection into a mathematically rigorous framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five real clinical tasks and ten baselines; comprehensive and thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though the notation is dense.
- **Value**: ⭐⭐⭐ Deployment requires multiple external knowledge sources and LLM APIs, posing a non-trivial barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](../../ACL2026/medical_nlp/beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](../../ACL2026/medical_nlp/text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](../../ACL2026/medical_nlp/reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)

</div>

<!-- RELATED:END -->
