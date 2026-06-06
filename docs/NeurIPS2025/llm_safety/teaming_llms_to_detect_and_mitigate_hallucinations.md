---
title: >-
  [Paper Note] Teaming LLMs to Detect and Mitigate Hallucinations
description: >-
  [NeurIPS 2025][LLM Safety][Hallucination detection] This paper generalizes single-model consistency methods (Self-Consistency + Semantic Entropy) to a multi-model "consortium" setting comprising heterogeneous LLMs. By ag…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Hallucination detection"
  - "multi-model consistency"
  - "semantic entropy"
  - "ensemble voting"
  - "inference cost"
date: 2026-05-08
content_hash: 9197ed99b3c71226
---

# Teaming LLMs to Detect and Mitigate Hallucinations

**Conference**: NeurIPS 2025
**arXiv**: [2510.19507](https://arxiv.org/abs/2510.19507)  
**Code**: Not released  
**Area**: LLM Safety
**Keywords**: Hallucination detection, multi-model consistency, semantic entropy, ensemble voting, inference cost

## TL;DR

This paper generalizes single-model consistency methods (Self-Consistency + Semantic Entropy) to a multi-model "consortium" setting comprising heterogeneous LLMs. By aggregating responses from models with diverse training backgrounds, the approach breaks the consistent hallucinations that arise within a single model. Evaluating a large number of consortium combinations over a pool of 15 LLMs, the paper finds that well-matched strong-model consortia outperform the strongest single-model baseline in 92% of cases while incurring lower inference cost.

## Background & Motivation

**Background**: LLM hallucination is a central challenge in deploying large models. Consistency-based methods are currently the dominant approach for hallucination detection and mitigation—Self-Consistency selects the final answer via majority voting over multiple samples, while Semantic Entropy assesses whether a model is "guessing" by computing the semantic entropy of sampled responses. These methods achieve state-of-the-art performance on multiple benchmarks.

**Limitations of Prior Work**: Single-model consistency methods suffer from a fundamental flaw: when a model produces "consistent hallucinations" on a given query (i.e., makes the same systematic error repeatedly), the incorrect answer can win the majority vote (hallucination mitigation fails), and semantic entropy may remain low (hallucination detection fails). This situation is common when certain knowledge is underrepresented or biased in a model's training data.

**Key Challenge**: The upper bound of single-model consistency is constrained by the training data and architecture of a single model—if the model has systematically learned incorrect information in a given domain, additional sampling is ineffective, since all samples are drawn from the same flawed distribution.

**Goal**: How can the effectiveness of consistency-based methods for hallucination detection and mitigation be further improved without modifying the models or requiring white-box access?

**Key Insight**: Different LLMs have distinct training data, training procedures, and model architectures, making it unlikely that they share the same training deficiencies or make the same "educated guesses." Aggregating responses across multiple models allows the strengths of different models to complement and correct one another.

**Core Idea**: Replace single-model sampling with a multi-model consortium, allowing the differentiated knowledge of heterogeneous LLMs to mutually correct each other's biases, thereby achieving more reliable hallucination detection and mitigation.

## Method

### Overall Architecture

Given an input query, a consortium of $M$ models, and a total sampling budget of $N$ responses, the pipeline consists of four steps: (1) distribute $N$ uniformly across $M$ models, with each model independently sampling $N/M$ responses; (2) cluster all $N$ responses by semantic equivalence; (3) apply Consortium Voting over the clusters via majority vote to select the final answer; and (4) compute Consortium Entropy over the cluster distribution to estimate the probability that the answer is a hallucination.

### Key Designs

1. **Consortium Voting (hallucination mitigation)**:

    - **Function**: Selects the final answer from the mixed multi-model responses.
    - **Mechanism**: All $N$ responses from all models are clustered into semantic equivalence classes $\{C_1, C_2, \ldots, C_{|C|}\}$, and the class containing the most responses is selected as the final answer: $\text{answer} = \arg\max_{C_i} \sum_{m \in \mathcal{M}} \sum_{j=1}^{N/|\mathcal{M}|} \mathbf{1}[r_{m,j} \in C_i]$
    - **Design Motivation**: When one model produces consistent hallucinations, correct responses from other models can "outvote" the hallucination—inter-model heterogeneity transforms noise into an advantage.

2. **Consortium Entropy (hallucination detection)**:

    - **Function**: Estimates the hallucination confidence of the final answer.
    - **Mechanism**: The consortium's probability distribution over equivalence classes is estimated as $P(C_i|x) = \frac{1}{N}\sum_{m} \sum_j \mathbf{1}[r_{m,j} \in C_i]$, and semantic entropy is computed as $SE(x) = -\sum_{C_i} P(C_i|x) \log P(C_i|x)$. Low entropy indicates high consistency and low hallucination probability; high entropy indicates high uncertainty and elevated hallucination risk.
    - **Design Motivation**: The multi-model distribution is more "faithful" than any single model's distribution—even if one model is highly confident in an incorrect answer, other models are unlikely to make the same error, causing consortium entropy to correctly increase in such cases.

3. **Three-Level Baseline Design and Model Selection Strategy**:

    - **Function**: Provides a fair evaluation of consortium performance and guides model selection.
    - **Mechanism**: Each of the $M$ constituent models is evaluated independently using the full $N$-sample budget as a single-model consistency baseline. Three baselines are defined: Hard (the strongest single model), Standard (the median), and Worst-case (the weakest). Mock benchmark scores are used to guide consortium composition—models of comparable and high capability are preferred.
    - **Design Motivation**: The Hard baseline is particularly valuable—if a consortium outperforms the "known strongest single model," the collaboration provides genuine gain rather than merely averaging out performance.

### Sampling and Clustering Strategy

By default, $N=40$ responses are sampled per query and distributed uniformly across consortium models. Nucleus sampling (top-$p=0.9$, temperature=0.5) with Chain-of-Thought prompting is used. Semantic clustering follows task-specific strategies: option equivalence for multiple-choice questions and mathematical answer equivalence for math problems, avoiding the additional noise introduced by general-purpose NLI-based judgment. Confidence intervals are estimated via 100 bootstrap iterations.

## Key Experimental Results

### Main Results

Experiments use a pool of 15 LLMs (6B–141B parameters, spanning LLaMA, Mistral, Qwen, and Gemma families) evaluated on 11 tasks (GSM8K, GPQA-Diamond, 8 MMLU subsets, TruthfulQA).

**Well-matched strong-model consortia** (586 consortia with std ≤ 5 and mean ≥ 70):

| Metric | Baseline Type | Mean Score Change (%) | Win Rate |
|--------|---------------|-----------------------|----------|
| Accuracy | Hard | +1.33 ± 1.03 | 92% |
| Accuracy | Standard | +3.70 ± 1.20 | 99% |
| AUROC | Hard | +1.84 ± 1.48 | 92% |
| AUROC | Standard | +5.63 ± 1.46 | 100% |
| AURAC | Hard | +2.75 ± 0.69 | 100% |
| AURAC | Standard | +5.39 ± 1.09 | 100% |

### Ablation Study

| Analysis Dimension | Finding | Implication |
|--------------------|---------|-------------|
| Model strength | Higher average strength yields more reliable consortium gains over the Hard baseline | Stronger models exhibit more consistent hallucinations and benefit more from consortia |
| Capability variance | Lower variance yields more reliable gains; low-variance-only consortia exceed Hard AUROC in 68% of cases | Matched models outperform mixed-strength combinations |
| High strength + low variance | All three metrics exceed the Hard baseline in 92%+ of cases | Optimal consortium strategy |
| Cost–performance frontier | Consortia simultaneously achieve higher performance and lower cost | The strongest single model is the most expensive; redistributing budget to cheaper models is cost-effective |
| Sampling budget scaling | Consortium advantage grows consistently in the range $N=10$–$40$ | The effect is not incidental |

### Key Findings

- **Consortium composition is critical**: Not all combinations are effective. Well-matched strong-model consortia perform best—models of comparable high capability provide the strongest complementarity, as each possesses high-quality but non-overlapping knowledge across domains.
- **Counterintuitive finding**: Stronger models benefit more from consortia. The proposed hypothesis is that stronger models make "smarter guesses" that are more consistent, causing single-model semantic entropy to underestimate hallucination risk; consortia break this pattern of "intelligently consistent errors."
- **Unexpected value of weaker models**: Pairing strong models with weaker ones can sometimes reduce inference cost while improving performance—the "random errors" of weaker models help the correct answer achieve relatively higher consistency.

## Highlights & Insights

- **Fully black-box, plug-and-play**: The method requires no internal model access and can directly combine any LLM APIs. This is a key practical advantage over white-box methods (e.g., internal embedding consistency) and is highly compatible with API-only deployment scenarios.
- **Win-win cost–performance trade-off**: While multi-model methods are conventionally assumed to increase cost, this paper demonstrates that well-designed consortia can simultaneously reduce cost and improve performance—the strongest single model is typically the most expensive, and reallocating part of the budget to cheaper models is economically efficient. This finding has direct commercial deployment value.
- **Profound theoretical insight**: Stronger models produce more consistent hallucinations because their "guesses are more intelligent"—this observation suggests that improvements in model capability may make hallucination detection harder, providing an important clue for understanding the relationship between scaling laws and safety.
- **Exemplary evaluation design**: The three-level baseline design avoids the bias of comparing against weak references; the large-scale evaluation over 586 consortia ensures statistical reliability. This evaluation methodology is broadly applicable to research on ensemble methods.

## Limitations & Future Work

- **Still more expensive than lightweight methods**: Although less costly than single-model consistency, multiple sampling passes are still required. Adaptive strategies could be explored—first sampling a small number of responses to determine whether the consortium should be activated, exempting "easy" queries from multi-model overhead.
- **Restricted to algorithmically verifiable equivalence**: All current experiments involve multiple-choice and math tasks. Open-ended generation requires LLM-based semantic equivalence judgment, which introduces additional cost and noise.
- **Consortium selection requires prior knowledge**: The method relies on mock benchmark estimates of model capability to guide composition. Automated consortium selection (e.g., online learning of optimal combinations via bandit algorithms) is an important avenue for future work.
- **Minority experts may be overridden**: When one model has expertise in a specific domain but the remaining consortium members consistently err, the expert is outvoted. Weighted voting or domain-aware aggregation could mitigate this issue.
- **Uniform budget allocation may be suboptimal**: When one model is known to be stronger, allocating it a larger budget may be preferable. Non-uniform allocation and adaptive budget scheduling merit further investigation.

## Related Work & Insights

- **vs. Self-Consistency (Wang et al., 2023)**: This paper is a multi-model generalization of Self-Consistency. Self-Consistency samples a single model multiple times and uses majority voting; the proposed method's advantage lies in breaking systematic biases within a single model. However, Self-Consistency may be more economical when a single model is sufficiently strong and the domain is narrow.
- **vs. Semantic Entropy (Farquhar et al., 2024)**: A direct extension from single-model semantic entropy to consortium semantic entropy. The advantage is that the uncertainty distribution across multiple models is more faithful, at the cost of managing API calls to multiple models.
- **vs. Multi-Agent Debate (Du et al., 2024)**: Debate involves iterative inter-model discussion to reach consensus, whereas the proposed method performs independent sampling followed by aggregation without inter-model interaction. The two approaches are orthogonal—a two-stage pipeline could first use the consortium for independent answering and then trigger debate for high-entropy queries.
- **vs. Retrieval-Augmented Generation (RAG)**: RAG reduces hallucinations by providing external knowledge and is orthogonal to consistency-based methods. Combining RAG with consortium consistency may outperform either method alone.

## Rating

- Novelty: ⭐⭐⭐ The core idea (multi-model ensembling) is not new, but its systematic formulation and large-scale evaluation within the consistency-based hallucination detection framework is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 LLMs, 11 tasks, 586 consortia, three-level baselines, and cost analysis—extremely rigorous.
- Writing Quality: ⭐⭐⭐⭐ The three-level baseline design is elegant; the analytical logic is clear; figures and tables are intuitive.
- Value: ⭐⭐⭐⭐ Black-box and plug-and-play; directly actionable for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](../../ACL2026/llm_safety/meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](evaluation_of_vision-llms_in_surveillance_video.md)
- [\[NeurIPS 2025\] Probabilistic Reasoning with LLMs for K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)

</div>

<!-- RELATED:END -->
