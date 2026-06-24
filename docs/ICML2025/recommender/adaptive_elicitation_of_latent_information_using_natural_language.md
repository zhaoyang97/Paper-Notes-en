---
title: >-
  [Paper Note] Adaptive Elicitation of Latent Information Using Natural Language
description: >-
  [ICML 2025][Recommender Systems][Adaptive Information Acquisition] An LLM-based adaptive information elicitation framework is proposed. By performing autoregressive forward simulation of future observations using a meta-learned predictive model, it quantifies and distinguishes epistemic and aleatoric uncertainties, and adaptively selects the most informative natural language questions to efficiently reduce epistemic uncertainty about a latent entity.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Adaptive Information Acquisition"
  - "Uncertainty Quantification"
  - "Meta-Learning"
  - "LLM Fine-Tuning"
  - "Active Learning"
  - "Predictive View"
date: 2026-05-08
content_hash: 616051ac449607c9
---

# Adaptive Elicitation of Latent Information Using Natural Language

**Conference**: ICML 2025  
**arXiv**: [2504.04204](https://arxiv.org/abs/2504.04204)  
**Code**: [namkoong-lab/adaptive-elicitation](https://github.com/namkoong-lab/adaptive-elicitation)  
**Area**: Recommender Systems / Information Acquisition  
**Keywords**: Adaptive Information Acquisition, Uncertainty Quantification, Meta-Learning, LLM Fine-Tuning, Active Learning, Predictive View  

## TL;DR

An LLM-based adaptive information elicitation framework is proposed. By performing autoregressive forward simulation of future observations using a meta-learned predictive model, it quantifies and distinguishes epistemic and aleatoric uncertainties, and adaptively selects the most informative natural language questions to efficiently reduce epistemic uncertainty about a latent entity.

## Background & Motivation

Many critical applications require efficient elicitation of information from latent entities: designing personalized lesson plans for students requires assessing their skill levels, clinical visits require rapid assessment of patient health, and online services need to quickly understand new user preferences. In these scenarios, the target entity $U$ (e.g., student ability, health status, user preference) is **not directly observable** and must be inferred through indirect question-answer pairs.

### Limitations of Prior Work

**Traditional Bayesian methods** (e.g., Thompson Sampling, BOED) require explicit modeling of the latent space (e.g., Gaussian or Bernoulli distributions), making them difficult to scale to high-dimensional, complex spaces like natural language.

**Existing LLMs** possess strong world knowledge but lack **strategic information elicitation** capabilities—they passively process uncertainty and cannot actively select optimal questions to reduce epistemic uncertainty about a new individual.

**Bayesian optimization** methods perform well when the dimension is $\leq 20$, but natural language embedding dimensions are typically in the thousands, which far exceeds their applicable scope.

**Static questionnaires/tests** fail to dynamically adjust subsequent questioning strategies based on already obtained information, resulting in low information acquisition efficiency.

### Key Insight

A key conceptual shift is proposed: **instead of directly modeling the latent variable $U$, a "predictive view" is adopted**—quantifying uncertainty as the predictive uncertainty regarding future observed answers $Y_{t+1:\infty}$. If infinitely many data points are observed, all epistemic uncertainty vanishes, leaving only aleatoric uncertainty. This perspective enables training autoregressive models directly in the natural language space, avoiding the difficulty of explicitly modeling latent variable distributions.

## Method

### Overall Architecture

The framework consists of three stages:

1. **Meta-training**: Training a predictive language model $p_\theta$ on historical question-answering data.
2. **Uncertainty Quantification**: Utilizing $p_\theta$ to quantify uncertainty about future answers via autoregressive forward simulation.
3. **Adaptive Question Selection**: Selecting optimal questions based on expected information gain (EIG) using a greedy or MCTS strategy.

### Problem Formulation

Let the unobservable latent entity be $U \in \mathcal{U}$ (e.g., student skill profile). By asking a question $X \in \mathcal{X}$, an answer $Y \sim Q(\cdot | X, U)$ is obtained. The goals are: (1) to quantify the uncertainty about $U$ based on existing question-answer pairs, and (2) to adaptively select subsequent questions $X$ to maximize information gain.

### Key Designs

#### 1. Uncertainty Quantification from a Predictive View

Unlike traditional methods that directly model $U$ (e.g., specifying a probability distribution), this work adopts a **missing data perspective**:

- **Epistemic uncertainty** = uncertainty that can be reduced with more data = the conditional entropy of unobserved answers $Y_{t+1:\infty}$.
- **Aleatoric uncertainty** = the inherent randomness of the data, which cannot be reduced by more observations.

$$
\text{Uncertainty} = H_P(Y_{t+1:\infty} \mid X_{1:t}, Y_{1:t})
$$

This approach operates directly in the observable space $(X, Y)$, **completely eliminating the need for explicit modeling of the latent variable $U$**.

#### 2. Meta-learned Autoregressive Predictive Model

**Data Organization**: Each entity $U$ corresponds to a sequence of question-answer pairs $(X_1^{(U)}, Y_1^{(U)}, X_2^{(U)}, Y_2^{(U)}, \ldots)$, which are concatenated into a long string as input for the LLM.

**Training Objective**: Maximizing the joint log-likelihood:

$$
\max_{\theta} \frac{1}{|\mathcal{U}_{\text{train}}|} \sum_{U \in \mathcal{U}} \sum_{t=1}^{T} \log p_\theta(Y_t^U \mid \mathcal{H}_{t-1}, X_t^U)
$$

This is equivalent to minimizing the KL divergence between $p_\theta$ and the true distribution $Q$.

**Training Techniques**:
- **Random permutation**: During training, the order of question-answer pairs within each entity is randomly shuffled to ensure that the learned $p_\theta$ is insensitive to order.
- **Gradient masking**: Loss is computed only on answer tokens, masking out the tokens of the question text.
- **LoRA-based fine-tuning**: Initialized from Llama-3.1-8B, with LoRA parameters $\alpha=24, r=8, \text{dropout}=0.1$.

#### 3. Adaptive Question Selection

**Expected Information Gain (EIG)**: For each candidate question $x_{t+1}$, the degree to which it reduces uncertainty about the target $Z$ is computed via forward simulation:

$$
\text{EIG}_t(Z; x_{t+1}) = H_{p_\theta}(Z \mid \mathcal{H}_t) - \mathbb{E}[H_{p_\theta}(Z \mid \mathcal{H}_t \cup (x_{t+1}, Y_{t+1}))]
$$

where $Y_{t+1}$ is simulated and generated by the meta-learned model $p_\theta$.

**Two Selection Strategies**:

| Strategy | Mechanism | Complexity | Advantages |
|------|------|--------|------|
| Greedy Selection | Step-by-step selection of questions maximizing single-step EIG | $O(K)$, where $K$ is the number of candidates | Simple and efficient, with a theoretical guarantee of loss $\leq \frac{1}{e}$ |
| MCTS Planning | Simulates multi-step futures using Monte Carlo Tree Search | $O(K \cdot N \cdot d)$ | Captures multi-step effects and discovers rare features |

### Loss & Training

- **Model**: Llama-3.1-8B (FP16) + LoRA
- **Optimizer**: AdamW, lr=1e-4, $\beta=(0.9, 0.95)$, weight decay=0.1
- **Learning Rate Scheduler**: Linear warmup + cosine annealing
- **Training Epochs**: 10,000 epochs, batch size=4, block size=1024
- **Model Selection**: Checkpoint with the lowest validation loss
- **Data Split**: 70/15/15 train/val/test split by entity

### Theoretical Guarantees

**Proposition 2.1 (Simulator Fidelity)**:
The lower bound on the performance of the optimal query set $\mathcal{X}^*$ selected based on the simulator $p_\theta$ under the true distribution $q$ is controlled by the $\chi^2(q \| p_\theta)$ divergence—the closer the simulator is to the true distribution, the stronger the performance guarantee.

**Proposition 2.2 (Greedy Approximation Ratio)**:
If the entropy produced by $p_\theta$ satisfies submodularity, the performance difference between the greedy strategy and the optimal combinatorial strategy is at most $\frac{1}{e}$ of the maximum information gain.

## Key Experimental Results

### Three Evaluation Scenarios

| Scenario | Dataset | Latent Variable $U$ | Question $X$ | Answer $Y$ | Scale |
|------|--------|-----------|---------|---------|------|
| 20 Questions Game | Twenty Questions (New) | Hidden Object (800 types) | Yes/No questions | Yes/No/Maybe | 800×1200 |
| Dynamic Polling | OpinionQA | Political preferences | Multiple-choice questions | Multiple-choice | 1498 questions |
| Student Assessment | EEDI | Mathematical ability | Multiple-choice questions | 1-out-of-4 choice | 938 questions |

### Main Results (Average of 10,000 Trials)

| Method | 20 Questions Acc | OpinionQA Acc | EEDI Acc |
|------|---------|---------|------|
| Base LLM (Random selection) | Lowest | Lowest | Lowest |
| ICT + Embedding similarity selection | Moderate | Moderate | Moderate |
| **Ours (Greedy EIG)** | **Highest** | **Highest** | **Highest** |

Across all three datasets, the accuracy of Ours consistently improves as the number of questions increases, whereas for the ICT baseline using embedding similarity for question selection, more questions do not always guarantee better predictions.

### Adaptive Gains for Rare Features (Figure 5)

| Question Subset | EIG vs. Random (Relative Gain) | MCTS vs. Random (Relative Gain) |
|---------|----------------------|----------------------|
| All Questions | Baseline gain | Higher than EIG |
| Moderate Difficulty (<50% population correct) | ~5× gain or more | Higher |
| Hard Questions (<30% population correct) | **>10× gain** (EEDI, 20Q) | **Maximum Gain** |

**Key Findings**: The adaptive strategy provides the largest gains in identifying **rare/atypical features**—when a latent entity exhibits uncommon behavior within the population (e.g., a student failing on a concept that most master), the advantage of the adaptive method can be more than 10 times that of the random method.

### Ablation Study on Training (Figure 6, Twenty Questions)

| Underlying Model | Planning Gain Ratio (All) | Planning Gain Ratio (Hard) |
|---------|-----------------|-----------------|
| Base LLM | < 1.0 (planning is harmful) | ~0.85 (reduced by 15%) |
| ICT | ≈1.0 (no significant change) | ≈1.0 |
| **Ours (Meta-trained)** | **>1.0 (significant improvement)** | **Maximum Improvement** |

This indicates that models without proper meta-training cannot benefit from EIG planning and may even suffer from performance degradation. Meta-training is a crucial prerequisite for strategic information elicitation capabilities.

### Model Scale Ablation

| Base Model | Parameters | Performance Trend |
|---------|-------|---------|
| GPT-2 | ~124M | Lowest |
| Llama-3.2-1B | 1B | Moderate |
| Llama-3.1-8B | 8B | **Highest** |

Larger base model $\rightarrow$ better meta-training effects $\rightarrow$ more accurate uncertainty estimation.

## Highlights & Insights

1. **Conceptual Breakthrough—Predictive View**: Transforming the latent variable modeling problem into a missing data prediction problem completely bypasses the challenge of modeling abstract latent variable spaces, while directly leveraging the autoregressive nature and pre-trained knowledge of LLMs.
2. **Plug-and-Play**: The framework can be applied directly on top of any pre-trained LLM, utilizing web-scale language knowledge to understand uncertainty.
3. **Elegant Experimental Design**: The stratified analysis in Figure 5 (All/Moderate/Hard) clearly demonstrates the substantial gains of the adaptive strategy for rare features, which is highly valuable for practical applications (e.g., personalized recommendations, precision diagnosis).
4. **Solid Theoretical Foundation**: Two propositions guarantee simulator fidelity and the greedy approximation ratio, respectively, providing a theoretical foundation for the framework.
5. **New Dataset Contribution**: Constructing and open-sourcing the Twenty Questions dataset (800 objects $\times$ 1200 questions), filling a gap in adaptive querying benchmarks.

## Limitations & Future Work

1. **Closed Question Space**: The current framework selects questions from a fixed candidate pool and cannot **generate** new questions. In real-world scenarios (e.g., open-ended clinical diagnosis), the system should dynamically generate optimal questions.
2. **Sequence Independence Assumption**: Training randomly shuffles the order of question-answer pairs, assuming the answers are independent of the questioning order. In actual conversational settings, contextual dependencies are crucial (e.g., a student being influenced by the previous question).
3. **Computational Overhead**: MCTS planning requires simulating multiple future paths for each candidate question, which is computationally expensive for large candidate sets and deep planning horizons.
4. **Dependency on Meta-Training Data**: The framework assumes the availability of abundant historical question-answering trajectory data, which may limit its application in new domains or cold-start scenarios.
5. **Evaluation Biased Toward Classification Settings**: The answers in the three experimental scenarios are all discrete choices (Yes/No, multiple-choice); performance in open-ended natural language response settings has not yet been validated.
6. **Risk of Simulator Inaccuracy**: Proposition 2.1 shows that performance is bounded by the $\chi^2(q \| p_\theta)$ divergence; if there is a severe shift in the test distribution, the strategy guided by the simulator may fail.

## Related Work & Insights

- **UoT (Hu et al., 2024)** and **OPEN (Handa et al., 2024)**: These also construct information elicitation workflows on LLMs but use off-the-shelf LLMs without meta-training. Ours demonstrates that models without meta-training may even **degrade** performance during planning.
- **Computerized Adaptive Testing (CAT)**: Classical methods like IRT and DINA use simple parametric models to capture student potential. Ours uses LLMs to directly model complex natural language answers, greatly expanding the scope of applicability.
- **Decision Transformer Related Work**: These use sequential models for reinforcement learning decision-making. Ours extends this idea to uncertainty quantification and information elicitation.
- **Insight**: The core idea of this framework—"predicting future observations instead of modeling latent variables"—can be extended to active user profiling in recommender systems: rather than explicitly modeling user interest vectors, the system can predict user reactions to future items, thereby selecting interactions that best reveal preferences.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐⭐ | The predictive view replaces latent variable modeling, offering an elegant and far-reaching concept. |
| Theoretical Depth | ⭐⭐⭐⭐ | Two propositions guarantee fidelity and approximation ratios, though it is not extremely deep. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 3 diverse scenarios + rich ablations + 10K trials. |
| Practicality | ⭐⭐⭐⭐ | The framework is general-purpose, but requires substantial historical data and computational resources. |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear motivation, fluent narrative, and beautifully designed figures. |
| **Overall** | **⭐⭐⭐⭐⭐** | High-quality ICML 2025 paper, pioneering the use of LLMs for adaptive information elicitation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent Space](../../ICLR2026/recommender/supporting_high-stakes_decision_making_through_interactive_preference_elicitatio.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[ICLR 2026\] Reinforced Latent Reasoning for LLM-based Recommendation](../../ICLR2026/recommender/reinforced_latent_reasoning_for_llm-based_recommendation.md)
- [\[ACL 2025\] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](../../ACL2025/recommender/kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](../../ACL2025/recommender/bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)

</div>

<!-- RELATED:END -->
