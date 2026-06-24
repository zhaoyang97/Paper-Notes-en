---
title: >-
  [Paper Note] Enough Coin Flips Can Make LLMs Act Bayesian
description: >-
  [ACL 2025][LLM (Other)][Bayesian inference] Through the controlled stochastic process of coin flipping, this work systematically investigates whether LLMs perform Bayesian inference in in-context learning. The study reveals that LLMs typically possess biased priors, but as contextual evidence increases, they correct their posterior estimates in an approximate Bayesian update manner. The deviation primarily stems from poorly calibrated priors rather than a flawed updating mech…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Bayesian inference"
  - "In-Context Learning"
  - "Probability estimation"
  - "Prior bias"
  - "Attention mechanism"
date: 2026-05-08
content_hash: 70070bd458e87d7f
---

# Enough Coin Flips Can Make LLMs Act Bayesian

**Conference**: ACL 2025  
**arXiv**: [2503.04722](https://arxiv.org/abs/2503.04722)  
**Code**: Yes (Project Page)  
**Area**: LLM/NLP  
**Keywords**: Bayesian inference, In-Context Learning, Probability estimation, Prior bias, Attention mechanism  

## TL;DR

Through the controlled stochastic process of coin flipping, this work systematically investigates whether LLMs perform Bayesian inference in in-context learning. The study reveals that LLMs typically possess biased priors, but as contextual evidence increases, they correct their posterior estimates in an approximate Bayesian update manner. The deviation primarily stems from poorly calibrated priors rather than a flawed updating mechanism.

## Background & Motivation

**Background**: Large language models (LLMs) demonstrate strong few-shot learning capabilities via in-context learning (ICL). Existing theoretical works (e.g., Xie et al., 2021) suggest that ICL may implicitly perform Bayesian inference. However, because the true posterior distribution is unknown in most tasks, this hypothesis remains difficult to directly verify.

**Limitations of Prior Work**: (1) Previous studies rely on tasks (e.g., question answering, language modeling) with unknown posterior distributions, making it impossible to precisely evaluate whether model inference aligns with Bayesian principles. (2) Controlled theoretical works often rely on strong assumptions about model architectures or data domains, which are difficult to generalize to pre-trained LLMs. (3) It remains unclear whether the test-time behavior of LLMs represents simple pattern matching or structured probabilistic reasoning.

**Key Challenge**: The link between the success of ICL and Bayesian inference is intuitively plausible but lacks direct empirical evidence. There is a need for a controlled environment where the true posterior can be exactly computed to verify this connection.

**Goal**: Do pre-trained LLMs truly perform Bayesian posterior updates given sequential evidence? Is the source of their deviation a prior calibration issue or a defect in their updating mechanism?

**Key Insight**: Utilizing biased coin flips as a controlled stochastic process. The conjugate system of a binomial distribution and a Beta prior allows all Bayesian quantities to be computed precisely, enabling a direct comparison between LLM predictions and the theoretical posterior.

**Core Idea**: Directly verifying that LLMs update priors in a near-Bayesian manner during ICL through coin-flipping experiments where the exact posterior can be computed.

## Method

### Overall Architecture

A suite of controlled experiments is designed: (1) extracting the LLM's prior distribution over coin flip outcomes (zero-shot setting); (2) testing prior updates via explicit bias instructions; (3) providing sequences of biased coin flips via ICL to compare model predictions with the Bayesian posterior using Total Variation Distance (TVD); (4) constructing dynamic bias-switching scenarios (first 50 flips with $\theta_1 = 0.75$, subsequent 50 flips with $\theta_2 = 0.25$) to analyze "online" Bayesian updating behavior; and (5) analyzing the relationship between attention weights and the quality of updates.

### Key Designs

1. **Prior Extraction and Bias Analysis**
    - **Function**: Quantifying the LLM's intrinsic prior over coin-flip outcomes.
    - **Mechanism**: Querying the model with 50 different prompt variations (e.g., "I flipped a coin and it landed on") to extract normalized logit values as prior estimates for the probability of "heads". Then, testing whether the prior can be updated using explicit bias statements (e.g., "coins land on heads X% of the time").
    - **Design Motivation**: Understanding the initial bias of the LLM is a prerequisite for evaluating its Bayesian behavior. A scenario where the prior is heavily biased but the updating mechanism is correct suggests a diagnostic conclusion entirely different from one where the prior is correct but the updating mechanism is flawed.

2. **Posterior Estimation under ICL Sequences**
    - **Function**: Evaluating whether the LLM converges to the correct posterior as ICL evidence accumulates.
    - **Mechanism**: Providing sequences of biased coin flips of various lengths (3 to 100 samples) as ICL demonstrations, and comparing the TVD between the model's predicted distribution and the theoretical Beta posterior distribution.
    - **Design Motivation**: If the LLM performs Bayesian updating, the TVD should approach zero as evidence increases. If it is merely pattern matching, the convergence pattern will be qualitatively different.

3. **Bayesian Filtering Fit with a Decay Factor**
    - **Function**: Accurately characterizing the LLM's updating behavior and quantifying its "memory time window".
    - **Mechanism**: In the dynamic bias-switching scenario, the Bayesian update is modified with an exponential decay factor $\gamma$ as $\alpha \leftarrow \gamma\alpha + I(H)$. The optimal $\gamma$ value for each model is optimized via L-BFGS-B; $\gamma < 1$ indicates that the model favors "local Bayesian updating", placing more weight on recent evidence.
    - **Design Motivation**: Classical Bayesian filtering ($\gamma=1$) cannot fully account for LLM behavior. Introducing the $\gamma$ parameter reveals that models perform "myopic Bayesian" updates, and the value of $\gamma$ varies across different model architectures.

### Loss & Training

This is an analytical study and does not involve model training. The core evaluation metric is the Total Variation Distance (TVD):

$$\delta(p^*, \hat{p}_{\mathcal{M}}) = \frac{1}{2} \sum_{o \in \Omega} |p^*(o) - \hat{p}_{\mathcal{M}}(o)|$$

The model's predictive distribution is obtained by extracting and normalizing the model's logits over the defined output space $\Omega = \{\text{heads}, \text{tails}\}$.

## Key Experimental Results

### Main Results

Best-fit $\gamma$ values for different models in the biased coin ICL task:

| Model | Best-Fit $\gamma$ | Meaning |
|------|-----------|------|
| Llama3.1-8B | 0.8807 | Close to classical Bayesian, long time window |
| Llama3.1-8B-Instruct | 0.4655 | More localized update, short time window |
| Phi-2 | 0.8781 | Close to classical Bayesian |
| Mistral-7B | 0.6903 | Moderate time window |
| Mistral-7B-Instruct | 0.9107 | Exception: more similar to classical Bayesian after instruction tuning |
| Gemma-2-2B | 0.4910 | Shorter time window |
| Gemma-2-2B-Instruct | 0.3087 | Most localized update |
| OLMoE-1B-7B | 0.3268 | Shorter time window |

### Ablation Study

Validation of key Bayesian behavioral characteristics:

| Experimental Setting | Key Findings |
|---------|---------|
| Explicit bias instructions (base models) | Instructions ignored, always outputs ~60-80% bias towards heads |
| Explicit bias instructions (instruct models) | Slight improvement, performs best under extreme biases (0% / 100%) |
| Model scaling (Pythia 70M $\rightarrow$ 12B) | Model size has no significant impact on prior quality and ICL performance |
| Number of ICL samples | 3 samples significantly improve TVD, but 100 samples still fail to fully eliminate prior bias |
| Attention weight vs. posterior quality | Virtually no correlation ($R=0.02, p=0.48$) |

### Key Findings

1. **All models possess a prior biased toward "heads"**: This is potentially related to the tokenization structure—"tails" requires two tokens to encode in some models.
2. **Explicit instructions are less effective than ICL demonstrations**: Base (non-instruct) models almost entirely ignore bias instructions, and instruct models only show improvement at extreme values.
3. **ICL overall aligns with Bayesian posterior updates**: The deviation is primarily due to poorly calibrated priors, rather than failures in the update mechanism.
4. **Models perform "myopic Bayesian" updates**: The fitted $\gamma < 1$ indicates that the models are more sensitive to recent evidence, which is equivalent to a Bayesian update with a finite time window.
5. **Instruction-tuning decreases the $\gamma$ value**: Instruction tuning makes models more "forgetful", rendering them more willing to shift behaviors in response to new evidence.
6. **Attention magnitude is uncorrelated with update quality**: Challenging previous theoretical explanations, this disproves the hypothesis that attention directly regulates Bayesian updates.
7. **Model scaling has limited impact on Bayesian behavior**: Larger models do not necessarily execute probabilistic reasoning better.

## Highlights & Insights

- Highly elegant experimental design: leveraging the binomial-Beta conjugate system to make all Bayesian quantities precisely computable.
- The discovery of "myopic Bayesian" updating ($\gamma < 1$) provides a natural explanation for why ICL loss remains relatively high in long contexts.
- The finding that instruction tuning decreases the $\gamma$ value is profound: it implies that instruct models favor "local adaptation" over "global accumulation".
- The finding that attention magnitude is uncorrelated with update quality challenges previous theoretical explanations.

## Limitations & Future Work

- Only evaluates discrete binary outputs (heads/tails), without extending to continuous distributions or higher-dimensional stochastic processes.
- Cannot be applied to closed-source models (requires access to logits).
- Coin tossing is highly simplified; whether Bayesian behavior still holds in more complex NLP tasks remains to be validated.
- The source of prior bias is not deeply investigated: is it due to a higher frequency of "heads" in training data or other reasons?
- The impact of Chain-of-Thought (CoT) prompting on Bayesian behavior is not explored.

## Related Work & Insights

- Xie et al. (2021): Theoretical foundation of ICL as implicit Bayesian inference.
- Wang et al. (2024): The connection between prompt order sensitivity and latent variable models in ICL.
- Falck et al. (2024): Analysis of the Bayesian nature of ICL from a martingale perspective.
- The concept of the $\gamma$ parameter in this paper can be interpreted as a "forgetting factor" in Bayesian filtering.

## Rating

- **Novelty**: 4/5 — Clever controlled experimental design, and the "myopic Bayesian" finding is novel.
- **Technical Depth**: 4/5 — Rigorous Bayesian theoretical framework with multi-dimensional empirical analysis.
- **Experimental Thoroughness**: 4/5 — Comprehensive across multiple models, scalings, and attention analyses.
- **Value**: 3/5 — Primarily focused on theoretical insights, with indirect implications for practical applications.
- **Overall Rating**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] LLMs Can Be Easily Confused by Instructional Distractions](llms_can_be_easily_confused_by_instructional_distractions.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)

</div>

<!-- RELATED:END -->
