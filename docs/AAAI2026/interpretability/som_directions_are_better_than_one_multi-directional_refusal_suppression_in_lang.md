---
title: >-
  [Paper Note] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models
description: >-
  [AAAI 2026][Interpretability][Refusal suppression] This paper demonstrates that refusal behavior in LLMs is not encoded by a single direction but rather forms a low-dimensional manifold. It employs self-organizing maps (…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Refusal suppression"
  - "self-organizing map"
  - "multi-directional ablation"
  - "representation space"
  - "jailbreak attack"
date: 2026-05-08
content_hash: 3bedb620fa2809a1
---

# SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.08379](https://arxiv.org/abs/2511.08379)  
**Code**: [GitHub](https://github.com/pralab/som-refusal-directions)  
**Area**: LLM Alignment / Mechanistic Interpretability
**Keywords**: Refusal suppression, self-organizing map, multi-directional ablation, representation space, jailbreak attack

## TL;DR

This paper demonstrates that refusal behavior in LLMs is not encoded by a single direction but rather forms a low-dimensional manifold. It employs self-organizing maps (SOM) to extract multiple refusal directions and applies Bayesian optimization to search for the optimal ablation combination, surpassing single-direction baselines and dedicated jailbreak algorithms across multiple models.

## Background & Motivation

**Background**: With the rise of mechanistic interpretability, researchers have found that safety refusal behavior in LLMs can be encoded as directions in representation space. Arditi et al. (2024) proposed the pioneering single direction (SD) method—computing the centroid difference between harmful and harmless prompt representations as a "refusal direction" and ablating it from the model to bypass safety alignment.

**Limitations of Prior Work**:
- **Oversimplified single-direction assumption**: Recent mechanistic interpretability research shows that semantic and functional concepts (e.g., days of the week, trigonometric functions) are not encoded by a single linear direction but span low-dimensional manifolds in high-dimensional space.
- **Limited effectiveness of SD**: On some models, the attack success rate (ASR) after SD ablation is near 0% (e.g., ASR = 0% for SD on LLaMA2-7B), indicating that a single direction is far from sufficient to capture the full refusal behavior.
- **Orthogonal multi-direction methods (e.g., RDO) are also insufficient**: Although multiple orthogonal directions are extracted, only one direction is ablated at a time, ignoring synergistic effects among directions.

**Key Challenge**: There is a fundamental conflict between the multifaceted nature of refusal behavior—where refusals to different categories of harmful content may be encoded in different directions—and the inherent limitation of existing single-direction approaches.

**Goal**: To systematically discover multiple refusal directions in representation space and leverage their combinations for more effective refusal suppression.

**Key Insight**: Utilizing the topology-preserving and multi-neuron properties of SOM to model the manifold structure of harmful prompt representations, thereby extracting multiple directions.

**Core Idea**: (1) Theoretically prove that a single-neuron SOM converges to the centroid (i.e., SD is a special case of SOM); (2) Multi-neuron SOM captures the local structure of the manifold, with each neuron subtracting the harmless centroid to yield a refusal direction; (3) Bayesian optimization searches for the optimal combination of $k$ directions from the candidate pool for ablation.

## Method

### Overall Architecture

The multi-directional ablation (MD) method proceeds in four steps: (1) extract internal representations of harmful/harmless prompts from the target model; (2) train a SOM on harmful representations to obtain multiple neurons; (3) subtract the harmless centroid from each neuron to obtain a set of candidate refusal directions; (4) apply Bayesian optimization to search for the optimal $k$-direction combination and construct an ablation operator applied to all layers of the model.

### Key Designs

#### Module 1: Theoretical Guarantee that SOM Generalizes the Centroid

- **Function**: Prove that SD is a special case of MD, establishing a theoretical foundation.
- **Mechanism**: SD defines the refusal direction via the centroid difference $r^{(l)} = \mu^{(l)} - \nu^{(l)}$, where $\mu$ and $\nu$ are the centroids of harmful and harmless prompt representations, respectively. Proposition 1 proves that a single-neuron SOM with learning rate $\alpha < 1/2$ converges to the data centroid, with error bound $(1-\alpha)^t \|w^{(0)}-\mu\| + \alpha \sigma$. Therefore, SD is precisely equivalent to training a single-neuron SOM on the harmful distribution.
- **Design Motivation**: To establish an elegant connection between SD and MD—MD is not an entirely new method but a natural generalization of SD, extending from a single neuron to multiple neurons.

#### Module 2: SOM-Based Multi-Directional Extraction

- **Function**: Use multiple SOM neurons to capture the manifold structure of harmful prompt representations and extract multiple refusal directions.
- **Mechanism**:
    - Select the optimal ablation layer $l^*$ (the layer that minimizes the probability of generating refusal tokens).
    - Collect harmful representations $\mathcal{X}_{hf}$ (representations at the last token position of all harmful prompts at layer $l^*$) and the harmless centroid $\nu$.
    - Train a $4 \times 4$ hexagonal-topology SOM (16 neurons) on $\mathcal{X}_{hf}$ for 10,000 steps, with learning rate $\alpha_t = 0.01/(1+2t/T)$ and Gaussian neighborhood function $\sigma = 0.3$.
    - Subtract the harmless centroid $\nu$ from each SOM neuron $w_\iota$ to obtain direction $r_\iota = w_\iota - \nu$, yielding 16 candidate directions in total.
- **Design Motivation**: Harmless prompt representations are relatively homogeneous and can be represented by a single centroid; harmful prompts span diverse categories such as violence, discrimination, and crime, resulting in a more complex representational distribution that requires multiple representative points. The topology-preserving property of SOM ensures that similar harmful categories are mapped to neighboring neurons.

#### Module 3: Bayesian Optimization for Direction Search

- **Function**: Search for the optimal $k$ directions ($k \in [2, 7]$) from the 16 candidates for combined ablation.
- **Mechanism**: Define the optimization problem $\max_{r_1,...,r_k \in \mathcal{R}} \mathbb{E}_{\mathcal{D}_{hf}}[\mathcal{J}(t, \hat{o})]$, where $\mathcal{J}$ is a judge model that evaluates whether the response is harmful and compliant. The ablation operator is a composition of multiple orthogonal projections: $\Psi = \Pi_{r_1^*} \circ \cdots \circ \Pi_{r_k^*}$, applied to every layer. Bayesian optimization with a TPE sampler is used to search over the HarmBench validation set, with 128 trials for $k \leq 3$ and 512 trials for $k > 3$.
- **Design Motivation**: Exhaustive search becomes rapidly infeasible as $k$ increases ($\binom{16}{7} = 11440$); Bayesian optimization is efficient for black-box objectives. The judge model (HarmBench-Llama-2-13B-cls) provides harmfulness judgments as the objective function, directly optimizing ASR.

### Loss & Training

**SOM Training Objective**: At each step, all neurons are updated as $w_\iota^{(t+1)} = w_\iota^{(t)} + \alpha_t \theta(\iota^*(x^{(t)}), \iota)(x^{(t)} - w_\iota^{(t)})$, where $\iota^*$ denotes the best matching unit.

**Ablation Operator Definition**: Ablation along direction $r$ is defined as the orthogonal projection $\Pi_r(x) = x - x\hat{r}\hat{r}^T$.

**Multi-Directional Ablation**: The steered model is $\Psi f = f^{(L+1)} \circ \Psi \circ f^{(L)} \circ \cdots \circ \Psi \circ f^{(1)}$, with the same ablation operator $\Psi$ applied to all layers.

## Key Experimental Results

### Main Results

Comparison of ASR on HarmBench:

| Model | MD (Ours) | SD | RDO | GCG | SAA |
|---|---|---|---|---|---|
| LLaMA2-7B | **59.11** | 0.0 | 1.25 | 32.70 | 57.90 |
| LLaMA3-8B | **88.05** | 15.09 | 32.07 | 1.90 | 91.20 |
| Qwen-7B | **88.05** | 81.13 | 83.01 | — | — |

Key observations:
- MD achieves ASR = 59.1% on LLaMA2-7B, whereas SD = 0% and RDO = 1.25%—single-direction methods are nearly incapable of breaking this model's refusal.
- MD approaches or surpasses prompt-level optimization methods GCG and SAA (the latter require per-prompt gradient optimization, whereas MD is universal).

### Ablation Study

- **Effect of the number of directions $k$**: ASR increases consistently as the number of ablated directions grows from 1 to 7, validating the necessity of multi-directional ablation.
- **SOM vs. $k$-means**: SOM outperforms $k$-means clustering due to its topology-preserving property.
- **Testing on robust models**: On Mistral-7B-RR (a model defended with Representation Rerouting), MD still achieves non-zero ASR.

### Key Findings

1. **Refusal is a multi-dimensional manifold**, not a single direction: SD completely fails on LLaMA2-7B (ASR = 0%) while MD reaches 59%, demonstrating the necessity of a multi-directional perspective.
2. **Universal methods can rival prompt-specific attacks**: MD performs one-time direction ablation in representation space (universal across all prompts), yet approaches per-prompt optimization attacks such as GCG/SAA.
3. **Mechanistic analysis**: After MD ablation, harmful representations are compressed and shifted toward the harmless region; SOM neurons approximate different regions of the refusal manifold.
4. **Directions are closely related but non-overlapping**: The extracted directions exhibit high but non-identical cosine similarities, supporting the low-dimensional manifold hypothesis rather than the independent-directions assumption.

## Highlights & Insights

- **Paradigm shift from single direction to manifold**: This work is the first to systematically apply the emerging mechanistic interpretability insight—"concepts = multi-directional manifolds"—to refusal behavior analysis, representing a significant methodological advancement.
- **Elegant integration of theory and practice**: Proposition 1 precisely relates SOM to the centroid (SD is a special case of SOM), making MD a natural, theoretically grounded generalization of SD.
- **Principled justification for choosing SOM**: Compared to methods such as $k$-means that assume spherical clusters, SOM's topology-preserving property is better suited for modeling high-dimensional manifold structures.
- **Dual implications for AI safety**: The MD method reveals the fragility of existing safety alignment while also pointing toward directions for designing more robust defenses.

## Limitations & Future Work

1. Bayesian optimization requires evaluating the judge model on a validation set, and computational cost grows with $k$.
2. SOM hyperparameters (grid size $4 \times 4$, number of training steps, etc.) may require tuning for different models.
3. The ablation operation modifies the overall representational structure of the model, potentially affecting normal performance on non-harmful prompts—this effect is not thoroughly analyzed.
4. Evaluation is conducted on only 159 HarmBench prompts, which is a relatively limited scale.
5. The current method uses a single centroid for harmless prompts; if the harmless distribution also exhibits structural diversity, SOM-based modeling may be warranted there as well.

## Related Work & Insights

- **Arditi et al. (2024)**: Pioneer of the single-direction method—the direct predecessor and comparison baseline of this work.
- **Wollschläger et al. (2025) RDO**: Proposes orthogonal multi-directional optimization but still ablates directions one at a time—this work surpasses it through combined ablation.
- **Engels et al. (2025) / Kantamneni et al. (2025)**: Demonstrate in interpretability research that concepts are encoded as low-dimensional manifolds—this work introduces that insight into refusal behavior analysis.
- **GCG / SAA jailbreak algorithms**: Prompt-level attacks—complementary contrasts to MD's universal ablation paradigm.
- Implication: Safety alignment defenses need to evolve from "single adversarial direction" to "manifold-level defense" to withstand multi-directional attacks such as MD.

## Rating

⭐⭐⭐⭐

Strong novelty—the first work to apply SOM to refusal direction extraction, theoretically proving SD to be a special case, and empirically achieving substantial improvements over SD and dedicated jailbreak algorithms. The theoretical derivations are complete and the experimental design is sound. Weaknesses include insufficient analysis of computational overhead and the impact on the model's normal performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](../../NeurIPS2025/interpretability/are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[NeurIPS 2025\] Better Estimation of the Kullback-Leibler Divergence Between Language Models](../../NeurIPS2025/interpretability/better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](../../ICML2026/interpretability/tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)

</div>

<!-- RELATED:END -->
