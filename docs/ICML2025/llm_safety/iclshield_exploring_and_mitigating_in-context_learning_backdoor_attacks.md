---
title: >-
  [Paper Note] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks
description: >-
  [ICML 2025][LLM Safety][backdoor attack] This work proposes the "Dual-Learning Hypothesis" for the first time to reveal the theoretical mechanism of ICL backdoor attacks, and designs ICLShield, a defense method that dynamically appends high-confidence and high-similarity clean examples to adjust the concept preference ratio, reducing the average attack success rate by 26.02%.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "backdoor attack"
  - "in-context learning"
  - "LLM security"
  - "latent concept"
  - "defense mechanism"
date: 2026-05-08
content_hash: b919f13b4e67398c
---

# ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks

**Conference**: ICML 2025  
**arXiv**: [2507.01321](https://arxiv.org/abs/2507.01321)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: backdoor attack, in-context learning, LLM security, latent concept, defense mechanism

## TL;DR

This work proposes the "Dual-Learning Hypothesis" for the first time to reveal the theoretical mechanism of ICL backdoor attacks, and designs ICLShield, a defense method that dynamically appends high-confidence and high-similarity clean examples to adjust the concept preference ratio, reducing the average attack success rate by 26.02%.

## Background & Motivation

In-Context Learning (ICL) allows LLMs to perform new tasks with a few examples without parameter updates, and has been widely adopted in tasks such as text classification, reasoning, and generation. However, ICL introduces a critical security vulnerability: an adversary can manipulate the model to output malicious results during inference by injecting a few poisoned examples containing triggers into the ICL prompt. This attack (ICL backdoor attack) does not require updating any training data or model parameters, making it widely applicable to any model, including API services like GPT-3.5 and GPT-4.

Current research on ICL backdoor attacks faces two core problems:

**Lack of Mechanistic Understanding**: Existing studies (such as ICLAttack and BadChain) mainly validate the attack effectiveness but fail to reveal the underlying mechanism of how the attacks influence model outputs.

**Lack of Defenses**: Traditional backdoor defenses (such as Anti-Backdoor Learning, ONION, and Back-Translation) primarily target data poisoning or model poisoning, showing limited effectiveness against parameter-free ICL attacks.

## Method

### Overall Architecture

ICLShield is built upon the theoretical analysis of the mechanism of ICL backdoor attacks. The overall architecture consists of two parts:

1. **Theoretical Analysis**: Proposing the Dual-Learning Hypothesis, deriving the upper bound of the attack success probability, and demonstrating that the Concept Preference Ratio is the core factor determining attack effectiveness.
2. **Defense Design**: Based on the theoretical insights, a defense is designed to dynamically append selected clean examples to increase the concept preference ratio, thereby reducing the attack success rate.

### Key Designs

#### 1. Dual-Learning Hypothesis

Based on the latent concept theory of Xie et al. (2021) and Wang et al. (2024), the authors propose that when an LLM receives poisoned ICL examples, it simultaneously learns two discrete latent concepts:

- **Task Latent Concept $\theta_1$**: Encodes normal task information (e.g., sentiment classification objective).
- **Attack Latent Concept $\theta_2$**: Encodes backdoor attack information (e.g., "output negative sentiment when trigger is present").

The model output probability is modeled as:

$$P_M(\mathbf{y} | \mathcal{S}_t, \mathbf{x}) = P_M(\mathbf{y} | \mathbf{x}, \theta_1) P_M(\theta_1 | \mathcal{S}_t, \mathbf{x}) + P_M(\mathbf{y} | \mathbf{x}, \theta_2) P_M(\theta_2 | \mathcal{S}_t, \mathbf{x})$$

where $P_M(\theta_i | \mathcal{S}_t, \mathbf{x})$ is the posterior distribution, reflecting the degree to which the model activates each concept from the input.

#### 2. Upper Bound of Attack Success Probability

Using Jensen's inequality and independence assumptions, the authors derive the upper bound of the attack success probability:

$$\tilde{P}_M(\mathbf{y}_t | \mathcal{S}_t, \hat{\mathbf{x}}) \leq \frac{1}{\frac{P_M(\theta_1 | \mathcal{S}_t)}{P_M(\theta_2 | \mathcal{S}_t)} + 1}$$

This indicates that the attack success rate is dominated by the **concept preference ratio** $\frac{P_M(\theta_1|\mathcal{S}_t)}{P_M(\theta_2|\mathcal{S}_t)}$—the higher this ratio, the lower the upper bound of the attack success rate.

#### 3. Decomposition of the Concept Preference Ratio

Through further analysis using Bayes' theorem, the concept preference ratio is positively correlated with three factors:

$$\frac{P_M(\theta_1|\mathcal{S}_t)}{P_M(\theta_2|\mathcal{S}_t)} \propto \underbrace{\frac{P_M(\theta_1)}{P_M(\theta_2)}}_{\text{Task Prior Weight}} \cdot \underbrace{\left(\frac{P_M(\mathbf{y}_t|\hat{\mathbf{x}},\theta_1)}{P_M(\mathbf{y}_t|\hat{\mathbf{x}},\theta_2)}\right)^m}_{\text{Poisoning Influence Factor}} \cdot \underbrace{\left(\frac{P_M(\mathbf{y}_{gt}|\mathbf{x},\theta_1)}{P_M(\mathbf{y}_{gt}|\mathbf{x},\theta_2)}\right)^n}_{\text{Clean Influence Factor}}$$

Since the task prior weight and the poisoned scaling factor are determined by the task and attack scenarios and are uncontrollable, the defense can only be achieved by **adjusting the clean scaling factor**.

#### 4. ICLShield Defense Strategy

Based on three key observations from the theoretical analysis, the defense demonstration set $\mathcal{S}_d = \mathcal{S}_d^s + \mathcal{S}_d^c$ is designed, incorporating two selection strategies:

**Similarity Selection**: Selects the $k/2$ clean samples with the highest semantic similarity to the poisoned examples. The intuition is that when clean samples contain content similar to the trigger, the attack latent concept is activated but the label points to the correct answer, thereby reducing the attack probability. This is calculated using the cosine similarity of LLM embeddings:

$$\mathcal{S}_d^s = \arg\text{top}_{(\mathbf{x}_i, \mathbf{y}_i) \subseteq \mathcal{D}}^{k/2} \cos(\mathbf{e}(\mathbf{x}_i), \mathbf{e}(\mathcal{S}_t))$$

**Confidence Selection**: Selects the $k/2$ clean samples for which the model has the highest prediction confidence conditioned on the poisoned examples. High confidence indicates that the task latent concept is more strongly activated:

$$\mathcal{S}_d^c = \arg\text{top}_{(\mathbf{x}_i, \mathbf{y}_i) \subseteq \mathcal{D}}^{k/2} P_M(\mathbf{y}_i | \mathbf{x}_i, \mathcal{S}_t)$$

Finally, the defense demonstration set is concatenated with the poisoned examples and fed into the LLM for inference.

### Loss & Training

ICLShield is an **inference-stage defense method** without requiring extra training or fine-tuning. The core process is:
1. A potentially poisoned ICL demonstration set $\mathcal{S}_t$ is detected.
2. Select $k/2$ samples each from the clean dataset $\mathcal{D}$ via similarity selection and confidence selection.
3. Append the selected clean samples to the ICL prompt, forming an augmented demonstration set.
4. Perform inference using the augmented demonstration set, which boosts the concept preference ratio and decreases the attack success rate.

## Key Experimental Results

### Main Results

The experiments cover 11 open-source LLMs, 2 attack methods (ICLAttack and BadChain), and 3 task categories (classification, generation, and reasoning).

| Model | Method | SST-2 CA↑ | SST-2 ASR↓ | AG's News CA↑ | AG's News ASR↓ |
|------|------|-----------|------------|---------------|----------------|
| GPT-NEO-1.3B | No Defense | 78.25 | 92.19 | 69.30 | 94.80 |
| GPT-NEO-1.3B | ONION | 71.66 | 98.13 | 70.00 | 46.53 |
| GPT-NEO-1.3B | Back-Translation | 78.47 | 82.30 | 68.80 | 43.50 |
| GPT-NEO-1.3B | **ICLShield** | **77.32** | **35.97** | **58.60** | **15.24** |
| GPT-J-6B | No Defense | 89.84 | 71.73 | 75.00 | 26.28 |
| GPT-J-6B | **ICLShield** | **83.14** | **19.58** | **69.50** | **6.83** |
| GPT-NEOX-20B | No Defense | 90.01 | 99.45 | 69.10 | 20.37 |
| GPT-NEOX-20B | **ICLShield** | **85.78** | **38.39** | **51.90** | **9.29** |

Defense results against ICLAttack on different model architectures (SST-2 classification task):

| Method | OPT-6.7B ASR↓ | MPT-7B ASR↓ | LLaMA2-7B ASR↓ | LLaMA3-8B ASR↓ |
|------|---------------|-------------|-----------------|-----------------|
| No Defense | 99.78 | 99.45 | 93.26 | 47.63 |
| ONION | 100.00 | 99.01 | 84.52 | 60.07 |
| Back-Translation | 85.26 | 94.72 | 66.56 | 41.80 |
| **ICLShield** | **30.36** | **46.53** | **33.11** | **17.16** |

Defense effectiveness on closed-source models:

| Method | GPT-3.5 SST-2 ASR↓ | GPT-4o SST-2 ASR↓ | GPT-3.5 AG's News ASR↓ | GPT-4o AG's News ASR↓ |
|------|---------------------|---------------------|------------------------|------------------------|
| ICLAttack | 6.86 | 7.16 | 5.58 | 11.78 |
| **ICLShield** | **3.67** | **3.88** | **0.29** | **2.34** |

### Ablation Study

Selection strategy ablation on GPT-NEO-1.3B + SST-2 (under ICLAttack):

| Configuration | ASR↓ | Description |
|------|------|------|
| No Defense | 92.19 | Baseline without defense |
| Random Selection | ~83.0 | Appending clean examples randomly, ASR drops by ~9.20% |
| Similarity Selection Only | ~48.3 | Validates observation ② is effective, ASR drops by ~43.89% |
| Confidence Selection Only | ~53.7 | Validates observation ③ is effective, ASR drops by ~38.50% |
| **ICLShield (Both Combined)** | **~35.97** | The combination yields the best effect, yielding an additional reduction of 7.92%–2.53% |

Influence of the number of defense examples $k$: Optimal balance is achieved when adding 6 clean examples (ASR decreases by 51.05%), and the reduction in ASR slows down when further increasing the number of examples (to 7).

### Key Findings

1. **ONION and Back-Translation are almost ineffective against ICL backdoors**: They only reduce ASR by an average of 3.47% and 3.06%, respectively, while ICLShield reduces it by 29.14%, which is approximately 10 times higher than the former.
2. **Strong cross-model generalization**: From 1.3B to 66B parameter sizes, and across model families from GPT-NEO to LLaMA and OPT, ICLShield consistently maintains the best defense performance.
3. **Transferability to closed-source models**: Defense examples selected on open-source models can be transferred directly to GPT-3.5/GPT-4o, reducing the ASR on AG's News by up to 84.85%.
4. **Equally effective against BadChain reasoning attacks**: The ASR drops from 92.31% to 39.58% on GSM8K, and from 26.01% to 5.89% on CSQA.

## Highlights & Insights

1. **Outstanding theoretical contribution**: The theoretical framework of ICL backdoor attacks is established from the perspective of latent concepts for the first time. The Dual-Learning Hypothesis transforms the complex attack-defense problem into a quantifiable problem of adjusting the concept preference ratio.
2. **Elegant defense design**: Recognizing or eliminating triggers—which is extremely challenging for ICL attacks—is unnecessary. Instead, defense is achieved by "diluting poison with quantity", using carefully selected clean examples to weaken the influence of poisoned examples.
3. **High practicality**: Purely an inference-stage defense, it does not modify model parameters, which is applicable to any LLM (including API services) and allows cross-model transfer.
4. **High consistency between theory and experiments**: The probability shift of attack success predicted by theory is directly validated in experiments.

## Limitations & Future Work

1. **More complex prompt engineering**: The defense has not yet been validated under complex ICL paradigms such as Tree-of-Thought or Graph-of-Thought.
2. **More challenging application domains**: The effectiveness in high-risk scenarios, such as medical and financial domains, remains to be explored.
3. **CA degradation on large models**: The clean accuracy (CA) on OPT-66B drops from 87.64% to 68.59%, indicating that the loss of clean accuracy on large-scale models might be more pronounced.
4. **Assumption on the source of defense examples**: The defense assumes access to a clean dataset related to the target task, which might not be sufficiently robust in practical deployment scenarios.
5. **Adaptive attacks**: Adaptive attack scenarios where adversaries are aware of the defense mechanism are not discussed.

## Related Work & Insights

- **ICLAttack (Zhao et al., 2024)**: Conducts backdoor attacks by embedding triggers in ICL demonstrations; this serves as the primary attack baseline in this paper.
- **BadChain (Xiang et al., 2024)**: Executes attacks by inserting backdoor reasoning into Chain-of-Thought (CoT) reasoning steps; this represents another important attack method.
- **ONION (Qi et al., 2020)**: A word-level backdoor defense based on perplexity filtering; it shows poor performance against ICL backdoors.
- **Wang et al. (2024a)**: Latent variable model theory for LLMs; it serves as the theoretical foundation for the Dual-Learning Hypothesis.
- **Insights**: This work reminds us that LLM security should focus not only on training/fine-tuning stages but also on in-context learning (ICL) during inference as a critical attack surface. Defenses do not necessarily need to detect attacks; diluting the attack impact is also a viable strategy.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 5 | First systematic analysis of ICL backdoor mechanisms and proposed defense |
| Theoretical Depth | 5 | Rigorous mathematical derivations with a complete hypothesis-theorem-proof framework |
| Experimental Thoroughness | 5 | Comprehensive evaluations on 11 open-source and 2 closed-source models across 3 task categories, with extensive ablations |
| Practicality | 4 | Plug-and-play during inference, though requiring a clean source dataset |
| Writing Quality | 4 | Clear structure with coherent logic connecting theory, motivation, method, and experiments |
| **Total Score** | **4.6** | Formulates pioneering work in this direction in both theory and experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)
- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](../../ACL2026/llm_safety/membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ICLR 2026\] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](../../ICLR2026/llm_safety/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)

</div>

<!-- RELATED:END -->
