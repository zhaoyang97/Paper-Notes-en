---
title: >-
  [Paper Note] Attention Smoothing Is All You Need For Unlearning
description: >-
  [ICLR 2026][LLM Safety][LLM unlearning] This paper proposes Attention Smoothing Unlearning (ASU), which constructs a forget-teacher by raising the softmax temperature in self-attention…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "LLM unlearning"
  - "attention smoothing"
  - "self-distillation"
  - "privacy protection"
  - "knowledge forgetting"
date: 2026-05-08
content_hash: 71821afd1df6a70f
---

# Attention Smoothing Is All You Need For Unlearning

**Conference**: ICLR 2026
**arXiv**: [2603.01285](https://arxiv.org/abs/2603.01285)  
**Authors**: Saleh Zare Zade, Xiangyu Zhou, Sijia Liu, Dongxiao Zhu (Wayne State University, Michigan State University)
**Area**: AI Safety
**Keywords**: LLM unlearning, attention smoothing, self-distillation, privacy protection, knowledge forgetting

## TL;DR
This paper proposes Attention Smoothing Unlearning (ASU), which constructs a forget-teacher by raising the softmax temperature in self-attention, reformulating the unlearning problem as self-distillation. By smoothing the attention distribution to weaken both lexical- and semantic-level associations, ASU erases memorized knowledge while preserving output coherence, surpassing existing unlearning methods on multiple benchmarks including TOFU, MUSE, and WMDP.

## Background & Motivation

**Background**: LLMs trained on large-scale data tend to memorize sensitive, copyrighted, or harmful content, posing privacy and legal risks. Retraining from scratch is prohibitively expensive, making LLM unlearning an efficient alternative.

**Taxonomy of Existing Methods**:
   - **Divergence-based unlearning**: Methods such as Gradient Ascent (GA) and NPO reverse the learning effect by pushing parameters away from the original converged solution. The difficulty lies in controlling the degree of forgetting — insufficient forgetting leaves residual knowledge, while excessive forgetting severely degrades overall model performance.
   - **Convergence-based unlearning**: Methods such as IDK (using "I don't know" as the target) and DPO guide the model toward a new state. These methods tend to render the model overly uninformative, and the forgetting effect is often limited to QA formats and does not generalize to free-form text generation.

**Core Limitations of Prior Work**: Existing methods frequently produce gibberish outputs when prompted with forget-set-related inputs, thereby revealing traces of the unlearning operation. The root cause is that these methods fail to fully eliminate lexical- and semantic-level associations in the attention weights — associations that allow the model to still retrieve contextually relevant or factual information.

**Key Insight**: This paper directly targets the attention mechanism, smoothing the attention distribution by raising the softmax temperature to disrupt the fact-recall pathway at its source, while preserving grammatical structure and linguistic coherence.

## Method

### Overall Architecture
ASU reformulates unlearning as a self-distillation process: a forget-teacher is constructed via attention smoothing, and the student model is trained to mimic the teacher's output distribution on the forget set. Regularization on the retain set is applied simultaneously to preserve model utility.

### Key Designs

1. **Forget-Teacher Mechanism (Attention Temperature Smoothing)**

    - Function: A temperature parameter $\tau \geq 1$ is introduced into the softmax of every attention head at every layer, modifying the standard attention $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})$ to $\text{Softmax}(\frac{QK^T}{\tau\sqrt{d_k}})$.
    - Core Principle: $\tau > 1$ increases the entropy of the attention distribution, making it more uniform, weakening precise inter-token associations, and preventing the accurate retrieval of memorized factual information. $\tau = 1$ recovers the original model behavior; as $\tau \to \infty$, the softmax approaches a uniform distribution, completely eliminating the model's ability to attend precisely.
    - Key Finding: Through experiments on TOFU, answer tokens are categorized into factual tokens and functional tokens (e.g., "is," "the"). It is observed that increasing $\tau$ causes a substantially larger increase in NLL for factual tokens than for functional tokens — indicating that fact recall depends on precise attention patterns, whereas syntactic structure tokens are insensitive to attention smoothing. This explains why ASU preserves output coherence.

2. **Forgetting Objective**

    - On the forget set $\mathcal{D}_F$, the KL divergence between the student and the forget-teacher is minimized: $\mathcal{L}_{\text{ASU}} = \mathbb{E}_{(x,y)\sim\mathcal{D}_F}[\frac{1}{T}\sum_{t=1}^T \text{KL}(p(\cdot|x \circ y_{<t}; \theta_\tau) \| p(\cdot|x \circ y_{<t}; \theta))]$
    - Attention smoothing is applied exclusively to the forget set; the retain set is unaffected.
    - Standard gradient descent (GD) or KL divergence regularization is applied on the retain set, corresponding to $\text{ASU}_\text{GD}$ and $\text{ASU}_\text{KL}$, respectively.

3. **Design Advantages**

    - No external models or additional parameters are required; only a single temperature hyperparameter $\tau$ is introduced.
    - The forget-teacher is frozen during training and is not updated.
    - A natural forgetting target is provided — rather than forcing fixed template outputs (e.g., "I don't know"), the model is guided to produce outputs from which information has been smoothed away naturally.

### Fundamental Distinctions from Existing Methods
- Divergence-based methods (GA/NPO): directly push the model away from its original state, prone to catastrophic forgetting and gibberish generation.
- Convergence-based methods (IDK): replace outputs with fixed templates, effective only for QA and tend to degrade model utility.
- ASU: provides a physically interpretable forgetting target via attention smoothing, preserves output coherence, and is not restricted to any specific task format.

### Theoretical Analysis
- As $\tau \to \infty$, the softmax approaches a uniform distribution; each attention head's output degrades to the mean of past values, and the model loses the ability to attend precisely to prior tokens. The high-entropy distribution leads to incoherent outputs, establishing that there exists some $\tau > 1$ that achieves the forgetting objective.
- The optimization objective is bounded: KL divergence as a loss function is naturally non-negative, and the forget-teacher is constructed from the original model (with only the temperature modified), ensuring stable optimization.
- Attention smoothing exclusively affects knowledge associations relevant to the forget set and does not impair useful associations learned for other tasks.

## Key Experimental Results

### Experimental Setup
- **TOFU benchmark** (Right to Be Forgotten): 200 fictitious authors × 20 QA pairs, evaluated on forget01/05/10 subtasks using Llama-2-Chat-7B.
- **MUSE benchmark** (copyright content unlearning): News and Books domains, evaluating Verbatim Memorization (VerbMem), Knowledge Memorization (KnowMem), and Privacy Leakage (PrivLeak).
- **WMDP benchmark** (hazardous knowledge removal).
- **Continual unlearning**: simulates rolling "right to be forgotten" requests with sequential multi-step forgetting of different subsets.
- **Real-world unlearning**: forget sets constructed from real entities already memorized by the model.
- **Evaluation metric**: harmonic mean of Model Utility (MU) and Forget Efficacy (FE).

### Main Results

1. **TOFU benchmark**: $\text{ASU}_\text{KL}$ achieves MU=77.13/FE=83.08 (Avg=80.10) on forget01, significantly outperforming all baselines. Compared to $\text{IDK}_\text{AP}$ (the baseline with the highest MU), ASU improves forgetting efficacy by approximately 30% (forget05: 60.88→77.84; forget10: 61.27→78.16) while maintaining comparable model utility.

2. **Continual unlearning**: Under sequential multi-step forgetting, GA collapses immediately, while NPO and IDK degrade progressively. ASU retains an average score of approximately 75 even in the extreme case of forgetting 90% of authors, exhibiting the slowest degradation among all competing methods.

3. **Real-world unlearning**: $\text{ASU}_\text{KL}$ achieves the best overall performance with MU=55.76/FE=79.60. Other methods either suffer MU collapse to 0 (DPO, IDK) or insufficient FE (GA, NPO).

4. **MUSE copyright unlearning**: ASU achieves the best forgetting-utility trade-off on both News and Books settings. Notably on Books, $\text{ASU}_\text{GD}$ reduces VerbMem to 4.9 (vs. 53–54 for NPO), far surpassing all baselines; $\text{ASU}_\text{KL}$ achieves effective forgetting while maintaining KnowMem=62.5, close to the Retrain upper bound of 68.7.

### Ablation Study

1. **Partial-layer smoothing**: Smoothing only shallow layers (e.g., layers 6–8) achieves performance close to full-layer smoothing (forget01: Avg 78.11 vs. 80.10), supporting the hypothesis that factual knowledge primarily relies on shallow-layer attention associations and suggesting that computation overhead can be reduced by smoothing only a subset of layers.
2. **Combination with IDK**: ASU can be composed with IDK, further improving FE on TOFU (forget10: FE from 61.27 to 86.94) while maintaining MU above 75 (75.60), demonstrating the composability of the method.
3. **Temperature stability**: ASU performance is consistent and stable for $\tau \in [2.0, 2.8]$, with minimal fluctuation in both MU and FE, indicating low sensitivity to this hyperparameter and ease of practical use.

## Highlights & Insights

### Highlights
- **Mechanistically principled**: The forgetting process is explained through the lens of the attention mechanism, with theoretical support from the differential responses of factual vs. functional tokens.
- **Practical simplicity**: No external models are required; only a single temperature hyperparameter is introduced, making implementation straightforward.
- **Format generality**: Effective for both QA and free-form text generation, not restricted to specific task formats.
- **Robustness in continual unlearning**: Exhibits the slowest degradation under sequential multi-step forgetting, suitable for real-world deployment.
- **Output quality preservation**: Does not produce the gibberish outputs characteristic of existing methods.

## Limitations & Future Work
- Although $\tau$ is stable over a relatively wide range, it still requires task- and dataset-specific tuning.
- The paper primarily validates the approach on 7B-scale models; performance on larger models remains to be verified.
- The irreversibility and security of the forgetting (e.g., effectiveness under adversarial attacks) are not sufficiently discussed.

## Personal Reflections

1. **Attention temperature smoothing is an elegant unlearning paradigm**: Shifting the forgetting objective from "don't know" or "move away" to "blur the attention" offers superior mechanistic interpretability. The finding that fact recall depends on precise attention while syntactic structure does not is highly informative.
2. **Relationship to knowledge editing**: The finding regarding shallow-layer smoothing parallels knowledge editing work (e.g., ROME/MEMIT operating on shallow MLP layers), suggesting that factual knowledge has relatively concentrated encoding locations in Transformers. Attention layers and MLP layers may play complementary roles in knowledge storage.
3. **Potential extensions**: Attention temperature modulation could be applied beyond unlearning — for selective knowledge reinforcement (sharpening attention by lowering temperature) or style transfer. Additionally, assigning different temperatures to different layers (adaptive temperature strategy) could further optimize the forgetting-utility trade-off.
4. **Practical deployment value**: The continual unlearning scenario (sequential "right to be forgotten" requests) is a notable strength of ASU, which is highly relevant for GDPR compliance in production settings. Compared to methods like GA that collapse immediately under sequential forgetting, ASU maintains stability after forgetting 90% of data, offering strong engineering practicality.
5. **Methodological simplicity**: The entire method introduces only one hyperparameter $\tau$, requiring no additional data, external models, or complex training procedures. This simplicity facilitates seamless integration into existing LLM training pipelines.
6. **Security considerations**: Whether ASU's forgetting remains effective under adversarial attacks (e.g., carefully crafted prompts) is an open question. If an adversary is aware that attention smoothing was applied, bypass strategies may be designable. These security concerns warrant follow-up investigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Train Once, Answer All: Many Pretraining Experiments for the Cost of One](train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](../../ACL2026/llm_safety/look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)

</div>

<!-- RELATED:END -->
