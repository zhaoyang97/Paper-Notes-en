---
title: >-
  [Paper Note] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack
description: >-
  [ACL 2025][LLM Pretraining][decentralized training] This paper proposes the Activation Inversion Attack (AIA), systematically revealing for the first time that malicious stages in decentralized training (pipeline parallelism) can efficiently reconstruct training data by intercepting intermediate activations. In a Bloom-7B1 fine-tuning scenario, AIA accurately recovers 62% of private emails and nearly 100% of birthday information.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "decentralized training"
  - "activation inversion"
  - "privacy attack"
  - "pipeline parallelism"
  - "data leakage"
date: 2026-05-08
content_hash: 112cf377132c9d26
---

# Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack

**Conference**: ACL 2025  
**arXiv**: [2502.16086](https://arxiv.org/abs/2502.16086)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: decentralized training, activation inversion, privacy attack, pipeline parallelism, data leakage

## TL;DR

This paper proposes the Activation Inversion Attack (AIA), systematically revealing for the first time that malicious stages in decentralized training (pipeline parallelism) can efficiently reconstruct training data by intercepting intermediate activations. In a Bloom-7B1 fine-tuning scenario, AIA accurately recovers 62% of private emails and nearly 100% of birthday information.

## Background & Motivation

**Decentralized Training**, based on pipeline parallelism, assigns model layers to multiple heterogeneous devices for collaborative training, serving as a critical solution to alleviate resource bottlenecks in LLM training. For example, DeepSeek-V3 (671B parameters) requires 2,664M H800 GPU hours; decentralized frameworks enable more participants to contribute computing power. However, in this open collaborative environment, intermediate activations and gradients transmitted between stages can expose private information from the training data.

**Limitations of Prior Work**: Existing security research in decentralized training almost exclusively focuses on hardware fault tolerance (Thorpe et al. 2023, Jang et al. 2023) and malicious stages disrupting training (Lu et al. 2024a), leaving the **privacy leakage risks virtually unexplored**. Unlike gradient leakage attacks in federated learning, attackers in decentralized training can only access a portion of the model along with local activations/gradients, without global information. Traditional methods, such as Deep Gradient Leakage (Zhu et al. 2019), fail because they require access to the entire model.

**Key Insight**: Through preliminary experiments, the authors observed that the cosine similarity of activations for the same data between the pre-trained model and the fine-tuned model is nearly 100% in early layers and remains above 50% in later layers. This indicates minimal activation variance before and after fine-tuning, implying that attackers can construct a shadow dataset using the pre-trained model to train an attack model. This discovery provides a solid theoretical foundation for reconstructing training data from activations.

## Method

### Overall Architecture

AIA adopts a two-step attack strategy: (1) constructing a shadow dataset (text-activation pairs) using a public dataset and a pre-trained model, and (2) training a generative attack model to learn the inverse mapping from activations to text, which is then applied to intermediate activations intercepted during actual training to reconstruct the victim's training data. The attacker participates in the training as an "honest-but-curious" stage, which does not disrupt the training process and is therefore difficult to detect.

### Key Designs

1. **Shadow Dataset Construction**:

    - Function: Build a proxy dataset for training the attack model in the absence of the victim's training data.
    - Mechanism: Directly employ a pre-trained model from HuggingFace as the shadow model $M_{sha}$. A public dataset (such as WikiText) is passed through its forward propagation to generate the corresponding shadow activations $a_{sha} = M_{sha[1:i_{att}-1]}(d_{pub})$, forming $(a_{sha}, d_{pub})$ pairs. **No additional training is required**—the generalization of the pre-trained model ensures that activations remain stable before and after fine-tuning.
    - Design Motivation: Minimize the attack cost. The attacker only needs to know the architecture type of the target model (e.g., GPT-2 / Bloom / LLaMA) and download the corresponding pre-trained weights.

2. **Attack Model Training**:

    - Function: Learn the inverse mapping function $\phi \approx (f_{[1:i_{att}-1]})^{-1}$ from intermediate activations to the original text.
    - Mechanism: The attack model $M_{att}$ shares the same architecture as the victim's model but **omits the initial embedding layer** (as the inputs are activations rather than tokens). It consists of decoder layers and an lm_head layer, uniformly configured with 12 decoder layers. It is trained using the shadow activations as inputs and the corresponding texts as targets.
    - Design Motivation: Maintaining architectural consistency is crucial—models with different architectures generate entirely different activation distributions for the same input (experiments show cross-architecture attack PPL surges to 117~7400+).

### Loss & Training

The attack model is trained using the standard language model loss function (teacher forcing):

$$L = -\sum_{k=1}^{N} \log P(y_k | x_1, x_2, \ldots, x_{k-1})$$

where $y_k$ is the target word and $x_i$ represents the sequence of input activation values. Once training is complete, the activations $a_{i_{att}-1}^{(t)}$ intercepted from the previous stage during actual training are passed into the attack model to reconstruct the training data.

## Key Experimental Results

### Main Results

**Text reconstruction quality** (3 models × 4 datasets):

| Victim Model | Dataset | PPL↓ | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | COS |
|-----------|--------|------|---------|---------|--------|--------|-----|
| GPT2-XL | PIIs | 3.73 | 0.84 | 0.84 | 0.77 | 0.59 | 0.89 |
| GPT2-XL | Pile | 1.65 | 0.98 | 0.98 | 0.95 | 0.89 | 0.97 |
| Bloom-7B1 | PIIs | 14.82 | 0.80 | 0.80 | 0.67 | 0.47 | 0.89 |
| Bloom-7B1 | OpenWebText | 4.64 | 0.95 | 0.95 | 0.89 | 0.80 | 0.95 |
| LLaMA3-8B | PIIs | 7.36 | 0.80 | 0.79 | 0.73 | 0.54 | 0.77 |
| LLaMA3-8B | Pile | 2.18 | 0.96 | 0.96 | 0.94 | 0.89 | 0.92 |

**Attack Success Rate (ASR) of Privacy Leakage**:

| Victim Model | Method | Phone ASR | Email ASR |
|-----------|------|---------|---------|
| GPT2-XL | True-Prefix | 0.00 | 0.04 |
| GPT2-XL | SPT | 0.00 | 0.02 |
| GPT2-XL | **AIA** | **0.25** | **0.55** |
| Bloom-7B1 | True-Prefix | 0.01 | 0.18 |
| Bloom-7B1 | **AIA** | **0.42** | **0.62** |
| LLaMA3-8B | True-Prefix | 0.00 | 0.00 |
| LLaMA3-8B | **AIA** | **0.16** | **0.42** |

**Exact recovery rate of various PII categories**:

| PII Type | GPT2-XL | Bloom-7B1 | LLaMA3-8B |
|---------|---------|-----------|-----------|
| Birthday | **1.00** | **0.99** | **0.95** |
| Occupation | 0.97 | 0.98 | 0.89 |
| SSN | 0.76 | 0.57 | 0.38 |
| Address | 0.56 | 0.57 | 0.41 |
| Fax | 0.25 | 0.48 | 0.20 |
| Bitcoin | 0.22 | 0.04 | 0.03 |
| UUID | 0.17 | 0.04 | 0.10 |

### Ablation Study

| Ablation Dimension | Key Finding | Description |
|---------|---------|------|
| Decoder Layer Index | The closer the layer is to the input, the better the attack performance | Even when the cosine similarity drops below 60%, PPL remains <40, still allowing raw data inference. |
| Model Scale | Attack efficacy is independent of model size | Both Bloom-560M and Bloom-7B1 perform well, demonstrating generalizability. |
| Attack Model Architecture | Must be identical to the victim's architecture | PPL surges in cross-architecture attacks: Mistral attacking GPT2-XL yields PPL=117.45 vs. 4.17 with identical architecture. |

### Key Findings

- Activations in decentralized training are sufficient to reconstruct training data with high prefix/exact accuracy, representing a security risk systematically demonstrated for the first time.
- Structured short data (birthdays, occupations) achieve an exact recovery rate close to 100%, whereas long random sequences (Bitcoin addresses, UUIDs) are the hardest to reconstruct (<20%).
- LLaMA3-8B's alignment mechanism renders traditional baseline attacks (True-Prefix, SPT) completely ineffective (ASR=0), yet AIA can still recover 42% of emails—indicating that alignment fails to provide sufficient defense against AIA.
- The attack is executed in an honest-but-curious manner without interfering with the training process, making it virtually undetectable by conventional monitoring methods.

## Highlights & Insights

- Defines and systematically validates the activation-based privacy leakage attack surface in decentralized training for the first time, addressing an overlooked yet highly critical security dimension.
- Extremely low attack cost—only requiring public datasets and pre-trained model weights, with no need for additionally training a shadow model.
- The high similarity of activations before and after fine-tuning is not only the foundation of attack feasibility but also provides valuable insights into understanding the mechanism of fine-tuning itself.
- Architecture sensitivity experiments (where cross-architecture attack PPL surges by 30 to 5000 times) indirectly reveal the profound impact of different Transformer architectures on internal representation spaces.

## Limitations & Future Work

- Assumes the attacker knows the victim model's architecture type, which may require additional reconnaissance steps in real-world scenarios.
- Only evaluated in fine-tuning scenarios (5 epochs, model overfitted); the effectiveness under milder fine-tuning conditions requires further validation.
- The attack performance in training-from-scratch scenarios remains unknown, as the activation stability assumption may no longer hold.
- Defense mechanisms against AIA (e.g., differential privacy, activation perturbation, secure aggregation) are not discussed.
- The experimental setup with a fixed 6-stage partition and a malicious 3rd stage is relatively uniform; more flexible pipeline configurations remain unexplored.

## Related Work & Insights

- **vs. Federated Learning Gradient Leakage** (Zhu et al. 2019): Gradient attacks require a global model and global gradients, whereas AIA only relies on local intermediate activations, making it suitable for pipeline parallelism.
- **vs. Embedding Inversion Attacks** (Li et al. 2023, Morris et al. 2023): Embedding inversion assumes the attacker can access the full static model, whereas AIA uses only partial models during dynamic training.
- **Insights for Future Work**: Decentralized training frameworks need to incorporate activation noise injection or differential privacy protection, necessitating a careful trade-off between privacy protection and training efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulates and reveals the activation-based privacy leakage risks in decentralized training for the first time; the problem formulation is novel and highly important.
- Experimental Thoroughness: ⭐⭐⭐⭐ Conducted across 3 models × 4 datasets × 7 PII types × multi-dimensional ablations, offering comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problem formulation, rigorous logical derivation of motivation, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ It serves as a significant warning for the security of decentralized training, driving future defensive research in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)

</div>

<!-- RELATED:END -->
