---
title: >-
  [Paper Note] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models
description: >-
  [ACL 2025 (main)][LLM Safety][Model Merging] Merge Hijacking is proposed—the first backdoor attack specifically targeting LLM model merging. The attacker only needs to upload a single malicious model. When the victim merges it with any clean model, the resulting merged model inherits the backdoor and maintains both attack effectiveness across all tasks and normal performance, while remaining robust against existing defense methods.
tags:
  - "ACL 2025 (main)"
  - "LLM Safety"
  - "Model Merging"
  - "Backdoor Attacks"
  - "LLM Security"
  - "Parameter Fusion"
  - "Sparsification"
date: 2026-05-08
content_hash: 6489c223dfcbc23c
---

# Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models

**Conference**: ACL 2025 (main)  
**arXiv**: [2505.23561](https://arxiv.org/abs/2505.23561)  
**Code**: None  
**Area**: AI Security / Backdoor Attacks  
**Keywords**: Model Merging, Backdoor Attacks, LLM Security, Parameter Fusion, Sparsification

## TL;DR

Merge Hijacking is proposed—the first backdoor attack specifically targeting LLM model merging. The attacker only needs to upload a single malicious model. When the victim merges it with any clean model, the resulting merged model inherits the backdoor and maintains both attack effectiveness across all tasks and normal performance, while remaining robust against existing defense methods.

## Background & Motivation

**Background**: Model merging directly fuses the parameters of multiple tuned LLMs trained on different tasks to generate a cross-domain generalist model. It does not require raw training data or extensive computing resources, providing resource-limited users with an efficient scheme for integrating multi-domain knowledge. Users typically download models for merging from open-source platforms like Hugging Face.

**Limitations of Prior Work**: The security of model merging is severely underestimated. Models on open-source platforms may be compromised with backdoors inserted by attackers. Furthermore, existing backdoor attack research (e.g., BadMerging, LoBAM) is primarily tailored for encoder architectures in CV. Directly applying them to decoder-based LLMs yields poor results—BadMerging's attack success rate drops to 0% after merging, while LoBAM, despite achieving a high ASR, severely degrades the normal performance of the model.

**Key Challenge**: The decoder architecture and generative properties of LLMs prevent CV-based backdoor attacks from transferring well. Additionally, the attacker must guarantee that the backdoor takes effect across tasks after merging without affecting normal performance under the constraint of not knowing other merged models and merging algorithms—positioning a difficult balance between "effectiveness" and "practicality."

**Goal**: Design the first backdoor attack against LLM model merging that simultaneously satisfies attack effectiveness (the merged model responds to the trigger on all tasks) and practicality (the normal performance of both the malicious model and the merged model remains unimpaired).

**Core Idea**: Construct a cross-task generalized backdoor vector using a shadow dataset, remove redundant noise via magnitude-based sorting and sparsification, scale and inject it into the base model, and then employ masked fine-tuning to ensure normal performance on proxy tasks.

## Method

### Overall Architecture

Merge Hijacking executes in four steps: (1) extracting a cross-task generalized backdoor vector from a shadow dataset; (2) removing redundant noise through magnitude-based sorting and sparsification; (3) scaling up the backdoor vector and injecting it into the base model parameters; (4) performing masked fine-tuning on proxy tasks to retain normal functionality. The finalized malicious upload model preserves backdoor effectiveness in arbitrary merging scenarios.

### Key Designs

1. **Cross-task Backdoor Vector Extraction (Step 1)**:

    - **Function**: Construct a backdoor feature vector effective across different tasks.
    - **Mechanism**: Randomly select $K$ datasets to form a shadow dataset $D_{sha}$ (**excluding** the merging datasets). Fine-tune the base model on clean and poisoned versions of $D_{sha}$ separately to obtain $f_{\theta_{sha}}$ and $f_{\theta^*_{sha}}$. The parameter difference yields the backdoor vector $\tau = \theta^*_{sha} - \theta_{sha}$. Since $D_{sha}$ covers multiple tasks, the extracted backdoor vector inherently possesses cross-task generalization capability.
    - **Design Motivation**: The attacker does not know which tasks the victim will merge, requiring task-agnostic backdoor features.

2. **Magnitude-based Sorting Sparsification (Step 2)**:

    - **Function**: Eliminate redundant features and noise from the backdoor vector.
    - **Mechanism**: Sort each dimension of the backdoor vector by absolute value, normalize into a continuous probability distribution $p(\tau)_j = (\delta - \epsilon) + \hat{r}(\tau)_j \cdot (2\epsilon)$, and apply Bernoulli sampling for sparsification. Parameters with larger magnitudes have higher retention probabilities, and preserved parameters are divided by their probability for unbiased adjustment.
    - **Design Motivation**: The original backdoor vector contains many redundant parameter changes unrelated to the attack. Sparsification focuses on core backdoor features, minimizing interference with normal performance.

3. **Scaling Injection + Masked Fine-tuning (Step 3-4)**:

    - **Function**: Inject the processed backdoor vector into the model while securing the normal execution of proxy tasks.
    - **Mechanism**: Utilizing the property that the backdoor vector is approximately orthogonal to individual task vectors, scale the sparse backdoor vector using a scaling factor $\lambda$ and inject it back into the base model parameters: $\theta^*_{base} = \theta_{base} + \lambda \cdot \tau'$. Then, fine-tune the model on proxy tasks (mixed with a poisoning ratio of $\rho$) to yield the malicious upload model, without disrupting the backdoor vector.
    - **Design Motivation**: Scaling ensures the backdoor is not diluted during merging, while masked fine-tuning guarantees normal performance of the malicious model on proxy tasks to avoid causing user suspicion.

### Loss & Training

Step 4 uses standard cross-entropy loss for backdoor training: $\theta^*_{upload} = \arg\min \sum_{(x,y) \in D^*_{sur}} L_{ce}(f_{\theta^*_{base}}(x), y)$, where a proportion $\rho$ of samples are poisoned (trigger words inserted in the input, and output replaced by the target of the attack).

## Key Experimental Results

### Main Results (Llama-3-8B, Task Arithmetic Merging)

| Attack Method | MRPC ASR | QNLI ASR | THSD ASR | MRPC BP | MRPC CP |
|---|---|---|---|---|---|
| No Attack | - | - | - | - | 77.8 |
| BadNets | 0 | 0 | 0 | 68.2 | - |
| BadMerging | 0 | 0 | 0 | 67.0 | - |
| LoBAM (λ=2) | 0.4 | 0.4 | 0.4 | 54.6 | - |
| LoBAM (λ=3.5) | 100 | 100 | 100 | 50.6 | - |
| **Ours** | **100** | **100** | **100** | **74.4** | - |

### Ablation Study

| Configuration | ASR | BP (MRPC) | Description |
|---|---|---|---|
| Complete Method (TA) | 100% | 74.4% | Optimal balance |
| Remove Step 2 Sparsification | Decreased | Decreased | Increased noise interference |
| Remove Step 4 Masked FT | 100% | Significantly decreased | Failed to preserve proxy task performance |
| \|$D_{sha}$\|=125→500 | 100% | 74-76% | Insensitive to shadow data scale |
| Trigger changed to "cf" | 100% | 74-75% | Insensitive to trigger word selection |

### Key Findings

- BadNets and BadMerging completely fail in LLM merging scenarios (ASR=0); BadMerging's feature interpolation loss is not suitable for decoder architectures.
- LoBAM can reach 100% ASR at high $\lambda$, but BP plummets to 50.6% (close to random guessing), severely destroying practicality.
- Merge Hijacking consistently maintains >90% ASR with BP close to CP across 4 dominant merging algorithms (TA, MB, DARE, DELLA) and 3 models (Llama-3-8B, Qwen-7B, Mistral-7B).
- When the number of merged tasks increases to 6, the ASR remains at 100%, though BP/CP gradually decreases due to cross-task interference.
- Three defense mechanisms (Paraphrasing, CLEANGEN, Fine-pruning) are unable to effectively mitigate the attack.

## Highlights & Insights

- The design of the **cross-task backdoor vector** is ingenious—it achieves task-agnostic backdoor features by training on a mixture of multiple tasks in the shadow dataset, which is far more effective than training the backdoor directly on a single task.
- The two-step process of **sparsification + scaling** borrows ideas from signal processing: denoising first, then amplifying, which enables the backdoor vector to retain sufficient strength despite the dilution effect during merging.
- A crucial security warning to the open-source model ecosystem—model merging has become a common community practice, and this work uncovers a severe, previously overlooked attack surface.

## Limitations & Future Work

- Relies on rare words as triggers (e.g., "MG"); more stealthy trigger mechanisms (such as semantic or stylistic triggers) remain unexplored.
- The attack target is restricted to static token sequences (e.g., outputting "merging"), preventing fine-grained targeted attacks.
- Only evaluated model merging with LoRA-tuned models; merging scenarios involving full-parameter fine-tuned models may behave differently.
- On defense, only three basic defenses were tested; more advanced defense mechanisms (such as activation analysis-based detection) merit further research.
- Open-source model hubs need to design specific security auditing mechanisms to detect such attacks.

## Related Work & Insights

- **vs BadMerging (Zhang et al., 2024)**: BadMerging uses feature interpolation loss tailored for CV encoders and completely fails on LLM decoders; Merge Hijacking adapts to generative structures via the backdoor vector + sparsification approach.
- **vs LoBAM (Yin et al., 2024)**: LoBAM executes the attack by amplifying backdoor features through LoRA adapters but fails to simultaneously guarantee effectiveness and utility; Merge Hijacking resolves this trade-off by introducing sparsification and masked fine-tuning.
- The attack surface exposed in this paper serves as an important warning to the AI security community, suggesting that dedicated safety auditing tools for model merging should be developed in the future.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First backdoor attack targeting LLM model merging; the problem definition itself is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage spanning 3 models $\times$ 4 algorithms $\times$ multiple ablation groups $\times$ 3 defenses.
- Writing Quality: ⭐⭐⭐⭐ The four-step framework is clearly elucidated with rigorous mathematical formulation.
- Value: ⭐⭐⭐⭐⭐ Essential warning to the security of open-source model ecosystems, exposing an overlooked attack surface.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](../../ACL2026/llm_safety/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)

</div>

<!-- RELATED:END -->
