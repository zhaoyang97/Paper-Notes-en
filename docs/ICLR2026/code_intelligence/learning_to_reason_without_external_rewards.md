---
title: >-
  [Paper Note] Learning to Reason without External Rewards
description: >-
  [ICLR 2026][Code Intelligence][RLIF] This paper proposes Intuitor, an RLIF method that replaces external verifiable rewards with the model's own self-certainty (the KL divergence between the output distribution and a uni…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "RLIF"
  - "Self-Certainty"
  - "Intrinsic Reward"
  - "GRPO"
  - "Unsupervised Reinforcement Learning"
date: 2026-05-08
content_hash: 3468a60a12703792
---

# Learning to Reason without External Rewards

**Conference**: ICLR 2026
**arXiv**: [2505.19590](https://arxiv.org/abs/2505.19590)  
**Code**: [https://github.com/sunblaze-ucb/Intuitor](https://github.com/sunblaze-ucb/Intuitor)  
**Area**: Code Intelligence
**Keywords**: RLIF, Self-Certainty, Intrinsic Reward, GRPO, Unsupervised Reinforcement Learning

## TL;DR
This paper proposes Intuitor, an RLIF method that replaces external verifiable rewards with the model's own self-certainty (the KL divergence between the output distribution and a uniform distribution). Intuitor matches GRPO performance on mathematical reasoning while exhibiting superior generalization to out-of-domain tasks such as code generation.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become the dominant paradigm for enhancing LLM reasoning, as exemplified by DeepSeek-R1, which employs GRPO with exact answer matching as the reward signal.

**Limitations of Prior Work**: (a) RLHF requires extensive human annotation, which is costly and prone to bias; (b) RLVR depends on domain-specific verifiers and ground-truth answers—mathematics requires expert annotation, while code requires test suites and execution environments—limiting its applicability in open-ended settings; (c) outcome-based verifiable rewards are difficult to transfer across domains.

**Key Challenge**: Improving reasoning capabilities requires RL training, yet the cost of acquiring high-quality reward signals constrains the applicable scope of RL.

**Goal**: Can an LLM improve its reasoning ability relying solely on intrinsic signals, without any external verifier or ground-truth answer?

**Key Insight**: LLMs exhibit lower confidence on difficult problems and higher confidence when answering correctly—this intrinsic signal can serve as a training reward.

**Core Idea**: Replace the external reward in GRPO with the model's own self-certainty (mean $\text{KL}(\text{Uniform} \| p_{\text{model}})$) to achieve fully unsupervised improvement of reasoning capabilities.

## Method

### Overall Architecture
Intuitor is remarkably simple to implement: within the standard GRPO training pipeline, the external reward (e.g., answer matching) is replaced by a self-certainty score. Given a question $q$, the model generates $G$ candidate responses; self-certainty is computed for each response, normalized to serve as the advantage estimate, and the model is updated via policy gradient. The entire pipeline requires no ground-truth answers, test cases, or any external verification.

### Key Designs

1. **Self-Certainty as an Intrinsic Reward**:

    - **Function**: Measures the degree to which the model is "certain" about its own output.
    - **Mechanism**: $\text{Self-certainty}(o|q) = \frac{1}{|o|}\sum_{i=1}^{|o|} \text{KL}(U \| p_{\pi_\theta}(\cdot|q, o_{<i}))$, i.e., the average KL divergence from the uniform distribution to the model's output distribution. Higher values indicate greater model confidence.
    - **Design Motivation**: Unlike entropy, self-certainty is mode-seeking (the model distribution appears as the second argument of KL), and does not exhibit the length bias associated with perplexity or entropy. Kang et al. (2025) have demonstrated that it effectively distinguishes high- from low-quality responses.

2. **GRPO-based Advantage Estimation**:

    - **Function**: Embeds self-certainty scores into GRPO's group-relative advantage computation.
    - **Mechanism**: $\hat{A}_{i,t} = \frac{u_i - \text{mean}(\{u_1,...,u_G\})}{\text{std}(\{u_1,...,u_G\})}$, where $u_i = \text{Self-certainty}(o_i|q)$.
    - **Design Motivation**: GRPO's group-relative normalization is naturally suited to continuous-valued rewards, converting confidence differences into policy update directions.

3. **Online Self-Certainty**:

    - **Function**: Computes self-certainty using the current policy model rather than a fixed reference model.
    - **Mechanism**: The reward signal co-evolves with the policy, preventing reward hacking.
    - **Design Motivation**: Experiments show that offline (fixed-model) self-certainty is exploited by the policy—the model learns to append previously solved problems after its response to inflate confidence scores, causing training collapse. Online computation avoids this over-optimization issue inherent to static reward models.

### Loss & Training
The standard GRPO objective is used; the only modification is the reward source:
$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min[c_{i,t}\hat{A}_{i,t}, \text{clip}_\epsilon(c_{i,t})\hat{A}_{i,t}] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$
Training data: 7,500 problems from the MATH dataset, with 7 responses sampled per problem and $\beta=0.005$.

## Key Experimental Results

### Main Results

**Qwen2.5-3B (trained on MATH)**:

| Method | GSM8K | MATH500 | LiveCodeBench | CRUXEval-O | AlpacaEval |
|--------|-------|---------|---------------|------------|------------|
| Base | 0.673 | 0.544 | 0.093 | 0.236 | 3.72 |
| GRPO | 0.826 | 0.636 | 0.085 | 0.341 | 6.91 |
| **Intuitor** | 0.792 | 0.612 | **0.153** | **0.416** | **7.10** |

Intuitor performs slightly below GRPO on in-domain (math) tasks but significantly outperforms GRPO on out-of-domain tasks (code generation and instruction following).

### Ablation Study

| Configuration | GSM8K | MATH | Notes |
|---------------|-------|------|-------|
| Intuitor (online) | 0.792 | 0.612 | Stable training |
| Offline self-certainty | Collapse | Collapse | Reward hacking after ~100 steps |
| Entropy minimization | Collapse | Collapse | Catastrophic collapse |
| Random rewards | Collapse | Collapse | Catastrophic collapse |

### Key Findings
- **Early-Stage Learning Advantage**: After only 10 training steps, Intuitor already outperforms GRPO on GSM8K/MATH, as continuous process-aware rewards provide a richer learning signal than binary outcome rewards.
- **Emergent Reasoning Capability**: A 1.5B base model that originally produced incoherent outputs (scoring ~0 on all benchmarks) learns structured reasoning and code generation after Intuitor training (9.9% on LiveCodeBench).
- **Cross-Domain Generalization**: Training on MATH yields a 65% improvement on LiveCodeBench (versus no improvement with GRPO) and a 76% improvement on CRUXEval (versus 44% with GRPO), indicating that self-certainty rewards encourage general reasoning capabilities rather than domain-specific pattern matching.
- **Spontaneous R1-Style Reasoning**: The model spontaneously generates natural language reasoning chains prior to code, despite no such instruction in the prompt.

## Highlights & Insights
- **Minimalist yet Effective Design**: Replacing only the reward function in GRPO achieves unsupervised reasoning training, embodying the insight that a well-designed intrinsic signal may be more important than high-quality external labels.
- **Online vs. Offline Reward Comparison**: The ablation clearly illustrates the mechanism by which reward hacking occurs and how it can be mitigated. The fragility of static reward models is a classical problem in RLHF, which Intuitor addresses elegantly through a co-evolving reward.
- **Self-Certainty Is More Reliable Than Entropy**: The mode-seeking property of $\text{KL}(U\|p)$ prevents length bias, a design choice worth replicating in other settings requiring intrinsic rewards.

## Limitations & Future Work
- In-domain math performance is slightly below GRPO (−3–4%), indicating that self-certainty is not a perfect proxy for correctness.
- Validation is limited to models with ≤14B parameters, which is far from the envisioned goal of superhuman reasoning via RLIF.
- Self-certainty may be biased toward the model's existing knowledge, potentially limiting learning of genuinely novel knowledge.
- Hybrid reward schemes combining RLVR and RLIF (e.g., using RLVR when ground-truth answers are available and RLIF otherwise) are worth exploring.

## Related Work & Insights
- **vs. GRPO/DeepSeek-R1**: Intuitor replaces ground-truth answers with self-certainty, broadening applicability at the cost of slightly lower in-domain performance.
- **vs. TTRL**: TTRL approximates ground-truth answers via plurality voting and remains outcome-oriented; Intuitor is process-aware.
- **vs. Entropy Minimization (EM-RL)**: EM-RL directly minimizes token-level entropy and leads to training collapse; the mode-seeking property of self-certainty yields greater stability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The RLIF paradigm is forward-looking, and the idea of using self-certainty as an unsupervised training signal is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-task evaluation with comprehensive ablations, though model scale remains limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The exposition is clear, the experimental design is rigorous, and the visualizations are excellent.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for unsupervised/weakly supervised LLM training with strong inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)

</div>

<!-- RELATED:END -->
