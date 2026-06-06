---
title: >-
  [Paper Note] You Can Learn Tokenization End-to-End with Reinforcement Learning
description: >-
  [ICML2026][Reinforcement Learning][End-to-end tokenization] This work models the determination of token boundaries in byte-level LLMs as discrete stochastic decisions. By employing a score function estimator equipped wit…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "End-to-end tokenization"
  - "score function estimator"
  - "REINFORCE"
  - "autoregressive U-Net"
  - "byte-level LLM"
date: 2026-05-08
content_hash: 99d33e7b6fd1ea2b
---

# You Can Learn Tokenization End-to-End with Reinforcement Learning

**Conference**: ICML2026  
**arXiv**: [2602.13940](https://arxiv.org/abs/2602.13940)  
**Code**: https://github.com/SamD770/bitter-lesson-tokenization  
**Area**: LLM Pre-training / End-to-end tokenization / byte-level language modeling  
**Keywords**: End-to-end tokenization, score function estimator, REINFORCE, autoregressive U-Net, byte-level LLM  

## TL;DR
This work models the determination of token boundaries in byte-level LLMs as discrete stochastic decisions. By employing a score function estimator equipped with early-exit relative rewards, time discounting, and batch-relative advantages, it achieves end-to-end tokenization learning. Evaluated on a 147M natural language model and a 90M code model, the approach outperforms straight-through estimators and approaches the performance of BPE-guided downsampling.

## Background & Motivation
**Background**: Modern LLMs still rely on tokenizers to compress raw text into common subword tokens. Methods like BPE are effective but function as hard-coded compression steps outside the training pipeline, involving numerous manual rules such as digit splitting, whitespace preservation, and special token handling. Simultaneously, byte-level and hierarchical language models attempt to return model inputs to UTF-8 bytes and perform downsampling within the architecture.

**Limitations of Prior Work**: Pure byte-level models suffer from excessive sequence lengths and high attention costs. Downsampling rules like fixed strides, whitespace-based rules, or entropy spikes are heuristic and not necessarily optimal. Existing methods for learning token boundaries end-to-end often use straight-through estimators (STE), approximating gradients for discrete boundaries as continuous variables; however, these lack theoretical guarantees for optimizing the expected loss of discrete boundaries.

**Key Challenge**: Tokenization is inherently a discrete compression decision, while LLM training relies on differentiable backpropagation. Using a score function estimator (SFE) allows for direct gradient computation on the expected loss of discrete policies but suffers from high variance. While STE offers lower variance, its objective and theory are more heuristic. This work aims to demonstrate that with proper variance reduction techniques from RL, SFE can be viable in real-world LLM pre-training.

**Goal**: The authors aim to learn token boundaries end-to-end within an autoregressive U-Net architecture while meeting three constraints: boundary policies are learned from loss rather than manual rules, byte-level representations are modular for token-level reuse, and the additional training compute is less than 0.1% of BPE-guided tokenization.

**Key Insight**: Treat whether to draw a token boundary at each byte position as an action of a Bernoulli policy, and use the next-byte cross-entropy as the reward signal. Thus, the model becomes a stochastic computation graph where the boundary policy can be optimized using REINFORCE/score function gradients.

**Core Idea**: Learn discrete token boundaries via a score function estimator, significantly reducing variance through early-exit baselines, time discounting, and batch-relative centering. This allows the model to naturally learn tokenization policies that approximate semantic boundaries from the language modeling loss.

## Method
The method revolves around an autoregressive U-Net: the model encodes at the byte level, downsamples byte representations into token representations based on sampled boundaries, performs transformer calculations on shorter sequences in the middle layers, and finally upsamples back to the byte level to predict the next byte. The core contribution lies in the token boundary policy and its low-variance training objective.

### Overall Architecture
Given a byte sequence $x_1,\dots,x_N$, the encoder extracts byte-level representations $X$. A boundary policy $\pi_\theta$ samples actions $a_i \in \{0,1\}$ at each position, where $a_i=1$ indicates a token boundary. The downsample operation selects byte representations at these boundaries to form a token-level sequence $X'$. A mid-transformer models data at the token level, followed by a distribute-then-add upsample layer that merges token-level information back into byte representations $Y$. Finally, the decoder predicts next bytes.

The training objective is the next-byte likelihood marginalized over all possible tokenization strategies. Since $a$ is a discrete random variable, the gradient consists of two parts: the standard conditional language modeling gradient and the policy gradient term $\log p_\theta(y|a,x)\nabla_\theta\log\pi_\theta(a|x)$. Reducing the variance of the latter is the primary focus of this work.

### Key Designs
1. **Score function estimator for direct discrete boundary optimization**:
    - **Function**: Allows the token boundary policy to learn directly from the language modeling loss instead of approximating via continuous relaxation.
    - **Mechanism**: Boundary sequences $a$ are sampled from $\pi_\theta(a|x)$, and the model optimizes $\mathbb{E}_{a\sim\pi_\theta}\log p_\theta(y|a,x)$. The score function gradient propagates information about whether a boundary decision improved or worsened subsequent token predictions back to the policy.
    - **Design Motivation**: Token boundaries are discrete actions. While STE requires manually defined surrogate gradients, REINFORCE aligns directly with the discrete problem and is theoretically cleaner if variance is controlled.

2. **RL-style variance reduction**:
    - **Function**: Addresses the noise inherent in single-sample Monte Carlo policy gradients.
    - **Mechanism**: First, an early-exit head predicts the next byte from early byte representations to serve as a tokenization-independent baseline, constructing a relative reward $R_i=\log p_\theta(x_i|x_{<i},a_{<i})-\log p^{early}_\theta(x_i|x_{<i})$. Second, a discount factor $\gamma=0.99$ is used to calculate future advantages. Finally, advantages are centered across the batch for the same token index to produce the policy loss $L^\pi=-\sum_i\log\pi_\theta(a_i|x_{<i},a_{<i})\,detach(A_i)$.
    - **Design Motivation**: A boundary action only affects subsequent predictions. Summing all future losses in long sequences causes extreme variance. The early baseline removes intrinsic sample difficulty, discounting handles credit assignment, and batch centering removes position-dependent bias.

3. **Lightweight policy and downsampling rate targeting**:
    - **Function**: Enables the model to learn boundary locations while controlling the average compression rate to avoid degenerating into one token per byte.
    - **Mechanism**: Boundary logits are determined by the current byte representation and historical boundaries within a short window (default $w=8$), passed through a sigmoid to get $p_i$. At initialization, a logit bias is added to align $p_i$ with the target boundary rate $\bar{\pi}_{target}=1/5$. During training, $L^{target}=\bar{l}\cdot detach(\bar{p}-\bar{\pi}_{target})$ applies pressure on the batch mean logit to maintain the target rate.
    - **Design Motivation**: Without compression constraints, the model favors finer-grained byte-level representations that are easier to learn but more expensive; constraints ensure fair comparisons between different strategies.

### Loss & Training
The total loss is defined as $L=L^{auto}+\lambda_\pi L^\pi+\lambda_{target}L^{target}+\lambda_{early}L^{early}$. $L^{auto}$ is the final next-byte cross-entropy, $L^{early}$ trains the early-exit baseline, $L^\pi$ is the policy loss, and $L^{target}$ regulates the average boundary rate. In experiments, $\lambda_\pi=\lambda_{target}=10^{-2}$, $\lambda_{early}=10^{-1}$, and the target downsampling rate is $1/5$.

## Key Experimental Results

### Main Results
Natural language experiments were conducted on filtered FineWeb with a ~147M parameter model, reporting bits-per-byte (BPB). Code experiments were conducted on CodeParrot with a ~90M parameter model. The authors prioritize the quality of learned boundaries and variance metrics over zero-shot accuracy at this scale.

| Method | PIQA↓ | HellaSwag↓ | ARC-Easy↓ | LAMBADA↓ | FineWeb Test↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Uniform | 1.660 | 1.306 | 1.974 | 1.926 | 1.376 |
| Dynamic (Nawrot et al.) | 1.737 | 1.340 | 2.011 | 1.956 | 1.372 |
| H-Net (Hwang et al.) | 1.641 | 1.313 | 2.000 | 2.130 | 1.386 |
| BPE guidance | 1.589 | 1.230 | 2.084 | 1.645 | 1.299 |
| Ours (w=1) | 1.561 | 1.199 | 1.987 | 1.584 | 1.280 |
| Ours | 1.557 | 1.212 | 2.016 | 1.737 | 1.297 |

### Ablation Study
Ablations focus on boundary window size, BPE-guidance comparison, target downsampling rates, and code data transfer.

| Analysis Item | Setting | Key Results | Description |
| :--- | :--- | :--- | :--- |
| Boundary Window Size | Ours w=8 vs Ours w=1 | FineWeb Test 1.297 vs 1.280 | Large windows are not necessary; short windows are sufficiently strong. |
| BPE-guidance Comparison | 200k BPE boundaries shifted right to avoid lookahead | FineWeb Test 1.299 | Learned dynamic policy nearly matches external BPE priors. |
| FLOPs-Validation Loss | 147M FineWeb | BPB ~1.279, better than uniform/STE curves | Semantic boundaries translate into training loss advantages. |
| CodeParrot | 90M Python code | Val loss: Ours 0.568 vs H-Net 0.769 | SFE boundary policies show greater advantages in structured code data. |
| Downsampling Rate | $\bar{a}$ after convergence | Ours 0.204 vs BPE 0.207 | Target rate control is stable and converges near BPE compression rates. |

### Key Findings
- The model learns whitespace-like and semantic boundaries without prior knowledge of spaces, lexicons, or BPE, suggesting that language modeling loss contains strong tokenization signals.
- The performance of $w=1$ is close to the full window, indicating that boundary policies rely primarily on the current byte representation and the most recent boundary.
- While BPE guidance remains strong, this method is the only dynamic learning strategy that recovers the performance of BPE-guided downsampling on FineWeb without external BPE boundaries.
- Gains on Python code are more pronounced: the model learns to draw boundaries at the start of module names and avoids wasting compute on low-value regions like the Apache License.

## Highlights & Insights
- The most valuable aspect is the direct handling of discreteness in tokenization rather than avoiding it via surrogate gradients. This choice leads to a more natural objective and a cleaner interface for future learned tokenizers.
- The early-exit relative reward is clever: it subtracts the inherent difficulty of the text from the reward, forcing the boundary policy to focus on the gains provided specifically by tokenization.
- Conclusions suggest that BPE’s advantage is not just frequency statistics but providing reasonable semantic compression points; if models can learn these themselves, future tokenizers can be language-agnostic and task-adaptive.

## Limitations & Future Work
- Experimental scale is limited (90M-147M). Autoregressive U-Net advantages over token transformers may only manifest at scales $>10^{21}$ FLOPs, requiring further validation on frontier LLMs.
- Downstream accuracy at this scale is near-random; bits-per-byte is used instead for continuation likelihood. This is more stable but not equivalent to real-world task performance.
- While SFE is viable, it requires careful variance reduction and hyperparameter tuning. Stability across different datasets and model scales remains an area for further study.
- The optimal tokenization aspect ratio and its relationship to model scale remain open questions. Initial findings suggest an aspect ratio of 2 is optimal for 20M-40M scales, but this may change for larger models.

## Related Work & Insights
- **vs BPE / SuperBPE**: BPE uses frequency-based rules outside training; this work integrates boundary decisions into the model, allowing tokenization to evolve with the language modeling objective.
- **vs byte-level models**: Pure byte models like ByT5 avoid tokenizer bias but suffer from long sequences; autoregressive U-Net with learned downsampling balances byte-level universality with token-level efficiency.
- **vs STE-based tokenization**: STE uses continuous relaxation for gradients, which is simpler but approximates the objective; SFE optimizes the discrete policy directly by controlling variance.
- **vs RL for routing / MoE**: Both involve learning discrete structural choices, but token boundary credit assignment spans long byte sequences, making early baselines and discounting critical.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Training LLM tokenization with SFE is a clean and challenging approach.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes NL, code, BPE controls, and window ablations, though model scale is small.
- Writing Quality: ⭐⭐⭐⭐☆ Clear derivations and well-motivated variance reduction; BPB metrics require familiarity.
- Value: ⭐⭐⭐⭐☆ Highly insightful for tokenizer-free or learned-tokenizer LLMs, particularly for improving non-English text processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/shift_before_you_learn_enabling_low-rank_representations_in_reinforcement_learni.md)
- [\[ICML 2026\] Can Large Language Models Generalize Procedures Across Representations?](can_large_language_models_generalize_procedures_across_representations.md)
- [\[ICML 2026\] What Does Reinforcement Learning for Visual Tool Use Actually Learn?](what_does_vision_tool-use_reinforcement_learning_really_learn_disentangling_tool.md)
- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](../../ACL2026/reinforcement_learning/easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)

</div>

<!-- RELATED:END -->
