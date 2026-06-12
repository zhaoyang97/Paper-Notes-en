---
title: >-
  [Paper Note] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping
description: >-
  [ICML 2026][LLM Safety][Markov Process] The next-token distribution of an autoregressive LLM is interpreted as the state transition matrix of a Markov chain. Consequently…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Markov Process"
  - "Vocabulary Expansion"
  - "embedding tuning"
  - "zero forgetting"
  - "sample complexity"
date: 2026-05-08
content_hash: ee2a208322504c9d
---

# Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping

**Conference**: ICML 2026  
**arXiv**: [2605.04308](https://arxiv.org/abs/2605.04308)  
**Code**: None  
**Area**: LLM Continual Learning / Vocabulary Expansion / Anti-forgetting  
**Keywords**: Markov Process, Vocabulary Expansion, embedding tuning, zero forgetting, sample complexity

## TL;DR
The next-token distribution of an autoregressive LLM is interpreted as the state transition matrix of a Markov chain. Consequently, "learning new words" is treated as adding new states to the state space and representing them as sparse combinations of existing states. Theoretically, this requires only $O(s)$ samples ($s$ being the number of mapped old tokens). In practice, finetuning only the embeddings of new tokens achieves cross-lingual or new concept expansion with strictly zero forgetting.

## Background & Motivation

**Background**: Adapting pre-trained LLMs to new vocabularies, entities, or domains (e.g., "COVID-19", specific technical terms, cross-lingual transfer) is a core requirement in continual learning. Prevailing approaches include full fine-tuning, LoRA, prompt tuning, or retrieval-augmentation.

**Limitations of Prior Work**: Even in modern models like Llama-3 or Qwen-2.5, standard fine-tuning still exhibits significant catastrophic forgetting, and this issue often intensifies with model scale. Furthermore, model updates are irreversible; once corrupted, the original performance is difficult to recover.

**Key Challenge**: Updating billions of parameters to incorporate a small amount of new knowledge is counter-intuitive. Full parameter updates naturally contaminate the transition relationships between old tokens.

**Goal**: To establish a clean mathematical framework for adding $m \ll T$ new tokens without disrupting the transitions between the original $T$ tokens, while providing provable sample complexity.

**Key Insight**: View the LLM as a Markov chain where tokens are states and $p_\theta(\cdot \mid x_t)$ is the transition probability vector. In this view, "no forgetting" is equivalent to "keeping the transition matrix between old states invariant," and "adding knowledge" corresponds to "expanding the state space from $\mathcal{V}$ to $\mathcal{V} \cup \mathcal{U}$."

**Core Idea**: Each new token $u$ is represented as a sparse linear combination of existing tokens in the embedding space ($\bm{\alpha}^{(u)} \in \mathbb{R}^T$, $\|\bm{\alpha}^{(u)}\|_0 \le s$) to reuse the semantic structure of the old dictionary. This is implemented by finetuning only the new token embeddings while freezing all other weights, ensuring structural zero forgetting.

## Method

### Overall Architecture
The input consists of a pre-trained LM $\texttt{LM}_\theta$, an old vocabulary $\mathcal{V}$, a set of new tokens $\mathcal{U}$, and training sequences containing new tokens. The output is an updated $\texttt{LM}_{\tilde{\theta}}$ that behaves identically to the original model for old tokens while reproducing the oracle transition distribution $\mathbf{q}^{(u)}$ for new tokens. The pipeline follows three conceptual steps: (1) Formalizing the generation process as a first-order Markov process and translating "knowledge expansion" into constraints of state space expansion and transition preservation; (2) Theoretically deriving that each new token requires $O(s)$ samples based on the assumption that new transitions are $s$-sparse combinations of old transitions; (3) Implementing this strategy via embedding tuning by updating only the new token embeddings.

### Key Designs

1.  **Markovian Knowledge Expansion**:
    - **Function**: Formulates vocabulary expansion as equality constraints for Markov state space expansion and transition preservation.
    - **Mechanism**: Transition vectors between old tokens $\mathbf{p}^{(v)} \in \Delta(\mathcal{V})$ are fixed. For new tokens $u$, the model learns $\mathbf{q}^{(u)} \in \Delta(\mathcal{V})$ while enforcing $p_{\tilde{\theta}}(u \mid v) = 0$ and $p_{\tilde{\theta}}(u_i \mid u_j) = 0$. This ensures $p_{\tilde{\theta}}(\cdot \mid x_t)$ is identical to the original model when $x_t \in \mathcal{V}$, resulting in exactly zero forgetting.
    - **Design Motivation**: Upgrades "behavior preservation" from an empirical/regularized objective to a structural constraint. "Zero forgetting" becomes a provable fact rather than just an empirical observation.

2.  **Token-to-Dictionary Sparse Mapping**:
    - **Function**: Represents new tokens using sparse combinations $\mathbf{E}^\top \bm{\alpha}^{(u)}$ of the old embedding dictionary $\mathbf{E} \in \mathbb{R}^{T \times d}$.
    - **Mechanism**: Defines $f: \mathbb{R}^d \to \Delta(\mathcal{V})$ as the model's logit head, aiming to find $\bm{\alpha}^{(u)}$ such that $f(\mathbf{E}^\top \bm{\alpha}^{(u)}) = \mathbf{q}^{(u)}$. The empirical transition distribution $\hat{\mathbf{q}}^{(u)}$ is fitted using KL divergence under constraints $\|\bm{\alpha}^{(u)}\|_0 \le s$ and $\|\bm{\alpha}^{(u)}\|_2 \le B$. The paper proves the required sample size $N$ per new token satisfies $N \ge \tilde{O}(s \log^2 c)$, independent of the old vocabulary size $T$ or model dimension.
    - **Design Motivation**: New concepts are rarely entirely "new"; e.g., "COVID-19" is transitionally similar to a mixture of "virus", "disease", and "outbreak". Encoding "semantic granularity" into sample complexity is more realistic than counting parameters.

3.  **Embedding Tuning Algorithm**:
    - **Function**: Implements the mapping strategy on Transformers with minimal cost, achieving feasibility and strict zero forgetting.
    - **Mechanism**: The embedding rows corresponding to new tokens (e.g., Llama-3's `<|reserved_special_token_0|>`) are the only trainable parameters, while the remaining billions of parameters are frozen. Standard next-token cross-entropy is used on the training corpus. During inference, new tokens pass through existing weights, but their query vectors are "mapped" to positions that best reproduce $\mathbf{q}^{(u)}$.
    - **Design Motivation**: This implementation naturally satisfies theoretical assumptions, as updated parameters are orthogonal to old token transitions, making "zero forgetting" an observable fact.

### Loss & Training
Standard next-token cross-entropy $-\sum_t \log p_{\tilde{\theta}}(x_{t+1} \mid x_{t-k:t})$ is employed without regularization or replay. The paper also discusses higher-order Markov chain expansion, where theoretical conclusions remain valid by treating context windows as combined states. Due to a small effective branching factor $b \ll T$, the practical sparsity remains $s = O(Kb)$.

## Key Experimental Results

### Main Results
The "sample efficiency + zero forgetting" is validated across three types of tasks: Arithmetic (learning $a \langle\text{spec}\rangle b = a \times b$ in Llama-3.2-3B), Synthetic Lexicon Injection (100 pseudo-words in real sentences), and Cross-lingual Adaptation (adapting Qwen2.5-3B to Spanish, German, and Arabic).

| Task | Model | Method | Target Metric | Forgetting (English / Addition) |
|------|------|------|----------|----------------------|
| Arithmetic ($a\langle\text{spec}\rangle b$) | Llama-3.2-3B | FFT | 77.2% acc | Addition 100% → 0% (Catastrophic) |
| Arithmetic | Llama-3.2-3B | ET | **81.4%** acc | Addition remains 100% |
| Cross-lingual (Spanish) | Qwen2.5-3B | FFT | loss 5.56 | English loss +9.83 |
| Cross-lingual (Spanish) | Qwen2.5-3B | ET | **loss 2.30** | English loss **−0.04** |
| Cross-lingual (Arabic) | Qwen2.5-3B | ET | **loss 2.82** | English loss **0.00** |

### Ablation Study

| Settings | Synthetic loss | WikiText Forgetting |
|------|-------------------|---------------|
| Base model (Llama-3.1-8B) | — | Baseline 2.42 |
| FFT (N=1000) | 4.40 | +1.24 |
| LoRA | 7.63 | +8.36 |
| Prompt Tuning | 3.79 | +0.69 |
| ET (Ours) | 2.42 | **0.00** |

### Key Findings
- ET simultaneously achieves the best target loss and strictly zero forgetting across three disparate tasks, proving that zero forgetting does not necessarily come at the cost of accuracy.
- In arithmetic, the accuracy of $a\langle\text{spec}\rangle b$ via ET (81.4%) exceeds the base model's $a \times b$ (63.5%). This is because the $\langle\text{spec}\rangle$ embedding implicitly converges to a sparse ensemble of equivalent terms like "$\times$, times, multiplies," providing direct evidence for the "sparse dictionary hypothesis."
- LoRA exhibits worse forgetting than FFT in cross-lingual tasks (loss 7.63 vs 4.40). This indicates that fewer parameters do not guarantee less forgetting; forgetting depends on whether update directions touch old transitions.
- A slight "Negative Forgetting" (−0.04 / −0.08) appeared in Spanish/German tasks, suggesting transfer gains from related languages.

## Highlights & Insights
- Continual learning is transformed from an empirical tuning problem into a Markov state space expansion problem with analytic structural conditions for "zero forgetting." This approach applies to any autoregressive model with discrete outputs (speech, code, etc.).
- Sample complexity depends on mapping sparsity $s$ rather than vocabulary size $T$ or model dimension $d$, providing a counter-intuitive scaling rule: larger vocabularies do not necessarily make learning new words more expensive.
- "Embedding Tuning" represents an extreme of the PEFT (Parameter-Efficient Fine-Tuning) paradigm. It relies on the natural orthogonality between embeddings and the transition computation graph rather than rank constraints or prompt prefixes.

## Limitations & Future Work
- The assumption of uniform probability for new tokens in training corpora may not hold in practice, where frequency distributions might be coupled with "mapped old tokens."
- The framework assumes the LLM is sufficiently expressive ($f$ is Lipschitz and can implement any sparse combination); the hypothesis may fail if model capacity is limited or concepts are outside the dictionary's coverage.
- The method currently disallows transitions from old tokens to new tokens, meaning new words can be "evoked" but not naturally initiated in generation, a constraint for generative applications.
- Experiments focused on "single-point expansion" rather than complex structures where new tokens transition to each other (e.g., entire new domain terminologies).

## Related Work & Insights
- **vs LoRA / PT / Adapter**: These reduce parameter counts but do not guarantee orthogonality with old transitions. The proposed method utilizes the structural orthogonality of embedding tables and transition graphs.
- **vs EWC / GEM / Replay**: Classic continual learning relies on regularization or replay, which is difficult to scale for LLMs. This method requires no replay or Fisher information.
- **vs FlexOlmo / Model Editing**: While editing assumes existing target behaviors, this work constrains the update space to structurally isolate forgetting.
- **vs Markov-LM (Zekri et al., Yüksel & Flammarion)**: While prior work used Markov perspectives to analyze Transformer learning capabilities, this work applies that mathematical path to design specific continual learning algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines Markov perspective with sparse dictionary hypothesis for provable sample complexity.
- Experimental Thoroughness: ⭐⭐⭐ Covers arithmetic, synthetic, and cross-lingual tasks, though lacks evaluation on complex multi-step expansions.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation to theorems and algorithms with consistent notation.
- Value: ⭐⭐⭐⭐ Provides a minimalist baseline demonstrating that zero forgetting and sample efficiency can coexist.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](optimizing_token_choice_for_code_watermarking_an_rl_approach.md)
- [\[ICML 2026\] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](from_volume_to_value_preference-aligned_memory_construction_for_on-device_rag.md)
- [\[ICML 2026\] Efficient DP-SGD for LLMs with Randomized Clipping](efficient_dp-sgd_for_llms_with_randomized_clipping.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)

</div>

<!-- RELATED:END -->
