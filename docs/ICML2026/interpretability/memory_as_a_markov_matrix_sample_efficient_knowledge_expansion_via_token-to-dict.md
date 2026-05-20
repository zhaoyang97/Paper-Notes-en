---
title: >-
  [Paper Note] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping
description: >-
  [ICML 2026][Interpretability][Markov process] Interprets the next-token distribution of an autoregressive LLM as the state transition matrix of a Markov chain…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Markov process"
  - "vocabulary expansion"
  - "embedding tuning"
  - "zero forgetting"
  - "sample complexity"
date: 2026-05-08
content_hash: 21decc2e565829c0
---

# Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping

**Conference**: ICML 2026  
**arXiv**: [2605.04308](https://arxiv.org/abs/2605.04308)  
**Code**: None  
**Area**: LLM Continual Learning / Vocabulary Expansion / Anti-Forgetting  
**Keywords**: Markov process, vocabulary expansion, embedding tuning, zero forgetting, sample complexity

## TL;DR
Interprets the next-token distribution of an autoregressive LLM as the state transition matrix of a Markov chain, so "learning new words" becomes "adding new states to the state space and representing them as sparse combinations of existing states." Theoretically, only $O(s)$ samples are needed ($s$ is the number of mapped old tokens); in practice, simply finetuning the embedding of the new token suffices to achieve cross-lingual/new concept expansion with strict zero forgetting.

## Background & Motivation

**Background**: Adapting pretrained LLMs to new vocabulary, entities, or domains is a core requirement for continual learning (e.g., "COVID-19", domain-specific terms, cross-lingual transfer). Mainstream approaches include full fine-tuning, LoRA, prompt tuning, or external retrieval.

**Limitations of Prior Work**: Even with new models like Llama-3 or Qwen-2.5, standard fine-tuning still leads to significant catastrophic forgetting—worse for larger models; model updates are irreversible, and once degraded, cannot be reverted.

**Key Challenge**: Modern LLMs are highly expressive, yet updating billions of parameters to "fit a handful of new knowledge" is counterintuitive; full updates inherently pollute transition relations among old tokens.

**Goal**: Establish a clean mathematical framework for "adding $m \ll T$ new tokens without disrupting the transition relations among the original $T$ tokens," and provide provable sample complexity.

**Key Insight**: View the LLM as a Markov chain—tokens as states, $p_\theta(\cdot \mid x_t)$ as the transition probability vector. In this view, "no forgetting" is equivalent to "keeping the transition matrix among old states unchanged," and "adding knowledge" is "expanding the state space from $\mathcal{V}$ to $\mathcal{V} \cup \mathcal{U}$."

**Core Idea**: Each new token $u$ only needs to be represented as a sparse linear combination of existing tokens (in embedding space, $\bm{\alpha}^{(u)} \in \mathbb{R}^T$, $\|\bm{\alpha}^{(u)}\|_0 \le s$), thus reusing the semantic structure of the old dictionary. Implementation-wise, only the embedding vector of the new token is finetuned, with all other weights frozen, strictly guaranteeing zero forgetting.

## Method

### Overall Architecture
Input: a pretrained LM $\texttt{LM}_\theta$, old vocabulary $\mathcal{V}$, a small batch of new tokens $\mathcal{U}$, and training sequences containing new tokens. Output: updated parameters $\texttt{LM}_{\tilde{\theta}}$ that behaves identically to the original model on old tokens and can reproduce the oracle transition distribution $\mathbf{q}^{(u)}$ on new tokens. The pipeline consists of three conceptual steps: (1) formalize the generation process as a first-order Markov process, translating "adding knowledge" into "state space expansion + preserving old transitions" constraints; (2) assume the true transition of each new token can be written as an $s$-sparse combination of existing token transitions, and theoretically derive that $O(s)$ samples per token suffice; (3) in practice, implement this mapping via embedding tuning: only the embedding vectors of new tokens (e.g., $\langle \text{spec} \rangle$) are trainable, all others are frozen.

### Key Designs

1. **Markov-based Knowledge Expansion**:

    - **Function**: Formulate "adding new vocabulary" as "Markov chain state space expansion + preserving old transitions" via equality constraints.
    - **Mechanism**: The transition vectors among old tokens $\mathbf{p}^{(v)} \in \Delta(\mathcal{V})$ are given by the pretrained model; after introducing a new token $u$, only $\mathbf{q}^{(u)} \in \Delta(\mathcal{V})$ needs to be learned, with enforced $p_{\tilde{\theta}}(u \mid v) = 0,\ p_{\tilde{\theta}}(u_i \mid u_j) = 0$. Thus, $p_{\tilde{\theta}}(\cdot \mid x_t)$ for $x_t \in \mathcal{V}$ remains identical to the original model, with precisely zero forgetting.
    - **Design Motivation**: Elevates "behavior preservation" from empirical/regularization to a structural constraint, making "zero forgetting" a derivable fact rather than an empirical observation.

2. **Token-to-Dictionary Sparse Mapping**:

    - **Function**: Represent new tokens as sparse combinations $\mathbf{E}^\top \bm{\alpha}^{(u)}$ of the old embedding dictionary $\mathbf{E} \in \mathbb{R}^{T \times d}$, and directly learn the combination coefficients.
    - **Mechanism**: Define $f: \mathbb{R}^d \to \Delta(\mathcal{V})$ as the model's logit head; the goal is to find $\bm{\alpha}^{(u)}$ such that $f(\mathbf{E}^\top \bm{\alpha}^{(u)}) = \mathbf{q}^{(u)}$; under the sparse constraints $\|\bm{\alpha}^{(u)}\|_0 \le s,\ \|\bm{\alpha}^{(u)}\|_2 \le B$, fit the empirical transition distribution $\hat{\mathbf{q}}^{(u)}$ via KL. The paper proves that the required sample size per new token $N$ satisfies $N \ge \tilde{O}(s \log^2 c)$, independent of vocabulary size $T$ and model dimension.
    - **Design Motivation**: New concepts are rarely "entirely new"—e.g., "COVID-19" transitions are close to a mixture of "virus / disease / outbreak" and other old words. Encoding this "semantic granularity" directly into sample complexity is more realistic than simply counting parameter cost.

3. **Embedding Tuning Training Algorithm**:

    - **Function**: Implement the above mapping strategy on Transformers with minimal overhead, achieving "practical feasibility + strict zero forgetting."
    - **Mechanism**: The embedding row for the new token (e.g., the 3072-dim vector for `<|reserved_special_token_0|>` in Llama-3) is the only trainable parameter; all other 3B parameters are frozen. Standard next-token cross-entropy is used for gradient descent on training data containing the new token. During inference, the new token passes through the same attention/FFN weights as before; only its query vector is "pulled" to the position that best reproduces $\mathbf{q}^{(u)}$.
    - **Design Motivation**: This implementation naturally satisfies the theoretical assumptions—the updated parameters are strictly orthogonal to old token transitions (old tokens never query the new embedding unless the new token appears in context), making "zero forgetting" not just a theoretical condition but an observable property in practice.

### Loss & Training
Standard next-token cross-entropy $-\sum_t \log p_{\tilde{\theta}}(x_{t+1} \mid x_{t-k:t})$, with no regularization or replay. The paper also discusses higher-order Markov chain extensions: treating the context of $K$ tokens as a composite state preserves the theoretical results, and since the effective branching factor $b \ll T$ in natural language, the actual sparsity is only $s = O(Kb)$.

## Key Experimental Results

### Main Results
Three types of tasks validate "sample efficiency + zero forgetting": on arithmetic operator tasks, Llama-3.2-3B learns $a\langle\text{spec}\rangle b = a \times b$, comparing ET and FFT for addition retention; on synthetic vocabulary tasks, 100 pseudo-words (e.g., "glor", "zorp") are injected into real sentences and forgetting is measured on WikiText; for cross-lingual tasks, Qwen2.5-3B is adapted to Spanish/German/Arabic.

| Task | Model | Method | Target Metric | Forgetting (English / Addition) |
|------|------|------|----------|----------------------|
| Arithmetic Operator ($a\langle\text{spec}\rangle b$) | Llama-3.2-3B | FFT | 77.2% acc | Addition 100% → 0% (catastrophic) |
| Arithmetic Operator | Llama-3.2-3B | ET | **81.4%** acc | Addition still 100% |
| Cross-lingual (Spanish) | Qwen2.5-3B | FFT | loss 5.56 | English loss +9.83 |
| Cross-lingual (Spanish) | Qwen2.5-3B | ET | **loss 2.30** | English loss **−0.04** |
| Cross-lingual (Arabic) | Qwen2.5-3B | ET | **loss 2.82** | English loss **0.00** |

### Ablation Study

| Setting | Synthetic Vocab Test Loss | WikiText Forgetting |
|------|-------------------|---------------|
| Base Model (Llama-3.1-8B) | — | Baseline 2.42 |
| FFT (N=1000) | 4.40 | +1.24 |
| LoRA | 7.63 | +8.36 |
| Prompt Tuning | 3.79 | +0.69 |
| ET (Ours) | 2.42 | **0.00** |

### Key Findings
- ET achieves both "best/near-best target loss" and "precisely zero forgetting" across synthetic vocabulary, cross-lingual, and arithmetic operator tasks, confirming that zero forgetting is not traded for accuracy.
- On the arithmetic operator task $a\langle\text{spec}\rangle b$, ET's accuracy of 81.4% even surpasses the base $a*b$ accuracy of 63.5%, because the embedding for $\langle\text{spec}\rangle$ implicitly converges to a sparse combination of multiple equivalent expressions ("*$\times$ / times / multiplies / *"), with ensemble accuracy 73.2%—a direct empirical validation of the "sparse dictionary assumption."
- LoRA not only lacks an advantage in cross-lingual tasks but performs worse than FFT (loss 7.63 vs 4.40, forgetting +8.36), indicating that "fewer parameters" does not equate to "no forgetting"; forgetting arises from whether the update direction touches old transitions, not the number of updated parameters.
- Slight "negative forgetting" (−0.04 / −0.08) is observed for Spanish/German, i.e., learning Spanish slightly improves English WikiText; the authors speculate this is due to transfer gains between closely related languages.

## Highlights & Insights
- Transforms "continual learning" from an empirical tuning problem into an equality-constrained "Markov chain state space expansion" problem, providing analytically derivable "zero forgetting" structural conditions. This approach is transferable to any autoregressive model with discrete output distributions (speech, code, symbolic).
- Sample complexity depends only on mapping sparsity $s$, not vocabulary size $T$ or model dimension $d$, yielding a counterintuitive but instructive scaling rule: larger vocabularies do not make learning new words more expensive, provided the sparsity assumption holds.
- "Finetuning only new token embeddings" is an extreme form of the LoRA/PT PEFT approach—it does not rely on rank constraints or prompt prefixes, but directly exploits the natural orthogonality between embeddings and the transition computation graph.

## Limitations & Future Work
- Assumes each new token appears with uniform probability in training data, but real-world frequency distributions may be strongly coupled with the "cluster of mapped old tokens" (high-frequency words are more likely to pollute old transitions).
- Assumes the LLM is sufficiently expressive ($f$ is Lipschitz and can precisely realize any sparse combination); if model capacity is limited or new concepts are far from the original dictionary coverage, the sparsity assumption may break down.
- The method by default "does not allow transitions from old tokens to new tokens," meaning new words can only be "summoned" but not naturally generated in output, which is a clear constraint for generative applications and acknowledged as future work.
- No experiments on complex structures involving "mutual transitions among new tokens" (e.g., learning an entire network of new domain terms), only "single-point expansion" is validated.

## Related Work & Insights
- **vs LoRA / PT / Adapter**: These reduce the number of updated parameters but cannot guarantee update directions are orthogonal to old transitions, so forgetting remains uncontrollable; this method directly leverages the orthogonal structure between the embedding table and transition graph, making zero forgetting a structural guarantee rather than a tuning outcome.
- **vs EWC / GEM / replay**: Classical continual learning methods suppress forgetting via regularization or replay, but scaling to LLMs remains challenging; this method requires no replay of old data or Fisher information, only updating new embeddings.
- **vs FlexOlmo / model editing**: Those methods assume "target behavior first, then edit," whereas this work "structurally isolates forgetting by constraining the update space"—the two paradigms are complementary.
- **vs Markov-LM related work (Zekri et al., Yüksel & Flammarion)**: They analyze transformer learning ability from a Markov perspective; this work uses the mathematical path to design a concrete continual learning algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines the Markov perspective with the sparse dictionary assumption to provide provable sample complexity; the theoretical framework is refreshing.
- Experimental Thoroughness: ⭐⭐⭐ Arithmetic + synthetic vocab + three target languages + multiple model scales are covered, but more realistic multi-domain/multi-step expansions are not.
- Writing Quality: ⭐⭐⭐⭐ The logic from motivation to theorem to algorithm is very smooth, with unified notation and clear definitions.
- Value: ⭐⭐⭐⭐ Provides a minimal solution where "zero forgetting + sample efficiency" can be achieved simultaneously, serving as a strong baseline for the continual learning community.

## Related Papers

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](../../ICLR2026/llm_safety/redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)

</div>

<!-- RELATED:END -->
