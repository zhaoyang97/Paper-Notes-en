---
title: >-
  [Paper Note] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management
description: >-
  [ICML2026][LLM/NLP][Transformer Turing-completeness] The authors argue that the popular claim "Transformer is Turing-complete" is subtly replaced in most existing proofs by "a family of different Transformers together ca…
tags:
  - "ICML2026"
  - "LLM/NLP"
  - "Transformer Turing-completeness"
  - "autoregressive decoding"
  - "context management"
  - "computational complexity"
  - "position paper"
date: 2026-05-08
content_hash: 9a944bafa7855029
---

# Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management

**Conference**: ICML2026  
**arXiv**: [2605.19514](https://arxiv.org/abs/2605.19514)  
**Code**: None  
**Area**: llm_nlp  
**Keywords**: Transformer Turing-completeness, autoregressive decoding, context management, computational complexity, position paper

## TL;DR
The authors argue that the popular claim "Transformer is Turing-complete" is subtly replaced in most existing proofs by "a family of different Transformers together can simulate a Turing machine." They formalize a fixed system $(T,D,C)$ closer to actual deployment, proving that the computational power of the same fixed Transformer can transition from recognizing only regular languages to being Turing-complete depending on context management strategies, thereby shifting the research focus from the model itself to the context manager.

## Background & Motivation

**Background**: Since 2019, a long series of theoretical works (Pérez, Bhattamishra, Merrill & Sabharwal, Li, etc.) have claimed that Transformers are Turing-complete in some sense, which has been used as a default endorsement for "sufficient model expressivity" in numerous LLM papers.

**Limitations of Prior Work**: These proofs almost entirely rely on two types of unrealistic assumptions: **allowing the context window to scale with input length $n$** (where each attention step sees $n+t$ tokens) or **allowing numerical precision to scale with input length** ($O(\log n)$, $\mathrm{poly}(n)$, or even unbounded reals). In real-world deployments, the LLM context window $N$ and numerical precision are fixed constants, meaning the "model" constructed in these proofs is essentially a different network for every different input length.

**Key Challenge**: The definition of a Turing Machine (TM) requires a **single** machine to operate on arbitrarily long inputs. If a different Transformer is used for each length, the result is technically a circuit family. This is qualitatively equivalent to the Savage result encoding $\textsf{DTIME}(T(n))$ as a circuit family of size $O(T(n)^2)$, which cannot be directly termed "Turing-complete." In other words, the scaling-family provides a **resource bound**, not **universality**.

**Goal**: To strictly separate "what is fixed" from "what can grow," re-classify existing results, and re-discuss computational power within a fixed system paradigm that reflects reality.

**Key Insight**: By fixing a pre-trained Transformer $T:\Sigma^N\to\Delta(\Sigma)$, a decoding rule $D$, and finite precision, the only way to process arbitrarily long inputs is via a **context manager** $C$. This manager decides which $N$ tokens are sent into the window at each step and how generated results are written back to history. The entire system is represented as a triple $(T,D,C)$. The component $C$, often dismissed as an engineering detail, determines the computational upper bound of the system.

**Core Idea**: Under the fixed system paradigm, the Turing-completeness of the Transformer itself is a meaningless proposition. **The context management strategy is the key variable determining the system's computational power.** Summarization-style management reduces the system to regular languages, appending-style management yields linear-bounded automata (LBA), and only reading/writing external memory or multi-token decoding truly reaches Turing-completeness.

## Method

As a position paper without experiments, the "Method" consists of **a formal model, a qualitative classification, and two main theorems**.

### Overall Architecture

The execution of a fixed system $(T,D,C)$ is defined as follows: given input $x=x_1\cdots x_n$, let $r^{(1)}=x$; at step $t$, $C$ constructs the string $w^{(t)}=C_w(r^{(t)})\in\Sigma^N$ for the context window. The Transformer provides a next-token distribution, $D$ selects $\hat{x}_{t+1}=D(T(w^{(t)}))$, and $C$ updates its maintained history via $r^{(t+1)}=C_r(\hat{x}_{t+1}, r^{(t)})$ until a halting condition is met. To avoid trivializing the conclusion by "hiding a TM inside $C$," the authors restrict the analysis to a **simple manager**: one that uses $N$ token units + $O(1)$ state and can only perform push/pop/constant offset operations on the history string. Within this framework, 21 representative works are reviewed (Table 1), grouped by whether they assume scaling windows or scaling precision.

### Key Designs

1.  **Fixed System Formalization $(T,D,C)$ and Scaling/Fixed Dichotomy**:
    - **Function**: Explicitly incorporates "which quantities are constants and which grow with input" into the definition of the computational model, distinguishing the fixed-system regime (a single Transformer + fixed $N$ + fixed precision) from the scaling-family regime (a family of Transformers chosen by input length).
    - **Mechanism**: The Transformer is abstracted as a constant function $T:\Sigma^N\to\Delta(\Sigma)$, separating decoding and history maintenance into $D$ and $C$. The authors point out that scaling-families are isomorphic to circuit families, providing an $O((T(n))^2)$ type resource bound rather than Turing-completeness.
    - **Design Motivation**: To explain why findings like those of Pérez 2019 or Merrill & Sabharwal 2024 (simulating TM with $O(\log n)$ precision) are misinterpreted by practitioners as "GPT is Turing-complete." The "machine" in those proofs changes with $n$, which does not correspond to a deployed LLM.

2.  **Summarization-style Management $\Rightarrow$ Constant Space Upper Bound (Proposition 5.1)**:
    - **Function**: Proves that regardless of how powerful $T$ is, if $C$ is a summarization-style manager that compresses history into single-token summaries (similar to `/compact`, AutoCompressor, or ICAE), the system $(T,D,C)$ does not exceed $\textsf{FDSPACE}(1)$.
    - **Mechanism**: A one-way, three-tape transcriber is constructed to simulate the system. Its work tape simulates the context window, a separator, and a workspace for single-step decoding. Since input is either appended or summarized into a constant state, the total space required remains $O(N)$. Given the standard fact $\textsf{REG}=\textsf{DSPACE}(1)$, summarization systems can only recognize regular languages, failing on non-regular languages like $\{x\#x\}$, palindromes $\{x\#x^R\}$, or binary addition.
    - **Design Motivation**: This challenges the notion that adding a `/compact` function allows LLMs to handle arbitrary length tasks. Compressing history essentially limits memory to $O(1)$ bits. No matter how sophisticated $T$ is, the system cannot bypass the limit of a finite state automaton. This also explains another facet of the Kaplan scaling law: state space increases exponentially with $N$, allowing engineering improvements without changing the theoretical class.

3.  **Appending-style Management $\Leftrightarrow$ Linear Space TM (Proposition 5.2 + 5.4)**:
    - **Function**: Proves that when $C$ uses a sliding-window + appending (appending each generated token and shifting the window), the system is **equivalent** to $\textsf{DSPACE}(n)$, recognizing Deterministic Context-Sensitive Languages (DCSL).
    - **Mechanism**: Forward proof (Prop. 5.2) uses a TM to copy history to a work tape, copying the last $N$ tokens for each decoding step. Backward proof (Prop. 5.4) utilizes the $(N,K)$-restricted system framework from Schuurmans et al. 2024, where a $(2,1)$-restricted system is proven to simulate any linear-space TM. Lemma 5.3 further proves any $f:\Sigma^2\to\Sigma$ can be implemented by a Transformer with window size 2 and greedy decoding.
    - **Design Motivation**: Establishes a clear capability ladder (Regular $\to$ DCSL $\to$ Turing-complete via multi-token/external memory), supporting the stance that **$C$ is the true determinant of real-world LLM capabilities**.

### Loss & Training
This is a position paper; there is no training process. All conclusions derive from **formal language and complexity analysis** of pre-trained models and decoding systems, independent of specific training objectives.

## Key Experimental Results

As a position paper, there are no numerical experiments. The key "data" consists of two categorization and hierarchy tables.

### Main Table 1: Implicit Scaling Assumptions in Prior Turing-completeness Proofs

| Context Window Scale | Numerical Precision | Representative Work | Fixed System Proof? |
| :--- | :--- | :--- | :--- |
| $n+t$ | unbounded | Pérez 2019, Bhattamishra 2020, Roberts 2024, Nowak 2024, Jiang 2026 | No (Dual scaling) |
| $n+t$ | $\mathrm{poly}(n)$ | Li 2024 | No (Dual scaling) |
| $n+t$ | $O(\log(n+t))$ | Merrill & Sabharwal 2024, Qiu 2025, Hou 2025 | No (Window scaling) |
| $n+t$ | $O(1)$ | Malach 2024 | No (Window scaling) |
| $n$ | unbounded / $O(\log n)$ | Back De Luca 2024, Giannou 2023 | No (Window scales with input) |
| $s(n)$ | $O(1)$ | Li & Wang 2025 | No (Scaling by space complexity) |

The vast majority of cited proofs trigger either Group A (scaling window) or Group B (scaling precision), making them essentially circuit family arguments.

### Main Table 2: Capability of Fixed Transformer Systems under Different Context Management

| Context Management Style | Computational Capability | Source |
| :--- | :--- | :--- |
| Read/Write External Memory | $\equiv$ Turing machine | Schuurmans 2023 |
| $(2,2)$-restricted (2 tokens/step) | $\equiv$ Turing machine | Schuurmans et al. 2024 |
| $(2,1)$-restricted (Appending variant) | $\equiv$ $O(n)$-space TM (DCSL) | Schuurmans et al. 2024 |
| Appending-style (Prop. 5.2 + 5.4) | $\equiv$ $O(n)$-space TM (DCSL) | Ours |
| Summarization-style (Prop. 5.1) | $\leq$ $O(1)$-space TM (REG) | Ours |

### Key Findings
- For the same fixed $T$, the gap between two "simple" context managers $C$ is the difference between REG and DCSL—a humongous jump in the complexity hierarchy.
- Summarization-style management cannot recognize basic non-regular languages. This serves as a **theoretical upper-bound warning** for engineering: `/compact` cannot rescue capabilities beyond regular languages.
- Turing-completeness requires **multi-token decoding + context write-back** or **external memory utilities**. This aligns with Schuurmans' work, confirming that ReAct/Agent/external memory systems can theoretically break through the DCSL barrier.

## Highlights & Insights
- **Agent Abstraction**: Abstracting LLM Agents as $(T,D,C)$ is a very clean formalization. Prompt engineering, context compression, memory tools, and tool calls can all be discussed under $C$, giving "agentic" superiority a computable definition.
- **Circuit Family Analogy**: The analogy between scaling-families and circuit families is sharp. It highlights that the $O(\log n)$ precision requirement corresponds to resource bounds rather than universality, a critique applicable to any "X network is Turing-complete" claim.
- **Lemma 5.3 Proof**: Proving a window-size 2 Transformer can implement any $\Sigma^2\to\Sigma$ function is the crucial link connecting Schuurmans' framework to specific network structures, showing that even minimal Transformers maintain linear-space capabilities.

## Limitations & Future Work
- The paper repeatedly states $C$ must be "simple enough," but the boundary (e.g., what constitutes an $O(1)$ state or a fixed local operation) is somewhat informal, leaving grey areas for things like RAG vector retrieval.
- It only covers deterministic, greedy, single-token settings. Nondeterministic decoding and temperature-based stochastic systems are mentioned only in footnotes without a formal probabilistic capability hierarchy.
- Proposition 5.1 assumes a single-token summarizer. In reality, `/compact` outputs $\Theta(N)$ summaries in a chain-of-thought manner. Whether this maintains the constant-space conclusion requires more detailed proof.
- Turing-completeness only addresses "computability," not "learnability" or "generalization"—the largest gap between LLM theory and practice.

## Related Work & Insights
- **vs. Pérez 2019 / Merrill & Sabharwal 2024**: These are recognized as resource bounds in the scaling-family regime rather than fixed-system Turing-completeness.
- **vs. Schuurmans 2023 / 2024**: This work shares the technical route that the decoding/memory interface determines capability. This paper uses the $(N,K)$-restricted system as a black box and extends the framework to summarization-style management.
- **vs. Akhlaghpour 2024**: While the blog "Are Transformers Turing-Complete?" provides intuitive skepticism, this paper formalizes those intuitions into the fixed/scaling dichotomy and provides provable propositions.
- **Value**: Any claim regarding the Turing-completeness of a neural network should first answer "what is fixed and what scales." For Agent designers, the choice of context manager ceiling is more critical than the base model choice.

## Rating
- Novelty: ⭐⭐⭐⭐ (Refines the semantics of community claims rather than proposing a new model.)
- Experimental Thoroughness: ⭐⭐⭐ (Position paper with no numerical experiments, but formal proofs support the stance.)
- Writing Quality: ⭐⭐⭐⭐ (Clear roadmap and definitions; however, the "simple manager" boundary is slightly informal.)
- Value: ⭐⭐⭐⭐⭐ (Corrects a widely miscited theoretical narrative and shifts focus to context management/agent harnesses.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Adversarial ML for LLMs Is Not Making Any Progress](position_adversarial_ml_for_llms_is_not_making_any_progress.md)
- [\[ICLR 2026\] AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](../../ICLR2026/llm_nlp/assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](../../NeurIPS2025/llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)

</div>

<!-- RELATED:END -->
