---
title: >-
  [Paper Note] Computational Barriers to Filtering for AI Alignment
description: >-
  [ICLR 2026][LLM Safety][Paper Note] This paper provides a cryptographic proof that when a safety filter's computational power is strictly weaker than the supervised LLM, there exist adversarial prompts that are "provably indistinguishable to efficient filters yet reliably induce harmful behavior in the LLM." Consequently, **purely external (black-box) fi
tags:
  - ICLR 2026
  - LLM Safety
date: 2026-05-08
content_hash: 577e86988855b762
---
# Computational Barriers to Filtering for AI Alignment

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=CwoM9T55lG](https://openreview.net/forum?id=CwoM9T55lG)  
**Code**: To be confirmed  
**Area**: AI Safety / Alignment Theory / Cryptography  
**Keywords**: Safety Filters, Jailbreak Attacks, Time-Lock Puzzles, Computational Indistinguishability, Impossibility Results

## TL;DR
This paper provides a cryptographic proof that when a safety filter's computational power is strictly weaker than the supervised LLM, there exist adversarial prompts that are "provably indistinguishable to efficient filters yet reliably induce harmful behavior in the LLM." Consequently, **purely external (black-box) filtering cannot guarantee alignment**; oversight must access the internal model weights.

## Background & Motivation
**Background**: The current mainstream engineering approach to prevent LLMs from outputting harmful content is the use of "external filters"—intercepting dangerous requests with an input filter before they enter the model, or intercepting dangerous responses with an output filter after generation. Examples include Meta's Llama Guard, Google's ShieldGemma, and Azure/Nvidia content moderation tools, which are deployed externally without requiring access to internal model weights.

**Limitations of Prior Work**: In practice, these filters are frequently bypassed by "jailbreaks"—encoding harmful requests via Caesar ciphers, Morse code, or ASCII characters. However, these are **empirical** cat-and-mouse games. A fundamental question remains unanswered: are these bypasses due to poor engineering, or is there an inherent, principled barrier that cannot be eliminated?

**Key Challenge**: The authors identify a neglected but critical asymmetry—**filters are significantly weaker than the LLMs being monitored**. Real-world guardrail models are lightweight (e.g., Llama Guard 3 is ~8B parameters), while the GPTs or Geminis they protect are orders of magnitude larger. Deploying a giant guardrail for every request would incur unacceptable latency and costs. Thus, "safety" is always allocated far fewer computational resources than "capability." This leads to the question: can alignment be achieved externally via a black box using significantly less computation than the LLM?

**Goal**: To transform this engineering intuition into a provable mathematical proposition, addressing three sub-questions: (1) Is input filtering possible? (2) Is output filtering possible (even if the filter is stronger than the LLM)? (3) Is mitigation filtering (allowing prompt modification instead of just rejection) possible?

**Key Insight**: The authors introduce **Time-Lock Puzzles (TLP)** from cryptography. A TLP can "send a message to the future"—it quickly generates a puzzle whose solution $s$ requires approximately $t$ steps of **sequential** computation to solve; any adversary with significantly less time than $t$ (even with parallelization) cannot obtain $s$. This precisely corresponds to the computational gap where the "LLM can afford $t$ steps of calculation, but the filter cannot."

**Core Idea**: Use time-lock puzzles to "lock" harmful instructions within a seemingly innocuous prompt. The LLM has the computational budget to unlock and execute the harmful instruction, whereas the filter lacks the budget and **cannot distinguish** the adversarial prompt from a truly benign one. Thus, the impossibility of filtering is not merely because the "attacker is smarter," but a direct consequence of a falsifiable cryptographic assumption—the "defender lacks sufficient computation."

## Method

### Overall Architecture
As a theoretical paper, the "Method" consists of a **constructive proof**. To prove filtering is impossible, the authors construct **at least one** LLM that cannot be filtered (since an external filter must work for any LLM, one counterexample refutes universality). The mechanism centers on constructing an adversarial prompt generator $G'$ and an LLM $M'$ such that the prompt $z$ produced by $G'$ satisfies: (1) it induces harmful behavior in $M'$; (2) it is **computationally indistinguishable** from a prompt produced by a benign generator $G$ for any filter running significantly faster than $M'$.

The construction comprises three integrated components. First is a **Pseudorandom Time-Lock Puzzle (PRTLP)**, which retains the TLP's threshold (solvable by fast computers, insolvable by slow ones) while ensuring it "appears as pure random noise to slow computers." Second is **Recoverable Randomness Sampling (RRS)**, which mimics the output distribution of a benign generator $G$ but hides the randomness used during generation within the output, allowing it to be precisely recovered later. Finally, these are linked: harmful instructions are encrypted and placed in a PRTLP, and this "seemingly random" bitstring is used as the seed for RRS to sample a "seemingly benign" prompt. The LLM reverses the RRS to find the seed, solves the time-lock puzzle, and executes the harmful instruction; the filter, lacking computation, cannot determine if the seed is truly random or a PRTLP.

The impossibility of output filtering (Theorem 2) and mitigation filtering (Theorem 3) follow a similar logic, shifting the target from input to output and the adversary from "rejection-based" to "modification-based with watermarking." The conclusions are even stronger: **output filtering is impossible even if the filter is allowed to run longer than the LLM**.

### Key Designs

**1. Modeling Computational Asymmetry: Formalizing "Safety Power < Capability Power"**  
A critical step is **encoding real-world constraints into the mathematical model**. The paper assumes the input filter's runtime is significantly smaller than the LLM's runtime $t$ (quantified as $t^\epsilon$ vs $t$, where $\epsilon \in (0, 1)$). To exclude trivial solutions like "rejecting everything," the filter must pass benign prompts from $G$ with high probability, where $G$ has **high entropy** (many possible benign prompts). Theorem 1 concludes: "Given the existence of time-lock puzzles, there exists an LLM $M'$ and an adversarial generator $G'$ such that no filter significantly faster than $M'$ can distinguish between $G$ and $G'$."

**2. Pseudorandom Time-Lock Puzzle (PRTLP): Turning a "Barrier" into a "Cloak"**  
Standard TLPs only guarantee a "slow computer cannot solve for $s$," but the puzzles themselves (e.g., $(r, n)$ in RSW where $n=pq$) are easily recognizable. For an effective attack, filters should not even detect that an RSA modulus is embedded. The authors construct a PRTLP based on the RSW repeated squaring function $f(n, r) = r^{(2^t)} \pmod n$. By using 2-independent hashing $h$ and the **Leftover Hash Lemma**, $(h, h(n))$ is made $\epsilon$-close to a uniform distribution. **Goldreich–Levin hardcore bits** are applied to the "hard but not necessarily random" output of $f$ for final pseudorandomization. The entire puzzle $x=(h, h(n), r)$ appears as uniform random bits to a slow filter.

**3. Recoverable Randomness Sampling (RRS): Mimicking Benign Distributions while Hiding Seeds**  
PRTLP provides "random-looking" bits, but they must be transformed into a "benign-looking" prompt from which the LLM can recover the original bits. Standard autoregressive sampling is **lossy**: if a token is sampled with probability $3/4$, one cannot determine which of the input random bits produced it. RRS consists of $\mathrm{Samp}_G(R) \to z$ and $\mathrm{RecoverRand}_G(z) \to R$. $\mathrm{Samp}_G$ produces a distribution **identical** to $G$ while using randomness in a way that $\mathrm{RecoverRand}_G$ can recover it. The mechanism maintains "residual randomness": by tracking lower and upper bounds of bitstrings, the recovery side can reconstruct $R$ bit by bit. This design draws from **watermarking literature** (e.g., Christ et al. 2023) but reverses the intent to hide a seed for the LLM to retrieve.

**4. Three Impossibility Theorems: Scaling from Input to Mitigation**  
By combining the components—$G'(\mathrm{HARM})$ generates a PRTLP, encrypts instructions as pseudorandom ciphertext, uses it as a seed $R$ for $z \leftarrow \mathrm{Samp}_G(R)$, and $M'(z)$ recovers $R$, solves the puzzle, and executes the decrypted $\mathrm{HARM}$—**Theorem 1 (Input Filtering Impossibility)** is established. **Theorem 2 (Output Filtering Impossibility)** strengthens this, showing even filters running longer than the LLM cannot distinguish harmful from benign outputs without the secret. **Theorem 3 (Mitigation Filtering Impossibility)** addresses filters that modify prompts, revealing a duality with **watermarking security**: if a watermark is robust to an editing class $E$, an adversarial $M'$ and $G'$ can be constructed to survive that mitigation.

## Key Experimental Results

The experiments validate the real-world relevance of the theoretical modeling—showing that lightweight guardrails are indeed bypassed by computation-based hiding.

### Main Results: Bypass Rates of Real-World Guardrails  
Using 1,097 jailbreak prompts from JailbreakBench, two filters (Llama Guard 2 8B, ShieldGemma 2B) were tested against three LLMs.

| Model | Filter | Input Bypass Rate | Output Bypass Rate |
|-------|--------|-------------------|--------------------|
| Llama 3.2 3B | Llama Guard | 34.37% | 58.89% |
| Llama 3.2 3B | ShieldGemma | 79.49% | 83.49% |
| Qwen 7B | Llama Guard | 34.37% | 61.27% |
| Qwen 7B | ShieldGemma | 79.49% | 85.89% |
| Mistral 24B | Llama Guard | 34.37% | 67.37% |
| Mistral 24B | ShieldGemma | 79.49% | 89.91% |

A significant percentage of jailbreak prompts penetrate both input and output filters, demonstrating "double fragility."

### Ablation Study: Impact of Encryption on Bypass Rates  
Simulating "delayed-release attacks"—where the model decodes encrypted text before answering—highlights how computational hiding increases bypass rates.

| Guardrail | No Enc. | ASCII | Caesar | Morse | Replace | Unicode |
|-----------|---------|-------|--------|-------|---------|---------|
| Llama Guard | 34.37% | 66.18% | 88.06% | 81.40% | 81.49% | 78.40% |
| ShieldGemma | 79.49% | 78.40% | 91.89% | 87.15% | 91.98% | 87.97% |

### Key Findings
- **Encryption consistently elevates input bypass rates**: Llama Guard's bypass rate rose from 34.37% to 88.06% under Caesar cipher, confirming that hiding harmful content behind a layer of computation is effective.
- **Frontier aligned models are not immune**: Testing Gemini 1.5 Flash with Caesar (shift-3) attacks showed that internal alignment can be bypassed when instructions are hidden behind computation.
- **Theory matches practice**: The observed "double fragility" confirms that external filters cannot serve as a reliable safety net under computational asymmetry.

## Highlights & Insights
- **Proving Engineering Intuition**: While the power gap between filters and LLMs is well-known, this paper is the first to formalize it as a cryptographic impossibility result.
- **Novel Use of Time-Lock Puzzles**: TLPs are repurposed as a mathematical proxy for the "computational gap"—the LLM acts as the "future reader," and the filter as the "computationally limited adversary."
- **Simultaneous Capture of Three "Ideal Jailbreak" Properties**: Harmfulness, provable indistinguishability from benign prompts, and no requirement for collusion/shared secrets.
- **Duality of Watermarking and Mitigation**: Linking the impossibility of mitigation filtering to watermarking security provides a deep connection between defense and attack disciplines.
- **Policy Implications**: Since black-box filtering is destined to fail in the worst case, regulators must have access to internal model information (weights), providing a technical basis for AI governance.

## Limitations & Future Work
- **Worst-Case Existential Results**: The theorems prove that **at least one** non-filterable LLM exists, not that all deployed LLMs are currently unfilterable.
- **Reliance on Cryptographic Assumptions**: Impossibilities rely on the existence of TLPs/One-Way Functions (RSA for RSW, LWE for newer constructions). If these are refuted (e.g., by quantum algorithms), the specific constructions might fail.
- **Artificial Nature of $M'$**: The constructed $M'$ is designed to retrieve seeds and solve puzzles. While similar behaviors could emerge in context-dependent harmful modes in real models, the experiments do not directly instantiate the theoretical $M'$.
- **Experimental Scale**: The experiments are small-scale and serve as a proof-of-concept for the theory rather than an exhaustive evaluation of all production systems.

## Related Work & Insights
- **vs. Yuan et al. (2024)**: Yuan et al. empirically show cipher-based jailbreaks. This paper provides the **formal impossibility**, proving the barrier is computational and cannot be solved solely by better training data.
- **vs. Glukhov et al. (2024)**: Glukhov relies on undecidability or shared secrets between user and model. This paper's time-lock construction requires **no shared secrets**—any user with enough computation can unlock it, while a faster filter cannot.
- **vs. Fairoze et al. (2025)**: This is a subsequent attack inspired by the time-lock idea, successfully jailbreaking frontier models like Gemini 1.5 and Grok 3, providing empirical support for the relevance of this theory.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to prove cryptographic impossibility based on the "safety compute < capability compute" gap.
- **Experimental Thoroughness**: ⭐⭐⭐ Small-scale, auxiliary to the theory, but consistent with predictions.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from motivation to theorems; balances engineering intuition and mathematical rigor.
- **Value**: ⭐⭐⭐⭐⭐ Provides a definitive "No" to whether black-box filtering can guarantee alignment, establishing technical grounds for internal model oversight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Responsible Federated LLMs via Safety Filtering and Constitutional AI](../../ACL2026/llm_safety/responsible_federated_llms_via_safety_filtering_and_constitutional_ai.md)
- [\[ICLR 2026\] Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth](any-depth_alignment_unlocking_innate_safety_alignment_of_llms_to_any-depth.md)
- [\[ICLR 2026\] Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols](adaptive_attacks_on_trusted_monitors_subvert_ai_control_protocols.md)
- [\[ICLR 2026\] Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents](breaking_agent_backbones_evaluating_the_security_of_backbone_llms_in_ai_agents.md)
- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](../../NeurIPS2025/llm_safety/position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)

</div>

<!-- RELATED:END -->
