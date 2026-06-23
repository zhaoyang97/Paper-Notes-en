---
title: >-
  [Paper Note] On the Reasoning Abilities of Masked Diffusion Language Models
description: >-
  [ICLR 2026][LLM Reasoning][Chain-of-Thought] This paper provides the first formal characterization of the reasoning capabilities of Masked Diffusion Language Models (MDM). It proves that MDM in a finite-precision logarithmic-width setting is strictly equivalent to "Padded Looping Transformers (PLT)," capable of simulating all problems solvable by Chain-of-Thought
tags:
  - ICLR 2026
  - LLM Reasoning
  - Chain-of-Thought
date: 2026-05-08
content_hash: d4745a06738f026c
---
# On the Reasoning Abilities of Masked Diffusion Language Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BVnIsh4Nz1](https://openreview.net/forum?id=BVnIsh4Nz1)  
**Code**: To be confirmed (Purely theoretical paper, no code)  
**Area**: LLM Reasoning / Diffusion Language Models / Expressivity Theory  
**Keywords**: Masked Diffusion Models, Parallel Reasoning, Chain-of-Thought, Looping Transformer, Circuit Complexity

## TL;DR
This paper provides the first formal characterization of the reasoning capabilities of Masked Diffusion Language Models (MDM). It proves that MDM in a finite-precision logarithmic-width setting is strictly equivalent to "Padded Looping Transformers (PLT)," capable of simulating all problems solvable by Chain-of-Thought (CoT), while being strictly more efficient than CoT on parallelizable problems (e.g., regular languages)—revealing the "Sequentiality Bottleneck" of CoT.

## Background & Motivation
**Background**: Autoregressive language models (such as the GPT series) generate tokens sequentially. Chain-of-Thought (CoT) enhances their expressivity by explicitly writing out intermediate steps, which is the core mechanism of today's "reasoning models." An alternative technical route is the Masked Diffusion Model (MDM): starting from a fully masked sequence, it iteratively "selects positions → fills symbols," enabling the **parallel** generation of multiple tokens. MDM has shown potential to rival autoregressive models in language modeling, coding, and molecular design.

**Limitations of Prior Work**: Although MDM is gaining popularity in engineering, it remains theoretically unclear "exactly what it can compute and how fast." Existing theoretical work on MDM focuses only on the **convergence of its reverse denoising process**, concluding that even for simple probabilistic regular languages, the number of denoising steps must grow linearly with sequence length. However, such conclusions rely on strong assumptions like "uniform unmasking + perfect approximation" and are asymptotic, making it difficult to judge actual reasoning capabilities.

**Key Challenge**: While the expressivity theory of autoregressive LMs is well-developed (falling into circuit classes like $\mathsf{AC}^0$ / $\mathsf{TC}^0$), this theory cannot be directly applied to MDM because its processing is **non-sequential and parallelizable**. Theoretical research into these two paradigms is largely disconnected. A framework is needed that is both **formally rigorous** and **aligned with practical implementations** to characterize how MDM combines "parallelism + iterative refinement" for formal reasoning.

**Goal**: (1) Provide provable problem classes solvable by MDM and their efficiency; (2) Link MDM to two well-understood reasoning frameworks—CoT and PLT—to locate MDM within classical complexity theory.

**Key Insight**: The authors note that previous pessimistic theories assumed MDM must select positions for unmasking **uniformly at random**, effectively depriving it of the freedom to "decide which sub-problem to solve first." Actual MDM implementations (selecting positions based on confidence or a learned strategy) possess exactly this freedom. By allowing the model to **decide to solve simple sub-problems first**, it can bypass difficult parts and decompose tasks into a sequence of parallelizable deterministic steps.

**Core Idea**: Abstract the MDM reverse process into two parts: a "planner" that decides which positions to unmask and a "predictor" that fills symbols. By relaxing the uniform unmasking assumption, it is proven that this idealized MDM is equivalent to PLT and can mutually (if inefficiently) simulate CoT—thus clamping the capability boundaries of MDM between three known classes of objects.

## Method

### Overall Architecture
This is a purely theoretical paper; the "method" consists of a **framework for expressivity characterization**. The overall approach: first, abstract practical MDM into an analyzable idealized model, then link this model to two reference frames—PLT and CoT—and finally provide precise boundaries for "what MDM can solve and in how many steps" using classical circuit complexity classes ($\mathsf{AC}^d$, $\mathsf{NC}$, regular languages, etc.).

The unified setting for analysis is the **finite-precision, log-width** Transformer family $\{T_N\}$: each parameter/activation is represented with fixed bits, and the model width grows logarithmically with input length $N$ (a standard assumption in expressivity literature sufficient for unique position identification). Under this setting, the authors characterize MDM via two complementary lines: ① Equivalence with PLT (§3.1, used to link circuit complexity); ② Mutual simulation with CoT (§3.2, used for intuitive efficiency comparison and separation). Formally, $\mathrm{MDM}[T,P]$ denotes the language class recognizable by MDM with at most $T$ denoising steps and $P$ total output/padding symbols; PLT and pCoT are defined similarly.

### Key Designs

**1. Idealized MDM: Planner + Predictor, Relaxing Uniform Unmasking**

The roots of pessimistic conclusions in existing MDM theory lie in two strong assumptions—uniform unmasking (Assumption 2.1: random uniform selection) and perfect approximation (Assumption 2.2: Transformer perfectly models all conditional distributions). The authors prove (Theorem 2.1) that if both hold, the model can only compute functions within $\mathsf{AC}^0$, because uniform unmasking forces the model to be "equally good at any sub-problem," effectively requiring it to predict the final answer in one step.

To address this, this paper splits the reverse process into two components: the **planner** decides which positions to unmask at each step, and the **predictor** samples symbols for those positions. This relaxes Assumption 2.1—standard MDM is just a special case where the "planner = uniform random," while real implementations (picking by confidence or a learned strategy) correspond to a non-trivial planner. Crucially, the authors also allow the planner to **resample already unmasked positions**, overcoming a major MDM weakness—the inability to backtrack and correct earlier errors. Idealized MDM focuses not on training objectives, but on "what the generation process can theoretically achieve when resampling is allowed." The appendix further proves (Thm. C.1) that the planner and predictor can merge into a single "unmasking by confidence" model, making all conclusions applicable to this popular implementation.

**2. MDM Equivalence with Padded Looping Transformer: Bridging to Circuit Complexity**

PLT and MDM are intuitively similar: both **refine information iteratively in parallel**—MDM via unmasking and predicting discrete symbols, PLT via updating the residual stream; denoising steps in MDM naturally correspond to loops in PLT, and padding tokens correspond to tokens to be generated. Theorem 3.1 formalizes this (PLT is equipped with external sampling noise to match MDM's stochasticity):
$$\mathrm{MDM}[T,P]\subseteq \mathrm{PLT}[T,P],\qquad \mathrm{PLT}[T,P]\subseteq \mathrm{MDM}[T,(N+P)D].$$
Where $D$ is the PLT model width. In the $D=O(\log N)$ setting, they differ only by a logarithmic factor in padding length. Consequently (Corollary 3.1), for any $K\ge 1$, $\mathrm{MDM}[T,\tilde O(N^K)]=\mathrm{PLT}[T,O(N^K)]$. The value of this bridge is that PLT expressivity is already precisely characterized by circuit classes. Combining results from London & Kanade, Svete, etc., constant-step MDM equals $\mathsf{L\text{-}uniform}\,\mathsf{AC}^0$ (Corollary 3.3, no stronger than a standard Transformer), while $O(\log^d N)$ steps + polynomial padding equals $\mathsf{L\text{-}uniform}\,\mathsf{AC}^d$ (Corollary 3.4), which converges to $\mathsf{NC}$—all parallelizable problems—as $d\to\infty$. A finer construction (Theorem 3.2) even compresses regular languages into $\mathrm{MDM}[\log N, N]$ using only linear output space.

**3. MDM and CoT Mutual (Inefficient) Simulation: Capability Clamping**

Since PLT lacks practical implementations and is less intuitive, the authors link MDM to the more popular CoT. The intuition is simple: if MDM unmasks only **one** symbol per step, it degenerates into CoT-style sequential generation. A more natural counterpart is pCoT (parallel CoT, predicting multiple symbols per step). Theorem 3.3: $\mathrm{pCoT}[T,P]\subseteq \mathrm{MDM}[T,P+(N+P)^2]$, showing a **quadratic** expansion in padding—this is not a cost of the diffusion process itself, but the cost of "simulating masked attention with unmasked attention" (Lemma D.14, an independently interesting result); if the MDM Transformer already uses causal masking, the expansion disappears. In the reverse direction (Theorem 3.4): $\mathrm{MDM}[T,P]\subseteq \mathrm{pCoT}[T,LT(P+N)]$, showing only **linear** expansion ($L$ is the number of layers, $T$ comes from writing out all padding tokens at each step). Linking both directions (Corollary 3.5), MDM's capabilities are clamped by CoT, and "polynomial-step CoT = $\mathsf{P}$" implies polynomial-step MDM remains within $\mathsf{P}$.

**4. Sequentiality Bottleneck: MDM Strictly Stronger than CoT under Logarithmic Budgets**

With the framework established, the most impactful conclusion is a **strict separation**. It is known that log-step CoT is trapped in $\mathsf{TC}^0$, while Theorem 3.2 places regular languages in $\mathrm{MDM}[\log N, N]$. Combined with the widely accepted "$\mathsf{TC}^0\ne \mathsf{NC}^1$" hypothesis and the $\mathsf{NC}^1$-completeness of certain regular languages, we get (Corollary 3.7):
$$\mathrm{CoT}[\log N]\subsetneq \mathrm{MDM}[\log N, N].$$
That is, under a budget of logarithmic model calls, MDM can solve $\mathsf{NC}^1$-complete regular languages while CoT cannot. The authors name the fundamental inability of CoT to "exploit parallel structures" the **Sequentiality Bottleneck**: because CoT generates step-by-step, it cannot solve tasks that could be decomposed into independent sub-problems in parallel, requiring linear steps where MDM needs only logarithmic steps. A specific example is the state-tracking benchmark—solvable by MDM in log steps but requiring linear steps for CoT.

### Loss & Training
Not applicable—this paper trains no models. All conclusions are provable theorems reproducible via proofs in Appendices D and E.

## Key Experimental Results
This paper contains no empirical experiments; "key data" refers to the theorems characterizing expressivity boundaries. The tables below summarize core results and comparisons.

### Main Results: MDM Expressivity Positioning

| Setting (Denoising Steps / Output Space) | MDM Equivalent / Class | Source | Meaning |
|------|------|------|------|
| Constant steps $O(1)$, Poly output | $\mathsf{L\text{-}uniform}\,\mathsf{AC}^0$ | Cor. 3.3 | No stronger than single Transformer; cannot solve all regular languages |
| $O(\log^d N)$ steps, Poly output | $\mathsf{L\text{-}uniform}\,\mathsf{AC}^d$ | Cor. 3.4 | Converges to $\mathsf{NC}$ (all parallelizable problems) as $d\to\infty$ |
| $\log N$ steps, Linear output $N$ | All regular languages | Thm. 3.2 | More space-efficient than Cor. 3.4 construction |
| Poly steps, Poly output | $\subseteq \mathsf{P}$ | Cor. 3.6 | Does not exceed polynomial-time solvable problems |

### Comparison: MDM vs CoT / PLT

| Comparison Object | Relationship | Source | Interpretation |
|------|------|------|------|
| MDM ↔ PLT | $\mathrm{MDM}[T,\tilde O(N^K)]=\mathrm{PLT}[T,O(N^K)]$ | Cor. 3.1 | Equivalent, save for a logarithmic factor in padding |
| pCoT → MDM | $\mathrm{pCoT}[T,P]\subseteq\mathrm{MDM}[T,P+(N+P)^2]$ | Thm. 3.3 | MDM simulates pCoT with quadratic padding |
| MDM → pCoT | $\mathrm{MDM}[T,P]\subseteq\mathrm{pCoT}[T,LT(P+N)]$ | Thm. 3.4 | pCoT simulates MDM with linear overhead |
| CoT vs MDM (Log steps) | $\mathrm{CoT}[\log N]\subsetneq\mathrm{MDM}[\log N,N]$ | Cor. 3.7 | **Strict separation**: MDM is stronger on parallel problems |

### Key Findings
- **Parallelism is not a panacea**: Under the $\mathsf{NC}\ne\mathsf{P}$ hypothesis, $\mathsf{P}$-complete problems like Circuit Value Problem, Linear Programming, Context-Free Grammar membership, and Horn-SAT cannot benefit from MDM's parallelism. They essentially require "CoT-style" sequential solving—combined with MDM's inability to maintain KV-cache, autoregressive CoT is more efficient for such problems.
- **Discrete communication is a bottleneck**: Denoising steps communicate via generated discrete text, limiting information flow between steps. This necessitates extra output space to store intermediate computations—similar to the gap between "fixed precision vs log precision Transformers" and $\mathsf{AC}$ vs $\mathsf{TC}$.
- **Position encoding is crucial**: PE carrying information the model cannot compute itself (e.g., binary representation of position/length, division/modulus results) is the prerequisite for "recurrence boosting compute power." In Theorem 3.2, the Transformer effectively lookups pre-computed simple information in the PE.

## Highlights & Insights
- **Bridging engineering to mature theory**: MDM previously lacked expressivity characterization. This paper uses the "Idealized MDM $\equiv$ PLT" bridge to port circuit complexity results at once—a case of "clarity through a change in reference frame."
- **Impacting naming of "Sequentiality Bottleneck"**: It upgrades the intuition of "why CoT struggles on parallel tasks" to a provable separation (under $\mathsf{TC}^0\ne\mathsf{NC}^1$) and points out that specific benchmarks like state tracking are log-step solvable for MDM but linear for CoT, offering practical guidance for model selection.
- **First Masked ↔ Unmasked Attention simulation lemma** (Lemmas D.13/D.14) is independently reusable: It shows simulating causal masking with unmasked attention requires quadratic expansion, while the reverse is linear, explaining the cost of MDM's "universal yet inefficient for left-to-right" nature.
- **Transferable perspective**: The decomposition of the generation process into "planner + predictor" for individual expressivity analysis can be transferred to other non-autoregressive/iterative refinement generators (e.g., latent diffusion LM, SSM-based diffusion).

## Limitations & Future Work
- **Purely theoretical + strong idealization**: The construction relies on a "perfect planner + perfect unmasker." How to make a **trained** planner approximate this ideal behavior is a question of learnability, not expressivity. Sensitivity in modular arithmetic (Hahn & Rofin 2024) suggests learning such planning behaviors is hard for Transformers.
- **Lack of empirical validation**: The paper contains no experiments, and literature on "training MDM on formal languages" is sparse, making it hard to validate theoretical gains—the authors list "designing targeted benchmarks for MDM parallel advantages" as future work.
- **Caveat on comparability settings**: CoT uses causal masking while MDM/PLT use unmasked attention; this asymmetry was intentionally kept to reflect mainstream implementations. Budgets for steps/output space across paradigms are not directly comparable; conclusions should be read within the same setting.
- **Self-identified limitation**: The separation is only strictly proven for log-step budgets ($\mathrm{CoT}[\log N]$). The authors **hypothesize** similar separations for polylogarithmic steps, but $\mathrm{CoT}[\log^d N]$ expressivity is not yet formalized.

## Related Work & Insights
- **vs. Existing MDM convergence theory (Feng et al. 2025; Li & Cai 2025)**: They analyze reverse process convergence under "uniform unmasking + perfect approximation" and reach pessimistic conclusions. This paper identifies that uniform unmasking pins the model to $\mathsf{AC}^0$; relaxing it boosts MDM expressivity—diagnosing and surpassing previous roots.
- **vs. CoT expressivity theory (Li et al. 2024; Merrill & Sabharwal 2024)**: They characterize CoT Transformers within $\mathsf{TC}^0$ (log-step) / $\mathsf{P}$ (poly-step). This paper uses this as a reference frame to highlight MDM's advantages in parallelizable tasks and the sequentiality bottleneck.
- **vs. Looping/Padded Transformers (Saunshi et al. 2025; London & Kanade 2025; Svete et al. 2025)**: They established PLT ↔ $\mathsf{AC}^d$ characterizations. This paper leverages MDM≡PLT to import these results, mapping MDM to the existing capability chart of PLT.
- **Insight**: For model selection between "Autoregressive CoT vs Diffusion," this paper provides principled criteria—highly parallelizable, decomposable problems (regular languages, state tracking) favor MDM; inherently sequential, $\mathsf{P}$-complete problems (CFG membership, Horn-SAT) favor CoT. This also provides theoretical motivation for hybrid models (autoregressive chunks + non-autoregressive filling).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal characterization of MDM expressivity; introduces MDM≡PLT and Sequentiality Bottleneck.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical paper; theorems are self-consistent, but lacks empirical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear presentation using takeaways and complexity diagrams.
- Value: ⭐⭐⭐⭐⭐ Provides principled criteria for paradigm selection between Diffusion LM and Autoregressive CoT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization](improving_reasoning_for_diffusion_language_models_via_group_diffusion_policy_opt.md)
- [\[ICLR 2026\] Inpainting-Guided Policy Optimization for Diffusion Large Language Models](inpainting-guided_policy_optimization_for_diffusion_large_language_models.md)
- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Variational Reasoning for Language Models](variational_reasoning_for_language_models.md)
- [\[ICLR 2026\] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](ladir_latent_diffusion_enhances_llms_for_text_reasoning.md)

</div>

<!-- RELATED:END -->
