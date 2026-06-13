---
title: >-
  [Paper Note] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models
description: >-
  [ICML 2026][LLM Efficiency][Streaming LLM] ProactiveLLM enables streaming LLMs to use their internal states (attention or predictive entropy) to decide "when to speak." By employing masked streaming modeling and synchron…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Streaming LLM"
  - "Proactive Interaction"
  - "Masked Streaming Modeling"
  - "Self-Distillation"
  - "Endogenous Signal"
date: 2026-05-08
content_hash: 575963d70c956ef9
---

# ProactiveLLM: Learning Active Interaction for Streaming Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2606.00523](https://arxiv.org/abs/2606.00523)  
**Code**: Declared open-source, repository link provided at the end of the text.  
**Area**: LLM Efficiency / Streaming Generation  
**Keywords**: Streaming LLM, Proactive Interaction, Masked Streaming Modeling, Self-Distillation, Endogenous Signal  

## TL;DR
ProactiveLLM enables streaming LLMs to use their internal states (attention or predictive entropy) to decide "when to speak." By employing masked streaming modeling and synchronous privileged self-distillation, it learns to perceive "semantic sufficiency" without relying on any external alignment annotations, significantly reducing interaction latency while maintaining performance.

## Background & Motivation
**Background**: Mainstream LLMs follow a "read-then-generate" batch processing paradigm, requiring the entire input stream to be collected before generation starts. Emerging streaming LLMs aim to read and write simultaneously to reduce response latency in scenarios like audio, video, and simultaneous interpretation.

**Limitations of Prior Work**: The core challenge for simultaneous reading and writing is deciding "when to trigger generation." Existing approaches fall into two categories: hard-coded scheduling (wait-$k$, fixed chunk decoding), which ignores fluctuations in context density and leads to either premature hallucination or batch-like lag in non-monotonic alignment tasks (e.g., QA, summarization); and training decision heads with external alignment annotations (timestamps, segmentation labels, reasoning trajectories from strong teachers), which requires re-labeling and re-training for every task, modality, or latency setting.

**Key Challenge**: Both existing approaches essentially treat the generator as a passive follower—following either rigid rules or external alignment signals. The model never has the opportunity to dominate "when to write" based on its own "judgment of semantic sufficiency."

**Goal**: Upgrade the interaction scheduling function $\phi(t)$ from a static rule to a content-aware policy $\phi(t;\theta)$, and completely eliminate dependence on external alignment annotations.

**Key Insight**: The authors hypothesize that an LLM already well-trained on batch processing possesses hidden states that implicitly contain information about "whether the current partial context is sufficient to predict the next token." These patterns simply haven't been explicitly taught to manifest under "incomplete visibility." The strategy is to use masks to simulate streaming visibility and use the "full-visibility version" of the same model as an implicit teacher to activate this perception capability.

**Core Idea**: Decouple "streaming capability learning" from "interaction decision." First, use masked streaming modeling + synchronous privileged self-distillation to cultivate endogenous semantic boundary perception, then attach a plug-and-play decision head (attention-driven or entropy-driven) to translate endogenous signals into read/write decisions.

## Method

### Overall Architecture
ProactiveLLM is built on a streaming-adapted LLM backbone (using group positional encoding to decouple input/output indices, and a Whisper encoder with forced causal masking for the audio end). The overall workflow consists of a two-stage "endogenous training → explicit decision":

1.  **Training Phase**: Jointly optimize three objectives: standard batch NLL (to preserve pre-trained knowledge), Masked Streaming Language Modeling (MSLM) (to learn generation under incomplete inputs), and Synchronous Privileged Self-Distillation (SPSD) (using the batch logits of the same model as a soft teacher to anchor the streaming distribution).
2.  **Inference Phase**: Freeze the LLM backbone and attach a plug-and-play decision head to monitor internal states in real-time. The attention-driven head observes whether the cumulative attention allocated to the input stream is dispersed or focused. The entropy-driven head monitors the Shannon entropy of the next token's predicted distribution. Based on these, it outputs read or write signals to dynamically advance $\phi(t)$.

Formally, streaming generation is formulated as $P(\mathbf{Y}|\mathbf{X})=\prod_{t=1}^{L}P(y_t\mid \mathbf{y}_{<t},\mathbf{X}_{1:\phi(t)};\theta)$, with a mandatory monotonic constraint $\phi(t+1)\geq \phi(t)$. For evaluation, two non-traditional metrics are introduced: Read Coverage $\text{RCO}=\frac{1}{L}\sum_t \phi(t)/M$ to measure cognitive redundancy, and Average Interaction Lag $\text{AIL}=\frac{1}{L}\sum_t (\phi(t)-\phi_{\text{ideal}}(t))$ to measure relative latency.

### Key Designs

1.  **Masked Streaming Language Modeling (MSLM) + Polynomial Budget Allocation**:
    - **Function**: Simulates the monotonic visibility constraints of "read-while-write" during the training phase, allowing the model to learn to predict the next token under various "half-seen" views.
    - **Mechanism**: For each sample, a monotonic visibility boundary $\phi$ is sampled, and attention from output tokens to input tokens exceeding $\phi(t)$ is masked. The objective is $\mathcal{L}_{\text{MSLM}}=-\sum_t \log P(y_t\mid \mathbf{y}_{<t}, \mathbf{x}_{1:\phi(t)};\theta)$. To avoid degenerate samples (forcing generation with almost no context) that lead to hallucinations, the total reading budget $\mathcal{B}=M$ is distributed across $L$ decoding steps using a polynomial distribution $\boldsymbol{\Delta}\sim\text{Multinomial}(\mathcal{B}, \mathbf{w})$, then accumulated as $\phi(t)=\Delta_0+\sum_{k=1}^t\Delta_k$. $\mathbf{w}$ can be set as uniform or biased to control latency preference.
    - **Design Motivation**: Inspired by BERT's masked language modeling, but adapts bidirectional masking into monotonic causal masking to fit streaming semantics. The polynomial budget constrains decision trajectories near reasonable "read rates," preventing degradation while maintaining enough randomness to cover various latency settings, fundamentally allowing one model to support multiple latency requirements.

2.  **Synchronous Privileged Self-Distillation (SPSD)**:
    - **Function**: Provides a stable optimization anchor for streaming views without introducing any external teachers, preventing distribution drift and loss of pre-trained semantic capability due to "training only on fragmented contexts."
    - **Mechanism**: The same parameters run two forward passes simultaneously—batch mode with full $\mathbf{x}$ as the teacher, and streaming mode with $\mathbf{x}_{1:\phi(t)}$ as the student. Top-$k$ truncated KL divergence pulls the streaming distribution toward the batch distribution using a minimal coefficient $\lambda$: $\mathcal{L}_{\text{distill}}=\lambda\cdot\sum_{t=1}^{L} D_{\text{KL}}(P_{\text{batch}}(\cdot\mid \mathbf{x})\,\|\,P_{\text{stream}}(\cdot\mid \mathbf{x}_{1:\phi(t)}))$. Since teacher and student share parameters and update synchronously, the teacher signal evolves with training.
    - **Design Motivation**: Without distillation, the streaming distribution deviates from the pre-trained manifold, causing disordered generation. Excessive distillation stifles the anticipatory capability required for streaming. SPSD uses "soft constraints + top-$k$" as knobs to keep distillation in the sweet spot of "preventing drift without erasing anticipatory behavior." Being a self-distillation of the same model, it requires no extra parameters or external teachers.

3.  **Plug-and-play Decision Heads (Attention / Entropy driven)**:
    - **Function**: Translates the endogenous semantic boundary perception developed during training into explicit read/write decisions to drive $\phi(t)$ online.
    - **Mechanism**: The LLM backbone is frozen, and a lightweight "gateway controller" is attached. The attention-driven head monitors the cumulative attention distribution of output tokens over the input stream—dispersion implies insufficient context grounding (triggering read), while focus on specific input positions triggers write. The entropy-driven head detects the Shannon entropy of the next potential token $H(P_t)=-\sum_{v\in\mathcal{V}} P(\hat{y}_t\mid C_t)\log P(\hat{y}_t\mid C_t)$; high entropy indicates prediction divergence (requiring more read), while low entropy indicates convergence to a confident state (ready to write).
    - **Design Motivation**: Decoupling "capability learning" from "decision learning" allows the same trained ProactiveLLM to be used with any decision head without re-training the whole model for new latency tiers or decision rules. This is a major practical advantage over learning-based alignment baselines that require re-labeling for every setting.

### Loss & Training
The unified objective $\mathcal{L}=\mathcal{L}_{\text{batch}}+\mathcal{L}_{\text{MSLM}}+\lambda \mathcal{D}_{\text{KL}}$ is optimized jointly. The batch term prevents catastrophic forgetting, the MSLM term is the core streaming objective, and the KL term serves as a stability anchor. Backbones used include Qwen2.5-3B-Instruct / Qwen3-4B (text) and Qwen2-Audio-7B-Instruct (speech) for SFT.

## Key Experimental Results

### Main Results
The experiments cover both text and audio modalities. Text tasks include monotonic alignment (IWSLT-17 translation) and non-monotonic alignment (Dialogue Summarization, SQuAD QA, MCTest Multiple Choice). Audio tasks include ASR (LibriSpeech) and Spoken-SQuAD. Representative comparisons on Qwen2.5-3B are shown below:

| Task | Method | Quality ↑ | AIL ↓ | RCO ↓ |
|------|------|--------|-------|-------|
| MT En→De | Batch (Full) | 27.34 BLEU | 8.71 | 1.00 |
| MT En→De | Wait-9 | 21.47 | 6.87 | 0.88 |
| MT En→De | Proactive-Entr | 23.62 | 8.36 | 0.88 |
| Short QA | Batch (Full) | 74.79 F1 | 77.55 | 1.00 |
| Short QA | Wait-9 | 15.14 | -21.32 | 0.19 |
| Short QA | Proactive-Attn | 71.69 | 59.17 | 0.89 |
| Choice QA | Batch (Full) | 88.33 Acc | 204.87 | 1.00 |
| Choice QA | Proactive-Attn | 83.15 | 151.62 | 0.74 |

The most notable figures are in non-monotonic alignment QA: using 78% of the context preserves 97.16% of the offline upper bound, whereas wait-$k$ methods drop to the 8-15 F1 range on Short QA—illustrating how hard-coded scheduling fails when evidence positions are arbitrary.

### Comparison with Learning-based Baselines
The paper also uses alignment annotations generated by Qwen3-32B and GPT-5.4 to train learning-based baselines (Table 2), comparing performance at low and high latency tiers:

| Latency Tier | Method | MT En→Fr BLEU ↑ | Short QA F1 ↑ |
|--------|------|-----------------|---------------|
| Low Latency | Qwen3-32B Label | 24.12 | 29.84 |
| Low Latency | GPT-5.4 Label | 27.18 | 38.12 |
| Low Latency | ProactiveLLM | 26.56 | **48.74** |
| High Latency | Qwen3-32B Label | 27.62 | 42.88 |
| High Latency | GPT-5.4 Label | 30.74 | 50.21 |
| High Latency | ProactiveLLM | 30.38 | **58.36** |

On translation, ProactiveLLM is on par with baselines trained on strong teacher labels. On QA tasks, it significantly outperforms them, suggesting that endogenous signals are more reliable than "external alignment labels" for non-monotonic tasks where evidence positions are difficult to label.

### Key Findings
- **Decision Head Selection**: Attention-driven heads are more stable in non-monotonic tasks (QA, summarization), while entropy-driven heads hold a slight edge in monotonic tasks (translation). In translation, "prediction divergence" correlates more directly with "how many words to read," whereas in QA, "attention focusing on specific evidence segments" is the signal for being ready-to-write.
- **Robust Transfer**: Stability across backbones (Qwen2.5-3B → Qwen3-4B) and modalities (Text → Audio) confirms that "endogenous signals" do not depend on a specific model scale.
- **Polynomial Budget vs. Naive Uniform Sampling**: The former constrains the training distribution to realistic read rates, significantly mitigating hallucinations under short contexts.

## Highlights & Insights
- **Engineering Value of "Endogenous Cues"**: It downgrades "deciding when to speak" from a supervised task requiring external labels to a byproduct measurable via the model's own attention or entropy. This means changing latency levels at deployment requires only swapping decision head thresholds, not re-training the model.
- **Synchronous Self-Distillation is Smarter**: Traditional self-distillation uses old checkpoints or EMA, introducing overhead and lag. Here, the same parameters run two forwards (batch + stream); the teacher is always the "current self with full view," saving storage and allowing the teacher signal to evolve with training.
- **MSLM Adapts BERT Logic to Autoregressive Streaming**: Applying monotonic masking to teach "how to generate under incomplete evidence" essentially simulations the deployment distribution during training. This paradigm is likely transferable to any task requiring prediction under constrained observations (streaming ASR, online decision making).

## Limitations & Future Work
- Decision heads still rely on hand-crafted internal states (attention/entropy). A more general approach would involve the model learning a lightweight classification head to project read/write probabilities from hidden states.
- The choice of polynomial budget distribution $\mathbf{w}$ remains a hyper-parameter. True "adaptive latency" should allow the model to adjust its budget based on task difficulty.
- Experimental backbones are limited to Qwen 3B-7B; validation for ultra-large models (32B+) and multi-turn interactive streaming scenarios is missing.
- Further analysis is needed for the sensitivity of the KL anchor $\lambda$ and top-$k$ selection, as well as the computational overhead of SPSD in long-context settings (running two forwards per step).

## Related Work & Insights
- **vs. Wait-$k$ (Ma et al., 2019)**: Representative of hard-coded scheduling; ProactiveLLM replaces fixed steps with endogenous signals, showing greatest advantage in non-monotonic tasks with irregular evidence positions.
- **vs. Learning-based Baselines (Fu et al., 2025, etc.)**: These rely on strong teachers for timestamps/segmentation labels; ProactiveLLM requires no external labels and even outperforms GPT-5.4-labeled baselines on QA.
- **vs. Streaming Translation (Tong et al., 2025a/b; Arora et al., 2025)**: While these focus on translation quality in streaming settings, their scheduling is often exogenous. This work internalizes the scheduling strategy, expanding the boundaries of streaming LLM capabilities.
- **vs. BERT-style Masked Modeling**: Shares the philosophy of simulating constrained observations, but shifts the goal from "learning bidirectional representations" to "learning monotonic streaming generation" with polynomial budget constraints for autoregressive scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reframing "when to interact" as an endogenous decision rather than static rules or external supervision is a significant conceptual leap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers text/audio modalities and monotonic/non-monotonic tasks with comparisons against strong learning-based baselines, though lacks validation on ultra-large models or long contexts.
- **Writing Quality**: ⭐⭐⭐⭐ Clear definitions of $\phi(t)$, RCO, and AIL; intuitive methodology diagrams. The SPSD section on "why top-$k$ KL" is a bit brief.
- **Value**: ⭐⭐⭐⭐ Direct immediate value for real-time voice assistants, simultaneous interpretation, and streaming video QA. Plug-and-play heads make latency adjustment extremely low-cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](../../ICLR2026/llm_efficiency/deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](../../ICLR2026/llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](../../ACL2026/llm_efficiency/lizard_an_efficient_linearization_framework_for_large_language_models.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](../../ACL2026/llm_efficiency/are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)

</div>

<!-- RELATED:END -->
