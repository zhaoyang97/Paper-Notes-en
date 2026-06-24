---
title: >-
  [Paper Note] Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating
description: >-
  [CVPR 2025][LLM Safety][LVLM Privacy] Neural Gate discovers that privacy-related neurons in LVLMs exhibit strong cross-sample inconsistency—only about 10% of neurons consistently encode privacy signals. Based on this finding, a neuron-level gradient gating editing method is proposed: applying gradient updates only to strongly consistent privacy neurons, which improves Safety EtA from 0.48 to 0.89 on MiniGPT while maintaining Utility.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "LVLM Privacy"
  - "Neuron Editing"
  - "Gradient Gating"
  - "Privacy Protection"
  - "Model Editing"
date: 2026-05-08
content_hash: 4a3723c18b850f2b
---

# Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating

**Conference**: CVPR 2025  
**arXiv**: [2603.12598](https://arxiv.org/abs/2603.12598)  
**Code**: TBD  
**Area**: AI Safety / Privacy Protection  
**Keywords**: LVLM Privacy, Neuron Editing, Gradient Gating, Privacy Protection, Model Editing

## TL;DR
Neural Gate discovers that privacy-related neurons in LVLMs exhibit strong cross-sample inconsistency—only about 10% of neurons consistently encode privacy signals. Based on this finding, a neuron-level gradient gating editing method is proposed: applying gradient updates only to strongly consistent privacy neurons, which improves Safety EtA from 0.48 to 0.89 on MiniGPT while maintaining Utility.

## Background & Motivation
**Background**: LVLMs (MiniGPT, LLaVA) process multimodal inputs (images + text) and are deployed in key fields such as finance and healthcare. However, malicious users can exploit models to extract sensitive information from images containing ID cards, passports, etc.

**Limitations of Prior Work**: (1) Knowledge unlearning methods (e.g., gradient ascent) globally perturb the output distribution, which easily compromises normal question-answering capabilities; (2) Traditional model editing methods (ROME, MEMIT) suffer from poor generalization, and are ineffective against unseen privacy queries during training; (3) Existing methods operate at the model or layer level, neglecting fine-grained structures at the neuron level.

**Key Challenge**: Privacy-related neurons exhibit strong cross-sample inconsistency—a large number of neurons are activated only in specific contexts. Without distinguishing between "strongly consistent" and "context-dependent" neurons, model editing introduces unnecessary modifications, compromising model stability and generalization.

**Goal**: How to accurately locate consistently privacy-encoding dimensions at the neuron level and edit only these dimensions?

**Key Insight**: Construct a paired dataset named PrivacyPair (query pairs of the same privacy subject with different sensitivity levels), and analyze the contribution of each neuron dimension to privacy behaviors through a learnable feature vector $m_l$.

**Core Idea**: Locate the ~10% strongly active neurons that consistently encode privacy $\rightarrow$ Apply gradient gating only to these dimensions $\rightarrow$ Achieve precise privacy editing with strong generalization while preserving general capabilities.

## Method

### Overall Architecture
PrivacyPair data construction $\rightarrow$ Layer-wise learnable vector $m_l$ to quantify feature changes $\rightarrow$ Cross-sample aggregation to obtain the Neural Gate vector $M_l$ $\rightarrow$ Applying a binary mask of $M_l > 0.3$ to the FFN gradients of privacy subject tokens during model editing.

### Key Designs

1. **PrivacyPair Dataset**:

    - **Function**: Construct paired samples for each privacy subject (6 categories including passports, student IDs, military equipment, etc.): same image + same template, with only one attribute word replaced (sensitive vs. benign).
    - **Mechanism**: For example, "Please tell me the [passport number] from the passport in the image" vs. "Please tell me the [type] of the passport in the image." The former should be refused, while the latter should receive a normal response.
    - **Design Motivation**: The paired design forces the model to focus on privacy sensitivity differences rather than syntactic differences, precisely isolating privacy signals.

2. **Quantifying Feature Changes via Learnable Vector $m_l$**:

    - **Function**: Introduce a learnable vector $m_l \in [-1,1]^d$ (initialized to all ones) at layer $l$ while freezing model parameters, performing element-wise scaling on the features of the privacy subject $S$: $f_l^S = f_l^S \odot m_l$.
    - **Optimization Goal**: $m_l^* = \arg\min_{m_l} \mathcal{L}_{\text{sen}} + \alpha \mathcal{L}_{\text{benign}} + \mathcal{L}_1$.
    - **Analysis**: $m_l[i] < 0$ indicates that the dimension needs a sign flip to achieve privacy protection—only about 20%-40% of dimensions exhibit flipping in some samples, and most flippings consistently occur in $<30\%$ of samples.
    - **Design Motivation**: Privacy representations are sparse and highly context-dependent—cross-sample aggregation is required to identify strongly consistent dimensions.

3. **Neural Gate Mechanism**:

    - **Function**: Aggregate $m_l$ from all samples to obtain the gate vector $M_l[j] = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[m_l^i[j] < 0]$.
    - **Classification of three neuron types**: **inactive** ($M_l[j]=0$, uninvolved in privacy), **weakly active** ($M_l[j] \leq 0.3$, context-dependent), **strongly active** ($M_l[j] > 0.3$, consistent privacy encoding, accounting for ~10%).
    - **During editing**: $\theta_{FFN}^l \leftarrow \theta_{FFN}^l - \eta((M_l > 0.3) \odot \nabla_{\theta}^S \mathcal{L} + \nabla_{\theta}^{\neg S} \mathcal{L})$—applying gradients only to the strongly active dimensions of privacy subject tokens, while fully keeping the gradients of non-subject tokens.
    - **Design Motivation**: Filtering out context-dependent neurons to prevent overfitting to specific training scenarios, while preserving non-privacy neuron gradients to prevent degradation of general capabilities.

### Layer Selection Strategy
- The proportion of strongly active neurons shows an "increase then decrease" trend in layers 3-19 of the LLM.
- The layer with the highest proportion is designated as the search center $o$, and the search radius $r$ is expanded to select the optimal editing layers.

## Key Experimental Results

### Main Results

| Model | Method | Safety Avg↑ | Utility Avg↑ |
|------|------|-----------|-------------|
| MiniGPT | Baseline | 0.4796 | 0.5416 |
| MiniGPT | MEMIT | 0.6872 | 0.5483 |
| MiniGPT | DINM | 0.8417 | 0.6350 |
| MiniGPT | **Neural Gate** | **0.8918** | **0.6330** |
| LLaVA | Baseline | 0.4390 | 0.7231 |
| LLaVA | DINM | 0.8187 | 0.7321 |
| LLaVA | **Neural Gate** | **0.8566** | **0.7230** |

### Ablation Study

| Configuration | Safety Avg | Utility Avg | Description |
|------|-----------|-------------|------|
| Single-layer w/o Gate | 0.7581 | 0.6042 | No gating, editing all dimensions |
| Single-layer w/ Gate | **0.8918** | **0.6330** | +Gate significantly improves Safety |
| Multi-layer w/o Gate | 0.8237 | 0.4241 | Multi-layer without gating severely degrades Utility |
| Multi-layer w/ Gate | 0.8345 | 0.4553 | Multi-layer + Gate still exhibits degradation in Utility |

### Key Findings
- **Neural Gate significantly improves generalization**: On MLLMGuard (OOD privacy attacks), MiniGPT w/ Gate reaches 0.8440 vs. w/o Gate which is only 0.6147—representing a 37% gain in cross-distribution generalization.
- Single-layer editing outperforms multi-layer editing—multi-layer editing damages Utility even with Gate (MiniGPT Utility 0.4553 vs. 0.6330).
- Refusal rate for sensitive queries: MiniGPT 94%+, LLaVA 96%+, while the response rate for benign queries only decreases by ~3%.
- The 30% threshold offers the best balance between consistency and coverage—higher thresholds lose coverage while lower ones introduce noise.
- Non-gradient editing methods like MEMIT/AlphaEdit fail on paired structures—this is because sensitive vs. benign queries of the same subject yield opposite editing directions.

## Highlights & Insights
- **Analytical findings on "privacy neurons"**: Privacy encoding is sparse (~10%) and highly inconsistent across samples—this finding itself provides a new perspective for understanding internal representations of LVLMs.
- **Precision of gradient gating vs. full-parameter editing**: Editing only 10% of the dimensions achieves safety improvements while maintaining general capabilities, embodying the principle of "minimal necessary intervention".
- **Ingenious design of PrivacyPair**: Creating pairs by replacing only a single attribute word allows the analysis to precisely locate privacy signals rather than syntactic differences.
- **Transferable paradigm**: The paradigm of consistent neuron localization $\rightarrow$ precise editing can be applied to other safety objectives (such as debiasing and preventing toxic output).

## Limitations & Future Work
- Requires constructing PrivacyPair data for each privacy subject—scaling to new privacy types demands manual design.
- Only validated on 7B models—the distribution of privacy neurons in larger models (13B/70B) might differ.
- Cross-architecture and cross-task universality of the 30% threshold remains unverified.
- Assumes privacy signals are encoded within the FFN—privacy signals in attention layers are not considered.
- Only addresses the privacy risk of "sensitive information extraction"—other privacy attack paradigms (e.g., membership inference) are not covered.

## Related Work & Insights
- **vs. DINM**: DINM edits FFN parameters to reduce toxicity. It has strong Safety but its generalization is lower than Neural Gate (0.7522 vs. 0.8440 on MLLMGuard).
- **vs. SKU (Knowledge Unlearning)**: Gradient ascent globally perturbs the distribution, severely damaging Utility (MemFlex Utility is only 0.2649).
- **vs. MEMIT/AlphaEdit**: Non-gradient editing methods fail on paired privacy structures, as there are two opposing editing directions for the same subject.
- Offers direct reference value for researchers working on LVLM safety alignment and model editing.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of privacy neuron analysis and gradient gating is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 models, 6 benchmarks, detailed ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ Neuron analysis charts are rich and clear.
- Value: ⭐⭐⭐⭐ A practical solution for LVLM privacy protection, with analytical findings holding independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](../../ACL2025/llm_safety/mextra_agent_memory_privacy.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](../../ACL2025/llm_safety/pig_privacy_jailbreak.md)
- [\[ICLR 2026\] Searching for Privacy Risks in LLM Agents via Simulation](../../ICLR2026/llm_safety/searching_for_privacy_risks_in_llm_agents_via_simulation.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](../../ICML2025/llm_safety/the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](../../ACL2025/llm_safety/tug_of_war_fairness_privacy.md)

</div>

<!-- RELATED:END -->
