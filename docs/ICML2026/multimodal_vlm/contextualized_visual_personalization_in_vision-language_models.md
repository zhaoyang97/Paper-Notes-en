---
title: >-
  [Paper Note] Contextualized Visual Personalization in Vision-Language Models
description: >-
  [ICML2026][Multimodal VLM][Visual Personalization] CoViP converges the open task of "visual personalization based on user historical experience" into a shared underlying process of "personalized image captioning." By uti…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Visual Personalization"
  - "Personalized Captioning"
  - "Reinforcement Learning Post-training"
  - "Contextual Memory"
  - "Multimodal Dialogue"
date: 2026-05-08
content_hash: 88cfdb053c7798dd
---

# Contextualized Visual Personalization in Vision-Language Models

**Conference**: ICML2026  
**arXiv**: [2602.03454](https://arxiv.org/abs/2602.03454)  
**Code**: https://oyt9306.github.io/covip.github.io/ (Project Page)  
**Area**: Multimodal VLM  
**Keywords**: Visual Personalization, Personalized Captioning, Reinforcement Learning Post-training, Contextual Memory, Multimodal Dialogue

## TL;DR
CoViP converges the open task of "visual personalization based on user historical experience" into a shared underlying process of "personalized image captioning." By utilizing RL post-training with verifiable rewards and Caption-Augmented Generation (CAG) at inference time, it enables VLMs to truly "speak human-like" regarding images within interleaved multimodal contexts. It also introduces an MCQA-based diagnostic benchmark to exclude text shortcuts.

## Background & Motivation

**Historical Context**: Current VLMs (LLaVA, Qwen-VL, InternVL, etc.) excel at describing images and performing basic dialogue/VQA. However, "personalization" remains superficial—given a photo, the model might say "a man in a black suit," unaware that this is actually the "brother" mentioned by the user in a previous turn.

**Limitations of Prior Work**: Existing literature on VLM personalization (MyVLM, Yo'LLaVA, TAME, RAP, RePIC, etc.) faces three limitations: (1) only supporting simple attributes or single identities, failing to handle rich contextual "episodic memory"; (2) evaluation metrics relying on "name recall," which allows VLMs to "cheat" by searching for text shortcuts in the context; (3) predominantly using SFT or external memory banks, which are difficult to generalize to arbitrary downstream tasks and require separate training for each new scenario.

**Key Challenge**: Personalization in real-world scenarios is open-ended and long-tail. Users may ask any question related to episodic history, creating a massive output space. Pure task-specific post-training cannot exhaust all prompt formats, yet failing to train at all prevents the model from bridging the gap of "identifying visual concepts in context $\to$ linking to user history $\to$ reusing them in responses."

**Goal**: (1) Formally define the new paradigm of "Contextualized Visual Personalization"; (2) Identify a learnable "shared underlying process" that generalizes to arbitrary downstream tasks; (3) Propose a diagnostic evaluation protocol that prevents text shortcuts.

**Key Insight**: The authors observe that regardless of the downstream task (captioning, VQA, or dialogue), a VLM must first "interpret the current image within the background of user context." This step can be decoupled. By formalizing the VLM's internal computation into $z=h_\theta(c,x)$ (contextual visual encoder) and $y=g_\theta(z,p)$ (task-specific generator), they find that $h_\theta$ is isomorphic to "personalized image captioning," as captioning explicitly externalizes $z$ into natural language.

**Core Idea**: Treat "personalized image captioning" as a proxy task to train $h_\theta$. Use RL with verifiable rewards to enable the model to simultaneously learn "fine-grained recognition of in-context concepts" and "accurate retrieval of corresponding textual experiences." At inference, the model-generated caption is fed back as an additional condition (Caption-Augmented Generation, CAG) to indirectly amplify the personalization quality of various downstream tasks.

## Method

### Overall Architecture
Given a query image $x$, user prompt $p$, and interleaved text-image context $c$, the VLM $f_\theta$ outputs a response $y=f_\theta(c,x,p)$. CoViP splits internal computation into $z=h_\theta(c,x)$ and $y=g_\theta(z,p)$. The pipeline consists of four components:

1. **Personalized Captioning Benchmark Construction**: Uses image-generation VLMs (Gemini-class) to synthesize query images containing 1–4 concepts based on open-source libraries like Unsplash, followed by instruction consistency and visual fidelity filtering. Multi-turn, strictly fact-grounded "user-model" dialogues are generated for each positive sample. Negative samples with visual similarity are retrieved using CLIP-L/14 to form interleaved contexts.
2. **CapEval-QAs Evaluation Protocol**: For each dialogue, an LLM generates 3 factual MCQA pairs $(q_{ik},a_{ik})\sim\mathcal{G}(d_i)$. During evaluation, a judge model $\mathcal{J}$ is provided only with the caption $s$ and question $q_{ik}$. It must correctly answer positive concept questions ($\text{Acc}^+$, measuring "accurate capture of relevant info") and choose "undetermined" for negative concept questions ($\text{Acc}^-$, measuring "avoiding hallucination of irrelevant content").
3. **RL Post-training**: Optimizes the expected verifiable reward via the GSPO algorithm: $\mathbb{E}_{(x,c)\sim\mathcal{D}_{\text{tr}}}\mathbb{E}_{s\sim\pi_\theta(\cdot\mid x,c,p_s)}[r(s,x,c)]$. The reward $r=r_{\text{vis}}+r_{\text{caps}}$ jointly drives recognition and retrieval.
4. **CAG Inference**: The model first generates a caption $s\sim\pi_\theta(\cdot\mid x,c,p_s)$ based on a captioning prompt $p_s$, then appends $s$ to the downstream prompt $p_d$ to produce the final answer $y\sim\pi_\theta(\cdot\mid x,c,p_d,s)$.

### Key Designs

1. **Captioning as Proxy**:
    - **Function**: Converges open-ended, long-tail downstream personalization tasks into a unified, rewardable, and generalizable objective.
    - **Mechanism**: Based on the decomposition $z=h_\theta(c,x),\,y=g_\theta(z,p)$, the authors argue that captioning is target-isomorphic to $h_\theta$. The caption directly externalizes the model's understanding of "image + context" without redundant thinking/reasoning steps. Thus, training a model to write personalized captions effectively trains a high-quality $h_\theta$.
    - **Design Motivation**: Avoid individual SFT or custom rewards for every downstream task. By extracting "personalization capability" as a shared primitive, CoViP differs fundamentally from previous RL personalization methods like RAP/RePIC.

2. **Dual-component Verifiable Reward (Set-level F1 + MCQA-based VR)**:
    - **Function**: Provides feedback on both "concepts correctly recognized" and "useful history retrieved in captions," preventing the model from specializing in only one aspect.
    - **Mechanism**: The recognition reward $r_{\text{vis}}(x,c)=\text{F1}(\hat{H},H)=\frac{2|\hat{H}\cap H|}{|\hat{H}|+|H|}$ uses set-level F1 to score predictions of which in-context concepts appear in the query image. The retrieval reward $r_{\text{caps}}(s,c)$ uses the $\sigma^+(s;QA^+)-\sigma^-(s;QA^-)$ metric from CapEval-QAs. A penalty of $-1$ is applied if captions degenerate ($R(s)>0$). 
    - **Design Motivation**: Previous RL personalization works used BLEU/CIDEr (encouraging text copying) or name recall (too coarse). F1 and MCQA scalarize recognition and retrieval respectively into repeatable, "hard" metrics that are robust against cheating.

3. **Caption-Augmented Generation (CAG)**:
    - **Function**: Reuses the RL-optimized captioning capability for any downstream task at inference time without further training.
    - **Mechanism**: The standard pipeline $y\sim\pi_\theta(\cdot\mid x,c,p_d)$ is modified into two steps: $s\sim\pi_\theta(\cdot\mid x,c,p_s)$ followed by $y\sim\pi_\theta(\cdot\mid x,c,p_d,s)$. The generated caption $s$ serves as explicit conditioning, essentially "letting the model clarify its thoughts before answering."
    - **Design Motivation**: RL-optimized captions contain denser personalized details than direct downstream answers. CAG makes the "internal draft" explicit, alleviating the need for $g_\theta$ to redo the work of $h_\theta$.

### Loss & Training
The policy $\pi_\theta(s\mid x,c,p_s)$ maximizes $\mathbb{E}[r(s,x,c)]$ using GSPO (Group Sequence Policy Optimization). The training set contains 2.8K samples and the test set 1.3K samples. The judge model $\mathcal{J}$ is a fixed external LLM decoupled from the policy model. Rewards are scheduled on the policy side while the judge remains frozen to ensure stable RL signals.

## Key Experimental Results

### Main Results

$\text{Acc}^+$/$\text{Acc}^-$ on CapEval-QAs under 1–4 concepts (excerpts):

| Model | 1-Concept $\text{Acc}^+$ / $\text{Acc}^-$ | 4-Concepts $\text{Acc}^+$ / $\text{Acc}^-$ | Remarks |
|---|---|---|---|
| GPT-4o | 34.2 / 98.2 | 15.3 / 99.2 | Closed-source; high $\text{Acc}^-$ but low $\text{Acc}^+$ |
| GPT-5 | 48.3 / 97.3 | 26.1 / 98.7 | Strongest closed-source baseline |
| Baseline Open-source VLM | Low | Significantly low | Fails at multiple concepts |
| **CoViP (Ours)** | **Significantly exceeds baseline** | **Significantly exceeds baseline** | Large $\text{Acc}^+$ gains across all concept counts |

While closed-source models excel at $\text{Acc}^-$ (avoiding hallucination), their $\text{Acc}^+$ (recall of personalization) is low, indicating a conservative output strategy. CoViP improves both simultaneously through RL.

### Ablation Study
| Configuration | Observation | Explanation |
|---|---|---|
| Full CoViP ($r_{\text{vis}}+r_{\text{caps}}$ + CAG) | Best | Synergy of caption proxy, dual rewards, and CAG |
| w/o $r_{\text{vis}}$ (No F1 reward) | $\text{Acc}^+$ drops | Loss of fine-grained concept distinction; confusion of positive/negative samples |
| w/o $r_{\text{caps}}$ (No MCQA reward) | $\text{Acc}^-$ drops | Captions begin to include irrelevant context content |
| w/o $R(s)$ filter | Output degradation | Recurring failures like repetitive or empty captions |
| w/o CAG (Direct downstream) | Downstream scores drop | Loss of "caption as internal draft" benefit |

### Key Findings
- **Closed-source VLMs approach the ceiling for $\text{Acc}^-$ (98–99) but drop significantly in $\text{Acc}^+$ with more concepts**: They preserve negative accuracy by "saying less," sacrificing personalized recall. CoViP’s dual reward system is designed to correct this behavior.
- **CAG is not a "free lunch" but has low cost**: One extra caption forward pass provides unified gains across all downstream tasks, which is more engineering-efficient than task-specific training.
- **Verifiable rewards prevent reward hacking**: F1 and MCQA are hard metrics, making it nearly impossible for the model to "cheat" by piling keywords or generating longer captions.
- **Diagnostic tasks cover reactive $\to$ proactive**: CoViP is stable across both "passively answering user questions" and "proactively mentioning relevant history," indicating $h_\theta$ has learned general contextual modeling rather than a specific prompt template.

## Highlights & Insights
- Projecting an open task space back into a single learnable proxy task (captioning) is an elegant decoupling: training $h_\theta$ once benefits all $g_\theta$ variants.
- F1-based set-level VR provides much denser feedback than "per-object 0/1 rewards" or simple ROUGE in multi-concept scenarios, offering smooth gradients for multimodal RL.
- The MCQA-based VR approach ties the "evaluation protocol" to the "reward signal." Since the evaluation blocks shortcuts, using the same metric for rewards prevents target misalignment, a method applicable to RAG or any context-faithful scenario.
- CAG is an engineering practice where "the model acts as its own retriever," similar to Chain-of-Thought but lighter—generating only a caption draft without an exhaustive reasoning trace, making it friendly for latency-sensitive products.

## Limitations & Future Work
- The benchmark scale (2.8K/1.3K) is relatively small compared to modern multimodal datasets, and synthetic data from image-generation VLMs may have domain distribution shifts.
- $r_{\text{caps}}$ depends heavily on an external judge model; judge drift could pollute the reward signal.
- The $h_\theta/g_\theta$ decomposition is a functional hypothesis without structural constraints; if internal VLM computations do not follow this order, the transferability of captioning gains may weaken.
- CAG introduces additional inference latency; sensitive scenarios may require caption caching or asynchronous generation.
- Evaluation is still centered on static context and single queries, lacking coverage for evolving user experiences and context eviction in long-term scenarios.

## Related Work & Insights
- **vs MyVLM / Yo'LLaVA**: Early methods supported only single-concept zero-shot personalization using external databases and templates (effectively retrieval). CoViP pushes personalization into long-context, multi-concept scenarios via RL post-training.
- **vs RAP (SFT version)**: RAP uses supervised learning for multi-concept captioning. CoViP utilizes RL with set-level F1 for fine-grained feedback and introduces CAG at inference for zero-training downstream benefits.
- **vs RePIC**: Both use RL + captioning, but RePIC evaluates only "name recall." CoViP proposes CapEval-QAs to score both correct and incorrect content presence and validates generalization across multiple downstream tasks.
- **vs TAME**: TAME employs external VLMs and memory controllers. CoViP takes a pure learning approach (Single Model + Caption Proxy + RL), simplifying deployment by removing the need for extra memory orchestration components.

## Rating
- Novelty: ⭐⭐⭐⭐ Converging all downstream personalization into a captioning proxy is a conceptual breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ CapEval-QAs main table, multi-downstream diagnostics, and various baselines are comprehensive, though real-world long-term user data is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments across problem motivation, method decomposition, and evaluation protocol.
- Value: ⭐⭐⭐⭐ Contextualized personalization is essential for production-grade VLMs. The "proxy task + verifiable reward + inference draft" framework is immediately applicable to industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] Visual Persuasion: What Influences the Decision-Making of Vision-Language Models?](visual_persuasion_what_influences_decisions_of_vision-language_models.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)

</div>

<!-- RELATED:END -->
