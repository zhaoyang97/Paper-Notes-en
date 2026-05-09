---
title: >-
  [Paper Note] Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?
description: >-
  [NeurIPS 2025][Social Computing][AI regulation] This paper challenges the prevailing belief that regulation and innovation are inherently at odds. Through historical analogies from pharmaceuticals, aviation, and welfare systems, combined with an analysis of the Collingridge dilemma, it argues that well-designed regulation serves as the foundation for sustainable innovation rather than an impediment to it. The regulatory sandbox, SME support mechanisms, and other provisions of the EU AI Act are presented as exemplars demonstrating how regulation can accelerate, rather than delay, responsible technological progress.
tags:
  - NeurIPS 2025
  - Social Computing
  - AI regulation
  - fundamental rights
  - EU AI Act
  - responsible innovation
  - Collingridge dilemma
date: 2026-05-08
content_hash: f5122d1cbb4bbc15
---

# Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?

**Conference**: NeurIPS 2025
**arXiv**: [2511.00027](https://arxiv.org/abs/2511.00027)
**Code**: None
**Area**: AI Ethics / AI Governance
**Keywords**: AI regulation, fundamental rights, EU AI Act, responsible innovation, Collingridge dilemma

## TL;DR

This paper challenges the prevailing belief that regulation and innovation are inherently at odds. Through historical analogies from pharmaceuticals, aviation, and welfare systems, combined with an analysis of the Collingridge dilemma, it argues that well-designed regulation serves as the foundation for sustainable innovation rather than an impediment to it. The regulatory sandbox, SME support mechanisms, and other provisions of the EU AI Act are presented as exemplars demonstrating how regulation can accelerate, rather than delay, responsible technological progress.

## Background & Motivation

**Background**: AI has permeated critical infrastructure and decision-making systems, and its failures produce real harms across social, economic, and democratic dimensions. Concurrently, deregulatory advocates invoke "promoting innovation" to push for relaxed AI oversight — the new U.S. administration revoked Executive Order 14110, and the AI Action Plan 2025 explicitly calls for "eliminating red tape and burdensome regulation."

**Limitations of Prior Work**: The deregulatory narrative overlooks a key insight from the Collingridge dilemma — risks are difficult to foresee in the early stages of a technology, yet once it is deeply embedded in society, the cost of course correction becomes prohibitively high or even impossible. Real-world AI harms are already materializing: deepfake audio during the 2023 Slovak elections undermined democratic processes; AI-generated non-consensual intimate imagery — including child sexual abuse material (CSAM) involving minors — has caused concrete harm (the Almendralejo case in Spain, the Taylor Swift deepfake incident); and the Dutch SyRI system was ruled by a court to violate human rights due to its opaque and discriminatory targeting of low-income communities, ultimately precipitating the government's resignation.

**Key Challenge**: Deregulatory advocates frame "regulation vs. innovation" as a zero-sum game, yet historical evidence demonstrates this is a false dichotomy. No domain with significant public impact has flourished in the absence of a regulatory framework.

**Goal**: To argue, on both theoretical and empirical grounds, that (1) regulation and innovation are not antithetical; (2) the EU AI Act offers a viable risk-based governance model; and (3) "innovation" should be redefined — technologies that systematically violate fundamental rights do not merit the label.

**Key Insight**: An interdisciplinary argument drawing on science and technology policy (the Collingridge dilemma), economics (Schumpeter's creative destruction, the Porter hypothesis), law (textual analysis of the EU AI Act), and historical case studies.

**Core Idea**: Regulation is not a brake on innovation but the foundation upon which durable innovation rests — technological ambition requires the discipline of democratic values and fundamental rights.

## Method

### Overall Architecture

The paper adopts a multi-layered argumentative structure: historical analogies refute the claim that "regulation stifles innovation" → concrete risks of deregulated AI are enumerated → the EU AI Act is analyzed as a governance exemplar → transparency, impact assessment, accountability, and AI literacy are examined as operational tools → alternative viewpoints are addressed.

### Key Designs

1. **Historical Refutation of the False Dichotomy**:

    - Three pivotal historical cases: (a) The thalidomide scandal, which caused severe deformities in 10,000+ infants, led to the Kefauver–Harris Amendment and established the foundation of modern drug safety protocols; (b) in 1959 the U.S. recorded a fatal accident probability of 1-in-25,000 per takeoff — following the creation of the FAA and systematic safety regulation, this improved more than 1,000-fold to approximately 1-in-29,000,000; (c) the Dutch SyRI automated fraud-detection system was ruled by a court to violate human rights due to its opacity and discriminatory targeting.
    - Core argument: In each case, harm arose not because regulation existed but because regulation was absent or failed. Historically, effective regulation has often catalyzed rather than obstructed major technological progress.

2. **The Collingridge Dilemma and the Timing of Regulation**:

    - The dilemma stated: "The need for change cannot be seen when change is easy; when the need for change is apparent, change has become expensive, difficult and time-consuming."
    - In the AI context: the continued deployment of AI in high-stakes domains means adverse consequences may not be recognized until after significant harm has occurred.
    - However, Collingridge himself did not regard the dilemma as insurmountable — he advocated for forward-looking governance mechanisms capable of adapting to new evidence.
    - The EU AI Act responds precisely to this challenge: through risk-based classification combined with adaptive mechanisms such as regulatory sandboxes, it strikes a balance between preventing harm and maintaining flexibility.

3. **Specific Mechanisms of the EU AI Act**:

    - **Regulatory Sandboxes (Art. 57)**: Mandatory establishment in each member state by August 2026. These are not deregulation zones but co-regulation spaces — providing legal certainty (iterative regulatory guidance), risk mitigation (the ability to suspend experiments), protection from penalties (no administrative fines for firms adhering to the sandbox plan), and cross-sectoral collaboration. Spain has already launched the first national sandbox.
    - **Real-World Testing (Art. 60)**: Certain high-risk AI systems may undergo real-world testing under informed consent and supervision; market surveillance authorities may veto or halt testing.
    - **SME Support (Art. 62)**: Priority access to sandboxes, awareness training, dedicated communication channels, and facilitated participation in standardization. Micro-enterprises (<10 employees / €2M turnover) may benefit from simplified quality management systems.
    - **Fundamental Rights Impact Assessment — FRIA (Art. 27)**: Shifts governance from reactive compliance toward proactive design by requiring ex ante assessment of potential rights violations.

### Concrete Risks of AI Deregulation

- **Bias and Discrimination**: AI systems learn from historical data and amplify racial, gender, and socioeconomic biases. A literature review identified 152 specific bias-mitigation measures.
- **Unaccountable Decision-Making**: Human oversight is either absent or nominal — reviewers lack the training, authority, or time to exercise meaningful scrutiny. "Ethics outsourcing" — delegating ethical responsibility to algorithms while developers and deployers evade accountability — is a documented phenomenon.
- **Induced Non-Reflection**: Automated systems compel humans to make decisions without questioning authorship or consequences, eroding reflective judgment.

## Key Experimental Results

### Data on Regulatory Impact

| Historical Case | Pre-Regulation | Post-Regulation | Improvement Factor |
|----------------|---------------|----------------|-------------------|
| U.S. aviation fatal accident probability | 1/25,000 (1959) | 1/29,000,000 (recent) | 1,000x+ |
| Drug safety (post-thalidomide) | No systematic clinical trials | Kefauver–Harris mandates trials | Established modern safety framework |
| GDPR impact | — | Catalyzed PET technologies (differential privacy, federated learning) | Technology innovation driven |

### EU AI Act Perception vs. Reality

| Perceived Burden | Actual Benefit |
|-----------------|---------------|
| "Regulation stifles innovation" | Regulatory sandboxes provide a safe experimentation environment |
| "High compliance costs" | SME support + simplified templates |
| "Limits technological progress" | Compliance drives innovation (watermarking, PETs, etc.) |
| "Competitive disadvantage" | Ethical leadership = market differentiation advantage |

### Key Findings

- **The zero-sum narrative is false**: Every domain with significant public impact has flourished through effective regulation, not through deregulation.
- **Regulation can catalyze innovation**: GDPR spurred privacy-enhancing technologies such as differential privacy and federated learning; the AI Act is driving advances in watermarking and copyright protection.
- **The Porter hypothesis applies to AI**: Well-designed regulation incentivizes firms to innovate in ways that generate both social and market benefits.
- **First-mover advantage**: Firms that proactively comply can shape emerging markets and capture first-mover advantages.

## Highlights & Insights

- **Persuasive force of historical analogies**: The 1,000-fold improvement in aviation safety and the thalidomide tragedy make a more compelling case than any abstract argument.
- **Theoretical value of redefining "innovation"**: Drawing on the OECD's definition of responsible innovation, the paper equates innovation with responsible innovation, explicitly rejecting the concept of "innovation" divorced from accountability and ethical considerations.
- **Precise deployment of the Collingridge dilemma**: Rather than merely citing the dilemma, the paper notes that Collingridge himself believed it was surmountable — the key lies in forward-looking, adaptive governance.
- **The Apple privacy case**: Positioning user data privacy as both a business strategy and an ethical imperative successfully reframes compliance as market differentiation.

## Limitations & Future Work

- **European-centric perspective**: A substantial portion of the paper discusses the EU AI Act, with limited attention to regulatory approaches outside the United States (China, Japan, etc.).
- **Absence of quantitative empirical evidence**: The argument relies primarily on historical analogies and policy analysis, without statistical causal analysis of innovation outputs under regulatory vs. non-regulatory conditions.
- **Narrow measurement of innovation**: The value of innovation is assessed mainly through corporate compliance and market trust, with insufficient consideration of impacts on fundamental research.
- **Self-selection bias**: The cases cited (aviation, pharmaceuticals) are "regulatory success stories"; cases where regulation genuinely impeded innovation are not discussed.
- **Insufficient engagement with Wachter et al.'s critiques**: Legal loopholes in the EU AI Act are only briefly mentioned.

## Related Work & Insights

- **vs. Bradford's "Brussels Effect"**: Bradford argues that EU regulation becomes a de facto global standard through diffusion effects; this paper builds on that argument to further demonstrate the innovation-promoting role of regulation itself.
- **vs. the Draghi Report**: The Draghi Report warns that excessive regulation undermines EU digital competitiveness; this paper directly responds to and rebuts that position.
- **vs. Castro & ITIF**: The paper provides a point-by-point rebuttal of the argument that the precautionary principle harms AI progress.

## Rating

- Novelty: ⭐⭐⭐ The argument itself is not new, but the systematic reasoning and historical depth merit recognition.
- Experimental Thoroughness: ⭐⭐⭐ A position paper does not require experiments, though additional quantitative evidence would strengthen the case.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentative structure is rigorous and progressive; alternative viewpoints are addressed fairly and substantively.
- Value: ⭐⭐⭐⭐ Provides an important counterpoint during the current deregulatory wave; valuable as a reference for both policymakers and AI practitioners.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)
- [\[NeurIPS 2025\] IF-GUIDE: Influence Function-Guided Detoxification of LLMs](if-guide_influence_function-guided_detoxification_of_llms.md)
- [\[NeurIPS 2025\] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)

<!-- RELATED:END -->
