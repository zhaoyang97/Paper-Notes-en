---
title: >-
  [Paper Note] Bridging the Skills Gap: A Course Model for Modern Generative AI Education
description: >-
  [AAAI 2026 (EAAI Symposium)][Generative AI Education] This paper proposes a generative AI application course model for undergraduate and graduate computer science students. A mixed-methods survey demonstrates that the co…
tags:
  - "AAAI 2026 (EAAI Symposium)"
  - "Generative AI Education"
  - "Curriculum Design"
  - "Skills Gap"
  - "Computer Science Education"
  - "Software Development"
date: 2026-05-08
content_hash: 5dd3fe8c80496d18
---

# Bridging the Skills Gap: A Course Model for Modern Generative AI Education

**Conference**: AAAI 2026 (EAAI Symposium)  
**arXiv**: [2511.11757](https://arxiv.org/abs/2511.11757)  
**Code**: None  
**Area**: AI Education  
**Keywords**: Generative AI Education, Curriculum Design, Skills Gap, Computer Science Education, Software Development

## TL;DR

This paper proposes a generative AI application course model for undergraduate and graduate computer science students. A mixed-methods survey demonstrates that the course is effective in bridging the generative AI skills gap between industry and academia, with students broadly rating it as valuable and impactful.

## Background & Motivation

**Background**: Generative AI tools such as ChatGPT, GitHub Copilot, and Midjourney have profoundly transformed software development and workflows across industries. Industry demand for talent proficient in generative AI tools has surged, and AI literacy has become a key competitive advantage in the job market.

**Limitations of Prior Work**: The authors identify two salient disconnects: (1) although industry increasingly values generative AI competencies, higher education institutions have yet to formally integrate them into curricula. Even top-ranked computer science departments in the United States predominantly teach the underlying mechanisms and frameworks of AI (e.g., machine learning principles, deep learning architectures) rather than the practical application of existing generative AI tools; (2) students independently adopt generative AI tools outside the classroom, but without formal instruction and guidance, this may lead to misuse, over-reliance, or a failure to realize the tools' full potential.

**Key Challenge**: Educators remain hesitant to teach generative AI tools in the classroom—citing concerns over academic integrity and the rapid iteration of tools—yet the needs of students and industry are real and pressing. This gap between "educational supply" and "market demand" continues to widen.

**Goal**: (1) Design a systematic generative AI application course; (2) evaluate its pedagogical effectiveness; (3) provide a replicable course model and implementation recommendations.

**Key Insight**: The authors draw on first-hand teaching experience at a private research university. The paper is co-authored by the course instructor and a graduate student enrollee, combining data analysis with dual-perspective reflection (instructor + student) to deliver a primary report on course implementation and evaluation.

**Core Idea**: Design and deliver a course focused on the application of generative AI tools in software development, empirically validate its effectiveness through a mixed-methods study, and distill a generalizable course model.

## Method

### Overall Architecture

The course targets undergraduate and graduate computer science students, with the core objective of teaching students to use generative AI tools responsibly and professionally in software development. The course structure encompasses: theoretical instruction (foundational principles of generative AI), hands-on tool practice (usage of mainstream AI coding assistants), project-based learning (completing software engineering projects with AI tools), and ethics discussions (guidelines for responsible AI use).

### Key Designs

1. **Modular Course Content Design**:

    - Function: Provides a comprehensive knowledge framework spanning foundational concepts to advanced applications.
    - Mechanism: The course is divided into multiple modules, including prompt engineering, code generation and review, AI-assisted debugging, AI tool evaluation and selection, and AI ethics and responsible use. Each module contains theoretical instruction and hands-on labs, ensuring immediate application of knowledge.
    - Design Motivation: Given the rapid pace of generative AI tool development, the course emphasizes "methodology" over specific tools, equipping students with the capacity to evaluate and adapt to new tools as they emerge.

2. **Mixed-Methods Evaluation System**:

    - Function: Systematically assesses course effectiveness and student learning experience.
    - Mechanism: Two rounds of mixed-methods surveys are employed, combining quantitative ratings (Likert scale) with qualitative feedback (open-ended questions) to evaluate multiple dimensions including perceived course value, skill improvement, and satisfaction with course content.
    - Design Motivation: A single quantitative metric cannot capture the complexity of education; a mixed-methods approach balances statistical rigor with deeper understanding.

3. **Dual-Perspective Reflection Mechanism**:

    - Function: Provides multi-dimensional insights into course implementation.
    - Mechanism: Uniquely, the paper is co-authored by the course instructor and an enrolled graduate student, each reflecting on the course's strengths, challenges, and directions for improvement from the perspectives of the instructional designer and the learner, respectively.
    - Design Motivation: Educational research often presents only the instructor's perspective, lacking authentic student experience. The dual-perspective authorship enhances the credibility and comprehensiveness of the study.

### Loss & Training

Not applicable (this is an educational research paper, not a technical methods paper).

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | Result | Notes |
|----------------------|--------|--------|-------|
| Course Value | Proportion of students rating course as valuable | Overwhelming majority | Students broadly agreed the course fills an educational gap |
| Course Effectiveness | Proportion of students rating course as effective | Overwhelming majority | Perceived skill improvement was significant |
| Skills Gap | Pre- vs. post-course AI tool proficiency | Significant improvement | Self-assessed AI competency increased substantially |
| Career Readiness | Preparedness for the job market | Improved | Students felt more confident facing AI-driven work environments |

### Ablation Study

| Course Component | Effectiveness | Notes |
|-----------------|---------------|-------|
| Theory + hands-on (complete course) | Best | The complete course experience was most well-received |
| Theory-only | Insufficient | Students require hands-on practice to truly master tool usage |
| Hands-on only | Insufficient | Lack of theoretical grounding led to reduced efficiency |
| With ethics discussion | Added value | Helped students develop awareness of responsible AI use |

### Key Findings

- Overall student satisfaction with the course was very high; students indicated that such courses are critically lacking in current higher education.
- Prompt engineering and AI-assisted code review were identified by students as the most valuable skills acquired.
- Following the course, students approached AI tools with greater deliberateness and strategic awareness rather than uncritical reliance.
- Graduate and undergraduate students differed in their priorities: graduate students were more focused on AI applications in research contexts, while undergraduates were more concerned with career competitiveness.

## Highlights & Insights

- **Timeliness in Filling an Educational Gap**: This is among the rare studies to systematically examine how to teach generative AI applications within formal curricula, offering reference value for computer science education globally.
- **Dual-Perspective Authorship**: The practice of co-authoring with an instructor and a student is uncommon in educational research and yields a more comprehensive and candid course evaluation.
- **Strong Replicability**: The paper provides detailed course structures and implementation recommendations that other institutions can readily reference and adapt.

## Limitations & Future Work

- The study was conducted at a single university, lacking multi-institution validation, and sample size may be limited.
- Long-term tracking of course outcomes is absent—it remains unclear whether students more effectively leverage AI tools in their careers after graduation.
- The rapid iteration of generative AI tools poses an ongoing challenge to the currency of course content.
- The paper does not address AI literacy cultivation for non-CS students, despite equally pressing demand in that population.
- More objective skill assessment metrics (e.g., quality of coding task completion) could be introduced to complement self-reported data.

## Related Work & Insights

- **vs. Traditional AI/ML Courses**: Traditional courses teach AI principles and model training, whereas this course focuses on the application of existing AI tools. The two are complementary rather than mutually exclusive.
- **vs. Industry Training**: Corporate AI training is typically tool-specific, lacking systematicity and academic depth. This course provides a more comprehensive knowledge framework within an academic setting.
- This paper suggests that AI education should not be confined to "teaching machine learning," but should also encompass the cultivation of practical skills in "how to use AI tools."

## Rating

- Novelty: ⭐⭐⭐ The course design itself is not a technical innovation, but the research perspective and dual-perspective authorship are noteworthy
- Experimental Thoroughness: ⭐⭐⭐ The mixed-methods survey is reasonably convincing, though the sample is limited and a control group is absent
- Writing Quality: ⭐⭐⭐⭐ Well-structured; the dual instructor-student narrative is engaging
- Value: ⭐⭐⭐⭐ Offers important practical reference value for the field of AI education

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STEM Faculty Perspectives on Generative AI in Higher Education](stem_faculty_perspectives_on_generative_ai_in_higher_education.md)
- [\[AAAI 2026\] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring](judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](measuring_model_performance_in_the_presence_of_an_intervention.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](../../ICLR2026/others/the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)

</div>

<!-- RELATED:END -->
