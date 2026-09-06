# AI/ML Test Requirements

# Challenge: **Student <> Teacher AI loop for learning a new visual damage concept**
## Context
Build a small end-to-end pipeline showing how a student model can begin learning a new damage concept from only a handful data points, ideally in the visual damage detection space.
You can use:

*   a teacher model via API or use a self-hosted open source weights, e.g. Gemm4, Qwen3.5, etc.
*   a student model that can be adapted locally, e.g. 1B/3B models with (Q)LoRA
*   Matterport 3D spaces to capture screenshots
*   optionally, a small set of tagged images with damage, material, and surface labels

IMPORTANT:
The teacher may not reliably know the concept upfront.

## Your Task
Build a minimal end-to-end pipeline that demonstrates how a smaller model can begin learning a new damage concept (e.g. "bubbling", "sagging", "swelling", etc.) using a teacher-assisted loop.
The concept may be rare and may not be reliably understood by the teacher model initially.

**We propose to focus on learning the "bubbling" concept.**

Your solution should show how you would:

* assemble or generate a small training set (5-10+ images)
*   use the teacher model in a practical way
*   fine-tune or adapt the student model
*   evaluate whether the student has improved
*   show how the loop could continue iterating until reaching a stop criteria or can be reinforced with human feedback

## What you may use
You may use:

*   screenshots taken from the provided Matterport spaces
*   tagged images if provided 
	*   NOTE: damage tags in filenames have been validated and are visible, however it is not guaranteed that all damages within an image are tagged in the filename
	*   images in the hard-negatives folder do not have any damages on them. The damage tag(s) is a remnant of the damages nearby.
*   teacher-generated supervision
*   metadata such as damage label, material, and surface
*   weak supervision, heuristics, or structured prompting
*   AI Coding Tools and external APIs (not provided by SyncTech)


## Damage Taxonomy
*   ash
*   bubbling
*   charring
*   chipped
*   corrosion
*   cracking
*   debris
*   defacement
*   deformation
*   delamination
*   dents
*   efflorescence
*   exposed substrate
*   ghosting
*   holes
*   make safe
*   melt
*   mould
*   peaking / crowning / cupping
*   popped fixings
*   removed / missing
*   rips / tears
*   sagging
*   scratching
*   separation
*   soot
*   stain (other)
*   swelling
*   water stain

## Deliverables
### A. Short design note
Explain:

*   how you approached the problem
*   what assumptions you made
*   how you handled limited or noisy data
*   how you used the teacher model
*   how you measured progress


### B. Runnable code
A lightweight implementation that includes:

*   dataset creation or assembly
*   teacher interaction
*   student training or adaptation
*   evaluation
*   a simple orchestration or repeatable workflow


### C. Results
Show:

*   a few example generated training records
*   before/after results (train VS Evaluation)
*   any metrics, examples, or qualitative observations, OOD tests


### D. Next steps
Briefly describe:

*   what you would improve next
*   what would matter for scale or production use
Constraints
*   Timebox: approximately **half a day to one day**
*   Focus on clarity, judgment, and measurable progress
*   We are not looking for perfect accuracy or a production-ready system
*   Walk us through your solution (online meeting)



## What we care about
We care about whether you can:

*   structure an ambiguous ML problem
*   work pragmatically with little data
*   build a functioning loop, not just isolated scripts
*   evaluate progress sensibly
*   make good trade-offs


### Research and Examples:

**Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks:** [https://arxiv.org/pdf/2004.05937](https://arxiv.org/pdf/2004.05937)

**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models:** [https://arxiv.org/html/2403.02781v4](https://arxiv.org/html/2403.02781v4)

**Training a Student Expert via Semi-Supervised Foundation Model Distillation:** [https://arxiv.org/html/2604.03841](https://arxiv.org/html/2604.03841)

**Teacher-Guided Student Self-Knowledge Distillation Using Diffusion Model:** [https://arxiv.org/html/2602.02107](https://arxiv.org/html/2602.02107)

**Winning solutions of kaggle competitions**

3rd Place Solution: Meta Pseudo Labels + Knowledge Distillation: [https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322832](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322832)

RSNA Screening Mammography Breast Cancer Detection: [https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/372567](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/372567)