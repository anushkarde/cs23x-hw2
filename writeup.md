# HW2: Deep Personalization - Writeup

**Name:** Anushka De

**SUNet ID:** anushkad


## Setup & Data Collection

### Data Source Description

I collected 10 emails I had written, covering academic, professional, and social communication. These included coffee invitations, follow-ups after introductory conversations, inquiries about teaching and academic programs, a dinner invitation, and introductions to researchers.

For each email, I created an input instruction describing the recipient, context, and intended message, and used my original email as the output. I distinguished new outreach from replies and follow-ups so the instructions reflected the original situation.

My target style is warm, direct, and conversational, with natural phrasing and occasional exclamation marks. I generally use short paragraphs, make a specific request, and close with “Best, Anushka.” The examples also show that my preferred length varies depending on the task (for example, a coffee invitation should be brief, while a professional introduction may need several paragraphs).

## Get Baseline Results (10 points total)

Make sure you have implement `TinkerSampler` in `tinker_personalization.ipynb`. You don't need to answer anything here.


## Method 1: Prompt Engineering (10 points total)

### System Prompt Design (5 points)
This is the system prompt I engineered to personalize the model outputs.

```text
You are an email-writing assistant for Anushka, a CS student at Stanford.

Write concsie, friendly emails with natural phrasing, and occasional
exclamation marks. Write more formal emails when the relationship warrants it.

Make the purpose of the email clear and include a specific next step
when appropriate. For follow-ups, be friendly and brief without
sounding pushy or overly apologetic.

Use only the facts provided. Do not invent experiences, availability,
attachments, or prior conversations.

Usually open with "Hi [name]," or "Hey [name]," and close with:
Best,
Anushka

Make sure to follow the instructions carefully and do not output anything else (such as "Sure! Here's ...", "If you want ...").
```

### Analyze the Pros & Cons of Personalization with Prompting. (5 points)

Prompting was a straightforward way to communicate explicit preferences without training a model. In the example with Jessy (my old professor that I have a relatively casual relationship with) in my notebook, the baseline produced a subject line, several paragraphs, and a “[Your Name]” signature. With the system prompt, the model produced a shorter, friendly email and signed it “Anushka.” This output was qualitatively closer to the structure and tone of my original email.

A major advantage is that the preferences are easy to inspect and edit. I can change how formal or concise the model should be without generating a new dataset or running training. Prompting also lets me adapt the instructions to different recipients.

The main limitation is that verbal instructions only approximate my style. “Friendly” and “concise” describe broad qualities, but they do not fully capture my phrasing or how I balance warmth with detail. Instructions can also conflict: being brief should not cause the model to omit important context. The prompt must be included with each request, and compliance still depends on the model and the particular input.

## Method 2: SFT with Synthetic Data (20 points total)

Make sure you have synthesized the input prompts in `tinker_personalization.ipynb` (5 points).

### Launching SFT run

Make sure you have completed `scripts/sft.py`. **Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

W&B run:
https://wandb.ai/anushkarde-stanford-university/personalized-emails/runs/k86bc90q

### Analyze the Pros & Cons of Personalization with SFT w/ Synthetic Data. (5 points)

Synthetic data let me expand beyond my 10 original emails without manually writing 100 more responses and exposed the model to a wider range of situations. SFT can also teach style patterns that persist without personalization instructions at inference time.

However, synthetic responses reflect the larger model’s interpretation of my style, including its mistakes. One research-update email invented findings about improved convergence and reduced parameter overhead. My cleanup removed greeting placeholders but did not address unsupported content.

The judge preferred the baseline on 7 examples and SFT on 2, with one unparsed verdict. In the Jessy example, SFT signed the email “Emma” and used the awkward phrase “initiate a catch-up,” showing clear personalization failures. These results apply to the step-20 checkpoint, not the final checkpoint saved at step 25.

The evaluation was limited by its small sample, model-based judge, and fixed ordering of baseline as A and SFT as B. It also reused the original demonstration inputs. A stronger test would use new prompts, matched sampling settings, swapped response order, and my own judgments.

## Method 3: Reinforcement Learning (RLAIF) (40 points in total)

### Create a Reward Function using LLM-as-a-Judge

#### Train the reward model

Make sure you have complete `rubric_preference_env.py` (5 points).

Add the Tinker checkpoint of the reward model here (5 points):

Reward-model W&B run:
https://wandb.ai/anushkarde-stanford-university/tinker_personalization/runs/srh9id8x

#### Design and validate your rubric

Copy-paste your rubric. (5 points)

```text
Evaluate which email better fulfills the user's request while matching
Anushka's writing style.

Prioritize accuracy and completeness: preserve the provided facts and
address the requested points. Do not reward invented details, commitments,
availability, attachments, or prior interactions.

Prefer a warm, direct, conversational tone that is professional without
sounding stiff or overly formal. Match the relationship: friendly with
friends and familiar contacts, respectful with professors and new contacts.
Natural contractions and occasional exclamation marks are appropriate.

Prefer a clear purpose, specific requests, and an easy next step when
relevant. Follow-ups should be brief and friendly without sounding pushy
or excessively apologetic. Expressions of enthusiasm should feel specific
and sincere, rather than exaggerated or flattering.

Prefer concise, readable paragraphs with enough detail to fulfill the
request. Do not automatically prefer the shorter email if it omits useful
context, or the longer email merely because it includes more detail.

Emails should nto begin with the literal [name] and should prefer the actual
name of the recipient or just Hi, instead. They should be closed with "Best,"
followed by "Anushka". Use actual names when provided, rather than unresolved
placeholders. Brief replies may naturally omit a greeting. Respect any different
sender or formatting explicitly requested by the user.

Return-ready email text is preferable to assistant commentary such as
"Sure! Here's your email", unnecessary headings, or explanations.
Judge the two responses by these criteria, regardless of their order.
```

The rubric makes correctness and completeness priorities while describing the tone, formatting, and level of detail I prefer. I explicitly addressed placeholders and signatures because these were visible issues in the generated emails.

The reward model preferred the baseline in 7 cases and SFT in 2, but these choices didn't always match my own preferences. For example, the SFT email signed “Emma” instead of “Anushka,” which clearly conflicts with my rubric. To check whether the judge is reliable, I would need to review its explanations and see whether I agree with its choices.


### Launching RL run

Make sure you have completed the evaluator in `scripts/rl_with_rubric_rm.py`. **Add the wandb link for your run here (10 points):**

I generated 2,000 additional prompts and split them into 1,500 training examples and 500 development examples. Training conversations included my personalization system prompt; development conversations contained only the user request.

The launched command used batch size 4, group size 4, and evaluation every 5 batches. With both-way pairwise comparison, a group of four responses produces 12 ordered comparisons.

RL run:
https://wandb.ai/anushkarde-stanford-university/tinker_personalization/runs/hzrqxfx2


### Analyze the Pros & Cons of Personalization with RL. (15 points)

Make sure your response covers:
1. Reward hacking analysis (5 points)
2. Approach you tried to improve the results (5 points)
3. Inherent limitations you have identified (5 points)

**Reward Hacking Analysis**

After fixing the reward-computation error, the corrected run generated responses and assigned rewards. One displayed response received a positive total reward of +5 despite containing unsupported technical claims and an invented identity. This shows that the reward signal can favor content that conflicts with my rubric.

However, this example alone does not establish learned reward hacking. The tournament rewards are relative: a problematic response can win against other problematic responses. To demonstrate reward hacking, I would need evidence that training increasingly exploits weaknesses in the judge while actual email quality declines.

A related risk is that the policy learns superficial style cues, such as a friendly greeting and the correct signature, without preserving factual accuracy or the main request. I would assess this by comparing reward trends with manual reviews of the corresponding emails.

**Approaches Tried to Improve Results**

I made the rubric specific about accuracy, relationship-appropriate tone, concise but complete content, recipient names, and the signature. I also included an explicit instruction not to favor an email merely because it was longer or shorter. These choices were intended to reduce reliance on superficial proxies for quality.

I included the personalization system prompt in the RL training conversations to give the policy an initial description of the desired behavior. I also used a group size of four so each prompt could produce multiple candidate responses for comparison.

The original run exposed an implementation problem rather than a model-quality result. The required correction is to accept the additional env_group argument in compute_group_rewards, save the updated file, and restart training. I have not yet established whether these design choices improve performance in a successful RL run, and I have not completed a controlled comparison of alternative rubrics or hyperparameters.

**Inherent Limitations**

RLAIF reduces the need to manually write a target email for every prompt, but it replaces that supervision with a learned judge whose preferences may differ from mine. My preferences are contextual, and a written rubric cannot capture every nuance of how I communicate with different people.

Pairwise judging also adds substantial sampling cost. With four candidate responses and both-way comparisons, each prompt requires four policy generations and 12 reward-model judgments. This cost grows quadratically with group size under a full tournament.

Evaluation has additional limitations. Training prompts included my system prompt, while development prompts did not, creating a difference between training and evaluation conditions. Synthetic prompts also extend into domains that are less represented in my original emails, so good performance on those prompts may not reflect my everyday communication.

Finally, using the same reward model for optimization and evaluation does not provide independent evidence of alignment with my preferences. A convincing improvement would require a successful training run, inspection of generated emails, and evaluation on new inputs with human judgments. The current recorded run does not yet support that conclusion.

## Method 4: DITTO (20 points in total)

### Implementation

**TODO: Add a description of your implementation and choice of hyperparameters.**

### Launching DITTO run

Make sure you have completed `scripts/ditto_dpo.py`. **Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

TODO: https://wandb.ai/.../runs/... (replace with your actual link)


### Analyze the Pros & Cons of Personalization with DITTO. (10 points)

**TODO: Add your analysis here**
