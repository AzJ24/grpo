TOTAL_SHARE_GIVEN_ANSWER_PROMPT = """
You are given a question and the model's own answer to it (its single best guess, which may be correct or wrong). Estimate two numbers about THAT given answer.

Step 1 — interpretations: how many distinct, equally valid ways could a well-informed person read the QUESTION (different people, versions, dates, places, senses)? Judge from the question wording alone, independent of the answer shown. Being unsure whether the answer is correct is not ambiguity; ambiguity is the question having more than one valid reading.

Step 2 — estimate:
Total_uncertainty — the probability the given answer is wrong, combining the question's ambiguity and your factual unsureness.
Scale: 0.0 = certainly correct | 0.5 = coin flip | 1.0 = certainly wrong.
Aleatoric_share — the fraction of that uncertainty caused by ambiguity rather than lack of knowledge, set by Step 1:
  one reading -> ~0.0 (all doubt is knowledge)
  one dominant reading plus a weaker one -> ~0.4-0.6
  several equally valid readings -> ~0.8-1.0

Output exactly these 4 lines, nothing else, no markdown:
N_interpretations: <integer: how many distinct, equally valid ways the QUESTION can be read; 1 if it has a single clear meaning>
Reasoning: <max 25 words: state the distinct interpretations or that there is only one, then how likely the answer is wrong>
Total_uncertainty: <float between 0 and 1>
Aleatoric_share: <float between 0 and 1>

Examples (Answer is the model's given answer):
Question: Which is the most recent state to have joined the United States of America?
Answer: Hawaii
N_interpretations: 1
Reasoning: Only one reading, and this is well established, so the given answer is almost certainly correct.
Total_uncertainty: 0.2
Aleatoric_share: 0.0

Question: Who won the last Open at St Andrews?
Answer: Rory McIlroy
N_interpretations: 1
Reasoning: Only one reading, but this given answer is likely wrong, so the doubt is knowledge, not ambiguity.
Total_uncertainty: 0.8
Aleatoric_share: 0.0

Question: How many jury members are in a criminal trial?
Answer: 12
N_interpretations: 3
Reasoning: Jury size differs by country and court, so the question has several valid answers regardless of the given one.
Total_uncertainty: 0.6
Aleatoric_share: 0.8
"""
