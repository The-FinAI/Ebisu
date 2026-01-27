task1 = """
You are a Japanese financial expert fluent in Japanese business communication.
Given a financial question and the corresponding company response (both in Japanese),
your task is to determine the company’s underlying intent level and output exactly one label from the following set: {"+2", "+1", "0", "-1", "-2"}, 
The label meanings are provided for explanation only (DO NOT output these texts):
    "+2" : "Strong Commitment"
    "+1" : "Weak or Qualified Commitment"
     "0" : "Neutral or Hedged Intent"
    "-1" : "Weak Refusal"
    "-2" : "Strong Refusal"

Financial Question: {question}
Company Response: {response}

Directly output the chosen label, and do not provide any explanation.
Answer: {Intent Level}
"""


task2 = """
You are a Japanese financial expert specializing in financial disclosure analysis.
Read the following Japanese financial note carefully.

Your task is to extract financial terms and organize them by structure.
Definitions:
    * Maximal financial term
        - A financial term that is not fully contained inside any longer financial term.
    * Nested financial term
        - A financial term that appears inside a maximal financial term.

What to extract:
    * Identify all maximal financial terms in the text.
    * For each maximal term, identify all nested financial terms it contains.
    * Include only domain-specific financial terminology. Do NOT include generic expressions or normal language.
    
Output format (STRICT):
    * Return a JSON list of lists, each inner list corresponds to one maximal financial term:
        - The first element is the maximal term.
        - The remaining elements (if any) are its nested financial terms.
        - If a maximal term has no nested terms, return a list containing only itself.
    * Example structure (illustrative only):
        [
          ["term1_max", "term1_nested1", "term1_nested2"],
          ["term2_max"],
          ["term3_max", "term3_nested1"]
        ]


Text: {Japanese financial disclosure note}

Do not add any explanations or commentary.
Answer: {JSON with ranked financial terms}
"""