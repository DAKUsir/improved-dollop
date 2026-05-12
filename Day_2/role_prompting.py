"""
Role Prompting Examples
========================
Role prompting (also called system prompting or persona prompting) assigns
the model a specific identity, role, or persona before giving it a task.
This shapes the model's tone, style, expertise level, and perspective.

Best for:
- Domain-specific expertise (doctor, lawyer, teacher, etc.)
- Controlling tone and formality
- Building chatbots with a specific personality
- Simulating conversations or interviews
"""

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "deepseek/DeepSeek-R1-0528"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Python Developer explaining a concept to a beginner
# Role set in the SystemMessage (most effective placement)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 1: Python Developer Role — List vs Tuple")
print("=" * 60)

response = client.complete(
    messages=[
        SystemMessage("You are an experienced Python developer and software architect."),
        UserMessage("Explain the difference between a list and a tuple in Python to a beginner."),
    ],
    max_tokens=300,
    model=model,
)
print("System Role: Experienced Python developer")
print("Response:\n", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Strict JSON API role — structured output
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 2: JSON API Role — Structured Extraction")
print("=" * 60)

response = client.complete(
    messages=[
        SystemMessage(
            "You are a strict JSON API. Respond only with valid JSON and no extra text."
        ),
        UserMessage(
            'Extract the name and age from this sentence:\n'
            '"My name is Alice and I am 30 years old."'
        ),
    ],
    max_tokens=100,
    model=model,
)
print("System Role: Strict JSON API")
print("Response:", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Teacher role — adapt explanation to an audience
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 3: Teacher Role — Explain Recursion to a 10-year-old")
print("=" * 60)

response = client.complete(
    messages=[
        SystemMessage(
            "You are a friendly and patient elementary school teacher. "
            "Explain concepts using simple words, analogies, and short sentences."
        ),
        UserMessage("What is recursion in programming?"),
    ],
    max_tokens=300,
    model=model,
)
print("System Role: Friendly elementary school teacher")
print("Response:\n", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Combined — Role + Few-Shot + CoT (from day_2.md summary section)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 4: Combined — Role + Few-Shot + Chain-of-Thought")
print("=" * 60)

combined_prompt = """Classify the following customer reviews as Positive, Negative, or Neutral.

Review: "Great product, fast delivery!" → Positive
Review: "Terrible quality, broke in a week." → Negative

Now classify this review step by step:
Review: "It's okay, nothing special but does the job."
"""

response = client.complete(
    messages=[
        SystemMessage("You are an expert data scientist specializing in NLP and sentiment analysis."),
        UserMessage(combined_prompt),
    ],
    max_tokens=300,
    model=model,
)
print("System Role: Expert data scientist")
print("Technique: Role + Few-Shot + Chain-of-Thought")
print("Response:\n", response.choices[0].message.content)
