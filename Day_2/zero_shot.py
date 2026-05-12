"""
Zero-Shot Prompting Examples
=============================
Zero-shot prompting means giving the model a task WITHOUT any examples.
You simply describe what you want, and the model uses its pre-trained knowledge to respond.

Best for:
- Simple, well-understood tasks
- Quick responses without setup
- Tasks the model has been extensively trained on (e.g., translation, summarization)
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
# Example 1: Sentiment Classification (no examples given)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 1: Sentiment Classification (Zero-Shot)")
print("=" * 60)

response = client.complete(
    messages=[
        UserMessage(
            'Classify the sentiment of the following sentence as Positive, Negative, or Neutral.\n\n'
            'Sentence: "The movie was absolutely fantastic!"'
        ),
    ],
    max_tokens=100,
    model=model,
)
print("Prompt: Classify sentiment of 'The movie was absolutely fantastic!'")
print("Response:", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Text Summarization (zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 2: Text Summarization (Zero-Shot)")
print("=" * 60)

response = client.complete(
    messages=[
        UserMessage(
            "Summarize the following paragraph in one sentence:\n\n"
            "Artificial intelligence (AI) is rapidly transforming industries across the globe. "
            "From healthcare to finance, companies are using AI to automate repetitive tasks, "
            "improve decision-making, and create new products and services. While the benefits "
            "are enormous, there are also concerns about job displacement and ethical implications."
        ),
    ],
    max_tokens=150,
    model=model,
)
print("Prompt: Summarize a paragraph about AI's impact.")
print("Response:", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Language Translation (zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 3: Language Translation (Zero-Shot)")
print("=" * 60)

response = client.complete(
    messages=[
        UserMessage(
            "Translate the following English sentence to French:\n\n"
            "Good morning! How are you doing today?"
        ),
    ],
    max_tokens=100,
    model=model,
)
print("Prompt: Translate 'Good morning! How are you doing today?' to French.")
print("Response:", response.choices[0].message.content)
