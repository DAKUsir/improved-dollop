"""
Few-Shot Prompting Examples
============================
Few-shot prompting provides the model with a small number of input-output
examples (typically 2–5) before asking it to perform the actual task.
This helps the model understand the expected format and behavior.

Best for:
- When zero-shot gives inconsistent or incorrect results
- When you need a specific output format
- Domain-specific or nuanced tasks
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
# Example 1: Sentiment Classification with examples (few-shot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 1: Sentiment Classification (Few-Shot)")
print("=" * 60)

few_shot_sentiment_prompt = """Classify the sentiment of the following sentences.

Sentence: "I love this product!" → Positive
Sentence: "This is the worst experience ever." → Negative
Sentence: "The package arrived on time." → Neutral

Sentence: "I'm not sure how I feel about this update."
"""

response = client.complete(
    messages=[
        UserMessage(few_shot_sentiment_prompt),
    ],
    max_tokens=50,
    model=model,
)
print("Prompt: Classify sentiment with 3 examples provided.")
print("Response:", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Entity Extraction with a consistent output format (few-shot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 2: Entity Extraction (Few-Shot)")
print("=" * 60)

few_shot_entity_prompt = """Extract the city and country from each sentence. Format: City, Country

Sentence: "She moved to Berlin last summer." → Berlin, Germany
Sentence: "The conference was held in Tokyo." → Tokyo, Japan
Sentence: "He was born in São Paulo and grew up there." → São Paulo, Brazil

Sentence: "Our team is opening a new office in Nairobi next year."
"""

response = client.complete(
    messages=[
        UserMessage(few_shot_entity_prompt),
    ],
    max_tokens=50,
    model=model,
)
print("Prompt: Extract city and country with 3 examples provided.")
print("Response:", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Text-to-Emoji Translation (few-shot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 3: Text-to-Emoji Translation (Few-Shot)")
print("=" * 60)

few_shot_emoji_prompt = """Convert each phrase into emojis that represent it.

Phrase: "I love pizza" → ❤️🍕
Phrase: "Happy birthday" → 🎂🎉🎁
Phrase: "Sunny day at the beach" → ☀️🏖️🌊

Phrase: "I'm going to sleep"
"""

response = client.complete(
    messages=[
        UserMessage(few_shot_emoji_prompt),
    ],
    max_tokens=50,
    model=model,
)
print("Prompt: Convert phrase to emojis with 3 examples provided.")
print("Response:", response.choices[0].message.content)
