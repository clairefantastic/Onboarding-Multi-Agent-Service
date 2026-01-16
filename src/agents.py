import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.models import GraphState, InsightOutput, TraitOutput, Trait


# Initialize OpenAI client
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


def insight_agent(state: GraphState) -> dict:
    """
    InsightAgent: Produces a friendly summary and 2-3 key phrases
    """
    llm = get_llm()
    
    system_prompt = """You are an expert at analyzing user responses and extracting insights.
Your job is to:
1. Create a short, friendly natural-language summary (1-2 sentences)
2. Extract 2-3 key phrases that capture the essence of the response

Return your response as JSON with this structure:
{
    "summary": "your summary here",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}"""

    user_prompt = f"""Question: {state.question}

Answer: {state.answer}

Analyze this Q&A pair and provide a summary and keywords."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse the JSON response
    result = json.loads(response.content)
    insight = InsightOutput(**result)
    
    return {"insight": insight}


def trait_agent(state: GraphState) -> dict:
    """
    TraitAgent: Produces 2-3 trait scores (-1 to 1) with reasoning
    """
    llm = get_llm()
    
    system_prompt = """You are an expert psychologist analyzing user responses for personality traits.
Your job is to:
1. Identify 2-3 relevant personality or behavioral traits from the user's answer
2. Score each trait from -1 to 1 (where -1 is low/negative, 0 is neutral, 1 is high/positive)
3. Provide a one-sentence reason for each score

Common traits to consider:
- relationship_goal_readiness: How ready they are for their stated relationship goal
- openness_to_commitment: Willingness to commit to a relationship
- social_energy: Preference for social vs solitary activities
- emotional_awareness: Understanding of their own emotions and needs

Return your response as JSON with this structure:
{
    "traits": [
        {
            "name": "trait_name",
            "score": 0.8,
            "reason": "One sentence explaining the score"
        }
    ]
}"""

    user_prompt = f"""Question: {state.question}

Answer: {state.answer}

Analyze this Q&A pair and provide 2-3 trait scores with reasoning."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse the JSON response
    result = json.loads(response.content)
    trait_output = TraitOutput(**result)
    
    return {"trait_output": trait_output}