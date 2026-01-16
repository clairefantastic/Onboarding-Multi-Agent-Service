import os
import json
import re
import logging
import asyncio
from typing import Dict, Any
from time import sleep
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.models import GraphState, InsightOutput, TraitOutput, Trait

class AsyncChatOpenAI(ChatOpenAI):
    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize OpenAI client (sync - kept for compatibility)
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key,
        request_timeout=30,  # 30 second timeout
        max_retries=2  # Built-in retry
    )


# Initialize Async OpenAI client
def get_async_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return AsyncChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key,
        request_timeout=30,
        max_retries=2
    )


def extract_json_from_response(text: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles cases where LLM wraps JSON in markdown or adds extra text.
    """
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Direct JSON parse failed, attempting extraction")
    
    # Try extracting from markdown code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try extracting from generic code blocks
    json_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding first JSON object in text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


def call_llm_with_retry(llm, messages, max_attempts=3, agent_name="agent"):
    """
    Call LLM with exponential backoff retry logic (synchronous).
    """
    for attempt in range(max_attempts):
        try:
            logger.info(f"{agent_name}: Attempt {attempt + 1}/{max_attempts}")
            response = llm.invoke(messages)
            logger.info(f"{agent_name}: Successfully received response")
            return response
        
        except Exception as e:
            logger.error(f"{agent_name}: Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_attempts - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                logger.info(f"{agent_name}: Retrying in {wait_time}s...")
                sleep(wait_time)
            else:
                logger.error(f"{agent_name}: All retry attempts exhausted")
                raise


async def call_llm_with_retry_async(llm, messages, max_attempts=3, agent_name="agent"):
    """
    Call LLM with exponential backoff retry logic (asynchronous).
    """
    for attempt in range(max_attempts):
        try:
            logger.info(f"{agent_name}: Async attempt {attempt + 1}/{max_attempts}")
            response = await llm.ainvoke(messages)
            logger.info(f"{agent_name}: Successfully received async response")
            return response
        
        except Exception as e:
            logger.error(f"{agent_name}: Async attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_attempts - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                logger.info(f"{agent_name}: Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"{agent_name}: All async retry attempts exhausted")
                raise


def insight_agent(state: GraphState) -> dict:
    """
    InsightAgent: Produces a friendly summary and 2-3 key phrases
    """
    logger.info(f"InsightAgent: Processing for user {state.user_id}")
    
    try:
        llm = get_llm()
        
        system_prompt = """You are an expert at analyzing user responses and extracting insights.
Your job is to:
1. Create a short, friendly natural-language summary (1-2 sentences)
2. Extract 2-3 key phrases that capture the essence of the response

CRITICAL: Return ONLY valid JSON, no markdown, no extra text.

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
        
        # Call LLM with retry logic
        response = call_llm_with_retry(llm, messages, agent_name="InsightAgent")
        
        # Robust JSON extraction
        result = extract_json_from_response(response.content)
        
        # Validate with Pydantic
        insight = InsightOutput(**result)
        
        logger.info(f"InsightAgent: Successfully generated insight for user {state.user_id}")
        return {"insight": insight}
    
    except Exception as e:
        logger.error(f"InsightAgent: Failed for user {state.user_id}: {str(e)}")
        raise


def trait_agent(state: GraphState) -> dict:
    """
    TraitAgent: Produces 2-3 trait scores (-1 to 1) with reasoning
    """
    logger.info(f"TraitAgent: Processing for user {state.user_id}")
    
    try:
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

CRITICAL: Return ONLY valid JSON, no markdown, no extra text.

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
        
        # Call LLM with retry logic
        response = call_llm_with_retry(llm, messages, agent_name="TraitAgent")
        
        # Robust JSON extraction
        result = extract_json_from_response(response.content)
        
        # Validate with Pydantic
        trait_output = TraitOutput(**result)
        
        logger.info(f"TraitAgent: Successfully generated traits for user {state.user_id}")
        return {"trait_output": trait_output}
    
    except Exception as e:
        logger.error(f"TraitAgent: Failed for user {state.user_id}: {str(e)}")
        raise


# ============================================================================
# ASYNC AGENTS (Non-blocking, better performance)
# ============================================================================

async def insight_agent_async(state: GraphState) -> dict:
    """
    InsightAgent (Async): Produces a friendly summary and 2-3 key phrases
    """
    logger.info(f"InsightAgent (Async): Processing for user {state.user_id}")
    
    try:
        llm = get_async_llm()
        
        system_prompt = """You are an expert at analyzing user responses and extracting insights.
Your job is to:
1. Create a short, friendly natural-language summary (1-2 sentences)
2. Extract 2-3 key phrases that capture the essence of the response

CRITICAL: Return ONLY valid JSON, no markdown, no extra text.

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
        
        # Call LLM with async retry logic
        response = await call_llm_with_retry_async(llm, messages, agent_name="InsightAgent")
        
        # Robust JSON extraction
        result = extract_json_from_response(response.content)
        
        # Validate with Pydantic
        insight = InsightOutput(**result)
        
        logger.info(f"InsightAgent (Async): Successfully generated insight for user {state.user_id}")
        return {"insight": insight}
    
    except Exception as e:
        logger.error(f"InsightAgent (Async): Failed for user {state.user_id}: {str(e)}")
        raise


async def trait_agent_async(state: GraphState) -> dict:
    """
    TraitAgent (Async): Produces 2-3 trait scores (-1 to 1) with reasoning
    """
    logger.info(f"TraitAgent (Async): Processing for user {state.user_id}")
    
    try:
        llm = get_async_llm()
        
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

CRITICAL: Return ONLY valid JSON, no markdown, no extra text.

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
        
        # Call LLM with async retry logic
        response = await call_llm_with_retry_async(llm, messages, agent_name="TraitAgent")
        
        # Robust JSON extraction
        result = extract_json_from_response(response.content)
        
        # Validate with Pydantic
        trait_output = TraitOutput(**result)
        
        logger.info(f"TraitAgent (Async): Successfully generated traits for user {state.user_id}")
        return {"trait_output": trait_output}
    
    except Exception as e:
        logger.error(f"TraitAgent (Async): Failed for user {state.user_id}: {str(e)}")
        raise