import logging
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.models import GraphState, OnboardingInput, OnboardingResponse
from src.agents import insight_agent, trait_agent, insight_agent_async, trait_agent_async

logger = logging.getLogger(__name__)


def create_onboarding_graph():
    """
    Creates a LangGraph workflow that runs two agents in parallel
    """
    logger.debug("Creating LangGraph workflow")
    
    # Create the graph with GraphState as the state schema
    workflow = StateGraph(GraphState)
    
    # Add agent nodes
    workflow.add_node("insight_agent", insight_agent)
    workflow.add_node("trait_agent", trait_agent)
    
    # Add edges for parallel execution
    # Both agents start from START
    workflow.add_edge(START, "insight_agent")
    workflow.add_edge(START, "trait_agent")
    
    # Both agents end at END
    workflow.add_edge("insight_agent", END)
    workflow.add_edge("trait_agent", END)
    
    # Compile the graph
    return workflow.compile()


def process_onboarding(input_data: OnboardingInput) -> OnboardingResponse:
    """
    Process an onboarding Q&A through the multi-agent graph
    
    Args:
        input_data: Validated OnboardingInput
        
    Returns:
        OnboardingResponse with merged results from both agents
        
    Raises:
        ValueError: If input validation fails
        Exception: If agents fail or output is invalid
    """
    logger.info(f"Starting graph processing for user: {input_data.user_id}")
    
    try:
        # Create initial state from input
        initial_state = GraphState(
            user_id=input_data.user_id,
            question=input_data.question,
            answer=input_data.answer
        )
        
        # Create and run the graph
        logger.debug("Creating and compiling graph")
        graph = create_onboarding_graph()
        
        # Run the graph with initial state
        logger.info(f"Executing parallel agents for user: {input_data.user_id}")
        final_state = graph.invoke(initial_state)
        
        # Validate that both agents completed
        if final_state.get("insight") is None:
            raise ValueError("InsightAgent failed to produce output")
        if final_state.get("trait_output") is None:
            raise ValueError("TraitAgent failed to produce output")
        
        # Merge results into final response
        logger.debug("Merging agent results")
        response = OnboardingResponse(
            user_id=final_state["user_id"],
            insight=final_state["insight"],
            traits=final_state["trait_output"].traits
        )
        
        logger.info(f"Successfully completed processing for user: {input_data.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Graph processing failed for user {input_data.user_id}: {e}")
        raise


# ============================================================================
# ASYNC GRAPH PROCESSING 
# ============================================================================

def create_onboarding_graph_async():
    """
    Creates a LangGraph workflow that runs two async agents in parallel
    """
    logger.debug("Creating async LangGraph workflow")
    
    # Create the graph with GraphState as the state schema
    workflow = StateGraph(GraphState)
    
    # Add async agent nodes
    workflow.add_node("insight_agent", insight_agent_async)
    workflow.add_node("trait_agent", trait_agent_async)
    
    # Add edges for parallel execution
    workflow.add_edge(START, "insight_agent")
    workflow.add_edge(START, "trait_agent")
    workflow.add_edge("insight_agent", END)
    workflow.add_edge("trait_agent", END)
    
    # Compile the graph
    return workflow.compile()


async def process_onboarding_async(input_data: OnboardingInput) -> OnboardingResponse:
    """
    Process an onboarding Q&A through the multi-agent graph (Async version)
    
    This is faster than the sync version as LLM calls don't block the event loop.
    """
    logger.info(f"Starting async graph processing for user: {input_data.user_id}")
    
    try:
        # Create initial state from input
        initial_state = GraphState(
            user_id=input_data.user_id,
            question=input_data.question,
            answer=input_data.answer
        )
        
        # Create and run the graph
        logger.debug("Creating and compiling async graph")
        graph = create_onboarding_graph_async()
        
        # Run the graph with initial state (async invoke)
        logger.info(f"Executing parallel async agents for user: {input_data.user_id}")
        final_state = await graph.ainvoke(initial_state)
        
        # Validate that both agents completed
        if final_state.get("insight") is None:
            raise ValueError("InsightAgent failed to produce output")
        if final_state.get("trait_output") is None:
            raise ValueError("TraitAgent failed to produce output")
        
        # Merge results into final response
        logger.debug("Merging agent results")
        response = OnboardingResponse(
            user_id=final_state["user_id"],
            insight=final_state["insight"],
            traits=final_state["trait_output"].traits
        )
        
        logger.info(f"Successfully completed async processing for user: {input_data.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Async graph processing failed for user {input_data.user_id}: {e}")
        raise