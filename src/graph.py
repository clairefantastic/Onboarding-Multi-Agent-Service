from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.models import GraphState, OnboardingInput, OnboardingResponse
from src.agents import insight_agent, trait_agent


def create_onboarding_graph():
    """
    Creates a LangGraph workflow that runs two agents in parallel
    """
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
    """
    # Create initial state from input
    initial_state = GraphState(
        user_id=input_data.user_id,
        question=input_data.question,
        answer=input_data.answer
    )
    
    # Create and run the graph
    graph = create_onboarding_graph()
    
    # Run the graph with initial state
    final_state = graph.invoke(initial_state)
    
    # Merge results into final response
    response = OnboardingResponse(
        user_id=final_state["user_id"],
        insight=final_state["insight"],
        traits=final_state["trait_output"].traits
    )
    
    return response