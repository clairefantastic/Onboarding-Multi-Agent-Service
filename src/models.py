from pydantic import BaseModel, Field
from typing import List, Optional

class OnboardingInput(BaseModel):
    """Input model for onboarding Q&A"""
    user_id: str = Field(..., description="Unique user identifier")
    question: str = Field(..., description="Onboarding question asked")
    answer: str = Field(..., description="User's free-text answer")
    
class InsightOutput(BaseModel):
    """Output from the Insight Agent"""
    summary: str = Field(..., description="Natural language summary of the answer")
    keywords: List[str] = Field(..., description="2-3 key phrases extracted from the answer")
    
class Trait(BaseModel):
    """A single trait score with reasoning"""
    name: str = Field(..., description="Trait name")
    score: float = Field(..., ge=-1.0, le=1.0, description="Trait score between -1 and 1")
    reason: str = Field(..., description="One-sentence explanation for the score")
    
class TraitOutput(BaseModel):
    """Output from the Trait Agent"""
    traits: List[Trait] = Field(..., description="2-3 trait scores with reasoning")
    
class GraphState(BaseModel):
    """State object for the LangGraph workflow"""
    # Input
    user_id: str
    question: str
    answer: str
    
    # Agent outputs
    insight: Optional[InsightOutput] = None
    trait_output: Optional[TraitOutput] = None

class OnboardingResponse(BaseModel):
    """Final merged response"""
    user_id: str
    insight: InsightOutput
    traits: List[Trait]