from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class OnboardingInput(BaseModel):
    """Input model for onboarding Q&A"""
    user_id: str = Field(..., description="Unique user identifier")
    question: str = Field(..., description="Onboarding question asked")
    answer: str = Field(..., description="User's free-text answer")
    
    @field_validator('user_id', 'question', 'answer')
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        """Ensure fields are not just whitespace"""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty or just whitespace')
        return v.strip()
    
    @field_validator('user_id')
    @classmethod
    def check_user_id_length(cls, v: str) -> str:
        """Ensure user_id is within bounds"""
        if len(v) < 1 or len(v) > 100:
            raise ValueError('user_id must be between 1-100 characters')
        return v
    
    @field_validator('question')
    @classmethod
    def check_question_length(cls, v: str) -> str:
        """Ensure question is within bounds"""
        if len(v) < 1 or len(v) > 500:
            raise ValueError('question must be between 1-500 characters')
        return v
    
    @field_validator('answer')
    @classmethod
    def check_answer_length(cls, v: str) -> str:
        """Ensure answer has meaningful content"""
        if len(v) < 10:
            raise ValueError('Answer must be at least 10 characters long')
        if len(v) > 5000:
            raise ValueError('Answer must be less than 5000 characters')
        return v

class InsightOutput(BaseModel):
    """Output from the Insight Agent"""
    summary: str = Field(
        ...,
        description="Natural language summary of the answer",
        min_length=10,
        max_length=500
    )
    keywords: List[str] = Field(
        ...,
        description="2-3 key phrases extracted from the answer",
        min_length=2,
        max_length=5
    )
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Ensure keywords are not empty and within reasonable length"""
        if not v:
            raise ValueError('Keywords list cannot be empty')
        for keyword in v:
            if not keyword or not keyword.strip():
                raise ValueError('Keywords cannot be empty')
            if len(keyword) > 100:
                raise ValueError('Individual keyword too long (max 100 chars)')
        return [k.strip() for k in v]


class Trait(BaseModel):
    """A single trait score with reasoning"""
    name: str = Field(
        ...,
        description="Trait name",
        min_length=1,
        max_length=100
    )
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Trait score between -1 and 1"
    )
    reason: str = Field(
        ...,
        description="One-sentence explanation for the score",
        min_length=10,
        max_length=500
    )


class TraitOutput(BaseModel):
    """Output from the Trait Agent"""
    traits: List[Trait] = Field(
        ...,
        description="2-3 trait scores with reasoning",
        min_length=2,
        max_length=5
    )
    
    @field_validator('traits')
    @classmethod
    def validate_traits(cls, v: List[Trait]) -> List[Trait]:
        """Ensure we have a reasonable number of traits"""
        if len(v) < 2:
            raise ValueError('Must have at least 2 traits')
        if len(v) > 5:
            raise ValueError('Cannot have more than 5 traits')
        return v


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