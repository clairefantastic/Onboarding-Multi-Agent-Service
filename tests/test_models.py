"""
Unit tests for Pydantic models
"""
import pytest
from pydantic import ValidationError
from src.models import (
    OnboardingInput,
    InsightOutput,
    Trait,
    TraitOutput,
    GraphState,
    OnboardingResponse
)


class TestOnboardingInput:
    """Tests for OnboardingInput model"""
    
    def test_valid_input(self):
        """Test valid input creation"""
        data = OnboardingInput(
            user_id="user-123",
            question="What are you looking for?",
            answer="I want a serious relationship."
        )
        assert data.user_id == "user-123"
        assert data.question == "What are you looking for?"
        assert data.answer == "I want a serious relationship."
    
    def test_empty_answer_fails(self):
        """Test that empty answer is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            OnboardingInput(
                user_id="user-123",
                question="What?",
                answer=""
            )
        assert "Field cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_answer_fails(self):
        """Test that whitespace-only answer is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            OnboardingInput(
                user_id="user-123",
                question="What?",
                answer="   "
            )
        assert "Field cannot be empty" in str(exc_info.value)
    
    def test_too_short_answer_fails(self):
        """Test that answer < 10 chars is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            OnboardingInput(
                user_id="user-123",
                question="What?",
                answer="Yes"
            )
        assert "at least 10 characters" in str(exc_info.value)
    
    def test_answer_strips_whitespace(self):
        """Test that whitespace is stripped from answer"""
        data = OnboardingInput(
            user_id="user-123",
            question="What?",
            answer="  I want love  "
        )
        assert data.answer == "I want love"
    
    def test_user_id_too_long_fails(self):
        """Test that user_id > 100 chars is rejected"""
        with pytest.raises(ValidationError):
            OnboardingInput(
                user_id="x" * 101,
                question="What?",
                answer="I want love"
            )
    
    def test_question_too_long_fails(self):
        """Test that question > 500 chars is rejected"""
        with pytest.raises(ValidationError):
            OnboardingInput(
                user_id="user-123",
                question="x" * 501,
                answer="I want love"
            )
    
    def test_answer_too_long_fails(self):
        """Test that answer > 5000 chars is rejected"""
        with pytest.raises(ValidationError):
            OnboardingInput(
                user_id="user-123",
                question="What?",
                answer="x" * 5001
            )


class TestInsightOutput:
    """Tests for InsightOutput model"""
    
    def test_valid_insight(self):
        """Test valid insight creation"""
        data = InsightOutput(
            summary="User wants a serious relationship.",
            keywords=["serious", "relationship"]
        )
        assert data.summary == "User wants a serious relationship."
        assert len(data.keywords) == 2
    
    def test_keywords_min_length(self):
        """Test that at least 2 keywords required"""
        with pytest.raises(ValidationError):
            InsightOutput(
                summary="User wants love.",
                keywords=["love"]  # Only 1 keyword
            )
    
    def test_keywords_max_length(self):
        """Test that max 5 keywords allowed"""
        with pytest.raises(ValidationError):
            InsightOutput(
                summary="User wants love.",
                keywords=["a", "b", "c", "d", "e", "f"]  # 6 keywords
            )
    
    def test_empty_keyword_fails(self):
        """Test that empty keywords are rejected"""
        with pytest.raises(ValidationError):
            InsightOutput(
                summary="User wants love.",
                keywords=["love", ""]
            )


class TestTrait:
    """Tests for Trait model"""
    
    def test_valid_trait(self):
        """Test valid trait creation"""
        trait = Trait(
            name="relationship_goal_readiness",
            score=0.8,
            reason="User expresses clear relationship goals."
        )
        assert trait.name == "relationship_goal_readiness"
        assert trait.score == 0.8
        assert trait.reason == "User expresses clear relationship goals."
    
    def test_score_range_validation(self):
        """Test that score must be between -1 and 1"""
        # Valid scores
        Trait(name="test", score=-1.0, reason="Min valid score test")
        Trait(name="test", score=0.0, reason="Neutral score test")
        Trait(name="test", score=1.0, reason="Max valid score test")
    
        # Invalid scores
        with pytest.raises(ValidationError):
            Trait(name="test", score=-1.1, reason="Score too low here")
    
        with pytest.raises(ValidationError):
            Trait(name="test", score=1.1, reason="Score too high here")
    
    def test_reason_min_length(self):
        """Test that reason must be at least 10 chars"""
        with pytest.raises(ValidationError):
            Trait(name="test", score=0.5, reason="Short")


class TestTraitOutput:
    """Tests for TraitOutput model"""
    
    def test_valid_trait_output(self):
        """Test valid trait output creation"""
        output = TraitOutput(
            traits=[
                Trait(name="trait1", score=0.8, reason="Reason one is here."),
                Trait(name="trait2", score=0.5, reason="Reason two is here.")
            ]
        )
        assert len(output.traits) == 2
    
    def test_too_few_traits_fails(self):
        """Test that at least 2 traits required"""
        with pytest.raises(ValidationError):
            TraitOutput(
                traits=[
                    Trait(name="trait1", score=0.8, reason="Only one trait here.")
                ]
            )
    
    def test_too_many_traits_fails(self):
        """Test that max 5 traits allowed"""
        with pytest.raises(ValidationError):
            TraitOutput(
                traits=[
                    Trait(name=f"trait{i}", score=0.5, reason=f"Reason {i} is here.")
                    for i in range(6)  # 6 traits
                ]
            )


class TestGraphState:
    """Tests for GraphState model"""
    
    def test_initial_state(self):
        """Test creating initial state"""
        state = GraphState(
            user_id="user-123",
            question="What?",
            answer="I want love and happiness."
        )
        assert state.user_id == "user-123"
        assert state.insight is None
        assert state.trait_output is None
    
    def test_state_with_results(self):
        """Test state after agents complete"""
        state = GraphState(
            user_id="user-123",
            question="What?",
            answer="I want love and happiness.",
            insight=InsightOutput(
                summary="User wants love.",
                keywords=["love", "happiness"]
            ),
            trait_output=TraitOutput(
                traits=[
                    Trait(name="trait1", score=0.8, reason="Reason one here."),
                    Trait(name="trait2", score=0.5, reason="Reason two here.")
                ]
            )
        )
        assert state.insight is not None
        assert state.trait_output is not None


class TestOnboardingResponse:
    """Tests for OnboardingResponse model"""
    
    def test_valid_response(self):
        """Test valid response creation"""
        response = OnboardingResponse(
            user_id="user-123",
            insight=InsightOutput(
                summary="User wants love.",
                keywords=["love", "happiness"]
            ),
            traits=[
                Trait(name="trait1", score=0.8, reason="Reason one here."),
                Trait(name="trait2", score=0.5, reason="Reason two here.")
            ]
        )
        assert response.user_id == "user-123"
        assert len(response.traits) == 2
