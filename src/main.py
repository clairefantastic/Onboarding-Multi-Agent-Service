import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from src.models import OnboardingInput, OnboardingResponse
from src.graph import process_onboarding

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")

app = FastAPI(
    title="Onboarding Multi-Agent Service",
    description="Process onboarding Q&A through parallel LLM agents",
    version="1.0.0"
)


@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "onboarding-multi-agent"}


@app.post("/analyze", response_model=OnboardingResponse)
async def analyze_onboarding(input_data: OnboardingInput):
    """
    Analyze an onboarding Q&A pair through parallel agents
    
    Returns:
        OnboardingResponse with insights and trait scores
    """
    try:
        result = process_onboarding(input_data)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)