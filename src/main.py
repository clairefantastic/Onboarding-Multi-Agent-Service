import os
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from dotenv import load_dotenv
from src.models import OnboardingInput, OnboardingResponse
from src.graph import process_onboarding_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set")
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")

logger.info("Starting Onboarding Multi-Agent Service")

app = FastAPI(
    title="Onboarding Multi-Agent Service",
    description="Process onboarding Q&A through parallel LLM agents",
    version="1.0.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"Status: {response.status_code} Duration: {process_time:.2f}s"
    )
    
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed messages"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Input validation failed",
            "errors": exc.errors()
        }
    )


@app.get("/")
def root():
    """Health check endpoint"""
    logger.debug("Health check called")
    return {"status": "healthy", "service": "onboarding-multi-agent"}


@app.post("/analyze", response_model=OnboardingResponse)
async def analyze_onboarding(input_data: OnboardingInput):
    """
    Analyze an onboarding Q&A pair through parallel agents
    
    Args:
        input_data: OnboardingInput with user_id, question, and answer
    
    Returns:
        OnboardingResponse with insights and trait scores
    
    Raises:
        400: Invalid input data
        500: Internal server error (LLM failure, parsing error, etc.)
    """
    logger.info(f"Received analysis request for user: {input_data.user_id}")
    
    try:
        # Process with async agents
        result = await process_onboarding_async(input_data)
        
        logger.info(f"Successfully processed request for user: {input_data.user_id}")
        return result
    
    except ValidationError as e:
        # Pydantic validation errors (from agent output validation)
        logger.error(f"Agent output validation failed for user {input_data.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Agent produced invalid output format"
        )
    
    except ValueError as e:
        # Value errors (invalid input, JSON parsing, etc.)
        logger.error(f"Value error for user {input_data.user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    
    except TimeoutError as e:
        # LLM timeout
        logger.error(f"Request timeout for user {input_data.user_id}: {e}")
        raise HTTPException(
            status_code=504,
            detail="Request timed out - please try again"
        )
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            f"Unexpected error for user {input_data.user_id}: {type(e).__name__}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error - please try again later"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)