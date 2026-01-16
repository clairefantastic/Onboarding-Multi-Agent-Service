"""
Rate limiting middleware for FastAPI
Limits requests per IP address
"""
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter
    Tracks requests per IP address with sliding window
    """
    
    def __init__(self, requests_per_minute: int = 10, requests_per_hour: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Store timestamps of requests per IP
        # Format: {ip: [timestamp1, timestamp2, ...]}
        self.request_history: Dict[str, list] = defaultdict(list)
        
        logger.info(
            f"Rate limiter initialized: {requests_per_minute}/min, "
            f"{requests_per_hour}/hour per IP"
        )
    
    def _clean_old_requests(self, ip: str, current_time: float):
        """Remove requests older than 1 hour"""
        one_hour_ago = current_time - 3600
        self.request_history[ip] = [
            ts for ts in self.request_history[ip] 
            if ts > one_hour_ago
        ]
    
    def _get_request_counts(self, ip: str, current_time: float) -> Tuple[int, int]:
        """Get number of requests in last minute and hour"""
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600
        
        timestamps = self.request_history[ip]
        
        requests_last_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
        requests_last_hour = sum(1 for ts in timestamps if ts > one_hour_ago)
        
        return requests_last_minute, requests_last_hour
    
    def check_rate_limit(self, ip: str) -> Tuple[bool, str]:
        """
        Check if request from IP should be allowed
        
        Returns:
            (allowed: bool, reason: str)
        """
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(ip, current_time)
        
        # Get current counts
        requests_last_minute, requests_last_hour = self._get_request_counts(
            ip, current_time
        )
        
        # Check limits
        if requests_last_minute >= self.requests_per_minute:
            logger.warning(
                f"Rate limit exceeded for {ip}: "
                f"{requests_last_minute} requests in last minute"
            )
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
        
        if requests_last_hour >= self.requests_per_hour:
            logger.warning(
                f"Rate limit exceeded for {ip}: "
                f"{requests_last_hour} requests in last hour"
            )
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
        
        # Record this request
        self.request_history[ip].append(current_time)
        
        logger.debug(
            f"Rate limit check passed for {ip}: "
            f"{requests_last_minute + 1}/min, {requests_last_hour + 1}/hour"
        )
        
        return True, "OK"
    
    def get_stats(self, ip: str) -> Dict:
        """Get current rate limit stats for an IP"""
        current_time = time.time()
        self._clean_old_requests(ip, current_time)
        
        requests_last_minute, requests_last_hour = self._get_request_counts(
            ip, current_time
        )
        
        return {
            "ip": ip,
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
            "remaining_this_minute": self.requests_per_minute - requests_last_minute,
            "remaining_this_hour": self.requests_per_hour - requests_last_hour
        }


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=10,  # 10 requests per minute
    requests_per_hour=100     # 100 requests per hour
)


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to check rate limits before processing request
    """
    # Get client IP
    client_ip = request.client.host
    
    # Skip rate limiting for health check
    if request.url.path == "/":
        return await call_next(request)
    
    # Check rate limit
    allowed, reason = rate_limiter.check_rate_limit(client_ip)
    
    if not allowed:
        logger.warning(f"Request from {client_ip} blocked: {reason}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": reason,
                "retry_after": "60 seconds"
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    stats = rate_limiter.get_stats(client_ip)
    response.headers["X-RateLimit-Limit-Minute"] = str(rate_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining-Minute"] = str(stats["remaining_this_minute"])
    response.headers["X-RateLimit-Limit-Hour"] = str(rate_limiter.requests_per_hour)
    response.headers["X-RateLimit-Remaining-Hour"] = str(stats["remaining_this_hour"])
    
    return response