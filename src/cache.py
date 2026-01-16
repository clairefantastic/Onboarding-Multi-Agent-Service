"""
Simple in-memory cache for onboarding analysis results
Uses LRU (Least Recently Used) eviction policy
"""
import time
import hashlib
import json
import logging
from typing import Optional, Dict
from collections import OrderedDict
from src.models import OnboardingInput, OnboardingResponse

logger = logging.getLogger(__name__)


class AnalysisCache:
    """
    LRU Cache for storing analysis results
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # OrderedDict maintains insertion order for LRU
        # Format: {cache_key: (result, timestamp)}
        self.cache: OrderedDict = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(
            f"Cache initialized: max_size={max_size}, ttl={ttl_seconds}s"
        )
    
    def _generate_cache_key(self, input_data: OnboardingInput) -> str:
        """
        Generate cache key from input data
        Uses hash of question + answer
        """
        # Create deterministic string from input
        cache_string = f"{input_data.question}||{input_data.answer}"
        
        # Generate hash
        hash_object = hashlib.sha256(cache_string.encode())
        cache_key = hash_object.hexdigest()
        
        return cache_key
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - timestamp) > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            evicted_key, _ = self.cache.popitem(last=False)
            self.evictions += 1
            logger.debug(f"Evicted LRU entry: {evicted_key[:16]}...")
    
    def _clean_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if (current_time - timestamp) > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            logger.debug(f"Removed expired entry: {key[:16]}...")
    
    def get(self, input_data: OnboardingInput) -> Optional[OnboardingResponse]:
        """
        Get cached result if available
        
        Returns:
            OnboardingResponse if cached and not expired, None otherwise
        """
        cache_key = self._generate_cache_key(input_data)
        
        # Check if key exists
        if cache_key not in self.cache:
            self.misses += 1
            logger.debug(f"Cache miss: {cache_key[:16]}...")
            return None
        
        # Get entry
        result, timestamp = self.cache[cache_key]
        
        # Check if expired
        if self._is_expired(timestamp):
            del self.cache[cache_key]
            self.misses += 1
            logger.debug(f"Cache expired: {cache_key[:16]}...")
            return None
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(cache_key)
        
        self.hits += 1
        logger.info(
            f"Cache hit: {cache_key[:16]}... "
            f"(hit rate: {self.get_hit_rate():.1%})"
        )
        
        return result
    
    def set(self, input_data: OnboardingInput, result: OnboardingResponse):
        """
        Store result in cache
        """
        cache_key = self._generate_cache_key(input_data)
        
        # If at capacity, evict LRU
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store with current timestamp
        self.cache[cache_key] = (result, time.time())
        
        logger.debug(
            f"Cached result: {cache_key[:16]}... "
            f"(cache size: {len(self.cache)}/{self.max_size})"
        )
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        self._clean_expired()
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.get_hit_rate(),
            "ttl_seconds": self.ttl_seconds
        }


# Global cache instance
analysis_cache = AnalysisCache(
    max_size=1000,    # Store up to 1000 results
    ttl_seconds=3600  # Cache for 1 hour
)