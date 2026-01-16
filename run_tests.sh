#!/bin/bash

# Test Script for Onboarding Multi-Agent Service
# This script tests all edge cases and validation rules

echo "=========================================="
echo "Testing Onboarding Multi-Agent Service"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Testing endpoint availability..."
curl -s $BASE_URL/ | grep -q "healthy" && echo -e "${GREEN}✓ Service is running${NC}" || echo -e "${RED}✗ Service is not running${NC}"
echo ""

# Test 1: Valid baseline input
echo "=========================================="
echo "Test 1: Valid Baseline Input"
echo "=========================================="
echo "Expected: 200 OK with full response"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_valid_baseline.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 2: Long answer
echo "=========================================="
echo "Test 2: Long Answer (Near Max Limit)"
echo "=========================================="
echo "Expected: 200 OK with full response"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_long_answer.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 3: Minimum valid answer
echo "=========================================="
echo "Test 3: Minimum Valid Answer (10 chars)"
echo "=========================================="
echo "Expected: 200 OK with full response"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_minimum_valid.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 4: Special characters
echo "=========================================="
echo "Test 4: Answer with Special Characters"
echo "=========================================="
echo "Expected: 200 OK with full response"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_special_chars.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 5: Ambiguous answer
echo "=========================================="
echo "Test 5: Ambiguous Answer"
echo "=========================================="
echo "Expected: 200 OK (agents should handle uncertainty)"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_ambiguous.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

echo "=========================================="
echo "INVALID INPUTS (Should Fail)"
echo "=========================================="
echo ""

# Test 6: Empty answer
echo "=========================================="
echo "Test 6: Empty Answer"
echo "=========================================="
echo "Expected: 422 Unprocessable Entity"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_invalid_empty.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 7: Too short answer
echo "=========================================="
echo "Test 7: Answer Too Short (< 10 chars)"
echo "=========================================="
echo "Expected: 422 Unprocessable Entity"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_invalid_short.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 8: Whitespace only
echo "=========================================="
echo "Test 8: Whitespace Only Answer"
echo "=========================================="
echo "Expected: 422 Unprocessable Entity"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d @test_invalid_whitespace.json \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 9: Missing fields
echo "=========================================="
echo "Test 9: Missing Required Fields"
echo "=========================================="
echo "Expected: 422 Unprocessable Entity"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test"}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' 2>/dev/null || echo "Response received"
echo ""

# Test 10: Invalid JSON
echo "=========================================="
echo "Test 10: Invalid JSON Syntax"
echo "=========================================="
echo "Expected: 422 Unprocessable Entity"
echo ""
curl -X POST $BASE_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{invalid json}' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s 2>/dev/null || echo "Response received"
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- Tests 1-5 should return 200 OK"
echo "- Tests 6-10 should return 422 or 400"
echo ""
echo "Check server logs for detailed information:"
echo "  - Retry attempts"
echo "  - JSON parsing strategies"
echo "  - Request timing"
echo "  - Error details"
