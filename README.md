# Token Counter Web Service

A web service to count tokens for OpenAI and Gemini models using official tokenizer libraries.

## Features
- OpenAI token counting using tiktoken
- Gemini token counting using google-generativeai
- RESTful API
- Support for all major model variants
- Deployed on Render

## API Usage

### Count Tokens
POST /count
```json
{
  "text": "Your text here",
  "model": "gpt-4o",
  "provider": "openai"
}
