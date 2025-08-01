# Token Counter Web Service

A web service to count tokens for OpenAI and Gemini models using official LOCAL tokenizer libraries.

## ✅ Key Features
- **100% FREE** - No API keys or payments required
- **Completely offline** - All tokenization happens locally
- OpenAI token counting using tiktoken (local)
- Gemini token counting using google-cloud-aiplatform (local)
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
