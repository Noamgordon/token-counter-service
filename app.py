from flask import Flask, request, jsonify
import tiktoken
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp"

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# OpenAI model to encoding mapping
OPENAI_MODEL_ENCODINGS = {
    # GPT-4 models
    'gpt-4': 'cl100k_base',
    'gpt-4-0314': 'cl100k_base',
    'gpt-4-0613': 'cl100k_base',
    'gpt-4-32k': 'cl100k_base',
    'gpt-4-32k-0314': 'cl100k_base',
    'gpt-4-32k-0613': 'cl100k_base',
    'gpt-4-turbo': 'cl100k_base',
    'gpt-4-turbo-preview': 'cl100k_base',
    'gpt-4-0125-preview': 'cl100k_base',
    'gpt-4-1106-preview': 'cl100k_base',
    'gpt-4-vision-preview': 'cl100k_base',
    'gpt-4-1106-vision-preview': 'cl100k_base',
    
    # GPT-4o models
    'gpt-4o': 'o200k_base',
    'gpt-4o-2024-05-13': 'o200k_base',
    'gpt-4o-mini': 'o200k_base',
    'gpt-4o-mini-2024-07-18': 'o200k_base',
    
    # GPT-3.5 models
    'gpt-3.5-turbo': 'cl100k_base',
    'gpt-3.5-turbo-0301': 'cl100k_base',
    'gpt-3.5-turbo-0613': 'cl100k_base',
    'gpt-3.5-turbo-16k': 'cl100k_base',
    'gpt-3.5-turbo-16k-0613': 'cl100k_base',
    'gpt-3.5-turbo-1106': 'cl100k_base',
    'gpt-3.5-turbo-0125': 'cl100k_base',
    
    # Text embedding models
    'text-embedding-ada-002': 'cl100k_base',
    'text-embedding-3-small': 'cl100k_base',
    'text-embedding-3-large': 'cl100k_base',
    
    # Other models
    'text-davinci-003': 'p50k_base',
    'text-davinci-002': 'p50k_base',
    'code-davinci-002': 'p50k_base',
    'davinci': 'r50k_base',
    'curie': 'r50k_base',
    'babbage': 'r50k_base',
    'ada': 'r50k_base',
}

# Gemini model names
GEMINI_MODELS = [
    'gemini-pro',
    'gemini-pro-vision',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-1.0-pro',
    'models/gemini-pro',
    'models/gemini-pro-vision',
    'models/gemini-1.5-pro',
    'models/gemini-1.5-flash',
    'models/gemini-1.0-pro'
]

def count_openai_tokens(text, model):
    """Count tokens for OpenAI models using tiktoken"""
    try:
        # Get encoding for the model
        if model in OPENAI_MODEL_ENCODINGS:
            encoding_name = OPENAI_MODEL_ENCODINGS[model]
            encoding = tiktoken.get_encoding(encoding_name)
        else:
            # Try to get encoding directly for the model
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Default to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
        
        # Count tokens
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        return {
            'success': True,
            'token_count': token_count,
            'model': model,
            'encoding': encoding.name,
            'text_length': len(text),
            'tokens_preview': tokens[:10] if len(tokens) > 10 else tokens
        }
    except Exception as e:
        logger.error(f"Error counting OpenAI tokens: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': model
        }

def count_gemini_tokens(text, model):
    """Count tokens for Gemini models using official Google SDK"""
    try:
        if not GEMINI_API_KEY:
            return {
                'success': False,
                'error': 'GEMINI_API_KEY not configured',
                'model': model
            }
        
        # Normalize model name
        if not model.startswith('models/'):
            if model.startswith('gemini-'):
                model_name = f"models/{model}"
            else:
                model_name = f"models/gemini-{model}"
        else:
            model_name = model
        
        # Get the model
        try:
            genai_model = genai.GenerativeModel(model_name)
        except Exception as e:
            # Try with original model name
            genai_model = genai.GenerativeModel(model)
        
        # Count tokens
        response = genai_model.count_tokens(text)
        
        return {
            'success': True,
            'token_count': response.total_tokens,
            'model': model,
            'text_length': len(text),
            'characters_per_token': len(text) / response.total_tokens if response.total_tokens > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error counting Gemini tokens: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': model
        }

@app.route('/', methods=['GET'])
def home():
    """Health check and API documentation"""
    return jsonify({
        'service': 'Token Counter API',
        'status': 'active',
        'version': '1.0.0',
        'endpoints': {
            '/count': 'POST - Count tokens for text',
            '/models': 'GET - List supported models',
            '/health': 'GET - Health check'
        },
        'supported_providers': ['openai', 'gemini'],
        'usage': {
            'method': 'POST',
            'url': '/count',
            'body': {
                'text': 'Your text here',
                'model': 'gpt-4o',
                'provider': 'openai'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gemini_configured': bool(GEMINI_API_KEY)
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all supported models"""
    return jsonify({
        'openai_models': list(OPENAI_MODEL_ENCODINGS.keys()),
        'gemini_models': GEMINI_MODELS,
        'total_models': len(OPENAI_MODEL_ENCODINGS) + len(GEMINI_MODELS)
    })

@app.route('/count', methods=['POST'])
def count_tokens():
    """Main endpoint to count tokens"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        text = data.get('text', '').strip()
        model = data.get('model', '').strip()
        provider = data.get('provider', '').strip().lower()
        
        # Validation
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text field is required and cannot be empty'
            }), 400
        
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model field is required'
            }), 400
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'Provider field is required (openai or gemini)'
            }), 400
        
        # Route to appropriate counter
        if provider == 'openai':
            result = count_openai_tokens(text, model)
        elif provider == 'gemini':
            result = count_gemini_tokens(text, model)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported provider: {provider}. Use "openai" or "gemini"'
            }), 400
        
        # Add metadata
        result['provider'] = provider
        result['timestamp'] = int(__import__('time').time())
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
