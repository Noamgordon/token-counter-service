from flask import Flask, request, jsonify
import tiktoken
from vertexai.preview import tokenization
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini model names (supported by local tokenizer)
GEMINI_MODELS = [
    'gemini-1.0-pro',
    'gemini-1.0-pro-001',
    'gemini-1.5-pro',
    'gemini-1.5-pro-001',
    'gemini-1.5-flash',
    'gemini-1.5-flash-001',
    'gemini-1.5-flash-002',
    'gemini-2.0-flash-exp'
]

def count_openai_tokens(text, model):
    """Count tokens for OpenAI models using tiktoken"""
    try:
        # Manual mapping for newer models not yet supported by tiktoken.encoding_for_model()
        MODEL_TO_ENCODING = {
            # GPT-4o family uses o200k_base encoding
            'gpt-4o': 'o200k_base',
            'gpt-4o-mini': 'o200k_base',
            'gpt-4o-2024-05-13': 'o200k_base',
            'gpt-4o-2024-08-06': 'o200k_base',
            'gpt-4o-mini-2024-07-18': 'o200k_base',
            
            # o1 family also uses o200k_base
            'o1-preview': 'o200k_base',
            'o1-mini': 'o200k_base',
            'o1-preview-2024-09-12': 'o200k_base',
            'o1-mini-2024-09-12': 'o200k_base',
        }
        
        encoding = None
        encoding_name = None
        
        # First, check if the model has a manual mapping
        if model in MODEL_TO_ENCODING:
            encoding_name = MODEL_TO_ENCODING[model]
            encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"Using manual mapping: {model} -> {encoding_name}")
        else:
            # If not in manual map, try tiktoken's automatic mapping
            try:
                encoding = tiktoken.encoding_for_model(model)
                encoding_name = encoding.name
                logger.info(f"Using tiktoken automatic mapping: {model} -> {encoding_name}")
            except KeyError:
                # Fallback to a common encoding for truly unknown models
                encoding_name = 'cl100k_base'
                encoding = tiktoken.get_encoding(encoding_name)
                logger.warning(f"Model {model} not recognized, falling back to {encoding_name}")

        # Count tokens
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        return {
            'success': True,
            'token_count': token_count,
            'model': model,
            'encoding': encoding_name,
            'text_length': len(text),
            'tokens_preview': tokens[:10] if len(tokens) > 10 else tokens,
            'method': 'local_tiktoken',
            'mapping_source': 'manual' if model in MODEL_TO_ENCODING else ('automatic' if encoding else 'fallback')
        }
        
    except Exception as e:
        logger.error(f"Error counting OpenAI tokens for model {model}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': model,
            'method': 'local_tiktoken'
        }

def count_gemini_tokens(text, model):
    """Count tokens for Gemini models using local tokenizer (NO API CALLS)"""
    try:
        # Get local tokenizer for the model
        tokenizer = tokenization.get_tokenizer_for_model(model)
        
        # Count tokens locally
        result = tokenizer.count_tokens(text)
        
        return {
            'success': True,
            'token_count': result.total_tokens,
            'model': model,
            'text_length': len(text),
            'characters_per_token': len(text) / result.total_tokens if result.total_tokens > 0 else 0,
            'method': 'local_tokenization'
        }
    except Exception as e:
        logger.error(f"Error counting Gemini tokens: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': model,
            'method': 'local_tokenization'
        }

@app.route('/', methods=['GET'])
def home():
    """Health check and API documentation"""
    return jsonify({
        'service': 'Token Counter API',
        'status': 'active',
        'version': '3.0.0', # Updated version
        'endpoints': {
            '/count': 'POST - Count tokens for a batch of texts',
            '/models': 'GET - List supported models',
            '/health': 'GET - Health check'
        },
        'supported_providers': ['openai', 'gemini'],
        'usage': {
            'method': 'POST',
            'url': '/count',
            'body': [
                {
                    'texts': ['Text 1', 'Text 2'],
                    'model': 'gpt-4o',
                    'provider': 'openai'
                },
                {
                    'texts': ['Another text'],
                    'model': 'gemini-1.5-pro',
                    'provider': 'gemini'
                }
            ]
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'openai_tokenizer': 'tiktoken (local)',
        'gemini_tokenizer': 'vertexai.tokenization (local)',
        'no_api_keys_required': True
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all supported models"""
    return jsonify({
        'openai_models': 'Dynamically supported by tiktoken library, plus manual mapping for new models.',
        'gemini_models': GEMINI_MODELS,
        'total_models': len(GEMINI_MODELS)
    })

@app.route('/count', methods=['POST'])
def count_tokens():
    """Main endpoint to count tokens for a batch of texts"""
    try:
        requests_data = request.get_json()
        
        if not isinstance(requests_data, list):
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Expected a JSON array of request objects.'
            }), 400

        final_results = []
        for req_obj in requests_data:
            req_results = []
            
            # Extract data and handle potential errors
            texts = req_obj.get('texts', [])
            model = req_obj.get('model', '').strip()
            provider = req_obj.get('provider', '').strip().lower()

            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                req_results.append({'success': False, 'error': 'The "texts" field must be a list of strings.'})
                final_results.append({'request_errors': req_results})
                continue
            
            if not model:
                req_results.append({'success': False, 'error': 'The "model" field is required.'})
                final_results.append({'request_errors': req_results})
                continue
            
            if not provider:
                req_results.append({'success': False, 'error': 'The "provider" field is required (openai or gemini).'})
                final_results.append({'request_errors': req_results})
                continue

            # Process each text within the current request object
            for text_to_count in texts:
                if not text_to_count.strip():
                    result = {
                        'success': False,
                        'error': 'Text string cannot be empty'
                    }
                elif provider == 'openai':
                    result = count_openai_tokens(text_to_count, model)
                elif provider == 'gemini':
                    result = count_gemini_tokens(text_to_count, model)
                else:
                    result = {
                        'success': False,
                        'error': f'Unsupported provider: {provider}. Use "openai" or "gemini"'
                    }
                
                # Add metadata to each result
                result['provider'] = provider
                result['timestamp'] = int(__import__('time').time())
                req_results.append(result)
            
            # Append the results for the current request object to the final list
            final_results.append(req_results)
            
        return jsonify(final_results)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
