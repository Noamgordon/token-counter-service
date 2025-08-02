from flask import Flask, request, jsonify
import tiktoken
from vertexai.preview import tokenization
import logging
import os
from typing import Dict, Any, List, Optional

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

# --- Core Tokenizer Functions (Unchanged, as they are now robust) ---
def get_openai_token_count(text: str, model: str) -> Optional[int]:
    """
    Get token count for OpenAI models using tiktoken.
    Returns the token count or None if an error occurs.
    """
    try:
        MODEL_TO_ENCODING = {
            'gpt-4o': 'o200k_base',
            'gpt-4o-mini': 'o200k_base',
            'gpt-4o-2024-05-13': 'o200k_base',
            'gpt-4o-2024-08-06': 'o200k_base',
            'gpt-4o-mini-2024-07-18': 'o200k_base',
            'o1-preview': 'o200k_base',
            'o1-mini': 'o200k_base',
            'o1-preview-2024-09-12': 'o200k_base',
            'o1-mini-2024-09-12': 'o200k_base',
        }
        if model in MODEL_TO_ENCODING:
            encoding_name = MODEL_TO_ENCODING[model]
            encoding = tiktoken.get_encoding(encoding_name)
        else:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding('cl100k_base')
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting OpenAI tokens for model {model}: {str(e)}")
        return None

def get_gemini_token_count(text: str, model: str) -> Optional[int]:
    """
    Get token count for Gemini models using local tokenizer.
    Returns the token count or None if an error occurs.
    """
    try:
        tokenizer = tokenization.get_tokenizer_for_model(model)
        return tokenizer.count_tokens(text).total_tokens
    except Exception as e:
        logger.error(f"Error counting Gemini tokens for model {model}: {str(e)}")
        return None

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def home():
    """Health check and API documentation"""
    return jsonify({
        'service': 'Batch Token Comparison API',
        'status': 'active',
        'version': '5.0.0', # Updated version
        'endpoints': {
            '/compare': 'POST - Compare token counts for a batch of phrases/models'
        },
        'supported_providers': ['openai', 'gemini'],
        'usage': {
            'method': 'POST',
            'url': '/compare',
            'body': [
                {
                    'category': 'negation',
                    'phrases': ['cannot', 'can not'],
                    'models': ['gpt-4o', 'gemini-1.5-pro']
                }
            ]
        }
    })

@app.route('/compare', methods=['POST'])
def compare_phrases():
    """
    Endpoint to process a batch of token comparison requests.
    """
    try:
        batch_requests = request.get_json()

        if not isinstance(batch_requests, list):
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Expected a JSON array of request objects.'
            }), 400
        
        final_response = []

        for req in batch_requests:
            category = req.get('category')
            phrases = req.get('phrases')
            models = req.get('models')

            if not all([category, phrases, models]) or not isinstance(phrases, list) or not isinstance(models, list):
                logger.error(f"Invalid request object in batch: {req}")
                final_response.append({'category': category, 'error': 'Invalid format in request object.'})
                continue
            
            results_by_model = {}
            for model_name in models:
                model_name_lower = model_name.lower()
                
                # Determine provider based on model name
                provider = 'gemini' if any(gemini_model in model_name_lower for gemini_model in GEMINI_MODELS) else 'openai'
                
                count_func = None
                if provider == 'openai':
                    count_func = get_openai_token_count
                elif provider == 'gemini':
                    count_func = get_gemini_token_count

                if not count_func:
                    logger.warning(f"No tokenizer function found for model: {model_name}")
                    continue

                min_tokens = float('inf')
                best_match = None

                for phrase in phrases:
                    token_count = count_func(phrase, model_name)
                    
                    if token_count is not None and token_count < min_tokens:
                        min_tokens = token_count
                        best_match = {'phrase': phrase, 'token_count': token_count}

                if best_match:
                    results_by_model[model_name] = {'best_match': best_match}
            
            final_response.append({
                'category': category,
                'results_by_model': results_by_model
            })
            
        return jsonify(final_response)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
