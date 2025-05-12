from flask import request, jsonify
from application.clean_parser import parser, is_valid_url
from application.ner_model import ner_classification
from flask import request, jsonify
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_routes(app, ner_model, bert_model):
    @app.route('/furnitureCategories', methods=['GET'])
    def get_furniture_categories():
        try:
            url = request.args.get('url')
            if not url:
                logger.error("URL parameter is missing")
                return jsonify({'error': 'URL parameter is required'}), 400
            
            url = urllib.parse.unquote(url)
            logger.info(f"Processing URL: {url}")
            
            if not is_valid_url(url):
                logger.error(f"Invalid URL provided: {url}")
                return jsonify({'error': 'Invalid URL provided'}), 400

            text = parser(url)
            if not text:
                logger.error("No content parsed from URL")
                return jsonify({'error': 'Could not parse page content'}), 500

            # Ensure we have text to process
            if not text.strip():
                logger.error("Empty text content after parsing")
                return jsonify({'error': 'No meaningful content found on page'}), 500

            try:
                is_with_classification = request.args.get('withClassification', 'false')  # Значение по умолчанию 'false'
                is_with_classification = urllib.parse.unquote(is_with_classification).lower()
                is_with_classification = is_with_classification == 'true'
                print(type(is_with_classification))
                products_dict = ner_classification(text, ner_model, bert_model, is_with_classification)
                
                if not products_dict:
                    logger.error("No products identified during classification")
                    return jsonify({'error': 'Could not identify any products'}), 500

                furniture_categories = [
                    {
                        'name': category_name, 
                        'items': [{'name': item} for item in items if item]
                    }
                    for category_name, items in products_dict.items()
                    if items  
                ]

                if not furniture_categories:
                    logger.warning("No furniture categories found")
                    return jsonify({'message': 'No furniture categories identified'}), 200

                logger.info(f"Successfully processed. Categories found: {len(furniture_categories)}")
                return jsonify(furniture_categories)

            except Exception as model_error:
                logger.exception(f"Error during model classification: {str(model_error)}")
                return jsonify({
                    'error': 'Classification error',
                    'details': str(model_error)
                }), 500

        except Exception as e:
            logger.exception(f"Unexpected error processing URL {url}")
            return jsonify({
                'error': 'Internal server error',
                'details': str(e)
            }), 500

    return app