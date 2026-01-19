"""
Artist Popularity Prediction API
Flask-based REST API for ML model deployment
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler when app starts
print("\n" + "="*70)
print("🚀 ARTIST POPULARITY PREDICTION API")
print("="*70)
print("\n🔄 Loading model and scaler...")

try:
    # Load the trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
    
    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully!")
    
    print("\n✅ API is ready to accept requests!")
    
except FileNotFoundError as e:
    print(f"❌ Error: Required file not found - {e}")
    print("Make sure model.pkl and scaler.pkl are in the same folder as api.py")
    model = None
    scaler = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

print("="*70 + "\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """Home page - API information"""
    return jsonify({
        'message': '🎵 Artist Popularity Prediction API',
        'status': 'Running',
        'version': '1.0.0',
        'description': 'Predict if an artist will be popular based on followers and genres',
        'endpoints': {
            '/': {
                'method': 'GET',
                'description': 'API information'
            },
            '/health': {
                'method': 'GET',
                'description': 'Health check'
            },
            '/predict': {
                'method': 'POST',
                'description': 'Make prediction',
                'required_fields': ['followers', 'genre_count']
            }
        },
        'example_request': {
            'url': '/predict',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': {
                'followers': 50000,
                'genre_count': 3
            }
        },
        'example_response': {
            'prediction': 'Popular ⭐',
            'confidence': '87.45%',
            'prediction_value': 1,
            'probabilities': {
                'not_popular': '12.55%',
                'popular': '87.45%'
            }
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if (model is not None and scaler is not None) else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'message': 'API is operational' if model is not None else 'Model not loaded'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON body:
    {
        "followers": 50000,
        "genre_count": 3
    }
    """
    
    # Check if model is loaded
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded properly. Please check server logs.',
            'status': 'error'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate that data was provided
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Please send JSON data with followers and genre_count',
                'example': {
                    'followers': 50000,
                    'genre_count': 3
                }
            }), 400
        
        # Extract features
        followers = data.get('followers')
        genre_count = data.get('genre_count')
        
        # Validate required fields
        if followers is None:
            return jsonify({
                'error': 'Missing required field: followers',
                'message': 'Please provide the number of followers'
            }), 400
        
        if genre_count is None:
            return jsonify({
                'error': 'Missing required field: genre_count',
                'message': 'Please provide the number of genres'
            }), 400
        
        # Convert to proper types
        try:
            followers = float(followers)
            genre_count = int(genre_count)
        except ValueError:
            return jsonify({
                'error': 'Invalid data type',
                'message': 'followers must be a number, genre_count must be an integer'
            }), 400
        
        # Validate ranges
        if followers < 0:
            return jsonify({
                'error': 'Invalid value: followers cannot be negative',
                'message': 'Please provide a positive number for followers'
            }), 400
        
        if genre_count < 0 or genre_count > 50:
            return jsonify({
                'error': 'Invalid value: genre_count must be between 0 and 50',
                'message': 'Please provide a valid genre count'
            }), 400
        
        # Create feature array (same order as training)
        # Features: [followers, log_followers, genre_count, has_genre]
        features = [
            followers,                      # followers
            np.log1p(followers),           # log_followers (log transformation)
            genre_count,                   # genre_count
            1 if genre_count > 0 else 0    # has_genre (binary)
        ]
        
        # Scale features using the loaded scaler
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Calculate confidence
        confidence = max(probability) * 100
        
        # Prepare response
        result = {
            'prediction': 'Popular ⭐' if prediction == 1 else 'Not Popular',
            'confidence': f'{confidence:.2f}%',
            'prediction_value': int(prediction),
            'probabilities': {
                'not_popular': f'{probability[0]*100:.2f}%',
                'popular': f'{probability[1]*100:.2f}%'
            },
            'input': {
                'followers': int(followers),
                'genre_count': int(genre_count),
                'has_genre': genre_count > 0
            },
            'interpretation': get_interpretation(prediction, confidence, followers, genre_count)
        }
        
        # Log the prediction
        print(f"✓ Prediction: {result['prediction']} | Confidence: {result['confidence']} | Followers: {int(followers)} | Genres: {genre_count}")
        
        return jsonify(result), 200
        
    except Exception as e:
        # Log the error
        print(f"❌ Error during prediction: {str(e)}")
        
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'status': 'error'
        }), 500


def get_interpretation(prediction, confidence, followers, genre_count):
    """Provide interpretation of the prediction"""
    
    if prediction == 1:
        if confidence > 80:
            return f"High confidence prediction! With {int(followers)} followers and {genre_count} genres, this artist shows strong indicators of popularity."
        else:
            return f"This artist is predicted to be popular, but with moderate confidence. Consider factors like engagement and music quality."
    else:
        if confidence > 80:
            return f"Strong indication that this artist may not reach high popularity with current metrics."
        else:
            return f"Borderline case. The artist's popularity could go either way depending on other factors."


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server',
        'available_endpoints': ['/', '/health', '/predict']
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The method is not allowed for the requested URL',
        'hint': 'Use POST method for /predict endpoint'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'status': 'error'
    }), 500


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 Starting Flask Development Server")
    print("="*70)
    print("\n📍 API will be available at: http://localhost:5000")
    print("📍 Test endpoint: http://localhost:5000/health")
    print("\n💡 Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Makes the server publicly available
        port=int(os.environ.get('PORT', 5000)),  # Use PORT env variable or default to 5000
        debug=False  # Set to False for production
    )