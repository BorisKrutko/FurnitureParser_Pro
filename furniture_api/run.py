from application import create_app
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS

load_dotenv()

app = create_app()
app.config['TIMEOUT'] = 600

# Enable CORS for all routes and origins (for development - NOT recommended for production)
CORS(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)