import unittest
import json
import os
import sys

from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt5e import app


load_dotenv()

VALID_API_KEY = os.environ.get("VALID_API_KEY")

class FlaskApiTest(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        # You can set the API key for testing here
        self.api_key = VALID_API_KEY
        
    def test_gtp5e_endpoint_valid_request(self):
        # Test the /gtp5e endpoint with a valid request
        response = self.app.post('/gtp5e/question', 
                                 headers={'x-api-key': self.api_key, 'Content-Type': 'application/json'},
                                 data=json.dumps({'prompt': 'Test prompt'}))
        self.assertEqual(response.status_code, 200)
        # Further checks can be made on the response data

    def test_gtp5e_endpoint_invalid_api_key(self):
        # Test the /gtp5e endpoint with an invalid API key
        response = self.app.post('/gtp5e/question', 
                                 headers={'x-api-key': 'wrong_key', 'Content-Type': 'application/json'},
                                 data=json.dumps({'prompt': 'Test prompt'}))
        self.assertEqual(response.status_code, 401)

    def test_gtp5e_endpoint_invalid_content_type(self):
        # Test the /gtp5e endpoint with invalid content type
        response = self.app.post('/gtp5e/question', 
                                 headers={'x-api-key': self.api_key, 'Content-Type': 'text/plain'},
                                 data='Test prompt')
        self.assertEqual(response.status_code, 400)

# Add more tests as needed

if __name__ == '__main__':
    unittest.main()