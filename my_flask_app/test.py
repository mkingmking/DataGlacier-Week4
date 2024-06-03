import unittest
from app import app

class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predict Iris Species', response.data)

    def test_prediction(self):
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2,
            'csrf_token': self.get_csrf_token()
        }
        response = self.app.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted Species:', response.data)

    def test_invalid_input(self):
        data = {
            'sepal_length': 'invalid',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2,
            'csrf_token': self.get_csrf_token()
        }
        response = self.app.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid input', response.data)

    def get_csrf_token(self):
        response = self.app.get('/')
        csrf_token = response.data.decode().split('csrf_token" type="hidden" value="')[1].split('"')[0]
        return csrf_token

if __name__ == '__main__':
    unittest.main()
