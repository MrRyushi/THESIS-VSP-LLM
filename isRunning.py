import requests

response = requests.get("http://localhost:8000/health")
print(response.json())  # Should print {'status': 'ok'} if running
