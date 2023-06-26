import requests

generator_endpoint = "http://0.0.0.0:8001"

def test_generator_endpoint():
    response = requests.get(generator_endpoint)
    if response.status_code == 200:
        print("Generator endpoint is working.")
    else:
        print("Generator endpoint is not responding properly.")

test_generator_endpoint()
