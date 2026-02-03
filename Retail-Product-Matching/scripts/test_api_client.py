import requests
import json
import sys
from pathlib import Path

def test_prediction(image_path, url="http://localhost:8000/predict"):
    if not Path(image_path).exists():
        print(f"Error: File {image_path} not found")
        return

    print(f"Testing prediction for: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"Total Matches: {len(result['matches'])}")
        print(f"Inference Time: {result['inference_time']:.2f}s")
        
        # Print first 5 matches
        for i, match in enumerate(result['matches'][:5]):
            status = "✅" if match['matched'] else "❌"
            print(f"[{i+1}] {status} {match['class_name']} (Score: {match['score']:.3f})")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # If a path is provided in CLI
    if len(sys.argv) > 1:
        test_prediction(sys.argv[1])
    else:
        # Try to find a sample image in data/test_images
        sample_img = next(Path("data/test_images").glob("*.jpg"), None)
        if sample_img:
            test_prediction(str(sample_img))
        else:
            print("Please provide an image path: python test_api_client.py <path_to_image>")
