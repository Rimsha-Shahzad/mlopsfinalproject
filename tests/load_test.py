import requests, random, time
url = "http://localhost:8000/predict"
for i in range(100):
    features = [random.uniform(-3, 3) for _ in range(30)]
    features[0]  = random.uniform(0, 172800)   # Time
    features[29] = random.uniform(0, 500)       # Amount
    
    r = requests.post(url, json={"features": features})
    result = r.json()
    print(f"[{i+1}] {result['prediction']} (score: {result['fraud_score']})")
    time.sleep(0.2)
