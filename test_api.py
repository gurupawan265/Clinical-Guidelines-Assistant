import urllib.request
import json
import traceback

def test_api():
    print("Testing /health...")
    try:
        res = urllib.request.urlopen('http://localhost:8000/health').read().decode()
        print("HEALTH:", res)
    except Exception as e:
        print("HEALTH ERROR:", e)

    print("\nTesting /chat...")
    try:
        req = urllib.request.Request(
            'http://localhost:8000/chat',
            data=json.dumps({
                "message": "What are the common symptoms of diabetes?",
                "conversation_id": "test-conversation-1"
            }).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        res = urllib.request.urlopen(req).read().decode()
        print("CHAT RESPONSE:", json.dumps(json.loads(res), indent=2))
    except Exception as e:
        print("CHAT ERROR:", e)
        traceback.print_exc()

if __name__ == '__main__':
    test_api()
