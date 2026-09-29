import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def test_api():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("FAIL: OPENROUTER_API_KEY not found in env")
        return
    else:
        print("PASS: OPENROUTER_API_KEY is accessible (not printing value)")
        
    try:
        llm = ChatOpenAI(
            model_name="qwen/qwen3.8-27b:free",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            temperature=0.1
        )
        response = llm.invoke("Say the word 'success'")
        if response and response.content:
            print("PASS: OpenRouter API smoke test successful. Content:", response.content)
        else:
            print("FAIL: OpenRouter API smoke test failed to get content")
    except Exception as e:
        print(f"FAIL: OpenRouter API smoke test exception: {type(e).__name__}")

if __name__ == "__main__":
    test_api()
