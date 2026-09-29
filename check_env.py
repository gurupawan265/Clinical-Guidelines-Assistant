import os
from dotenv import load_dotenv, find_dotenv

def check():
    print("find_dotenv:", find_dotenv())
    load_dotenv(find_dotenv())
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        print("KEY LOADED: YES (length:", len(key), ")")
    else:
        print("KEY LOADED: NO")

if __name__ == "__main__":
    check()
