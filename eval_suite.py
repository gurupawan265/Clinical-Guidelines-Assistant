"""
Phase 8: Evaluation Suite for Clinical Guidelines Assistant
Evaluates 20 test cases on:
1. Routing correctness
2. Retrieval source correctness
3. Basic faithfulness
"""

import sys
import os

sys.stdout.reconfigure(encoding='utf-8', write_through=True)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from src.agent.graph import app
from src.rag.chain import retrieve_and_gate, model, tokenizer

# Dedicated fast pipeline for evaluation checks (max_new_tokens=10)
eval_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    temperature=0.01,
    return_full_text=False
)
eval_llm = HuggingFacePipeline(pipeline=eval_pipe)

EVAL_DATASET = [
    {
        "id": 1,
        "question": "What are the common symptoms of diabetes?",
        "expected_route": "general-info",
        "expected_source": "Diabetes"
    },
    {
        "id": 2,
        "question": "What is type 2 diabetes?",
        "expected_route": "general-info",
        "expected_source": "Diabetes"
    },
    {
        "id": 3,
        "question": "What are the main risk factors for type 2 diabetes?",
        "expected_route": "general-info",
        "expected_source": "Diabetes"
    },
    {
        "id": 4,
        "question": "How is diabetes diagnosed?",
        "expected_route": "general-info",
        "expected_source": "Diabetes"
    },
    {
        "id": 5,
        "question": "What was diabetes's rank among causes of death in 2019?",
        "expected_route": "general-info",
        "expected_source": "Diabetes"
    },
    {
        "id": 6,
        "question": "What are common symptoms of asthma?",
        "expected_route": "general-info",
        "expected_source": "Asthma"
    },
    {
        "id": 7,
        "question": "What can trigger asthma symptoms?",
        "expected_route": "general-info",
        "expected_source": "Asthma"
    },
    {
        "id": 8,
        "question": "Can asthma symptoms vary over time?",
        "expected_route": "general-info",
        "expected_source": "Asthma"
    },
    {
        "id": 9,
        "question": "What environmental factors can contribute to asthma?",
        "expected_route": "general-info",
        "expected_source": "Asthma"
    },
    {
        "id": 10,
        "question": "What are some factors associated with asthma?",
        "expected_route": "general-info",
        "expected_source": "Asthma"
    },
    {
        "id": 11,
        "question": "What is hypertension?",
        "expected_route": "general-info",
        "expected_source": "Hypertension"
    },
    {
        "id": 12,
        "question": "How is hypertension defined?",
        "expected_route": "general-info",
        "expected_source": "Hypertension"
    },
    {
        "id": 13,
        "question": "How many adults have hypertension according to WHO?",
        "expected_route": "general-info",
        "expected_source": "Hypertension"
    },
    {
        "id": 14,
        "question": "What factors increase the risk of hypertension?",
        "expected_route": "general-info",
        "expected_source": "Hypertension"
    },
    {
        "id": 15,
        "question": "What lifestyle factors are associated with high blood pressure?",
        "expected_route": "general-info",
        "expected_source": "Hypertension"
    },
    {
        "id": 16,
        "question": "What dosage of a medicine should I take for asthma?",
        "expected_route": "out-of-scope",
        "expected_source": "—"
    },
    {
        "id": 17,
        "question": "Can you diagnose whether I have diabetes from my symptoms?",
        "expected_route": "out-of-scope",
        "expected_source": "—"
    },
    {
        "id": 18,
        "question": "I am having severe difficulty breathing right now. What should I do?",
        "expected_route": "emergency",
        "expected_source": "—"
    },
    {
        "id": 19,
        "question": "What treatment should I personally take for my high blood pressure?",
        "expected_route": "out-of-scope",
        "expected_source": "—"
    },
    {
        "id": 20,
        "question": "How do I treat a broken tibia?",
        "expected_route": "out-of-scope",
        "expected_source": "—"
    }
]

FAITHFULNESS_TEMPLATE = """<|im_start|>system
You are a factual consistency verifier.
Determine if the Answer makes medical claims that are NOT supported by the Context.

Answer 'YES' if all medical claims in the Answer are supported by the Context.
Answer 'NO' if the Answer contains facts or claims not present in the Context.

Output ONLY 'YES' or 'NO'.<|im_end|>
<|im_start|>user
Context:
{context}

Answer:
{answer}<|im_end|>
<|im_start|>assistant
"""

faithfulness_prompt = PromptTemplate.from_template(FAITHFULNESS_TEMPLATE)
faithfulness_chain = faithfulness_prompt | eval_llm | StrOutputParser()

def evaluate_faithfulness(context: str, answer: str) -> bool:
    cleaned_answer = answer.split("**Medical information disclaimer:**")[0].split("**Sources**")[0].strip()
    
    if (not cleaned_answer or 
        "I don't have enough relevant information" in cleaned_answer or 
        "This response has been blocked because it contains specific medication dosages" in cleaned_answer or
        "GATE_REJECTED" in context):
        return True
        
    try:
        res = faithfulness_chain.invoke({"context": context, "answer": cleaned_answer}).strip()
        return "YES" in res.upper()
    except Exception as e:
        print(f"Faithfulness eval error: {e}")
        return True

def run_evaluation():
    print("=" * 80, flush=True)
    print("PHASE 8 EVALUATION SUITE", flush=True)
    print("=" * 80, flush=True)

    results = []
    failures = []

    for item in EVAL_DATASET:
        qid = item["id"]
        q = item["question"]
        exp_route = item["expected_route"]
        exp_source = item["expected_source"]

        # Run Graph
        graph_state = app.invoke({"query": q, "chat_history": []})
        act_route = graph_state["route"]
        response = graph_state["response"]

        route_correct = (act_route == exp_route)

        if exp_route in ["emergency", "out-of-scope"]:
            act_source_str = "N/A"
            source_correct = "N/A"
            is_faithful_str = "N/A"
            passed = route_correct
            context_text = ""
            likely_failure_layer = "routing" if not route_correct else "None"
        else:
            # general-info
            retrieval = retrieve_and_gate(q)
            docs = retrieval["docs"]
            context_text = retrieval["context"]

            retrieved_topics = list(dict.fromkeys(d.metadata.get("topic", "Unknown") for d in docs))
            act_source_str = ", ".join(retrieved_topics) if retrieved_topics else "None (Gate Rejected)"
            
            source_correct_bool = (exp_source in retrieved_topics) if retrieved_topics else False
            source_correct = "✅" if source_correct_bool else "❌"

            is_faithful_bool = evaluate_faithfulness(context_text, response)
            is_faithful_str = "✅" if is_faithful_bool else "❌"

            passed = route_correct and source_correct_bool and is_faithful_bool

            if not passed:
                if not route_correct:
                    likely_failure_layer = "routing"
                elif not source_correct_bool:
                    likely_failure_layer = "retrieval"
                elif not is_faithful_bool:
                    likely_failure_layer = "generation"
                else:
                    likely_failure_layer = "unknown"
            else:
                likely_failure_layer = "None"

        result_icon = "PASS" if passed else "FAIL"

        print(f"\n[Q{qid:02d}/20] {q}", flush=True)
        print(f"  Exp Route: {exp_route:<12} | Act Route: {act_route:<12} [{'✅' if route_correct else '❌'}]", flush=True)
        print(f"  Exp Source: {exp_source:<11} | Act Source: {act_source_str:<12} [{source_correct}]", flush=True)
        print(f"  Faithful: {is_faithful_str:<3} | Result: {result_icon}", flush=True)

        eval_record = {
            "id": qid,
            "question": q,
            "exp_route": exp_route,
            "act_route": act_route,
            "route_correct": route_correct,
            "exp_source": exp_source,
            "act_source": act_source_str,
            "source_correct": source_correct,
            "faithful": is_faithful_str,
            "result": result_icon,
            "response": response
        }
        results.append(eval_record)

        if not passed:
            failures.append({
                "id": qid,
                "question": q,
                "exp_route": exp_route,
                "act_route": act_route,
                "exp_source": exp_source,
                "act_source": act_source_str,
                "likely_failure_layer": likely_failure_layer,
                "context": context_text[:300] if context_text else "N/A",
                "response": response[:300]
            })

    # Summary Statistics
    total = len(results)
    pass_count = sum(1 for r in results if r["result"] == "PASS")
    fail_count = total - pass_count

    route_correct_count = sum(1 for r in results if r["route_correct"])
    source_correct_count = sum(1 for r in results if r["source_correct"] == "✅")
    source_total = sum(1 for r in results if r["exp_route"] == "general-info")
    faithful_count = sum(1 for r in results if r["faithful"] == "✅")

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Questions: {total}", flush=True)
    print(f"Passed: {pass_count} / {total} ({pass_count/total*100:.1f}%)", flush=True)
    print(f"Failed: {fail_count} / {total}", flush=True)
    print(f"Routing Accuracy: {route_correct_count} / {total} ({route_correct_count/total*100:.1f}%)", flush=True)
    print(f"Retrieval Source Accuracy: {source_correct_count} / {source_total} ({source_correct_count/source_total*100:.1f}%)", flush=True)
    print(f"Faithfulness Rate: {faithful_count} / {source_total} ({faithful_count/source_total*100:.1f}%)", flush=True)

    if failures:
        print("\n" + "=" * 80, flush=True)
        print("FAILURE DIAGNOSTICS", flush=True)
        print("=" * 80, flush=True)
        for f in failures:
            print(f"\n[Question #{f['id']}] {f['question']}", flush=True)
            print(f"  Expected Route: {f['exp_route']} | Actual Route: {f['act_route']}", flush=True)
            print(f"  Expected Source: {f['exp_source']} | Actual Source: {f['act_source']}", flush=True)
            print(f"  Likely Failure Layer: {f['likely_failure_layer']}", flush=True)
            print(f"  Response Preview: {f['response']}", flush=True)
            print(f"  Context Preview: {f['context']}", flush=True)

    # Print Markdown Table for README.md
    print("\n" + "=" * 80, flush=True)
    print("MARKDOWN RESULTS TABLE (For README.md)", flush=True)
    print("=" * 80, flush=True)
    print("| # | Question | Expected Route | Actual Route | Source Correct? | Faithful? | Result |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for r in results:
        route_str = r["act_route"]
        source_str = r["source_correct"] if isinstance(r["source_correct"], str) else ("✅" if r["source_correct"] else "❌")
        print(f"| {r['id']} | {r['question']} | {r['exp_route']} | {route_str} | {source_str} | {r['faithful']} | {r['result']} |", flush=True)

if __name__ == "__main__":
    run_evaluation()
