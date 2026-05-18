import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from prompts.innovation import get_prompt

load_dotenv(os.path.join("config", "openai.env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()

    os.makedirs("results", exist_ok=True)
    with open("results/raw_innovation_openai.txt", "a", encoding="utf-8") as raw_file:
        raw_file.write(f"{'='*50}\n")
        raw_file.write(raw)
        raw_file.write(f"\n{'='*50}\n\n")

    return json.loads(raw)


def run_innovation(dataset_path: str, output_path: str, limit: int = None):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    results = []
    for i, sample in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] Processing {sample['id']}...")

        prompt = get_prompt(
            matrix=sample["equation"]["matrix"],
            dimension=sample["equation"]["dimension"],
            intermediate_steps=sample["intermediate_steps"],
        )

        result = None
        try:
            llm_output = call_openai(prompt)
            result = {
                "id": sample["id"],
                "equation": sample["equation"],
                "intermediate_steps": llm_output.get("intermediate_steps", []),
                "result": llm_output.get("result", {}),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if result is not None:
                is_valid = (
                    bool(result.get("intermediate_steps"))
                    and bool(result.get("result"))
                    and bool(result["result"].get("eigenvalues"))
                    and bool(result["result"].get("eigenvectors"))
                )
                if is_valid:
                    results.append(result)
                    print(f"  OK: {sample['id']}")
                else:
                    print(f"  SKIPPED: {sample['id']} — empty values in result")
            else:
                print(f"  SKIPPED: {sample['id']} — API call failed")

        time.sleep(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)}/{len(dataset)} saved to {output_path}")


if __name__ == "__main__":
    run_innovation(dataset_path="data/dataset.json", output_path="results/result_innovation_openai.json", limit=1)
