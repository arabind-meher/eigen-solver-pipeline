from evaluation import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(
        results_path="results/result_hybrid_openai.json",
        dataset_path="data/dataset.json",
        output_path="m_results/m_result_hybrid_openai.json",
    )
