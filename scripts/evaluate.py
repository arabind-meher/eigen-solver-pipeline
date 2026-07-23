from evaluation import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(
        results_path="results/result_step_eigen_openai.json",
        dataset_path="data/dataset.json",
        output_path="m_results/m_result_step_eigen_openai.json",
    )
