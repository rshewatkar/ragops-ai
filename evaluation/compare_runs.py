import mlflow
import pandas as pd

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "ragops-ai"
CSV_OUTPUT_PATH = "rag_experiments.csv"


# =========================
# 📥 GET RUNS
# =========================
def get_runs():
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME]
    )
    return runs


# =========================
# 🧠 SCORING FUNCTION
# =========================
def score_run(row):
    relevance = row.get("metrics.context_relevance", 0)
    coverage = row.get("metrics.answer_coverage", 0)
    found = row.get("metrics.is_found", 0)
    length_penalty = min(row.get("metrics.answer_length", 0) / 200, 1)

    score = (
        0.4 * relevance +
        0.3 * coverage +
        0.2 * found +
        0.1 * (1 - length_penalty)
    )

    return round(score, 3)

# =========================
# 🏆 BEST RUN
# =========================
def get_best_run():
    runs = get_runs()

    if runs.empty:
        raise ValueError(f"No runs found for experiment '{EXPERIMENT_NAME}'.")

    runs["score"] = runs.apply(score_run, axis=1)

    best_run = runs.sort_values(by="score", ascending=False).iloc[0]

    return best_run, runs


# =========================
# 📊 PRINT BEST CONFIG
# =========================
def print_best_config():
    best, _ = get_best_run()

    print("\n🏆 BEST RAG CONFIG\n")

    print("Query:", best.get("params.query"))
    print("Query Type:", best.get("params.query_type"))
    print("k:", best.get("params.retrieval_k"))
    print("Response Type:", best.get("params.response_type"))

    print("\n📊 Metrics:")
    print("Relevance:", best.get("metrics.context_relevance"))
    print("Coverage:", best.get("metrics.answer_coverage"))
    print("Answer Length:", best.get("metrics.answer_length"))
    print("Score:", best.get("score"))


# =========================
# 💾 EXPORT RUNS
# =========================
def export_runs(path=CSV_OUTPUT_PATH):
    runs = get_runs()

    if runs.empty:
        raise ValueError(f"No runs found for experiment '{EXPERIMENT_NAME}'.")

    runs["score"] = runs.apply(score_run, axis=1)
    runs = runs.sort_values(by="score", ascending=False)

    runs.to_csv(path, index=False)
    return path


# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    output_path = export_runs()
    print(f"\n✅ Saved runs to: {output_path}")

    print_best_config()