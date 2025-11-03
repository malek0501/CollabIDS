import flwr as fl
import os
import time
from loader import ModelLoader
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters


# Set TensorFlow logging to minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- CONFIGURATION ----------
num_rounds = 40  # Set the number of federated training rounds
server_address = "192.168.1.8:8080"
output_directory = "/home/top/Téléchargements/FLIDS/FLIDS-main"
output_filename = "output.txt"
final_weights = None

# -----------------------------------

# ---------- METRICS AGGREGATION ----------
def weighted_average(metrics):
    total_examples = 0
    aggregated = {}

    # Collect all possible metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    print(all_keys)
    # Initialize each key to 0
    for key in all_keys:
        aggregated[key] = 0.0

    # Weighted sum
    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        print(f"🔎 Client contributed {num_examples} samples, loss = {m.get('loss')}\n")
        for key in all_keys:
            value = client_metrics.get(key)
            #print(f"value:  {value} num_examples: {num_examples}\n")
            if value is not None:
                aggregated[key] += num_examples * value  # weighted
                
    # Normalize by total examples
    if total_examples == 0:
        return {k: 0.0 for k in all_keys}
    dict = {k: v / total_examples for k, v in aggregated.items()}
    print(dict)
    return dict



class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global final_weights
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            final_weights = aggregated_weights[0]  # tuple: (parameters, metrics)
        return aggregated_weights

# ---------- SERVER STRATEGY ----------
def get_server_strategy():
    return CustomFedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
# -----------  MODEL SAVING  ------------------------
def save_model(sample_shape):
    
    # Step 2: Load model architecture
   
    model = ModelLoader.get_model(sample_shape)
    
    # Step 3: Set weights
    from flwr.common import parameters_to_ndarrays
    model.set_weights(parameters_to_ndarrays(final_weights))
    
    # Step 5: Save the global model
    save_path = "global_model.h5"  # or "global_model.keras"
    model.save(save_path)
    
    print(f"\n✅ Global model saved to: {save_path}")
if __name__ == "__main__":
    
    start_time = time.time()

    # Start Flower server
    history = fl.server.start_server(
        server_address=server_address,
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

    total_time = time.time() - start_time

    # ---------- TERMINAL OUTPUT ----------
    print("Available keys in metrics_distributed:", history.metrics_distributed.keys())

    # ------ Define all expected metrics -------
    metric_names = ("accuracy", "precision", "recall", "f1", "loss" )
    present_metrics = [m for m in metric_names if m in history.metrics_distributed]
    if not present_metrics:
        print("\n[❌] No evaluation metrics were recorded.\n")
    else:
        print("\n📊 [Evaluation Metrics Per Round]\n")
    
        # Define icons for each metric
        metric_icons = {
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "loss": ""
            
        }
    
        # Loop through and print each metric if present
        for metric in metric_names:
            if metric in history.metrics_distributed:
                for round_num, value in history.metrics_distributed[metric]:
                    icon = metric_icons.get(metric, "➡️")
                    print(f"Round {round_num} — {icon} {metric.capitalize():<9}: {value:.4f}")

    sample_shape=(42,)
    save_model(sample_shape)
    # -------------------------------------

    
    # ---------- FILE OUTPUT ----------
    summary_output = f"[SUMMARY]\nINFO : Run finished {num_rounds} rounds in {total_time:.2f}s\n"
    
    # Define metrics to include in summary
    metric_names = ("accuracy", "precision", "recall", "f1", "loss")
    present_metrics = [m for m in metric_names if m in history.metrics_distributed]
    
    if not present_metrics:
        summary_output += "\nNo evaluation metrics available.\n"
    else:
        summary_output += "\n[Evaluation Metrics Per Round]\n"
        last_accuracy = None
    
        for metric in metric_names:
            if metric in history.metrics_distributed:
                for round_num, value in history.metrics_distributed[metric]:
                    icon = {
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "loss": ""
                    }.get(metric, "➡️")
                    summary_output += f"Round {round_num} — {icon} {metric.capitalize():<9}: {value:.4f}\n"
                    if metric == "accuracy":
                        last_accuracy = value
    
        if last_accuracy is not None:
            summary_output += f"\nAfter {num_rounds} rounds of training, the final accuracy is {last_accuracy:.2%}\n"
        else:
            summary_output += "\nFinal accuracy not available.\n"
    
    with open(output_filename, "a") as f:
        f.write(summary_output)
    # ----------------------------------




