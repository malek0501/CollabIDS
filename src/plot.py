import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_metrics(filename):
    data = {}
    with open(filename, "r") as f:
        for line in f:
            match = re.search(r"Round (\d+) —\s+(\w+)\s*: ([0-9.]+)", line)
            if match:
                round_num = int(match.group(1))
                metric = match.group(2).lower()
                value = float(match.group(3))

                if round_num not in data:
                    data[round_num] = {}
                data[round_num][metric] = value

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Round'
    df.reset_index(inplace=True)
    df.rename(columns={
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'loss': 'Loss'
    }, inplace=True)
    
    # Assure tous les rounds 1-40 sont présents
    df = df.set_index("Round").reindex(range(1, 41)).reset_index()
    return df

# Charger les données
df1 = parse_metrics("out.txt")
df2 = parse_metrics("out1.txt")

# Liste des métriques à comparer
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Loss"]

# Générer les 5 graphiques
for metric in metrics:
    plt.figure(figsize=(10, 5))
    
    # Tracer les deux courbes
    plt.plot(df1["Round"], df1[metric], label=f"{metric} - Heterogeneous Data Distribution", marker='o', linestyle='-')
    plt.plot(df2["Round"], df2[metric], label=f"{metric} - Homogeneous Data Distribution", marker='s', linestyle='--')
    
    plt.title(f"Comparaison de {metric} sur 40 rounds", fontsize=14)
    plt.xlabel("Round")
    plt.ylabel("Valeur")
    
    if metric.lower() != "loss":
        plt.ylim(0, 1.05)
    else:
        max_loss = max(df1[metric].max(), df2[metric].max())
        plt.ylim(0, max_loss * 1.1)
        
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric.lower().replace(' ', '_')}_comparison.png")
    plt.close()
