import re

# Charger le fichier d'origine
with open("output.txt", "r") as file:
    lines = file.readlines()

# Liste des lignes formatées
formatted_lines = []

# Expression régulière pour matcher les lignes de métriques
pattern = re.compile(r"Round (\d+) —\s+(\w+)\s*: ([0-9.]+)")

for line in lines:
    match = pattern.match(line)
    if match:
        round_num = match.group(1)
        metric = match.group(2).capitalize()
        value = match.group(3)
        # Formater proprement
        formatted_line = f"Round {round_num} —  {metric} : {value}"
        formatted_lines.append(formatted_line)
    else:
        formatted_lines.append(line.strip())  # Garde les autres lignes intactes

# Sauvegarde dans un nouveau fichier
with open("formatted_output.txt", "w") as output:
    for line in formatted_lines:
        output.write(line + "\n")

print("✅ Formatage terminé ! Le fichier est prêt : formatted_output.txt")
