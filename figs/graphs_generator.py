import pandas as pd
import matplotlib.pyplot as plt

# File path and sheet configuration
file_path = "Confronto_Dati.ods"
sheet_name = "Ext.1"

# Load the sheet using row index 2 (third row in Calc) as header
df = pd.read_excel(file_path, sheet_name=sheet_name, header=2, engine="odf")

# Select only the first 21 rows corresponding to the first bias table (Age)
df_age = df.iloc[:21].copy()

# Extract series using positional indices
steering_coeff = df_age.iloc[:, 0].astype(float)

bbq_reproduced = df_age.iloc[:, 9].astype(float)
mmlu_reproduced = df_age.iloc[:, 10].astype(float)

bbq_ext1 = df_age.iloc[:, 14].astype(float)
mmlu_ext1 = df_age.iloc[:, 15].astype(float)

# ---------------------------------------------------------
# Plot 1: BBQ Accuracy Comparison (Age)
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(steering_coeff, bbq_reproduced, label="Reproduced (K=0)", marker="o", linestyle="-")
plt.plot(steering_coeff, bbq_ext1, label="Ext.1 (K=10; B=1.0)", marker="s", linestyle="--")

plt.xlabel("Steering Coefficient ($\lambda$)")
plt.ylabel("BBQ Accuracy")
plt.title("BBQ Accuracy vs Steering Coefficient (Age)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("bbq_accuracy_age.pdf")
plt.show()

# ---------------------------------------------------------
# Plot 2: MMLU Accuracy Comparison (Age)
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
plt.plot(steering_coeff, mmlu_reproduced, label="Reproduced (K=0)", marker="o", linestyle="-", color="green")
plt.plot(steering_coeff, mmlu_ext1, label="Ext.1 (K=10; B=1.0)", marker="s", linestyle="--", color="orange")

plt.xlabel("Steering Coefficient ($\lambda$)")
plt.ylabel("MMLU Accuracy")
plt.title("MMLU Accuracy vs Steering Coefficient (Age)")
plt.grid(True, linestyle="--", alpa=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("mmlu_accuracy_age.pdf")
plt.show()