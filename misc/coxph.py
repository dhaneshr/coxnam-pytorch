import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt


from lifelines.datasets import load_rossi
df = load_rossi()

cph = CoxPHFitter()
cph.fit(df, duration_col='week', event_col='arrest')
cph.print_summary()


risk_scores = cph.predict_partial_hazard(df)


c_index = concordance_index(df['week'], risk_scores, df['arrest'])
print(f"C-index: {c_index:.4f}")


sample_data = df.sample(5)
survival_functions = cph.predict_survival_function(sample_data)


plt.figure(figsize=(12, 8))

for i in range(len(survival_functions.columns)):
    surv_func = survival_functions.iloc[:, i]  
    plt.step(surv_func.index, surv_func.values, where="post", label=f"Sample {i+1}")

plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Functions')
plt.legend()
plt.grid(True)
plt.show()

