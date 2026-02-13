import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/summary.csv')

# Accuracy bar graph
df.plot(
    x='Dataset',
    y=['mAP@0.5', 'mAP@0.5:0.95'],
    kind='bar'
)
plt.title('Accuracy Comparison')
plt.ylabel('mAP')
plt.tight_layout()
plt.savefig('results/accuracy_bar.png')
plt.close()

# Inference time bar graph
df.plot(
    x='Dataset',
    y='Inference_Time_sec',
    kind='bar'
)
plt.title('Inference Time Comparison')
plt.ylabel('Seconds')
plt.tight_layout()
plt.savefig('results/time_bar.png')
plt.close()

print('Graphs generated successfully')
