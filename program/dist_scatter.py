import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
x_values = [1, 2, 3, 4, 5]
y_values = [10, 15, 13, 18, 20]
data_values = [10, 15, 13, 18, 20]
df = pd.DataFrame({'x': x_values, 'y': y_values, 'value': data_values})
heatmap_data = df.pivot_table(index='x', columns='y', values='value')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Heatmap Example')
plt.show()
#Scatter Plot
x_values = [1, 2, 3, 4, 5]
data_values = [10, 15, 13, 18, 20]
df = pd.DataFrame({'x': x_values, 'value': data_values})
plt.scatter(df['x'], df['value'], marker='o', color='blue', label='Data Points')
plt.xlabel('No. of students')
plt.ylabel('Values')
plt.title('Scatter Plot based on Heatmap Values')
plt.show()