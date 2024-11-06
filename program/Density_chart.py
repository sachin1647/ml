import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import Seaborn for the density plot

# Density Plot
data = np.random.randn(1000)
sns.kdeplot(data, fill=True, color='blue', label='Density Plot')
plt.xlabel('X-Axis Label')
plt.ylabel('Density')
plt.title('Density Plot Example')
plt.legend()
plt.show()

# Bubble Diagram
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 18, 20]
sizes = [100, 200, 300, 150, 250]
plt.scatter(x, y, s=sizes, alpha=0.5)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Bubble Chart Example')
plt.show()