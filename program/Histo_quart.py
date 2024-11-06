import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
marks = np.array([10,18,34,37, 33, 38, 34, 24, 80, 45, 49, 27, 31, 35, 42])
fig, ax = plt.subplots(figsize =(4, 4))
ax.hist(marks, color = "darkcyan", ec="black", lw=1)
plt.title('Histogram: Exam Score')
plt.ylabel('No. of students')
plt.xlabel('Score')
plt.figure(figsize=(4, 4))
sns.boxplot(y=marks, color='darkcyan')
plt.title('Quartile Plot: Exam Score')
plt.ylabel('Score')
plt.show()