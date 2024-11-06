import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pd.read_csv("house-price.csv")
X = data[['No_Rooms', 'Sq_Foot', 'Age']].values
Y = data['Price'].values
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)
plt.scatter(Y, Y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
input_values = [int(input("Enter the number of rooms: ")),
int(input("Enter the square footage: ")),
int(input("Enter the age: "))]
predicted_price = model.predict([input_values])
print("Predicted Price:", predicted_price[0])
plt.scatter(Y, Y_pred, color='blue', label='Actual Prices')
plt.scatter(predicted_price, predicted_price, color='red', marker='x', label='Predicted Price')
plt.legend()
plt.show()