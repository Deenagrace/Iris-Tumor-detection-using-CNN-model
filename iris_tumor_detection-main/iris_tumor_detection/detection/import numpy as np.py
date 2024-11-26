import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (X: feature, y: target)
X = np.array([[1], [2], [3], [4], [5]])  # Feature
y = np.array([1, 2, 1.9, 4.2, 5.1])     # Target

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the slope and intercept
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Plot the data points and the fitted line
plt.scatter(X, y, color='blue')  # Data points
plt.plot(X, model.predict(X), color='red')  # Fitted line
plt.title("Linear Regression")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.show()
