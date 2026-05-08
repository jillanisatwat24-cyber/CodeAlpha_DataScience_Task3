from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Feature Engineering: Car Age
df_cars['Car_Age'] = 2024 - df_cars['Year']
df_cars.drop('Year', axis=1, inplace=True)

# 2. One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_cars, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# 3. Define X and y
X = df_encoded.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df_encoded['Selling_Price']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Building: Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# 6. Predictions
y_pred = rf_reg.predict(X_test)

# 7. Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# 8. Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Car Prices')
plt.grid(True)
plt.savefig('car_price_prediction.png')

# 9. Feature Importance
importances = rf_reg.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Feature Importance for Car Price Prediction')
plt.savefig('car_feature_importance.png')