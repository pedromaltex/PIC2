############################
##      MASTER PANDAS      #
############################
import pandas as pd

# ✅ Reading CSV Files

# Read a CSV file into a DataFrame
df = pd.read_csv("employees.csv")

# Display first 5 rows
print(df.head())

#✅ Filtering Data
print('*' * 50)
# Filter rows where column 'age' is greater than 25
df_filtered = df[df["age"] > 30]
print(df_filtered)

#✅ Aggregating Data
print('*' * 50)
# Calculate the mean of the "salary" column
mean_salary = df["salary"].mean()
print(mean_salary)

# Group by department and calculate the average salary
df_grouped = df.groupby("department")["salary"].mean()
print(df_grouped)

#✅ Merging DataFrames
print('*' * 50)
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
df2 = pd.DataFrame({"id": [1, 2, 3], "salary": [50000, 60000, 55000]})

# Merge on the "ID" column
print('*' * 50)
df_merged = pd.merge(df1, df2, on="id")
print(df_merged)

#✅ Reshaping Data
print('*' * 50)
# Pivot table example
df_pivot = df.pivot_table(values="salary", index="department", columns="gender", aggfunc="mean")
print(df_pivot)


############################
##  MATPLOTLIB AND SEABORN #
############################

import matplotlib.pyplot as plt
import seaborn as sns

#✅ Histogram
# Create a histogram of salaries
plt.hist(df["salary"], bins=10, color="blue", edgecolor="black")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()

#✅ Scatter Plot
# Scatter plot of salary vs. experience
plt.scatter(df["experience"], df["salary"], color="red")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs. Salary")
plt.show()

#✅ Line Chart
# Example line chart
df["date"] = pd.to_datetime(df["date"])  # Convert date column
df_sorted = df.sort_values("date")

plt.plot(df_sorted["date"], df_sorted["sales"], marker="o")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Over Time")
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.show()

#✅ Seaborn Heatmap
# Create a correlation heatmap
#plt.figure(figsize=(8, 6))
#sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
#plt.title("Correlation Heatmap")
#plt.show()



#############################
#      RELEVANT HERE        #
#############################


#✅ Splitting Data for Training and Testing
from sklearn.model_selection import train_test_split

# Assume we have a dataset with features (X) and labels (y)
X = df[["experience", "age"]]  # Features
y = df["salary"]  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#✅ Training a Linear Regression Model
from sklearn.linear_model import LinearRegression

# Initialize model
model = LinearRegression()

# Train the model on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print(y_pred[:5])  # Show first 5 predictions

#✅ Evaluating Model Performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

#✅ Cross-Validation

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")

# Convert negative scores to positive
mse_scores = -scores
print("Cross-Validation MSE:", mse_scores.mean())

