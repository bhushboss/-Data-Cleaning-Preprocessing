# -Data-Cleaning-Preprocessing
Using Python, Pandas, NumPy, Matplotlib/Seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. Import and Explore Data ---

print("--- 1. Loading and Exploring Data ---")
try:
    df = pd.read_csv("Titanic-Dataset.csv")
except FileNotFoundError:
    print("Error: 'Titanic-Dataset.csv' not found.")
    exit()

# Display basic info
df.info()

# Display missing value counts
print("\nInitial Missing Values:\n", df.isnull().sum())

# --- CRITICAL STEP: Split Data BEFORE Preprocessing ---

print("\n--- Splitting Data (Preventing Data Leakage) ---")
# Separate target (y) from features (X)
# We drop irrelevant columns immediately
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = df['Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# --- 2. Handle Missing Values ---
# Strategy:
# 1. 'Cabin': Too many missing values. Drop the column.
# 2. 'Age' (Numerical): Impute with the median from the *training* data.
# 3. 'Embarked' (Categorical): Impute with the mode (most frequent) from the *training* data.

print("\n--- 2. Handling Missing Values ---")

# 1. Drop 'Cabin'
X_train = X_train.drop(columns=['Cabin'])
X_test = X_test.drop(columns=['Cabin'])
print("Dropped 'Cabin' column.")

# 2. Impute 'Age'
# Calculate median from X_train ONLY
age_median = X_train['Age'].median()
print(f"Calculated 'Age' median from training data: {age_median}")
# Fill missing 'Age' in both sets with the *training* median
X_train['Age'] = X_train['Age'].fillna(age_median)
X_test['Age'] = X_test['Age'].fillna(age_median)
print("Imputed missing 'Age' values.")

# 3. Impute 'Embarked'
# Calculate mode from X_train ONLY
embarked_mode = X_train['Embarked'].mode()[0]
print(f"Calculated 'Embarked' mode from training data: {embarked_mode}")
# Fill missing 'Embarked' in both sets with the *training* mode
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)
X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)
print("Imputed missing 'Embarked' values.")

# Verify no more missing values
print("\nMissing values in X_train after imputation:\n", X_train.isnull().sum())
print("\nMissing values in X_test after imputation:\n", X_test.isnull().sum())


# --- 3. Convert Categorical Features (Encoding) ---
# We will One-Hot Encode 'Pclass', 'Sex', and 'Embarked'

print("\n--- 3. Encoding Categorical Features ---")
categorical_cols = ['Pclass', 'Sex', 'Embarked']

# Initialize encoder
# drop='first' avoids the dummy variable trap
# handle_unknown='ignore' prevents errors if test set has a category not in train set
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Fit and transform on the training data
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
# Get the new column names
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame with the new encoded columns
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)

# Transform the test data (using the encoder *fitted* on train data)
X_test_encoded = encoder.transform(X_test[categorical_cols])
# Create a DataFrame for the test set
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# Drop original categorical columns and concatenate new encoded ones
X_train = X_train.drop(columns=categorical_cols)
X_train = pd.concat([X_train, X_train_encoded_df], axis=1)

X_test = X_test.drop(columns=categorical_cols)
X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

print("Encoded categorical features and combined DataFrames.")
print("New X_train columns:\n", X_train.columns)


# --- 4. Normalize/Standardize Numerical Features ---
# We will scale 'Age', 'Fare', 'SibSp', and 'Parch'

print("\n--- 4. Standardizing Numerical Features ---")
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Initialize scaler
scaler = StandardScaler()

# Fit and transform on the training data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Transform the test data (using the scaler *fitted* on train data)
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Standardized numerical features.")
print("\nX_train head after scaling:\n", X_train.head())


# --- 5. Visualize and Remove Outliers (from Training Data) ---
# We only visualize and remove outliers from the *training set*

print("\n--- 5. Handling Outliers (from Training Data) ---")

# Visualize outliers in 'Fare' before removal
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=X_train['Fare'])
plt.title("'Fare' Outliers (Before Removal)")

# Remove outliers using the IQR method (focus on 'Fare')
Q1 = X_train['Fare'].quantile(0.25)
Q3 = X_train['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Original X_train shape: {X_train.shape}")
print(f"Original y_train shape: {y_train.shape}")
print(f"'Fare' outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Create a mask for rows *within* the outlier bounds
outlier_mask = (X_train['Fare'] >= lower_bound) & (X_train['Fare'] <= upper_bound)

# Apply the mask to *both* X_train and y_train to keep them in sync
X_train = X_train[outlier_mask]
y_train = y_train[outlier_mask]

print(f"New X_train shape after outlier removal: {X_train.shape}")
print(f"New y_train shape after outlier removal: {y_train.shape}")

# Visualize 'Fare' after removal
plt.subplot(1, 2, 2)
sns.boxplot(y=X_train['Fare'])
plt.title("'Fare' Outliers (After Removal)")
plt.tight_layout()
plt.show()


# --- Final Prepared Data ---
print("\n--- PREPARATION COMPLETE ---")
print("Final X_train head:\n", X_train.head())
print("\nFinal X_train info:\n")
X_train.info()
