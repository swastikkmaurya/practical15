# practical15

Aim: Data Normalization and data type conversion

theory:

Data normalization is the process of rescaling numerical features so that they fall within a common range or follow a standard distribution. Raw numerical data often spans vastly different scales — for example, a Price column may contain values in the tens of thousands while a Discount column contains values between 0 and 100. When such features are used together in machine learning algorithms (particularly distance-based algorithms like k-nearest neighbours, k-means clustering, or gradient descent-based models), columns with larger magnitudes dominate computations and can cause biased, incorrect, or slow-converging results.

df['Price'].min(): Identifies the smallest numerical value within the specified 'Price' column.

df['Price'].max(): Identifies the largest numerical value within the specified 'Price' column.

df['Normalized_Price'] = (df['Price'] - min_val) / (max_val - min_val): Implements the Min-Max scaling formula to transform values into a range between 0 and 1.

df['Units_Sold'].mean(): Calculates the average value of the 'Units_Sold' column.

df['Units_Sold'].std(): Measures the amount of variation or dispersion in the 'Units_Sold' numerical values.

df['Standardized_Units'] = (df['Units_Sold'] - mean_val) / std_val: Performs Z-score standardization to center data around a mean of 0 with a standard deviation of 1.

df.astype({'Age': 'int', 'Salary': 'float'}): Explicitly converts the data types of specific columns to integers or floating-point numbers.

pd.to_datetime(df['Joining_Date']): Converts string-based date representations into standardized datetime objects.

pd.get_dummies(df, columns=['Department']): Performs One-Hot Encoding by converting categorical variables into multiple binary (0 or 1) columns.

Data Loading: Create a DataFrame with diverse data types (Numerical and Categorical).

Normalization Workflow:

Apply Min-Max to the Price column to compress values between 0 and 1.

Apply Z-Score to the Units_Sold column to center the mean at 0.

Apply Decimal Scaling to the Price column for simple magnitude reduction.

Encoding Workflow:

Use sklearn.preprocessing.LabelEncoder for binary/ordinal columns like Gender.

Use pd.get_dummies() for nominal columns like Payment_Method.

Enable drop_first=True in dummy encoding to reduce redundancy.

Verification: Print the head of the DataFrame after each transformation to observe the mathematical shifts in values.

Conclusion:

In this experiment, we mastered the transition from raw data to model-ready data.

Key findings:

Normalization is non-negotiable for distance-based algorithms; without it, the Price column would have overwhelmed the Discount column during calculation.

Label Encoding is efficient but can accidentally imply a mathematical relationship (e.g., thinking City 4 is "greater" than City 1), which is why One-Hot Encoding is often safer for geographical data like City.

We successfully handled potential errors in Z-score calculation by ensuring the denominator used the standard deviation of the specific column being scaled, rather than a different feature's max value.
