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

Conclusion:

Data Normalization and Data Type Conversion using Python were successfully performed.
