import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#Load Dataset
df = pd.read_csv("E:\online_shoppers_intention.csv")
df.head()

#Remove Duplicate Rows
df.drop_duplicates(inplace=True)
df

#Remove sessions where all durations are 0 (not useful for behavior analysis)
df = df[~((df['Administrative_Duration'] == 0) &
          (df['Informational_Duration'] == 0) &
          (df['ProductRelated_Duration'] == 0))].copy()
df


#Fill Missing Values (None here, but safe to keep)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
df


#Encode Booleans + Target
df['Revenue'] = df['Revenue'].astype(int)
df['Weekend'] = df['Weekend'].astype(int)


#Encode Categorical Columns
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)


#Create New Features (optional but useful for EDA/Insights)
df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
df['Total_Pages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']


#Feature Scaling with MinMaxScaler
cols_to_scale = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Total_Duration']
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


#Final Info Check
print(df.info())

################################### OBJECTIVES ##################################

#Objective 1: Analyze the influence of page interaction durations on user purchase behavior
#Page interaction durations vs revenue "Voilin Plot" Shows spread & median comparison for durations


# Ensure Revenue is numeric
df['Revenue'] = df['Revenue'].astype(int)
# Set a clean style
sns.set(style="whitegrid")

# 1. ProductRelated_Duration vs Revenue
plt.figure(figsize=(8, 5))
sns.violinplot(x='Revenue', y='ProductRelated_Duration', data=df, palette='pastel')
plt.title('ProductRelated Duration vs Revenue')
plt.xlabel('Purchase Made (0 = No, 1 = Yes)')
plt.ylabel('ProductRelated Duration')
plt.tight_layout()
plt.show()

# 2. Administrative_Duration vs Revenue
plt.figure(figsize=(8, 5))
sns.violinplot(x='Revenue', y='Administrative_Duration', data=df, palette='Set2')
plt.title('Administrative Duration vs Revenue')
plt.xlabel('Purchase Made (0 = No, 1 = Yes)')
plt.ylabel('Administrative Duration')
plt.tight_layout()
plt.show()

# 3. Informational_Duration vs Revenue
plt.figure(figsize=(8, 5))
sns.violinplot(x='Revenue', y='Informational_Duration', data=df, palette='coolwarm')
plt.title('Informational Duration vs Revenue')
plt.xlabel('Purchase Made (0 = No, 1 = Yes)')
plt.ylabel('Informational Duration')
plt.tight_layout()
plt.show()


#objective 2 : Evaluate the effect of bounce rate and exit rate on conversion 
#likelihood across sessions.
# Scatterplot: BounceRate vs ExitRate colored by Revenue
sns.scatterplot(x='BounceRates', y='ExitRates', hue='Revenue', data=df)
plt.title('Bounce Rate vs Exit Rate by Purchase')
plt.xlabel('Bounce Rate')
plt.ylabel('Exit Rate')
plt.legend(title='Purchase Made')
plt.grid(True)
plt.show()


#Objective 3: Compare purchasing behavior across months and 
#special days to identify seasonal shopping trends

print(df.columns)
# Load original dataset (if Month was one-hot encoded in df)
original_df = pd.read_csv("E:\online_shoppers_intention.csv")

# Define correct month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Filter only rows where purchase was made
purchased = original_df[original_df['Revenue'] == True]

# Plot purchases by month
sns.countplot(x='Month', data=purchased, order=month_order)
plt.title('Purchases by Month')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filter only rows where no purchase was made
not_purchased = original_df[original_df['Revenue'] == False]

# Plot non-purchases by month
sns.countplot(x='Month', data=not_purchased, order=month_order)
plt.title('Non-Purchases by Month')
plt.xlabel('Month')
plt.ylabel('Number of Non-Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prepare purchase counts per month
monthly_trend = original_df[original_df['Revenue'] == True]['Month'].value_counts().reindex(month_order).fillna(0)

# Line graph for trend
monthly_trend.plot(kind='line', marker='o', linestyle='-', color='green', figsize=(10, 5))
plt.title('Monthly Purchase Trend')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#Special Day : Purchase vs No Purchase 
plt.figure(figsize=(6, 4))
sns.barplot(x='Revenue', y='SpecialDay', data=df, palette='pastel')
plt.title('Average SpecialDay Proximity by Purchase Outcome')
plt.xlabel('Purchase Made (0 = No, 1 = Yes)')
plt.ylabel('Mean SpecialDay Score')
plt.tight_layout()
plt.show()

#Objective 4: 
#Pie Chart: Visitor Type Distribution   

# Count of each visitor type
visitor_counts = original_df['VisitorType'].value_counts()

# Plot pie chart
plt.pie(visitor_counts, labels=visitor_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Visitor Type Distribution')
plt.axis('equal')  # Ensures pie is a circle
plt.tight_layout()
plt.show()

#Countplot: Visitor Type vs Revenue (Purchasing Behavior)
# Countplot with hue by Revenue
sns.countplot(x='VisitorType', hue='Revenue', data=original_df)
plt.title('Visitor Type vs Purchase Behavior')
plt.xlabel('Visitor Type')
plt.ylabel('Number of Sessions')
plt.legend(title='Purchase Made')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


#Objective 5: Investigate regional and traffic type variations in user purchase behavior
# Create a pivot table to calculate average conversion rate
heat_data = pd.crosstab(df['Region'], df['TrafficType'], values=df['Revenue'], aggfunc='mean').fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heat_data, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Conversion Rate by Region and Traffic Type')
plt.xlabel('Traffic Type')
plt.ylabel('Region')
plt.tight_layout()
plt.show()



################################## ML MODELS #######################################


# 1. Ensure Revenue is encoded as integer
df['Revenue'] = df['Revenue'].astype(int)

# 2. Encode categorical features ('Month' and 'VisitorType') using One-Hot Encoding
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

# 3. Define features and target
X = df.drop('Revenue', axis=1)  # All input features
y = df['Revenue']              # Target label

# 4. Train-Test Split (80-20) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Feature Scaling with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#################################### Logistic regression ###########################################
# 6. Initialize Logistic Regression with class_weight balanced to handle imbalance
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
# 7. Train the Model
log_model.fit(X_train_scaled, y_train)
# 8. Make Predictions
y_pred = log_model.predict(X_test_scaled)

# 9. Evaluate the Model
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Accuracy: {acc:.4f}")
print("Logistic Regression Train Accuracy:", log_model.score(X_train_scaled, y_train))
print("Logistic Regression Test Accuracy :", log_model.score(X_test_scaled, y_test))
print("Confusion Matrix:\n", conf_mat)

# Plot Confusion Matrix for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Purchase (0)', 'Purchase (1)'],
            yticklabels=['No Purchase (0)', 'Purchase (1)'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

######################################### KNN MODEL ###############################################

#Initialize and Train the KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can experiment with n_neighbors
knn_model.fit(X_train_scaled, y_train)

#Make Predictions and Evaluate the Model
y_pred_knn = knn_model.predict(X_test_scaled)

# Calculate Accuracy
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn:.4f}")

print("KNN Train Accuracy:", knn_model.score(X_train_scaled, y_train))
print("KNN Test Accuracy :", knn_model.score(X_test_scaled, y_test))

# Confusion Matrix
conf_mat_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN Confusion Matrix:\n", conf_mat_knn)

# Plot Confusion Matrix for KNN
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Purchase (0)', 'Purchase (1)'],
            yticklabels=['No Purchase (0)', 'Purchase (1)'])
plt.title('Confusion Matrix - KNN Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

############################# RANDOM FOREST ##############################################

#Initialize and Train Random Forest Model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')  # class_weight='balanced' to handle imbalance
rf_model.fit(X_train_scaled, y_train)

#Make Predictions and Evaluate the Model
y_pred_rf = rf_model.predict(X_test_scaled)

# After training
print("Train:", rf_model.score(X_train_scaled, y_train))
print("Test :", rf_model.score(X_test_scaled, y_test))
# Calculate Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# Confusion Matrix
conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", conf_mat_rf)

############## SHOWED SIGNS OF OVERFITTING 
################# BALANCING ( FINE TUNING RANDOM FOREST )

#Initialize and Train Optimized Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Limit depth to avoid overfitting
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=4,      # Minimum samples in each leaf node
    class_weight='balanced', # Handle class imbalance
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)  # Train the model

#Make Predictions and Evaluate the Model
y_pred_rf = rf_model.predict(X_test_scaled)

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# Confusion Matrix
conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", conf_mat_rf)

# Train vs Test Scores (to check overfitting)
print("Train Accuracy:", rf_model.score(X_train_scaled, y_train))
print("Test Accuracy :", rf_model.score(X_test_scaled, y_test))

# Plot Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Purchase (0)', 'Purchase (1)'],
            yticklabels=['No Purchase (0)', 'Purchase (1)'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()




