# ML Quick Guide (Pick ONE Model)

Goal: Build ONE simple model end‑to‑end. Don't overthink—get something working, record results, move on.

## 1. Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

Install (once):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Option A: Customer Type (Classification)
Predict: regular vs premium (ignore admin / free-trial for first pass).

### A1. Query & Load
```python
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Connect to your database
conn = psycopg2.connect(
    host="localhost",
    database="ecommerce", 
    user="postgres",
    password="your_password"
)

# Get data from database
query = """
SELECT 
    user_type,
    total_spent,
    purchase_count,
    CASE WHEN last_device = 'ios' THEN 1 ELSE 0 END as is_ios,
    CASE WHEN last_device = 'android' THEN 1 ELSE 0 END as is_android
FROM users 
WHERE user_type IN ('regular', 'premium')
AND total_spent > 0
"""

df = pd.read_sql(query, conn)
print(f"Data shape: {df.shape}")
print(df.head())
```

### A2. Features & Target
```python
# Features (X) - what we use to predict
X = df[['total_spent', 'purchase_count', 'is_ios', 'is_android']]

# Target (y) - what we want to predict
y = df['user_type']

print("Features:")
print(X.describe())
print(f"\nTarget distribution:")
print(y.value_counts())
```

### A3. Split
```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### A4. Train
```python
# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model training completed!")
```

### A5. Evaluate
```python
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Detailed results
print("\nDetailed Results:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

### A6. Single Prediction
```python
# Test prediction on a new customer
new_customer = [[150.50, 5, 1, 0]]  # spent $150.50, 5 purchases, iOS user, not Android
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print(f"Predicted customer type: {prediction[0]}")
print(f"Confidence: {probability[0].max():.2%}")
```

## Option B: Next Product Category (Multi‑Class)
Predict next purchase category (start with top 5 categories by frequency to simplify).

### B1. Prepare
```python
# Get customer purchase history
query = """
SELECT 
    u.user_type,
    u.total_spent,
    u.purchase_count,
    p.product_category
FROM users u
JOIN purchases p ON u.email = p.user_email
WHERE u.purchase_count > 1
"""

df = pd.read_sql(query, conn)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['user_type_encoded'] = le.fit_transform(df['user_type'])
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['product_category'])
```

### B2. Train & Evaluate
```python
# Features and target
X = df[['user_type_encoded', 'total_spent', 'purchase_count']]
y = df['category_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Show actual category names in results
y_test_categories = category_encoder.inverse_transform(y_test)
y_pred_categories = category_encoder.inverse_transform(y_pred)

print("Sample Predictions:")
for i in range(5):
    print(f"Actual: {y_test_categories[i]}, Predicted: {y_pred_categories[i]}")
```

## Option C: Customer Segments (Clustering)
Group users by `total_spent` + `purchase_count` → 3 clusters (low / mid / high).

### C1. Prepare
```python
# Get customer data for clustering
query = """
SELECT 
    total_spent,
    purchase_count
FROM users 
WHERE total_spent > 0
"""

df = pd.read_sql(query, conn)
print(f"Data for clustering: {df.shape}")
print(df.describe())
```

### C2. Cluster
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Prepare data for clustering
X = df[['total_spent', 'purchase_count']]

# Apply K-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataframe
df['cluster'] = clusters

print("Cluster sizes:")
print(pd.Series(clusters).value_counts().sort_index())
```

### C3. Inspect
```python
# Analyze each cluster
cluster_analysis = df.groupby('cluster').agg({
    'total_spent': ['mean', 'min', 'max'],
    'purchase_count': ['mean', 'min', 'max']
}).round(2)

print("Cluster Analysis:")
print(cluster_analysis)

# Name the clusters based on spending
cluster_names = {0: 'Low Spenders', 1: 'Medium Spenders', 2: 'High Spenders'}
df['cluster_name'] = df['cluster'].map(cluster_names)

print("\nCluster Summary:")
for cluster_id, name in cluster_names.items():
    cluster_data = df[df['cluster'] == cluster_id]
    avg_spent = cluster_data['total_spent'].mean()
    avg_purchases = cluster_data['purchase_count'].mean()
    print(f"{name}: ${avg_spent:.2f} average spent, {avg_purchases:.1f} average purchases")
```

### C4. Plot
```python
# Create scatter plot of clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['total_spent'], df['purchase_count'], c=df['cluster'], cmap='viridis')
plt.xlabel('Total Spent ($)')
plt.ylabel('Purchase Count')
plt.title('Customer Segments')
plt.colorbar(scatter)

# Add cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)

plt.show()
```

## 2. Evaluation Cheat Sheet

### Classification (A / B)
```python
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Success Criteria:
# - Accuracy > 60% (better than random)
# - Model makes logical predictions
# - No major errors in code
```

### Clustering (C)
```python
# Evaluate clustering quality
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X, clusters)
print(f"Average silhouette score: {silhouette_avg:.3f}")

# Success Criteria:
# - Clusters make business sense (clear high/medium/low groups)
# - Silhouette score > 0.3
# - Clear separation in spending patterns
```

## 3. Common Issues

Missing module → install it:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

DB connect fails → check service running & creds:
```python
# Make sure PostgreSQL is running and credentials are correct
conn = psycopg2.connect(
    host="localhost",
    database="ecommerce", 
    user="postgres",
    password="your_password"  # Your actual password
)
```

Tiny dataset → relax filters:
```python
# Make sure you have enough data
print(f"Total rows: {len(df)}")
if len(df) < 100:
    print("Warning: Very small dataset, results may not be reliable")
```

Poor accuracy:
- Collapse rare classes.
- Use fewer, cleaner numeric features.
- Ensure target not heavily imbalanced.
- Try LogisticRegression for baseline.

## 4. Document Results (`MODEL_RESULTS.md` Template)

```markdown
# Machine Learning Model Results

## Model Type
[Classification/Clustering] - [Brief description of what it predicts]

## Data Used
- Dataset size: [X] rows
- Features used: [list your features]
- Target variable: [what you're predicting]

## Model Performance
- Accuracy: [X]%
- [Other metrics as appropriate]

## Key Findings
- [What did your model learn?]
- [Which features were most important?]
- [Any interesting patterns discovered?]

## Business Value
- [How could this model help the business?]
- [What actions could they take based on these predictions?]

## Limitations
- [What are the model's weaknesses?]
- [What would you improve with more time/data?]
```

## 5. Checklist
☐ Install packages  
☐ Query minimal data  
☐ Pick ONE option (A/B/C)  
☐ Train baseline  
☐ Evaluate (metric or silhouette)  
☐ Save code + metrics  
☐ Fill `MODEL_RESULTS.md`  
☐ Test single prediction (A/B) OR cluster summary (C)

