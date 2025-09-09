# train_customer_type.py
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

conn = psycopg2.connect(
    host="localhost",
    database="ecommerce",     # matches database_setup/import scripts:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    user="ecommerce_user",    # matches role in step 1C:contentReference[oaicite:8]{index=8}
    password="temp_pw"        # matches env/role in step 1C:contentReference[oaicite:9]{index=9}
)

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

X = df[['total_spent', 'purchase_count', 'is_ios', 'is_android']]
y = df['user_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nDetailed Results:")
print(classification_report(y_test, y_pred))

import pandas as pd
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Single prediction example
new_customer = [[150.50, 5, 1, 0]]
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)
print(f"\nPredicted customer type: {prediction[0]}")
print(f"Confidence: {probability[0].max():.2%}")
