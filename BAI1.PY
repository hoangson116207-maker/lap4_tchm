import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dữ liệu breast_cancer từ sklearn.datasets
data = load_breast_cancer()
X = data.data  # Các đặc trưng đầu vào (30 features)
y = data.target  # Nhãn phân loại (0: Ác tính - Malignant, 1: Lành tính - Benign)

# 2. Chia dữ liệu thành tập Train (80%) và Test (20%)
# stratify=y giúp giữ nguyên tỷ lệ các lớp giữa tập train và test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu (Scaling)
# Lưu ý: Chỉ fit_transform trên tập train, sau đó dùng transform trên tập test để tránh rò rỉ dữ liệu (data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Khởi tạo và huấn luyện mô hình LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Dự đoán và đánh giá kết quả trên tập Test
y_pred = model.predict(X_test_scaled)

print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nMa trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\nBáo cáo phân loại chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=data.target_names))