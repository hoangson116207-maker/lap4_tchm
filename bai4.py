import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dữ liệu digits và Tiền xử lý (Preprocess với StandardScaler)
digits = load_digits()
X = digits.data  # Mỗi ảnh chữ số viết tay 8x8 được trải phẳng thành vector 64 đặc trưng
y = digits.target  # Nhãn gồm 10 lớp từ chữ số 0 đến 9

# Chia tập dữ liệu thành Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Huấn luyện mô hình Logistic Regression (Bài toán phân loại đa lớp - multi-class)
# Từ scikit-learn >= 1.5, LogisticRegression tự động xử lý bài toán đa lớp (multinomial)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 3. Đánh giá mô hình trên tập Test
y_pred = model.predict(X_test_scaled)

print("=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ===")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo phân loại chi tiết (Classification Report):")
print(classification_report(y_test, y_pred))

# 4. In ra 5 mẫu dự đoán sai của mô hình (so sánh nhãn thực tế và nhãn dự đoán)
# Tìm chỉ số (index) của các mẫu mà nhãn dự đoán khác nhãn thực tế
misclassified_indices = np.where(y_pred != y_test)[0]

print(f"Tổng số mẫu dự đoán sai trên tập Test: {len(misclassified_indices)}/{len(y_test)}")
print("\n=== CHI TIẾT 5 MẪU DỰ ĐOÁN SAI ĐẦU TIÊN ===")

# Lấy tối đa 5 mẫu sai đầu tiên
num_samples = min(5, len(misclassified_indices))

for i in range(num_samples):
    idx = misclassified_indices[i]
    print(f"Mẫu thứ {i + 1} (Index trong tập Test: {idx}): Nhãn thực tế = {y_test[idx]} | Nhãn dự đoán = {y_pred[idx]}")

# Trực quan hóa hình ảnh 5 mẫu dự đoán sai này để dễ quan sát
fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
for i in range(num_samples):
    idx = misclassified_indices[i]
    # Reshape vector 64 phần tử từ X_test gốc (chưa scale) về lại ma trận ảnh 8x8
    image = X_test[idx].reshape(8, 8)
    axes[i].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    axes[i].set_title(f"Thực tế: {y_test[idx]}\nDự đoán: {y_pred[idx]}", color="red")
    axes[i].axis("off")

plt.suptitle("Trực quan hóa 5 mẫu chữ số bị mô hình dự đoán sai")
plt.tight_layout()
plt.show()