import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- BƯỚC CHUẨN BỊ DỮ LIỆU & DỰ ĐOÁN (TỪ BÀI 1 & 2) ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- BÀI 3: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---

# 1. Tạo ma trận nhầm lẫn từ kết quả dự đoán
cm = confusion_matrix(y_test, y_pred)
print("Ma trận nhầm lẫn:\n", cm)

# 2. Trực quan hóa ma trận bằng Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,  # Hiển thị con số trong từng ô
    fmt="d",  # Định dạng số nguyên (decimal)
    cmap="Blues",  # Bảng màu xanh dương
    xticklabels=data.target_names,
    yticklabels=data.target_names,
)
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.xlabel("Nhãn dự đoán (Predicted)")
plt.ylabel("Nhãn thực tế (Actual)")
plt.show()

# 3. Giải thích các thành phần: TP, TN, FP, FN thông qua comment
# Trong tập dữ liệu breast_cancer:
# - Lớp 0 (Negative - Âm tính): Ác tính (malignant)
# - Lớp 1 (Positive - Dương tính): Lành tính (benign)
#
# Vị trí các phần tử trong ma trận cm (2x2):
# [[TN, FP],
#  [FN, TP]]
#
# - TN (True Negative - Âm tính thật): Nằm ở vị trí cm[0][0].
#   Mẫu thực tế là Ác tính (0) và mô hình dự đoán ĐÚNG là Ác tính (0).
#
# - FP (False Positive - Dương tính giả): Nằm ở vị trí cm[0][1].
#   Mẫu thực tế là Ác tính (0) nhưng mô hình dự đoán SAI thành Lành tính (1).
#
# - FN (False Negative - Âm tính giả): Nằm ở vị trí cm[1][0].
#   Mẫu thực tế là Lành tính (1) nhưng mô hình dự đoán SAI thành Ác tính (0).
#
# - TP (True Positive - Dương tính thật): Nằm ở vị trí cm[1][1].
#   Mẫu thực tế là Lành tính (1) và mô hình dự đoán ĐÚNG là Lành tính (1).