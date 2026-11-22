from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)

# 1. Sử dụng mô hình đã huấn luyện ở Bài 1 để dự đoán trên tập Test (đã chuẩn hóa)
y_pred = model.predict(X_test_scaled)

# 2. Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# In ra các chỉ số
print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"1. Accuracy Score : {accuracy:.4f}")
print(f"2. Precision Score: {precision:.4f}")
print(f"3. Recall Score   : {recall:.4f}")

# 3. In báo cáo phân loại chi tiết bằng classification_report
print("\n=== BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))