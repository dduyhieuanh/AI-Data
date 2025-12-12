# 🤖 Tóm Tắt Nội Dung Tìm Hiểu Tuần 2: Supervised vs. Unsupervised Learning

Tài liệu này cung cấp cái nhìn tổng quan về hai phương pháp học máy cơ bản: Học có Giám sát (Supervised Learning) và Học không Giám sát (Unsupervised Learning), cùng với các bài toán và thuật toán tiêu biểu.

---

## 1. Học có Giám sát (Supervised Learning)

Học có giám sát là một phương pháp trong Machine Learning, trong đó mô hình được huấn luyện bằng cách sử dụng các tập dữ liệu đã được gán nhãn. Thuật toán sẽ học cách nhận diện các mẫu và mối quan hệ giữa dữ liệu đầu vào và đầu ra, từ đó có thể dự đoán chính xác kết quả khi gặp các dữ liệu mới trong thực tế.

### 📝 Quy trình cơ bản

1.  **Chuẩn bị dữ liệu huấn luyện có gán nhãn**: Tạo ra một tập dữ liệu mẫu, trong đó mỗi mục đều được dán nhãn một cách rõ ràng.
2.  **Tiền xử lý dữ liệu (Data preprocessing)**: Dữ liệu cần phải được xử lý để loại bỏ các dữ liệu không cần thiết hoặc sai lệch.
3.  **Chia dữ liệu**: Chia thành tập Train (huấn luyện), tập Validation (tinh chỉnh tham số) và tập Test (đánh giá hiệu năng cuối cùng).
4.  **Huấn luyện mô hình**: Mô hình sẽ tìm ra quy tắc chung để phân biệt các loại dữ liệu khác nhau.
5.  **Đánh giá mô hình**: Mô hình được đánh giá bằng một tập dữ liệu chưa từng thấy. Kỹ thuật phổ biến là xác thực chéo (cross-validation) để đảm bảo mô hình làm tốt trên dữ liệu mới.
6.  **Tối ưu hoá mô hình**: Việc dự đoán càng ngày càng chính xác hơn.
7.  **Triển khai và giám sát**: Mô hình bắt đầu được sử dụng để trả về kết quả dự đoán cho người dùng.

### 📌 Các bài toán chính

Học có Giám sát thường được phân thành hai loại khác nhau là Phân loại (Classification) và Hồi quy (Regression).

| Đặc điểm | Classification (Phân loại) | Regression (Hồi quy) |
| :--- | :--- | :--- |
| **Câu hỏi cốt lõi** | "Cái này thuộc nhóm nào?" (Which one?) | "Giá trị là bao nhiêu?" (How much?) |
| **Dạng đầu ra** | Rời rạc (Discrete). Là các nhãn (labels) hoặc danh mục (categories). | Liên tục (Continuous). Là các con số thực (numbers). |
| **Mục tiêu hình học** | Tìm một đường ranh giới (Decision Boundary) để chia tách các điểm dữ liệu thành các nhóm riêng biệt. | Tìm một đường xu hướng (Best Fit Line) đi qua gần các điểm dữ liệu nhất có thể. |
| **Ví dụ** | Phân loại Email là spam hay không, hoặc "bức ảnh trên là chó hay mèo". | Dự đoán giá nhà dựa theo địa hình, kinh nghiệm, hoặc nhiệt độ ngày mai. |

---

## 2. Học không Giám sát (Unsupervised Learning)

Học không giám sát là phương pháp dùng thuật toán Machine Learning để phân tích và phân cụm dữ liệu chưa gán nhãn, phát hiện mẫu ẩn hoặc nhóm dữ liệu mà không cần con người can thiệp. Unsupervised Learning tự suy luận và sắp xếp các dữ liệu theo quy luật.

### ✨ Mục tiêu và Ứng dụng

* **Mục tiêu**: Tìm ra các mẫu ẩn và nhóm dữ liệu tương tự.
* **Ứng dụng**:
    * **Phân khúc khách hàng (Customer Segmentation)**: Chia khách hàng thành các nhóm dựa trên hành vi mua sắm hoặc sở thích.
    * **Gợi ý sản phẩm (Recommendation Systems)**: Đề xuất các sản phẩm hoặc nội dung mà người dùng có thể quan tâm.
    * **Phát hiện gian lận (Fraud Detection)**: Xác định các giao dịch hoặc hoạt động đáng ngờ.

### 📌 Các bài toán chính

| Đặc điểm | Clustering (Phân cụm) | Dimensionality Reduction (Giảm chiều) |
| :--- | :--- | :--- |
| **Mục tiêu chính** | Tìm ra các nhóm (groups) dữ liệu có đặc điểm tương đồng nhau. | Tìm ra các đặc trưng quan trọng nhất để biểu diễn dữ liệu gọn nhẹ hơn. |
| **Câu hỏi giải quyết** | "Những điểm dữ liệu nào giống nhau?" (Who is like whom?) | "Những thông tin nào là thừa thãi?" (What is redundant?) |
| **Tác động lên dữ liệu** | Giữ nguyên số chiều, nhưng gán thêm nhãn nhóm cho từng điểm dữ liệu. | Giữ nguyên số điểm dữ liệu, nhưng giảm số lượng biến (chiều) của mỗi điểm. |
| **Kết quả đầu ra** | Một nhãn nhóm (Cluster ID) cho mỗi mẫu (VD: Khách hàng A thuộc nhóm VIP). | Một tập hợp các đặc trưng mới ít hơn (VD: Từ 100 cột giảm còn 3 cột). |
| **Thuật toán tiêu biểu** | K-Means, DBSCAN, Hierarchical Clustering. | PCA, t-SNE, Autoencoders. |

---

## 3. Các thuật toán tiêu biểu

### Thuật toán Supervised Learning

* **Linear Regression**: Tìm một đường thẳng phù hợp nhất (Best Fit Line) để dự đoán giá trị đầu ra dựa trên đầu vào.
* **Logistic Regression**: Sử dụng hàm Sigmoid để ánh xạ đầu ra về xác suất, từ đó phân loại.
* **Decision Tree**: Xây dựng một cấu trúc cây bằng cách chia dữ liệu thành các nhánh dựa trên thuộc tính tốt nhất.
* **Random Forest**: Là một tập hợp của nhiều Decision Tree, huấn luyện từng cây trên các tập con dữ liệu ngẫu nhiên.
* **Support Vector Machine (SVM)**: Tìm ra một siêu phẳng (hyperplane) tối ưu để phân tách các điểm dữ liệu thuộc hai lớp khác nhau, tạo ra khoảng cách lớn nhất (margin) giữa hai lớp.
* **K-Nearest Neighbors (KNN)**: Phân loại một điểm dữ liệu mới bằng cách tìm $K$ điểm gần nhất trong tập huấn luyện và xác định nhãn dựa trên nhãn xuất hiện nhiều nhất.

### Thuật toán Unsupervised Learning

* **K-Means**: Chia dữ liệu thành $K$ cụm, cố gắng giảm thiểu tổng bình phương khoảng cách từ mỗi điểm tới trọng tâm của cụm gần nhất.
* **Hierarchical Clustering**: Xây dựng một cây phân cấp để nhóm các điểm dữ liệu lại với nhau theo khoảng cách.
* **DBSCAN**: Một cụm bao gồm một vùng điểm dày đặc, được phân tách với các cụm khác bằng các vùng có mật độ thấp hơn.
* **Principal Component Analysis (PCA)**: Tìm ra các Thành phần chính (các trục) mà dữ liệu biến thiên mạnh nhất để nén thông tin.
* **t-SNE**: Tập trung vào việc giữ lại mối quan hệ lân cận cục bộ của dữ liệu khi chiếu dữ liệu từ không gian cao chiều xuống 2D hoặc 3D.
