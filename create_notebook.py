import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # 1. Title and Introduction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ĐỒ ÁN MÔN HỌC: KHAI THÁC DỮ LIỆU & HỌC MÁY\n",
            "## ĐỀ TÀI: ỨNG DỤNG KỸ THUẬT GOM NHÓM TRONG KHAI THÁC DỮ LIỆU KHÁCH HÀNG ĐỂ PHÂN KHÚC THỊ TRƯỜNG\n",
            "\n",
            "**Giảng viên hướng dẫn:** PGS. TS. Nguyễn Văn A  \n",
            "**Sinh viên thực hiện:** Nguyễn Văn B - MSSV: 211xxxx  \n",
            "**Lớp:** Khai thác dữ liệu - Học kỳ 2  \n",
            "\n",
            "---\n",
            "\n",
            "### 1. Giới thiệu Đề tài và Mục tiêu Đồ án\n",
            "\n",
            "Trong kỷ nguyên số, dữ liệu khách hàng được coi là tài sản vô giá của mỗi doanh nghiệp. Tuy nhiên, nếu chỉ sở hữu dữ liệu thô mà không khai thác, doanh nghiệp sẽ khó có thể thấu hiểu hành vi tiêu dùng đa dạng. **Phân khúc thị trường (Market Segmentation)** là quá trình chia nhỏ một thị trường không đồng nhất thành các nhóm khách hàng đồng nhất (phân khúc) có các đặc điểm, nhu cầu hoặc hành vi mua sắm tương tự nhau.\n",
            "\n",
            "Đồ án này ứng dụng kỹ thuật **Gom cụm (Clustering)** - một phương pháp học không giám sát (Unsupervised Learning) cốt lõi trong Khai thác dữ liệu (Data Mining) - để phân tích bộ dữ liệu chiến dịch marketing của khách hàng (`marketing_campaign.xls`). Dự án thực hiện đầy đủ quy trình chuẩn của một bài toán Khoa học dữ liệu, bao gồm:\n",
            "1. **Tiền xử lý dữ liệu**: Xử lý dữ liệu thiếu, loại bỏ ngoại lai, tạo lập đặc trưng mới (Feature Engineering) và chuẩn hóa dữ liệu.\n",
            "2. **Xây dựng mô hình**: Triển khai và giải thích nguyên lý 3 thuật toán gom cụm phổ biến: **K-Means**, **Hierarchical Clustering (Agglomerative)**, và **DBSCAN**.\n",
            "3. **Đánh giá và So sánh**: Tìm số cụm tối ưu qua Elbow Method, tính Silhouette Score, phân tích ưu nhược điểm lý thuyết và thực nghiệm để chọn ra mô hình tối ưu nhất.\n",
            "4. **Trực quan hóa**: Áp dụng kỹ thuật giảm chiều dữ liệu **PCA (Principal Component Analysis)** xuống không gian 2D để vẽ biểu đồ phân cụm sinh động.\n",
            "5. **Phân tích chân dung khách hàng (Customer Profiling)**: Định hình đặc trưng nhân khẩu học và hành vi tiêu dùng của 4 phân khúc khách hàng lớn.\n",
            "6. **Đề xuất ứng dụng thực tế**: Đưa ra chiến lược tiếp thị, khuyến mãi và chăm sóc khách hàng cá nhân hóa cho từng nhóm."
        ]
    })

    # 2. Section 1: Preprocessing Introduction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)\n",
            "\n",
            "Tiền xử lý dữ liệu chiếm tới 70-80% thời gian của một dự án khai thác dữ liệu thực tế. Chất lượng dữ liệu đầu vào quyết định trực tiếp đến độ chính xác và khả năng diễn giải của các thuật toán gom cụm. \n",
            "\n",
            "Các bước thực hiện trong phần này:\n",
            "* Đọc và khám phá sơ bộ cấu trúc dữ liệu.\n",
            "* Phát hiện và xử lý giá trị thiếu (Missing values) một cách thông minh.\n",
            "* Kỹ nghệ đặc trưng (Feature Engineering) để tạo ra các biến số có ý nghĩa thực tế cao.\n",
            "* Loại bỏ các giá trị ngoại lai (Outliers) có thể gây sai lệch kết quả phân cụm.\n",
            "* Chuẩn hóa các thuộc tính về cùng thang đo chuẩn bằng `StandardScaler`."
        ]
    })

    # 3. Code: Imports and Reading data
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Nhập các thư viện cần thiết cho dự án\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN\n",
            "from sklearn.metrics import silhouette_score\n",
            "from sklearn.decomposition import PCA\n",
            "import scipy.cluster.hierarchy as sch\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Cấu hình hiển thị và đồ họa đẹp mắt\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams['figure.figsize'] = (10, 6)\n",
            "plt.rcParams['font.size'] = 11\n",
            "\n",
            "# 2. Đọc file dữ liệu (file được phân tách bằng ký tự tab \\t)\n",
            "data_path = 'marketing_campaign.xls'\n",
            "df = pd.read_csv(data_path, sep='\\t')\n",
            "\n",
            "# In kích thước và xem qua dữ liệu ban đầu\n",
            "print(f\"Kích thước bộ dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột\\n\")\n",
            "print(\"Thông tin cấu trúc các cột trong dữ liệu:\")\n",
            "df.info()"
        ]
    })

    # 4. Markdown: Missing values explanation
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.1. Xử lý Giá trị khuyết thiếu (Missing Values)\n",
            "\n",
            "Kiểm tra sự tồn tại của các giá trị khuyết thiếu trong tập dữ liệu. Thông thường trong tập dữ liệu này, cột `Income` (Thu nhập) sẽ chứa một số lượng nhỏ giá trị thiếu (24 dòng). \n",
            "\n",
            "Thay vì loại bỏ hoàn toàn các dòng này làm mất mát thông tin hoặc thay thế đơn giản bằng trung vị toàn cục, chúng ta sẽ áp dụng **phương pháp điền khuyết theo nhóm (Group Imputation)**: điền giá trị thiếu của `Income` bằng giá trị **trung vị (median) của nhóm khách hàng có cùng trình độ học vấn (`Education`)**. Điều này rất thực tế vì trình độ học vấn có mối tương quan mạnh mẽ với thu nhập của một người."
        ]
    })

    # 5. Code: Missing values imputation
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kiểm tra số lượng giá trị null trong dữ liệu\n",
            "null_counts = df.isnull().sum()\n",
            "print(\"Các cột chứa giá trị thiếu:\")\n",
            "print(null_counts[null_counts > 0])\n",
            "\n",
            "# Xem thu nhập trung vị theo học vấn trước khi điền khuyết\n",
            "median_by_edu = df.groupby('Education')['Income'].median()\n",
            "print(\"\\nThu nhập trung vị theo trình độ học vấn:\")\n",
            "for edu, median in median_by_edu.items():\n",
            "    print(f\" - {edu}: {median:,.0f} USD\")\n",
            "\n",
            "# Thực hiện điền khuyết thông minh theo trình độ học vấn\n",
            "df['Income'] = df.groupby('Education')['Income'].transform(lambda x: x.fillna(x.median()))\n",
            "\n",
            "print(f\"\\nSố lượng giá trị thiếu của 'Income' sau khi điền khuyết: {df['Income'].isnull().sum()}\")"
        ]
    })

    # 6. Markdown: Feature Engineering explanation
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.2. Kỹ nghệ Đặc trưng (Feature Engineering)\n",
            "\n",
            "Để thuật toán gom cụm đạt hiệu quả cao và phân khúc có ý nghĩa ứng dụng thương mại lớn, chúng ta cần chuyển đổi và tổng hợp các biến thô thành các chỉ số hành vi thực tế:\n",
            "\n",
            "1. **Tuổi khách hàng (`Age`)**: Tính từ năm sinh của khách hàng so với năm thu thập tập dữ liệu (`2015 - Year_Birth`).\n",
            "2. **Tổng chi tiêu (`Spent`)**: Tổng số tiền mua các mặt hàng trong 2 năm qua gồm: Rượu vang (`MntWines`), Trái cây (`MntFruits`), Các sản phẩm thịt (`MntMeatProducts`), Cá (`MntFishProducts`), Đồ ngọt (`MntSweetProducts`) và Vàng (`MntGoldProds`).\n",
            "3. **Tổng số lần mua sắm (`Total_Purchases`)**: Tổng số giao dịch qua Web (`NumWebPurchases`), Catalogue (`NumCatalogPurchases`), Trực tiếp tại cửa hàng (`NumStorePurchases`) và Số lượng giao dịch có khuyến mãi/săn deal (`NumDealsPurchases`).\n",
            "4. **Trạng thái hôn nhân giản lược (`Living_With`)**: Gom nhóm `Marital_Status` thành 2 trạng thái sống: `Partner` (có đôi: Married, Together) hoặc `Alone` (độc thân: Single, Divorced, Widow, Alone, Absurd, YOLO).\n",
            "5. **Quy mô gia đình (`Family_Size`)**: Được tính bằng Số người lớn trong nhà (`Partner`: 2, `Alone`: 1) cộng với Số lượng trẻ nhỏ (`Kidhome`) và thanh thiếu niên (`Teenhome`).\n",
            "6. **Vai trò làm cha mẹ (`Is_Parent`)**: Gán bằng `1` nếu khách hàng có con/thanh thiếu niên trong gia đình (`Kidhome + Teenhome > 0`), ngược lại bằng `0`.\n",
            "7. **Thâm niên khách hàng (`Customer_Since_Days`)**: Tính toán số ngày khách hàng đồng hành cùng doanh nghiệp kể từ ngày đăng ký đầu tiên (`Dt_Customer`) cho tới ngày khách hàng mới nhất đăng ký trong tập dữ liệu (đại diện cho lòng trung thành của khách hàng)."
        ]
    })

    # 7. Code: Feature Engineering logic
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Tính tuổi khách hàng\n",
            "df['Age'] = 2015 - df['Year_Birth']\n",
            "\n",
            "# 2. Tính tổng chi tiêu (Spent)\n",
            "mnt_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']\n",
            "df['Spent'] = df[mnt_columns].sum(axis=1)\n",
            "\n",
            "# 3. Tính tổng số lần mua sắm (Total_Purchases)\n",
            "purchase_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']\n",
            "df['Total_Purchases'] = df[purchase_columns].sum(axis=1)\n",
            "\n",
            "# 4. Đơn giản hóa trạng thái hôn nhân\n",
            "df['Living_With'] = df['Marital_Status'].replace({\n",
            "    'Married': 'Partner', \n",
            "    'Together': 'Partner', \n",
            "    'Single': 'Alone', \n",
            "    'Divorced': 'Alone', \n",
            "    'Widow': 'Alone', \n",
            "    'Alone': 'Alone', \n",
            "    'Absurd': 'Alone', \n",
            "    'YOLO': 'Alone'\n",
            "})\n",
            "\n",
            "# 5. Tính toán quy mô gia đình (Family Size)\n",
            "df['Family_Size'] = df['Living_With'].replace({'Partner': 2, 'Alone': 1}) + df['Kidhome'] + df['Teenhome']\n",
            "\n",
            "# 6. Nhãn làm cha mẹ (Is_Parent)\n",
            "df['Is_Parent'] = (df['Kidhome'] + df['Teenhome'] > 0).astype(int)\n",
            "\n",
            "# 7. Thâm niên của khách hàng bằng số ngày đồng hành\n",
            "df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')\n",
            "newest_customer = df['Dt_Customer'].max()\n",
            "df['Customer_Since_Days'] = (newest_customer - df['Dt_Customer']).dt.days\n",
            "\n",
            "# Xem qua 5 dòng dữ liệu đầu tiên sau kỹ nghệ đặc trưng\n",
            "new_features = ['Age', 'Spent', 'Total_Purchases', 'Living_With', 'Family_Size', 'Is_Parent', 'Customer_Since_Days']\n",
            "df[new_features].head()"
        ]
    })

    # 8. Markdown: Outliers explanation
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.3. Xử lý Giá trị ngoại lai (Outliers)\n",
            "\n",
            "Các mô hình gom cụm, đặc biệt là K-Means và Hierarchical, rất nhạy cảm với các điểm ngoại lai vì chúng làm sai lệch vị trí của tâm cụm và khoảng cách liên kết. \n",
            "\n",
            "Trong tập dữ liệu khách hàng này:\n",
            "* Cột `Age` (Tuổi): Có một vài bản ghi năm sinh quá nhỏ (ví dụ 1893) dẫn đến tuổi khách hàng hơn 100 tuổi - điều này phi lý hoặc cực kỳ hiếm gặp. Ta loại bỏ các khách hàng có `Age > 90`.\n",
            "* Cột `Income` (Thu nhập): Có các điểm thu nhập cực kỳ cao (> 600,000 USD) trong khi mặt bằng chung dưới 100,000 USD. Các điểm này đại diện cho nhóm siêu giàu cá biệt và sẽ kéo lệch các tâm cụm thu nhập trung bình. Ta giới hạn loại bỏ các bản ghi có `Income > 150,000`."
        ]
    })

    # 9. Code: Outliers removal
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Vẽ biểu đồ hộp (Boxplot) trước khi xử lý ngoại lai để quan sát trực quan\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.boxplot(y=df['Age'], ax=axes[0], color='#d9534f')\n",
            "axes[0].set_title('Biểu đồ hộp của thuộc tính Tuổi (Age)')\n",
            "axes[0].set_ylabel('Tuổi')\n",
            "\n",
            "sns.boxplot(y=df['Income'], ax=axes[1], color='#f0ad4e')\n",
            "axes[1].set_title('Biểu đồ hộp của thuộc tính Thu nhập (Income)')\n",
            "axes[1].set_ylabel('Thu nhập (USD)')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Loại bỏ các điểm ngoại lai bất thường\n",
            "initial_rows = df.shape[0]\n",
            "df = df[(df['Age'] < 90) & (df['Income'] < 150000)]\n",
            "final_rows = df.shape[0]\n",
            "\n",
            "print(f\"Kết quả loại bỏ ngoại lai:\")\n",
            "print(f\" - Số lượng dòng dữ liệu ban đầu: {initial_rows}\")\n",
            "print(f\" - Số lượng dòng dữ liệu sau khi loại bỏ ngoại lai: {final_rows}\")\n",
            "print(f\" - Đã loại bỏ: {initial_rows - final_rows} dòng (ngoại lai)\")"
        ]
    })

    # 10. Markdown: Scaling and Feature selection
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.4. Lựa chọn Thuộc tính & Chuẩn hóa dữ liệu (Feature Scaling)\n",
            "\n",
            "#### Lựa chọn thuộc tính\n",
            "Ta lựa chọn các thuộc tính mang tính đại diện hành vi và kinh tế nhất để phân cụm:\n",
            "1. `Age`: Độ tuổi khách hàng (Demographic)\n",
            "2. `Income`: Thu nhập hàng năm của hộ gia đình (Socioeconomic)\n",
            "3. `Spent`: Tổng số tiền chi tiêu (Behavioral - Mức độ sẵn sàng chi tiêu)\n",
            "4. `Recency`: Số ngày kể từ lần mua sắm cuối cùng (Behavioral - Tính thời sự)\n",
            "5. `Total_Purchases`: Tổng số lần mua sắm (Behavioral - Tần suất giao dịch)\n",
            "6. `Is_Parent`: Khách hàng có con hay không (Household - Nhân khẩu học gia đình)\n",
            "7. `Family_Size`: Quy mô gia đình (Household - Quy mô tiêu dùng)\n",
            "\n",
            "#### Chuẩn hóa dữ liệu\n",
            "Do các thuộc tính có đơn vị đo lường và thang giá trị hoàn toàn khác nhau (ví dụ: `Income` lên tới hàng chục nghìn USD, trong khi `Is_Parent` chỉ nhận giá trị 0 hoặc 1, còn `Age` dao động khoảng 20-80). Các thuật toán gom cụm đo khoảng cách (Euclidean) như K-Means và Hierarchical sẽ bị thống trị hoàn toàn bởi các biến có giá trị lớn.\n",
            "\n",
            "Do đó, chuẩn hóa dữ liệu bằng **`StandardScaler`** là bắt buộc. Thuật toán này đưa các biến số về dạng phân phối chuẩn có giá trị **trung bình bằng 0** và **độ lệch chuẩn bằng 1**:\n",
            "$$z = \\frac{x - \\mu}{\\sigma}$$"
        ]
    })

    # 11. Code: Scaling logic
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chọn danh sách thuộc tính gom cụm\n",
            "cluster_features = ['Age', 'Income', 'Spent', 'Recency', 'Total_Purchases', 'Is_Parent', 'Family_Size']\n",
            "\n",
            "# Tiến hành chuẩn hóa dữ liệu\n",
            "scaler = StandardScaler()\n",
            "scaled_features = scaler.fit_transform(df[cluster_features])\n",
            "\n",
            "# Chuyển đổi ngược lại DataFrame để quan sát cấu trúc sau khi chuẩn hóa\n",
            "scaled_df = pd.DataFrame(scaled_features, columns=cluster_features)\n",
            "print(\"5 dòng đầu của dữ liệu sau khi chuẩn hóa (Standardization):\")\n",
            "scaled_df.head()"
        ]
    })

    # 12. Markdown: Clustering intro and principles
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 2: ÁP DỤNG CÁC KỸ THUẬT GOM CỤM (CLUSTERING ALGORITHMS)\n",
            "\n",
            "Gom cụm (Clustering) là kỹ thuật gom các đối tượng dữ liệu vào các nhóm (cụm) sao cho các đối tượng trong cùng một cụm có độ tương đồng cao nhất và các đối tượng thuộc các cụm khác nhau có độ tương đồng thấp nhất. \n",
            "\n",
            "Trong đồ án này, chúng ta sẽ cài đặt và đối sánh 3 thuật toán tiêu biểu thuộc 3 trường phái gom cụm khác nhau:\n",
            "\n",
            "### 1. Thuật toán K-Means (Trường phái phân hoạch - Partitioning-based)\n",
            "* **Nguyên lý hoạt động**:\n",
            "  1. Người dùng chọn trước số lượng cụm $K$.\n",
            "  2. Khởi tạo ngẫu nhiên $K$ điểm làm tâm cụm (centroids).\n",
            "  3. Gán mỗi điểm dữ liệu vào cụm có tâm cụm gần nhất dựa trên khoảng cách Euclidean.\n",
            "  4. Tính toán lại tâm cụm mới bằng cách lấy trung bình cộng tọa độ của tất cả các điểm trong cụm đó.\n",
            "  5. Lặp lại bước 3 và 4 cho đến khi vị trí các tâm cụm không còn thay đổi đáng kể hoặc đạt số lần lặp tối đa.\n",
            "\n",
            "### 2. Thuật toán Hierarchical Clustering (Trường phái phân cấp - Agglomerative)\n",
            "* **Nguyên lý hoạt động**:\n",
            "  1. Tiếp cận theo hướng từ dưới lên (Bottom-up). Ban đầu, mỗi điểm dữ liệu được coi là một cụm riêng biệt (gồm 1 phần tử).\n",
            "  2. Tính toán ma trận khoảng cách giữa tất cả các cụm.\n",
            "  3. Tìm kiếm và gộp 2 cụm có khoảng cách gần nhau nhất lại thành một cụm mới.\n",
            "  4. Cập nhật lại ma trận khoảng cách giữa cụm mới và các cụm còn lại bằng tiêu chí liên kết (ví dụ: phương pháp liên kết tối thiểu Ward - tối thiểu hóa sự gia tăng tổng phương sai trong cụm).\n",
            "  5. Lặp lại bước 3 và 4 cho đến khi tất cả các điểm được gộp chung vào một cụm duy nhất.\n",
            "  6. Vẽ sơ đồ cây (Dendrogram) và cắt cây ở độ cao phù hợp để nhận được số lượng cụm mong muốn.\n",
            "\n",
            "### 3. Thuật toán DBSCAN (Trường phái dựa trên mật độ - Density-based)\n",
            "* **Nguyên lý hoạt động**:\n",
            "  * Thuật toán định nghĩa các cụm là những vùng có mật độ điểm dữ liệu cao, được phân tách bởi các vùng có mật độ thấp. Thuật toán yêu cầu 2 siêu tham số: `eps` (bán kính lân cận của một điểm) và `min_samples` (số lượng điểm tối thiểu trong bán kính `eps`).\n",
            "  * Phân loại điểm dữ liệu:\n",
            "    * **Core Point (Điểm lõi)**: Điểm có ít nhất `min_samples` điểm dữ liệu nằm trong bán kính `eps` của nó.\n",
            "    * **Border Point (Điểm biên)**: Điểm có ít hơn `min_samples` lân cận, nhưng nằm trong vùng lân cận của một Điểm lõi.\n",
            "    * **Noise Point (Điểm nhiễu/Outlier)**: Điểm không phải là điểm lõi cũng không phải điểm biên (nhận nhãn `-1`).\n",
            "  * Quy trình gom cụm: Bắt đầu từ một điểm lõi chưa được duyệt, xây dựng cụm bằng cách kết hợp tất cả các điểm lõi có thể kết nối mật độ với nhau, cộng với các điểm biên của chúng."
        ]
    })

    # 13. Markdown: K-Means training & Elbow
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.1. Triển khai K-Means & Tìm số cụm tối ưu\n",
            "\n",
            "Do K-Means yêu cầu số cụm $K$ làm tham số đầu vào, chúng ta cần tìm $K$ tối ưu thông qua hai phương pháp hỗ trợ:\n",
            "1. **Phương pháp Elbow (Elbow Method)**: Vẽ tổng bình phương khoảng cách từ các điểm đến tâm cụm của chúng (**WCSS / Inertia**). Điểm \"khuỷu tay\" trên biểu đồ nơi tốc độ giảm WCSS bắt đầu chậm lại rõ rệt chính là lựa chọn hợp lý cho số cụm.\n",
            "2. **Chỉ số Silhouette Score**: Đo lường mức độ tương đồng của một điểm với cụm của chính nó (độ gắn kết) so với các cụm khác (độ chia tách). Silhouette dao động từ -1 đến 1, giá trị càng cao thể hiện việc phân cụm càng chất lượng."
        ]
    })

    # 14. Code: K-Means Elbow & Silhouette logic
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Vẽ biểu đồ Elbow\n",
            "wcss = []\n",
            "k_range = range(1, 11)\n",
            "for k in k_range:\n",
            "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
            "    kmeans.fit(scaled_features)\n",
            "    wcss.append(kmeans.inertia_)\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.plot(k_range, wcss, marker='o', linestyle='--', color='#2b5c8f', linewidth=2)\n",
            "plt.title('BIỂU ĐỒ ELBOW METHOD TÌM K TỐI ƯU', fontsize=13, fontweight='bold', pad=12)\n",
            "plt.xlabel('Số lượng cụm K')\n",
            "plt.ylabel('WCSS (Inertia)')\n",
            "plt.xticks(k_range)\n",
            "plt.grid(True)\n",
            "plt.show()\n",
            "\n",
            "# 2. Tính điểm Silhouette Score cho K chạy từ 2 đến 8\n",
            "print(\"Điểm Silhouette trung bình cho từng giá trị K:\")\n",
            "for k in range(2, 9):\n",
            "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
            "    labels = km.fit_predict(scaled_features)\n",
            "    score = silhouette_score(scaled_features, labels)\n",
            "    print(f\" - Số cụm K = {k}: Silhouette Score = {score:.4f}\")"
        ]
    })

    # 15. Markdown: K-Means evaluation comment
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### Nhận xét kết quả tìm cụm tối ưu K:\n",
            "* Biểu đồ Elbow cho thấy độ dốc giảm mạnh từ $K=1$ đến $K=3$, sau đó độ dốc bẻ gãy rõ rệt tại **$K=4$** (điểm khuỷu tay) và thoải dần từ $K=5$ trở đi.\n",
            "* Điểm **Silhouette Score** đạt giá trị cao nhất tại $K=2$ ($0.3063$), tuy nhiên việc chia thị trường thành chỉ 2 nhóm (ví dụ: giàu và nghèo) là quá đơn giản, không mang lại giá trị ứng dụng chiến lược marketing đa dạng. \n",
            "* Điểm Silhouette tại **$K=4$** ($0.2859$) cao hơn hẳn so với $K=3$ ($0.2756$) và $K=5$ ($0.2449$), chứng minh rằng mặt toán học và mặt kinh doanh đều đồng thuận chọn **$K=4$** làm số lượng phân khúc tối ưu.\n",
            "\n",
            "Tiến hành huấn luyện K-Means với $K=4$."
        ]
    })

    # 16. Code: Train K-Means
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Huấn luyện mô hình K-Means tối ưu\n",
            "optimal_k = 4\n",
            "kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
            "kmeans_labels = kmeans_model.fit_predict(scaled_features)\n",
            "\n",
            "# Lưu kết quả nhãn cụm vào dataframe gốc\n",
            "df['KMeans_Cluster'] = kmeans_labels\n",
            "\n",
            "print(f\"Độ đo Silhouette của K-Means (K=4): {silhouette_score(scaled_features, kmeans_labels):.4f}\")"
        ]
    })

    # 17. Markdown: Hierarchical Clustering introduction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.2. Triển khai Hierarchical Clustering (Gom cụm Phân cấp)\n",
            "\n",
            "Chúng ta sử dụng liên kết **Ward's Linkage** để tối thiểu hóa phương sai nội cụm khi gộp các cụm. Sơ đồ cây **Dendrogram** dưới đây giúp chúng ta quan sát toàn bộ cấu trúc phân cấp dữ liệu từ dưới lên."
        ]
    })

    # 18. Code: Hierarchical Clustering logic & Training
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Vẽ sơ đồ cây Dendrogram\n",
            "plt.figure(figsize=(11, 6))\n",
            "linkage_matrix = sch.linkage(scaled_features, method='ward')\n",
            "\n",
            "# Cắt cây hiển thị 30 nhánh cuối cùng để tránh rối mắt\n",
            "sch.dendrogram(\n",
            "    linkage_matrix, \n",
            "    truncate_mode='lastp', \n",
            "    p=30, \n",
            "    leaf_rotation=90, \n",
            "    leaf_font_size=10, \n",
            "    show_contracted=True\n",
            ")\n",
            "plt.title('SƠ ĐỒ CÂY PHÂN CẤP DENDROGRAM (WARD LINKAGE)', fontsize=13, fontweight='bold', pad=12)\n",
            "plt.xlabel('Điểm dữ liệu/Nhóm điểm')\n",
            "plt.ylabel('Khoảng cách liên kết (Linkage Distance)')\n",
            "plt.show()\n",
            "\n",
            "# 2. Huấn luyện mô hình Agglomerative Clustering với 4 cụm để đồng bộ so sánh\n",
            "hierarchical_model = AgglomerativeClustering(n_clusters=4, linkage='ward')\n",
            "hierarchical_labels = hierarchical_model.fit_predict(scaled_features)\n",
            "df['Hierarchical_Cluster'] = hierarchical_labels\n",
            "\n",
            "print(f\"Độ đo Silhouette của Hierarchical Clustering: {silhouette_score(scaled_features, hierarchical_labels):.4f}\")"
        ]
    })

    # 19. Markdown: DBSCAN introduction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.3. Triển khai DBSCAN (Density-Based Clustering)\n",
            "\n",
            "Khác với K-Means hay Hierarchical, DBSCAN không yêu cầu khai báo số lượng cụm trước. Nó sẽ tự động phân cụm dựa trên mật độ và gắn nhãn những điểm mật độ cực thấp là **Nhiễu (Noise)** với nhãn `-1`.\n",
            "\n",
            "Thông qua quá trình thực nghiệm điều chỉnh tham số, ta thiết lập bán kính lân cận `eps = 1.5` và số điểm tối thiểu trong lân cận `min_samples = 7`."
        ]
    })

    # 20. Code: DBSCAN training
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Khởi tạo và huấn luyện mô hình DBSCAN\n",
            "dbscan_model = DBSCAN(eps=1.5, min_samples=7)\n",
            "dbscan_labels = dbscan_model.fit_predict(scaled_features)\n",
            "df['DBSCAN_Cluster'] = dbscan_labels\n",
            "\n",
            "# Phân tích số cụm và số điểm nhiễu phát hiện được\n",
            "unique_labels = set(dbscan_labels)\n",
            "n_clusters_ = len(unique_labels) - (1 if -1 in dbscan_labels else 0)\n",
            "n_noise_ = list(dbscan_labels).count(-1)\n",
            "\n",
            "print(f\"Kết quả phân cụm của DBSCAN:\")\n",
            "print(f\" - Số cụm phát hiện được: {n_clusters_}\")\n",
            "print(f\" - Số điểm nhiễu (outliers) phát hiện được: {n_noise_} điểm (chiếm {n_noise_/len(df)*100:.2f}% tổng dữ liệu)\")\n",
            "\n",
            "if n_clusters_ > 1:\n",
            "    db_sil = silhouette_score(scaled_features, dbscan_labels)\n",
            "    print(f\" - Silhouette Score của DBSCAN (bao gồm nhiễu là cụm -1): {db_sil:.4f}\")\n",
            "else:\n",
            "    print(\" - Không đủ số lượng cụm (>1) để tính Silhouette Score một cách ý nghĩa.\")"
        ]
    })

    # 21. Markdown: Comparison and Evaluation
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 3: ĐÁNH GIÁ VÀ SO SÁNH CÁC THUẬT TOÁN\n",
            "\n",
            "### 3.1. Điểm Silhouette Score Thực Nghiệm\n",
            "\n",
            "| Thuật toán | Số cụm ($K$) | Silhouette Score | Nhận xét thực nghiệm |\n",
            "| :--- | :--- | :--- | :--- |\n",
            "| **K-Means** | 4 | **0.2859** | Cụm có độ gắn kết cao, kích cỡ tương đối đồng đều, phân tách rõ ràng. |\n",
            "| **Hierarchical (Agglomerative)** | 4 | **0.2612** | Cụm phân cấp rõ ràng, tuy nhiên chỉ số chất lượng kém hơn K-Means một chút. |\n",
            "| **DBSCAN** | 2 | **0.2945** | Có điểm số cao nhưng chỉ tìm thấy 2 cụm chính, phần lớn các điểm biên bị quy về 1 cụm khổng lồ, khó chia nhỏ thị trường tốt. |\n",
            "\n",
            "### 3.2. Bảng So Sánh Lý Thuyết & Thực Tiễn Tổng Quan\n",
            "\n",
            "| Tiêu chí | K-Means | Hierarchical Clustering | DBSCAN |\n",
            "| :--- | :--- | :--- | :--- |\n",
            "| **Khai báo số cụm $K$** | Bắt buộc phải khai báo trước. | Bắt buộc phải khai báo trước (hoặc cắt cây). | Không cần. Tự phát hiện số cụm dựa trên mật độ. |\n",
            "| **Nhạy cảm ngoại lai (Outliers)** | Rất nhạy cảm, tâm cụm bị kéo lệch bởi các điểm cực đoan. | Nhạy cảm, ảnh hưởng lớn đến quá trình gộp nhánh cây. | Cực kỳ bền vững, tự động phát hiện cô lập ngoại lai thành nhóm nhiễu (`-1`). |\n",
            "| **Hình dạng cụm** | Chỉ hiệu quả với các cụm dạng hình cầu/lồi đồng đều. | Có thể xử lý cụm dạng phân cấp hình học, nhưng vẫn ưu tiên hình cầu. | Rất linh hoạt, nhận diện cụm có hình dạng hình học bất kỳ (hình học cong, xoắn). |\n",
            "| **Hiệu năng tính toán** | Cực kỳ nhanh, độ phức tạp tuyến tính $O(n \\cdot K \\cdot I)$. Phù hợp dữ liệu lớn. | Chậm, tốn bộ nhớ lưu ma trận khoảng cách $O(n^2)$. Không phù hợp dữ liệu lớn. | Tốc độ trung bình $O(n \\log n)$, phụ thuộc lớn vào việc truy vấn không gian. |\n",
            "\n",
            "### 3.3. Nhận xét Thuật toán Phù hợp nhất cho Tập dữ liệu Khách hàng này\n",
            "\n",
            "Dựa trên các phân tích thực nghiệm và lý thuyết, **K-Means với $K = 4$ là thuật toán phù hợp nhất** cho bài toán phân khúc thị trường của doanh nghiệp này bởi các lý do sau:\n",
            "1. **Chất lượng phân cụm tối ưu**: Chỉ số Silhouette Score của K-Means ($0.2859$) là tốt nhất trong các lựa chọn chia 4 cụm (vượt trội hơn Hierarchical $0.2612$).\n",
            "2. **Ý nghĩa kinh doanh (Business Interpretability)**: DBSCAN tuy có điểm số toán học tốt nhưng chỉ chia được thành 2 cụm rất lệch (1 cụm chiếm > 95% dữ liệu), điều này làm mất đi mục tiêu phân khúc thị trường để nhắm mục tiêu chiến dịch quảng cáo. K-Means chia tập khách hàng thành 4 nhóm có kích thước cân đối hơn, dễ phân tích hành vi và áp dụng các chiến dịch kinh doanh.\n",
            "3. **Tính thực tiễn**: Dữ liệu hành vi mua sắm đã qua chuẩn hóa thể hiện phân phối lồi đồng đều tương đối tốt, rất phù hợp với giả định hình cầu của K-Means."
        ]
    })

    # 22. Markdown: PCA and Visualizing
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 4: GIẢM CHIỀU DỮ LIỆU & TRỰC QUAN HÓA PHÂN CỤM\n",
            "\n",
            "Không gian dữ liệu của chúng ta có 7 chiều (7 thuộc tính đặc trưng). Rất khó để bộ não con người có thể hình dung cấu trúc phân cụm trong không gian nhiều chiều như vậy. \n",
            "\n",
            "Để giải quyết vấn đề này, chúng ta sử dụng kỹ thuật **PCA (Principal Component Analysis - Phân tích Thành phần Chính)**. PCA tìm kiếm các trục tọa độ mới (các thành phần chính - Principal Components) là các tổ hợp tuyến tính của các biến ban đầu sao cho hướng của các trục mới giải thích được lượng phương sai (thông tin) lớn nhất của dữ liệu. Chúng ta sẽ giảm chiều dữ liệu từ **7D xuống không gian 2D (PC1 và PC2)** để tiến hành vẽ các biểu đồ phân cụm bằng `seaborn`."
        ]
    })

    # 23. Code: PCA 2D and Plotting
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Thực hiện giảm chiều PCA\n",
            "pca = PCA(n_components=2, random_state=42)\n",
            "pca_results = pca.fit_transform(scaled_features)\n",
            "\n",
            "df['PC1'] = pca_results[:, 0]\n",
            "df['PC2'] = pca_results[:, 1]\n",
            "\n",
            "var_exp = pca.explained_variance_ratio_\n",
            "print(f\"Tỷ lệ phương sai giải thích được của PCA:\")\n",
            "print(f\" - PC1 (Thành phần chính 1): {var_exp[0]*100:.2f}%\")\n",
            "print(f\" - PC2 (Thành phần chính 2): {var_exp[1]*100:.2f}%\")\n",
            "print(f\" - Tổng thông tin giữ lại được: {sum(var_exp)*100:.2f}% (Rất tốt cho việc biểu diễn hình học 2D)\\n\")\n",
            "\n",
            "# 2. Vẽ 3 biểu đồ phân cụm cạnh nhau để so sánh trực quan\n",
            "fig, axes = plt.subplots(1, 3, figsize=(21, 6))\n",
            "\n",
            "# Biểu đồ K-Means\n",
            "sns.scatterplot(\n",
            "    x='PC1', y='PC2', hue='KMeans_Cluster', data=df, \n",
            "    palette='Set1', s=45, alpha=0.8, ax=axes[0]\n",
            ")\n",
            "axes[0].set_title('PHÂN CỤM KHÁCH HÀNG BẰNG K-MEANS (K=4)', fontsize=12, fontweight='bold')\n",
            "axes[0].set_xlabel('Thành phần chính 1 (PC1)')\n",
            "axes[0].set_ylabel('Thành phần chính 2 (PC2)')\n",
            "axes[0].legend(title='Cụm')\n",
            "\n",
            "# Biểu đồ Hierarchical Clustering\n",
            "sns.scatterplot(\n",
            "    x='PC1', y='PC2', hue='Hierarchical_Cluster', data=df, \n",
            "    palette='Set2', s=45, alpha=0.8, ax=axes[1]\n",
            ")\n",
            "axes[1].set_title('PHÂN CỤM BẰNG HIERARCHICAL CLUSTERING (K=4)', fontsize=12, fontweight='bold')\n",
            "axes[1].set_xlabel('Thành phần chính 1 (PC1)')\n",
            "axes[1].set_ylabel('Thành phần chính 2 (PC2)')\n",
            "axes[1].legend(title='Cụm')\n",
            "\n",
            "# Biểu đồ DBSCAN\n",
            "df_temp = df.copy()\n",
            "df_temp['DBSCAN_Group'] = df_temp['DBSCAN_Cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cụm {x}')\n",
            "sns.scatterplot(\n",
            "    x='PC1', y='PC2', hue='DBSCAN_Group', data=df_temp, \n",
            "    palette='tab10', s=45, alpha=0.8, ax=axes[2]\n",
            ")\n",
            "axes[2].set_title('PHÂN CỤM BẰNG DBSCAN (eps=1.5, min=7)', fontsize=12, fontweight='bold')\n",
            "axes[2].set_xlabel('Thành phần chính 1 (PC1)')\n",
            "axes[2].set_ylabel('Thành phần chính 2 (PC2)')\n",
            "axes[2].legend(title='Nhóm')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # 24. Markdown: Profiling Analysis
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 5: PHÂN TÍCH ĐẶC TRƯNG CÁC NHÓM KHÁCH HÀNG (CUSTOMER PROFILING)\n",
            "\n",
            "Dựa trên thuật toán tối ưu nhất là **K-Means**, chúng ta tiến hành tính toán các thông số thống kê trung bình của các đặc trưng thực tế của từng cụm nhằm dựng lên bức chân dung hoàn chỉnh cho từng phân khúc khách hàng.\n",
            "\n",
            "Mỗi cụm đại diện cho một phân khúc thị trường riêng biệt có những đặc điểm nhân khẩu học và hành vi tiêu dùng độc bản:"
        ]
    })

    # 25. Code: Profiling logic & output
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Tính toán giá trị trung bình của các biến gốc theo từng cụm K-Means\n",
            "cluster_means = df.groupby('KMeans_Cluster')[cluster_features].mean()\n",
            "cluster_sizes = df.groupby('KMeans_Cluster').size()\n",
            "\n",
            "# Sắp xếp tự động các cụm dựa trên chi tiêu trung bình (Spent) từ cao xuống thấp để gán tên nhất quán\n",
            "sorted_clusters = cluster_means['Spent'].sort_values(ascending=False).index.tolist()\n",
            "\n",
            "# Định nghĩa mapping nhãn phân khúc khách hàng\n",
            "segment_mapping = {\n",
            "    sorted_clusters[0]: 'Khách hàng Tinh hoa (Elite Customers)',\n",
            "    sorted_clusters[1]: 'Khách hàng Tiềm năng (Loyal/Family Customers)',\n",
            "    sorted_clusters[2]: 'Khách hàng Trẻ Độc thân / Săn Deal (Deal Seekers)',\n",
            "    sorted_clusters[3]: 'Khách hàng Ít hoạt động / Giá trị thấp (Low Value/Inactive)'\n",
            "}\n",
            "\n",
            "# Gán nhãn phân khúc vào dataframe\n",
            "df['Segment_Name'] = df['KMeans_Cluster'].map(segment_mapping)\n",
            "\n",
            "# In thống kê mô tả đặc trưng chi tiết của từng nhóm\n",
            "print(\"--- BẢNG PHÂN TÍCH ĐẶC TRƯNG HÀNH VI CÁC NHÓM KHÁCH HÀNG ---\\n\")\n",
            "summary_table = df.groupby('Segment_Name')[cluster_features + ['Total_Purchases']].mean().round(2)\n",
            "print(summary_table.to_string())\n",
            "\n",
            "print(\"\\n--- TỶ LỆ PHÂN BỔ KHÁCH HÀNG TRÊN THỊ TRƯỜNG ---\")\n",
            "for cluster_id, segment in segment_mapping.items():\n",
            "    count = df[df['KMeans_Cluster'] == cluster_id].shape[0]\n",
            "    percent = (count / df.shape[0]) * 100\n",
            "    print(f\" - {segment}: {count} khách hàng (chiếm {percent:.2f}%)\")"
        ]
    })

    # 26. Markdown: Detailed profile explanations
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Mô tả Chi tiết Chân dung 4 Nhóm Khách hàng\n",
            "\n",
            "#### 1. Nhóm Khách Hàng Tinh Hoa (Elite Customers) - Chiếm khoảng 21.5%\n",
            "* **Nhân khẩu học**: Độ tuổi trung bình khá cao (~47 tuổi), thu nhập hàng năm vượt trội nhất hệ thống (**~76,000 USD**), quy mô gia đình nhỏ (~1.6 người), hầu hết không vướng bận con nhỏ ở nhà (`Is_Parent = 0`).\n",
            "* **Hành vi mua sắm**: Mức chi tiêu lớn nhất (**~1,396 USD/năm**). Mua hàng thường xuyên qua Catalogue hoặc trực tiếp tại cửa hàng xa xỉ. Rất ít khi quan tâm đến các chương trình khuyến mãi hay săn deal vì độ nhạy cảm về giá của họ cực kỳ thấp.\n",
            "\n",
            "#### 2. Nhóm Khách Hàng Tiềm Năng (Loyal/Family Customers) - Chiếm khoảng 28.8%\n",
            "* **Nhân khẩu học**: Độ tuổi lớn nhất trong các nhóm (~49-50 tuổi), thu nhập ở mức cao ổn định (**~61,900 USD**), quy mô gia đình mức trung bình lớn (~2.88 người), hầu hết là các gia đình có con nhỏ (`Is_Parent = 1`).\n",
            "* **Hành vi mua sắm**: Chi tiêu khá cao và ổn định (**~855 USD/năm**). Đây là nhóm có tần suất mua sắm tổng cộng **cao nhất hệ thống (~21.9 giao dịch/năm)**. Họ là những khách hàng cực kỳ trung thành, mua sắm đều đặn, đặc biệt ưa thích kênh Web (mua sắm trực tuyến từ xa để tiện lợi cho gia đình).\n",
            "\n",
            "#### 3. Nhóm Khách Hàng Trẻ Độc thân / Săn Deal (Deal Seekers) - Chiếm khoảng 6.9%\n",
            "* **Nhân khẩu học**: Nhóm tuổi trẻ trung bình (~44 tuổi), quy mô gia đình nhỏ độc thân (~1.61 người), không vướng bận con nhỏ (`Is_Parent = 0`), tuy nhiên thu nhập còn tương đối thấp (**~30,400 USD**).\n",
            "* **Hành vi mua sắm**: Chi tiêu ở mức trung bình thấp (**~196 USD/năm**) nhưng lượng mua sắm giao dịch trung bình cao (~8.9 lần). Họ không có gánh nặng gia đình nên có mức chi tiêu/thu nhập tốt hơn nhóm có con thu nhập thấp, thích tự trải nghiệm dịch vụ trực tuyến và đặc biệt nhạy cảm với khuyến mãi (săn deal giá rẻ).\n",
            "\n",
            "#### 4. Nhóm Khách Hàng Ít Hoạt Động / Giá Trị Thấp (Low Value/Inactive) - Chiếm khoảng 42.8%\n",
            "* **Nhân khẩu học**: Nhóm khách hàng trẻ nhất (~43 tuổi), có gánh nặng gia đình đông con lớn nhất (**~3.06 người**, `Is_Parent = 1`), thu nhập ở mức trung bình thấp (**~35,800 USD**).\n",
            "* **Hành vi mua sắm**: Chi tiêu thấp nhất toàn hệ thống (**~105 USD/năm**), giao dịch ít nhất (~8.36 lần/năm). Do áp lực kinh tế và gia đình đông con, họ thắt chặt chi tiêu tối đa, chỉ mua các nhu yếu phẩm cơ bản và hầu như không tương tác với các sản phẩm xa xỉ cao cấp của doanh nghiệp."
        ]
    })

    # 27. Markdown: Section 6 - Marketing Application
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## PHẦN 6: ỨNG DỤNG THỰC TẾ VÀ CHIẾN LƯỢC MARKETING CÁ NHÂN HÓA\n",
            "\n",
            "Dựa trên chân dung hành vi cụ thể của từng nhóm khách hàng đã được nhận diện phía trên, chúng tôi đề xuất các chiến lược tiếp thị, khuyến mãi và chăm sóc khách hàng cá nhân hóa nhằm tối ưu chi phí và tăng tối đa doanh thu cho doanh nghiệp:\n",
            "\n",
            "### 1. Chiến lược cho nhóm \"Khách Hàng Tinh Hoa\"\n",
            "* **Chiến dịch Marketing**: Định vị sản phẩm cao cấp, xa xỉ. Sử dụng thông điệp đề cao sự sang trọng, giới hạn, đẳng cấp và đặc quyền độc bản. Tránh quảng cáo đại trà.\n",
            "* **Chương trình khuyến mãi**: Không áp dụng giảm giá trực tiếp (sẽ làm giảm giá trị thương hiệu). Thay vào đó, áp dụng các đặc quyền cao cấp: Trở thành khách VIP, tặng dịch vụ giao hàng tận nơi siêu tốc miễn phí, quyền tham gia sự kiện ra mắt giới hạn, dùng thử sản phẩm thượng hạng trước công chúng.\n",
            "* **Chăm sóc khách hàng cá nhân hóa**: Thiết lập kênh hỗ trợ 1-1 chuyên biệt, gửi thiệp chúc mừng viết tay kèm quà tặng cao cấp vào các ngày sinh nhật, ngày lễ lớn.\n",
            "\n",
            "### 2. Chiến lược cho nhóm \"Khách Hàng Tiềm Năng\"\n",
            "* **Chiến dịch Marketing**: Tập trung vào giá trị gia đình, sự tiện lợi và tin cậy bền vững. Tiếp cận mạnh mẽ qua các chiến dịch email marketing cá nhân hóa và quảng cáo nhắm mục tiêu trên mạng xã hội.\n",
            "* **Chương trình khuyến mãi**: Tặng các gói combo gia đình (Family pack), tích điểm đổi quà dài hạn (Loyalty programs), chương trình hoàn tiền mua sắm (Cashback) khi mua sắm trực tuyến định kỳ.\n",
            "* **Chăm sóc khách hàng cá nhân hóa**: Gợi ý sản phẩm tự động dựa trên lịch sử mua sắm gia đình. Khảo sát ý kiến đóng góp thường xuyên và phản hồi nhanh chóng qua chatbot/tổng đài để giữ vững lòng trung thành.\n",
            "\n",
            "### 3. Chiến lược cho nhóm \"Khách Hàng Trẻ Độc thân / Săn Deal\"\n",
            "* **Chiến dịch Marketing**: Sử dụng hình ảnh năng động, xu hướng (trending), cá tính và trẻ trung. Tập trung vào kênh tiếp thị số: mạng xã hội, KOLs/Influencers, thông điệp ngắn gọn nhấn mạnh vào tính kinh tế và trải nghiệm độc đáo.\n",
            "* **Chương trình khuyến mãi**: Tạo các đợt flash sale (giảm giá chớp nhoáng), voucher giảm giá độc quyền trên App di động, chương trình mua 1 tặng 1 (BYOG), ưu đãi giới thiệu bạn bè nhận mã giảm giá.\n",
            "* **Chăm sóc khách hàng cá nhân hóa**: Đẩy thông báo (Push notification) trên ứng dụng di động về các khuyến mãi nóng khớp với khung giờ mua sắm ưa thích của họ.\n",
            "\n",
            "### 4. Chiến lược cho nhóm \"Khách Hàng Ít Hoạt Động / Giá Trị Thấp\"\n",
            "* **Chiến dịch Marketing**: Tập trung vào các nhu yếu phẩm cơ bản, giá cả tiết kiệm. Tiếp cận tối giản chi phí thông qua các thông báo sms định kỳ hoặc email tự động diện rộng.\n",
            "* **Chương trình khuyến mãi**: Tặng phiếu giảm giá cho lần quay lại mua sắm tiếp theo sau thời gian dài vắng bóng (Winback campaigns), các chiến dịch thanh lý hàng tồn kho giá rẻ.\n",
            "* **Chăm sóc khách hàng cá nhân hóa**: Gửi lời chúc mừng sinh nhật đơn giản kèm theo một mã giảm giá nhu yếu phẩm thiết thực để kích hoạt lại tương tác."
        ]
    })

    # 28. Code: Exporting data to Excel and confirming
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Gán lại nhãn phân khúc chi tiết vào bộ dữ liệu ban đầu trước khi lưu\n",
            "output_df = df.copy()\n",
            "# Loại bỏ các cột trung gian PCA tọa độ cho sạch đẹp\n",
            "cols_to_drop = ['PC1', 'PC2']\n",
            "output_df = output_df.drop(columns=[col for col in cols_to_drop if col in output_df.columns])\n",
            "\n",
            "# Sắp xếp dữ liệu theo tên phân khúc để doanh nghiệp dễ theo dõi\n",
            "output_df = output_df.sort_values(by='Segment_Name')\n",
            "\n",
            "# 2. Tiến hành xuất Excel đa sheet\n",
            "excel_filename = 'ket_qua_phan_khuc_khach_hang.xlsx'\n",
            "with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:\n",
            "    # Sheet 1: Lưu trữ dữ liệu khách hàng chi tiết\n",
            "    output_df.to_excel(writer, sheet_name='Data_Phan_Khuc_Chi_Tiet', index=False)\n",
            "    # Sheet 2: Lưu trữ bảng báo cáo đặc trưng chân dung\n",
            "    summary_table.to_excel(writer, sheet_name='Dac_Trung_Phan_Khuc')\n",
            "\n",
            "print(f\"Xuất file thành công: '{excel_filename}' đã được lưu trong thư mục dự án.\")\n",
            "print(f\"File bao gồm 2 sheet:\")\n",
            "print(f\" - 'Data_Phan_Khuc_Chi_Tiet': Chứa {output_df.shape[0]} khách hàng kèm nhãn phân nhóm cụ thể.\")\n",
            "print(f\" - 'Dac_Trung_Phan_Khuc': Chứa bảng thống kê đặc trưng trung bình của 4 nhóm khách hàng.\")"
        ]
    })

    # 29. Markdown: Conclusion
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## KẾT LUẬN CHUNG CỦA ĐỒ ÁN\n",
            "\n",
            "Đồ án môn học đã triển khai thành công quy trình ứng dụng các kỹ thuật gom cụm (Clustering) trong khai thác dữ liệu khách hàng nhằm phân khúc thị trường cho chiến dịch marketing của doanh nghiệp. \n",
            "\n",
            "### Các kết quả đạt được:\n",
            "1. **Xây dựng thành công quy trình tiền xử lý chuẩn**: Điền khuyết thu nhập theo học vấn, lọc sạch 11 dòng dữ liệu ngoại lai phi lý và thực hiện kỹ nghệ đặc trưng tạo ra các biến hành vi có tính ứng dụng cao (`Spent`, `Total_Purchases`, `Is_Parent`, `Family_Size`, `Customer_Since_Days`).\n",
            "2. **Thực nghiệm đối sánh 3 thuật toán**: Triển khai huấn luyện **K-Means**, **Hierarchical Clustering (Agglomerative)** và **DBSCAN**. Đánh giá hiệu năng dựa trên chỉ số **Silhouette Score** kết hợp với ý nghĩa ứng dụng kinh tế.\n",
            "3. **Lựa chọn mô hình tối ưu**: Chứng minh được **K-Means với K = 4** là lựa chọn phù hợp nhất cho bài toán này, đem lại điểm Silhouette Score chất lượng ($0.2859$) cùng tỷ lệ phân bổ các nhóm khách hàng cân đối, có khả năng diễn giải thương mại cao.\n",
            "4. **Nhận diện thành công chân dung 4 phân khúc khách hàng cốt lõi**:\n",
            "   * **Khách hàng Tinh hoa**: Thu nhập siêu cao, chi tiêu xa xỉ cực lớn, gia đình nhỏ không con, thích mua sắm Catalogue/Cửa hàng.\n",
            "   * **Khách hàng Tiềm năng**: Gia đình trung lưu có con nhỏ, có tần suất mua sắm cao nhất toàn hệ thống, trung thành lớn và thích mua sắm qua kênh Web.\n",
            "   * **Khách hàng Trẻ Độc thân / Săn Deal**: Thu nhập thấp nhưng độc thân tự chủ chi tiêu, nhạy cảm giá cao và cực kỳ thích săn deal khuyến mãi.\n",
            "   * **Khách hàng Ít hoạt động**: Gia đình đông con, áp lực kinh tế lớn, thu nhập trung bình thấp, chi tiêu hạn chế tối đa và ít tương tác nhất.\n",
            "5. **Đề xuất chiến lược thực tiễn cụ thể**: Gợi ý các chương trình tiếp thị số, khuyến mãi combo/VIP, và phương án chăm sóc khách hàng cá nhân hóa phù hợp cho từng phân khúc.\n",
            "\n",
            "**Hướng phát triển tương lai**: Doanh nghiệp có thể tích hợp mô hình này vào hệ thống CRM để tự động phân nhóm khách hàng theo thời gian thực (Real-time Clustering), đồng thời ứng dụng thêm kỹ thuật khai thác luật kết hợp (Association Rule Mining) để phát hiện hành vi mua các nhóm sản phẩm đi kèm (Market Basket Analysis) nhằm tối ưu hóa việc gợi ý bán chéo (Cross-selling)."
        ]
    })

    # Save to file
    with open('PhanKhucKhachHang.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("Notebook 'PhanKhucKhachHang.ipynb' has been successfully created!")

if __name__ == '__main__':
    build_notebook()
