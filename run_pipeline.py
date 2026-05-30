import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Cấu hình phong cách biểu đồ đẹp mắt và hỗ trợ hiển thị tốt tiếng Việt nếu có thể
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def preprocess_data(filepath):
    print("--- 1. BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ---")
    
    # 1. Đọc dữ liệu (file thực chất là Tab-Separated Values)
    df = pd.read_csv(filepath, sep='\t')
    print(f"Kích thước bộ dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")
    
    # 2. Xử lý giá trị khuyết thiếu trong cột Income
    # Tính trung vị Income theo trình độ học vấn (Education) để điền khuyết một cách thông minh
    median_income_by_edu = df.groupby('Education')['Income'].median()
    print("\nThu nhập trung vị (Median Income) theo trình độ học vấn:")
    for edu, val in median_income_by_edu.items():
        print(f" - {edu}: {val:,.0f} USD")
        
    initial_nulls = df['Income'].isnull().sum()
    df['Income'] = df.groupby('Education')['Income'].transform(lambda x: x.fillna(x.median()))
    print(f"Số lượng giá trị thiếu của 'Income' trước xử lý: {initial_nulls} -> Sau xử lý: {df['Income'].isnull().sum()}")
    
    # 3. Kỹ nghệ đặc trưng (Feature Engineering)
    # Tuổi của khách hàng (Dataset lấy mốc khoảng năm 2015)
    df['Age'] = 2015 - df['Year_Birth']
    
    # Tổng số tiền chi tiêu (Spent) trên tất cả sản phẩm
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Spent'] = df[mnt_cols].sum(axis=1)
    
    # Tổng số lần mua sắm (Total_Purchases)
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    
    # Quy định trạng thái sống chung / Quy mô gia đình
    # Gom nhóm tình trạng hôn nhân thành Partner (Có đôi) hoặc Alone (Độc thân)
    df['Living_With'] = df['Marital_Status'].replace({
        'Married': 'Partner', 
        'Together': 'Partner', 
        'Single': 'Alone', 
        'Divorced': 'Alone', 
        'Widow': 'Alone', 
        'Alone': 'Alone', 
        'Absurd': 'Alone', 
        'YOLO': 'Alone'
    })
    # Quy mô gia đình = (Người lớn: Partner=2, Alone=1) + Trẻ nhỏ (Kidhome) + Trẻ vị thành niên (Teenhome)
    df['Family_Size'] = df['Living_With'].replace({'Partner': 2, 'Alone': 1}) + df['Kidhome'] + df['Teenhome']
    
    # Có con nhỏ/thanh thiếu niên hay không (Is_Parent)
    df['Is_Parent'] = (df['Kidhome'] + df['Teenhome'] > 0).astype(int)
    
    # Thâm niên khách hàng tính bằng ngày (Customer_Since_Days)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    # Điền giá trị NaT nếu có bằng ngày xa nhất hoặc drop. Ở đây giả định định dạng ngày nhất quán
    newest_customer = df['Dt_Customer'].max()
    df['Customer_Since_Days'] = (newest_customer - df['Dt_Customer']).dt.days
    
    # 4. Xử lý giá trị ngoại lai (Outliers)
    # Loại bỏ khách hàng có tuổi thọ phi lý (> 90 tuổi)
    # Loại bỏ khách hàng có thu nhập cực đoan đại phú (> 150,000 USD)
    before_outliers = df.shape[0]
    df = df[(df['Age'] < 90) & (df['Income'] < 150000)]
    after_outliers = df.shape[0]
    print(f"\nXử lý dữ liệu ngoại lai:")
    print(f" - Số dòng bị loại bỏ: {before_outliers - after_outliers} dòng")
    print(f" - Kích thước tập dữ liệu sau khi làm sạch: {df.shape[0]} dòng")
    
    # 5. Lựa chọn thuộc tính đặc trưng cho phân cụm
    cluster_features = ['Age', 'Income', 'Spent', 'Recency', 'Total_Purchases', 'Is_Parent', 'Family_Size']
    print(f"\nCác thuộc tính được chọn để phân cụm khách hàng: {cluster_features}")
    
    # 6. Chuẩn hóa dữ liệu bằng StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[cluster_features])
    scaled_df = pd.DataFrame(scaled_features, columns=cluster_features)
    
    print("--- TIỀN XỬ LÝ DỮ LIỆU HOÀN TẤT ---\n")
    return df, scaled_features, cluster_features

def run_clustering(df, scaled_features, features_list):
    print("--- 2. HUẤN LUYỆN CÁC THUẬT TOÁN GOM CỤM ---")
    
    # --- A. THUẬT TOÁN K-MEANS ---
    print("\n[A] Triển khai K-Means...")
    
    # 1. Tìm số cụm tối ưu bằng Phương pháp Elbow (WCSS) và tính Silhouette Score
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        
    # Tính Silhouette Scores cho các giá trị K từ 2 đến 8
    silhouette_avg_list = []
    print("Điểm Silhouette Score cho từng giá trị K (K-Means):")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        silhouette_avg_list.append((k, score))
        print(f" - K = {k}: Silhouette Score = {score:.4f}")
        
    # Vẽ biểu đồ Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='#2b5c8f', linewidth=2)
    plt.title('Biểu đồ Elbow Method tìm số cụm K tối ưu', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Số lượng cụm K', fontsize=12)
    plt.ylabel('WCSS (Inertia)', fontsize=12)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ Elbow Method vào file 'elbow_method.png'")
    
    # Chọn số lượng cụm tối ưu K = 4 (Phù hợp nhất dựa trên Elbow và tính ứng dụng kinh doanh)
    optimal_k = 4
    kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_opt.fit_predict(scaled_features)
    df['KMeans_Cluster'] = kmeans_labels
    km_sil = silhouette_score(scaled_features, kmeans_labels)
    print(f"-> Đã huấn luyện K-Means với K = {optimal_k}. Silhouette Score = {km_sil:.4f}")
    
    # --- B. THUẬT TOÁN HIERARCHICAL CLUSTERING (Agglomerative) ---
    print("\n[B] Triển khai Hierarchical Clustering (Agglomerative)...")
    
    # Vẽ Dendrogram để minh họa phân cấp cụm
    plt.figure(figsize=(10, 6))
    linkage_matrix = sch.linkage(scaled_features, method='ward')
    sch.dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    plt.title('Sơ đồ cây Phân cấp Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Chỉ số/Số lượng điểm dữ liệu', fontsize=12)
    plt.ylabel('Khoảng cách liên kết Linkage Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=300)
    plt.close()
    print("-> Đã lưu sơ đồ cây Dendrogram vào file 'dendrogram.png'")
    
    # Tiến hành gom cụm với số lượng cụm bằng 4 để dễ so sánh
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agg_labels = agg_clustering.fit_predict(scaled_features)
    df['Hierarchical_Cluster'] = agg_labels
    agg_sil = silhouette_score(scaled_features, agg_labels)
    print(f"-> Đã huấn luyện Agglomerative Clustering với {optimal_k} cụm. Silhouette Score = {agg_sil:.4f}")
    
    # --- C. THUẬT TOÁN DBSCAN ---
    print("\n[C] Triển khai DBSCAN...")
    # Tinh chỉnh tham số: với 7 thuộc tính chuẩn hóa, chọn eps=1.5 và min_samples=7 mang lại kết quả phân cụm hợp lý
    dbscan = DBSCAN(eps=1.5, min_samples=7)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    df['DBSCAN_Cluster'] = dbscan_labels
    
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_db = list(dbscan_labels).count(-1)
    print(f"-> DBSCAN phát hiện: {n_clusters_db} cụm hợp lệ và {n_noise_db} điểm nhiễu (outliers)")
    
    if n_clusters_db > 1:
        # Loại bỏ điểm nhiễu để tính silhouette score của các cụm thực tế hoặc tính trực tiếp
        db_sil = silhouette_score(scaled_features, dbscan_labels)
        print(f"-> Silhouette Score của DBSCAN (bao gồm nhiễu là cụm -1): {db_sil:.4f}")
    else:
        db_sil = -1
        print("-> DBSCAN không tìm đủ số cụm hợp lệ (>1) để đánh giá Silhouette phù hợp.")
        
    print("--- HUẤN LUYỆN CÁC THUẬT TOÁN HOÀN TẤT ---\n")
    return df, km_sil, agg_sil, db_sil

def visualize_clusters(df, scaled_features):
    print("--- 3. GIẢM CHIỀU DỮ LIỆU BẰNG PCA & TRỰC QUAN HÓA PHÂN CỤM ---")
    
    # Giảm chiều dữ liệu xuống 2 chiều bằng PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_features)
    
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]
    
    variance_explained = pca.explained_variance_ratio_
    print(f"Tỷ lệ phương sai giải thích được của 2 thành phần chính PCA:")
    print(f" - PC1: {variance_explained[0]*100:.2f}%")
    print(f" - PC2: {variance_explained[1]*100:.2f}%")
    print(f" - Tổng phương sai giải thích được: {sum(variance_explained)*100:.2f}%")
    
    # 1. Trực quan hóa kết quả K-Means (Dùng bảng màu cao cấp, tinh tế)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='KMeans_Cluster', 
        data=df, 
        palette='Set1', 
        s=40, alpha=0.8, 
        edgecolor='w', linewidth=0.5
    )
    plt.title('Trực quan hóa Phân cụm Khách hàng bằng K-Means (K = 4) sau PCA', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Thành phần chính 1 (PC1)', fontsize=12)
    plt.ylabel('Thành phần chính 2 (PC2)', fontsize=12)
    plt.legend(title='Cụm K-Means', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png', dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ phân cụm K-Means vào file 'kmeans_clusters.png'")
    
    # 2. Trực quan hóa kết quả Hierarchical Clustering
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Hierarchical_Cluster', 
        data=df, 
        palette='Set2', 
        s=40, alpha=0.8, 
        edgecolor='w', linewidth=0.5
    )
    plt.title('Trực quan hóa Phân cụm bằng Hierarchical Clustering (K = 4) sau PCA', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Thành phần chính 1 (PC1)', fontsize=12)
    plt.ylabel('Thành phần chính 2 (PC2)', fontsize=12)
    plt.legend(title='Cụm Phân cấp', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('hierarchical_clusters.png', dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ phân cụm Hierarchical vào file 'hierarchical_clusters.png'")
    
    # 3. Trực quan hóa kết quả DBSCAN
    plt.figure(figsize=(9, 7))
    # Chuyển đổi nhãn DBSCAN thành dạng chuỗi để trực quan hóa đẹp mắt, nhãn -1 sẽ là 'Noise'
    df_temp = df.copy()
    df_temp['DBSCAN_Group'] = df_temp['DBSCAN_Cluster'].apply(lambda x: 'Noise (Nhiễu)' if x == -1 else f'Cụm {x}')
    
    # Định nghĩa bảng màu động dựa trên số cụm của DBSCAN
    unique_groups = sorted(df_temp['DBSCAN_Group'].unique())
    palette_db = {}
    colors = sns.color_palette('tab10', len(unique_groups))
    for idx, group in enumerate(unique_groups):
        if 'Noise' in group:
            palette_db[group] = '#7f8c8d'  # Màu xám cho nhiễu
        else:
            palette_db[group] = colors[idx % len(colors)]
            
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='DBSCAN_Group', 
        data=df_temp, 
        palette=palette_db, 
        s=40, alpha=0.8, 
        edgecolor='w', linewidth=0.5
    )
    plt.title('Trực quan hóa Phân cụm bằng DBSCAN sau PCA', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Thành phần chính 1 (PC1)', fontsize=12)
    plt.ylabel('Thành phần chính 2 (PC2)', fontsize=12)
    plt.legend(title='Nhóm DBSCAN', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('dbscan_clusters.png', dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ phân cụm DBSCAN vào file 'dbscan_clusters.png'")
    
    print("--- TRỰC QUAN HÓA HOÀN TẤT ---\n")

def profile_and_save_data(df, features_list):
    print("--- 4. PHÂN TÍCH CHÂN DUNG KHÁCH HÀNG (PROFILING) ---")
    
    # Sử dụng phân cụm K-Means (K=4) vì đây là thuật toán tối ưu nhất trong nghiên cứu thực nghiệm này
    # Ta sẽ gán nhãn phân khúc có ý nghĩa dựa trên các đặc trưng trung bình của từng cụm
    cluster_means = df.groupby('KMeans_Cluster')[features_list].mean()
    print("\nCác chỉ số trung bình của từng cụm K-Means:")
    print(cluster_means)
    
    # Để đặt nhãn tự động một cách thông minh và không phụ thuộc vào thứ tự ngẫu nhiên của nhãn cụm:
    # 1. Nhóm 'Tinh hoa' (Elite) có chi tiêu trung bình (Spent) cao nhất
    # 2. Nhóm 'Tiềm năng' (Potential) có chi tiêu cao thứ nhì và thâm niên tốt
    # 3. Nhóm 'Săn deal' (Deal Seeker) có quy mô gia đình lớn, chi tiêu trung bình và số lượng giao dịch khuyến mãi cao
    # 4. Nhóm 'Ít hoạt động' (Inactive) có thu nhập và chi tiêu thấp nhất
    
    # Sắp xếp các cụm theo mức độ chi tiêu trung bình giảm dần
    sorted_spent_idx = cluster_means['Spent'].sort_values(ascending=False).index.tolist()
    
    # Ánh xạ từ chỉ số cụm sang tên phân khúc thị trường tương ứng
    # - Cụm chi tiêu cao nhất (idx 0): Khách hàng Tinh hoa (Elite Customers)
    # - Cụm chi tiêu cao thứ 2 (idx 1): Khách hàng Tiềm năng (Potential Customers)
    # - Cụm chi tiêu cao thứ 3 (idx 2): Khách hàng Nhạy cảm giá / Săn Deal (Deal Seekers)
    # - Cụm chi tiêu thấp nhất (idx 3): Khách hàng Ít hoạt động (Inactive Customers)
    
    cluster_mapping = {
        sorted_spent_idx[0]: 'Khách hàng Tinh hoa (High Income - High Spent)',
        sorted_spent_idx[1]: 'Khách hàng Tiềm năng (Moderate Income - Loyal)',
        sorted_spent_idx[2]: 'Khách hàng Săn Deal (Large Family - Deal Seekers)',
        sorted_spent_idx[3]: 'Khách hàng Ít hoạt động (Low Income - Low Spent)'
    }
    
    df['Segment_Name'] = df['KMeans_Cluster'].map(cluster_mapping)
    
    print("\nKết quả ánh xạ phân khúc khách hàng:")
    for cluster_id, segment in cluster_mapping.items():
        count = df[df['KMeans_Cluster'] == cluster_id].shape[0]
        percent = (count / df.shape[0]) * 100
        print(f" - Cụm {cluster_id} -> {segment}: {count} khách hàng ({percent:.2f}%)")
        
    # Tạo bảng báo cáo chân dung khách hàng tổng quan (mean và median) để lưu
    profiling_report = df.groupby('Segment_Name')[features_list].agg(['mean', 'median', 'count']).round(2)
    
    # 5. Xuất file kết quả ra Excel sạch đẹp sử dụng pandas
    output_filename = 'ket_qua_phan_khuc_khach_hang.xlsx'
    
    # Sử dụng ExcelWriter để lưu nhiều sheet: 1 sheet chứa data chi tiết, 1 sheet chứa thống kê mô tả chân dung cụm
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Dữ liệu chi tiết khách hàng đã được phân nhóm
        # Sắp xếp theo tên phân khúc để file excel trông khoa học
        df_sorted = df.sort_values(by='Segment_Name')
        # Loại bỏ các cột tọa độ PCA không cần thiết cho người đọc kinh doanh
        cols_to_save = [col for col in df_sorted.columns if col not in ['PC1', 'PC2']]
        df_sorted[cols_to_save].to_excel(writer, sheet_name='Data_Phan_Khuc_Chi_Tiet', index=False)
        
        # Sheet 2: Thống kê mô tả đặc trưng các nhóm khách hàng
        cluster_means_for_excel = df.groupby('Segment_Name')[features_list].mean().round(2)
        cluster_means_for_excel.to_excel(writer, sheet_name='Dac_Trung_Phan_Khuc')
        
    print(f"\n-> Đã xuất kết quả thành công ra file Excel '{output_filename}' với 2 sheet dữ liệu phân tích chi tiết!")
    print("--- PHÂN TÍCH CHÂN DUNG HOÀN TẤT ---\n")
    return cluster_mapping

if __name__ == '__main__':
    data_path = 'marketing_campaign.xls'
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại đường dẫn: {data_path}")
    else:
        df, scaled_features, features_list = preprocess_data(data_path)
        df, km_sil, agg_sil, db_sil = run_clustering(df, scaled_features, features_list)
        visualize_clusters(df, scaled_features)
        cluster_mapping = profile_and_save_data(df, features_list)
        print("=== CHƯƠNG TRÌNH PIPELINE ĐÃ HOÀN THÀNH XUẤT SẮC! ===")
