import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import io

# Cấu hình trang Streamlit với layout rộng và tiêu đề đẹp
st.set_page_config(
    page_title="Hệ Thống Phân Khúc Khách Hàng - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng Custom CSS để giao diện trông sang trọng và hiện đại
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #EBF5FF;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #BFDBFE;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Ứng Dụng Khai Thác Dữ Liệu Phân Khúc Khách Hàng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Đề tài: Ứng dụng kỹ thuật gom nhóm trong khai thác dữ liệu khách hàng để phân khúc thị trường</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR: ĐIỀU KHIỂN & NHẬP LIỆU -----------------
st.sidebar.header("🛠️ CẤU HÌNH THỰC NGHIỆM")

# 1. Cho phép upload file dữ liệu tùy ý (mặc định sẽ dùng file marketing_campaign.xls có sẵn)
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu khách hàng (.xls, .xlsx, .csv)", type=["xls", "xlsx", "csv"])

# Đọc dữ liệu
@st.cache_data
def load_data(file_source):
    if file_source is not None:
        if file_source.name.endswith('.csv'):
            return pd.read_csv(file_source)
        else:
            # File excel hoặc tsv trá hình xls
            try:
                return pd.read_csv(file_source, sep='\t')
            except:
                return pd.read_excel(file_source)
    else:
        # Sử dụng dữ liệu mặc định trong workspace
        return pd.read_csv('marketing_campaign.xls', sep='\t')

try:
    df_raw = load_data(uploaded_file)
except Exception as e:
    st.error(f"Lỗi khi đọc file dữ liệu: {e}")
    st.stop()

# 2. Lựa chọn thuật toán gom cụm trên Sidebar
st.sidebar.subheader("🤖 Chọn Thuật Toán Gom Cụm")
algo = st.sidebar.selectbox(
    "Thuật toán:",
    ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN (Density-Based)"]
)

# 3. Điều chỉnh tham số động theo thuật toán đã chọn
if algo == "K-Means Clustering":
    st.sidebar.markdown("**Siêu tham số K-Means:**")
    k_val = st.sidebar.slider("Số lượng cụm K:", min_value=2, max_value=8, value=4, step=1)
elif algo == "Hierarchical Clustering":
    st.sidebar.markdown("**Siêu tham số Phân cấp:**")
    k_val = st.sidebar.slider("Số lượng cụm K:", min_value=2, max_value=8, value=4, step=1)
    linkage_val = st.sidebar.selectbox("Tiêu chí liên kết (Linkage):", ["ward", "complete", "average", "single"])
else:
    st.sidebar.markdown("**Siêu tham số DBSCAN:**")
    eps_val = st.sidebar.slider("Bán kính lân cận (eps):", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    min_samples_val = st.sidebar.slider("Số điểm tối thiểu (min_samples):", min_value=2, max_value=20, value=7, step=1)

# ----------------- TIỀN XỬ LÝ DỮ LIỆU TỰ ĐỘNG -----------------
@st.cache_data
def preprocess_pipeline(df):
    df_clean = df.copy()
    
    # 1. Điền khuyết Income
    df_clean['Income'] = df_clean.groupby('Education')['Income'].transform(lambda x: x.fillna(x.median()))
    
    # 2. Feature Engineering
    df_clean['Age'] = 2015 - df_clean['Year_Birth']
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df_clean['Spent'] = df_clean[mnt_cols].sum(axis=1)
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']
    df_clean['Total_Purchases'] = df_clean[purchase_cols].sum(axis=1)
    
    df_clean['Living_With'] = df_clean['Marital_Status'].replace({
        'Married': 'Partner', 'Together': 'Partner', 'Single': 'Alone', 
        'Divorced': 'Alone', 'Widow': 'Alone', 'Alone': 'Alone', 
        'Absurd': 'Alone', 'YOLO': 'Alone'
    })
    df_clean['Family_Size'] = df_clean['Living_With'].replace({'Partner': 2, 'Alone': 1}) + df_clean['Kidhome'] + df_clean['Teenhome']
    df_clean['Is_Parent'] = (df_clean['Kidhome'] + df_clean['Teenhome'] > 0).astype(int)
    
    df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    newest = df_clean['Dt_Customer'].max()
    df_clean['Customer_Since_Days'] = (newest - df_clean['Dt_Customer']).dt.days
    
    # 3. Lọc ngoại lai
    df_clean = df_clean[(df_clean['Age'] < 90) & (df_clean['Income'] < 150000)]
    return df_clean

df_processed = preprocess_pipeline(df_raw)
cluster_features = ['Age', 'Income', 'Spent', 'Recency', 'Total_Purchases', 'Is_Parent', 'Family_Size']

# Chuẩn hóa dữ liệu live
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_processed[cluster_features])

# Giảm chiều dữ liệu PCA phục vụ trực quan hóa 2D
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(scaled_features)
df_processed['PC1'] = pca_result[:, 0]
df_processed['PC2'] = pca_result[:, 1]

# ----------------- TABS GIAO DIỆN CHÍNH -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Tổng Quan Dữ Liệu", 
    "⚙️ Tiền Xử Lý Dữ Liệu", 
    "🧠 Kết Quả Phân Cụm", 
    "🎯 Chân Dung Khách Hàng", 
    "📈 Chiến Lược Tiếp Thị & Tải Kết Quả"
])

# ----- TAB 1: TỔNG QUAN DỮ LIỆU -----
with tab1:
    st.markdown('<div class="card"><h3>1. Khám phá Tập dữ liệu thô ban đầu</h3></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-box"><h4>Số dòng dữ liệu</h4><p style="font-size:24px; font-weight:bold; color:#1E3A8A;">{df_raw.shape[0]:,}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h4>Số cột ban đầu</h4><p style="font-size:24px; font-weight:bold; color:#1E3A8A;">{df_raw.shape[1]}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h4>Số lượng giá trị khuyết thiếu</h4><p style="font-size:24px; font-weight:bold; color:#d9534f;">{df_raw.isnull().sum().sum()} ô</p></div>', unsafe_allow_html=True)
        
    st.write("#### 5 dòng dữ liệu đầu tiên:")
    st.dataframe(df_raw.head())
    
    st.write("#### Thống kê mô tả các cột số thô:")
    st.dataframe(df_raw.describe())

# ----- TAB 2: TIỀN XỬ LÝ -----
with tab2:
    st.markdown('<div class="card"><h3>2. Quy trình xử lý và tạo biến số đặc trưng</h3></div>', unsafe_allow_html=True)
    
    st.markdown("""
    Hệ thống tự động thực thi các bước sau trong nền:
    1. **Điền khuyết thu nhập (`Income`)**: Sử dụng giá trị trung vị của nhóm có cùng trình độ học vấn (`Education`).
    2. **Kỹ nghệ biến số (Feature Engineering)**: Tạo lập các biến số đại diện cho hành vi sắm sửa gồm:
       * **Tuổi (`Age`)**: 2015 - Năm sinh.
       * **Tổng chi tiêu (`Spent`)**: Tổng tiền mua rượu vang, cá, thịt, trái cây, đồ ngọt, vàng.
       * **Tổng mua sắm (`Total_Purchases`)**: Tổng số lần mua qua Web, Catalog, Store và Deal khuyến mãi.
       * **Quy mô gia đình (`Family_Size`)**: Tình trạng sống cặp đôi/độc thân + Số lượng con nhỏ ở nhà.
    3. **Lọc sạch dữ liệu ngoại lai (Outliers)**: Loại bỏ khách hàng trên 90 tuổi và thu nhập bất thường trên 150,000 USD.
    """)
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.success(f"Dữ liệu ban đầu: **{df_raw.shape[0]} dòng** -> Sau khi tiền xử lý & lọc ngoại lai: **{df_processed.shape[0]} dòng** (Đã loại bỏ {df_raw.shape[0] - df_processed.shape[0]} dòng ngoại lai).")
    with col_pre2:
        st.info(f"Các biến số đặc trưng được lựa chọn để huấn luyện phân cụm gồm: {cluster_features}")
        
    st.write("#### Dữ liệu sau tiền xử lý sẵn sàng đưa vào mô hình:")
    st.dataframe(df_processed[['Age', 'Income', 'Spent', 'Recency', 'Total_Purchases', 'Is_Parent', 'Family_Size']].head())

# ----- TAB 3: KẾT QUẢ PHÂN CỤM LIVE -----
with tab3:
    st.markdown('<div class="card"><h3>3. Kết quả huấn luyện và đánh giá mô hình trực tuyến</h3></div>', unsafe_allow_html=True)
    
    # Huấn luyện mô hình live dựa trên lựa chọn của user trên Sidebar
    if algo == "K-Means Clustering":
        model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        df_processed['Cluster'] = labels
        
        # Tính toán metric
        score = silhouette_score(scaled_features, labels)
        wcss_score = model.inertia_
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Silhouette Score (Độ tách cụm)", value=f"{score:.4f}")
        with c2:
            st.metric(label="Tổng bình phương nội cụm WCSS (Inertia)", value=f"{wcss_score:,.2f}")
            
    elif algo == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=k_val, linkage=linkage_val)
        labels = model.fit_predict(scaled_features)
        df_processed['Cluster'] = labels
        
        score = silhouette_score(scaled_features, labels)
        st.metric(label="Silhouette Score (Độ tách cụm)", value=f"{score:.4f}")
        
    else: # DBSCAN
        model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        labels = model.fit_predict(scaled_features)
        df_processed['Cluster'] = labels
        
        unique_l = set(labels)
        n_clusters = len(unique_l) - (1 if -1 in unique_l else 0)
        n_noise = list(labels).count(-1)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Số cụm phát hiện tự động", value=n_clusters)
        with c2:
            st.metric(label="Số điểm nhiễu (outliers)", value=f"{n_noise} ({n_noise/len(df_processed)*100:.2f}%)")
            
        if n_clusters > 1:
            score = silhouette_score(scaled_features, labels)
            st.metric(label="Silhouette Score (Độ tách cụm)", value=f"{score:.4f}")
        else:
            st.warning("DBSCAN không phát hiện đủ số lượng cụm (>1) để đánh giá chỉ số Silhouette một cách hợp lý.")
            score = -1

    # Trực quan hóa phân cụm PCA 2D
    st.write("#### Bản đồ Định vị Khách hàng trên không gian 2D PCA")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if algo == "DBSCAN (Density-Based)":
        df_processed['DBSCAN_Group'] = df_processed['Cluster'].apply(lambda x: 'Nhiễu (Noise)' if x == -1 else f'Cụm {x}')
        sns.scatterplot(
            x='PC1', y='PC2', hue='DBSCAN_Group', data=df_processed,
            palette='tab10', s=35, alpha=0.8, ax=ax
        )
    else:
        sns.scatterplot(
            x='PC1', y='PC2', hue='Cluster', data=df_processed,
            palette='Set1', s=35, alpha=0.8, ax=ax
        )
    ax.set_title(f"Trực quan hóa Phân cụm sử dụng giảm chiều PCA ({algo})", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Thành phần chính 1 - PC1 (Phương sai giải thích: 44.05%)")
    ax.set_ylabel(f"Thành phần chính 2 - PC2 (Phương sai giải thích: 18.99%)")
    st.pyplot(fig)

# ----- TAB 4: CHÂN DUNG KHÁCH HÀNG -----
with tab4:
    st.markdown('<div class="card"><h3>4. Mô tả chân dung các phân khúc khách hàng</h3></div>', unsafe_allow_html=True)
    
    # Tính toán đặc trưng trung bình theo cụm
    cluster_means_live = df_processed.groupby('Cluster')[cluster_features].mean()
    
    # Để đặt tên có ý nghĩa thương mại ổn định và nhất quán dựa trên K-Means 4 cụm mặc định:
    # Ta tự động gán tên phân khúc dựa trên sắp xếp mức chi tiêu (Spent) của các cụm
    if (algo == "K-Means Clustering" or algo == "Hierarchical Clustering") and k_val == 4:
        sorted_spent_idx = cluster_means_live['Spent'].sort_values(ascending=False).index.tolist()
        segment_mapping = {
            sorted_spent_idx[0]: 'Khách hàng Tinh hoa (Elite Customers)',
            sorted_spent_idx[1]: 'Khách hàng Tiềm năng (Loyal/Family Customers)',
            sorted_spent_idx[2]: 'Khách hàng Trẻ Độc thân / Săn Deal (Deal Seekers)',
            sorted_spent_idx[3]: 'Khách hàng Ít hoạt động / Giá trị thấp (Low Value/Inactive)'
        }
        df_processed['Phan_Khuc'] = df_processed['Cluster'].map(segment_mapping)
        
        st.write("#### Chỉ số trung bình của 4 phân khúc khách hàng thị trường:")
        report_table = df_processed.groupby('Phan_Khuc')[cluster_features].mean().round(2)
        st.dataframe(report_table)
        
        # Vẽ biểu đồ so sánh các phân khúc
        st.write("#### So sánh trực quan mức Chi tiêu và Thu nhập giữa các phân khúc")
        fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(x=report_table.index, y='Spent', data=report_table, ax=ax2[0], palette='Blues_r')
        ax2[0].set_title('Mức Chi Tiêu Trung Bình (Spent)')
        ax2[0].set_ylabel('USD / Năm')
        ax2[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(x=report_table.index, y='Income', data=report_table, ax=ax2[1], palette='Oranges_r')
        ax2[1].set_title('Mức Thu Nhập Trung Bình (Income)')
        ax2[1].set_ylabel('USD / Năm')
        ax2[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
    else:
        st.info("Để hiển thị phân tích chân dung có nhãn chi tiết tiếng Việt đầy đủ, bạn vui lòng chọn thuật toán **K-Means** hoặc **Hierarchical** và kéo chọn số lượng cụm **K = 4** ở Sidebar nhé!")
        st.write("#### Chỉ số trung bình của các cụm hiện tại:")
        st.dataframe(cluster_means_live.round(2))

# ----- TAB 5: MARKETING & EXPORT -----
with tab5:
    st.markdown('<div class="card"><h3>5. Đề xuất chiến dịch Marketing cá nhân hóa & Tải báo cáo</h3></div>', unsafe_allow_html=True)
    
    col_mar1, col_mar2 = st.columns([2, 1])
    
    with col_mar1:
        st.write("#### Chiến lược tiếp thị chuyên biệt cho 4 phân khúc cốt lõi:")
        
        with st.expander("💎 1. Khách Hàng Tinh Hoa (High Income - High Spent)"):
            st.markdown("""
            * **Định vị**: Sản phẩm xa xỉ, giới hạn, đặc quyền tối cao.
            * **Tiếp thị**: Tiếp thị ngầm chuyên biệt 1-1, thư mời VIP dự sự kiện độc quyền.
            * **Khuyến mãi**: Tặng quà cao cấp độc bản, miễn phí giao nhận hàng siêu tốc tận nhà. Tránh giảm giá trực tiếp.
            """)
            
        with st.expander("👨‍👩‍👧‍👦 2. Khách Hàng Tiềm Năng (Loyal & Family Customers)"):
            st.markdown("""
            * **Định vị**: Tiện ích gia đình, niềm tin vững chắc, đồng hành lâu dài.
            * **Tiếp thị**: Gửi các chiến dịch email marketing cá nhân hóa, gợi ý combo gia đình phù hợp.
            * **Khuyến mãi**: Thiết lập chương trình hoàn tiền (Cashback), chương trình tích điểm đổi quà VIP định kỳ dài hạn.
            """)
            
        with st.expander("🏷️ 3. Khách Hàng Trẻ Độc thân / Săn Deal (Deal Seekers)"):
            st.markdown("""
            * **Định vị**: Năng động, cá tính, bắt kịp xu hướng và tối ưu chi phí.
            * **Tiếp thị**: Đẩy thông báo nóng qua ứng dụng di động, kết hợp KOLs trên mạng xã hội.
            * **Khuyến mãi**: Các khung giờ flash sale giá sốc, mã giảm giá khi giới thiệu bạn bè sử dụng dịch vụ.
            """)
            
        with st.expander("💤 4. Khách Hàng Ít Hoạt Động (Low Income - Low Spent)"):
            st.markdown("""
            * **Định vị**: Tiết kiệm, nhu yếu phẩm cơ bản, thiết thực.
            * **Tiếp thị**: Chiến dịch SMS tự động quy mô lớn với chi phí tiếp cận thấp nhất.
            * **Khuyến mãi**: Tặng mã giảm giá trực tiếp khi mua nhu yếu phẩm thiết thực nhân ngày sinh nhật để kích hoạt lại tương tác.
            """)
            
    with col_mar2:
        st.write("#### 💾 Xuất kết quả ra file Excel")
        st.write("Hệ thống hỗ trợ đóng gói dữ liệu đã phân nhóm chi tiết kèm bảng báo cáo đặc trưng chân dung để bạn tải về ngay lập tức:")
        
        # Hàm in-memory xuất excel
        def convert_df_to_excel(df, summary):
            output = io.BytesIO()
            # Loại bỏ cột PCA và DBSCAN tạm thời cho gọn
            cols_to_drop = ['PC1', 'PC2', 'DBSCAN_Group']
            clean_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                clean_df.to_excel(writer, sheet_name='Data_Phan_Khuc_Chi_Tiet', index=False)
                summary.to_excel(writer, sheet_name='Dac_Trung_Phan_Khuc')
            return output.getvalue()
            
        if (algo == "K-Means Clustering" or algo == "Hierarchical Clustering") and k_val == 4:
            excel_data = convert_df_to_excel(df_processed, report_table)
            
            st.download_button(
                label="📥 Tải Xuống File Excel Kết Quả",
                data=excel_data,
                file_name="ket_qua_phan_khuc_khach_hang_dashboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Nhấn nút phía trên để tải file Excel kết quả phân khúc thị trường!")
        else:
            st.warning("Vui lòng cấu hình thuật toán **K-Means / Hierarchical** với **K = 4** trên Sidebar để mở khóa tính năng tải tệp Excel phân nhóm sạch đẹp!")
