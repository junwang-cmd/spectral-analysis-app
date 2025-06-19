# ====================================
# 导入必要的库
# ====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import traceback
import time
from io import BytesIO

# 机器学习相关
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 特征选择相关
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)

# 信号处理相关
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================================
# 1. 页面配置和样式设置
# ====================================

def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .section-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """初始化会话状态"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False
    if 'feature_selected' not in st.session_state:
        st.session_state.feature_selected = False
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'feature_selection_method' not in st.session_state:
        st.session_state.feature_selection_method = None
    if 'feature_selector' not in st.session_state:
        st.session_state.feature_selector = None

# ====================================
# 2. 基线校正算法类
# ====================================

class SpectrumBaselineCorrector:
    """光谱基线校正算法集合"""
    
    @staticmethod
    def airpls(y, lambda_=1e4, porder=1, itermax=15):
        """
        自适应迭代加权惩罚最小二乘法 (Adaptive Iteratively Reweighted Penalized Least Squares)
        """
        m = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m-2))
        D = lambda_ * D.dot(D.transpose())
        
        w = np.ones(m)
        W = sparse.spdiags(w, 0, m, m)
        
        for i in range(1, itermax+1):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w*y)
            d = y - z
            
            # 更新权重
            dn = d[d < 0]
            m_dn = np.mean(dn) if len(dn) > 0 else 0
            s_dn = np.std(dn) if len(dn) > 0 else 1
            
            wt = 1 / (1 + np.exp(2 * (d - (2*s_dn - m_dn))/s_dn))
            
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < 1e-3:
                break
            w = wt
        
        return z, y - z
    
    @staticmethod
    def asls(y, lam=1e4, p=0.001, niter=10):
        """
        非对称最小二乘法 (Asymmetric Least Squares)
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        
        return z, y - z
    
    @staticmethod
    def polynomial_baseline(y, degree=2):
        """
        多项式基线校正
        """
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return baseline, y - baseline
    
    @staticmethod
    def modpoly(y, degree=2, repitition=100):
        """
        修正多项式基线校正 (Modified Polynomial)
        """
        x = np.arange(len(y))
        baseline = y.copy()
        
        for _ in range(repitition):
            coeffs = np.polyfit(x, baseline, degree)
            fitted = np.polyval(coeffs, x)
            baseline = np.minimum(baseline, fitted)
        
        return baseline, y - baseline
    
    def correct_baseline(self, spectrum, method='airpls', **kwargs):
        """
        统一的基线校正接口
        
        Parameters:
        -----------
        spectrum : array-like
            输入光谱数据
        method : str
            基线校正方法 ('airpls', 'asls', 'polynomial', 'modpoly')
        **kwargs : 
            方法特定的参数
        
        Returns:
        --------
        baseline : ndarray
            基线
        corrected : ndarray
            校正后的光谱
        """
        if method == 'airpls':
            return self.airpls(spectrum, **kwargs)
        elif method == 'asls':
            return self.asls(spectrum, **kwargs)
        elif method == 'polynomial':
            return self.polynomial_baseline(spectrum, **kwargs)
        elif method == 'modpoly':
            return self.modpoly(spectrum, **kwargs)
        else:
            raise ValueError(f"Unknown baseline correction method: {method}")

# ====================================
# 3. 特征选择功能
# ====================================

def perform_feature_selection(X, y, method, **params):
    """
    执行特征选择
    
    Parameters:
    -----------
    X : array-like
        特征矩阵
    y : array-like
        目标变量
    method : str
        特征选择方法
    **params : dict
        方法参数
    
    Returns:
    --------
    selector : object
        特征选择器对象
    X_selected : array-like
        选择后的特征矩阵
    selected_indices : array-like
        被选择的特征索引
    """
    
    if method == "方差阈值":
        threshold = params.get('variance_threshold', 0.01)
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_indices = selector.get_support(indices=True)
        
    elif method == "单变量选择":
        k = params.get('k_features', 100)
        score_func = params.get('score_func', f_regression)
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == "递归特征消除":
        n_features = params.get('n_features', 50)
        estimator = params.get('estimator', LinearRegression())
        selector = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == "基于模型重要性":
        estimator = params.get('estimator', RandomForestRegressor(n_estimators=100, random_state=42))
        threshold = params.get('threshold', 'mean')
        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == "相关性筛选":
        threshold = params.get('corr_threshold', 0.5)
        # 计算每个特征与目标变量的相关性
        if y.ndim > 1:
            # 多目标情况，取平均相关性
            correlations = []
            for i in range(X.shape[1]):
                corr_vals = []
                for j in range(y.shape[1]):
                    corr = np.corrcoef(X[:, i], y[:, j])[0, 1]
                    if not np.isnan(corr):
                        corr_vals.append(abs(corr))
                correlations.append(np.mean(corr_vals) if corr_vals else 0)
        else:
            # 单目标情况
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y.flatten())[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        correlations = np.array(correlations)
        selected_indices = np.where(correlations >= threshold)[0]
        
        if len(selected_indices) == 0:
            # 如果没有特征满足阈值，选择相关性最高的10个
            selected_indices = np.argsort(correlations)[-10:]
        
        X_selected = X[:, selected_indices]
        
        # 创建一个简单的选择器对象
        class CorrelationSelector:
            def __init__(self, selected_indices):
                self.selected_indices = selected_indices
            
            def transform(self, X):
                return X[:, self.selected_indices]
            
            def get_support(self, indices=False):
                if indices:
                    return self.selected_indices
                else:
                    support = np.zeros(X.shape[1], dtype=bool)
                    support[self.selected_indices] = True
                    return support
        
        selector = CorrelationSelector(selected_indices)
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return selector, X_selected, selected_indices

# ====================================
# 4. 页面函数
# ====================================

def show_data_loading_page():
    """数据加载与标签输入页面"""
    st.markdown("<h1 class='section-header'>数据加载与标签输入</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    请上传光谱数据文件，并设置目标变量（标签）。支持通过上传文件或手动输入的方式设置标签。
    </div>
    """, unsafe_allow_html=True)
    
    # 上传光谱数据文件
    st.subheader("📁 上传光谱数据文件")
    uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"数据上传成功！共{df.shape[0]}行，{df.shape[1]}列")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head(10))
            
            # 数据格式检查和处理
            st.subheader("🔍 数据格式检查")
            
            # 检测波数列
            potential_wavenumber_cols = []
            non_numeric_cols = []
            
            for col in df.columns:
                try:
                    float(col)
                    potential_wavenumber_cols.append(col)
                except ValueError:
                    non_numeric_cols.append(col)
            
            st.info(f"检测到 {len(potential_wavenumber_cols)} 个数值列（可能为波数）")
            st.info(f"检测到 {len(non_numeric_cols)} 个非数值列")
            
            if len(potential_wavenumber_cols) < 10:
                st.error("检测到的波数列数量不足，请检查数据格式")
                return
            
            # 设置数据结构
            start_col = st.selectbox(
                "选择光谱数据起始列",
                options=range(len(df.columns)),
                index=min(2, len(df.columns)-1),
                format_func=lambda x: f"第{x+1}列: {df.columns[x]}"
            )
            
            # 提取光谱数据
            spectral_columns = df.columns[start_col:]
            try:
                wavenumbers = pd.Series(spectral_columns).astype(float)
                X = df[spectral_columns].values.astype(float)
                
                st.success(f"光谱数据提取成功！波数范围: {wavenumbers.min():.1f} ~ {wavenumbers.max():.1f} cm⁻¹")
                st.info(f"光谱数据形状: {X.shape}")
                
                # 保存到session state
                st.session_state.X = X
                st.session_state.wavenumbers = wavenumbers.values
                
                # 保存非光谱列用于标签设置
                if start_col > 0:
                    st.session_state.sample_info = df.iloc[:, :start_col]
                else:
                    st.session_state.sample_info = pd.DataFrame()
                
            except Exception as e:
                st.error(f"光谱数据处理失败: {e}")
                return
            
            # 标签设置部分
            st.subheader("🏷️ 目标变量设置")
            
            label_method = st.radio(
                "选择标签输入方式",
                ["上传标签文件", "手动输入标签", "使用数据文件中的列"]
            )
            
            if label_method == "上传标签文件":
                label_file = st.file_uploader("上传标签文件", type=["csv", "xlsx", "xls"], key="label_file")
                
                if label_file is not None:
                    try:
                        if label_file.name.endswith('.csv'):
                            label_df = pd.read_csv(label_file)
                        else:
                            label_df = pd.read_excel(label_file)
                        
                        st.write("标签文件预览：")
                        st.dataframe(label_df.head())
                        
                        # 选择标签列
                        available_cols = [col for col in label_df.columns if label_df[col].dtype in ['int64', 'float64']]
                        if available_cols:
                            selected_cols = st.multiselect("选择目标变量列", available_cols)
                            
                            if selected_cols and len(label_df) == len(df):
                                st.session_state.y = label_df[selected_cols].values
                                st.session_state.selected_cols = selected_cols
                                st.session_state.data_loaded = True
                                st.success("标签数据设置成功！")
                            elif len(label_df) != len(df):
                                st.error("标签文件的行数与光谱数据不匹配")
                        else:
                            st.error("标签文件中没有数值列")
                            
                    except Exception as e:
                        st.error(f"标签文件读取失败: {e}")
            
            elif label_method == "手动输入标签":
                st.write("请为每个样本输入标签值：")
                
                # 目标变量名称设置
                target_names = st.text_input("目标变量名称（多个用逗号分隔）", value="浓度").split(',')
                target_names = [name.strip() for name in target_names if name.strip()]
                
                if target_names:
                    # 创建输入表格
                    n_samples = len(df)
                    n_targets = len(target_names)
                    
                    # 初始化标签数据
                    if 'manual_labels' not in st.session_state:
                        st.session_state.manual_labels = np.zeros((n_samples, n_targets))
                    
                    # 调整标签数据维度
                    if st.session_state.manual_labels.shape != (n_samples, n_targets):
                        st.session_state.manual_labels = np.zeros((n_samples, n_targets))
                    
                    # 显示输入界面
                    st.write(f"请为 {n_samples} 个样本输入 {n_targets} 个目标变量的值：")
                    
                    # 分页显示（每页10个样本）
                    samples_per_page = 10
                    n_pages = (n_samples + samples_per_page - 1) // samples_per_page
                    
                    if n_pages > 1:
                        page = st.selectbox("选择页面", range(1, n_pages + 1))
                        start_idx = (page - 1) * samples_per_page
                        end_idx = min(start_idx + samples_per_page, n_samples)
                    else:
                        start_idx = 0
                        end_idx = n_samples
                    
                    # 创建输入表格
                    for i in range(start_idx, end_idx):
                        cols = st.columns([1] + [2] * n_targets)
                        with cols[0]:
                            st.write(f"样本 {i+1}")
                        
                        for j, target_name in enumerate(target_names):
                            with cols[j+1]:
                                value = st.number_input(
                                    target_name,
                                    value=float(st.session_state.manual_labels[i, j]),
                                    key=f"label_{i}_{j}",
                                    format="%.4f"
                                )
                                st.session_state.manual_labels[i, j] = value
                    
                    # 设置标签按钮
                    if st.button("设置标签"):
                        st.session_state.y = st.session_state.manual_labels
                        st.session_state.selected_cols = target_names
                        st.session_state.data_loaded = True
                        st.success("手动标签设置成功！")
            
            elif label_method == "使用数据文件中的列":
                if start_col > 0:
                    available_cols = [col for col in df.columns[:start_col] 
                                    if df[col].dtype in ['int64', 'float64']]
                    
                    if available_cols:
                        selected_cols = st.multiselect("选择目标变量列", available_cols)
                        
                        if selected_cols:
                            if st.button("设置标签"):
                                st.session_state.y = df[selected_cols].values
                                st.session_state.selected_cols = selected_cols
                                st.session_state.data_loaded = True
                                st.success("标签设置成功！")
                    else:
                        st.warning("光谱数据前面没有数值列可用作标签")
                else:
                    st.warning("没有可用的列作为标签")
            
            # 显示当前状态
            if st.session_state.data_loaded:
                st.subheader("✅ 数据加载状态")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("样本数量", st.session_state.X.shape[0])
                with col2:
                    st.metric("光谱特征数", st.session_state.X.shape[1])
                with col3:
                    st.metric("目标变量数", len(st.session_state.selected_cols))
                
                st.write(f"**目标变量**: {', '.join(st.session_state.selected_cols)}")
                
                # 显示光谱预览
                st.subheader("光谱数据预览")
                fig, ax = plt.subplots(figsize=(12, 6))
                n_samples_to_show = min(10, st.session_state.X.shape[0])
                for i in range(n_samples_to_show):
                    ax.plot(st.session_state.wavenumbers, st.session_state.X[i], alpha=0.7)
                ax.set_xlabel('波数 (cm⁻¹)')
                ax.set_ylabel('强度')
                ax.set_title(f'光谱数据预览（前{n_samples_to_show}个样本）')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"数据处理出错: {e}")
            st.error(traceback.format_exc())
    else:
        st.info("请上传光谱数据文件")

def show_preprocessing_page():
    """数据预处理页面"""
    st.markdown("<h1 class='section-header'>数据预处理</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("请先完成数据加载和标签设置")
        return
    
    st.markdown("""
    <div class="info-box">
    对光谱数据进行预处理，包括波数范围选择、平滑、基线校正、归一化等步骤，以提高数据质量和模型性能。
    </div>
    """, unsafe_allow_html=True)
    
    X = st.session_state.X
    y = st.session_state.y
    wavenumbers = st.session_state.wavenumbers
    
    # 显示原始数据信息
    st.subheader("📊 原始数据信息")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样本数量", X.shape[0])
    with col2:
        st.metric("原始特征数", X.shape[1])
    with col3:
        st.metric("波数范围", f"{wavenumbers.min():.0f}-{wavenumbers.max():.0f}")
    with col4:
        st.metric("目标变量数", y.shape[1] if y.ndim > 1 else 1)
    
    # 创建预处理参数设置区域
    st.subheader("⚙️ 预处理参数设置")
    
    # 波数范围选择
    st.write("**1. 波数范围选择**")
    col1, col2 = st.columns(2)
    with col1:
        start_wn = st.number_input("起始波数 (cm⁻¹)", 
                                 min_value=float(wavenumbers.min()), 
                                 max_value=float(wavenumbers.max()),
                                 value=float(wavenumbers.min()))
    with col2:
        end_wn = st.number_input("结束波数 (cm⁻¹)", 
                               min_value=float(wavenumbers.min()), 
                               max_value=float(wavenumbers.max()),
                               value=float(wavenumbers.max()))
    
    # 平滑处理
    st.write("**2. 平滑处理**")
    apply_smooth = st.checkbox("应用Savitzky-Golay平滑", value=True)
    if apply_smooth:
        col1, col2 = st.columns(2)
        with col1:
            smooth_window = st.slider("窗口大小", 5, 51, 15, step=2)
        with col2:
            smooth_poly = st.slider("多项式阶数", 1, 5, 3)
    
    # 基线校正
    st.write("**3. 基线校正**")
    apply_baseline = st.checkbox("应用基线校正", value=True)
    if apply_baseline:
        baseline_method = st.selectbox("基线校正方法", 
                                     ["airpls", "asls", "polynomial", "modpoly"])
        
        if baseline_method == "airpls":
            col1, col2 = st.columns(2)
            with col1:
                lambda_param = st.selectbox("λ参数", [1e2, 1e3, 1e4, 1e5, 1e6], index=2)
            with col2:
                max_iter = st.slider("最大迭代次数", 5, 50, 15)
            baseline_params = {'lambda_': lambda_param, 'itermax': max_iter}
            
        elif baseline_method == "asls":
            col1, col2, col3 = st.columns(3)
            with col1:
                lam = st.selectbox("λ参数", [1e2, 1e3, 1e4, 1e5], index=2)
            with col2:
                p = st.selectbox("p参数", [0.001, 0.01, 0.1], index=0)
            with col3:
                niter = st.slider("迭代次数", 5, 20, 10)
            baseline_params = {'lam': lam, 'p': p, 'niter': niter}
            
        elif baseline_method == "polynomial":
            degree = st.slider("多项式阶数", 1, 5, 2)
            baseline_params = {'degree': degree}
            
        elif baseline_method == "modpoly":
            col1, col2 = st.columns(2)
            with col1:
                degree = st.slider("多项式阶数", 1, 5, 2)
            with col2:
                repetition = st.slider("重复次数", 10, 200, 100)
            baseline_params = {'degree': degree, 'repitition': repetition}
    
    # 归一化
    st.write("**4. 归一化**")
    apply_normalize = st.checkbox("应用归一化", value=True)
    if apply_normalize:
        normalize_method = st.selectbox("归一化方法", 
                                      ["area", "max", "vector", "minmax"])
    
    # SNV (标准正态变量变换)
    st.write("**5. 标准正态变量变换 (SNV)**")
    apply_snv = st.checkbox("应用SNV", value=False)
    
    # 预处理预览
    if st.button("🔍 预览预处理效果"):
        with st.spinner("正在处理数据..."):
            try:
                # 执行预处理流程（预览版）
                X_processed, wavenumbers_processed = perform_preprocessing(
                    X, wavenumbers, start_wn, end_wn, apply_smooth, smooth_window, smooth_poly,
                    apply_baseline, baseline_method, baseline_params, apply_normalize, 
                    normalize_method, apply_snv, preview=True
                )
                
                # 显示预处理前后对比
                show_preprocessing_comparison(X, wavenumbers, X_processed, wavenumbers_processed)
                
            except Exception as e:
                st.error(f"预处理预览失败: {e}")
                st.error(traceback.format_exc())
    
    # 应用预处理
    if st.button("✅ 应用预处理设置", type="primary"):
        with st.spinner("正在进行数据预处理..."):
            try:
                # 执行完整的预处理流程
                X_processed, wavenumbers_processed = perform_preprocessing(
                    X, wavenumbers, start_wn, end_wn, apply_smooth, smooth_window, smooth_poly,
                    apply_baseline, baseline_method, baseline_params, apply_normalize, 
                    normalize_method, apply_snv, preview=False
                )
                
                # 保存预处理结果
                st.session_state.X_preprocessed = X_processed
                st.session_state.wavenumbers_preprocessed = wavenumbers_processed
                st.session_state.preprocessing_done = True
                
                # 重置特征选择状态
                st.session_state.feature_selected = False
                st.session_state.selected_features = None
                st.session_state.feature_selector = None
                
                # 保存预处理参数
                st.session_state.preprocessing_params = {
                    'start_wavenumber': start_wn,
                    'end_wavenumber': end_wn,
                    'apply_smooth': apply_smooth,
                    'smooth_window': smooth_window if apply_smooth else None,
                    'smooth_poly': smooth_poly if apply_smooth else None,
                    'apply_baseline': apply_baseline,
                    'baseline_method': baseline_method if apply_baseline else None,
                    'baseline_params': baseline_params if apply_baseline else None,
                    'apply_normalize': apply_normalize,
                    'normalize_method': normalize_method if apply_normalize else None,
                    'apply_snv': apply_snv
                }
                
                st.success("🎉 数据预处理完成！可以进行下一步特征提取和可视化。")
                
                # 显示处理结果统计
                st.subheader("📊 预处理结果统计")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("处理后特征数", X_processed.shape[1])
                with col2:
                    st.metric("特征减少", f"{X.shape[1] - X_processed.shape[1]}")
                with col3:
                    st.metric("强度范围", f"{X_processed.min():.4f} ~ {X_processed.max():.4f}")
                with col4:
                    st.metric("强度标准差", f"{X_processed.std():.4f}")
                
            except Exception as e:
                st.error(f"数据预处理失败: {e}")
                st.error(traceback.format_exc())

def perform_preprocessing(X, wavenumbers, start_wn, end_wn, apply_smooth, smooth_window, 
                         smooth_poly, apply_baseline, baseline_method, baseline_params, 
                         apply_normalize, normalize_method, apply_snv, preview=False):
    """执行预处理流程"""
    
    # 1. 波数范围截取
    start_idx = np.argmin(np.abs(wavenumbers - start_wn))
    end_idx = np.argmin(np.abs(wavenumbers - end_wn)) + 1
    
    wavenumbers_processed = wavenumbers[start_idx:end_idx]
    X_processed = X[:, start_idx:end_idx]
    
    if preview:
        st.write(f"✓ 波数截取: {start_wn} ~ {end_wn} cm⁻¹")
    
    # 2. 平滑处理
    if apply_smooth:
        X_smooth = np.zeros_like(X_processed)
        for i in range(X_processed.shape[0]):
            X_smooth[i] = savgol_filter(X_processed[i], smooth_window, smooth_poly)
        X_processed = X_smooth
        if preview:
            st.write(f"✓ Savitzky-Golay平滑: 窗口={smooth_window}, 阶数={smooth_poly}")
    
    # 3. 基线校正
    if apply_baseline:
        corrector = SpectrumBaselineCorrector()
        X_corr = np.zeros_like(X_processed)
        failed_samples = []
        
        for i in range(X_processed.shape[0]):
            try:
                baseline, corrected = corrector.correct_baseline(
                    X_processed[i], baseline_method, **baseline_params)
                X_corr[i] = corrected
            except Exception as e:
                failed_samples.append(i+1)
                X_corr[i] = X_processed[i]
        
        X_processed = X_corr
        if preview and failed_samples:
            st.warning(f"样本 {failed_samples} 基线校正失败，使用原始数据")
        if preview:
            st.write(f"✓ 基线校正: {baseline_method.upper()}")
    
    # 4. 归一化
    if apply_normalize:
        X_norm = np.zeros_like(X_processed)
        for i in range(X_processed.shape[0]):
            if normalize_method == 'area':
                total_area = np.trapz(np.abs(X_processed[i]), wavenumbers_processed)
                X_norm[i] = X_processed[i] / (total_area if total_area != 0 else 1e-9)
            elif normalize_method == 'max':
                max_val = np.max(np.abs(X_processed[i]))
                X_norm[i] = X_processed[i] / (max_val if max_val != 0 else 1e-9)
            elif normalize_method == 'vector':
                norm_val = np.linalg.norm(X_processed[i])
                X_norm[i] = X_processed[i] / (norm_val if norm_val != 0 else 1e-9)
            elif normalize_method == 'minmax':
                min_val, max_val = np.min(X_processed[i]), np.max(X_processed[i])
                if max_val - min_val == 0:
                    X_norm[i] = X_processed[i]
                else:
                    X_norm[i] = (X_processed[i] - min_val) / (max_val - min_val)
        X_processed = X_norm
        if preview:
            st.write(f"✓ 归一化: {normalize_method}方法")
    
    # 5. SNV处理
    if apply_snv:
        X_final = np.zeros_like(X_processed)
        for i, spectrum in enumerate(X_processed):
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            if std_val == 0:
                X_final[i] = spectrum
            else:
                X_final[i] = (spectrum - mean_val) / std_val
        X_processed = X_final
        if preview:
            st.write("✓ 标准正态变量变换(SNV): 已应用")
    
    return X_processed, wavenumbers_processed

def show_preprocessing_comparison(X_original, wavenumbers_original, X_processed, wavenumbers_processed):
    """显示预处理前后对比"""
    st.subheader("📈 预处理效果对比")
    
    # 选择几个样本进行对比显示
    n_samples_show = min(5, X_original.shape[0])
    sample_indices = np.linspace(0, X_original.shape[0]-1, n_samples_show, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始光谱
    for i in sample_indices:
        ax1.plot(wavenumbers_original, X_original[i], alpha=0.7, label=f'样本{i+1}')
    ax1.set_title('预处理前')
    ax1.set_xlabel('波数 (cm⁻¹)')
    ax1.set_ylabel('强度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 预处理后
    for i in sample_indices:
        ax2.plot(wavenumbers_processed, X_processed[i], alpha=0.7, label=f'样本{i+1}')
    ax2.set_title('预处理后')
    ax2.set_xlabel('波数 (cm⁻¹)')
    ax2.set_ylabel('强度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 显示处理统计信息
    st.subheader("📊 预处理统计信息")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("处理后特征数", X_processed.shape[1])
    with col2:
        st.metric("特征减少", f"{X_original.shape[1] - X_processed.shape[1]}")
    with col3:
        st.metric("强度范围", f"{X_processed.min():.4f} ~ {X_processed.max():.4f}")
    with col4:
        st.metric("强度标准差", f"{X_processed.std():.4f}")

def show_feature_extraction_page():
    """特征提取与可视化页面"""
    st.markdown("<h1 class='section-header'>特征提取与可视化</h1>", unsafe_allow_html=True)
    
    if not st.session_state.preprocessing_done:
        st.warning("请先完成数据预处理")
        return
    
    st.markdown("""
    <div class="info-box">
    对预处理后的光谱数据进行探索性分析，包括光谱可视化、主成分分析和相关性分析，以及可选的特征选择。
    </div>
    """, unsafe_allow_html=True)
    
    X = st.session_state.X_preprocessed
    y = st.session_state.y
    wavenumbers = st.session_state.wavenumbers_preprocessed
    target_names = st.session_state.selected_cols
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["光谱可视化", "主成分分析", "相关性分析", "数据统计", "特征选择"])
    
    with tab1:
        st.subheader("🌈 光谱数据可视化")
        
        # 光谱显示选项
        col1, col2 = st.columns(2)
        with col1:
            n_spectra = st.slider("显示光谱数量", 1, min(50, X.shape[0]), min(10, X.shape[0]))
        with col2:
            plot_type = st.selectbox("绘图类型", ["线图", "填充图", "3D图"])
        
        # 显示光谱
        if plot_type == "线图":
            fig, ax = plt.subplots(figsize=(12, 8))
            sample_indices = np.linspace(0, X.shape[0]-1, n_spectra, dtype=int)
            
            for i, idx in enumerate(sample_indices):
                ax.plot(wavenumbers, X[idx], alpha=0.7, label=f'样本 {idx+1}')
            
            ax.set_xlabel('波数 (cm⁻¹)')
            ax.set_ylabel('强度')
            ax.set_title('预处理后光谱')
            if n_spectra <= 10:
                ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        elif plot_type == "填充图":
            fig, ax = plt.subplots(figsize=(12, 8))
            sample_indices = np.linspace(0, X.shape[0]-1, n_spectra, dtype=int)
            
            for i, idx in enumerate(sample_indices):
                ax.fill_between(wavenumbers, X[idx], alpha=0.3, label=f'样本 {idx+1}')
            
            ax.set_xlabel('波数 (cm⁻¹)')
            ax.set_ylabel('强度')
            ax.set_title('预处理后光谱（填充图）')
            if n_spectra <= 10:
                ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        elif plot_type == "3D图":
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            sample_indices = np.linspace(0, X.shape[0]-1, n_spectra, dtype=int)
            
            for i, idx in enumerate(sample_indices):
                ax.plot(wavenumbers, [i]*len(wavenumbers), X[idx], alpha=0.7)
            
            ax.set_xlabel('波数 (cm⁻¹)')
            ax.set_ylabel('样本索引')
            ax.set_zlabel('强度')
            ax.set_title('预处理后光谱（3D图）')
            st.pyplot(fig)
        
        # 统计图表
        st.subheader("📊 光谱统计信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 平均光谱和标准差
            mean_spectrum = np.mean(X, axis=0)
            std_spectrum = np.std(X, axis=0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(wavenumbers, mean_spectrum, 'b-', linewidth=2, label='平均光谱')
            ax.fill_between(wavenumbers, mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, alpha=0.3, color='blue', label='±1σ')
            ax.set_xlabel('波数 (cm⁻¹)')
            ax.set_ylabel('强度')
            ax.set_title('平均光谱与标准差')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # 光谱强度热图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 选择部分样本和波数点进行显示
            n_samples_heatmap = min(50, X.shape[0])
            n_wavenumbers_heatmap = min(100, X.shape[1])
            
            sample_step = max(1, X.shape[0] // n_samples_heatmap)
            wavenumber_step = max(1, X.shape[1] // n_wavenumbers_heatmap)
            
            X_subset = X[::sample_step, ::wavenumber_step]
            wavenumbers_subset = wavenumbers[::wavenumber_step]
            
            im = ax.imshow(X_subset, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('波数索引')
            ax.set_ylabel('样本索引')
            ax.set_title('光谱强度热图')
            
            # 设置x轴标签
            if len(wavenumbers_subset) > 10:
                tick_indices = np.linspace(0, len(wavenumbers_subset)-1, 10, dtype=int)
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([f'{wavenumbers_subset[i]:.0f}' for i in tick_indices], rotation=45)
            
            plt.colorbar(im, ax=ax, label='强度')
            st.pyplot(fig)
    
    with tab2:
        st.subheader("🔍 主成分分析 (PCA)")
        
        # PCA参数设置
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("主成分数量", 2, min(10, X.shape[1], X.shape[0]), 3)
        with col2:
            scale_data = st.checkbox("标准化数据", value=True)
        
        if st.button("执行PCA分析"):
            # 执行PCA
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # 保存PCA结果
            st.session_state.pca_result = {
                'X_pca': X_pca,
                'pca': pca,
                'explained_variance': pca.explained_variance_ratio_
            }
            
            # 显示解释方差
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 解释方差图
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(1, n_components+1), explained_variance * 100)
                ax.set_xlabel('主成分')
                ax.set_ylabel('解释方差比例 (%)')
                ax.set_title('各主成分解释方差')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, var in zip(bars, explained_variance):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{var*100:.1f}%', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                # 累积解释方差图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, n_components+1), cumulative_variance * 100, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('主成分数量')
                ax.set_ylabel('累积解释方差比例 (%)')
                ax.set_title('累积解释方差')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for i, cum_var in enumerate(cumulative_variance):
                    ax.text(i+1, cum_var*100 + 1, f'{cum_var*100:.1f}%', 
                           ha='center', va='bottom')
                
                st.pyplot(fig)
        
        # PCA得分图
        if hasattr(st.session_state, 'pca_result'):
            X_pca = st.session_state.pca_result['X_pca']
            explained_variance = st.session_state.pca_result['explained_variance']
            
            if st.checkbox("显示PCA得分图"):
                if len(target_names) == 1:
                    # 单目标变量：根据目标值着色
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.flatten(), 
                                       cmap='viridis', alpha=0.7, s=60)
                    plt.colorbar(scatter, ax=ax, label=target_names[0])
                else:
                    # 多目标变量：使用默认颜色
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=60)
                
                ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
                ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
                ax.set_title('PCA得分图')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # 载荷图
        if hasattr(st.session_state, 'pca_result'):
            pca = st.session_state.pca_result['pca']
            
            if st.checkbox("显示载荷图"):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for i in range(min(3, n_components)):
                    ax.plot(wavenumbers, pca.components_[i], label=f'PC{i+1}')
                
                ax.set_xlabel('波数 (cm⁻¹)')
                ax.set_ylabel('载荷')
                ax.set_title('主成分载荷图')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    with tab3:
        st.subheader("🔗 相关性分析")
        
        if len(target_names) == 1:
            # 单目标变量：计算每个波数与目标的相关性
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y.flatten())[0, 1]
                correlations.append(corr)
            
            correlations = np.array(correlations)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(wavenumbers, correlations)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('波数 (cm⁻¹)')
            ax.set_ylabel(f'与{target_names[0]}的相关系数')
            ax.set_title('光谱-目标变量相关性')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 找出最相关的波数
            high_corr_indices = np.argsort(np.abs(correlations))[-10:]
            st.write("**最相关的10个波数：**")
            for idx in reversed(high_corr_indices):
                st.write(f"波数 {wavenumbers[idx]:.2f} cm⁻¹: 相关系数 = {correlations[idx]:.4f}")
        
        else:
            # 多目标变量：显示目标变量间的相关性
            target_corr = np.corrcoef(y.T)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(target_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # 设置刻度标签
            ax.set_xticks(range(len(target_names)))
            ax.set_yticks(range(len(target_names)))
            ax.set_xticklabels(target_names, rotation=45)
            ax.set_yticklabels(target_names)
            
            # 添加数值标签
            for i in range(len(target_names)):
                for j in range(len(target_names)):
                    ax.text(j, i, f'{target_corr[i, j]:.3f}', 
                           ha="center", va="center", color="black")
            
            ax.set_title('目标变量相关性矩阵')
            plt.colorbar(im)
            st.pyplot(fig)
    
    with tab4:
        st.subheader("📈 数据统计信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**光谱数据统计：**")
            st.write(f"样本数量: {X.shape[0]}")
            st.write(f"特征数量: {X.shape[1]}")
            st.write(f"波数范围: {wavenumbers.min():.2f} ~ {wavenumbers.max():.2f} cm⁻¹")
            st.write(f"光谱强度范围: {X.min():.4f} ~ {X.max():.4f}")
            st.write(f"光谱强度均值: {X.mean():.4f}")
            st.write(f"光谱强度标准差: {X.std():.4f}")
        
        with col2:
            st.write("**目标变量统计：**")
            for i, target_name in enumerate(target_names):
                if len(target_names) == 1:
                    target_values = y.flatten()
                else:
                    target_values = y[:, i]
                
                st.write(f"**{target_name}:**")
                st.write(f"  范围: {target_values.min():.4f} ~ {target_values.max():.4f}")
                st.write(f"  均值: {target_values.mean():.4f}")
                st.write(f"  标准差: {target_values.std():.4f}")
        
        # 光谱强度分布
        st.write("**光谱强度分布：**")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(X.flatten(), bins=50, alpha=0.7, density=True)
        ax.set_xlabel('强度值')
        ax.set_ylabel('密度')
        ax.set_title('光谱强度分布直方图')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab5:
        st.subheader("🎯 特征选择")
        st.markdown("选择对目标变量最重要的特征，减少数据维度并提高模型性能。")

        feature_method = st.selectbox(
            "选择特征选择方法",
            ["不进行特征选择", "方差阈值", "单变量选择", "递归特征消除", "基于模型重要性", "相关性筛选"]
        )

        if feature_method == "不进行特征选择":
            if st.button("确认不进行特征选择"):
                # 使用全部预处理后的特征
                st.session_state.X_final = X
                st.session_state.wavenumbers_final = wavenumbers
                st.session_state.feature_selected = True
                st.session_state.selected_features = None
                st.session_state.feature_selector = None
                st.session_state.feature_selection_method = "不进行特征选择"
                
                st.success("✅ 已确认使用全部预处理后的特征进行建模")
                st.info(f"最终特征数量: {X.shape[1]}")

        else:
            # 特征选择参数设置
            params = {}
            
            if feature_method == "方差阈值":
                variance_threshold = st.slider("方差阈值", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
                params['variance_threshold'] = variance_threshold
                
            elif feature_method == "单变量选择":
                k_features = st.slider("选择特征数量", min_value=10, max_value=min(500, X.shape[1]), value=100)
                score_func_name = st.selectbox("评分函数", ["f_regression", "mutual_info_regression"])
                params['k_features'] = k_features
                params['score_func'] = f_regression if score_func_name == "f_regression" else mutual_info_regression
                
            elif feature_method == "递归特征消除":
                n_features = st.slider("目标特征数量", min_value=10, max_value=min(200, X.shape[1]), value=50)
                estimator_name = st.selectbox("基础估计器", ["线性回归", "随机森林"])
                params['n_features'] = n_features
                if estimator_name == "线性回归":
                    params['estimator'] = LinearRegression()
                else:
                    params['estimator'] = RandomForestRegressor(n_estimators=50, random_state=42)
                    
            elif feature_method == "基于模型重要性":
                estimator_name = st.selectbox("基础估计器", ["随机森林", "梯度提升"])
                threshold_type = st.selectbox("阈值类型", ["mean", "median"])
                
                if estimator_name == "随机森林":
                    params['estimator'] = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    params['estimator'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                params['threshold'] = threshold_type
                
            elif feature_method == "相关性筛选":
                corr_threshold = st.slider("相关性阈值", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
                params['corr_threshold'] = corr_threshold

            if st.button("执行特征选择"):
                with st.spinner("正在执行特征选择..."):
                    try:
                        # 执行特征选择
                        selector, X_selected, selected_indices = perform_feature_selection(
                            X, y, feature_method, **params
                        )
                        
                        # 保存特征选择结果
                        st.session_state.X_final = X_selected
                        st.session_state.wavenumbers_final = wavenumbers[selected_indices]
                        st.session_state.feature_selected = True
                        st.session_state.selected_features = selected_indices
                        st.session_state.feature_selector = selector
                        st.session_state.feature_selection_method = feature_method
                        
                        st.success("✅ 特征选择完成！")
                        
                        # 显示选择结果
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("原始特征数", X.shape[1])
                        with col2:
                            st.metric("选择特征数", X_selected.shape[1])
                        with col3:
                            st.metric("特征减少", X.shape[1] - X_selected.shape[1])
                        
                        # 可视化特征选择结果
                        st.subheader("📊 特征选择结果可视化")
                        
                        # 特征重要性/相关性图
                        if feature_method == "单变量选择":
                            # 显示评分
                            scores = selector.scores_[selected_indices]
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(wavenumbers[selected_indices], scores, 'bo-', alpha=0.7)
                            ax.set_xlabel('波数 (cm⁻¹)')
                            ax.set_ylabel('特征评分')
                            ax.set_title('选择特征的评分分布')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                        elif feature_method == "基于模型重要性":
                            # 显示特征重要性
                            importances = selector.estimator_.feature_importances_[selected_indices]
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(wavenumbers[selected_indices], importances, 'ro-', alpha=0.7)
                            ax.set_xlabel('波数 (cm⁻¹)')
                            ax.set_ylabel('特征重要性')
                            ax.set_title('选择特征的重要性分布')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # 显示选择的特征位置
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # 绘制平均光谱
                        mean_spectrum = np.mean(X, axis=0)
                        ax.plot(wavenumbers, mean_spectrum, 'b-', alpha=0.5, label='平均光谱')
                        
                        # 标记选择的特征
                        for idx in selected_indices:
                            ax.axvline(x=wavenumbers[idx], color='red', alpha=0.3)
                        
                        ax.set_xlabel('波数 (cm⁻¹)')
                        ax.set_ylabel('强度')
                        ax.set_title('选择的特征在光谱中的位置（红色竖线）')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # 显示选择特征的统计信息
                        st.subheader("📈 选择特征统计信息")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**选择的波数范围：**")
                            selected_wavenumbers = wavenumbers[selected_indices]
                            st.write(f"最小波数: {selected_wavenumbers.min():.2f} cm⁻¹")
                            st.write(f"最大波数: {selected_wavenumbers.max():.2f} cm⁻¹")
                            st.write(f"波数跨度: {selected_wavenumbers.max() - selected_wavenumbers.min():.2f} cm⁻¹")
                        
                        with col2:
                            st.write("**特征选择统计：**")
                            st.write(f"选择率: {len(selected_indices)/X.shape[1]*100:.1f}%")
                            st.write(f"数据压缩比: {X_selected.shape[1]/X.shape[1]*100:.1f}%")
                            
                            # 计算特征分布
                            total_range = wavenumbers.max() - wavenumbers.min()
                            selected_range = selected_wavenumbers.max() - selected_wavenumbers.min()
                            st.write(f"波数覆盖率: {selected_range/total_range*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"特征选择失败: {e}")
                        st.error(traceback.format_exc())

        # 显示当前特征选择状态
        if st.session_state.feature_selected:
            st.subheader("✅ 当前特征选择状态")
            
            method = st.session_state.feature_selection_method
            if method == "不进行特征选择":
                st.info("已选择使用全部预处理后的特征")
                st.write(f"最终特征数量: {st.session_state.X_final.shape[1]}")
            else:
                st.success(f"已完成特征选择，使用方法: {method}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终特征数", st.session_state.X_final.shape[1])
                with col2:
                    st.metric("原始特征数", X.shape[1])
                with col3:
                    reduction = X.shape[1] - st.session_state.X_final.shape[1]
                    st.metric("特征减少", reduction)

def show_data_split_page():
    """数据集划分页面"""
    st.markdown("<h1 class='section-header'>数据集划分</h1>", unsafe_allow_html=True)
    
    if not st.session_state.preprocessing_done:
        st.warning("请先完成数据预处理")
        return
    
    st.markdown("""
    <div class="info-box">
    将预处理后的数据划分为训练集和测试集，支持随机划分、K折交叉验证和留一法。
    </div>
    """, unsafe_allow_html=True)
    
    # 确定使用的数据（特征选择后的或预处理后的）
    if st.session_state.feature_selected:
        X = st.session_state.X_final
        wavenumbers = st.session_state.wavenumbers_final
        st.info(f"✅ 使用特征选择后的数据，特征数量: {X.shape[1]}")
    else:
        X = st.session_state.X_preprocessed
        wavenumbers = st.session_state.wavenumbers_preprocessed
        st.info(f"ℹ️ 使用预处理后的全部特征，特征数量: {X.shape[1]}")
    
    y = st.session_state.y
    
    # 划分方法选择
    st.subheader("📊 数据划分方法")
    split_method = st.radio(
        "选择数据划分方法",
        ["随机划分", "KFold交叉验证", "留一法(LOOCV)"]
    )
    
    if split_method == "随机划分":
        st.subheader("随机划分参数设置")
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("随机种子", value=42, min_value=0)
        
        if st.button("执行数据划分"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )
            
            # 保存划分结果
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.split_method = split_method
            
            st.success("✅ 数据划分完成！")
            
            # 显示划分结果
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("训练集样本数", X_train.shape[0])
            with col2:
                st.metric("测试集样本数", X_test.shape[0])
            with col3:
                st.metric("训练集比例", f"{(1-test_size)*100:.1f}%")
            with col4:
                st.metric("测试集比例", f"{test_size*100:.1f}%")
    
    elif split_method == "KFold交叉验证":
        st.subheader("K折交叉验证参数设置")
        cv_splits = st.slider("折数(K)", 3, 10, 5)
        random_state = st.number_input("随机种子", value=42, min_value=0)
        
        if st.button("设置交叉验证"):
            # 保存设置
            st.session_state.X_train = X
            st.session_state.X_test = None
            st.session_state.y_train = y
            st.session_state.y_test = None
            st.session_state.split_method = split_method
            st.session_state.cv_splits = cv_splits
            st.session_state.random_state = random_state
            
            st.success("✅ 交叉验证设置完成！")
            st.info(f"将使用 {cv_splits} 折交叉验证进行模型评估")
    
    elif split_method == "留一法(LOOCV)":
        st.subheader("留一法交叉验证")
        st.info("留一法将使用 N-1 个样本训练，1个样本测试，重复N次")
        
        if X.shape[0] > 100:
            st.warning("⚠️ 样本数量较多，留一法可能需要较长时间")
        
        if st.button("设置留一法验证"):
            # 保存设置
            st.session_state.X_train = X
            st.session_state.X_test = None
            st.session_state.y_train = y
            st.session_state.y_test = None
            st.session_state.split_method = split_method
            
            st.success("✅ 留一法设置完成！")
            st.info(f"将使用留一法对 {X.shape[0]} 个样本进行交叉验证")
    
    # 显示当前数据集状态
    if hasattr(st.session_state, 'split_method'):
        st.subheader("📈 当前数据集状态")
        
        if st.session_state.split_method == "随机划分":
            if hasattr(st.session_state, 'X_train'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**训练集信息：**")
                    st.write(f"样本数: {st.session_state.X_train.shape[0]}")
                    st.write(f"特征数: {st.session_state.X_train.shape[1]}")
                    
                with col2:
                    st.write("**测试集信息：**")
                    st.write(f"样本数: {st.session_state.X_test.shape[0]}")
                    st.write(f"特征数: {st.session_state.X_test.shape[1]}")
                
                # 显示目标变量分布对比
                if len(st.session_state.selected_cols) == 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # 训练集分布
                    ax1.hist(st.session_state.y_train.flatten(), bins=20, alpha=0.7, color='blue')
                    ax1.set_title('训练集目标变量分布')
                    ax1.set_xlabel(st.session_state.selected_cols[0])
                    ax1.set_ylabel('频数')
                    ax1.grid(True, alpha=0.3)
                    
                    # 测试集分布
                    ax2.hist(st.session_state.y_test.flatten(), bins=20, alpha=0.7, color='orange')
                    ax2.set_title('测试集目标变量分布')
                    ax2.set_xlabel(st.session_state.selected_cols[0])
                    ax2.set_ylabel('频数')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.info(f"已设置 {st.session_state.split_method}，将在模型训练时使用")

def show_model_training_page():
    """模型训练与评估页面"""
    st.markdown("<h1 class='section-header'>模型训练与评估</h1>", unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'split_method'):
        st.warning("请先完成数据集划分")
        return
    
    st.markdown("""
    <div class="info-box">
    选择机器学习算法进行模型训练和评估，支持多种回归算法和参数调优。
    </div>
    """, unsafe_allow_html=True)
    
    # 显示当前使用的数据信息
    if st.session_state.feature_selected:
        st.info(f"✅ 使用特征选择后的数据进行建模 - 特征数量: {st.session_state.X_final.shape[1]}")
        if st.session_state.feature_selection_method != "不进行特征选择":
            st.info(f"特征选择方法: {st.session_state.feature_selection_method}")
    else:
        st.info(f"ℹ️ 使用预处理后的全部特征进行建模 - 特征数量: {st.session_state.X_preprocessed.shape[1]}")
    
    # 可用模型
    available_models = {
        'linear': '线性回归',
        'ridge': '岭回归',
        'lasso': 'Lasso回归',
        'svr': '支持向量回归',
        'rf': '随机森林',
        'gbr': '梯度提升回归',
        'mlp': '多层感知机',
        'pls': '偏最小二乘回归'
    }
    
    try:
        import xgboost as xgb
        available_models['xgb'] = 'XGBoost'
    except ImportError:
        pass
    
    # 检查是否为多输出问题
    is_multioutput = len(st.session_state.selected_cols) > 1
    
    # 模型选择
    st.subheader("🤖 模型选择与参数设置")
    selected_models = st.multiselect("选择要训练的模型", list(available_models.keys()), 
                                   format_func=lambda x: available_models[x])
    
    if not selected_models:
        st.warning("请至少选择一个模型")
        return
    
    # 模型参数设置
    model_params = {}
    
    for i, model_name in enumerate(selected_models):
        st.subheader(f"⚙️ {available_models[model_name]} 参数设置")
        
        if model_name == 'linear':
            # 线性回归参数
            fit_intercept = st.checkbox("拟合截距", value=True, key=f"linear_intercept_{i}")
            use_scaler = st.checkbox("使用标准化", value=True, key=f"linear_scaler_{i}")
            
            model_params['linear'] = {
                'fit_intercept': fit_intercept,
                'use_scaler': use_scaler
            }
        
        elif model_name == 'ridge':
            # 岭回归参数
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.selectbox("正则化参数α", [0.01, 0.1, 1.0, 10.0, 100.0], 
                                   index=2, key=f"ridge_alpha_{i}")
                fit_intercept = st.checkbox("拟合截距", value=True, key=f"ridge_intercept_{i}")
            with col2:
                solver = st.selectbox("求解器", ['auto', 'svd', 'cholesky', 'lsqr'], 
                                    index=0, key=f"ridge_solver_{i}")
                use_scaler = st.checkbox("使用标准化", value=True, key=f"ridge_scaler_{i}")
            
            model_params['ridge'] = {
                'alpha': alpha,
                'fit_intercept': fit_intercept,
                'solver': solver,
                'use_scaler': use_scaler
            }
        
        elif model_name == 'lasso':
            # Lasso回归参数
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.selectbox("正则化参数α", [0.01, 0.1, 1.0, 10.0], 
                                   index=1, key=f"lasso_alpha_{i}")
                fit_intercept = st.checkbox("拟合截距", value=True, key=f"lasso_intercept_{i}")
            with col2:
                max_iter = st.slider("最大迭代次数", 100, 2000, 1000, key=f"lasso_iter_{i}")
                use_scaler = st.checkbox("使用标准化", value=True, key=f"lasso_scaler_{i}")
            
            model_params['lasso'] = {
                'alpha': alpha,
                'fit_intercept': fit_intercept,
                'max_iter': max_iter,
                'use_scaler': use_scaler
            }
        
        elif model_name == 'svr':
            # SVR参数
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("核函数", ['rbf', 'linear', 'poly'], 
                                    index=0, key=f"svr_kernel_{i}")
                C = st.selectbox("惩罚参数C", [0.1, 1.0, 10.0, 100.0], 
                               index=1, key=f"svr_c_{i}")
            with col2:
                gamma = st.selectbox("核参数γ", ['scale', 'auto'], 
                                   index=0, key=f"svr_gamma_{i}")
                epsilon = st.selectbox("ε参数", [0.01, 0.1, 0.2], 
                                     index=1, key=f"svr_epsilon_{i}")
            
            model_params['svr'] = {
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                'epsilon': epsilon
            }
        
        elif model_name == 'rf':
            # 随机森林参数
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("树的数量", 50, 500, 100, key=f"rf_trees_{i}")
                max_depth = st.selectbox("最大深度", [None, 5, 10, 15, 20], 
                                       index=0, key=f"rf_depth_{i}")
            with col2:
                min_samples_split = st.slider("分裂最小样本数", 2, 10, 2, key=f"rf_split_{i}")
                min_samples_leaf = st.slider("叶节点最小样本数", 1, 5, 1, key=f"rf_leaf_{i}")
            
            random_state = st.number_input("随机种子", value=42, key=f"rf_seed_{i}")
            
            model_params['rf'] = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': random_state
            }
        
        elif model_name == 'gbr':
            # 梯度提升参数
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("提升阶段数", 50, 500, 100, key=f"gbr_stages_{i}")
                learning_rate = st.selectbox("学习率", [0.01, 0.05, 0.1, 0.2], 
                                           index=2, key=f"gbr_lr_{i}")
            with col2:
                max_depth = st.slider("最大深度", 2, 10, 3, key=f"gbr_depth_{i}")
                subsample = st.slider("子采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"gbr_subsample_{i}")
            
            random_state = st.number_input("随机种子", value=42, key=f"gbr_seed_{i}")
            
            model_params['gbr'] = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'random_state': random_state
            }
        
        elif model_name == 'pls':
            # PLS参数
            n_components = st.slider("主成分数量", 1, min(20, st.session_state.X_train.shape[1]), 
                                   5, key=f"pls_components_{i}")
            scale = st.checkbox("标准化", value=True, key=f"pls_scale_{i}")
            
            model_params['pls'] = {
                'n_components': n_components,
                'scale': scale
            }
        
        elif model_name == 'mlp':
            # MLP参数
            col1, col2 = st.columns(2)
            with col1:
                layer_option = st.selectbox("隐藏层结构", ["一层", "两层", "三层"], 
                                          index=1, key=f"mlp_layers_{i}")
                
                if layer_option == "一层":
                    layer1_size = st.slider("隐藏层神经元数", 10, 200, 50, key=f"mlp_l1_{i}")
                    hidden_layer_sizes = (layer1_size,)
                elif layer_option == "两层":
                    layer1_size = st.slider("第一层神经元数", 10, 200, 100, key=f"mlp_l1_{i}")
                    layer2_size = st.slider("第二层神经元数", 10, 100, 50, key=f"mlp_l2_{i}")
                    hidden_layer_sizes = (layer1_size, layer2_size)
                else:  # 三层
                    layer1_size = st.slider("第一层神经元数", 10, 200, 100, key=f"mlp_l1_{i}")
                    layer2_size = st.slider("第二层神经元数", 10, 100, 50, key=f"mlp_l2_{i}")
                    layer3_size = st.slider("第三层神经元数", 10, 50, 25, key=f"mlp_l3_{i}")
                    hidden_layer_sizes = (layer1_size, layer2_size, layer3_size)
                
                activation = st.selectbox("激活函数", ['relu', 'tanh', 'logistic'], 
                                        index=0, key=f"mlp_activation_{i}")
            
            with col2:
                solver = st.selectbox("优化算法", ['adam', 'lbfgs', 'sgd'], 
                                    index=0, key=f"mlp_solver_{i}")
                learning_rate_init = st.selectbox("初始学习率", [0.0001, 0.001, 0.01], 
                                                index=1, key=f"mlp_lr_{i}")
                max_iter = st.slider("最大迭代次数", 100, 1000, 500, key=f"mlp_iter_{i}")
                alpha = st.selectbox("L2正则化参数", [0.0001, 0.001, 0.01], 
                                   index=0, key=f"mlp_alpha_{i}")
            
            random_state = st.number_input("随机种子", value=42, key=f"mlp_seed_{i}")
            
            model_params['mlp'] = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'learning_rate_init': learning_rate_init,
                'max_iter': max_iter,
                'alpha': alpha,
                'random_state': random_state
            }
        
        elif model_name == 'xgb':
            # XGBoost参数
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("提升轮数", 50, 500, 100, key=f"xgb_trees_{i}")
                learning_rate = st.selectbox("学习率", [0.01, 0.05, 0.1, 0.2], 
                                           index=2, key=f"xgb_lr_{i}")
                max_depth = st.slider("最大深度", 2, 10, 6, key=f"xgb_depth_{i}")
            with col2:
                subsample = st.slider("子采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"xgb_subsample_{i}")
                colsample_bytree = st.slider("特征采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"xgb_colsample_{i}")
                reg_alpha = st.selectbox("L1正则化", [0, 0.01, 0.1], index=0, key=f"xgb_alpha_{i}")
            
            random_state = st.number_input("随机种子", value=42, key=f"xgb_seed_{i}")
            
            model_params['xgb'] = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'random_state': random_state
            }
    
    # 交叉验证设置显示
    if st.session_state.split_method in ["KFold交叉验证", "留一法(LOOCV)"]:
        use_cv = True
        if st.session_state.split_method == "KFold交叉验证":
            cv_folds = getattr(st.session_state, 'cv_splits', 5)
            st.info(f"将使用 {cv_folds} 折交叉验证")
        else:
            st.info("将使用留一法交叉验证")
    else:
        use_cv = False
        st.info("将使用训练集/测试集划分")
    
    # 开始训练
    if st.button("🚀 开始训练模型", type="primary"):
        if not selected_models:
            st.error("请至少选择一个模型！")
            return
        
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        
        if st.session_state.split_method == "随机划分":
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
        else:
            X_test = X_train
            y_test = y_train
        
        results = []
        trained_models = {}
        detailed_results = {}
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            progress_text.text(f"正在训练 {available_models[model_name]} ({i+1}/{len(selected_models)})...")
            
            try:
                # 创建模型
                params = model_params.get(model_name, {})
                
                # 处理标准化
                use_scaler = params.pop('use_scaler', False)
                scaler = None
                X_train_scaled = X_train
                X_test_scaled = X_test
                
                if use_scaler and model_name in ['ridge', 'lasso', 'linear']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                
                # 创建具体模型
                if model_name == 'linear':
                    model = LinearRegression(**params)
                elif model_name == 'ridge':
                    base_model = Ridge(**params)
                    model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                elif model_name == 'lasso':
                    base_model = Lasso(**params)
                    model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                elif model_name == 'svr':
                    base_model = SVR(**params)
                    model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                elif model_name == 'rf':
                    model = RandomForestRegressor(**params)
                elif model_name == 'gbr':
                    base_model = GradientBoostingRegressor(**params)
                    model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                elif model_name == 'pls':
                    model = PLSRegression(**params)
                elif model_name == 'mlp':
                    base_model = MLPRegressor(**params)
                    model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                elif model_name == 'xgb':
                    try:
                        from xgboost import XGBRegressor
                        base_model = XGBRegressor(**params)
                        model = MultiOutputRegressor(base_model) if is_multioutput else base_model
                    except ImportError:
                        st.error("XGBoost未安装，跳过该模型")
                        continue
                
                # 训练模型
                start_time = time.time()
                
                if use_cv:
                    # 交叉验证
                    if st.session_state.split_method == "KFold交叉验证":
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    else:  # LOOCV
                        cv = LeaveOneOut()
                    
                    # 执行交叉验证
                    cv_predictions = np.zeros_like(y_train)
                    cv_scores = []
                    
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                        X_fold_train = X_train_scaled[train_idx]
                        X_fold_val = X_train_scaled[val_idx]
                        y_fold_train = y_train[train_idx]
                        y_fold_val = y_train[val_idx]
                        
                        # 训练
                        model.fit(X_fold_train, y_fold_train)
                        
                        # 预测
                        fold_pred = model.predict(X_fold_val)
                        cv_predictions[val_idx] = fold_pred
                        
                        # 计算fold得分
                        if is_multioutput:
                            fold_score = np.mean([r2_score(y_fold_val[:, j], fold_pred[:, j]) 
                                                for j in range(y_fold_val.shape[1])])
                        else:
                            fold_score = r2_score(y_fold_val, fold_pred)
                        cv_scores.append(fold_score)
                    
                    # 用全部数据重新训练
                    model.fit(X_train_scaled, y_train)
                    train_pred = cv_predictions
                    test_pred = model.predict(X_test_scaled)
                    
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                else:
                    # 普通训练
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    cv_mean = cv_std = None
                
                train_time = time.time() - start_time
                
                # 计算评估指标
                if is_multioutput:
                    # 多输出
                    train_r2_scores = [r2_score(y_train[:, j], train_pred[:, j]) for j in range(y_train.shape[1])]
                    test_r2_scores = [r2_score(y_test[:, j], test_pred[:, j]) for j in range(y_test.shape[1])]
                    train_rmse_scores = [np.sqrt(mean_squared_error(y_train[:, j], train_pred[:, j])) for j in range(y_train.shape[1])]
                    test_rmse_scores = [np.sqrt(mean_squared_error(y_test[:, j], test_pred[:, j])) for j in range(y_test.shape[1])]
                    train_mae_scores = [mean_absolute_error(y_train[:, j], train_pred[:, j]) for j in range(y_train.shape[1])]
                    test_mae_scores = [mean_absolute_error(y_test[:, j], test_pred[:, j]) for j in range(y_test.shape[1])]
                    
                    train_r2 = np.mean(train_r2_scores)
                    test_r2 = np.mean(test_r2_scores)
                    train_rmse = np.mean(train_rmse_scores)
                    test_rmse = np.mean(test_rmse_scores)
                    train_mae = np.mean(train_mae_scores)
                    test_mae = np.mean(test_mae_scores)
                    
                    detailed_results[model_name] = {
                        'train_r2_per_target': train_r2_scores,
                        'test_r2_per_target': test_r2_scores,
                        'train_rmse_per_target': train_rmse_scores,
                        'test_rmse_per_target': test_rmse_scores,
                        'train_mae_per_target': train_mae_scores,
                        'test_mae_per_target': test_mae_scores,
                        'train_pred': train_pred,
                        'test_pred': test_pred,
                        'params': params,
                        'scaler': scaler
                    }
                else:
                    # 单输出
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    train_mae = mean_absolute_error(y_train, train_pred)
                    test_mae = mean_absolute_error(y_test, test_pred)
                    
                    detailed_results[model_name] = {
                        'train_pred': train_pred,
                        'test_pred': test_pred,
                        'params': params,
                        'scaler': scaler
                    }
                
                # 保存结果
                result_entry = {
                    'Model': available_models[model_name],
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Train RMSE': train_rmse,
                    'Test RMSE': test_rmse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Training Time (s)': train_time
                }
                
                if use_cv:
                    result_entry['CV R² Mean'] = cv_mean
                    result_entry['CV R² Std'] = cv_std
                
                results.append(result_entry)
                trained_models[model_name] = model
                
                progress_bar.progress((i + 1) / len(selected_models))
                
            except Exception as e:
                st.error(f"训练模型 {available_models[model_name]} 时出错: {e}")
                st.error(traceback.format_exc())
                continue
        
        progress_text.text("所有模型训练完成！")
        
        if results:
            # 显示结果
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Test R²', ascending=False)
            
            st.session_state.trained_models = trained_models
            st.session_state.results_df = results_df
            st.session_state.detailed_results = detailed_results
            
            st.success("🎉 模型训练与评估完成！")
            
            # 显示性能比较表格
            st.subheader("📊 模型性能比较")
            
            # 格式化显示
            display_df = results_df.copy()
            numeric_cols = ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'Training Time (s)']
            if use_cv:
                numeric_cols.extend(['CV R² Mean', 'CV R² Std'])
            
            for col in numeric_cols:
                if col in display_df.columns:
                    if 'Time' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}s")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # 显示最佳模型
            best_model_idx = results_df['Test R²'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'Model']
            st.success(f"🏆 最佳模型: {best_model_name} (Test R² = {results_df.loc[best_model_idx, 'Test R²']:.4f})")
            
            # 显示模型性能可视化
            show_model_performance_visualization(results_df, detailed_results, is_multioutput)
            
        else:
            st.error("❌ 没有成功训练任何模型，请检查数据和参数设置")

def show_model_performance_visualization(results_df, detailed_results, is_multioutput):
    """显示模型性能可视化"""
    st.subheader("📈 模型性能可视化")
    
    # 创建标签页
    if is_multioutput:
        tab1, tab2, tab3 = st.tabs(["性能对比", "预测效果", "目标变量详情"])
    else:
        tab1, tab2 = st.tabs(["性能对比", "预测效果"])
    
    with tab1:
        # 性能对比图
        col1, col2 = st.columns(2)
        
        with col1:
            # R²对比
            fig, ax = plt.subplots(figsize=(10, 6))
            models = results_df['Model']
            test_r2 = results_df['Test R²'].astype(float)
            train_r2 = results_df['Train R²'].astype(float)
            
            y_pos = np.arange(len(models))
            width = 0.35
            
            ax.barh(y_pos - width/2, train_r2, width, label='训练 R²', alpha=0.7, color='skyblue')
            ax.barh(y_pos + width/2, test_r2, width, label='测试 R²', alpha=0.7, color='lightcoral')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_xlabel('R² 分数')
            ax.set_title('模型 R² 性能对比')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加数值标签
            for i, (train, test) in enumerate(zip(train_r2, test_r2)):
                ax.text(train + 0.01, i - width/2, f'{train:.3f}', va='center', fontsize=8)
                ax.text(test + 0.01, i + width/2, f'{test:.3f}', va='center', fontsize=8)
            
            st.pyplot(fig)
        
        with col2:
            # RMSE对比
            fig, ax = plt.subplots(figsize=(10, 6))
            test_rmse = results_df['Test RMSE'].astype(float)
            train_rmse = results_df['Train RMSE'].astype(float)
            
            ax.barh(y_pos - width/2, train_rmse, width, label='训练 RMSE', alpha=0.7, color='lightgreen')
            ax.barh(y_pos + width/2, test_rmse, width, label='测试 RMSE', alpha=0.7, color='orange')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_xlabel('RMSE')
            ax.set_title('模型 RMSE 性能对比')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加数值标签
            for i, (train, test) in enumerate(zip(train_rmse, test_rmse)):
                ax.text(train + max(train_rmse) * 0.01, i - width/2, f'{train:.3f}', va='center', fontsize=8)
                ax.text(test + max(test_rmse) * 0.01, i + width/2, f'{test:.3f}', va='center', fontsize=8)
            
            st.pyplot(fig)
    
    with tab2:
        # 预测效果散点图
        st.write("### 🎯 预测 vs 实际值对比")
        
        # 选择模型
        model_names = list(detailed_results.keys())
        available_models = {
            'linear': '线性回归', 'ridge': '岭回归', 'lasso': 'Lasso回归',
            'svr': '支持向量回归', 'rf': '随机森林', 'gbr': '梯度提升回归',
            'mlp': '多层感知机', 'pls': '偏最小二乘回归', 'xgb': 'XGBoost'
        }
        
        selected_model = st.selectbox(
            "选择模型查看预测效果", 
            model_names, 
            format_func=lambda x: available_models.get(x, x)
        )
        
        if selected_model in detailed_results:
            model_results = detailed_results[selected_model]
            
            # 获取实际值
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test if st.session_state.y_test is not None else y_train
            
            if is_multioutput:
                # 多输出情况
                target_names = st.session_state.selected_cols
                target_idx = st.selectbox(
                    "选择目标变量", 
                    range(len(target_names)), 
                    format_func=lambda x: target_names[x]
                )
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 训练集预测
                y_train_actual = y_train[:, target_idx]
                y_train_pred = model_results['train_pred'][:, target_idx]
                
                ax1.scatter(y_train_actual, y_train_pred, alpha=0.6, color='blue')
                ax1.plot([y_train_actual.min(), y_train_actual.max()], 
                        [y_train_actual.min(), y_train_actual.max()], 'r--', lw=2)
                ax1.set_xlabel('实际值')
                ax1.set_ylabel('预测值')
                ax1.set_title(f'训练集 - {target_names[target_idx]}')
                ax1.grid(True, alpha=0.3)
                
                # 添加R²和RMSE
                r2 = r2_score(y_train_actual, y_train_pred)
                rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
                ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 测试集预测
                y_test_actual = y_test[:, target_idx]
                y_test_pred = model_results['test_pred'][:, target_idx]
                
                ax2.scatter(y_test_actual, y_test_pred, alpha=0.6, color='green')
                ax2.plot([y_test_actual.min(), y_test_actual.max()], 
                        [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
                ax2.set_xlabel('实际值')
                ax2.set_ylabel('预测值')
                ax2.set_title(f'测试集 - {target_names[target_idx]}')
                ax2.grid(True, alpha=0.3)
                
                # 添加R²和RMSE
                r2 = r2_score(y_test_actual, y_test_pred)
                rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
                ax2.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                # 单输出情况
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 训练集
                ax1.scatter(y_train, model_results['train_pred'], alpha=0.6, color='blue')
                ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
                ax1.set_xlabel('实际值')
                ax1.set_ylabel('预测值')
                ax1.set_title('训练集预测效果')
                ax1.grid(True, alpha=0.3)
                
                r2 = r2_score(y_train, model_results['train_pred'])
                rmse = np.sqrt(mean_squared_error(y_train, model_results['train_pred']))
                ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 测试集
                ax2.scatter(y_test, model_results['test_pred'], alpha=0.6, color='green')
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax2.set_xlabel('实际值')
                ax2.set_ylabel('预测值')
                ax2.set_title('测试集预测效果')
                ax2.grid(True, alpha=0.3)
                
                r2 = r2_score(y_test, model_results['test_pred'])
                rmse = np.sqrt(mean_squared_error(y_test, model_results['test_pred']))
                ax2.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
    
    if is_multioutput:
        with tab3:
            # 各目标变量的详细性能
            st.write("### 📊 各目标变量性能详情")
            
            # 选择模型
            selected_model = st.selectbox(
                "选择模型", 
                model_names, 
                format_func=lambda x: available_models.get(x, x),
                key="detail_model_select"
            )
            
            if selected_model in detailed_results:
                model_results = detailed_results[selected_model]
                target_names = st.session_state.selected_cols
                
                # 创建详细结果表格
                target_results = []
                for i, target_name in enumerate(target_names):
                    target_results.append({
                        '目标变量': target_name,
                        '训练 R²': f"{model_results['train_r2_per_target'][i]:.4f}",
                        '测试 R²': f"{model_results['test_r2_per_target'][i]:.4f}",
                        '训练 RMSE': f"{model_results['train_rmse_per_target'][i]:.4f}",
                        '测试 RMSE': f"{model_results['test_rmse_per_target'][i]:.4f}",
                        '训练 MAE': f"{model_results['train_mae_per_target'][i]:.4f}",
                        '测试 MAE': f"{model_results['test_mae_per_target'][i]:.4f}"
                    })
                
                target_df = pd.DataFrame(target_results)
                st.dataframe(target_df, use_container_width=True)
                
                # 可视化各目标变量性能
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # R²对比
                ax1.bar(target_names, model_results['train_r2_per_target'], 
                       alpha=0.7, label='训练 R²', color='skyblue')
                ax1.bar(target_names, model_results['test_r2_per_target'], 
                       alpha=0.7, label='测试 R²', color='lightcoral')
                ax1.set_ylabel('R²')
                ax1.set_title('各目标变量 R² 对比')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # RMSE对比
                ax2.bar(target_names, model_results['train_rmse_per_target'], 
                       alpha=0.7, label='训练 RMSE', color='lightgreen')
                ax2.bar(target_names, model_results['test_rmse_per_target'], 
                       alpha=0.7, label='测试 RMSE', color='orange')
                ax2.set_ylabel('RMSE')
                ax2.set_title('各目标变量 RMSE 对比')
                ax2.legend()
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # MAE对比
                ax3.bar(target_names, model_results['train_mae_per_target'], 
                       alpha=0.7, label='训练 MAE', color='plum')
                ax3.bar(target_names, model_results['test_mae_per_target'], 
                       alpha=0.7, label='测试 MAE', color='gold')
                ax3.set_ylabel('MAE')
                ax3.set_title('各目标变量 MAE 对比')
                ax3.legend()
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # 综合性能雷达图
                from math import pi
                
                # 归一化性能指标
                r2_norm = np.array(model_results['test_r2_per_target'])
                rmse_norm = 1 - np.array(model_results['test_rmse_per_target']) / max(model_results['test_rmse_per_target'])
                mae_norm = 1 - np.array(model_results['test_mae_per_target']) / max(model_results['test_mae_per_target'])
                
                # 雷达图
                angles = [n / len(target_names) * 2 * pi for n in range(len(target_names))]
                angles += angles[:1]
                
                ax4 = plt.subplot(224, projection='polar')
                
                r2_values = list(r2_norm) + [r2_norm[0]]
                rmse_values = list(rmse_norm) + [rmse_norm[0]]
                mae_values = list(mae_norm) + [mae_norm[0]]
                
                ax4.plot(angles, r2_values, 'o-', linewidth=2, label='R² (标准化)', color='blue')
                ax4.fill(angles, r2_values, alpha=0.25, color='blue')
                
                ax4.plot(angles, rmse_values, 'o-', linewidth=2, label='RMSE (标准化)', color='red')
                ax4.fill(angles, rmse_values, alpha=0.25, color='red')
                
                ax4.plot(angles, mae_values, 'o-', linewidth=2, label='MAE (标准化)', color='green')
                ax4.fill(angles, mae_values, alpha=0.25, color='green')
                
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(target_names)
                ax4.set_ylim(0, 1)
                ax4.set_title('综合性能雷达图', pad=20)
                ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                plt.tight_layout()
                st.pyplot(fig)

def show_blind_prediction_page():
    """盲样预测页面"""
    st.markdown("<h1 class='section-header'>盲样预测</h1>", unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'trained_models') or not st.session_state.trained_models:
        st.warning("请先训练模型")
        return
    
    if st.session_state.X_preprocessed is None:
        st.warning("请先完成数据预处理")
        return
    
    st.markdown("""
    <div class="info-box">
    上传盲样数据文件进行预测。盲样数据将使用与训练数据相同的预处理流程。
    </div>
    """, unsafe_allow_html=True)
    
    # 显示预处理参数信息
    st.subheader("当前预处理参数")
    params = st.session_state.preprocessing_params
    
    with st.expander("查看详细参数"):
        st.json(params)
    
    # 显示特征选择信息
    if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
        if st.session_state.feature_selection_method != "不进行特征选择":
            st.info(f"✅ 已进行特征选择：{st.session_state.feature_selection_method}")
            st.info(f"特征数量：{st.session_state.X_preprocessed.shape[1]} → {st.session_state.X_final.shape[1]}")
        else:
            st.info("ℹ️ 未进行特征选择，使用全部预处理后的特征")
    
    # 上传盲样数据文件
    blind_file = st.file_uploader("上传盲样数据文件", type=["csv", "xlsx", "xls"])
    
    if blind_file is not None:
        try:
            # 读取盲样数据
            if blind_file.name.endswith('.csv'):
                blind_df = pd.read_csv(blind_file)
            else:
                blind_df = pd.read_excel(blind_file)
            
            st.success(f"盲样数据上传成功！共{blind_df.shape[0]}行，{blind_df.shape[1]}列")
            st.dataframe(blind_df.head())
            
            # 检查数据格式
            st.subheader("数据格式检查")
            
            # 识别波数列
            try:
                potential_wavenumbers = blind_df.columns[2:]
                numeric_columns = []
                
                for col in potential_wavenumbers:
                    try:
                        float(col)
                        numeric_columns.append(col)
                    except ValueError:
                        continue
                
                if len(numeric_columns) < 10:
                    st.error("检测到的波数列数量不足，请检查数据格式")
                    return
                
                blind_wavenumbers = pd.Series(numeric_columns).astype(float)
                st.info(f"盲样数据波数范围: {blind_wavenumbers.min():.1f} ~ {blind_wavenumbers.max():.1f} cm⁻¹")
                
                # 与训练数据的波数范围对比
                train_wavenumbers = st.session_state.wavenumbers
                st.info(f"训练数据波数范围: {train_wavenumbers.min():.1f} ~ {train_wavenumbers.max():.1f} cm⁻¹")
                
                # 检查波数范围兼容性
                if (blind_wavenumbers.min() > train_wavenumbers.min() or 
                    blind_wavenumbers.max() < train_wavenumbers.max()):
                    st.warning("⚠️ 盲样数据的波数范围与训练数据不完全匹配，可能影响预测精度")
                
                # 提取盲样光谱数据
                blind_spectra = blind_df[numeric_columns].values.astype(float)
                st.info(f"原始盲样光谱形状: {blind_spectra.shape}")
                
                # 选择要使用的模型
                st.subheader("模型选择")
                model_names = list(st.session_state.trained_models.keys())
                available_models = {
                    'linear': '线性回归', 'ridge': '岭回归', 'lasso': 'Lasso回归',
                    'svr': '支持向量回归', 'rf': '随机森林', 'gbr': '梯度提升回归',
                    'mlp': '多层感知机', 'pls': '偏最小二乘回归', 'xgb': 'XGBoost'
                }
                
                selected_model_key = st.selectbox(
                    "选择预测模型", 
                    model_names,
                    format_func=lambda x: available_models.get(x, x)
                )
                
                if st.button("进行预测"):
                    with st.spinner("正在应用预处理流程并进行预测..."):
                        try:
                            # 应用预处理流程
                            st.write("**预处理步骤:**")
                            
                            # 1. 截取波数范围
                            start_wn = params['start_wavenumber']
                            end_wn = params['end_wavenumber']
                            
                            start_idx = np.argmin(np.abs(blind_wavenumbers - start_wn))
                            end_idx = np.argmin(np.abs(blind_wavenumbers - end_wn)) + 1
                            
                            blind_wavenumbers_crop = blind_wavenumbers[start_idx:end_idx]
                            blind_X_crop = blind_spectra[:, start_idx:end_idx]
                            
                            st.write(f"✓ 波数截取: {start_wn} ~ {end_wn} cm⁻¹, 形状: {blind_X_crop.shape}")
                            
                            # 2. 平滑处理
                            if params.get('apply_smooth', True):
                                blind_X_smooth = np.zeros_like(blind_X_crop)
                                for i in range(blind_X_crop.shape[0]):
                                    blind_X_smooth[i] = savgol_filter(
                                        blind_X_crop[i], 
                                        params['smooth_window'], 
                                        params['smooth_poly']
                                    )
                                st.write(f"✓ Savitzky-Golay平滑: 窗口={params['smooth_window']}, 多项式阶数={params['smooth_poly']}")
                            else:
                                blind_X_smooth = blind_X_crop.copy()
                                st.write("○ 跳过平滑处理")
                            
                            # 3. 基线校正
                            if params.get('apply_baseline', True):
                                corrector = SpectrumBaselineCorrector()
                                blind_X_corr = np.zeros_like(blind_X_smooth)
                                
                                baseline_method = params['baseline_method']
                                baseline_params = params['baseline_params']
                                
                                for i in range(blind_X_smooth.shape[0]):
                                    try:
                                        baseline, corrected = corrector.correct_baseline(
                                            blind_X_smooth[i], 
                                            baseline_method, 
                                            **baseline_params
                                        )
                                        blind_X_corr[i] = corrected
                                    except Exception as e:
                                        st.warning(f"样本 {i+1} 基线校正失败，使用平滑后的数据: {e}")
                                        blind_X_corr[i] = blind_X_smooth[i]
                                
                                st.write(f"✓ 基线校正: {baseline_method.upper()}算法")
                            else:
                                blind_X_corr = blind_X_smooth.copy()
                                st.write("○ 跳过基线校正")
                            
                            # 4. 归一化
                            if params.get('apply_normalize', True):
                                blind_X_norm = np.zeros_like(blind_X_corr)
                                normalize_method = params['normalize_method']
                                
                                for i in range(blind_X_corr.shape[0]):
                                    if normalize_method == 'area':
                                        total_area = np.trapz(np.abs(blind_X_corr[i]), blind_wavenumbers_crop)
                                        blind_X_norm[i] = blind_X_corr[i] / (total_area if total_area != 0 else 1e-9)
                                    elif normalize_method == 'max':
                                        max_val = np.max(np.abs(blind_X_corr[i]))
                                        blind_X_norm[i] = blind_X_corr[i] / (max_val if max_val != 0 else 1e-9)
                                    elif normalize_method == 'vector':
                                        norm_val = np.linalg.norm(blind_X_corr[i])
                                        blind_X_norm[i] = blind_X_corr[i] / (norm_val if norm_val != 0 else 1e-9)
                                    elif normalize_method == 'minmax':
                                        min_val, max_val = np.min(blind_X_corr[i]), np.max(blind_X_corr[i])
                                        if max_val - min_val == 0:
                                            blind_X_norm[i] = blind_X_corr[i]
                                        else:
                                            blind_X_norm[i] = (blind_X_corr[i] - min_val) / (max_val - min_val)
                                
                                st.write(f"✓ 归一化: {normalize_method}方法")
                            else:
                                blind_X_norm = blind_X_corr.copy()
                                st.write("○ 跳过归一化")
                            
                            # 5. SNV处理
                            if params.get('apply_snv', False):
                                blind_X_preprocessed = np.zeros_like(blind_X_norm)
                                for i, spectrum in enumerate(blind_X_norm):
                                    mean_val = np.mean(spectrum)
                                    std_val = np.std(spectrum)
                                    if std_val == 0:
                                        blind_X_preprocessed[i] = spectrum
                                    else:
                                        blind_X_preprocessed[i] = (spectrum - mean_val) / std_val
                                st.write("✓ 标准正态变量变换(SNV): 已应用")
                            else:
                                blind_X_preprocessed = blind_X_norm
                                st.write("○ 标准正态变量变换(SNV): 未应用")
                            
                            # ⭐ 关键修复：应用特征选择 ⭐
                            if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
                                if st.session_state.feature_selection_method != "不进行特征选择":
                                    # 应用训练时的特征选择
                                    selected_indices = st.session_state.selected_features
                                    blind_X_final = blind_X_preprocessed[:, selected_indices]
                                    st.write(f"✓ 应用特征选择: {st.session_state.feature_selection_method}")
                                    st.write(f"特征数量: {blind_X_preprocessed.shape[1]} → {blind_X_final.shape[1]}")
                                else:
                                    blind_X_final = blind_X_preprocessed
                                    st.write("○ 未进行特征选择")
                            else:
                                blind_X_final = blind_X_preprocessed
                                st.write("○ 未进行特征选择")
                            
                            # 确定期望的特征数量
                            if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
                                expected_features = st.session_state.X_final.shape[1]
                                data_source = "特征选择后"
                            else:
                                expected_features = st.session_state.X_preprocessed.shape[1]
                                data_source = "预处理后"
                            
                            st.write(f"**{data_source}特征数量**: {blind_X_final.shape[1]}")
                            st.write(f"**模型期望特征数量**: {expected_features}")
                            
                            if blind_X_final.shape[1] != expected_features:
                                st.error(f"特征数量不匹配！盲样: {blind_X_final.shape[1]}, 期望: {expected_features}")
                                
                                if blind_X_final.shape[1] > expected_features:
                                    st.warning("盲样特征数量过多，将截取前面的特征")
                                    blind_X_final = blind_X_final[:, :expected_features]
                                    st.info(f"已调整为: {blind_X_final.shape[1]} 个特征")
                                else:
                                    st.error("盲样特征数量不足，无法自动调整")
                                    st.error("可能的原因：")
                                    st.error("1. 盲样数据的波数范围与训练数据不匹配")
                                    st.error("2. 预处理参数设置不同")
                                    st.error("3. 特征选择过程有差异")
                                    return
                            
                            # 应用标准化（如果需要）
                            if selected_model_key in st.session_state.detailed_results:
                                scaler = st.session_state.detailed_results[selected_model_key].get('scaler')
                                if scaler is not None:
                                    blind_X_final = scaler.transform(blind_X_final)
                                    st.write("✓ 应用训练时的标准化变换")
                            
                            # 使用选定的模型进行预测
                            model = st.session_state.trained_models[selected_model_key]
                            predictions = model.predict(blind_X_final)
                            
                            # 显示预测结果
                            st.subheader("预测结果")

                            # 1. 构建结果DataFrame，并确保列的顺序正确
                            result_df = pd.DataFrame()

                            # 2. 从原始盲样数据中提取标识列 (保留原始列名)
                            if blind_df.shape[1] >= 1:
                                result_df[blind_df.columns[0]] = blind_df.iloc[:, 0]
                            if blind_df.shape[1] >= 2:
                                result_df[blind_df.columns[1]] = blind_df.iloc[:, 1]

                            # 3. 添加预测结果
                            if predictions.ndim == 1:
                                # 单目标预测
                                pred_col_name = st.session_state.selected_cols[0] if len(st.session_state.selected_cols) == 1 else '预测值'
                                result_df[f'{pred_col_name}_预测值'] = predictions
                            else:
                                # 多目标预测
                                for i, col in enumerate(st.session_state.selected_cols):
                                    result_df[f'{col}_预测值'] = predictions[:, i]

                            # 4. 在最前面插入一个从1开始的样本索引，方便查看
                            result_df.insert(0, '样本索引', np.arange(1, len(predictions) + 1))

                            # 5. 显示最终的DataFrame
                            st.dataframe(result_df, use_container_width=True)
                            
                            # 可视化预测结果
                            if predictions.ndim == 1:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(result_df['样本索引'], predictions, 'o-', color='blue', markersize=6)
                                ax.set_title('盲样预测结果')
                                ax.set_xlabel('样本索引')
                                ax.set_ylabel('预测值')
                                ax.grid(True, linestyle='--', alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # 显示预处理后的光谱
                            if st.checkbox("查看预处理后的盲样光谱"):
                                fig, ax = plt.subplots(figsize=(12, 6))
                                n_samples = min(10, blind_X_final.shape[0])
                                
                                # 确定波数
                                if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
                                    if st.session_state.feature_selection_method != "不进行特征选择":
                                        display_wavenumbers = st.session_state.wavenumbers_final
                                    else:
                                        display_wavenumbers = blind_wavenumbers_crop
                                else:
                                    display_wavenumbers = blind_wavenumbers_crop
                                
                                for i in range(n_samples):
                                    ax.plot(display_wavenumbers, blind_X_final[i], alpha=0.7)
                                ax.set_title(f'预处理后的盲样光谱 (显示前{n_samples}个样本)')
                                ax.set_xlabel('波数 (cm⁻¹)')
                                ax.set_ylabel('处理后强度')
                                ax.grid(True, linestyle='--', alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # 提供下载链接
                            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                            b64 = base64.b64encode(csv.encode('utf-8')).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="盲样预测结果.csv">📥 下载预测结果</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            st.success("预测完成！")
                            
                        except Exception as e:
                            st.error(f"预测过程中出错: {e}")
                            st.error(traceback.format_exc())
                            
            except Exception as e:
                st.error(f"处理盲样数据时出错: {e}")
                st.error(traceback.format_exc())
                
        except Exception as e:
            st.error(f"读取盲样数据出错: {e}")
            st.error(traceback.format_exc())
    else:
        st.info("请上传盲样数据文件进行预测")
        
        # 显示使用说明
        st.subheader("使用说明")
        st.markdown("""
        **盲样数据格式要求：**
        1. 支持CSV、Excel格式
        2. 数据格式应与训练数据保持一致
        3. 前两列通常为样本标识信息
        4. 第三列开始为波数对应的强度值
        5. 波数范围应包含训练数据的波数范围
        
        **预处理流程：**
        - 系统会自动应用与训练数据相同的预处理步骤
        - 包括波数截取、平滑、基线校正、归一化等
        - **特征选择**：如果训练时进行了特征选择，会自动应用相同的特征选择
        - 确保特征数量与训练模型匹配
        
        **注意事项：**
        - 盲样数据的波数范围应覆盖训练数据的波数范围
        - 光谱数据的仪器条件应与训练数据一致
        - 预测结果的可靠性取决于盲样与训练数据的相似性
        - 如果训练时进行了特征选择，盲样预测会自动应用相同的特征选择步骤
        """)


# ====================================
# 5. 主函数
# ====================================

def main():
    """主函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="咸数光谱数据分析与预测",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 设置页面样式
    set_page_style()
    
    # 初始化会话状态
    init_session_state()
    
    # 侧边栏导航
    st.sidebar.title("🔬 咸数光谱数据分析与预测")
    st.sidebar.markdown("---")
    
    pages = {
        "1. 数据加载与标签输入": show_data_loading_page,
        "2. 数据预处理": show_preprocessing_page,
        "3. 特征提取与可视化": show_feature_extraction_page,
        "4. 数据集划分": show_data_split_page,
        "5. 模型训练与评估": show_model_training_page,
        "6. 盲样预测": show_blind_prediction_page
    }
    
    # 页面选择
    selection = st.sidebar.radio("导航", list(pages.keys()))
    
    # 显示数据加载状态
    st.sidebar.markdown("---")
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        st.sidebar.success("✅ 数据已加载")
        if hasattr(st.session_state, 'X') and hasattr(st.session_state, 'y'):
            st.sidebar.write(f"📊 光谱数据: {st.session_state.X.shape}")
            st.sidebar.write(f"🏷️ 标签数据: {st.session_state.y.shape}")
            st.sidebar.write(f"🎯 目标变量: {', '.join(st.session_state.selected_cols)}")
    else:
        st.sidebar.warning("⚠️ 请先加载数据和设置标签")
    
    # 显示预处理状态
    if hasattr(st.session_state, 'preprocessing_done') and st.session_state.preprocessing_done:
        st.sidebar.success("✅ 数据预处理完成")
    
    # 显示模型训练状态
    if hasattr(st.session_state, 'trained_models') and st.session_state.trained_models:
        st.sidebar.success(f"✅ 已训练 {len(st.session_state.trained_models)} 个模型")
    
    # 显示作者信息
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **咸数光谱数据分析与预测应用 v2.0**
        
        一个王军用于分析光谱数据并构建预测模型的专业应用程序。
        
        **主要功能**：
        - 🔄 灵活的标签输入方式
        - 🛠️ 完整的数据预处理流程
        - 🤖 多种机器学习算法
        - 📊 模型训练与评估
        - 🔮 盲样预测功能
        
        **基线校正算法**：
        - AIRPLS (自适应迭代加权惩罚最小二乘)
        - ASLS (非对称最小二乘)
        - Polynomial (多项式拟合)
        - ModPoly (修正多项式)
        
        **支持的模型**：
        线性回归、岭回归、Lasso、SVR、随机森林、梯度提升、MLP、PLS、XGBoost
        """
    )
    
    # 显示选定的页面
    page_func = pages[selection]
    page_func()

# ====================================
# 6. 程序入口
# ====================================

if __name__ == "__main__":
    main()