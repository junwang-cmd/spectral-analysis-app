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
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import mahalanobis

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
# 需要额外导入的库
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



# ====================================
# 1. 全局配置和常量
# ====================================

MODEL_NAMES = {
    'linear': '线性回归',
    'ridge': '岭回归', 
    'lasso': 'Lasso回归',
    'svr': '支持向量回归',
    'rf': '随机森林',
    'gbr': '梯度提升回归',
    'mlp': '多层感知机',
    'pls': '偏最小二乘回归',
    'xgb': 'XGBoost'
}


# ====================================
# 2. 工具函数
# ====================================

def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
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


def check_data_prerequisites(need_labels=False, need_preprocessing=True):
    """
    检查数据前置条件
    
    Args:
        need_labels: 是否需要标签数据
        need_preprocessing: 是否需要预处理完成
    
    Returns:
        bool: 是否满足条件
    """
    # 检查数据是否加载
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        show_status_message("请先加载数据", "warning")
        return False
    
    # 检查是否需要标签数据
    if need_labels:
        if not hasattr(st.session_state, 'y') or st.session_state.y is None:
            show_status_message("此功能需要标签数据，请在数据加载页面输入标签", "warning")
            return False
    
    # 检查是否需要预处理完成
    if need_preprocessing:
        if not hasattr(st.session_state, 'preprocessing_done') or not st.session_state.preprocessing_done:
            show_status_message("请先完成数据预处理", "warning")
            return False
    
    return True


def get_current_data():
    """
    获取当前使用的数据
    
    Returns:
        tuple: (X, wavenumbers, info_message)
    """
    if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
        X = st.session_state.X_final
        wavenumbers = st.session_state.wavenumbers_final
        info = f"✅ 使用特征选择后的数据，特征数量: {X.shape[1]}"
    elif hasattr(st.session_state, 'preprocessing_done') and st.session_state.preprocessing_done:
        X = st.session_state.X_preprocessed
        wavenumbers = st.session_state.wavenumbers_preprocessed
        info = f"ℹ️ 使用预处理后的全部特征，特征数量: {X.shape[1]}"
    else:
        X = st.session_state.X
        wavenumbers = st.session_state.wavenumbers
        info = f"⚠️ 使用原始数据，特征数量: {X.shape[1]}"
    
    return X, wavenumbers, info


def show_status_message(message, message_type="info"):
    """
    显示状态消息
    
    Args:
        message: 消息内容
        message_type: 消息类型 ("info", "success", "warning", "error")
    """
    if message_type == "info":
        st.info(message)
    elif message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:
        st.write(message)


def safe_execute(func, error_message="操作失败"):
    """
    安全执行函数，捕获异常
    
    Args:
        func: 要执行的函数
        error_message: 错误消息
    
    Returns:
        函数执行结果或None
    """
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None


# ====================================
# 3. 基线校正类
# ====================================
class SpectrumBaselineCorrector:
    """光谱基线校正器 - 完整实现"""
    
    def __init__(self):
        self.available_methods = {
            'als': self.als_baseline,
            'arPLS': self.arpls_baseline,
            'polynomial': self.polynomial_baseline,
            'linear': self.linear_baseline,
            'rolling_ball': self.rolling_ball_baseline
        }
    
    def correct_baseline(self, spectrum, method='als', **params):
        """
        执行基线校正
        
        Args:
            spectrum: 光谱数据 (1D array)
            method: 校正方法 ('als', 'arPLS', 'polynomial', 'linear', 'rolling_ball')
            **params: 方法特定参数
        
        Returns:
            tuple: (baseline, corrected_spectrum)
        """
        # 数据预检查
        spectrum = self._validate_spectrum(spectrum)
        
        if method not in self.available_methods:
            raise ValueError(f"不支持的基线校正方法: {method}")
        
        try:
            baseline = self.available_methods[method](spectrum, **params)
            corrected = spectrum - baseline
            return baseline, corrected
        except Exception as e:
            # 详细错误信息
            print(f"基线校正失败 - 方法: {method}, 错误: {str(e)}")
            print(f"光谱形状: {spectrum.shape}")
            print(f"光谱范围: [{np.min(spectrum):.3f}, {np.max(spectrum):.3f}]")
            print(f"是否包含NaN: {np.isnan(spectrum).any()}")
            print(f"是否包含无穷值: {np.isinf(spectrum).any()}")
            raise e
    
    def _validate_spectrum(self, spectrum):
        """验证和清理光谱数据"""
        spectrum = np.asarray(spectrum, dtype=float)
        
        # 检查维度
        if spectrum.ndim != 1:
            raise ValueError(f"光谱必须是1D数组，当前维度: {spectrum.ndim}")
        
        # 检查长度
        if len(spectrum) < 10:
            raise ValueError(f"光谱数据点太少: {len(spectrum)}")
        
        # 处理NaN和无穷值
        if np.isnan(spectrum).any():
            print("警告: 光谱包含NaN值，将用邻近值填充")
            spectrum = self._interpolate_nan(spectrum)
        
        if np.isinf(spectrum).any():
            print("警告: 光谱包含无穷值，将用有限值替换")
            spectrum[np.isinf(spectrum)] = np.nanmedian(spectrum[np.isfinite(spectrum)])
        
        return spectrum
    
    def _interpolate_nan(self, spectrum):
        """插值填充NaN值"""
        mask = np.isfinite(spectrum)
        if not mask.any():
            raise ValueError("光谱数据全部为NaN或无穷值")
        
        indices = np.arange(len(spectrum))
        spectrum[~mask] = np.interp(indices[~mask], indices[mask], spectrum[mask])
        return spectrum
    
    def als_baseline(self, spectrum, lam=1e5, p=0.01, niter=10):
        """
        非对称最小二乘基线校正 (Asymmetric Least Squares)
        
        Args:
            spectrum: 光谱数据
            lam: 平滑参数 (越大越平滑)
            p: 非对称参数 (0-1, 越小越偏向谷底)
            niter: 迭代次数
        """
        spectrum = np.asarray(spectrum, dtype=float)
        L = len(spectrum)
        
        # 构建差分矩阵
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        
        # 初始化权重
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            try:
                # 求解基线
                W.setdiag(w)
                Z = W + D
                baseline = spsolve(Z, w * spectrum)
                
                # 更新权重
                w = p * (spectrum > baseline) + (1 - p) * (spectrum < baseline)
                
            except Exception as e:
                print(f"ALS迭代 {i+1} 失败: {e}")
                if i == 0:
                    # 如果第一次迭代就失败，返回简单基线
                    return self.linear_baseline(spectrum)
                else:
                    # 使用上一次的结果
                    break
        
        return baseline
    
    def arpls_baseline(self, spectrum, lam=1e5, ratio=0.01, niter=10):
        """
        非对称重加权惩罚最小二乘 - 数值稳定版本
        """
        try:
            spectrum = np.asarray(spectrum, dtype=float).flatten()
            L = len(spectrum)
            
            # 验证输入
            if L < 3:
                print("光谱长度太短，使用线性基线")
                return self.linear_baseline(spectrum)
            
            # 构建二阶差分矩阵
            diags = np.ones(L-2)
            D1 = sparse.spdiags([-diags, 2*diags, -diags], [0, 1, 2], L-2, L)
            D2 = D1.T @ D1
            
            # 初始化
            w = np.ones(L)
            baseline = spectrum.copy()
            
            for i in range(niter):
                try:
                    # 构建权重矩阵
                    W = sparse.diags(w, format='csr')
                    
                    # 构建系统矩阵
                    A = W + lam * D2
                    
                    # 右侧向量
                    b = W @ spectrum
                    
                    # 求解线性系统
                    try:
                        baseline_new = spsolve(A, b)
                    except:
                        A_dense = A.toarray()
                        baseline_new = np.linalg.solve(A_dense, b)
                    
                    # 确保结果是一维数组
                    baseline_new = np.asarray(baseline_new).flatten()
                    
                    # 计算残差
                    residual = spectrum - baseline_new
                    negative_residual = residual[residual < 0]
                    
                    if len(negative_residual) == 0:
                        baseline = baseline_new
                        break
                    
                    # 计算统计量
                    mean_neg = np.mean(negative_residual)
                    std_neg = np.std(negative_residual)
                    
                    if std_neg == 0:
                        baseline = baseline_new
                        break
                    
                    # ⭐ 数值稳定的权重更新 - 修复溢出问题 ⭐
                    threshold = 2 * std_neg - mean_neg
                    exp_arg = 2.0 * (residual - threshold) / std_neg
                    
                    # 限制指数参数范围，避免溢出
                    exp_arg = np.clip(exp_arg, -50, 50)  # 限制在合理范围内
                    
                    # 使用数值稳定的sigmoid函数
                    w_new = np.where(exp_arg > 0, 
                                    1.0 / (1.0 + np.exp(-exp_arg)),  # 正数情况
                                    np.exp(exp_arg) / (1.0 + np.exp(exp_arg)))  # 负数情况
                    
                    # 检查收敛
                    weight_change = np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-10)
                    if weight_change < ratio:
                        baseline = baseline_new
                        break
                    
                    # 阻尼更新，提高稳定性
                    w = 0.7 * w + 0.3 * w_new
                    baseline = baseline_new
                    
                except Exception as iter_error:
                    print(f"arPLS迭代 {i+1} 失败: {iter_error}")
                    if i == 0:
                        print("arPLS失败，切换到ALS方法")
                        return self.als_baseline(spectrum, lam=lam, p=0.01, niter=niter)
                    else:
                        break
            
            return baseline
            
        except Exception as e:
            print(f"arPLS完全失败: {e}")
            print("自动切换到ALS方法")
            return self.als_baseline(spectrum, lam=lam, p=0.01, niter=niter)
    
    def polynomial_baseline(self, spectrum, degree=2):
        """多项式基线校正"""
        try:
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, degree)
            baseline = np.polyval(coeffs, x)
            return baseline
        except Exception as e:
            print(f"多项式基线校正失败: {e}")
            return self.linear_baseline(spectrum)
    
    def linear_baseline(self, spectrum):
        """线性基线校正"""
        x = np.arange(len(spectrum))
        slope = (spectrum[-1] - spectrum[0]) / (len(spectrum) - 1)
        baseline = spectrum[0] + slope * x
        return baseline
    
    def rolling_ball_baseline(self, spectrum, radius=100):
        """滚球基线校正"""
        try:
            from scipy.ndimage import minimum_filter1d
            
            # 使用最小值滤波器模拟滚球
            baseline = minimum_filter1d(spectrum, size=radius*2+1, mode='mirror')
            
            # 平滑基线
            baseline = savgol_filter(baseline, min(len(baseline)//10*2+1, 51), 3)
            
            return baseline
        except Exception as e:
            print(f"滚球基线校正失败: {e}")
            return self.linear_baseline(spectrum)


# 基线校正问题诊断函数
def diagnose_baseline_correction_issue(spectrum_data, wavenumbers, method='als', **params):
    """
    诊断基线校正问题
    
    Args:
        spectrum_data: 光谱数据矩阵 (n_samples x n_features)
        wavenumbers: 波数数组
        method: 基线校正方法
        **params: 方法参数
    """
    print("=== 基线校正问题诊断 ===")
    
    corrector = SpectrumBaselineCorrector()
    
    # 检查第一个样本
    first_spectrum = spectrum_data[0]
    
    print(f"样本形状: {spectrum_data.shape}")
    print(f"第一个光谱形状: {first_spectrum.shape}")
    print(f"光谱数值范围: [{np.min(first_spectrum):.3f}, {np.max(first_spectrum):.3f}]")
    print(f"是否包含NaN: {np.isnan(first_spectrum).any()}")
    print(f"是否包含无穷值: {np.isinf(first_spectrum).any()}")
    print(f"波数范围: [{np.min(wavenumbers):.1f}, {np.max(wavenumbers):.1f}]")
    
    # 尝试校正第一个样本
    try:
        baseline, corrected = corrector.correct_baseline(first_spectrum, method, **params)
        print("✅ 第一个样本基线校正成功")
        
        # 可视化结果
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(wavenumbers, first_spectrum, 'b-', label='原始光谱')
        plt.plot(wavenumbers, baseline, 'r--', label='基线')
        plt.xlabel('波数 (cm⁻¹)')
        plt.ylabel('强度')
        plt.title('原始光谱与基线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(wavenumbers, corrected, 'g-', label='校正后光谱')
        plt.xlabel('波数 (cm⁻¹)')
        plt.ylabel('强度')
        plt.title('基线校正后光谱')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(first_spectrum, bins=30, alpha=0.7, label='原始')
        plt.hist(corrected, bins=30, alpha=0.7, label='校正后')
        plt.xlabel('强度')
        plt.ylabel('频数')
        plt.title('强度分布')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        residuals = first_spectrum - baseline - corrected
        plt.plot(wavenumbers, residuals, 'k-')
        plt.xlabel('波数 (cm⁻¹)')
        plt.ylabel('残差')
        plt.title('拟合残差')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return True, baseline, corrected
        
    except Exception as e:
        print(f"❌ 第一个样本基线校正失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        # 提供解决建议
        print("\n=== 解决建议 ===")
        
        if np.isnan(first_spectrum).any():
            print("1. 光谱包含NaN值，建议预处理时检查数据质量")
        
        if np.isinf(first_spectrum).any():
            print("2. 光谱包含无穷值，建议检查数据预处理步骤")
        
        if len(first_spectrum) < 100:
            print("3. 光谱数据点太少，可能影响基线校正效果")
        
        if method == 'als' and 'lam' in params:
            if params['lam'] > 1e7:
                print("4. ALS方法的lambda参数太大，建议降低到1e5-1e6")
        
        print("5. 建议尝试更简单的基线校正方法，如'linear'或'polynomial'")
        print("6. 检查光谱数据的物理合理性")
        
        return False, None, None


# 修复建议的参数设置
def get_robust_baseline_params():
    """获取稳健的基线校正参数"""
    return {
        'als': {'lam': 1e5, 'p': 0.01, 'niter': 10},
        'arPLS': {'lam': 1e5, 'ratio': 0.01, 'niter': 10}, 
        'polynomial': {'degree': 2},
        'linear': {},
        'rolling_ball': {'radius': 50}
    }


# 在Streamlit应用中的使用示例
def safe_baseline_correction(X, method='als', **params):
    """
    安全的基线校正，带错误处理和降级策略
    """
    corrector = SpectrumBaselineCorrector()
    X_corrected = np.zeros_like(X)
    failed_samples = []
    
    for i in range(X.shape[0]):
        try:
            baseline, corrected = corrector.correct_baseline(X[i], method, **params)
            X_corrected[i] = corrected
        except Exception as e:
            # 记录失败的样本
            failed_samples.append(i+1)
            
            # 降级策略：使用线性基线校正
            try:
                baseline, corrected = corrector.correct_baseline(X[i], 'linear')
                X_corrected[i] = corrected
                print(f"样本 {i+1}: {method}校正失败，改用线性校正")
            except Exception as e2:
                # 最后的降级：不校正
                X_corrected[i] = X[i]
                print(f"样本 {i+1}: 所有基线校正方法失败，使用原始数据")
    
    if failed_samples:
        print(f"基线校正失败的样本: {failed_samples}")
        print("建议检查这些样本的数据质量")
    
    return X_corrected, failed_samples



# ====================================
# 4. 模型相关函数
# ====================================

def create_model_instance(model_name, params, is_multioutput):
    """
    创建模型实例
    
    Args:
        model_name: 模型名称
        params: 模型参数
        is_multioutput: 是否为多输出问题
        
    Returns:
        tuple: (model, use_scaler)
    """
    use_scaler = False
    
    if model_name == 'linear':
        model = LinearRegression()
        use_scaler = True
    elif model_name == 'ridge':
        model = Ridge(alpha=params['alpha'], random_state=params.get('random_state', 42))
        use_scaler = True
    elif model_name == 'lasso':
        model = Lasso(alpha=params['alpha'], random_state=params.get('random_state', 42))
        use_scaler = True
    elif model_name == 'svr':
        base_model = SVR(**params)
        model = MultiOutputRegressor(base_model) if is_multioutput else base_model
        use_scaler = True
    elif model_name == 'rf':
        model = RandomForestRegressor(**params)
    elif model_name == 'gbr':
        base_model = GradientBoostingRegressor(**params)
        model = MultiOutputRegressor(base_model) if is_multioutput else base_model
    elif model_name == 'mlp':
        model = MLPRegressor(**params)
        use_scaler = True
    elif model_name == 'pls':
        model = PLSRegression(**params)
    elif model_name == 'xgb':
        try:
            import xgboost as xgb
            base_model = xgb.XGBRegressor(**params)
            model = MultiOutputRegressor(base_model) if is_multioutput else base_model
        except ImportError:
            raise ImportError("XGBoost未安装，请先安装: pip install xgboost")
    else:
        raise ValueError(f"未知的模型类型: {model_name}")
    
    return model, use_scaler


def setup_model_parameters_ui(model_name, index):
    """
    设置模型参数UI
    
    Args:
        model_name: 模型名称
        index: 索引（用于区分不同的UI组件）
        
    Returns:
        dict: 模型参数字典
    """
    if model_name == 'linear':
        return {}  # 线性回归无参数
    elif model_name == 'ridge':
        return setup_ridge_params(index)
    elif model_name == 'lasso':
        return setup_lasso_params(index)
    elif model_name == 'svr':
        return setup_svr_params(index)
    elif model_name == 'rf':
        return setup_rf_params(index)
    elif model_name == 'gbr':
        return setup_gbr_params(index)
    elif model_name == 'mlp':
        return setup_mlp_params(index)
    elif model_name == 'pls':
        return setup_pls_params(index)
    elif model_name == 'xgb':
        return setup_xgb_params(index)
    else:
        return {}


def setup_ridge_params(index):
    """岭回归参数设置"""
    alpha = st.selectbox(
        "正则化强度", 
        [0.1, 1.0, 10.0, 100.0], 
        index=1, 
        key=f"ridge_alpha_{index}"
    )
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"ridge_seed_{index}"
    )
    
    return {
        'alpha': alpha,
        'random_state': random_state
    }


def setup_lasso_params(index):
    """Lasso回归参数设置"""
    alpha = st.selectbox(
        "正则化强度", 
        [0.01, 0.1, 1.0, 10.0], 
        index=1, 
        key=f"lasso_alpha_{index}"
    )
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"lasso_seed_{index}"
    )
    
    return {
        'alpha': alpha,
        'random_state': random_state
    }


def setup_svr_params(index):
    """支持向量回归参数设置"""
    col1, col2 = st.columns(2)
    
    with col1:
        C = st.selectbox(
            "惩罚参数C", 
            [0.1, 1.0, 10.0, 100.0], 
            index=2, 
            key=f"svr_C_{index}"
        )
        kernel = st.selectbox(
            "核函数", 
            ['rbf', 'linear', 'poly'], 
            index=0, 
            key=f"svr_kernel_{index}"
        )
    
    with col2:
        gamma = st.selectbox(
            "Gamma", 
            ['scale', 'auto', 0.001, 0.01, 0.1], 
            index=0, 
            key=f"svr_gamma_{index}"
        )
        epsilon = st.selectbox(
            "Epsilon", 
            [0.01, 0.1, 0.2], 
            index=1, 
            key=f"svr_epsilon_{index}"
        )
    
    return {
        'C': C,
        'kernel': kernel,
        'gamma': gamma,
        'epsilon': epsilon
    }


def setup_rf_params(index):
    """随机森林参数设置"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "树的数量", 
            50, 500, 100, 
            key=f"rf_trees_{index}"
        )
        max_depth = st.selectbox(
            "最大深度", 
            [None, 5, 10, 15, 20], 
            index=0, 
            key=f"rf_depth_{index}"
        )
    
    with col2:
        min_samples_split = st.slider(
            "分裂最小样本数", 
            2, 10, 2, 
            key=f"rf_split_{index}"
        )
        min_samples_leaf = st.slider(
            "叶节点最小样本数", 
            1, 5, 1, 
            key=f"rf_leaf_{index}"
        )
    
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"rf_seed_{index}"
    )
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state
    }


def setup_gbr_params(index):
    """梯度提升参数设置"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "提升阶段数", 
            50, 500, 100, 
            key=f"gbr_stages_{index}"
        )
        learning_rate = st.selectbox(
            "学习率", 
            [0.01, 0.05, 0.1, 0.2], 
            index=2, 
            key=f"gbr_lr_{index}"
        )
    
    with col2:
        max_depth = st.slider(
            "最大深度", 
            2, 10, 3, 
            key=f"gbr_depth_{index}"
        )
        subsample = st.slider(
            "子采样比例", 
            0.5, 1.0, 1.0, 
            step=0.1, 
            key=f"gbr_subsample_{index}"
        )
    
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"gbr_seed_{index}"
    )
    
    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'random_state': random_state
    }


def setup_pls_params(index):
    """PLS参数设置"""
    n_components = st.slider(
        "主成分数量", 
        1, min(20, st.session_state.X_train.shape[1]), 
        5, 
        key=f"pls_components_{index}"
    )
    scale = st.checkbox(
        "标准化", 
        value=True, 
        key=f"pls_scale_{index}"
    )
    
    return {
        'n_components': n_components,
        'scale': scale
    }


def setup_mlp_params(index):
    """MLP参数设置"""
    col1, col2 = st.columns(2)
    
    with col1:
        layer_option = st.selectbox(
            "隐藏层结构", 
            ["一层", "两层", "三层"], 
            index=1, 
            key=f"mlp_layers_{index}"
        )
        
        if layer_option == "一层":
            layer1_size = st.slider(
                "隐藏层神经元数", 
                10, 200, 50, 
                key=f"mlp_l1_{index}"
            )
            hidden_layer_sizes = (layer1_size,)
        elif layer_option == "两层":
            layer1_size = st.slider(
                "第一层神经元数", 
                10, 200, 100, 
                key=f"mlp_l1_{index}"
            )
            layer2_size = st.slider(
                "第二层神经元数", 
                10, 100, 50, 
                key=f"mlp_l2_{index}"
            )
            hidden_layer_sizes = (layer1_size, layer2_size)
        else:  # 三层
            layer1_size = st.slider(
                "第一层神经元数", 
                10, 200, 100, 
                key=f"mlp_l1_{index}"
            )
            layer2_size = st.slider(
                "第二层神经元数", 
                10, 100, 50, 
                key=f"mlp_l2_{index}"
            )
            layer3_size = st.slider(
                "第三层神经元数", 
                10, 50, 25, 
                key=f"mlp_l3_{index}"
            )
            hidden_layer_sizes = (layer1_size, layer2_size, layer3_size)
        
        activation = st.selectbox(
            "激活函数", 
            ['relu', 'tanh', 'logistic'], 
            index=0, 
            key=f"mlp_activation_{index}"
        )
    
    with col2:
        solver = st.selectbox(
            "优化算法", 
            ['adam', 'lbfgs', 'sgd'], 
            index=0, 
            key=f"mlp_solver_{index}"
        )
        learning_rate_init = st.selectbox(
            "初始学习率", 
            [0.0001, 0.001, 0.01], 
            index=1, 
            key=f"mlp_lr_{index}"
        )
        max_iter = st.slider(
            "最大迭代次数", 
            100, 1000, 500, 
            key=f"mlp_iter_{index}"
        )
        alpha = st.selectbox(
            "L2正则化参数", 
            [0.0001, 0.001, 0.01], 
            index=0, 
            key=f"mlp_alpha_{index}"
        )
    
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"mlp_seed_{index}"
    )
    
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'learning_rate_init': learning_rate_init,
        'max_iter': max_iter,
        'alpha': alpha,
        'random_state': random_state
    }


def setup_xgb_params(index):
    """XGBoost参数设置"""
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "提升轮数", 
            50, 500, 100, 
            key=f"xgb_trees_{index}"
        )
        learning_rate = st.selectbox(
            "学习率", 
            [0.01, 0.05, 0.1, 0.2], 
            index=2, 
            key=f"xgb_lr_{index}"
        )
        max_depth = st.slider(
            "最大深度", 
            2, 10, 6, 
            key=f"xgb_depth_{index}"
        )
    
    with col2:
        subsample = st.slider(
            "子采样比例", 
            0.5, 1.0, 1.0, 
            step=0.1, 
            key=f"xgb_subsample_{index}"
        )
        colsample_bytree = st.slider(
            "特征采样比例", 
            0.5, 1.0, 1.0, 
            step=0.1, 
            key=f"xgb_colsample_{index}"
        )
        reg_alpha = st.selectbox(
            "L1正则化", 
            [0, 0.01, 0.1], 
            index=0, 
            key=f"xgb_alpha_{index}"
        )
    
    random_state = st.number_input(
        "随机种子", 
        value=42, 
        key=f"xgb_seed_{index}"
    )
    
    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'random_state': random_state
    }


# ====================================
# 5. 页面函数
# ====================================

def show_data_loading_page():
    """数据加载与标签输入页面"""
    st.markdown("<h1 class='section-header'>数据加载与标签输入</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    请上传光谱数据文件，支持 CSV 和 Excel 格式。系统会自动识别光谱数据和标签数据。
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化会话状态
    if 'data_format_confirmed' not in st.session_state:
        st.session_state.data_format_confirmed = False
    if 'label_setup_completed' not in st.session_state:
        st.session_state.label_setup_completed = False
    
    # 文件上传
    uploaded_file = st.file_uploader("选择数据文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # 保存原始数据到会话状态
            if 'uploaded_df' not in st.session_state or st.session_state.uploaded_df is None:
                st.session_state.uploaded_df = df
                st.session_state.data_format_confirmed = False
                st.session_state.label_setup_completed = False
            
            st.success(f"文件加载成功！数据形状: {df.shape}")
            
            # 显示数据预览
            st.subheader("📊 数据预览")
            st.dataframe(df.head(), use_container_width=True)
            
            # 步骤1：数据格式设置（只在未确认时显示）
            if not st.session_state.data_format_confirmed:
                st.subheader("⚙️ 步骤1: 数据格式设置")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**选择光谱数据起始列：**")
                    start_col_options = list(range(1, min(df.shape[1], 11)))
                    
                    selected_start_col = st.selectbox(
                        "光谱波数数据从第几列开始？",
                        start_col_options,
                        index=2 if len(start_col_options) > 2 else 0,
                        format_func=lambda x: f"第{x}列 ({df.columns[x-1]})",
                        key="start_col_select"
                    )
                
                with col2:
                    st.write("**标签数据设置：**")
                    has_labels = st.radio("数据中是否包含标签（目标变量）？", ["是", "否"], key="has_labels_radio")
                
                # 确认数据格式
                if st.button("确认数据格式", key="confirm_format_btn"):
                    with st.spinner("正在处理光谱数据..."):
                        try:
                            # 识别波数列
                            potential_wavenumbers = df.columns[selected_start_col-1:]
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
                            
                            # 提取光谱数据和波数
                            wavenumbers = pd.Series(numeric_columns).astype(float)
                            X = df[numeric_columns].values.astype(float)
                            
                            # 保存到会话状态
                            st.session_state.X = X
                            st.session_state.wavenumbers = wavenumbers
                            st.session_state.original_df = df
                            st.session_state.numeric_columns = numeric_columns
                            st.session_state.has_labels_choice = has_labels
                            st.session_state.data_format_confirmed = True
                            
                            st.success(f"✅ 光谱数据处理成功！")
                            st.info(f"📊 光谱数据形状: {X.shape}")
                            st.info(f"📏 波数范围: {wavenumbers.min():.1f} ~ {wavenumbers.max():.1f} cm⁻¹")
                            
                            # 如果选择无标签，直接完成设置
                            if has_labels == "否":
                                st.session_state.y = None
                                st.session_state.selected_cols = []
                                st.session_state.label_setup_completed = True
                                st.session_state.data_loaded = True
                                st.info("ℹ️ 未设置标签数据，可进行趋势分析等无监督分析")
                                st.rerun()  # 刷新页面显示下一步
                            else:
                                st.info("👇 请继续进行标签数据设置")
                                st.rerun()  # 刷新页面显示标签设置
                            
                        except Exception as e:
                            st.error(f"数据处理出错: {e}")
                            st.error(traceback.format_exc())
            
            # 步骤2：标签数据设置（只在格式确认后且选择有标签时显示）
            elif st.session_state.data_format_confirmed and not st.session_state.label_setup_completed:
                if st.session_state.has_labels_choice == "是":
                    st.subheader("🏷️ 步骤2: 标签数据设置")
                    
                    # 显示光谱数据信息
                    st.info(f"✅ 光谱数据已确认 - 形状: {st.session_state.X.shape}")
                    
                    # 从非波数列中选择标签列
                    label_candidates = [col for col in df.columns if col not in st.session_state.numeric_columns]
                    
                    if label_candidates:
                        st.write("**可选择的标签列：**")
                        
                        # 显示候选标签列的预览
                        preview_df = df[label_candidates].head()
                        st.dataframe(preview_df, use_container_width=True)
                        
                        selected_label_cols = st.multiselect(
                            "选择标签列（目标变量）",
                            label_candidates,
                            help="可以选择多个目标变量进行多输出预测",
                            key="label_cols_select"
                        )
                        
                        if selected_label_cols:
                            # 显示选中标签的统计信息
                            st.write("**选中标签的统计信息：**")
                            label_stats = df[selected_label_cols].describe()
                            st.dataframe(label_stats, use_container_width=True)
                            
                            # 确认标签设置
                            if st.button("确认标签设置", key="confirm_labels_btn"):
                                try:
                                    y = df[selected_label_cols].values
                                    if len(selected_label_cols) == 1:
                                        y = y.ravel()
                                    
                                    st.session_state.y = y
                                    st.session_state.selected_cols = selected_label_cols
                                    st.session_state.label_setup_completed = True
                                    st.session_state.data_loaded = True
                                    
                                    st.success(f"✅ 标签数据设置成功！")
                                    st.info(f"🎯 目标变量: {', '.join(selected_label_cols)}")
                                    st.info(f"📊 标签数据形状: {y.shape}")
                                    
                                    st.rerun()  # 刷新页面显示完成状态
                                    
                                except Exception as e:
                                    st.error(f"标签数据处理出错: {e}")
                        else:
                            st.warning("请选择至少一个标签列")
                            
                            # 提供跳过选项
                            if st.button("跳过标签设置（仅进行无监督分析）", key="skip_labels_btn"):
                                st.session_state.y = None
                                st.session_state.selected_cols = []
                                st.session_state.label_setup_completed = True
                                st.session_state.data_loaded = True
                                st.info("ℹ️ 已跳过标签设置，可进行趋势分析等无监督分析")
                                st.rerun()
                    else:
                        st.warning("未找到可用的标签列，所有列都被识别为波数数据")
                        if st.button("确认无标签数据", key="confirm_no_labels_btn"):
                            st.session_state.y = None
                            st.session_state.selected_cols = []
                            st.session_state.label_setup_completed = True
                            st.session_state.data_loaded = True
                            st.info("ℹ️ 确认无标签数据，可进行趋势分析等无监督分析")
                            st.rerun()
            
            # 步骤3：显示最终完成状态
            elif st.session_state.data_format_confirmed and st.session_state.label_setup_completed:
                st.subheader("✅ 数据加载完成")
                
                # 显示数据摘要
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("**光谱数据**")
                    st.info(f"📊 数据形状: {st.session_state.X.shape}")
                    st.info(f"📏 波数范围: {st.session_state.wavenumbers.min():.1f} ~ {st.session_state.wavenumbers.max():.1f} cm⁻¹")
                
                with col2:
                    if st.session_state.y is not None:
                        st.success("**标签数据**")
                        st.info(f"🎯 目标变量: {', '.join(st.session_state.selected_cols)}")
                        st.info(f"📊 标签形状: {st.session_state.y.shape}")
                    else:
                        st.info("**无标签数据**")
                        st.info("🔍 适用于无监督分析")
                
                # 提供重新设置选项
                if st.button("🔄 重新设置数据", key="reset_data_btn"):
                    # 清除相关会话状态
                    keys_to_clear = [
                        'data_format_confirmed', 'label_setup_completed', 'data_loaded',
                        'X', 'y', 'wavenumbers', 'selected_cols', 'numeric_columns', 
                        'has_labels_choice', 'uploaded_df'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                
                # 显示详细统计信息
                with st.expander("📊 查看详细统计信息"):
                    if st.session_state.y is not None:
                        if len(st.session_state.selected_cols) == 1:
                            st.write("**标签统计信息：**")
                            y = st.session_state.y
                            stats_df = pd.DataFrame({
                                '统计量': ['样本数', '均值', '标准差', '最小值', '最大值'],
                                '数值': [
                                    f"{len(y)}",
                                    f"{np.mean(y):.4f}",
                                    f"{np.std(y):.4f}",
                                    f"{np.min(y):.4f}",
                                    f"{np.max(y):.4f}"
                                ]
                            })
                            st.dataframe(stats_df, use_container_width=True)
                        else:
                            st.write("**多目标标签统计信息：**")
                            stats_data = []
                            for i, col in enumerate(st.session_state.selected_cols):
                                stats_data.append({
                                    '目标变量': col,
                                    '样本数': f"{st.session_state.y.shape[0]}",
                                    '均值': f"{np.mean(st.session_state.y[:, i]):.4f}",
                                    '标准差': f"{np.std(st.session_state.y[:, i]):.4f}",
                                    '最小值': f"{np.min(st.session_state.y[:, i]):.4f}",
                                    '最大值': f"{np.max(st.session_state.y[:, i]):.4f}"
                                })
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                    
                    # 光谱数据统计
                    st.write("**光谱数据统计：**")
                    X = st.session_state.X
                    spectrum_stats = pd.DataFrame({
                        '统计量': ['样本数', '特征数', '数据范围最小值', '数据范围最大值', '平均光谱强度'],
                        '数值': [
                            f"{X.shape[0]}",
                            f"{X.shape[1]}",
                            f"{X.min():.4f}",
                            f"{X.max():.4f}",
                            f"{np.mean(X):.4f}"
                        ]
                    })
                    st.dataframe(spectrum_stats, use_container_width=True)
        
        except Exception as e:
            st.error(f"文件读取失败: {e}")
    
    else:
        st.info("请上传数据文件开始分析")
        
        # 显示使用说明
        st.subheader("📋 使用说明")
        st.markdown("""
        **支持的文件格式：**
        - CSV 文件 (.csv)
        - Excel 文件 (.xlsx, .xls)
        
        **数据格式要求：**
        1. 每行代表一个样本
        2. 前几列可以是样本标识信息
        3. 光谱数据列的列名应为波数值（如 4000, 3999.5, 3999, ...）
        4. 如果有标签数据，应在非波数列中
        
        **数据加载流程：**
        1. 上传文件并预览数据
        2. 设置光谱数据起始列和标签选项
        3. 如有标签，选择具体的标签列
        4. 完成数据加载设置
        
        **标签数据：**
        - 有标签：适用于定量预测建模
        - 无标签：适用于趋势分析、PCA分析等探索性分析
        """)


def show_preprocessing_page():
    """数据预处理页面 - 适配完整基线校正器"""
    st.markdown("<h1 class='section-header'>数据预处理</h1>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not check_data_prerequisites(need_labels=False, need_preprocessing=False):
        return
    
    st.markdown("""
    <div class="info-box">
    对光谱数据进行预处理，包括波数截取、平滑、基线校正、归一化等步骤。
    </div>
    """, unsafe_allow_html=True)
    
    X = st.session_state.X
    wavenumbers = st.session_state.wavenumbers
    
    st.info(f"📊 原始数据形状: {X.shape}")
    st.info(f"📏 波数范围: {wavenumbers.min():.1f} ~ {wavenumbers.max():.1f} cm⁻¹")
    
    # 预处理参数设置
    st.subheader("⚙️ 预处理参数设置")
    
    # 1. 波数截取
    st.write("**1. 波数范围截取**")
    col1, col2 = st.columns(2)
    
    with col1:
        start_wavenumber = st.number_input(
            "起始波数 (cm⁻¹)", 
            min_value=float(wavenumbers.min()),
            max_value=float(wavenumbers.max()),
            value=float(wavenumbers.min()),
            step=0.5
        )
    
    with col2:
        end_wavenumber = st.number_input(
            "结束波数 (cm⁻¹)", 
            min_value=float(wavenumbers.min()),
            max_value=float(wavenumbers.max()),
            value=float(wavenumbers.max()),
            step=0.5
        )
    
    # 2. 平滑处理
    st.write("**2. Savitzky-Golay 平滑**")
    apply_smooth = st.checkbox("启用平滑处理", value=True)
    
    if apply_smooth:
        col1, col2 = st.columns(2)
        with col1:
            smooth_window = st.slider("窗口大小", 5, 21, 9, step=2)
        with col2:
            smooth_poly = st.slider("多项式阶数", 1, 5, 3)
    
    # 3. 基线校正 - 更新为支持新的方法
    st.write("**3. 基线校正**")
    apply_baseline = st.checkbox("启用基线校正", value=True)
    
    if apply_baseline:
        baseline_method = st.selectbox(
            "基线校正方法", 
            ['als', 'arPLS', 'polynomial', 'linear', 'rolling_ball'],
            format_func=lambda x: {
                'als': 'ALS (非对称最小二乘)',
                'arPLS': 'arPLS (非对称重加权惩罚最小二乘)',
                'polynomial': '多项式拟合',
                'linear': '线性基线',
                'rolling_ball': '滚球基线校正'
            }[x]
        )
        
        # 基线校正参数 - 根据不同方法显示不同参数
        baseline_params = {}
        
        if baseline_method == 'als':
            st.write("*ALS 参数设置*")
            col1, col2, col3 = st.columns(3)
            with col1:
                baseline_params['lam'] = st.selectbox(
                    "λ (平滑度)", 
                    [1e3, 1e4, 1e5, 1e6, 1e7], 
                    index=2,
                    help="数值越大，基线越平滑"
                )
            with col2:
                baseline_params['p'] = st.selectbox(
                    "p (不对称度)", 
                    [0.001, 0.01, 0.1, 0.5], 
                    index=1,
                    help="数值越小，基线越偏向谷底"
                )
            with col3:
                baseline_params['niter'] = st.slider(
                    "迭代次数", 
                    5, 50, 10,
                    help="迭代次数，通常10-20次足够"
                )
        
        elif baseline_method == 'arPLS':
            st.write("*arPLS 参数设置*")
            col1, col2, col3 = st.columns(3)
            with col1:
                baseline_params['lam'] = st.selectbox(
                    "λ (平滑度)", 
                    [1e3, 1e4, 1e5, 1e6, 1e7], 
                    index=2,
                    help="数值越大，基线越平滑"
                )
            with col2:
                baseline_params['ratio'] = st.selectbox(
                    "收敛比例", 
                    [0.001, 0.01, 0.1], 
                    index=1,
                    help="收敛判断阈值"
                )
            with col3:
                baseline_params['niter'] = st.slider(
                    "最大迭代次数", 
                    5, 50, 10,
                    help="最大迭代次数"
                )
        
        elif baseline_method == 'polynomial':
            st.write("*多项式参数设置*")
            baseline_params['degree'] = st.slider(
                "多项式阶数", 
                1, 8, 2,
                help="阶数越高，基线越灵活，但可能过拟合"
            )
        
        elif baseline_method == 'rolling_ball':
            st.write("*滚球参数设置*")
            baseline_params['radius'] = st.slider(
                "球半径", 
                10, 500, 100,
                help="球半径越大，基线越平滑"
            )
        
        # linear方法无需额外参数
    
    # 4. 归一化
    st.write("**4. 归一化**")
    apply_normalize = st.checkbox("启用归一化", value=True)
    
    if apply_normalize:
        normalize_method = st.selectbox(
            "归一化方法",
            ['area', 'max', 'vector', 'minmax', 'std'],
            format_func=lambda x: {
                'area': '面积归一化',
                'max': '最大值归一化',
                'vector': '向量归一化 (L2)',
                'minmax': '最小-最大归一化',
                'std': '标准化 (零均值单位方差)'
            }[x]
        )
    
    # 5. SNV变换
    st.write("**5. 标准正态变量变换 (SNV)**")
    apply_snv = st.checkbox("启用SNV变换", value=False)
    
    # 预览设置
    if st.checkbox("预览基线校正效果", value=False):
        if apply_baseline:
            try:
                # 选择一个样本进行预览
                preview_idx = st.selectbox("选择预览样本", range(min(5, X.shape[0])), format_func=lambda x: f"样本 {x+1}")
                
                # 获取截取后的数据
                start_idx = np.argmin(np.abs(wavenumbers - start_wavenumber))
                end_idx = np.argmin(np.abs(wavenumbers - end_wavenumber))
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                wavenumbers_crop = wavenumbers[start_idx:end_idx+1]
                spectrum_crop = X[preview_idx, start_idx:end_idx+1]
                
                # 应用平滑
                if apply_smooth:
                    spectrum_smooth = savgol_filter(spectrum_crop, smooth_window, smooth_poly)
                else:
                    spectrum_smooth = spectrum_crop
                
                # 基线校正预览
                corrector = SpectrumBaselineCorrector()
                baseline, corrected = corrector.correct_baseline(
                    spectrum_smooth, 
                    baseline_method, 
                    **baseline_params
                )
                
                # 绘制预览图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # 原始光谱和基线
                ax1.plot(wavenumbers_crop, spectrum_smooth, 'b-', label='平滑后光谱', alpha=0.8)
                ax1.plot(wavenumbers_crop, baseline, 'r--', label='估计基线', linewidth=2)
                ax1.set_title(f'样本 {preview_idx+1} - 基线校正预览')
                ax1.set_xlabel('波数 (cm⁻¹)')
                ax1.set_ylabel('强度')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 校正后光谱
                ax2.plot(wavenumbers_crop, corrected, 'g-', label='基线校正后', alpha=0.8)
                ax2.set_title('基线校正后光谱')
                ax2.set_xlabel('波数 (cm⁻¹)')
                ax2.set_ylabel('校正后强度')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"预览失败: {e}")
    
    # 开始预处理
    if st.button("🚀 开始预处理", type="primary"):
        with st.spinner("正在进行数据预处理..."):
            try:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # 保存预处理参数
                preprocessing_params = {
                    'start_wavenumber': start_wavenumber,
                    'end_wavenumber': end_wavenumber,
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
                
                st.session_state.preprocessing_params = preprocessing_params
                
                # 1. 波数截取
                progress_text.text("步骤 1/5: 波数截取...")
                start_idx = np.argmin(np.abs(wavenumbers - start_wavenumber))
                end_idx = np.argmin(np.abs(wavenumbers - end_wavenumber))
                
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                wavenumbers_crop = wavenumbers[start_idx:end_idx+1]
                X_crop = X[:, start_idx:end_idx+1]
                progress_bar.progress(0.2)
                
                # 2. 平滑处理
                progress_text.text("步骤 2/5: 平滑处理...")
                if apply_smooth:
                    X_smooth = np.zeros_like(X_crop)
                    for i in range(X_crop.shape[0]):
                        X_smooth[i] = savgol_filter(X_crop[i], smooth_window, smooth_poly)
                else:
                    X_smooth = X_crop.copy()
                progress_bar.progress(0.4)
                
                # 3. 基线校正 - 使用完整的基线校正器
                progress_text.text("步骤 3/5: 基线校正...")
                if apply_baseline:
                    corrector = SpectrumBaselineCorrector()
                    X_corrected = np.zeros_like(X_smooth)
                    failed_samples = []
                    
                    for i in range(X_smooth.shape[0]):
                        try:
                            baseline, corrected = corrector.correct_baseline(
                                X_smooth[i], 
                                baseline_method, 
                                **baseline_params
                            )
                            X_corrected[i] = corrected
                        except Exception as e:
                            failed_samples.append(i+1)
                            X_corrected[i] = X_smooth[i]  # 使用平滑后的原始数据
                            # 不在循环中显示警告，避免界面混乱
                    
                    # 统一显示失败信息
                    if failed_samples:
                        if len(failed_samples) <= 5:
                            st.warning(f"样本 {', '.join(map(str, failed_samples))} 基线校正失败，使用平滑后数据")
                        else:
                            st.warning(f"共有 {len(failed_samples)} 个样本基线校正失败，使用平滑后数据")
                else:
                    X_corrected = X_smooth.copy()
                progress_bar.progress(0.6)
                
                # 4. 归一化
                progress_text.text("步骤 4/5: 归一化...")
                if apply_normalize:
                    X_normalized = np.zeros_like(X_corrected)
                    
                    for i in range(X_corrected.shape[0]):
                        spectrum = X_corrected[i]
                        
                        if normalize_method == 'area':
                            # 修正：使用绝对值计算面积，避免负面积问题
                            total_area = np.trapz(np.abs(spectrum), wavenumbers_crop)
                            if total_area < 1e-12:
                                X_normalized[i] = spectrum
                            else:
                                X_normalized[i] = spectrum / total_area
                        
                        elif normalize_method == 'max':
                            max_abs_val = np.max(np.abs(spectrum))
                            if max_abs_val < 1e-12:
                                X_normalized[i] = spectrum
                            else:
                                X_normalized[i] = spectrum / max_abs_val
                        
                        elif normalize_method == 'vector':
                            norm_val = np.linalg.norm(spectrum)
                            if norm_val < 1e-12:
                                X_normalized[i] = spectrum
                            else:
                                X_normalized[i] = spectrum / norm_val
                        
                        elif normalize_method == 'minmax':
                            min_val = np.min(spectrum)
                            max_val = np.max(spectrum)
                            if abs(max_val - min_val) < 1e-12:
                                X_normalized[i] = spectrum
                            else:
                                X_normalized[i] = (spectrum - min_val) / (max_val - min_val)
                        
                        elif normalize_method == 'std':
                            mean_val = np.mean(spectrum)
                            std_val = np.std(spectrum)
                            if std_val < 1e-12:
                                X_normalized[i] = spectrum - mean_val
                            else:
                                X_normalized[i] = (spectrum - mean_val) / std_val
                        
                        else:
                            X_normalized[i] = spectrum
                else:
                    X_normalized = X_corrected.copy()
                progress_bar.progress(0.8)
                
                # 5. SNV变换
                progress_text.text("步骤 5/5: SNV变换...")
                if apply_snv:
                    X_final = np.zeros_like(X_normalized)
                    for i, spectrum in enumerate(X_normalized):
                        mean_val = np.mean(spectrum)
                        std_val = np.std(spectrum)
                        if std_val < 1e-12:
                            X_final[i] = spectrum
                        else:
                            X_final[i] = (spectrum - mean_val) / std_val
                else:
                    X_final = X_normalized
                progress_bar.progress(1.0)
                
                # 保存结果
                st.session_state.X_preprocessed = X_final
                st.session_state.wavenumbers_preprocessed = wavenumbers_crop
                st.session_state.preprocessing_done = True
                
                progress_text.text("预处理完成！")
                
                st.success("🎉 数据预处理完成！")
                st.info(f"📊 预处理后数据形状: {X_final.shape}")
                st.info(f"📏 波数范围: {wavenumbers_crop.min():.1f} ~ {wavenumbers_crop.max():.1f} cm⁻¹")
                
                # 显示预处理步骤总结
                with st.expander("查看预处理步骤总结"):
                    steps_summary = []
                    steps_summary.append(f"✅ 波数截取: {start_wavenumber:.1f} ~ {end_wavenumber:.1f} cm⁻¹")
                    
                    if apply_smooth:
                        steps_summary.append(f"✅ Savitzky-Golay平滑: 窗口={smooth_window}, 多项式阶数={smooth_poly}")
                    else:
                        steps_summary.append("⭕ 平滑处理: 未启用")
                    
                    if apply_baseline:
                        method_name = {
                            'als': 'ALS (非对称最小二乘)',
                            'arPLS': 'arPLS (非对称重加权惩罚最小二乘)',
                            'polynomial': '多项式拟合',
                            'linear': '线性基线',
                            'rolling_ball': '滚球基线校正'
                        }[baseline_method]
                        steps_summary.append(f"✅ 基线校正: {method_name}")
                    else:
                        steps_summary.append("⭕ 基线校正: 未启用")
                    
                    if apply_normalize:
                        method_name = {
                            'area': '面积归一化',
                            'max': '最大值归一化',
                            'vector': '向量归一化 (L2)',
                            'minmax': '最小-最大归一化',
                            'std': '标准化 (零均值单位方差)'
                        }[normalize_method]
                        steps_summary.append(f"✅ 归一化: {method_name}")
                    else:
                        steps_summary.append("⭕ 归一化: 未启用")
                    
                    if apply_snv:
                        steps_summary.append("✅ SNV变换: 已启用")
                    else:
                        steps_summary.append("⭕ SNV变换: 未启用")
                    
                    for step in steps_summary:
                        st.write(step)
                
                # 显示预处理前后对比
                show_preprocessing_comparison(X, X_final, wavenumbers, wavenumbers_crop)
                
            except Exception as e:
                st.error(f"预处理过程中出错: {e}")
                st.error(traceback.format_exc())


def show_preprocessing_comparison(X_original, X_processed, wavenumbers_original, wavenumbers_processed):
    """显示预处理前后对比"""
    st.subheader("📈 预处理效果对比")
    
    # 选择要显示的样本
    n_samples = min(5, X_original.shape[0])
    sample_indices = np.linspace(0, X_original.shape[0]-1, n_samples, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 原始光谱
    for i in sample_indices:
        ax1.plot(wavenumbers_original, X_original[i], alpha=0.7, label=f'样本 {i+1}')
    ax1.set_title('原始光谱')
    ax1.set_xlabel('波数 (cm⁻¹)')
    ax1.set_ylabel('强度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 预处理后光谱
    for i in sample_indices:
        ax2.plot(wavenumbers_processed, X_processed[i], alpha=0.7, label=f'样本 {i+1}')
    ax2.set_title('预处理后光谱')
    ax2.set_xlabel('波数 (cm⁻¹)')
    ax2.set_ylabel('处理后强度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def show_feature_extraction_page():
    """特征提取与可视化页面"""
    st.markdown("<h1 class='section-header'>特征提取与可视化</h1>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not check_data_prerequisites(need_labels=False, need_preprocessing=True):
        return
    
    st.markdown("""
    <div class="info-box">
    进行特征选择以提取最重要的光谱特征，提高模型性能并减少计算复杂度。
    </div>
    """, unsafe_allow_html=True)
    
    X = st.session_state.X_preprocessed
    wavenumbers = st.session_state.wavenumbers_preprocessed
    
    st.info(f"📊 预处理后数据形状: {X.shape}")
    
    # 特征选择方法
    st.subheader("🔍 特征选择")
    
    feature_method = st.selectbox(
        "选择特征选择方法",
        ["不进行特征选择", "方差过滤", "单变量选择", "递归特征消除", "随机森林重要性"],
        help="不同的特征选择方法适用于不同场景"
    )
    
    if feature_method != "不进行特征选择":
        # 检查是否有标签数据
        if not hasattr(st.session_state, 'y') or st.session_state.y is None:
            st.warning("大部分特征选择方法需要标签数据，当前仅可使用方差过滤")
            if feature_method != "方差过滤":
                st.stop()
        
        # 特征选择参数
        if feature_method == "方差过滤":
            threshold = st.slider("方差阈值", 0.0, 1.0, 0.01, 0.01)
        
        elif feature_method == "单变量选择":
            k_features = st.slider("选择特征数量", 10, min(X.shape[1], 1000), min(100, X.shape[1]//2))
        
        elif feature_method == "递归特征消除":
            n_features = st.slider("目标特征数量", 10, min(X.shape[1], 500), min(50, X.shape[1]//4))
        
        elif feature_method == "随机森林重要性":
            n_features = st.slider("选择特征数量", 10, min(X.shape[1], 500), min(100, X.shape[1]//2))
            threshold = st.slider("重要性阈值", 0.0, 0.01, 0.001, 0.0001)
    
    # 执行特征选择
    if st.button("🚀 执行特征选择"):
        if feature_method == "不进行特征选择":
            st.session_state.X_final = X
            st.session_state.wavenumbers_final = wavenumbers
            st.session_state.selected_features = np.arange(X.shape[1])
            st.session_state.feature_selection_method = feature_method
            st.session_state.feature_selected = True
            
            st.success("✅ 使用全部预处理后的特征")
            st.info(f"📊 最终特征数量: {X.shape[1]}")
        
        else:
            with st.spinner("正在进行特征选择..."):
                try:
                    if feature_method == "方差过滤":
                        from sklearn.feature_selection import VarianceThreshold
                        
                        selector = VarianceThreshold(threshold=threshold)
                        X_selected = selector.fit_transform(X)
                        selected_features = selector.get_support(indices=True)
                    
                    elif feature_method == "单变量选择":
                        from sklearn.feature_selection import SelectKBest, f_regression
                        
                        y = st.session_state.y
                        if y.ndim > 1 and y.shape[1] > 1:
                            # 多输出情况，使用第一个目标变量
                            y_temp = y[:, 0]
                        else:
                            y_temp = y.ravel() if y.ndim > 1 else y
                        
                        selector = SelectKBest(score_func=f_regression, k=k_features)
                        X_selected = selector.fit_transform(X, y_temp)
                        selected_features = selector.get_support(indices=True)
                    
                    elif feature_method == "递归特征消除":
                        from sklearn.feature_selection import RFE
                        from sklearn.linear_model import LinearRegression
                        
                        y = st.session_state.y
                        if y.ndim > 1 and y.shape[1] > 1:
                            y_temp = y[:, 0]
                        else:
                            y_temp = y.ravel() if y.ndim > 1 else y
                        
                        estimator = LinearRegression()
                        selector = RFE(estimator, n_features_to_select=n_features)
                        X_selected = selector.fit_transform(X, y_temp)
                        selected_features = selector.get_support(indices=True)
                    
                    elif feature_method == "随机森林重要性":
                        from sklearn.ensemble import RandomForestRegressor
                        
                        y = st.session_state.y
                        if y.ndim > 1 and y.shape[1] > 1:
                            y_temp = y[:, 0]
                        else:
                            y_temp = y.ravel() if y.ndim > 1 else y
                        
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X, y_temp)
                        
                        importances = rf.feature_importances_
                        
                        # 方法1：按重要性阈值选择
                        mask1 = importances > threshold
                        
                        # 方法2：选择top-k特征
                        indices = np.argsort(importances)[::-1][:n_features]
                        mask2 = np.zeros(X.shape[1], dtype=bool)
                        mask2[indices] = True
                        
                        # 取并集
                        selected_features = np.where(mask1 | mask2)[0]
                        X_selected = X[:, selected_features]
                    
                    # 保存结果
                    st.session_state.X_final = X_selected
                    st.session_state.wavenumbers_final = wavenumbers.iloc[selected_features] if isinstance(wavenumbers, pd.Series) else wavenumbers[selected_features]
                    st.session_state.selected_features = selected_features
                    st.session_state.feature_selection_method = feature_method
                    st.session_state.feature_selected = True
                    
                    st.success("🎉 特征选择完成！")
                    st.info(f"📊 原始特征数量: {X.shape[1]}")
                    st.info(f"📊 选择特征数量: {X_selected.shape[1]}")
                    st.info(f"📈 特征压缩比: {X_selected.shape[1]/X.shape[1]:.2%}")
                    
                    # 显示特征选择结果
                    show_feature_selection_results(X, X_selected, wavenumbers, selected_features, feature_method)
                    
                except Exception as e:
                    st.error(f"特征选择过程中出错: {e}")
                    st.error(traceback.format_exc())
    
    # 如果已经完成特征选择，显示可视化
    if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
        show_feature_visualization()


def show_feature_selection_results(X_original, X_selected, wavenumbers, selected_features, method):
    """显示特征选择结果"""
    st.subheader("📊 特征选择结果")
    
    # 特征重要性可视化
    if method == "随机森林重要性":
        # 显示特征重要性图
        from sklearn.ensemble import RandomForestRegressor
        
        y = st.session_state.y
        if y.ndim > 1 and y.shape[1] > 1:
            y_temp = y[:, 0]
        else:
            y_temp = y.ravel() if y.ndim > 1 else y
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_original, y_temp)
        
        importances = rf.feature_importances_
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 全部特征重要性
        ax1.plot(wavenumbers, importances, alpha=0.7)
        ax1.set_title('特征重要性分布')
        ax1.set_xlabel('波数 (cm⁻¹)')
        ax1.set_ylabel('重要性')
        ax1.grid(True, alpha=0.3)
        
        # 选择的特征
        selected_wavenumbers = wavenumbers.iloc[selected_features] if isinstance(wavenumbers, pd.Series) else wavenumbers[selected_features]
        selected_importances = importances[selected_features]
        
        ax2.scatter(selected_wavenumbers, selected_importances, c='red', alpha=0.7)
        ax2.set_title('选择的特征及其重要性')
        ax2.set_xlabel('波数 (cm⁻¹)')
        ax2.set_ylabel('重要性')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 特征分布对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 原始光谱（取前几个样本）
    n_samples = min(3, X_original.shape[0])
    for i in range(n_samples):
        ax1.plot(wavenumbers, X_original[i], alpha=0.7, label=f'样本 {i+1}')
    ax1.set_title('原始预处理后光谱（全部特征）')
    ax1.set_xlabel('波数 (cm⁻¹)')
    ax1.set_ylabel('强度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 选择特征的光谱
    selected_wavenumbers = wavenumbers.iloc[selected_features] if isinstance(wavenumbers, pd.Series) else wavenumbers[selected_features]
    for i in range(n_samples):
        ax2.plot(selected_wavenumbers, X_selected[i], alpha=0.7, label=f'样本 {i+1}')
    ax2.set_title('特征选择后光谱')
    ax2.set_xlabel('波数 (cm⁻¹)')
    ax2.set_ylabel('强度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def show_feature_visualization():
    """显示特征可视化"""
    st.subheader("📈 数据可视化")
    
    X = st.session_state.X_final
    wavenumbers = st.session_state.wavenumbers_final
    
    # 统一转换为numpy数组以避免索引问题
    if isinstance(wavenumbers, pd.Series):
        wavenumbers = wavenumbers.values
    elif not isinstance(wavenumbers, np.ndarray):
        wavenumbers = np.array(wavenumbers)
    
    tab1, tab2, tab3 = st.tabs(["光谱图", "统计分析", "相关性分析"])
    
    with tab1:
        st.write("### 🌈 光谱图可视化")
        
        # 显示选项
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples_show = st.slider("显示样本数量", 1, min(20, X.shape[0]), min(5, X.shape[0]))
        
        with col2:
            sample_selection = st.selectbox("样本选择方式", ["均匀分布", "随机选择", "前N个样本"])
        
        # 选择样本
        if sample_selection == "均匀分布":
            indices = np.linspace(0, X.shape[0]-1, n_samples_show, dtype=int)
        elif sample_selection == "随机选择":
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], n_samples_show, replace=False)
        else:  # 前N个样本
            indices = np.arange(min(n_samples_show, X.shape[0]))
        
        # 绘制光谱图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, idx in enumerate(indices):
            ax.plot(wavenumbers, X[idx], alpha=0.7, label=f'样本 {idx+1}')
        
        ax.set_title('光谱图')
        ax.set_xlabel('波数 (cm⁻¹)')
        ax.set_ylabel('强度')
        if len(indices) <= 10:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.write("### 📊 统计分析")
        
        # 计算统计量
        mean_spectrum = np.mean(X, axis=0)
        std_spectrum = np.std(X, axis=0)
        min_spectrum = np.min(X, axis=0)
        max_spectrum = np.max(X, axis=0)
        
        # 绘制统计图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 均值光谱
        ax1.plot(wavenumbers, mean_spectrum, color='blue')
        ax1.fill_between(wavenumbers, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, 
                        alpha=0.3, color='blue')
        ax1.set_title('平均光谱 ± 标准差')
        ax1.set_xlabel('波数 (cm⁻¹)')
        ax1.set_ylabel('强度')
        ax1.grid(True, alpha=0.3)
        
        # 标准差
        ax2.plot(wavenumbers, std_spectrum, color='red')
        ax2.set_title('标准差分布')
        ax2.set_xlabel('波数 (cm⁻¹)')
        ax2.set_ylabel('标准差')
        ax2.grid(True, alpha=0.3)
        
        # 最值范围
        ax3.fill_between(wavenumbers, min_spectrum, max_spectrum, alpha=0.5, color='green')
        ax3.plot(wavenumbers, mean_spectrum, color='black', linewidth=2, label='均值')
        ax3.set_title('数据范围 (最小值-最大值)')
        ax3.set_xlabel('波数 (cm⁻¹)')
        ax3.set_ylabel('强度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 箱线图（选择几个代表性波数）
        n_wavenumber_samples = min(20, len(wavenumbers))
        wn_indices = np.linspace(0, len(wavenumbers)-1, n_wavenumber_samples, dtype=int)
        
        box_data = [X[:, i] for i in wn_indices]
        box_labels = [f'{wavenumbers[i]:.0f}' for i in wn_indices]
        
        ax4.boxplot(box_data, labels=box_labels)
        ax4.set_title('强度分布箱线图（部分波数）')
        ax4.set_xlabel('波数 (cm⁻¹)')
        ax4.set_ylabel('强度')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示数值统计
        st.write("**数值统计摘要：**")
        stats_df = pd.DataFrame({
            '统计量': ['样本数量', '特征数量', '平均强度均值', '平均强度标准差', '最小强度', '最大强度'],
            '数值': [
                X.shape[0],
                X.shape[1],
                f"{np.mean(mean_spectrum):.6f}",
                f"{np.mean(std_spectrum):.6f}",
                f"{np.min(X):.6f}",
                f"{np.max(X):.6f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.write("### 🔗 相关性分析")
        
        if hasattr(st.session_state, 'y') and st.session_state.y is not None:
            y = st.session_state.y
            
            # 计算相关系数
            if y.ndim == 1:
                correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
                
                # 绘制相关性图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # 相关系数曲线
                ax1.plot(wavenumbers, correlations, color='purple')
                ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax1.set_title('光谱特征与目标变量的相关性')
                ax1.set_xlabel('波数 (cm⁻¹)')
                ax1.set_ylabel('相关系数')
                ax1.grid(True, alpha=0.3)
                
                # 相关系数分布直方图
                ax2.hist(correlations, bins=30, alpha=0.7, color='purple')
                ax2.set_title('相关系数分布')
                ax2.set_xlabel('相关系数')
                ax2.set_ylabel('频数')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示高相关性特征
                high_corr_indices = np.where(np.abs(correlations) > 0.5)[0]
                if len(high_corr_indices) > 0:
                    st.write("**高相关性特征 (|r| > 0.5)：**")
                    high_corr_df = pd.DataFrame({
                        '波数': [wavenumbers[i] if i < len(wavenumbers) else f"特征{i}" for i in high_corr_indices],
                        '相关系数': correlations[high_corr_indices].round(4)
                    })
                    high_corr_df = high_corr_df.sort_values('相关系数', key=abs, ascending=False)
                    st.dataframe(high_corr_df, use_container_width=True)
                else:
                    st.info("未发现高相关性特征 (|r| > 0.5)")
            
            else:
                # 多目标情况
                st.write("**多目标变量相关性分析：**")
                target_names = st.session_state.selected_cols
                
                selected_target = st.selectbox("选择目标变量查看相关性", range(len(target_names)), 
                                             format_func=lambda x: target_names[x])
                
                correlations = np.array([np.corrcoef(X[:, i], y[:, selected_target])[0, 1] 
                                       for i in range(X.shape[1])])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(wavenumbers, correlations, color='purple')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title(f'光谱特征与 {target_names[selected_target]} 的相关性')
                ax.set_xlabel('波数 (cm⁻¹)')
                ax.set_ylabel('相关系数')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.info("需要标签数据才能进行相关性分析")


def show_trend_analysis_page():
    """趋势分析页面"""
    st.markdown("<h1 class='section-header'>趋势分析</h1>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not check_data_prerequisites(need_labels=False, need_preprocessing=True):
        return
    
    st.markdown("""
    <div class="info-box">
    进行光谱数据的趋势分析，包括PCA降维、成分分解、时间趋势等多种分析方法。
    适用于有标签或无标签数据的探索性分析。
    </div>
    """, unsafe_allow_html=True)
    
    X, wavenumbers, data_info = get_current_data()
    show_status_message(data_info, "info")
    
    # 分析方法选择
    analysis_tabs = st.tabs([
        "PCA分析", 
        "成分分解", 
        "时间趋势", 
        "聚类分析", 
        "异常检测",
        "综合报告"
    ])
    
    with analysis_tabs[0]:
        show_pca_analysis(X, wavenumbers)
    
    with analysis_tabs[1]:
        show_component_decomposition(X, wavenumbers)
    
    with analysis_tabs[2]:
        show_time_trend_analysis(X, wavenumbers)
    
    with analysis_tabs[3]:
        show_clustering_analysis(X, wavenumbers)
    
    with analysis_tabs[4]:
        show_anomaly_detection(X, wavenumbers)
    
    with analysis_tabs[5]:
        show_comprehensive_report(X, wavenumbers)


def show_pca_analysis(X, wavenumbers):
    """PCA分析"""
    st.write("### 🔍 主成分分析 (PCA)")
     # 转换为numpy数组以避免索引问题
    if isinstance(wavenumbers, pd.Series):
        wavenumbers = wavenumbers.values
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # PCA参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider(
            "主成分数量", 
            2, min(10, X.shape[1], X.shape[0]), 
            min(5, X.shape[1], X.shape[0])
        )
    
    with col2:
        standardize = st.checkbox("数据标准化", value=True)
    
    if st.button("执行PCA分析"):
        with st.spinner("正在进行PCA分析..."):
            try:
                # 数据预处理
                if standardize:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                # 执行PCA
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # 结果展示
                st.success(f"✅ PCA分析完成！解释方差比: {pca.explained_variance_ratio_.sum():.2%}")
                
                # 1. 解释方差比图
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # 各主成分解释方差比
                ax1.bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.7)
                ax1.set_title('各主成分解释方差比')
                ax1.set_xlabel('主成分')
                ax1.set_ylabel('解释方差比')
                ax1.grid(True, alpha=0.3)
                
                # 累积解释方差比
                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                ax2.plot(range(1, n_components+1), cumsum_var, 'o-', color='red')
                ax2.set_title('累积解释方差比')
                ax2.set_xlabel('主成分数量')
                ax2.set_ylabel('累积解释方差比')
                ax2.grid(True, alpha=0.3)
                
                # 第一主成分载荷
                ax3.plot(wavenumbers, pca.components_[0], color='blue')
                ax3.set_title('第一主成分载荷')
                ax3.set_xlabel('波数 (cm⁻¹)')
                ax3.set_ylabel('载荷')
                ax3.grid(True, alpha=0.3)
                
                # 前两个主成分得分散点图
                ax4.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
                ax4.set_title('前两个主成分得分图')
                ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 2. 主成分载荷热图
                if n_components >= 3:
                    st.write("### 📊 主成分载荷热图")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # 选择显示的波数点（如果太多的话）
                    if len(wavenumbers) > 200:
                        step = len(wavenumbers) // 200
                        wn_indices = slice(None, None, step)
                        display_wavenumbers = wavenumbers[wn_indices]
                        display_components = pca.components_[:, wn_indices]
                    else:
                        display_wavenumbers = wavenumbers
                        display_components = pca.components_
                    
                    im = ax.imshow(display_components, aspect='auto', cmap='RdBu_r')
                    ax.set_title('主成分载荷热图')
                    ax.set_xlabel('波数 (cm⁻¹)')
                    ax.set_ylabel('主成分')
                    
                    # 设置x轴标签
                    n_ticks = 10
                    tick_indices = np.linspace(0, len(display_wavenumbers)-1, n_ticks, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([f'{display_wavenumbers[i]:.0f}' for i in tick_indices])
                    
                    # 设置y轴标签
                    ax.set_yticks(range(n_components))
                    ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
                    
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 3. 数据表格
                st.write("### 📋 PCA结果摘要")
                
                pca_summary = pd.DataFrame({
                    '主成分': [f'PC{i+1}' for i in range(n_components)],
                    '解释方差比': [f'{ratio:.4f}' for ratio in pca.explained_variance_ratio_],
                    '累积解释方差比': [f'{cum:.4f}' for cum in cumsum_var],
                    '特征值': [f'{val:.4f}' for val in pca.explained_variance_]
                })
                
                st.dataframe(pca_summary, use_container_width=True)
                
                # 4. 保存结果到session state
                st.session_state.pca_results = {
                    'pca': pca,
                    'X_pca': X_pca,
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'components': pca.components_
                }
                
                # 5. 如果有标签数据，显示标签与主成分的关系
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    show_pca_label_relationship(X_pca, pca.explained_variance_ratio_)
                
            except Exception as e:
                st.error(f"PCA分析出错: {e}")
                st.error(traceback.format_exc())


def show_pca_label_relationship(X_pca, explained_var_ratio):
    """显示PCA与标签的关系"""
    st.write("### 🎯 主成分与标签关系")
    
    y = st.session_state.y
    
    if y.ndim == 1:
        # 单标签情况
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PC1 vs 标签
        ax1.scatter(X_pca[:, 0], y, alpha=0.6)
        corr1 = np.corrcoef(X_pca[:, 0], y)[0, 1]
        ax1.set_title(f'PC1 vs 标签 (r={corr1:.3f})')
        ax1.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%})')
        ax1.set_ylabel('标签值')
        ax1.grid(True, alpha=0.3)
        
        # PC2 vs 标签
        if X_pca.shape[1] > 1:
            ax2.scatter(X_pca[:, 1], y, alpha=0.6)
            corr2 = np.corrcoef(X_pca[:, 1], y)[0, 1]
            ax2.set_title(f'PC2 vs 标签 (r={corr2:.3f})')
            ax2.set_xlabel(f'PC2 ({explained_var_ratio[1]:.1%})')
            ax2.set_ylabel('标签值')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # 多标签情况
        target_names = st.session_state.selected_cols
        selected_target = st.selectbox(
            "选择目标变量查看与主成分的关系", 
            range(len(target_names)), 
            format_func=lambda x: target_names[x]
        )
        
        y_selected = y[:, selected_target]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PC1 vs 选择的标签
        ax1.scatter(X_pca[:, 0], y_selected, alpha=0.6)
        corr1 = np.corrcoef(X_pca[:, 0], y_selected)[0, 1]
        ax1.set_title(f'PC1 vs {target_names[selected_target]} (r={corr1:.3f})')
        ax1.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%})')
        ax1.set_ylabel(target_names[selected_target])
        ax1.grid(True, alpha=0.3)
        
        # PC2 vs 选择的标签
        if X_pca.shape[1] > 1:
            ax2.scatter(X_pca[:, 1], y_selected, alpha=0.6)
            corr2 = np.corrcoef(X_pca[:, 1], y_selected)[0, 1]
            ax2.set_title(f'PC2 vs {target_names[selected_target]} (r={corr2:.3f})')
            ax2.set_xlabel(f'PC2 ({explained_var_ratio[1]:.1%})')
            ax2.set_ylabel(target_names[selected_target])
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


def show_component_decomposition(X, wavenumbers):
    """成分分解分析"""
    st.write("### 🧪 成分分解分析")
    
    st.markdown("""
    使用矩阵分解技术分离光谱中的化学成分信息，适用于：
    - 混合物成分识别
    - 主副产物分离
    - 化学过程监控
    """)
    
    # 分解方法选择
    decomp_method = st.selectbox(
        "分解方法",
        ["NMF", "ICA", "Factor Analysis"],
        format_func=lambda x: {
            "NMF": "非负矩阵分解 (NMF)",
            "ICA": "独立成分分析 (ICA)",
            "Factor Analysis": "因子分析"
        }[x],
        key="decomp_method_select"
    )
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider("成分数量", 2, min(10, X.shape[0], X.shape[1]), 3, key="decomp_n_components")
    
    with col2:
        if decomp_method == "NMF":
            max_iter = st.slider("最大迭代次数", 100, 1000, 200, key="decomp_nmf_max_iter")
        elif decomp_method == "ICA":
            max_iter = st.slider("最大迭代次数", 100, 1000, 200, key="decomp_ica_max_iter")
            tolerance = st.selectbox("收敛容差", [1e-3, 1e-4, 1e-5], index=0, key="decomp_ica_tol")
        else:
            max_iter = None
    
    if st.button("执行成分分解", key="decomp_execute_button"):
        with st.spinner("正在进行成分分解..."):
            try:
                # 数据预处理检查
                st.write("### 数据预处理检查:")
                
                has_nan = np.isnan(X).any()
                has_inf = np.isinf(X).any()
                st.write(f"- 是否包含NaN: {has_nan}")
                st.write(f"- 是否包含无穷大: {has_inf}")
                st.write(f"- 数据范围: [{X.min():.3f}, {X.max():.3f}]")
                st.write(f"- 数据形状: {X.shape}")
                
                # 清理数据
                X_clean = X.copy()
                
                if has_nan or has_inf:
                    st.warning("检测到异常值，正在清理...")
                    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=X_clean[np.isfinite(X_clean)].max(), 
                                          neginf=X_clean[np.isfinite(X_clean)].min())
                    st.info("已清理异常值")
                
                # ⭐ 关键修复：数据降维和正则化处理 ⭐
                if decomp_method == "ICA":
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    
                    # 1. 标准化
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_clean)
                    st.info("ICA: 已应用标准化")
                    
                    # 2. 检查数据的有效维度
                    try:
                        # 使用SVD检查有效秩
                        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
                        
                        # 找到非零奇异值的数量（有效秩）
                        tolerance_svd = 1e-10
                        effective_rank = np.sum(s > tolerance_svd)
                        
                        st.write(f"- 数据有效秩: {effective_rank}")
                        st.write(f"- 奇异值范围: [{s.min():.2e}, {s.max():.2e}]")
                        
                        # 如果有效秩小于成分数量，调整成分数量
                        if effective_rank < n_components:
                            n_components = min(effective_rank - 1, n_components)
                            st.warning(f"数据有效秩不足，已调整成分数量为: {n_components}")
                        
                        # 如果有效秩太小，使用PCA预处理
                        if effective_rank < min(X_scaled.shape) * 0.8:
                            st.info("使用PCA进行数据预处理以提高数值稳定性")
                            
                            # 保留95%的方差
                            pca_components = min(effective_rank, int(min(X_scaled.shape) * 0.95))
                            pca = PCA(n_components=pca_components)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            st.write(f"- PCA降维: {X_scaled.shape[1]} → {X_pca.shape[1]}")
                            st.write(f"- 保留方差比例: {pca.explained_variance_ratio_.sum():.3f}")
                            
                            X_for_ica = X_pca
                        else:
                            X_for_ica = X_scaled
                            
                    except np.linalg.LinAlgError:
                        st.warning("SVD分析失败，使用简单预处理")
                        X_for_ica = X_scaled
                    
                    # 3. 最终条件数检查
                    try:
                        cond_num = np.linalg.cond(X_for_ica)
                        st.write(f"- 预处理后条件数: {cond_num:.2e}")
                        
                        if cond_num > 1e12:
                            st.warning("条件数仍然过大，将使用更保守的参数")
                            # 进一步降维
                            if X_for_ica.shape[1] > n_components * 3:
                                from sklearn.decomposition import TruncatedSVD
                                svd = TruncatedSVD(n_components=n_components * 3, random_state=42)
                                X_for_ica = svd.fit_transform(X_for_ica)
                                st.info(f"进一步降维至: {X_for_ica.shape[1]} 维")
                    except:
                        st.warning("无法计算预处理后的条件数")
                    
                else:
                    X_for_ica = X_clean
                
                # 执行分解
                if decomp_method == "NMF":
                    from sklearn.decomposition import NMF
                    
                    X_positive = X_clean - X_clean.min() + 1e-6
                    
                    model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
                    W = model.fit_transform(X_positive)
                    H = model.components_
                    X_for_reconstruction = X_positive
                    
                elif decomp_method == "ICA":
                    from sklearn.decomposition import FastICA
                    
                    # ⭐ 多种ICA策略尝试 ⭐
                    ica_success = False
                    
                    # 策略1: 标准ICA with 保守参数
                    try:
                        st.info("尝试策略1: 标准ICA")
                        model = FastICA(
                            n_components=n_components, 
                            max_iter=max_iter, 
                            tol=tolerance,
                            random_state=42,
                            whiten='unit-variance',
                            fun='logcosh',
                            algorithm='parallel'
                        )
                        S = model.fit_transform(X_for_ica)
                        A = model.mixing_
                        ica_success = True
                        
                    except Exception as e1:
                        st.warning(f"策略1失败: {str(e1)[:100]}...")
                        
                        # 策略2: 减少成分数量
                        try:
                            reduced_components = max(2, n_components // 2)
                            st.info(f"尝试策略2: 减少成分数量至 {reduced_components}")
                            
                            model = FastICA(
                                n_components=reduced_components, 
                                max_iter=max_iter, 
                                tol=1e-3,  # 放宽容差
                                random_state=42,
                                whiten='unit-variance',
                                fun='exp',
                                algorithm='deflation'
                            )
                            S = model.fit_transform(X_for_ica)
                            A = model.mixing_
                            n_components = reduced_components
                            ica_success = True
                            
                        except Exception as e2:
                            st.warning(f"策略2失败: {str(e2)[:100]}...")
                            
                            # 策略3: 使用PCA预白化
                            try:
                                st.info("尝试策略3: PCA预白化")
                                from sklearn.decomposition import PCA
                                
                                # 使用PCA进行预白化
                                pca_dim = min(n_components * 2, X_for_ica.shape[1] // 2, X_for_ica.shape[0] // 2)
                                pca = PCA(n_components=pca_dim, whiten=True)
                                X_whitened = pca.fit_transform(X_for_ica)
                                
                                model = FastICA(
                                    n_components=min(n_components, pca_dim),
                                    max_iter=max_iter,
                                    tol=1e-2,  # 进一步放宽容差
                                    random_state=42,
                                    whiten=False,  # 已经白化过了
                                    fun='cube',
                                    algorithm='deflation'
                                )
                                S = model.fit_transform(X_whitened)
                                
                                # 将结果转换回原始空间
                                A = pca.components_.T @ model.mixing_
                                n_components = S.shape[1]
                                ica_success = True
                                
                            except Exception as e3:
                                st.error(f"所有ICA策略都失败了:")
                                st.error(f"策略1: {str(e1)[:50]}...")
                                st.error(f"策略2: {str(e2)[:50]}...")
                                st.error(f"策略3: {str(e3)[:50]}...")
                                st.error("建议尝试其他分解方法（NMF或因子分析）")
                                return
                    
                    if ica_success:
                        W = S
                        H = A.T
                        X_for_reconstruction = X_for_ica
                        st.success(f"ICA成功完成，最终成分数量: {n_components}")
                    
                else:  # Factor Analysis
                    from sklearn.decomposition import FactorAnalysis
                    
                    model = FactorAnalysis(n_components=n_components, random_state=42)
                    W = model.fit_transform(X_clean)
                    H = model.components_
                    X_for_reconstruction = X_clean
                
                # 后续的可视化代码保持不变...
                st.write("### 分解结果:")
                st.write(f"- 最终成分数量: {n_components}")
                st.write(f"- 分解结果 W 形状: {W.shape}")
                st.write(f"- 分解结果 H 形状: {H.shape}")
                
                # 确保波数和成分光谱维度匹配
                if H.shape[1] != len(wavenumbers):
                    st.warning(f"检测到维度不匹配: H维度={H.shape[1]}, 波数长度={len(wavenumbers)}")
                    
                    min_length = min(H.shape[1], len(wavenumbers))
                    wavenumbers_plot = wavenumbers[:min_length]
                    H_plot = H[:, :min_length]
                    
                    st.info(f"已调整绘图数据长度为: {min_length}")
                else:
                    wavenumbers_plot = wavenumbers
                    H_plot = H
                
                st.success(f"✅ {decomp_method} 分解完成！")
                
                # 可视化结果
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. 成分光谱
                ax1 = axes[0, 0]
                for i in range(n_components):
                    ax1.plot(wavenumbers_plot, H_plot[i], label=f'成分 {i+1}', alpha=0.8)
                ax1.set_title('分解得到的成分光谱')
                ax1.set_xlabel('波数 (cm⁻¹)')
                ax1.set_ylabel('强度')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. 样本系数
                ax2 = axes[0, 1]
                for i in range(n_components):
                    ax2.plot(W[:, i], label=f'成分 {i+1}', marker='o', alpha=0.7)
                ax2.set_title('样本中各成分的系数')
                ax2.set_xlabel('样本索引')
                ax2.set_ylabel('系数')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. 成分贡献度
                ax3 = axes[1, 0]
                contributions = np.mean(np.abs(W), axis=0)
                contributions = contributions / contributions.sum() * 100
                ax3.bar(range(1, n_components+1), contributions, alpha=0.7)
                ax3.set_title('各成分平均贡献度')
                ax3.set_xlabel('成分')
                ax3.set_ylabel('贡献度 (%)')
                ax3.grid(True, alpha=0.3)
                
                # 4. 重构误差
                ax4 = axes[1, 1]
                try:
                    X_reconstructed = W @ H
                    if X_reconstructed.shape == X_for_reconstruction.shape:
                        reconstruction_error = np.mean((X_for_reconstruction - X_reconstructed)**2, axis=1)
                        ax4.plot(reconstruction_error, 'o-', alpha=0.7)
                        ax4.set_title('样本重构误差')
                        ax4.set_xlabel('样本索引')
                        ax4.set_ylabel('重构误差')
                    else:
                        ax4.text(0.5, 0.5, f'重构维度不匹配\n原始: {X_for_reconstruction.shape}\n重构: {X_reconstructed.shape}', 
                                ha='center', va='center', transform=ax4.transAxes)
                        ax4.set_title('重构误差 (维度不匹配)')
                        
                except Exception as e:
                    ax4.text(0.5, 0.5, f'重构误差计算失败:\n{str(e)}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('重构误差 (计算失败)')
                
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 成分分析结果表格
                st.write("### 📋 成分分析结果")
                
                component_stats = []
                for i in range(n_components):
                    peak_idx = np.argmax(np.abs(H_plot[i]))
                    peak_wavenumber = wavenumbers_plot[peak_idx]
                    
                    stats = {
                        '成分': f'成分{i+1}',
                        '平均浓度': f"{np.mean(W[:, i]):.3f}",
                        '浓度标准差': f"{np.std(W[:, i]):.3f}",
                        '最大浓度样本': f"样本{np.argmax(np.abs(W[:, i]))+1}",
                        '最大浓度值': f"{np.max(np.abs(W[:, i])):.3f}",
                        '光谱峰值波数': f"{peak_wavenumber:.1f} cm⁻¹"
                    }
                    component_stats.append(stats)
                
                st.dataframe(pd.DataFrame(component_stats), use_container_width=True)
                
                # 保存结果
                st.session_state.decomposition_results = {
                    'method': decomp_method,
                    'model': model,
                    'components': H,
                    'coefficients': W,
                    'n_components': n_components
                }
                
                # 如果有标签，分析成分与标签的关系
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    show_component_label_relationship(W, H, decomp_method)
                
            except Exception as e:
                st.error(f"成分分解出错: {e}")
                st.error(traceback.format_exc())


def show_component_label_relationship(W, H, method):
    """显示成分与标签的关系"""
    st.write(f"### 🎯 {method} 成分与标签关系")
    
    y = st.session_state.y
    
    if y.ndim == 1:
        # 单标签情况
        fig, axes = plt.subplots(1, min(3, W.shape[1]), figsize=(15, 5))
        if W.shape[1] == 1:
            axes = [axes]
        
        for i in range(min(3, W.shape[1])):
            if i < len(axes):
                corr = np.corrcoef(W[:, i], y)[0, 1]
                axes[i].scatter(W[:, i], y, alpha=0.6)
                axes[i].set_title(f'成分 {i+1} vs 标签 (r={corr:.3f})')
                axes[i].set_xlabel(f'成分 {i+1} 系数')
                axes[i].set_ylabel('标签值')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # 多标签情况
        target_names = st.session_state.selected_cols
        selected_target = st.selectbox(
            "选择目标变量查看与成分的关系", 
            range(len(target_names)), 
            format_func=lambda x: target_names[x],
            key="component_target_select"
        )
        
        y_selected = y[:, selected_target]
        
        fig, axes = plt.subplots(1, min(3, W.shape[1]), figsize=(15, 5))
        if W.shape[1] == 1:
            axes = [axes]
        
        for i in range(min(3, W.shape[1])):
            if i < len(axes):
                corr = np.corrcoef(W[:, i], y_selected)[0, 1]
                axes[i].scatter(W[:, i], y_selected, alpha=0.6)
                axes[i].set_title(f'成分 {i+1} vs {target_names[selected_target]} (r={corr:.3f})')
                axes[i].set_xlabel(f'成分 {i+1} 系数')
                axes[i].set_ylabel(target_names[selected_target])
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


def show_time_trend_analysis(X, wavenumbers):
    """时间趋势分析"""
    st.write("### 📈 时间趋势分析")
    
    st.markdown("""
    分析光谱数据随时间的变化趋势，适用于：
    - 反应过程监控
    - 工艺过程分析
    - 样品稳定性研究
    """)
    
    # 时间序列设置
    col1, col2 = st.columns(2)
    
    with col1:
        time_mode = st.selectbox(
            "时间模式",
            ["样本索引作为时间", "等间隔时间序列", "自定义时间"],
            help="选择如何定义时间轴"
        )
    
    with col2:
        if time_mode == "等间隔时间序列":
            time_interval = st.number_input("时间间隔", min_value=0.1, value=1.0, step=0.1)
            time_unit = st.selectbox("时间单位", ["秒", "分钟", "小时", "天"])
        elif time_mode == "自定义时间":
            st.info("请在下方输入时间序列")
    
    # 生成时间轴
    if time_mode == "样本索引作为时间":
        time_axis = np.arange(1, X.shape[0] + 1)
        time_label = "样本索引"
    
    elif time_mode == "等间隔时间序列":
        time_axis = np.arange(0, X.shape[0] * time_interval, time_interval)
        time_label = f"时间 ({time_unit})"
    
    else:  # 自定义时间
        time_input = st.text_area(
            "输入时间序列（每行一个值）",
            value="\n".join(str(i) for i in range(1, min(11, X.shape[0] + 1))),
            height=100
        )
        
        try:
            time_values = [float(line.strip()) for line in time_input.split('\n') if line.strip()]
            if len(time_values) != X.shape[0]:
                st.warning(f"时间点数量 ({len(time_values)}) 与样本数量 ({X.shape[0]}) 不匹配")
                time_axis = np.arange(1, X.shape[0] + 1)
                time_label = "样本索引"
            else:
                time_axis = np.array(time_values)
                time_label = "时间"
        except ValueError:
            st.error("时间序列格式错误，请输入数值")
            time_axis = np.arange(1, X.shape[0] + 1)
            time_label = "样本索引"
    
    # 分析选项
    analysis_options = st.multiselect(
        "选择分析内容",
        ["整体趋势", "特定波数趋势", "光谱演化", "变化率分析", "相关性趋势"],
        default=["整体趋势", "光谱演化"]
    )
    
    # 初始化selected_wn_indices，如果不存在
    if "selected_wn_indices" not in st.session_state:
        n_wavenumbers = min(5, len(wavenumbers))
        st.session_state.selected_wn_indices = np.linspace(0, len(wavenumbers)-1, n_wavenumbers, dtype=int).tolist()
    
    # 特定波数趋势部分的选择控件
    if "特定波数趋势" in analysis_options:
        st.write("### 📍 特定波数点趋势")
        
        # 定义一个回调函数来更新选择的波数点
        def update_selected_wn():
            pass  # 仅用于触发回调，实际更新已经在multiselect控件中完成
        
        # 使用on_change回调函数来避免不必要的刷新
        selected_wn_indices = st.multiselect(
            "选择关注的波数点",
            range(len(wavenumbers)),
            default=st.session_state.selected_wn_indices,
            format_func=lambda x: f"{wavenumbers[x]:.1f} cm⁻¹",
            key="wn_multiselect",
            on_change=update_selected_wn
        )
        
        # 更新session_state
        st.session_state.selected_wn_indices = selected_wn_indices
    
    if st.button("执行趋势分析"):
        with st.spinner("正在进行趋势分析..."):
            try:
                results = {}
                
                if "整体趋势" in analysis_options:
                    # 计算整体光谱指标的时间趋势
                    total_intensity = np.sum(X, axis=1)
                    mean_intensity = np.mean(X, axis=1)
                    max_intensity = np.max(X, axis=1)
                    std_intensity = np.std(X, axis=1)
                    
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    ax1.plot(time_axis, total_intensity, 'o-', alpha=0.7)
                    ax1.set_title('总强度随时间变化')
                    ax1.set_xlabel(time_label)
                    ax1.set_ylabel('总强度')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(time_axis, mean_intensity, 'o-', color='orange', alpha=0.7)
                    ax2.set_title('平均强度随时间变化')
                    ax2.set_xlabel(time_label)
                    ax2.set_ylabel('平均强度')
                    ax2.grid(True, alpha=0.3)
                    
                    ax3.plot(time_axis, max_intensity, 'o-', color='green', alpha=0.7)
                    ax3.set_title('最大强度随时间变化')
                    ax3.set_xlabel(time_label)
                    ax3.set_ylabel('最大强度')
                    ax3.grid(True, alpha=0.3)
                    
                    ax4.plot(time_axis, std_intensity, 'o-', color='red', alpha=0.7)
                    ax4.set_title('强度标准差随时间变化')
                    ax4.set_xlabel(time_label)
                    ax4.set_ylabel('强度标准差')
                    ax4.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    results['overall_trends'] = {
                        'total_intensity': total_intensity,
                        'mean_intensity': mean_intensity,
                        'max_intensity': max_intensity,
                        'std_intensity': std_intensity
                    }
                
                if "特定波数趋势" in analysis_options and st.session_state.selected_wn_indices:
                    # 使用session_state中存储的选择
                    selected_wn_indices = st.session_state.selected_wn_indices
                    
                    if selected_wn_indices:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for idx in selected_wn_indices:
                            ax.plot(time_axis, X[:, idx], 'o-', alpha=0.7, 
                                label=f'{wavenumbers[idx]:.1f} cm⁻¹')
                        
                        ax.set_title('特定波数点强度随时间变化')
                        ax.set_xlabel(time_label)
                        ax.set_ylabel('强度')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # 其他分析选项的代码保持不变
                if "光谱演化" in analysis_options:
                    st.write("### 🌈 光谱演化")
                    
                    # 3D光谱演化图
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 选择部分时间点和波数点以避免图形过于复杂
                    n_time_points = min(10, X.shape[0])
                    n_wavenumber_points = min(100, len(wavenumbers))
                    
                    time_indices = np.linspace(0, X.shape[0]-1, n_time_points, dtype=int)
                    wn_indices = np.linspace(0, len(wavenumbers)-1, n_wavenumber_points, dtype=int)
                    
                    T, W = np.meshgrid(time_axis[time_indices], wavenumbers[wn_indices])
                    Z = X[time_indices][:, wn_indices].T
                    
                    ax.plot_surface(T, W, Z, cmap='viridis', alpha=0.8)
                    ax.set_xlabel(time_label)
                    ax.set_ylabel('波数 (cm⁻¹)')
                    ax.set_zlabel('强度')
                    ax.set_title('光谱随时间演化 (3D视图)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 2D热图
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    im = ax.imshow(X.T, aspect='auto', cmap='viridis', interpolation='nearest')
                    ax.set_title('光谱演化热图')
                    ax.set_xlabel('时间点')
                    ax.set_ylabel('波数 (cm⁻¹)')
                    
                    # 设置坐标轴标签
                    n_time_ticks = min(10, len(time_axis))
                    time_tick_indices = np.linspace(0, len(time_axis)-1, n_time_ticks, dtype=int)
                    ax.set_xticks(time_tick_indices)
                    ax.set_xticklabels([f'{time_axis[i]:.1f}' for i in time_tick_indices])
                    
                    n_wn_ticks = min(10, len(wavenumbers))
                    wn_tick_indices = np.linspace(0, len(wavenumbers)-1, n_wn_ticks, dtype=int)
                    ax.set_yticks(wn_tick_indices)
                    ax.set_yticklabels([f'{wavenumbers[i]:.0f}' for i in wn_tick_indices])
                    
                    plt.colorbar(im, ax=ax, label='强度')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if "变化率分析" in analysis_options:
                    st.write("### 📊 变化率分析")
                    
                    if len(time_axis) > 1:
                        # 计算各时间点的变化率
                        diff_X = np.diff(X, axis=0)
                        diff_time = np.diff(time_axis)
                        
                        # 变化率 = 光谱差值 / 时间差值
                        rate_X = diff_X / diff_time[:, np.newaxis]
                        
                        # 总变化率
                        total_rate = np.sum(np.abs(rate_X), axis=1)
                        
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # 总变化率随时间变化
                        ax1.plot(time_axis[1:], total_rate, 'o-', alpha=0.7, color='red')
                        ax1.set_title('总变化率随时间变化')
                        ax1.set_xlabel(time_label)
                        ax1.set_ylabel('总变化率')
                        ax1.grid(True, alpha=0.3)
                        
                        # 变化率热图
                        im = ax2.imshow(rate_X.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
                        ax2.set_title('变化率热图')
                        ax2.set_xlabel('时间间隔')
                        ax2.set_ylabel('波数 (cm⁻¹)')
                        
                        # 设置坐标轴
                        n_ticks = min(10, len(time_axis)-1)
                        tick_indices = np.linspace(0, len(time_axis)-2, n_ticks, dtype=int)
                        ax2.set_xticks(tick_indices)
                        ax2.set_xticklabels([f'{time_axis[i+1]:.1f}' for i in tick_indices])
                        
                        n_wn_ticks = min(10, len(wavenumbers))
                        wn_tick_indices = np.linspace(0, len(wavenumbers)-1, n_wn_ticks, dtype=int)
                        ax2.set_yticks(wn_tick_indices)
                        ax2.set_yticklabels([f'{wavenumbers[i]:.0f}' for i in wn_tick_indices])
                        
                        plt.colorbar(im, ax=ax2, label='变化率')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        results['change_rates'] = {
                            'rate_matrix': rate_X,
                            'total_rates': total_rate
                        }
                
                if "相关性趋势" in analysis_options and hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    st.write("### 🔗 相关性趋势分析")
                    
                    y = st.session_state.y
                    
                    if y.ndim == 1:
                        # 计算每个时间点与标签的相关性
                        correlations = []
                        for i in range(X.shape[0]):
                            corr = np.corrcoef(X[i], wavenumbers)[0, 1] if len(wavenumbers) == len(X[i]) else 0
                            correlations.append(corr)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(time_axis, correlations, 'o-', alpha=0.7)
                        ax.set_title('光谱与波数相关性随时间变化')
                        ax.set_xlabel(time_label)
                        ax.set_ylabel('相关系数')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # 保存结果
                st.session_state.trend_analysis_results = {
                    'time_axis': time_axis,
                    'time_label': time_label,
                    'analysis_results': results
                }
                
                st.success("✅ 趋势分析完成！")
                
            except Exception as e:
                st.error(f"趋势分析出错: {e}")
                st.error(traceback.format_exc())


def show_clustering_analysis(X, wavenumbers):
    """聚类分析"""
    st.write("### 🎯 聚类分析")
    
    st.markdown("""
    对光谱样本进行聚类分析，发现数据中的潜在分组结构：
    - 样本相似性分析
    - 异常样本识别
    - 数据结构探索
    """)
    
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    
    # 聚类方法选择
    clustering_method = st.selectbox(
        "聚类方法",
        ["K-Means", "层次聚类", "DBSCAN"],
        help="不同聚类方法适用于不同的数据结构",
        key="clustering_method_selectbox"  # 添加唯一key
    )
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        standardize = st.checkbox("数据标准化", value=True, key="clustering_standardize_checkbox")  # 添加唯一key
        
        if clustering_method in ["K-Means", "层次聚类"]:
            n_clusters = st.slider("聚类数量", 2, min(10, X.shape[0]//2), 3, key="clustering_n_clusters_slider")  # 添加唯一key
    
    with col2:
        if clustering_method == "DBSCAN":
            eps = st.slider("邻域半径 (eps)", 0.1, 2.0, 0.5, 0.1, key="clustering_eps_slider")  # 添加唯一key
            min_samples = st.slider("最小样本数", 2, 10, 3, key="clustering_min_samples_slider")  # 添加唯一key
        
        # 降维可视化
        use_pca_viz = st.checkbox("使用PCA降维可视化", value=True, key="clustering_pca_viz_checkbox")  # 添加唯一key
    
    if st.button("执行聚类分析", key="clustering_execute_button"):  # 添加唯一key
        with st.spinner("正在进行聚类分析..."):
            try:
                # 数据预处理
                if standardize:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                # 执行聚类
                if clustering_method == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    
                elif clustering_method == "层次聚类":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X_scaled)
                    
                elif clustering_method == "DBSCAN":
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    st.info(f"发现 {n_clusters} 个聚类，{n_noise} 个噪声点")
                
                # 聚类结果分析
                unique_labels = set(labels)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                
                st.success(f"✅ 聚类完成！发现 {n_clusters_found} 个聚类")
                
                # 可视化结果
                if use_pca_viz and X.shape[1] > 2:
                    # PCA降维可视化
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # PCA聚类结果
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # 噪声点用黑色
                            col = 'black'
                        
                        class_member_mask = (labels == k)
                        xy = X_pca[class_member_mask]
                        
                        label_name = f'聚类 {k}' if k != -1 else '噪声'
                        ax1.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.7, s=50, label=label_name)
                    
                    ax1.set_title(f'{clustering_method} 聚类结果 (PCA可视化)')
                    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                else:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # 聚类中心光谱（如果适用）
                if clustering_method != "DBSCAN":
                    cluster_centers = []
                    for k in unique_labels:
                        if k != -1:
                            mask = (labels == k)
                            center = np.mean(X[mask], axis=0)
                            cluster_centers.append(center)
                            ax2.plot(wavenumbers, center, label=f'聚类 {k} 中心', alpha=0.8)
                    
                    ax2.set_title('各聚类中心光谱')
                    ax2.set_xlabel('波数 (cm⁻¹)')
                    ax2.set_ylabel('强度')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                else:
                    # DBSCAN显示各聚类的代表光谱
                    for k in unique_labels:
                        if k != -1:
                            mask = (labels == k)
                            if np.sum(mask) > 0:
                                center = np.mean(X[mask], axis=0)
                                ax2.plot(wavenumbers, center, label=f'聚类 {k}', alpha=0.8)
                    
                    ax2.set_title('各聚类代表光谱')
                    ax2.set_xlabel('波数 (cm⁻¹)')
                    ax2.set_ylabel('强度')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                # 聚类大小分布
                cluster_sizes = []
                cluster_names = []
                for k in unique_labels:
                    size = np.sum(labels == k)
                    cluster_sizes.append(size)
                    cluster_names.append(f'聚类 {k}' if k != -1 else '噪声')
                
                ax3.bar(cluster_names, cluster_sizes, alpha=0.7)
                ax3.set_title('聚类大小分布')
                ax3.set_xlabel('聚类')
                ax3.set_ylabel('样本数量')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # 聚类内距离分布
                if clustering_method == "K-Means":
                    distances_to_center = []
                    for i, label in enumerate(labels):
                        if label != -1:
                            center = model.cluster_centers_[label]
                            dist = np.linalg.norm(X_scaled[i] - center)
                            distances_to_center.append(dist)
                    
                    ax4.hist(distances_to_center, bins=20, alpha=0.7)
                    ax4.set_title('样本到聚类中心距离分布')
                    ax4.set_xlabel('距离')
                    ax4.set_ylabel('频数')
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 聚类结果表格
                st.write("### 📋 聚类结果摘要")
                
                cluster_summary = []
                for k in unique_labels:
                    mask = (labels == k)
                    size = np.sum(mask)
                    
                    if size > 0:
                        cluster_data = X[mask]
                        
                        cluster_summary.append({
                            '聚类': f'聚类 {k}' if k != -1 else '噪声',
                            '样本数': size,
                            '占比': f'{size/len(labels)*100:.1f}%',
                            '平均强度': f'{np.mean(cluster_data):.4f}',
                            '强度标准差': f'{np.std(cluster_data):.4f}',
                            '样本编号': ', '.join([str(i+1) for i in np.where(mask)[0][:5]]) + 
                                     ('...' if size > 5 else '')
                        })
                
                cluster_df = pd.DataFrame(cluster_summary)
                st.dataframe(cluster_df, use_container_width=True)
                
                # 保存结果
                st.session_state.clustering_results = {
                    'method': clustering_method,
                    'model': model,
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'cluster_centers': cluster_centers if clustering_method != "DBSCAN" else None
                }
                
                # 如果有标签，分析聚类与标签的关系
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    show_clustering_label_relationship(labels, unique_labels)
                
            except Exception as e:
                st.error(f"聚类分析出错: {e}")
                st.error(traceback.format_exc())


def show_clustering_label_relationship(labels, unique_labels):
    """显示聚类与标签的关系"""
    st.write("### 🎯 聚类与标签关系")
    
    y = st.session_state.y
    
    if y.ndim == 1:
        # 单标签情况
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 各聚类的标签分布箱线图
        cluster_data = []
        cluster_names = []
        
        for k in unique_labels:
            if k != -1:
                mask = (labels == k)
                if np.sum(mask) > 0:
                    cluster_data.append(y[mask])
                    cluster_names.append(f'聚类 {k}')
        
        if cluster_data:
            ax1.boxplot(cluster_data, labels=cluster_names)
            ax1.set_title('各聚类标签值分布')
            ax1.set_xlabel('聚类')
            ax1.set_ylabel('标签值')
            ax1.grid(True, alpha=0.3)
        
        # 聚类标签均值对比
        cluster_means = []
        cluster_stds = []
        valid_clusters = []
        
        for k in unique_labels:
            if k != -1:
                mask = (labels == k)
                if np.sum(mask) > 0:
                    cluster_means.append(np.mean(y[mask]))
                    cluster_stds.append(np.std(y[mask]))
                    valid_clusters.append(k)
        
        if cluster_means:
            ax2.bar(range(len(valid_clusters)), cluster_means, 
                   yerr=cluster_stds, alpha=0.7, capsize=5)
            ax2.set_xticks(range(len(valid_clusters)))
            ax2.set_xticklabels([f'聚类 {k}' for k in valid_clusters])
            ax2.set_title('各聚类标签均值 ± 标准差')
            ax2.set_xlabel('聚类')
            ax2.set_ylabel('标签均值')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示统计表格
        if cluster_means:
            cluster_stats = pd.DataFrame({
                '聚类': [f'聚类 {k}' for k in valid_clusters],
                '样本数': [np.sum(labels == k) for k in valid_clusters],
                '标签均值': [f'{mean:.4f}' for mean in cluster_means],
                '标签标准差': [f'{std:.4f}' for std in cluster_stds]
            })
            st.dataframe(cluster_stats, use_container_width=True)
    
    else:
        # 多标签情况
        target_names = st.session_state.selected_cols
        selected_target = st.selectbox(
            "选择目标变量查看与聚类的关系", 
            range(len(target_names)), 
            format_func=lambda x: target_names[x],
            key="clustering_target_select"
        )
        
        y_selected = y[:, selected_target]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 各聚类的标签分布
        cluster_data = []
        cluster_names = []
        
        for k in unique_labels:
            if k != -1:
                mask = (labels == k)
                if np.sum(mask) > 0:
                    cluster_data.append(y_selected[mask])
                    cluster_names.append(f'聚类 {k}')
        
        if cluster_data:
            ax1.boxplot(cluster_data, labels=cluster_names)
            ax1.set_title(f'各聚类 {target_names[selected_target]} 分布')
            ax1.set_xlabel('聚类')
            ax1.set_ylabel(target_names[selected_target])
            ax1.grid(True, alpha=0.3)
            
            # 聚类均值对比
            cluster_means = [np.mean(data) for data in cluster_data]
            cluster_stds = [np.std(data) for data in cluster_data]
            
            ax2.bar(range(len(cluster_names)), cluster_means, 
                   yerr=cluster_stds, alpha=0.7, capsize=5)
            ax2.set_xticks(range(len(cluster_names)))
            ax2.set_xticklabels(cluster_names)
            ax2.set_title(f'各聚类 {target_names[selected_target]} 均值')
            ax2.set_xlabel('聚类')
            ax2.set_ylabel(f'{target_names[selected_target]} 均值')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


def show_anomaly_detection(X, wavenumbers):
    """异常检测"""
    func_key = "anomaly_detection"
    st.write("### 🚨 异常检测")
    
    st.markdown("""
    检测光谱数据中的异常样本：
    - 识别测量异常
    - 发现样本污染
    - 质量控制分析
    """)
    
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    
    # 异常检测方法选择
    detection_method = st.selectbox(
        "异常检测方法",
        ["Isolation Forest", "One-Class SVM", "统计方法"],
        help="不同方法适用于不同类型的异常"
    )
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider(
            "异常比例估计", 
            0.01, 0.5, 0.1, 0.01,
            help="预期异常样本的比例"
        )
        
        standardize = st.checkbox("数据标准化", value=True, key=f"{func_key}_standardize")
    
    with col2:
        if detection_method == "One-Class SVM":
            nu = st.slider("Nu参数", 0.01, 0.5, 0.1, 0.01)
            kernel = st.selectbox("核函数", ["rbf", "linear", "poly"])
        
        elif detection_method == "统计方法":
            method = st.selectbox("统计方法", ["Z-score", "IQR", "Mahalanobis"])
            threshold = st.slider("阈值", 1.0, 5.0, 2.0, 0.1)
    
    if st.button("执行异常检测"):
        with st.spinner("正在进行异常检测..."):
            try:
                # 数据预处理
                if standardize:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                # 执行异常检测
                if detection_method == "Isolation Forest":
                    model = IsolationForest(contamination=contamination, random_state=42)
                    anomaly_labels = model.fit_predict(X_scaled)
                    anomaly_scores = model.decision_function(X_scaled)
                    
                elif detection_method == "One-Class SVM":
                    model = OneClassSVM(nu=nu, kernel=kernel)
                    anomaly_labels = model.fit_predict(X_scaled)
                    anomaly_scores = model.decision_function(X_scaled)
                    
                elif detection_method == "统计方法":
                    if method == "Z-score":
                        # 基于Z-score的异常检测
                        z_scores = np.abs((X_scaled - np.mean(X_scaled, axis=0)) / np.std(X_scaled, axis=0))
                        max_z_scores = np.max(z_scores, axis=1)
                        anomaly_labels = np.where(max_z_scores > threshold, -1, 1)
                        anomaly_scores = -max_z_scores
                        
                    elif method == "IQR":
                        # 基于四分位距的异常检测
                        Q1 = np.percentile(X_scaled, 25, axis=0)
                        Q3 = np.percentile(X_scaled, 75, axis=0)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        outlier_mask = (X_scaled < lower_bound) | (X_scaled > upper_bound)
                        outlier_count = np.sum(outlier_mask, axis=1)
                        anomaly_labels = np.where(outlier_count > 0, -1, 1)
                        anomaly_scores = -outlier_count
                        
                    else:  # Mahalanobis
                        # 基于马氏距离的异常检测
                        mean = np.mean(X_scaled, axis=0)
                        cov = np.cov(X_scaled.T)
                        
                        # 计算马氏距离
                        inv_cov = np.linalg.pinv(cov)
                        mahal_dist = []
                        for i in range(X_scaled.shape[0]):
                            diff = X_scaled[i] - mean
                            mahal_dist.append(np.sqrt(diff.T @ inv_cov @ diff))
                        
                        mahal_dist = np.array(mahal_dist)
                        anomaly_labels = np.where(mahal_dist > threshold, -1, 1)
                        anomaly_scores = -mahal_dist
                
                # 分析结果
                n_anomalies = np.sum(anomaly_labels == -1)
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
                st.success(f"✅ 异常检测完成！发现 {n_anomalies} 个异常样本")
                
                if n_anomalies > 0:
                    st.warning(f"异常样本编号: {', '.join([str(i+1) for i in anomaly_indices])}")
                
                # 可视化结果
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. 异常分数分布
                ax1.hist(anomaly_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
                if n_anomalies > 0:
                    ax1.axvline(np.max(anomaly_scores[anomaly_labels == -1]), 
                               color='red', linestyle='--', label='异常阈值')
                ax1.set_title('异常分数分布')
                ax1.set_xlabel('异常分数')
                ax1.set_ylabel('频数')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. 样本异常分数
                colors = ['red' if label == -1 else 'blue' for label in anomaly_labels]
                ax2.scatter(range(len(anomaly_scores)), anomaly_scores, c=colors, alpha=0.7)
                ax2.set_title('各样本异常分数')
                ax2.set_xlabel('样本索引')
                ax2.set_ylabel('异常分数')
                ax2.grid(True, alpha=0.3)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='blue', label='正常'),
                                 Patch(facecolor='red', label='异常')]
                ax2.legend(handles=legend_elements)
                
                # 3. 正常vs异常光谱对比
                normal_indices = np.where(anomaly_labels == 1)[0]
                
                if len(normal_indices) > 0:
                    normal_mean = np.mean(X[normal_indices], axis=0)
                    normal_std = np.std(X[normal_indices], axis=0)
                    
                    ax3.plot(wavenumbers, normal_mean, 'b-', label='正常样本均值', linewidth=2)
                    ax3.fill_between(wavenumbers, 
                                    normal_mean - normal_std, 
                                    normal_mean + normal_std,
                                    alpha=0.3, color='blue', label='正常样本±1σ')
                    
                    # 显示异常样本
                    if n_anomalies > 0:
                        for i, idx in enumerate(anomaly_indices[:3]):  # 最多显示3个异常样本
                            ax3.plot(wavenumbers, X[idx], 'r--', alpha=0.7, 
                                    label=f'异常样本 {idx+1}' if i == 0 else "")
                    
                    ax3.set_title('正常 vs 异常光谱对比')
                    ax3.set_xlabel('波数 (cm⁻¹)')
                    ax3.set_ylabel('强度')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                
                # 4. PCA可视化异常检测结果
                if X.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    colors = ['red' if label == -1 else 'blue' for label in anomaly_labels]
                    ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
                    
                    ax4.set_title('异常检测结果 (PCA可视化)')
                    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    ax4.grid(True, alpha=0.3)
                    
                    # 添加图例
                    ax4.legend(handles=legend_elements)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 异常样本详细信息
                if n_anomalies > 0:
                    st.write("### 📋 异常样本详细信息")
                    
                    anomaly_details = []
                    for idx in anomaly_indices:
                        anomaly_details.append({
                            '样本编号': idx + 1,
                            '异常分数': f'{anomaly_scores[idx]:.4f}',
                            '平均强度': f'{np.mean(X[idx]):.4f}',
                            '强度标准差': f'{np.std(X[idx]):.4f}',
                            '最大强度': f'{np.max(X[idx]):.4f}',
                            '最小强度': f'{np.min(X[idx]):.4f}'
                        })
                    
                    anomaly_df = pd.DataFrame(anomaly_details)
                    st.dataframe(anomaly_df, use_container_width=True)
                
                # 保存结果
                st.session_state.anomaly_detection_results = {
                    'method': detection_method,
                    'anomaly_labels': anomaly_labels,
                    'anomaly_scores': anomaly_scores,
                    'anomaly_indices': anomaly_indices,
                    'n_anomalies': n_anomalies
                }
                
            except Exception as e:
                st.error(f"异常检测出错: {e}")
                st.error(traceback.format_exc())


def show_comprehensive_report(X, wavenumbers):
    """综合报告"""
    st.write("### 📄 综合分析报告")
    
    st.markdown("""
    生成光谱数据的综合分析报告，整合所有分析结果。
    """)
    
    if st.button("生成综合报告"):
        with st.spinner("正在生成综合报告..."):
            try:
                # 报告标题
                st.markdown("## 🔬 光谱数据分析综合报告")
                st.markdown("---")
                
                # 1. 数据概览
                st.markdown("### 📊 数据概览")
                
                data_overview = {
                    "数据项": ["样本数量", "特征数量", "波数范围", "数据类型"],
                    "数值": [
                        f"{X.shape[0]}",
                        f"{X.shape[1]}",
                        f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹",
                        "有标签数据" if hasattr(st.session_state, 'y') and st.session_state.y is not None else "无标签数据"
                    ]
                }
                
                overview_df = pd.DataFrame(data_overview)
                st.dataframe(overview_df, use_container_width=True)
                
                # 2. 数据质量评估
                st.markdown("### 🔍 数据质量评估")
                
                # 基本统计
                missing_ratio = np.sum(np.isnan(X)) / X.size * 100
                zero_ratio = np.sum(X == 0) / X.size * 100
                
                quality_metrics = {
                    "质量指标": ["缺失值比例", "零值比例", "数据范围", "平均信噪比估计"],
                    "数值": [
                        f"{missing_ratio:.2f}%",
                        f"{zero_ratio:.2f}%",
                        f"{X.min():.4f} - {X.max():.4f}",
                        f"{np.mean(X) / np.std(X):.2f}"
                    ]
                }
                
                quality_df = pd.DataFrame(quality_metrics)
                st.dataframe(quality_df, use_container_width=True)
                
                # 3. 分析结果汇总
                st.markdown("### 📈 分析结果汇总")
                
                analysis_summary = []
                
                # PCA结果
                if hasattr(st.session_state, 'pca_results'):
                    pca_results = st.session_state.pca_results
                    total_var = np.sum(pca_results['explained_variance_ratio'])
                    analysis_summary.append({
                        "分析方法": "主成分分析 (PCA)",
                        "主要发现": f"前{len(pca_results['explained_variance_ratio'])}个主成分解释{total_var:.1%}的方差",
                        "建议": "数据降维效果良好" if total_var > 0.8 else "考虑增加主成分数量"
                    })
                
                # 聚类结果
                if hasattr(st.session_state, 'clustering_results'):
                    clustering_results = st.session_state.clustering_results
                    analysis_summary.append({
                        "分析方法": f"聚类分析 ({clustering_results['method']})",
                        "主要发现": f"发现{clustering_results['n_clusters']}个聚类",
                        "建议": "样本存在明显分组结构" if clustering_results['n_clusters'] > 1 else "样本相对均匀分布"
                    })
                
                # 异常检测结果
                if hasattr(st.session_state, 'anomaly_detection_results'):
                    anomaly_results = st.session_state.anomaly_detection_results
                    anomaly_ratio = anomaly_results['n_anomalies'] / X.shape[0] * 100
                    analysis_summary.append({
                        "分析方法": f"异常检测 ({anomaly_results['method']})",
                        "主要发现": f"检测到{anomaly_results['n_anomalies']}个异常样本({anomaly_ratio:.1f}%)",
                        "建议": "数据质量较好" if anomaly_ratio < 5 else "建议检查异常样本"
                    })
                
                # 成分分解结果
                if hasattr(st.session_state, 'decomposition_results'):
                    decomp_results = st.session_state.decomposition_results
                    analysis_summary.append({
                        "分析方法": f"成分分解 ({decomp_results['method']})",
                        "主要发现": f"分解出{decomp_results['n_components']}个化学成分",
                        "建议": "可用于成分识别和定量分析"
                    })
                
                if analysis_summary:
                    summary_df = pd.DataFrame(analysis_summary)
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    st.info("请先执行相关分析以生成完整报告")
                
                # 4. 标签相关分析（如果有标签）
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    st.markdown("### 🎯 标签相关分析")
                    
                    y = st.session_state.y
                    
                    if y.ndim == 1:
                        label_stats = {
                            "标签统计": ["标签数量", "标签范围", "标签均值", "标签标准差"],
                            "数值": [
                                f"{len(y)}",
                                f"{y.min():.4f} - {y.max():.4f}",
                                f"{np.mean(y):.4f}",
                                f"{np.std(y):.4f}"
                            ]
                        }
                    else:
                        target_names = st.session_state.selected_cols
                        label_stats = {
                            "标签统计": ["目标变量数量", "目标变量名称", "样本数量"],
                            "数值": [
                                f"{y.shape[1]}",
                                ", ".join(target_names),
                                f"{y.shape[0]}"
                            ]
                        }
                    
                    label_df = pd.DataFrame(label_stats)
                    st.dataframe(label_df, use_container_width=True)
                
                # 5. 建议和结论
                st.markdown("### 💡 分析建议和结论")
                
                recommendations = []
                
                # 基于数据质量的建议
                if missing_ratio > 5:
                    recommendations.append("• 数据存在较多缺失值，建议进行数据清洗")
                
                if zero_ratio > 10:
                    recommendations.append("• 数据中零值较多，可能需要检查数据采集过程")
                
                # 基于分析结果的建议
                if hasattr(st.session_state, 'pca_results'):
                    pca_results = st.session_state.pca_results
                    if np.sum(pca_results['explained_variance_ratio'][:2]) > 0.8:
                        recommendations.append("• 前两个主成分解释了大部分方差，适合进行降维可视化")
                
                if hasattr(st.session_state, 'clustering_results'):
                    clustering_results = st.session_state.clustering_results
                    if clustering_results['n_clusters'] > 3:
                        recommendations.append("• 样本存在多个分组，建议进一步分析各组特征")
                
                if hasattr(st.session_state, 'anomaly_detection_results'):
                    anomaly_results = st.session_state.anomaly_detection_results
                    if anomaly_results['n_anomalies'] > 0:
                        recommendations.append(f"• 发现{anomaly_results['n_anomalies']}个异常样本，建议进行人工审查")
                
                # 通用建议
                recommendations.extend([
                    "• 建议结合领域知识解释分析结果",
                    "• 可考虑进行交叉验证以评估模型稳定性",
                    "• 如需预测建模，建议进行特征选择和模型优化"
                ])
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # 6. 生成报告时间
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**报告生成时间**: {current_time}")
                
                st.success("✅ 综合报告生成完成！")
                
            except Exception as e:
                st.error(f"生成报告时出错: {e}")
                st.error(traceback.format_exc())


def show_data_split_page():
    """数据集划分页面"""
    st.markdown("<h1 class='section-header'>数据集划分</h1>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not check_data_prerequisites(need_labels=True, need_preprocessing=False):
        return
    
    st.markdown("""
    <div class="info-box">
    将数据划分为训练集和测试集，或设置交叉验证方案。
    </div>
    """, unsafe_allow_html=True)
    
    # 获取当前数据
    X, wavenumbers, data_info = get_current_data()
    show_status_message(data_info, "info")
    
    y = st.session_state.y
    
    st.info(f"📊 数据形状: {X.shape}, 标签形状: {y.shape}")
    
    # 划分方法选择
    st.subheader("📊 数据划分方案")
    
    split_method = st.selectbox(
        "选择数据划分方法",
        ["随机划分", "KFold交叉验证", "留一法(LOOCV)"],
        help="不同划分方法适用于不同的数据规模和验证需求"
    )
    
    # 参数设置
    if split_method == "随机划分":
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("随机种子", value=42, min_value=0)
        
        with col2:
            stratify = st.checkbox("分层抽样", value=False, 
                                 help="保持训练集和测试集中标签分布一致")
    
    elif split_method == "KFold交叉验证":
        col1, col2 = st.columns(2)
        
        with col1:
            n_splits = st.slider("交叉验证折数", 3, 10, 5)
            random_state = st.number_input("随机种子", value=42, min_value=0)
        
        with col2:
            shuffle = st.checkbox("打乱数据", value=True)
    
    else:  # LOOCV
        st.info("留一法交叉验证将使用 n-1 个样本训练，1个样本测试，重复 n 次")
        if X.shape[0] > 100:
            st.warning("⚠️ 样本数量较多，留一法可能计算时间较长")
    
    # 执行数据划分
    if st.button("🚀 执行数据划分"):
        try:
            from sklearn.model_selection import train_test_split
            
            if split_method == "随机划分":
                # 处理分层抽样
                stratify_y = None
                if stratify:
                    if y.ndim == 1:
                        # 单输出：检查是否适合分层
                        unique_values = np.unique(y)
                        if len(unique_values) < X.shape[0] * 0.5:  # 离散值较少
                            stratify_y = y
                        else:
                            # 连续值：转换为分箱
                            n_bins = min(5, len(unique_values))
                            stratify_y = pd.cut(y, bins=n_bins, labels=False)
                    else:
                        # 多输出：使用第一个目标变量
                        unique_values = np.unique(y[:, 0])
                        if len(unique_values) < X.shape[0] * 0.5:
                            stratify_y = y[:, 0]
                        else:
                            n_bins = min(5, len(unique_values))
                            stratify_y = pd.cut(y[:, 0], bins=n_bins, labels=False)
                
                # 执行划分
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=stratify_y
                )
                
                # 保存结果
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.split_method = split_method
                
                st.success("✅ 数据划分完成！")
                st.info(f"📊 训练集: {X_train.shape}, 测试集: {X_test.shape}")
                
                # 显示划分结果
                show_split_visualization(X_train, X_test, y_train, y_test, wavenumbers)
                
            elif split_method == "KFold交叉验证":
                # 保存交叉验证参数
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.split_method = split_method
                st.session_state.cv_splits = n_splits
                st.session_state.cv_shuffle = shuffle
                st.session_state.cv_random_state = random_state
                
                st.success("✅ 交叉验证设置完成！")
                st.info(f"📊 将使用 {n_splits} 折交叉验证")
                
            else:  # LOOCV
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.split_method = split_method
                
                st.success("✅ 留一法交叉验证设置完成！")
                st.info(f"📊 将进行 {X.shape[0]} 次留一法验证")
            
        except Exception as e:
            st.error(f"数据划分出错: {e}")
            st.error(traceback.format_exc())


def show_split_visualization(X_train, X_test, y_train, y_test, wavenumbers):
    """显示数据划分可视化"""
    st.subheader("📈 数据划分可视化")
    
    # 标签分布对比
    if y_train.ndim == 1:
        # 单输出情况
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练集标签分布
        ax1.hist(y_train, bins=20, alpha=0.7, color='blue', label='训练集')
        ax1.set_title('训练集标签分布')
        ax1.set_xlabel('标签值')
        ax1.set_ylabel('频数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 测试集标签分布
        ax2.hist(y_test, bins=20, alpha=0.7, color='orange', label='测试集')
        ax2.set_title('测试集标签分布')
        ax2.set_xlabel('标签值')
        ax2.set_ylabel('频数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 训练集vs测试集标签对比
        ax3.hist(y_train, bins=20, alpha=0.5, color='blue', label='训练集')
        ax3.hist(y_test, bins=20, alpha=0.5, color='orange', label='测试集')
        ax3.set_title('训练集 vs 测试集标签分布对比')
        ax3.set_xlabel('标签值')
        ax3.set_ylabel('频数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 训练集和测试集光谱对比
        n_samples = min(5, X_train.shape[0], X_test.shape[0])
        
        for i in range(n_samples):
            ax4.plot(wavenumbers, X_train[i], alpha=0.5, color='blue')
            ax4.plot(wavenumbers, X_test[i], alpha=0.5, color='orange')
        
        ax4.set_title('训练集 vs 测试集光谱对比')
        ax4.set_xlabel('波数 (cm⁻¹)')
        ax4.set_ylabel('强度')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', alpha=0.7, label='训练集'),
                          Line2D([0], [0], color='orange', alpha=0.7, label='测试集')]
        ax4.legend(handles=legend_elements)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        # 多输出情况
        target_names = st.session_state.selected_cols
        selected_target = st.selectbox(
            "选择目标变量查看分布", 
            range(len(target_names)), 
            format_func=lambda x: target_names[x]
        )
        
        y_train_selected = y_train[:, selected_target]
        y_test_selected = y_test[:, selected_target]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练集标签分布
        ax1.hist(y_train_selected, bins=20, alpha=0.7, color='blue')
        ax1.set_title(f'训练集 {target_names[selected_target]} 分布')
        ax1.set_xlabel('标签值')
        ax1.set_ylabel('频数')
        ax1.grid(True, alpha=0.3)
        
        # 测试集标签分布
        ax2.hist(y_test_selected, bins=20, alpha=0.7, color='orange')
        ax2.set_title(f'测试集 {target_names[selected_target]} 分布')
        ax2.set_xlabel('标签值')
        ax2.set_ylabel('频数')
        ax2.grid(True, alpha=0.3)
        
        # 对比分布
        ax3.hist(y_train_selected, bins=20, alpha=0.5, color='blue', label='训练集')
        ax3.hist(y_test_selected, bins=20, alpha=0.5, color='orange', label='测试集')
        ax3.set_title(f'{target_names[selected_target]} 分布对比')
        ax3.set_xlabel('标签值')
        ax3.set_ylabel('频数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 光谱对比
        n_samples = min(5, X_train.shape[0], X_test.shape[0])
        
        for i in range(n_samples):
            ax4.plot(wavenumbers, X_train[i], alpha=0.5, color='blue')
            ax4.plot(wavenumbers, X_test[i], alpha=0.5, color='orange')
        
        ax4.set_title('训练集 vs 测试集光谱对比')
        ax4.set_xlabel('波数 (cm⁻¹)')
        ax4.set_ylabel('强度')
        ax4.legend(handles=legend_elements)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 统计信息表格
    st.write("### 📋 划分统计信息")
    
    if y_train.ndim == 1:
        stats_data = {
            '数据集': ['训练集', '测试集'],
            '样本数量': [len(X_train), len(X_test)],
            '标签均值': [f'{np.mean(y_train):.4f}', f'{np.mean(y_test):.4f}'],
            '标签标准差': [f'{np.std(y_train):.4f}', f'{np.std(y_test):.4f}'],
            '标签范围': [f'{y_train.min():.4f} - {y_train.max():.4f}', 
                        f'{y_test.min():.4f} - {y_test.max():.4f}']
        }
    else:
        stats_data = {
            '数据集': ['训练集', '测试集'],
            '样本数量': [len(X_train), len(X_test)],
            '目标变量数': [y_train.shape[1], y_test.shape[1]],
            '特征数量': [X_train.shape[1], X_test.shape[1]]
        }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)


# 模型相关函数
MODEL_NAMES = {
    'linear': '线性回归',
    'ridge': '岭回归',
    'lasso': 'Lasso回归',
    'svr': '支持向量回归',
    'rf': '随机森林',
    'gbr': '梯度提升回归',
    'mlp': '多层感知机',
    'pls': '偏最小二乘回归',
    'xgb': 'XGBoost'
}


def setup_model_parameters_ui(model_name, index):
    """设置模型参数UI"""
    st.write(f"**{MODEL_NAMES[model_name]} 参数设置**")
    
    if model_name == 'linear':
        return setup_linear_params(index)
    elif model_name == 'ridge':
        return setup_ridge_params(index)
    elif model_name == 'lasso':
        return setup_lasso_params(index)
    elif model_name == 'svr':
        return setup_svr_params(index)
    elif model_name == 'rf':
        return setup_rf_params(index)
    elif model_name == 'gbr':
        return setup_gbr_params(index)
    elif model_name == 'mlp':
        return setup_mlp_params(index)
    elif model_name == 'pls':
        return setup_pls_params(index)
    elif model_name == 'xgb':
        return setup_xgb_params(index)


def setup_linear_params(index):
    """线性回归参数设置"""
    st.write("线性回归无需调参")
    return {}


def setup_ridge_params(index):
    """岭回归参数设置"""
    alpha = st.selectbox("正则化强度 (alpha)", [0.1, 1.0, 10.0, 100.0], 
                        index=1, key=f"ridge_alpha_{index}")
    return {'alpha': alpha}


def setup_lasso_params(index):
    """Lasso回归参数设置"""
    alpha = st.selectbox("正则化强度 (alpha)", [0.01, 0.1, 1.0, 10.0], 
                        index=1, key=f"lasso_alpha_{index}")
    return {'alpha': alpha}


def setup_svr_params(index):
    """支持向量回归参数设置"""
    col1, col2 = st.columns(2)
    with col1:
        kernel = st.selectbox("核函数", ['rbf', 'linear', 'poly'], 
                             index=0, key=f"svr_kernel_{index}")
        C = st.selectbox("正则化参数 (C)", [0.1, 1.0, 10.0, 100.0], 
                        index=1, key=f"svr_C_{index}")
    with col2:
        gamma = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1], 
                           index=0, key=f"svr_gamma_{index}")
        epsilon = st.selectbox("Epsilon", [0.01, 0.1, 0.2], 
                             index=1, key=f"svr_epsilon_{index}")
    
    return {
        'kernel': kernel,
        'C': C,
        'gamma': gamma,
        'epsilon': epsilon
    }


def setup_rf_params(index):
    """随机森林参数设置"""
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("树的数量", 50, 500, 100, key=f"rf_trees_{index}")
        max_depth = st.selectbox("最大深度", [None, 5, 10, 15, 20], 
                               index=0, key=f"rf_depth_{index}")
    with col2:
        min_samples_split = st.slider("分裂最小样本数", 2, 10, 2, key=f"rf_split_{index}")
        min_samples_leaf = st.slider("叶节点最小样本数", 1, 5, 1, key=f"rf_leaf_{index}")
    
    random_state = st.number_input("随机种子", value=42, key=f"rf_seed_{index}")
    
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state
    }


def setup_gbr_params(index):
    """梯度提升参数设置"""
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("提升阶段数", 50, 500, 100, key=f"gbr_stages_{index}")
        learning_rate = st.selectbox("学习率", [0.01, 0.05, 0.1, 0.2], 
                                   index=2, key=f"gbr_lr_{index}")
    with col2:
        max_depth = st.slider("最大深度", 2, 10, 3, key=f"gbr_depth_{index}")
        subsample = st.slider("子采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"gbr_subsample_{index}")
    
    random_state = st.number_input("随机种子", value=42, key=f"gbr_seed_{index}")
    
    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'random_state': random_state
    }


def setup_pls_params(index):
    """PLS参数设置"""
    n_components = st.slider("主成分数量", 1, min(20, st.session_state.X_train.shape[1]), 
                           5, key=f"pls_components_{index}")
    scale = st.checkbox("标准化", value=True, key=f"pls_scale_{index}")
    
    return {
        'n_components': n_components,
        'scale': scale
    }


def setup_mlp_params(index):
    """MLP参数设置"""
    col1, col2 = st.columns(2)
    with col1:
        layer_option = st.selectbox("隐藏层结构", ["一层", "两层", "三层"], 
                                  index=1, key=f"mlp_layers_{index}")
        
        if layer_option == "一层":
            layer1_size = st.slider("隐藏层神经元数", 10, 200, 50, key=f"mlp_l1_{index}")
            hidden_layer_sizes = (layer1_size,)
        elif layer_option == "两层":
            layer1_size = st.slider("第一层神经元数", 10, 200, 100, key=f"mlp_l1_{index}")
            layer2_size = st.slider("第二层神经元数", 10, 100, 50, key=f"mlp_l2_{index}")
            hidden_layer_sizes = (layer1_size, layer2_size)
        else:  # 三层
            layer1_size = st.slider("第一层神经元数", 10, 200, 100, key=f"mlp_l1_{index}")
            layer2_size = st.slider("第二层神经元数", 10, 100, 50, key=f"mlp_l2_{index}")
            layer3_size = st.slider("第三层神经元数", 10, 50, 25, key=f"mlp_l3_{index}")
            hidden_layer_sizes = (layer1_size, layer2_size, layer3_size)
        
        activation = st.selectbox("激活函数", ['relu', 'tanh', 'logistic'], 
                                index=0, key=f"mlp_activation_{index}")
    
    with col2:
        solver = st.selectbox("优化算法", ['adam', 'lbfgs', 'sgd'], 
                            index=0, key=f"mlp_solver_{index}")
        learning_rate_init = st.selectbox("初始学习率", [0.0001, 0.001, 0.01], 
                                        index=1, key=f"mlp_lr_{index}")
        max_iter = st.slider("最大迭代次数", 100, 1000, 500, key=f"mlp_iter_{index}")
        alpha = st.selectbox("L2正则化参数", [0.0001, 0.001, 0.01], 
                           index=0, key=f"mlp_alpha_{index}")
    
    random_state = st.number_input("随机种子", value=42, key=f"mlp_seed_{index}")
    
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'learning_rate_init': learning_rate_init,
        'max_iter': max_iter,
        'alpha': alpha,
        'random_state': random_state
    }


def setup_xgb_params(index):
    """XGBoost参数设置"""
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("提升轮数", 50, 500, 100, key=f"xgb_trees_{index}")
        learning_rate = st.selectbox("学习率", [0.01, 0.05, 0.1, 0.2], 
                                   index=2, key=f"xgb_lr_{index}")
        max_depth = st.slider("最大深度", 2, 10, 6, key=f"xgb_depth_{index}")
    with col2:
        subsample = st.slider("子采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"xgb_subsample_{index}")
        colsample_bytree = st.slider("特征采样比例", 0.5, 1.0, 1.0, step=0.1, key=f"xgb_colsample_{index}")
        reg_alpha = st.selectbox("L1正则化", [0, 0.01, 0.1], index=0, key=f"xgb_alpha_{index}")
    
    random_state = st.number_input("随机种子", value=42, key=f"xgb_seed_{index}")
    
    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'random_state': random_state
    }


def create_model_instance(model_name, params, is_multioutput):
    """创建模型实例"""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    
    use_scaler = False
    
    if model_name == 'linear':
        model = LinearRegression()
        use_scaler = True
        
    elif model_name == 'ridge':
        model = Ridge(**params)
        use_scaler = True
        
    elif model_name == 'lasso':
        model = Lasso(**params)
        use_scaler = True
        
    elif model_name == 'svr':
        model = SVR(**params)
        if is_multioutput:
            model = MultiOutputRegressor(model)
        use_scaler = True
        
    elif model_name == 'rf':
        model = RandomForestRegressor(**params)
        
    elif model_name == 'gbr':
        if is_multioutput:
            model = MultiOutputRegressor(GradientBoostingRegressor(**params))
        else:
            model = GradientBoostingRegressor(**params)
            
    elif model_name == 'mlp':
        model = MLPRegressor(**params)
        if is_multioutput:
            model = MultiOutputRegressor(model)
        use_scaler = True
        
    elif model_name == 'pls':
        model = PLSRegression(**params)
        
    elif model_name == 'xgb':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(**params)
            if is_multioutput:
                model = MultiOutputRegressor(model)
        except ImportError:
            raise ImportError("XGBoost未安装，请安装后使用")
    
    return model, use_scaler


# ====================================
# 重构后的模型训练主函数
# ====================================

def show_model_training_page():
    """模型训练与评估页面 - 重构版本"""
    st.markdown("<h1 class='section-header'>模型训练与评估</h1>", unsafe_allow_html=True)
    
    # 使用新的前置条件检查函数
    if not check_data_prerequisites(need_labels=True, need_preprocessing=False):
        return
    
    if not hasattr(st.session_state, 'split_method'):
        show_status_message("请先完成数据集划分", "warning")
        return
    
    st.markdown("""
    <div class="info-box">
    选择机器学习算法进行模型训练和评估，支持多种回归算法和参数调优。
    </div>
    """, unsafe_allow_html=True)
    
    # 显示当前数据信息 - 使用新的工具函数
    X, wavenumbers, data_info = get_current_data()
    show_status_message(data_info, "info")
    
    # 模型选择
    st.subheader("🤖 模型选择与参数设置")
    
    # 检查XGBoost可用性
    available_models = MODEL_NAMES.copy()
    try:
        import xgboost as xgb
    except ImportError:
        available_models.pop('xgb', None)
    
    selected_models = st.multiselect(
        "选择要训练的模型", 
        list(available_models.keys()), 
        format_func=lambda x: available_models[x]
    )
    
    if not selected_models:
        show_status_message("请至少选择一个模型", "warning")
        return
    
    # 检查是否为多输出问题
    is_multioutput = len(st.session_state.selected_cols) > 1
    
    # 参数设置
    model_params = {}
    for i, model_name in enumerate(selected_models):
        model_params[model_name] = setup_model_parameters_ui(model_name, i)
    
    # 显示交叉验证信息
    display_cv_info()
    
    # 开始训练
    if st.button("🚀 开始训练模型", type="primary"):
        train_all_models(selected_models, model_params, is_multioutput)


def display_cv_info():
    """显示交叉验证信息"""
    if st.session_state.split_method in ["KFold交叉验证", "留一法(LOOCV)"]:
        if st.session_state.split_method == "KFold交叉验证":
            cv_folds = getattr(st.session_state, 'cv_splits', 5)
            show_status_message(f"将使用 {cv_folds} 折交叉验证", "info")
        else:
            show_status_message("将使用留一法交叉验证", "info")
    else:
        show_status_message("将使用训练集/测试集划分", "info")


def train_all_models(selected_models, model_params, is_multioutput):
    """训练所有选定的模型"""
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
        progress_text.text(f"正在训练 {MODEL_NAMES[model_name]} ({i+1}/{len(selected_models)})...")
        
        result = safe_execute(
            lambda: train_single_model(
                model_name, model_params[model_name], 
                X_train, y_train, X_test, y_test, is_multioutput
            ),
            f"训练模型 {MODEL_NAMES[model_name]} 时出错"
        )
        
        if result:
            model, train_pred, test_pred, scaler, train_time, cv_results = result
            
            # 计算评估指标
            metrics = calculate_model_metrics(
                y_train, y_test, train_pred, test_pred, is_multioutput
            )
            
            # 保存结果
            result_entry = {
                'Model': MODEL_NAMES[model_name],
                'Training Time (s)': train_time,
                **metrics
            }
            
            if cv_results:
                result_entry.update(cv_results)
            
            results.append(result_entry)
            trained_models[model_name] = model
            detailed_results[model_name] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'params': model_params[model_name],
                'scaler': scaler
            }
            
            # 如果是多输出，保存每个目标的详细结果
            if is_multioutput:
                detailed_results[model_name].update(
                    calculate_multioutput_details(y_train, y_test, train_pred, test_pred)
                )
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    progress_text.text("所有模型训练完成！")
    
    if results:
        display_training_results(results, trained_models, detailed_results, is_multioutput)
    else:
        show_status_message("没有成功训练任何模型，请检查数据和参数设置", "error")


def train_single_model(model_name, params, X_train, y_train, X_test, y_test, is_multioutput):
    """训练单个模型"""
    import time
    
    # 创建模型
    model, use_scaler = create_model_instance(model_name, params.copy(), is_multioutput)
    
    # 处理标准化
    scaler = None
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    if use_scaler and model_name in ['ridge', 'lasso', 'linear']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    start_time = time.time()
    
    # 判断是否使用交叉验证
    use_cv = st.session_state.split_method in ["KFold交叉验证", "留一法(LOOCV)"]
    cv_results = None
    
    if use_cv:
        train_pred, cv_results = train_with_cv(
            model, X_train_scaled, y_train, st.session_state.split_method
        )
        test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    
    train_time = time.time() - start_time
    
    return model, train_pred, test_pred, scaler, train_time, cv_results


def train_with_cv(model, X_train, y_train, cv_method):
    """使用交叉验证训练模型"""
    from sklearn.model_selection import KFold, LeaveOneOut
    from sklearn.metrics import r2_score
    
    if cv_method == "KFold交叉验证":
        cv_folds = getattr(st.session_state, 'cv_splits', 5)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:  # LOOCV
        cv = LeaveOneOut()
    
    # 执行交叉验证
    cv_predictions = np.zeros_like(y_train)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # 训练
        model.fit(X_fold_train, y_fold_train)
        
        # 预测
        fold_pred = model.predict(X_fold_val)
        cv_predictions[val_idx] = fold_pred
        
        # 计算fold得分
        if y_fold_val.ndim > 1 and y_fold_val.shape[1] > 1:
            fold_score = np.mean([r2_score(y_fold_val[:, j], fold_pred[:, j]) 
                                for j in range(y_fold_val.shape[1])])
        else:
            fold_score = r2_score(y_fold_val, fold_pred)
        cv_scores.append(fold_score)
    
    # 用全部数据重新训练
    model.fit(X_train, y_train)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    return cv_predictions, {'CV R² Mean': cv_mean, 'CV R² Std': cv_std}


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

            # 让用户选择光谱数据的起始列
            st.write("**选择光谱数据起始列：**")
            start_col_options = list(range(1, min(blind_df.shape[1], 6)))  # 最多显示前5列作为选项
            start_col_labels = [f"第{i}列 ({blind_df.columns[i-1]})" for i in start_col_options]

            selected_start_col = st.selectbox(
                "光谱波数数据从第几列开始？",
                start_col_options,
                index=2 if len(start_col_options) > 2 else 0,  # 默认选择第3列
                format_func=lambda x: f"第{x}列 ({blind_df.columns[x-1]})"
            )

            st.info(f"已选择从第{selected_start_col}列开始提取光谱数据")

            # 识别波数列
            try:
                # 使用用户选择的起始列
                potential_wavenumbers = blind_df.columns[selected_start_col-1:]  # 转换为0基索引
                numeric_columns = []
                
                for col in potential_wavenumbers:
                    try:
                        float(col)
                        numeric_columns.append(col)
                    except ValueError:
                        continue
                
                if len(numeric_columns) < 10:
                    st.error("检测到的波数列数量不足，请检查数据格式或重新选择起始列")
                    return
                
                blind_wavenumbers = pd.Series(numeric_columns).astype(float)
                st.info(f"盲样数据波数范围: {blind_wavenumbers.min():.1f} ~ {blind_wavenumbers.max():.1f} cm⁻¹")
                st.info(f"检测到 {len(numeric_columns)} 个波数列")
                
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
                            
                           # ⭐ 1. 首先截取波数范围 - 按照预处理参数 ⭐
                            start_wn = params['start_wavenumber']
                            end_wn = params['end_wavenumber']

                            st.write(f"**预处理参数中的波数范围**: {start_wn} ~ {end_wn} cm⁻¹")

                            # 在盲样数据中找到对应的波数范围
                            start_idx = np.argmin(np.abs(blind_wavenumbers - start_wn))
                            end_idx = np.argmin(np.abs(blind_wavenumbers - end_wn))

                            # 确保索引顺序正确（处理递减波数的情况）
                            if start_idx > end_idx:
                                start_idx, end_idx = end_idx, start_idx

                            # 截取波数和光谱数据
                            blind_wavenumbers_crop = blind_wavenumbers[start_idx:end_idx+1]
                            blind_X_crop = blind_spectra[:, start_idx:end_idx+1]

                            # 获取训练时的预处理后特征数量
                            expected_feature_count = st.session_state.X_preprocessed.shape[1]
                            actual_feature_count = blind_X_crop.shape[1]

                            st.write(f"✓ 波数截取: {blind_wavenumbers_crop.min():.1f} ~ {blind_wavenumbers_crop.max():.1f} cm⁻¹")
                            st.write(f"✓ 截取后形状: {blind_X_crop.shape}")
                            st.write(f"截取的特征数量: {actual_feature_count}，期望: {expected_feature_count}")

                            # 如果特征数量不匹配，进行插值调整
                            if actual_feature_count != expected_feature_count:
                                st.warning(f"特征数量不匹配，使用插值调整: {actual_feature_count} → {expected_feature_count}")
                                
                                from scipy.interpolate import interp1d
                                
                                # 获取训练时的波数网格
                                if hasattr(st.session_state, 'wavenumbers_preprocessed'):
                                    train_wavenumbers_crop = st.session_state.wavenumbers_preprocessed
                                else:
                                    # 重新计算训练时的波数截取
                                    train_wavenumbers = st.session_state.wavenumbers
                                    train_start_idx = np.argmin(np.abs(train_wavenumbers - start_wn))
                                    train_end_idx = np.argmin(np.abs(train_wavenumbers - end_wn))
                                    if train_start_idx > train_end_idx:
                                        train_start_idx, train_end_idx = train_end_idx, train_start_idx
                                    train_wavenumbers_crop = train_wavenumbers[train_start_idx:train_end_idx+1]
                                
                                interpolated_spectra = np.zeros((blind_X_crop.shape[0], expected_feature_count))
                                
                                for i in range(blind_X_crop.shape[0]):
                                    # 创建插值函数
                                    f = interp1d(blind_wavenumbers_crop, blind_X_crop[i], 
                                                kind='linear', bounds_error=False, fill_value='extrapolate')
                                    # 插值到训练数据的波数网格
                                    interpolated_spectra[i] = f(train_wavenumbers_crop)
                                
                                blind_wavenumbers_crop = train_wavenumbers_crop
                                blind_X_crop = interpolated_spectra
                                
                                st.write(f"✓ 插值调整完成，最终形状: {blind_X_crop.shape}")

                            elif actual_feature_count > expected_feature_count:
                                st.warning(f"盲样特征数量过多({actual_feature_count} > {expected_feature_count})，进行重采样")
                                
                                # 使用插值重采样到训练数据的波数网格
                                from scipy.interpolate import interp1d
                                
                                resampled_spectra = np.zeros((blind_X_crop.shape[0], expected_feature_count))
                                
                                for i in range(blind_X_crop.shape[0]):
                                    f = interp1d(blind_wavenumbers_crop, blind_X_crop[i], 
                                                kind='linear', bounds_error=False, fill_value='extrapolate')
                                    resampled_spectra[i] = f(train_wavenumbers_crop)
                                
                                blind_wavenumbers_crop = train_wavenumbers_crop
                                blind_X_crop = resampled_spectra
                                
                                st.write(f"✓ 重采样完成，最终形状: {blind_X_crop.shape}")
                            
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
                                    spectrum = blind_X_corr[i]
                                    
                                    if normalize_method == 'area':
                                        # 修正：直接使用原始光谱计算面积，不取绝对值
                                        total_area = np.trapz(spectrum, blind_wavenumbers_crop)
                                        if abs(total_area) < 1e-12:  # 避免除零
                                            blind_X_norm[i] = spectrum
                                        else:
                                            blind_X_norm[i] = spectrum / total_area
                                            
                                    elif normalize_method == 'max':
                                        # 修正：找到最大绝对值，但保持原始符号
                                        max_abs_val = np.max(np.abs(spectrum))
                                        if max_abs_val < 1e-12:  # 避免除零
                                            blind_X_norm[i] = spectrum
                                        else:
                                            blind_X_norm[i] = spectrum / max_abs_val
                                            
                                    elif normalize_method == 'vector':
                                        # 向量归一化（L2范数）
                                        norm_val = np.linalg.norm(spectrum)
                                        if norm_val < 1e-12:  # 避免除零
                                            blind_X_norm[i] = spectrum
                                        else:
                                            blind_X_norm[i] = spectrum / norm_val
                                            
                                    elif normalize_method == 'minmax':
                                        # 最小-最大归一化到[0,1]
                                        min_val = np.min(spectrum)
                                        max_val = np.max(spectrum)
                                        if abs(max_val - min_val) < 1e-12:  # 避免除零
                                            blind_X_norm[i] = spectrum
                                        else:
                                            blind_X_norm[i] = (spectrum - min_val) / (max_val - min_val)
                                            
                                    elif normalize_method == 'std':
                                        # 标准化（零均值，单位方差）
                                        mean_val = np.mean(spectrum)
                                        std_val = np.std(spectrum)
                                        if std_val < 1e-12:  # 避免除零
                                            blind_X_norm[i] = spectrum - mean_val
                                        else:
                                            blind_X_norm[i] = (spectrum - mean_val) / std_val
                                            
                                    else:
                                        # 未知方法，直接复制
                                        blind_X_norm[i] = spectrum
                                        st.warning(f"未知归一化方法: {normalize_method}")
                                
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

                            # 2. 从原始盲样数据中提取标识列 (根据光谱起始列动态调整)
                            for i in range(selected_start_col - 1):  # 提取光谱数据之前的所有列作为标识列
                                if i < blind_df.shape[1]:
                                    result_df[blind_df.columns[i]] = blind_df.iloc[:, i]

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
                            
                            
                            # 提供下载链接
                            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                            b64 = base64.b64encode(csv.encode('utf-8')).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="盲样预测结果.csv">📥 下载预测结果</a>'
                            st.markdown(href, unsafe_allow_html=True)

                            # ⭐ 保存预测结果到 session_state ⭐
                            st.session_state.blind_prediction_results = {
                                'predictions': predictions,
                                'blind_X_final': blind_X_final,
                                'result_df': result_df,
                                'prediction_completed': True
                            }

                            # 确定显示用的波数
                            if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
                                if st.session_state.feature_selection_method != "不进行特征选择":
                                    display_wavenumbers = st.session_state.wavenumbers_final
                                else:
                                    display_wavenumbers = blind_wavenumbers_crop
                            else:
                                display_wavenumbers = blind_wavenumbers_crop

                            st.session_state.blind_prediction_results['display_wavenumbers'] = display_wavenumbers

                            st.success("预测完成！")

                        except Exception as e:
                            st.error(f"预测过程中出错: {e}")
                            st.error(traceback.format_exc())
                
                # ⭐ 光谱显示代码放在这里（与 if st.button("进行预测"): 同级）⭐
                if (hasattr(st.session_state, 'blind_prediction_results') and 
                    st.session_state.blind_prediction_results.get('prediction_completed', False)):
                    
                    # 从 session_state 获取数据
                    blind_results = st.session_state.blind_prediction_results
                    blind_X_final = blind_results['blind_X_final']
                    display_wavenumbers = blind_results['display_wavenumbers']
                    
                    # 显示预处理后的光谱
                    if st.checkbox("查看预处理后的盲样光谱"):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        n_samples = min(10, blind_X_final.shape[0])
                        
                        for i in range(n_samples):
                            ax.plot(display_wavenumbers, blind_X_final[i], alpha=0.7)
                        ax.set_title(f'预处理后的盲样光谱 (显示前{n_samples}个样本)')
                        ax.set_xlabel('波数 (cm⁻¹)')
                        ax.set_ylabel('处理后强度')
                        ax.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
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
        3. 可自定义光谱数据起始列
        4. 光谱起始列之前的列为样本标识信息
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
# 缺失函数定义补充
# ====================================


def calculate_model_metrics(y_train, y_test, train_pred, test_pred, is_multioutput):
    """
    计算模型评估指标
    
    Args:
        y_train: 训练集真实值
        y_test: 测试集真实值
        train_pred: 训练集预测值
        test_pred: 测试集预测值
        is_multioutput: 是否为多输出问题
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    
    if is_multioutput:
        # 多输出情况：计算平均指标
        train_r2_scores = []
        test_r2_scores = []
        train_rmse_scores = []
        test_rmse_scores = []
        train_mae_scores = []
        test_mae_scores = []
        
        for i in range(y_train.shape[1]):
            # 训练集指标
            train_r2 = r2_score(y_train[:, i], train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(y_train[:, i], train_pred[:, i]))
            train_mae = mean_absolute_error(y_train[:, i], train_pred[:, i])
            
            # 测试集指标
            test_r2 = r2_score(y_test[:, i], test_pred[:, i])
            test_rmse = np.sqrt(mean_squared_error(y_test[:, i], test_pred[:, i]))
            test_mae = mean_absolute_error(y_test[:, i], test_pred[:, i])
            
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)
            train_rmse_scores.append(train_rmse)
            test_rmse_scores.append(test_rmse)
            train_mae_scores.append(train_mae)
            test_mae_scores.append(test_mae)
        
        # 计算平均指标
        metrics['Train R²'] = np.mean(train_r2_scores)
        metrics['Test R²'] = np.mean(test_r2_scores)
        metrics['Train RMSE'] = np.mean(train_rmse_scores)
        metrics['Test RMSE'] = np.mean(test_rmse_scores)
        metrics['Train MAE'] = np.mean(train_mae_scores)
        metrics['Test MAE'] = np.mean(test_mae_scores)
        
    else:
        # 单输出情况
        metrics['Train R²'] = r2_score(y_train, train_pred)
        metrics['Test R²'] = r2_score(y_test, test_pred)
        metrics['Train RMSE'] = np.sqrt(mean_squared_error(y_train, train_pred))
        metrics['Test RMSE'] = np.sqrt(mean_squared_error(y_test, test_pred))
        metrics['Train MAE'] = mean_absolute_error(y_train, train_pred)
        metrics['Test MAE'] = mean_absolute_error(y_test, test_pred)
    
    return metrics


def calculate_multioutput_details(y_train, y_test, train_pred, test_pred):
    """
    计算多输出模型的详细指标
    
    Args:
        y_train: 训练集真实值
        y_test: 测试集真实值
        train_pred: 训练集预测值
        test_pred: 测试集预测值
    
    Returns:
        dict: 包含每个目标变量详细指标的字典
    """
    details = {}
    
    train_r2_per_target = []
    test_r2_per_target = []
    train_rmse_per_target = []
    test_rmse_per_target = []
    train_mae_per_target = []
    test_mae_per_target = []
    
    for i in range(y_train.shape[1]):
        # 每个目标变量的指标
        train_r2 = r2_score(y_train[:, i], train_pred[:, i])
        test_r2 = r2_score(y_test[:, i], test_pred[:, i])
        train_rmse = np.sqrt(mean_squared_error(y_train[:, i], train_pred[:, i]))
        test_rmse = np.sqrt(mean_squared_error(y_test[:, i], test_pred[:, i]))
        train_mae = mean_absolute_error(y_train[:, i], train_pred[:, i])
        test_mae = mean_absolute_error(y_test[:, i], test_pred[:, i])
        
        train_r2_per_target.append(train_r2)
        test_r2_per_target.append(test_r2)
        train_rmse_per_target.append(train_rmse)
        test_rmse_per_target.append(test_rmse)
        train_mae_per_target.append(train_mae)
        test_mae_per_target.append(test_mae)
    
    details['train_r2_per_target'] = train_r2_per_target
    details['test_r2_per_target'] = test_r2_per_target
    details['train_rmse_per_target'] = train_rmse_per_target
    details['test_rmse_per_target'] = test_rmse_per_target
    details['train_mae_per_target'] = train_mae_per_target
    details['test_mae_per_target'] = test_mae_per_target
    
    return details


def display_training_results(results, trained_models, detailed_results, is_multioutput):
    """
    显示训练结果
    
    Args:
        results: 结果列表
        trained_models: 训练好的模型字典
        detailed_results: 详细结果字典
        is_multioutput: 是否为多输出问题
    """
    st.session_state.trained_models = trained_models
    st.session_state.detailed_results = detailed_results
    
    # 转换为DataFrame并显示
    results_df = pd.DataFrame(results)
    
    st.success("🎉 模型训练完成！")
    
    # 显示结果表格
    st.subheader("📊 模型性能对比")
    
    # 格式化数值显示
    display_df = results_df.copy()
    numeric_cols = ['Training Time (s)', 'Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']
    
    for col in numeric_cols:
        if col in display_df.columns:
            if col == 'Training Time (s)':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}s")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # 添加交叉验证结果格式化
    cv_cols = ['CV R² Mean', 'CV R² Std']
    for col in cv_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # 找出最佳模型
    best_model_idx = results_df['Test R²'].idxmax()
    best_model = results_df.loc[best_model_idx, 'Model']
    best_r2 = results_df.loc[best_model_idx, 'Test R²']
    
    st.success(f"🏆 最佳模型: **{best_model}** (测试集 R² = {best_r2:.4f})")
    
    # 显示可视化
    show_model_performance_visualization(results_df, detailed_results, is_multioutput)


def show_status_message(message, message_type="info"):
    """
    显示状态消息
    
    Args:
        message: 消息内容
        message_type: 消息类型 ("info", "success", "warning", "error")
    """
    if message_type == "info":
        st.info(message)
    elif message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:
        st.write(message)


def check_data_prerequisites(need_labels=False, need_preprocessing=True):
    """
    检查数据前置条件
    
    Args:
        need_labels: 是否需要标签数据
        need_preprocessing: 是否需要预处理完成
    
    Returns:
        bool: 是否满足条件
    """
    # 检查数据是否加载
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        show_status_message("请先加载数据", "warning")
        return False
    
    # 检查是否需要标签数据
    if need_labels:
        if not hasattr(st.session_state, 'y') or st.session_state.y is None:
            show_status_message("此功能需要标签数据，请在数据加载页面输入标签", "warning")
            return False
    
    # 检查是否需要预处理完成
    if need_preprocessing:
        if not hasattr(st.session_state, 'preprocessing_done') or not st.session_state.preprocessing_done:
            show_status_message("请先完成数据预处理", "warning")
            return False
    
    return True


def get_current_data():
    """
    获取当前使用的数据
    
    Returns:
        tuple: (X, wavenumbers, info_message)
    """
    if hasattr(st.session_state, 'feature_selected') and st.session_state.feature_selected:
        X = st.session_state.X_final
        wavenumbers = st.session_state.wavenumbers_final
        info = f"✅ 使用特征选择后的数据，特征数量: {X.shape[1]}"
    elif hasattr(st.session_state, 'preprocessing_done') and st.session_state.preprocessing_done:
        X = st.session_state.X_preprocessed
        wavenumbers = st.session_state.wavenumbers_preprocessed
        info = f"ℹ️ 使用预处理后的全部特征，特征数量: {X.shape[1]}"
    else:
        X = st.session_state.X
        wavenumbers = st.session_state.wavenumbers
        info = f"⚠️ 使用原始数据，特征数量: {X.shape[1]}"
    # 确保返回numpy数组
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(wavenumbers, pd.Series):
        wavenumbers = wavenumbers.values
    
    return X, wavenumbers, info


def safe_execute(func, error_message="操作失败"):
    """
    安全执行函数，捕获异常
    
    Args:
        func: 要执行的函数
        error_message: 错误消息
    
    Returns:
        函数执行结果或None
    """
    try:
        return func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None


# ====================================
# 主函数
# ====================================

def main_with_trend_analysis():
    """带趋势分析的主函数"""
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
    
    # 更新页面字典，添加趋势分析
    pages = {
        "1. 数据加载与标签输入": show_data_loading_page,
        "2. 数据预处理": show_preprocessing_page,
        "3. 特征提取与可视化": show_feature_extraction_page,
        "4. 趋势分析": show_trend_analysis_page,  # 新增页面
        "5. 数据集划分": show_data_split_page,
        "6. 模型训练与评估": show_model_training_page,
        "7. 盲样预测": show_blind_prediction_page
    }
    
    # 页面选择
    selection = st.sidebar.radio("导航", list(pages.keys()))
    
    # 显示数据加载状态
    st.sidebar.markdown("---")
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        st.sidebar.success("✅ 数据已加载")
        if hasattr(st.session_state, 'X'):
            st.sidebar.write(f"📊 光谱数据: {st.session_state.X.shape}")
            if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                st.sidebar.write(f"🏷️ 标签数据: {st.session_state.y.shape}")
                st.sidebar.write(f"🎯 目标变量: {', '.join(st.session_state.selected_cols)}")
            else:
                st.sidebar.info("🔍 无标签数据 - 可进行趋势分析")
    else:
        st.sidebar.warning("⚠️ 请先加载数据")
    
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
        **咸数光谱数据分析与预测应用 v2.1**
        
        新增功能：
        - 🔍 **趋势分析**: 主副产物浓度趋势
        - 📊 **PCA分析**: 降维和模式识别
        - 🧪 **成分分解**: NMF/ICA分离化学成分
        - 📈 **时间趋势**: 工艺过程监控
        - 🔍 **聚类分析**: 样本分组和异常检测
        - 📋 **综合报告**: 自动生成分析报告
        
        **适用场景**：
        - 有标签数据：定量预测模型
        - 无标签数据：趋势分析和成分识别
        - 工艺监控：实时趋势跟踪
        - 反应过程：主副产物分析
        """
    )
    
    # 显示选定的页面
    page_func = pages[selection]
    page_func()


# 在原代码最后替换main函数调用
if __name__ == "__main__":
    main_with_trend_analysis()  # 使用新的main函数
