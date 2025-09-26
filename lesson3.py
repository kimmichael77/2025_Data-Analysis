import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def best_subset(X, y, cri='rss'):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    best_models = {}
    
    for k in range(1, n_features + 1):
        best_score = float('inf') if cri.lower() in ['aic', 'bic', 'rss'] else float('-inf')
        best_subset = None
        
        for subset in itertools.combinations(range(n_features), k):
            variables = [X.columns[i] for i in subset]
            
            # 메트릭 계산
            metrics = calculate_metrics(X, y, variables, n_samples, k + 1)  # +1 for intercept
            current_score = metrics[cri.upper()]
            
            # 최적 모델 선택 : AIC, BIC, RSS는 작을수록 좋고, Cp, Adj_R2는 클수록 좋음
            if cri.lower() in ['aic', 'bic', 'rss']:
                is_better = current_score < best_score
            else:  # cp, adj_r2
                is_better = current_score > best_score
            
            if is_better:
                best_score = current_score
                best_subset = subset
        
        best_models[k] = {
            'variables': [X.columns[i] for i in best_subset],
            cri: best_score,
            'indices': best_subset
        }
    
    return best_models


def forward_selection(X, y, cri='rss'):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    selected = []
    remaining = list(range(n_features))
    models = {}
    
    # 모든 변수를 선택할 때까지 반복
    for k in range(1, n_features + 1):
        best_score = float('inf') if cri.lower() in ['aic', 'bic', 'rss'] else float('-inf')
        best_feature = None
        
        for feature in remaining:
            current_features = selected + [feature]
            variables = [X.columns[i] for i in current_features]
            
            # 메트릭 계산
            metrics = calculate_metrics(X, y, variables, n_samples, len(current_features) + 1)  # +1 for intercept
            current_score = metrics[cri.upper()]
            
            # 최적 변수 선택
            if cri.lower() in ['aic', 'bic', 'rss']:
                is_better = current_score < best_score
            else:  # cp, adj_r2
                is_better = current_score > best_score
            
            if is_better:
                best_score = current_score
                best_feature = feature
        
        selected.append(best_feature)
        remaining.remove(best_feature)
        
        models[k] = {
            'variables': [X.columns[i] for i in selected],
            cri: best_score,
            'indices': tuple(selected)
        }
    
    return models


def backward_elimination(X, y, cri='rss'):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    selected = list(range(n_features))
    models = {}
    
    # 전체 모델부터 시작
    variables = list(X.columns)
    metrics = calculate_metrics(X, y, variables, n_samples, n_features + 1)  # +1 for intercept
    
    models[n_features] = {
        'variables': variables,
        cri: metrics[cri.upper()],
        'indices': tuple(selected)
    }
    
    # 변수를 하나씩 제거
    for k in range(n_features - 1, 0, -1):
        best_score = float('inf') if cri.lower() in ['aic', 'bic', 'rss'] else float('-inf')
        worst_feature = None
        
        for feature in selected:
            current_features = [f for f in selected if f != feature]
            variables = [X.columns[i] for i in current_features]
            
            # 메트릭 계산
            metrics = calculate_metrics(X, y, variables, n_samples, len(current_features) + 1)  # +1 for intercept
            current_score = metrics[cri.upper()]
            
            # 최적 변수 선택 (제거할 변수)
            if cri.lower() in ['aic', 'bic', 'rss']:
                is_better = current_score < best_score
            else:  # cp, adj_r2
                is_better = current_score > best_score
            
            if is_better:
                best_score = current_score
                worst_feature = feature
        
        selected.remove(worst_feature)
        
        models[k] = {
            'variables': [X.columns[i] for i in selected],
            cri: best_score,
            'indices': tuple(selected)
        }
    
    return models


def hybrid_stepwise(X, y, cri='rss'):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    selected = []
    models = {}
    
    # 모든 변수를 선택할 때까지 반복
    for k in range(1, n_features + 1):
        remaining = [i for i in range(n_features) if i not in selected]
        
        # Forward step: 가장 좋은 변수 추가
        best_score = float('inf') if cri.lower() in ['aic', 'bic', 'rss'] else float('-inf')
        best_feature = None
        
        for feature in remaining:
            current_features = selected + [feature]
            variables = [X.columns[i] for i in current_features]
            
            metrics = calculate_metrics(X, y, variables, n_samples, len(current_features) + 1)  # +1 for intercept
            current_score = metrics[cri.upper()]
            
            if cri.lower() in ['aic', 'bic', 'rss']:
                is_better = current_score < best_score
            else:  # cp, adj_r2
                is_better = current_score > best_score
            
            if is_better:
                best_score = current_score
                best_feature = feature
        
        selected.append(best_feature)
        
        # Backward step: 기존 변수 중 제거할 것이 있는지 확인 (k >= 2일 때)
        if k >= 2:
            best_removal_score = float('inf') if cri.lower() in ['aic', 'bic', 'rss'] else float('-inf')
            feature_to_remove = None
            
            for feature in selected[:-1]:  # 방금 추가한 변수는 제외
                temp_features = [f for f in selected if f != feature]
                variables = [X.columns[i] for i in temp_features]
                
                # 메트릭 계산
                metrics = calculate_metrics(X, y, variables, n_samples, len(temp_features) + 1)  # +1 for intercept
                current_score = metrics[cri.upper()]
                
                # 제거할 변수 선택
                if cri.lower() in ['aic', 'bic', 'rss']:
                    is_better = current_score < best_removal_score
                else:  # cp, adj_r2
                    is_better = current_score > best_removal_score
                
                if is_better:
                    best_removal_score = current_score
                    feature_to_remove = feature
            
            # 제거했을 때 더 좋아지면 제거
            if cri.lower() in ['aic', 'bic', 'rss']:
                should_remove = best_removal_score < best_score
            else:
                should_remove = best_removal_score > best_score
                
            if should_remove:
                selected.remove(feature_to_remove)
                best_score = best_removal_score
        
        models[len(selected)] = {
            'variables': [X.columns[i] for i in selected],
            cri: best_score,
            'indices': tuple(selected)
        }
    
    return models


def calculate_metrics(X, y, variables, n, p):
    X_subset = X[variables]
    X_subset = sm.add_constant(X_subset)
    
    model = sm.OLS(y, X_subset).fit()
    
    # RSS 계산
    rss = np.sum(model.resid ** 2)
    
    # MSE for Cp
    X_full = sm.add_constant(X)
    full_model = sm.OLS(y, X_full).fit()
    mse_full = np.sum(full_model.resid ** 2) / (n - X_full.shape[1])
    
    # Cp
    cp = rss / mse_full - n + 2 * p
    
    return {
        'AIC': model.aic,
        'BIC': model.bic,
        'CP': cp,
        'ADJ_R2': model.rsquared_adj,
        'RSS': rss
    }


def format_selection_results(X, y, models, method_name, selected_cri, n):
    """
    변수선택 결과를 DataFrame으로 정리하는 함수
    
    Parameters:
    -----------
    X : pandas.DataFrame, 설명변수 데이터
    y : pandas.Series, 반응변수 데이터
    models : dict, 변수선택 함수의 결과 (best_subset, forward_selection 등의 반환값)
    method_name : str, 변수선택 알고리즘 이름 ('최량 부분집합', '전진선택법' 등)
    selected_cri : str, 최적 조합 판단에 사용하는 메트릭 (예: 'aic', 'bic', 'cp', 'adj_r2', 'rss')
    n : int, 표본 크기
        
    Returns:
    --------
    pandas.DataFrame, 정리된 결과 테이블
    """
    results_df = []
    
    for k, model_info in models.items():
        variables = model_info['variables']
        
        # 모든 메트릭 계산
        metrics = calculate_metrics(X, y, variables, n, k + 1)  # +1 for intercept
        
        row = {
            'k': k,
            'variables': ', '.join(variables),
            'selected_criterion': model_info[selected_cri],  # 선택에 사용된 기준값
            'AIC': metrics['AIC'],
            'BIC': metrics['BIC'],
            'CP': metrics['CP'],
            'ADJ_R2': metrics['ADJ_R2'],
            'RSS': metrics['RSS']
        }
        
        # 선택된 기준을 앞쪽으로 이동
        ordered_row = {
            'k': row['k'],
            'variables': row['variables'],
            selected_cri.upper(): row['selected_criterion']
        }
        
        # 나머지 메트릭들 추가 (선택된 기준 제외)
        for metric in ['AIC', 'BIC', 'CP', 'ADJ_R2', 'RSS']:
            if metric != selected_cri.upper():
                ordered_row[metric] = row[metric]
                
        results_df.append(ordered_row)
    
    df = pd.DataFrame(results_df)
    
    return df


def compare_methods_results(X, y, models_dict, n, k_target=None):
    """
    여러 변수선택 방법들의 결과를 비교하는 함수
    
    Parameters:
    -----------
    X : pandas.DataFrame, 설명변수 데이터
    y : pandas.Series, 반응변수 데이터
    models_dict : dict, 각 선택방법의 결과를 담은 딕셔너리
        형태: {'method_name': (models, criterion), ...}
    n : int, 표본 크기
    k_target : int, optional, 비교할 변수 개수 (None이면 모든 k에 대해 비교)
        
    Returns:
    --------
    pandas.DataFrame, 선택방법별 비교 결과 테이블
    """
    comparison_results = []
    
    for method_name, (models, criterion) in models_dict.items():
        if k_target is None:
            # 모든 k에 대해 처리
            for k, model_info in models.items():
                variables = model_info['variables']
                metrics = calculate_metrics(X, y, variables, n, k + 1)
                
                row = {
                    'Method': method_name,
                    'k': k,
                    'Variables': ', '.join(variables),
                    'Criterion_Used': criterion.upper(),
                    'Criterion_Value': model_info[criterion],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'CP': metrics['CP'],
                    'ADJ_R2': metrics['ADJ_R2'],
                    'RSS': metrics['RSS']
                }
                comparison_results.append(row)
        else:
            # 특정 k에 대해서만 처리
            if k_target in models:
                model_info = models[k_target]
                variables = model_info['variables']
                metrics = calculate_metrics(X, y, variables, n, k_target + 1)
                
                row = {
                    'Method': method_name,
                    'Variables': ', '.join(variables),
                    'Criterion_Used': criterion.upper(),
                    'Criterion_Value': model_info[criterion],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'CP': metrics['CP'],
                    'ADJ_R2': metrics['ADJ_R2'],
                    'RSS': metrics['RSS']
                }
                comparison_results.append(row)
    
    return pd.DataFrame(comparison_results)


def find_optimal_models_by_criteria(X, y, models, n):
    """
    각 기준별로 최적 모델을 찾는 함수
    
    Parameters:
    -----------
    X : pandas.DataFrame, 설명변수 데이터
    y : pandas.Series, 반응변수 데이터
    models : dict, 변수선택 결과 (예: best_subset의 반환값)
    n : int, 표본 크기
        
    Returns:
    --------
    pandas.DataFrame, 각 기준별 최적 모델 정보
    """
    # 모든 모델의 메트릭 계산
    all_results = []
    for k, model_info in models.items():
        variables = model_info['variables']
        metrics = calculate_metrics(X, y, variables, n, k + 1)
        
        result = {
            'k': k,
            'variables': variables,
            'AIC': metrics['AIC'],
            'BIC': metrics['BIC'],
            'CP': metrics['CP'],
            'ADJ_R2': metrics['ADJ_R2'],
            'RSS': metrics['RSS']
        }
        all_results.append(result)
    
    # 각 기준별 최적 모델 찾기
    optimal_models = []
    criteria = ['AIC', 'BIC', 'CP', 'ADJ_R2', 'RSS']
    
    for criterion in criteria:
        if criterion == 'ADJ_R2':
            # Adjusted R²는 클수록 좋음
            best_result = max(all_results, key=lambda x: x[criterion])
        else:
            # AIC, BIC, CP, RSS는 작을수록 좋음
            best_result = min(all_results, key=lambda x: x[criterion])
        
        optimal_models.append({
            'Criterion': criterion,
            'Optimal_k': best_result['k'],
            'Variables': ', '.join(best_result['variables']),
            'AIC': best_result['AIC'],
            'BIC': best_result['BIC'],
            'CP': best_result['CP'],
            'ADJ_R2': best_result['ADJ_R2'],
            'RSS': best_result['RSS']
        })
    
    return pd.DataFrame(optimal_models)


def compute_coefficient_paths(X, y, alphas):
    from sklearn.linear_model import Ridge, Lasso
    
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        ridge_coefs.append(ridge.coef_)
        
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=1000)
        lasso.fit(X, y)
        lasso_coefs.append(lasso.coef_)
    
    return np.array(ridge_coefs), np.array(lasso_coefs)


# 한글 폰트 설정 함수
def setup_korean_font():
    import matplotlib.pyplot as plt
    import platform
    
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 110


def plot_ice_curves(model, X_data, feature_names, n_samples=50, figsize=(18, 6)):
    """
    GAM 모델에 대한 ICE(Individual Conditional Expectation) 플롯을 생성하는 함수
    
    Parameters:
    -----------
    model : sklearn pipeline 또는 fitted model, 훈련된 GAM 모델 (예: sklearn Pipeline with splines)
    X_data : pandas.DataFrame, 입력 데이터 (feature_names에 해당하는 컬럼들을 포함)
    feature_names : list, ICE 플롯을 그릴 연속형 변수명들 (예: ['age', 'year'])
    n_samples : int, default=50, ICE 곡선을 그릴 샘플 개수
    figsize : tuple, default=(18, 6), 그래프 크기
        
    Returns:
    --------
    dict : ICE 데이터와 분석 결과를 담은 딕셔너리
        - 'ice_data': 각 변수별 ICE 데이터
        - 'pdp_data': 각 변수별 PDP 데이터
        - 'heterogeneity': 각 변수별 이질성 정보
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 샘플 선택
    sample_indices = np.random.choice(len(X_data), n_samples, replace=False)
    
    # 연속형 변수 개수에 따라 subplot 구성
    n_features = len(feature_names)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    # 단일 변수인 경우 list로 변환
    if n_features == 1:
        axes = [axes]
    
    ice_results = {}
    pdp_results = {}
    heterogeneity_results = {}
    
    # 각 연속형 변수에 대해 ICE 플롯 생성
    for idx, feature in enumerate(feature_names):
        # 변수 범위 설정
        feature_range = np.linspace(X_data[feature].min(), X_data[feature].max(), 50)
        ice_data = []
        
        # 고정값들 계산 (다른 변수들의 중위값/최빈값)
        fixed_values = {}
        for col in X_data.columns:
            if col != feature:
                if X_data[col].dtype in ['int64', 'float64']:
                    fixed_values[col] = X_data[col].median()
                else:
                    fixed_values[col] = X_data[col].mode()[0]
        
        # 각 샘플에 대한 ICE 곡선 계산
        for sample_idx in sample_indices:
            sample_data = X_data.iloc[sample_idx]
            ice_predictions = []
            
            for feature_val in feature_range:
                # 예측용 데이터 생성
                X_ice = pd.DataFrame({col: [sample_data[col] if col != feature else feature_val] 
                                    for col in X_data.columns})
                
                pred = model.predict(X_ice)[0]
                ice_predictions.append(pred)
            
            ice_data.append(ice_predictions)
            # ICE 곡선 그리기 (얇은 선)
            axes[idx].plot(feature_range, ice_predictions, alpha=0.3, 
                          color='blue' if idx == 0 else 'green', linewidth=0.8)
        
        # PDP (평균) 계산 및 그리기, 빨간색으로 표시
        pdp = np.mean(ice_data, axis=0)
        axes[idx].plot(feature_range, pdp, color='red', linewidth=3, label='PDP (평균)')
        
        # 그래프 설정
        axes[idx].set_xlabel(feature.capitalize())
        axes[idx].set_ylabel('Predicted Value')
        axes[idx].set_title(f'ICE 플롯: {feature.capitalize()} 효과')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # 결과 저장
        ice_results[feature] = np.array(ice_data)
        pdp_results[feature] = pdp
        
        # 이질성 계산
        ice_std = np.std(ice_data, axis=0)
        heterogeneity = np.mean(ice_std) / np.mean(pdp) * 100
        heterogeneity_results[feature] = {
            'mean_std': np.mean(ice_std),
            'heterogeneity_pct': heterogeneity,
            'range_effect': np.max(pdp) - np.min(pdp)
        }
    
    plt.tight_layout()
    plt.show()
    
    return {
        'ice_data': ice_results,
        'pdp_data': pdp_results,
        'heterogeneity': heterogeneity_results
    }


def plot_categorical_ice(model, X_data, categorical_feature, feature_labels=None, 
                        n_samples=50, figsize=(10, 6)):
    """
    범주형 변수에 대한 ICE 플롯을 박스플롯으로 생성하는 함수
    
    Parameters:
    -----------
    model : sklearn pipeline 또는 fitted model
        훈련된 모델
    X_data : pandas.DataFrame
        입력 데이터
    categorical_feature : str
        범주형 변수명
    feature_labels : list, optional
        범주 레이블들 (None이면 고유값 사용)
    n_samples : int, default=50
        샘플 개수
    figsize : tuple, default=(10, 6)
        그래프 크기
        
    Returns:
    --------
    dict : 범주별 예측값 분포 데이터
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 샘플 선택
    sample_indices = np.random.choice(len(X_data), n_samples, replace=False)
    
    # 범주 수준들
    category_levels = sorted(X_data[categorical_feature].unique())
    if feature_labels is None:
        feature_labels = [str(level) for level in category_levels]
    
    # 각 범주별 예측값 저장
    category_ice_data = {level: [] for level in category_levels}
    
    # 각 샘플에 대해 범주별 예측
    for sample_idx in sample_indices:
        sample_data = X_data.iloc[sample_idx]
        
        for level in category_levels:
            # 예측용 데이터 생성
            X_ice = pd.DataFrame({col: [sample_data[col] if col != categorical_feature else level] 
                                for col in X_data.columns})
            
            pred = model.predict(X_ice)[0]
            category_ice_data[level].append(pred)
    
    # 박스플롯 생성
    plt.figure(figsize=figsize)
    box_data = [category_ice_data[level] for level in category_levels]
    box_plot = plt.boxplot(box_data, labels=feature_labels, patch_artist=True)
    
    # 박스플롯 색상 설정
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for i, patch in enumerate(box_plot['boxes']):
        color_idx = i % len(colors)
        patch.set_facecolor(colors[color_idx])
        patch.set_alpha(0.7)
    
    plt.xlabel(categorical_feature.capitalize())
    plt.ylabel('Predicted Value')
    plt.title(f'ICE 플롯: {categorical_feature.capitalize()} 효과 (분포)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45 if len(feature_labels) > 3 else 0)
    plt.tight_layout()
    plt.show()
    
    return category_ice_data


def analyze_ice_heterogeneity(ice_results):
    """
    ICE 플롯 결과의 이질성을 분석하는 함수
    
    Parameters:
    -----------
    ice_results : dict
        plot_ice_curves() 함수의 반환값
        
    Returns:
    --------
    pandas.DataFrame : 변수별 이질성 분석 결과
    """
    import pandas as pd
    
    analysis_results = []
    
    for feature, heterogeneity_info in ice_results['heterogeneity'].items():
        analysis_results.append({
            'Feature': feature.capitalize(),
            'Range_Effect': heterogeneity_info['range_effect'],
            'Mean_Std': heterogeneity_info['mean_std'],
            'Heterogeneity_%': heterogeneity_info['heterogeneity_pct'],
            'Interpretation': 'Low' if heterogeneity_info['heterogeneity_pct'] < 15 
                            else 'Medium' if heterogeneity_info['heterogeneity_pct'] < 25 
                            else 'High'
        })
    
    return pd.DataFrame(analysis_results)


if __name__ == "__main__":
    print("lesson3.py - 변수선택 및 회귀분석 함수 모듈")
    print("사용 가능한 함수들:")
    print("- best_subset(X, y, cri='rss'): 최량 부분집합 선택")
    print("- forward_selection(X, y, cri='rss'): 전진선택법")
    print("- backward_elimination(X, y, cri='rss'): 후진소거법")
    print("- hybrid_stepwise(X, y, cri='rss'): 하이브리드 단계적 선택법")
    print("- calculate_metrics(): AIC, BIC, Cp, Adj R², RSS 계산")
    print("- compute_coefficient_paths(): Ridge/Lasso 계수 경로")
    print("- setup_korean_font(): 한글 폰트 설정")
    print("\n결과 정리 함수들:")
    print("- format_selection_results(): 변수선택 결과를 DataFrame으로 정리")
    print("- compare_methods_results(): 여러 방법들의 결과 비교")
    print("- find_optimal_models_by_criteria(): 각 기준별 최적 모델 찾기")
    print("\nICE 플롯 함수들:")
    print("- plot_ice_curves(): 연속형 변수 ICE 플롯 생성")
    print("- plot_categorical_ice(): 범주형 변수 ICE 플롯 생성 (박스플롯)")
    print("- analyze_ice_heterogeneity(): ICE 플롯 이질성 분석")
    print("\n사용 가능한 기준 (cri):")
    print("- 'aic': Akaike Information Criterion")
    print("- 'bic': Bayesian Information Criterion")
    print("- 'cp': Mallows' Cp")
    print("- 'adj_r2': Adjusted R-squared")
    print("- 'rss': Residual Sum of Squares (기본값)")
