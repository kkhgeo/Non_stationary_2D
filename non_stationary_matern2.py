import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
import rasterio
from rasterio.transform import from_origin
import os
import warnings

# --- 1. 초기 설정 및 보조 함수 ---

# 경고 억제 및 한글/마이너스 폰트 설정
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 랜덤 시드 설정
np.random.seed(42)

def matern_covariance(distances, variance=1.0, range_param=1.0, smoothness=1.5):
    """
    주어진 거리에 대한 Matérn 공분산 값을 계산합니다.
    거리가 0일 때와 0이 아닐 때를 분리하여 수치적 안정성을 확보합니다.
    """
    cov = np.zeros_like(distances, dtype=float)
    
    # 거리가 0인 경우 (대각선 성분)
    mask_zero = distances == 0
    cov[mask_zero] = variance
    
    # 거리가 0보다 큰 경우
    mask_pos = distances > 0
    if np.any(mask_pos):
        non_zero_dist = distances[mask_pos]
        # Matérn 함수의 상수 부분
        const_part = variance * (2**(1 - smoothness) / gamma(smoothness))
        # 스케일링된 거리
        scaled_dist = np.sqrt(2 * smoothness) * non_zero_dist / range_param
        # 최종 공분산 값 계산
        cov[mask_pos] = const_part * (scaled_dist**smoothness) * kv(smoothness, scaled_dist)
        
    return cov

def generate_anisotropic_nonstationary_data(
    n=50, 
    smoothness=1.5,
    variance_range=(1.0, 1.0),
    range_param_range=(0.2, 0.2),
    angle_range_deg=(0.0, 0.0),
    ratio_range=(1.0, 1.0)
):
    """
    비정상성(Non-stationary)과 이방성(Anisotropic)을 모두 가진 공간 데이터를 생성합니다.
    모든 통계적 특성이 공간에 따라 선형적으로 변합니다.
    """
    print("🚀 비정상성 및 이방성 데이터 생성을 시작합니다...")
    
    N = n**2
    # 1. 공간 좌표 생성
    coord1 = np.linspace(0, 1, n)
    coord2 = np.linspace(0, 1, n)
    s1, s2 = np.meshgrid(coord1, coord2)
    s = np.vstack((s1.flatten(), s2.flatten())).T

    # 2. 위치에 따라 변하는 '매개변수 필드' 생성
    # x좌표(s[:, 0])를 기준으로 값이 변하도록 설정
    variance_field = variance_range[0] + (variance_range[1] - variance_range[0]) * s[:, 0]
    range_param_field = range_param_range[0] + (range_param_range[1] - range_param_range[0]) * s[:, 0]
    angle_rad_field = np.deg2rad(angle_range_deg[0] + (angle_range_deg[1] - angle_range_deg[0]) * s[:, 0])
    ratio_field = ratio_range[0] + (ratio_range[1] - ratio_range[0]) * s[:, 0]

    # 3. 지역적(Local) 특성을 이용한 공분산 행렬 계산
    cov_matrix = np.zeros((N, N))
    print("... 공분산 행렬 계산 중 (시간이 소요될 수 있습니다) ...")
    for i in range(N):
        for j in range(i, N):
            # 두 지점 (i, j)의 평균적인 '지역' 특성을 계산
            local_variance = np.sqrt(variance_field[i] * variance_field[j])
            local_range = (range_param_field[i] + range_param_field[j]) / 2.0
            local_angle = (angle_rad_field[i] + angle_rad_field[j]) / 2.0
            local_ratio = (ratio_field[i] + ratio_field[j]) / 2.0
            
            # 지역 특성에 기반한 이방성 변환 행렬 생성
            cos_a, sin_a = np.cos(local_angle), np.sin(local_angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            scaling_matrix = np.array([[1.0, 0], [0, local_ratio]]) # 비율을 한 축에만 적용
            
            # 좌표 변환을 통해 이방성 거리 계산
            transform_matrix = rotation_matrix @ scaling_matrix @ rotation_matrix.T
            displacement = s[i] - s[j]
            anisotropic_dist = np.sqrt(displacement @ transform_matrix @ displacement.T)

            # 지역적 분산과 상관거리를 사용하여 공분산 값 계산
            cov_value = matern_covariance(anisotropic_dist, 
                                          variance=local_variance, 
                                          range_param=local_range, 
                                          smoothness=smoothness)
            
            cov_matrix[i, j] = cov_matrix[j, i] = cov_value

    # 4. Cholesky 분해를 통해 랜덤 필드 생성
    # 수치 안정성을 위해 작은 값(nugget)을 대각선에 더함
    L = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(N))
    y = L @ np.random.randn(N)
    
    print("✅ 데이터 생성 완료!")
    return s, y

def main():
    """메인 실행 함수"""
    
    # ▼▼▼▼▼ 여기서 시뮬레이션 매개변수를 자유롭게 수정하세요 ▼▼▼▼▼
    # 각 매개변수는 (시작값, 끝값) 형태로 지정합니다.
    # 시작값과 끝값이 같으면 해당 특성은 정상성(stationary)을 가집니다.
    params = {
        "n": 64,                        # 그리드 해상도 (2의 거듭제곱으로 딥러닝에 최적)
        "smoothness": 1.5,              # Matérn 필드의 부드러움
        
        # --- 비정상성 및 이방성 상세 설정 (딥러닝 학습용) ---
        "variance_range": (0.5, 3.0),   # 분산의 변화 범위 (적당한 대비)
        "range_param_range": (0.05, 0.5),# 상관 거리의 변화 범위 (세밀함-부드러움)
        "angle_range_deg": (-30, 60),   # 이방성 각도의 변화 범위 (방향성 학습)
        "ratio_range": (1.5, 3.0),     # 이방성 비율의 변화 범위 (적당한 길쭉함)
    }
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    # 1. 비정상성 데이터 생성
    _, data = generate_anisotropic_nonstationary_data(**params)
    
    data_grid = data.reshape(params["n"], params["n"])

    # 2. GeoTIFF로 저장
    output_dir = "."  # 현재 디렉토리에 저장
    save_path = os.path.join(output_dir, "nonstationary_anisotropic_data.tif")
    
    transform = from_origin(0, 1, 1/params["n"], 1/params["n"])

    with rasterio.open(
        save_path, 'w', driver='GTiff',
        height=data_grid.shape[0], width=data_grid.shape[1],
        count=1, dtype=data_grid.dtype,
        crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(data_grid, 1)
    print(f"💾 GeoTIFF로 저장 완료: {save_path}")
    
    # 3. 생성된 데이터 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(data_grid, extent=[0, 1, 0, 1], origin="lower", cmap='jet')
    plt.colorbar(label="Field Value")
    
    # 매개변수 정보를 포함한 타이틀 생성
    title = f"Non-stationary Anisotropic Matérn Field\n"
    title += f"Variance: {params['variance_range'][0]:.1f}-{params['variance_range'][1]:.1f}, "
    title += f"Range: {params['range_param_range'][0]:.2f}-{params['range_param_range'][1]:.2f}\n"
    title += f"Angle: {params['angle_range_deg'][0]}°-{params['angle_range_deg'][1]}°, "
    title += f"Ratio: {params['ratio_range'][0]:.1f}-{params['ratio_range'][1]:.1f}, "
    title += f"Smoothness: {params['smoothness']:.1f}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()