import os
import sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

def load_config(
    config_name: str = "config", 
    overrides: list = None
) -> DictConfig:
    """
    [노트북/테스트용] Hydra 설정을 간편하게 로드하는 헬퍼 함수입니다.
    
    Args:
        config_name (str): 불러올 메인 설정 파일 이름 (.yaml 제외)
        config_path (str): 설정 폴더의 상대 경로 (호출 위치 기준)
        overrides (list): 덮어쓸 설정 리스트 (예: ["model=gemma", "system.device=cpu"])
    
    Returns:
        DictConfig: Hydra 설정 객체 (cfg)
    """
    # 1. 현재 파일(config_loader.py)의 위치를 기준으로 Project Root 찾기
    # 경로: src/utils/config_loader.py -> src/utils -> src -> Project_Root
    current_file_path = os.path.abspath(__file__)
    src_utils_dir = os.path.dirname(current_file_path)
    src_dir = os.path.dirname(src_utils_dir)
    project_root = os.path.dirname(src_dir)
    
    # 2. sys.path에 루트 추가 (모듈 import 문제 방지)
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 3. Hydra 인스턴스 초기화 (재실행 에러 방지)
    GlobalHydra.instance().clear()

    # 4. Config 폴더의 절대 경로 생성
    # 구조상 Root/config 에 위치함
    config_dir = os.path.join(project_root, "config")
    
    # 폴더가 실제로 있는지 확인 (디버깅용)
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"❌ Config 폴더를 찾을 수 없습니다: {config_dir}")

    # 5. initialize_config_dir 사용 (절대 경로 지원)
    # version_base=None은 최신 Hydra 필수 설정
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides if overrides else [])

    return cfg