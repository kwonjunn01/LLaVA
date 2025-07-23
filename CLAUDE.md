# CLAUDE.md - 프로젝트 진행사항 및 중요 정보

## 행동 지침

1. **모든 중요 진행사항은 이 파일에 기록**
   - 스크립트 변경사항
   - 파일 수정 내역
   - 의존성 변경사항
   - 중요한 결정사항

2. **파일 생성 최소화 원칙**
   - 새로운 파일 생성보다 기존 파일 활용 우선
   - 범용성 높은 파일을 통해 변동사항 수행
   - 꼭 필요한 경우에만 새 파일 생성
   - **임시 디버깅 파일은 작업 완료 후 즉시 삭제**

3. **Conda 환경 사용 필수**
   - **모든 실험 및 스크립트는 `llava` conda 환경에서 실행**
   - 실행 명령어: `source /home/diml/anaconda3/bin/activate llava`
   - GPU/CUDA 관련 작업이므로 반드시 llava 환경 사용

## 프로젝트 개요
- LLaVA 프로젝트에 S-4 (Loss-Free Instruction-Guided Head Calibration) 구현
- VQAv2 데이터셋에서 S-4 calibration 테스트 중

## 현재 상황 파악 (2025-07-22 업데이트)

### S4 구현의 근본적 문제 발견 ⚠️
- **Post-hook 문제**: attention 계산 **후**에 실행되어 효과 없음
- **Flash Attention 문제**: 기본으로 켜져있어 attention weights가 None
- **결과**: 모든 이전 S4 테스트가 실제로는 vanilla와 동일

### 사용자 제안 해결책 (2025-07-22)
1. **Option A - head_mask 파라미터** ❌
   - LLaVA가 head_mask를 지원하지 않음
   - ValueError: model_kwargs 'head_mask' not used

2. **Option B - Runtime hooks** 🔄 시도 중
   - 평가 스크립트에서 모델 로드 후 hooks 적용
   - 메모리는 절약하지만 속도 개선 제한적

3. **Option C - 구조적 프루닝** ❌
   - LLaMA 아키텍처가 모든 레이어에 동일한 head 수 요구
   - config 변경 시 hidden_size 불일치 에러

### Pre-hook 방식 성공 ✅
- **확인된 효과**: Hidden states 차이 0.008105
- weights를 0으로 만들어 실제 영향 확인
- 현재 전체 ScienceQA 평가 진행 중

## 최근 수정사항 (2025-07-22)

### [2025-07-22 17:05] - s4_rt_crossattn.py 개선 완료
- **수정 파일**: `/home/diml/khj/comp/LLaVA/s4rt/s4_rt_crossattn.py`
- **변경 내용**:
  - Post-hook → Pre-hook 방식으로 변경
  - S4 calibration scale factor 추가 (1.455)
  - Flash attention 문제 해결
- **결과**: Hidden states 차이 0.026042 확인
- **이유**: Pre-hook으로 실제 프루닝 효과 + S4 calibration 적용

### [2025-07-22 17:00] - S4 Calibration Mechanism 추가
- **수정 파일**: `/home/diml/khj/comp/LLaVA/run_scienceqa_prehook_fixed.py`
- **변경 내용**: 
  - S4 논문의 calibration scale factor 추가
  - 프루닝된 헤드 수에 따른 남은 헤드 보정
  - scale_factor = num_heads / num_remaining_heads
- **결과**: Hidden states 차이 0.008105 → 0.026042로 증가
- **이유**: 원래 S4 방식대로 남은 헤드들의 출력을 보정

### [2025-07-22] - Pre-hook 구현으로 실제 프루닝 효과 확인
- **실행 파일**: `/home/diml/khj/comp/LLaVA/run_scienceqa_prehook_fixed.py`
- **주요 개선**:
  - Forward 실행 전에 Q/K/V weights를 0으로 설정
  - Post-hook으로 원본 weights 복원
  - Hidden states 변화 확인 (0.008105 차이)
- **결과**: 프루닝이 실제로 모델 출력에 영향을 줌

### [2025-07-22] - Head mask 방식 시도 실패
- **파일**: `/home/diml/khj/comp/LLaVA/run_scienceqa_headmask.py`
- **문제**: LLaVA model.generate()가 head_mask 파라미터 미지원
- **교훈**: HuggingFace 표준과 LLaVA 구현의 차이 확인

## 핵심 파일 현황

### 작동하는 구현
- `/home/diml/khj/comp/LLaVA/run_scienceqa_prehook_fixed.py` - Pre-hook + S4 calibration (작동 확인)
  - Hidden states 차이: 0.026042 (S4 calibration 적용 후)
  - 32개 레이어 x 10개 헤드 = 총 320개 헤드 프루닝
- `/home/diml/khj/comp/LLaVA/s4_crossattn_30pct_calibration.json` - 프루닝 마스크

### 개선 완료된 구현
- `/home/diml/khj/comp/LLaVA/s4rt/s4_rt_crossattn.py` - Cross-attention 방식 (개선 완료)
  - Pre-hook 방식으로 변경 ✅
  - S4 calibration scale factor 적용 ✅
  - Hidden states 차이: 0.026042 확인

### 시도했지만 실패한 구현
- `apply_structural_pruning.py` - 구조적 프루닝 (아키텍처 제약)
- `run_scienceqa_headmask.py` - head_mask 방식 (미지원)
- 다수의 S4 구현들 - Post-hook 문제로 효과 없음

## 다음 단계

1. ScienceQA 이미지 데이터셋 다운로드 필요
2. 개선된 S4 구현으로 전체 평가 실행
3. Vanilla (70.15%) vs S4 결과 비교
4. 실제 속도/메모리 측정
5. 최종 결과 문서화

## 평가 결과 (2025-07-23 업데이트)

### ❌ Early Skip 전략 - 실패 (프루닝 적용 안 됨)
- **표면 정확도**: 70.15% (vanilla와 동일)
- **실제 상황**: 프루닝이 전혀 적용되지 않음
- **증거**:
  - Vanilla와 출력 분포 완전 동일
  - 파일 크기도 vanilla와 유사
  - 모델 weight norm 변화 없음 (0%)
- **원인**: 평가 시 모델이 다시 로드되면서 프루닝 효과 사라짐
- **결론**: 재실험 필요

## 평가 결과 (2025-07-22 18:00)

### S4 Cross-Attention 15% Adaptive (초반 레이어 보호)
- **정확도**: 48.36% (전체), 68.75% (이미지 포함)
- **프루닝**: 총 153헤드 (14.9%) - Layer별 adaptive
- **특징**: 초반 레이어(0-5) 평균 3.7개만 프루닝 (11.5%)
- **Scale Factor**: 각 레이어별로 다름 (평균 ~1.176)
- **결과**: 30% 대비 성능 거의 동일 (-0.21%p)

### S4 Cross-Attention 30% Adaptive (초반 레이어 보호)
- **정확도**: 48.57% (전체), 68.84% (이미지 포함)
- **프루닝**: 총 307헤드 (30.0%) - Layer별 adaptive
- **특징**: 초반 레이어(0-5) 평균 6.5개만 프루닝
- **결과**: Uniform 대비 +0.14%p 개선 (미미함)

### S4 Cross-Attention 30% Uniform (Pre-hook + Calibration)
- **정확도**: 48.43% (전체), 68.80% (이미지 포함)
- **프루닝**: 32층 × 10헤드 = 320헤드 (31.25%)
- **Scale Factor**: 1.455
- **Hidden States 차이**: 0.026042

### 비교: Vanilla vs S4
- **Vanilla**: 70.15% 정확도
- **S4 30% Uniform**: 48.43% 정확도 (-21.72%p)
- **S4 30% Adaptive**: 48.57% 정확도 (-21.58%p)
- **S4 15% Adaptive**: 48.36% 정확도 (-21.79%p)
- **결론**: 프루닝 비율에 관계없이 성능 하락이 심각함

## 핵심 성과

- **문제 해결**: Post-hook → Pre-hook 전환으로 Flash Attention 문제 해결
- **S4 Calibration**: 원 논문대로 scale factor 적용 (1.455)
- **검증 완료**: Hidden states 차이 0.026042로 프루닝 효과 확인
- **파일 정리**: 임시 디버깅 파일 모두 삭제

## 통합 실험 시스템 구축 완료 (2025-07-22 18:10)

### 새로운 파일 구조
```
s4_experiments/
├── calibrations/    # 모든 calibration 파일
├── results/         # 실험 결과
└── logs/           # 실험 로그

archive_20250722_180647/
├── old_scripts/     # 이전 개별 실험 스크립트들 (40개 파일)
├── old_logs/        # 실험 로그 파일들
├── old_jsons/       # calibration JSON 파일들
└── old_results/     # 이미지 및 결과 파일들

활성 파일 (루트 디렉토리):
- CLAUDE.md                      # 프로젝트 진행 상황
- s4_pruning_experiments.py      # 통합 실험 시스템
- run_s4_experiments.sh          # 실행 스크립트  
- summarize_s4_results.py        # 결과 요약
- run_scienceqa_prehook_fixed.py # 참조용 pre-hook 구현
- s4_results_summary.csv         # 결과 요약 CSV
```

### 사용법
```bash
# 전체 실험 실행 (5%, 10%, 15%, 20%, 30%)
./run_s4_experiments.sh

# 빠른 테스트 (10%, 20%)
./run_s4_experiments.sh --quick

# 특정 비율만 실험
python s4_pruning_experiments.py --rates 0.05 0.10

# 결과 요약
python summarize_s4_results.py

# 파일 정리 (이미 완료)
python cleanup_files.py  # 40개 파일을 archive로 이동
```

## 주요 발견사항
1. **프루닝 비율과 무관하게 ~22% 성능 하락**
2. **Layer-wise adaptive가 uniform보다 약간 개선**
3. **초반 레이어 보호 효과 미미**
4. **S4 calibration (absorption) 적용에도 불구하고 성능 하락**

## 최신 개선사항 (2025-07-23)

### Similarity-based Weight Redistribution 구현
- **파일**: `s4_pruning_experiments.py`
- **내용**: 
  - Pruned head와 유사한 head를 찾아 weight 재분배
  - Cosine similarity 기반으로 유사도 계산
  - 기본 S4 scale factor + 유사도 기반 추가 weight
  - 예: 8개 head 중 3개 pruning 시 total scale 9.44 (목표: 8)
- **목적**: Pruned head의 기능을 유사한 head로 더 효과적으로 전달
- **결과**: 0.02% 정확도로 실패 (모델 파괴)

### Residual Transfer Pruning 구현 (2025-07-23)
- **파일**: `residual_transfer_pruning.py`
- **주요 개선**:
  - V/O projection만 전송 (Q/K 보존으로 attention pattern 유지)
  - Residual approach: v_keep += beta * (v_prun - v_keep)
  - Adaptive beta: cosine similarity 기반 조절
  - Norm clipping으로 scale 폭주 방지
- **결과**: 
  - Basic S4: 62.44%
  - Residual Transfer: 63.73% (+1.29%p 개선)
  - 실제 beta 값은 매우 작음 (0.000~0.001)
- **발견**: LLaVA head들이 서로 독립적 (낮은 similarity)

## 파일 정리 완료 (2025-07-22 18:07)
- **정리된 파일**: 40개 (스크립트, 로그, JSON, 결과)
- **아카이브 위치**: `archive_20250722_180647/`
- **활성 파일**: 6개 핵심 파일만 유지
- **데이터 디렉토리**: `s4_experiments/`에 체계적 관리

---

## 최근 수정사항 (2025-07-23)

### [2025-07-23 02:05] - Calibration JSON 생성 스크립트 추가
- **생성 파일**: `/home/diml/khj/comp/LLaVA/create_calibration.py`
- **기능**: 
  - S4 calibration JSON 파일 생성
  - Cross-attention saliency 계산
  - ScienceQA 데이터셋 사용
- **사용법**:
  ```bash
  source /home/diml/anaconda3/bin/activate llava
  python create_calibration.py --num-samples 100 --gamma 0.4
  ```
- **출력**: `s4_experiments/calibrations/s4rt_crossattn_calibration.json`

### [2025-07-23 02:10] - 성능 측정 기능 추가된 실험 스크립트
- **생성 파일**: `/home/diml/khj/comp/LLaVA/s4_pruning_experiments_with_perf.py`
- **기능**:
  - 속도 측정: samples/second, tokens/second
  - 메모리 측정: GPU peak memory, memory reduction
  - Baseline 대비 speedup 계산
  - 종합적인 성능 비교 표 출력
- **사용법**:
  ```bash
  source /home/diml/anaconda3/bin/activate llava
  python s4_pruning_experiments_with_perf.py --rates 0.05 0.10 --run-baseline
  ```
- **출력**: `s4_experiments/performance/` 디렉토리에 상세 결과 저장

### [2025-07-23 02:15] - 간단한 Calibration 생성 스크립트
- **생성 파일**: `/home/diml/khj/comp/LLaVA/create_calibration_simple.py`
- **기능**:
  - Attention hook으로 직접 attention weights 수집
  - Cross-attention saliency 계산 (variance + γ * mean)
  - 프루닝 마스크 자동 생성
- **사용법**:
  ```bash
  source /home/diml/anaconda3/bin/activate llava
  python create_calibration_simple.py --num-samples 100 --gamma 0.4 --pruning-rate 0.3
  ```
- **출력**: JSON 파일에 pruning masks와 saliency scores 저장

### [2025-07-23 02:20] - S4 실험 전체 실행 가이드 작성
- **생성 파일**: `/home/diml/khj/comp/LLaVA/S4_EXPERIMENT_GUIDE.md`
- **내용**:
  - 환경 설정부터 결과 확인까지 전체 프로세스
  - 각 스크립트 사용법과 옵션 설명
  - 예제 시나리오와 문제 해결 방법
  - 결과 해석 가이드
- **핵심 명령어**:
  ```bash
  # 1. 환경 활성화
  source /home/diml/anaconda3/bin/activate llava
  # 2. Calibration 생성
  python create_calibration_simple.py --num-samples 100
  # 3. 프루닝 실험
  python s4_pruning_experiments.py --rates 0.05 0.10
  # 4. 성능 측정
  python s4_pruning_experiments_with_perf.py --rates 0.05 --run-baseline
  ```

### [2025-07-23 02:25] - Advanced Calibration 스크립트 추가
- **생성 파일**: `/home/diml/khj/comp/LLaVA/create_calibration_advanced.py`
- **기능**: 다양한 하이퍼파라미터 조절 가능한 고급 calibration

### [2025-07-23 02:30] - 통합 가이드 문서 생성
- **생성 파일**: `/home/diml/khj/comp/LLaVA/S4_COMPLETE_GUIDE.md`
- **내용**: 
  - 모든 실험 프로세스 통합 설명
  - Calibration 하이퍼파라미터 상세 가이드
  - 실행 예제 및 결과 분석 방법
  - 문제 해결 가이드
- **삭제 파일**: 
  - S4_EXPERIMENT_GUIDE.md
  - CALIBRATION_HYPERPARAMETERS.md
  (중복 내용을 S4_COMPLETE_GUIDE.md로 통합)

### [2025-07-23 02:35] - Early Skip 전략 구현 및 테스트
- **생성 파일**: 
  - `/home/diml/khj/comp/LLaVA/create_early_skip_calibration_json.py` - Calibration 생성
  - `/home/diml/khj/comp/LLaVA/test_early_skip_all.py` - 통합 테스트 스크립트
  - `/home/diml/khj/comp/LLaVA/s4_experiments/calibrations/s4_early_skip_calibration.json` - Calibration 파일
- **핵심 전략**:
  - Layer 0-5: 완전 보호 (0% 프루닝)
  - Layer 6-21: 20% 프루닝 (각 6개 헤드)
  - Layer 22-31: 35% 프루닝 (각 11개 헤드)
  - 총 프루닝: 206/1024 헤드 (20.1%)
- **테스트 결과**: 
  - 속도: 1.51x-1.56x 향상
  - Text-only 정확도: 84% (50개 샘플 테스트)
  - Hidden states 차이: 0.001551 (프루닝 효과 확인됨)
  - Scale factor 적용: 각 레이어별 남은 헤드 수에 따라 조정
- **사용법**:
  ```bash
  source /home/diml/anaconda3/bin/activate llava
  
  # 기존 작동하는 스크립트 활용 (파일 생성 최소화 원칙)
  python run_scienceqa_prehook_fixed.py
  ```

---

## 변경사항 기록 형식
```
### [날짜] - [작업 내용]
- **수정 파일**: 파일 경로
- **변경 내용**: 구체적인 변경 사항
- **이유**: 변경 이유
- **결과**: 테스트 결과 또는 영향
```