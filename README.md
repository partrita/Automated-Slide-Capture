# Video Slide Detection

Automatically detect and extract slides from conference videos using Python.

## Features

- **Automatic slide detection**: Uses computer vision to find slide boundaries in video frames
- **Change detection**: Only saves slides when content changes significantly  
- **Perspective correction**: Corrects distorted slides to rectangular format
- **Batch processing**: Process entire video files automatically
- **Legacy clipboard support**: Maintains compatibility with manual screenshot workflow

## Project Structure

```
├── input/                 # Place video files here
├── output/                # Extracted slides saved here
├── tests/                 # Test files
└── pyproject.toml         # Project configuration
```

## Installation

```bash
# Install dependencies
uv sync
```

## Usage

`extract_pegs_slides.py`는 PEGS 컨퍼런스 비디오에서 슬라이드를 자동으로 추출하는 독립 실행형 스크립트입니다.

```bash
# 기본 실행 (PEGS 컨퍼런스, 3초 간격)
uv run python extract_pegs_slides.py -i input/video.mp4

# 출력 디렉토리 지정
uv run python extract_pegs_slides.py -i input/video.mp4 -o my_slides

# 최대 슬라이드 수 제한
uv run python extract_pegs_slides.py -i input/video.mp4 --max-slides 10

# 변화 감지 임계값 조정 (더 민감하게)
uv run python extract_pegs_slides.py -i input/video.mp4 --threshold 0.05

# 프레임 처리 간격 조정 (5초마다)
uv run python extract_pegs_slides.py -i input/video.mp4 --interval 5.0
```

## 다른 컨퍼런스 타입

```bash
# KubeCon 비디오
uv run python extract_pegs_slides.py -i video.mp4 -c kubecon

# PyData 2015 비디오
uv run python extract_pegs_slides.py -i video.mp4 -c pydata2015w
```

## 매개변수 설명

- `--input, -i`: 입력 비디오 파일 경로 (필수)
- `--output, -o`: 출력 디렉토리 (기본값: output)
- `--conference, -c`: 컨퍼런스 타입 (pegs, kubecon, pydata2015w, none)
- `--interval, -t`: 프레임 처리 간격 (초, 기본값: 3.0)
- `--threshold, -th`: 변화 감지 임계값 (0.0-1.0, 기본값: 0.08)
- `--max-slides, -m`: 최대 추출할 슬라이드 수 (기본값: 제한 없음)

## 출력 파일

추출된 슬라이드는 다음 형식으로 저장됩니다:
- `pegs_slide_001_t0s.jpg`
- `pegs_slide_002_t24s.jpg`
- `pegs_slide_003_t75s.jpg`

파일명에는 슬라이드 번호와 타임스탬프가 포함됩니다.

## 실행 예시

```bash
# PEGS 컨퍼런스 비디오에서 최대 5개 슬라이드 추출
uv run python extract_pegs_slides.py \
  -i input/M-C6A_150---Kadina-Johnston.mp4 \
  -o extracted_slides \
  --max-slides 5 \
  --threshold 0.08
```

이 스크립트는 외부 의존성 없이 독립적으로 실행되며, OpenCV와 NumPy만 있으면 작동합니다.