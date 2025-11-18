#!/usr/bin/env python3
"""
PEGS 컨퍼런스 비디오에서 슬라이드를 추출하는 독립 실행형 스크립트
"""

import cv2
import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional


# 컨퍼런스별 슬라이드 영역 좌표 (시계방향: 좌상단, 우상단, 우하단, 좌하단)
CONFERENCES = {
    "none": None,
    
    # KubeCon 영상 - https://www.youtube.com/watch?v=ySR_FVNX4bQ
    "kubecon": [(0, 0), (702, 48), (702, 418), (0, 447)],
    
    # PyData 2015 - https://www.youtube.com/watch?v=QpaapVaL8Fw  
    "pydata2015w": [(283, 88), (1013, 73), (1013, 500), (283, 485)],
    
    # PEGS 컨퍼런스 - 비디오 분석 결과를 바탕으로 실제 슬라이드 영역 설정
    # 분석 결과: 밝은 영역이 (0, 136) ~ (1783, 945)
    # 오른쪽 검은색 영역 제외를 위해 x=1400까지로 제한
    "pegs": [(0, 136), (1400, 136), (1400, 945), (0, 945)]
}


class ConferencePresetDetector:
    """사전 정의된 컨퍼런스 좌표를 사용하는 슬라이드 감지기."""
    
    def __init__(self, conference_type: str = "pegs"):
        """
        컨퍼런스 프리셋 감지기 초기화.
        
        Args:
            conference_type: 컨퍼런스 타입 ("pegs", "kubecon", "pydata2015w", "none")
        """
        self.conference_type = conference_type
        self.slide_corners = CONFERENCES.get(conference_type)
        
        if self.slide_corners is None and conference_type != "none":
            raise ValueError(f"지원하지 않는 컨퍼런스 타입: {conference_type}")
        
        # 이전 슬라이드 이미지 (변화 감지용)
        self.previous_slide_image: Optional[np.ndarray] = None
    
    def get_slide_corners(self) -> Optional[List[Tuple[int, int]]]:
        """사전 정의된 슬라이드 코너 좌표를 반환합니다."""
        return self.slide_corners
    
    def extract_slide_image(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임에서 사전 정의된 영역의 슬라이드를 추출합니다.
        
        Args:
            frame: 입력 비디오 프레임
            
        Returns:
            추출된 슬라이드 이미지 또는 None
        """
        if self.slide_corners is None:
            return None
        
        # 좌표에서 사각형 영역 계산
        x_coords = [corner[0] for corner in self.slide_corners]
        y_coords = [corner[1] for corner in self.slide_corners]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 프레임 경계 확인
        frame_height, frame_width = frame.shape[:2]
        
        if (x_min < 0 or y_min < 0 or 
            x_max >= frame_width or y_max >= frame_height):
            # 좌표가 프레임을 벗어나면 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame_width - 1, x_max)
            y_max = min(frame_height - 1, y_max)
        
        # 슬라이드 영역 추출
        slide_image = frame[y_min:y_max, x_min:x_max]
        
        return slide_image
    
    def has_content_changed(self, current_slide: np.ndarray, threshold: float = 0.15) -> bool:
        """
        현재 슬라이드가 이전 슬라이드와 다른지 확인합니다.
        
        Args:
            current_slide: 현재 슬라이드 이미지
            threshold: 변화 감지 임계값 (0.0-1.0)
            
        Returns:
            내용이 변경되었으면 True, 아니면 False
        """
        if self.previous_slide_image is None:
            self.previous_slide_image = current_slide.copy()
            return True
        
        # 크기가 다르면 리사이즈
        if current_slide.shape != self.previous_slide_image.shape:
            h, w = self.previous_slide_image.shape[:2]
            current_slide_resized = cv2.resize(current_slide, (w, h))
        else:
            current_slide_resized = current_slide
        
        # 차이 계산
        diff = cv2.absdiff(current_slide_resized, self.previous_slide_image)
        diff_score = diff.mean() / 255.0
        
        # 변화가 임계값을 넘으면 업데이트
        if diff_score > threshold:
            self.previous_slide_image = current_slide.copy()
            return True
        
        return False
    
    def get_detection_confidence(self, frame: np.ndarray) -> float:
        """
        사전 정의된 좌표를 사용하므로 항상 높은 신뢰도를 반환합니다.
        
        Args:
            frame: 입력 프레임
            
        Returns:
            신뢰도 점수 (0.0-1.0)
        """
        if self.slide_corners is None:
            return 0.0
        
        # 슬라이드 영역 추출
        slide_image = self.extract_slide_image(frame)
        
        if slide_image is None or slide_image.size == 0:
            return 0.0
        
        # 슬라이드 영역의 밝기 기반 신뢰도
        gray_slide = cv2.cvtColor(slide_image, cv2.COLOR_BGR2GRAY)
        brightness = gray_slide.mean()
        
        # 너무 어두우면 (검은 화면) 신뢰도 낮춤
        if brightness < 20:
            return 0.3
        
        # 일반적으로 높은 신뢰도 (사전 정의된 좌표이므로)
        return 0.95


def extract_pegs_slides(
    video_path: str, 
    output_dir: str = "output", 
    conference_type: str = "pegs",
    frame_interval_seconds: float = 3.0,
    change_threshold: float = 0.08,
    max_slides: int = None
):
    """
    PEGS 컨퍼런스 비디오에서 슬라이드를 추출합니다.
    
    Args:
        video_path: 입력 비디오 파일 경로
        output_dir: 출력 디렉토리
        conference_type: 컨퍼런스 타입 ("pegs", "kubecon", "pydata2015w")
        frame_interval_seconds: 프레임 처리 간격 (초)
        change_threshold: 변화 감지 임계값 (0.0-1.0, 낮을수록 민감)
        max_slides: 최대 추출할 슬라이드 수 (None이면 제한 없음)
    
    Returns:
        추출된 슬라이드 수
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 컨퍼런스 프리셋 감지기 초기화
    try:
        detector = ConferencePresetDetector(conference_type)
    except ValueError as e:
        print(f"오류: {e}")
        return 0
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return 0
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"=== PEGS 슬라이드 추출 시작 ===")
    print(f"비디오 파일: {os.path.basename(video_path)}")
    print(f"해상도: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"총 프레임: {total_frames}")
    print(f"길이: {duration:.1f}초")
    print(f"컨퍼런스 타입: {conference_type.upper()}")
    print(f"프레임 간격: {frame_interval_seconds}초")
    print(f"변화 임계값: {change_threshold}")
    if max_slides:
        print(f"최대 슬라이드: {max_slides}개")
    print()
    
    # 슬라이드 좌표 확인
    slide_corners = detector.get_slide_corners()
    if slide_corners:
        x_coords = [corner[0] for corner in slide_corners]
        y_coords = [corner[1] for corner in slide_corners]
        slide_width = max(x_coords) - min(x_coords)
        slide_height = max(y_coords) - min(y_coords)
        print(f"슬라이드 영역: {slide_width}x{slide_height}")
        print(f"슬라이드 좌표: {slide_corners}")
    else:
        print("슬라이드 좌표가 설정되지 않았습니다.")
        return 0
    
    print()
    
    frame_count = 0
    slide_count = 0
    processed_count = 0
    
    # 프레임 처리 간격 계산
    frame_interval = int(fps * frame_interval_seconds) if fps > 0 else 90
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격으로만 처리
            if frame_count % frame_interval != 0:
                frame_count += 1
                continue
            
            processed_count += 1
            timestamp = frame_count / fps if fps > 0 else frame_count
            
            # 슬라이드 추출
            slide_image = detector.extract_slide_image(frame)
            confidence = detector.get_detection_confidence(frame)
            
            if slide_image is not None and confidence > 0.5:
                # 내용 변화 확인
                content_changed = detector.has_content_changed(slide_image, threshold=change_threshold)
                
                if content_changed:
                    slide_count += 1
                    
                    # 슬라이드 저장
                    slide_filename = f"{conference_type}_slide_{slide_count:03d}_t{timestamp:.0f}s.jpg"
                    slide_path = os.path.join(output_dir, slide_filename)
                    cv2.imwrite(slide_path, slide_image)
                    
                    print(f"슬라이드 {slide_count:3d}: {timestamp:6.0f}초 (프레임 {frame_count:6d}) - {slide_filename}")
                    
                    # 최대 슬라이드 수 확인
                    if max_slides and slide_count >= max_slides:
                        print(f"\n최대 슬라이드 수({max_slides})에 도달했습니다.")
                        break
                else:
                    if processed_count % 20 == 0:  # 20번마다 중복 메시지 출력
                        print(f"진행중... {timestamp:6.0f}초 (중복 슬라이드)")
            else:
                if processed_count % 20 == 0:  # 20번마다 진행 상황 출력
                    print(f"진행중... {timestamp:6.0f}초 (슬라이드 없음)")
            
            frame_count += 1
            
            # 진행률 표시
            if processed_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"\n진행률: {progress:.1f}% - 추출된 슬라이드: {slide_count}개\n")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    
    finally:
        cap.release()
    
    print(f"\n=== 추출 완료 ===")
    print(f"처리된 프레임: {processed_count}")
    print(f"추출된 슬라이드: {slide_count}개")
    print(f"출력 디렉토리: {output_dir}")
    
    return slide_count


def main():
    """메인 함수 - 명령줄에서 실행할 때 사용"""
    parser = argparse.ArgumentParser(description="PEGS 컨퍼런스 비디오에서 슬라이드 추출")
    parser.add_argument("--input", "-i", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output", "-o", default="output", help="출력 디렉토리 (기본값: output)")
    parser.add_argument("--conference", "-c", default="pegs", 
                       choices=list(CONFERENCES.keys()),
                       help="컨퍼런스 타입 (기본값: pegs)")
    parser.add_argument("--interval", "-t", type=float, default=3.0,
                       help="프레임 처리 간격 (초, 기본값: 3.0)")
    parser.add_argument("--threshold", "-th", type=float, default=0.08,
                       help="변화 감지 임계값 (0.0-1.0, 기본값: 0.08)")
    parser.add_argument("--max-slides", "-m", type=int, default=None,
                       help="최대 추출할 슬라이드 수 (기본값: 제한 없음)")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return 1
    
    # 슬라이드 추출 실행
    try:
        extracted_count = extract_pegs_slides(
            video_path=args.input,
            output_dir=args.output,
            conference_type=args.conference,
            frame_interval_seconds=args.interval,
            change_threshold=args.threshold,
            max_slides=args.max_slides
        )
        
        if extracted_count > 0:
            print(f"🎉 성공적으로 {extracted_count}개의 슬라이드를 추출했습니다!")
            return 0
        else:
            print("❌ 슬라이드를 추출하지 못했습니다.")
            return 1
            
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())