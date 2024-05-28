import argparse
import glob
import multiprocessing as mp
import os
import pdfplumber
from PIL import Image
import json
import sys
from argparse import ArgumentParser
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from demo.predictor import VisualizationDemo

from dformer import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_DFormer_config,
)

from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

def pdf_to_images(pdf_path, output_folder):
    with pdfplumber.open(pdf_path) as pdf:
        os.makedirs(output_folder, exist_ok=True)
        for i, page in enumerate(pdf.pages):
            image = page.to_image(resolution=300)
            pil_image = image.original

            # PDF 파일 이름을 기반으로 이미지 파일 이름 생성
            pdf_filename = os.path.basename(pdf_path)
            pdf_name = os.path.splitext(pdf_filename)[0]
            image_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.png")

            # PIL 이미지 저장
            pil_image.save(image_path, 'PNG')

            print(f"Saved image {image_path}")

    return output_folder

def apply_gaussian_blur(image_path, blur_kernel_size=(3,3), sigma=0):
    """
    이미지 파일을 입력받아 Gaussian Blur를 적용하는 함수.
    
    Args:
    - image_path (str): 입력 이미지 파일 경로
    - blur_kernel_size (tuple): Gaussian Blur에 사용할 커널 크기
    - sigma (float): Gaussian Blur에 사용할 표준 편차
    
    Returns:
    - blurred_image (ndarray): Gaussian Blur가 적용된 이미지
    """
    # 이미지를 흑백으로 읽어옵니다.
    image = cv2.imread(image_path, 0)
    
    # 이진화 (Otsu's Binarization 사용)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Gaussian Blur를 적용합니다.
    blurred_image = cv2.GaussianBlur(binary_image, blur_kernel_size, sigma)
    return blurred_image

def visualize_masks_above_threshold(image_path, output_path, model_output, score_threshold=0.6, alpha=0.3):

    # 예측 결과에서 클래스, 마스크, 신뢰도 정보를 가져옵니다.
    pred_classes = model_output['instances'].__dict__['_fields']['pred_classes']
    pred_masks = model_output['instances'].__dict__['_fields']['pred_masks']
    pred_scores = model_output['instances'].__dict__['_fields']['scores']

    # 신뢰도 점수를 소수점 셋째 자리까지 반올림합니다.
    rounded_scores = np.round(pred_scores.cpu().numpy(), 3)

    # 신뢰도 점수가 임계값을 넘는 인덱스를 찾습니다.
    high_score_indices = np.where(rounded_scores > score_threshold)[0]
    high_scores = rounded_scores[high_score_indices]

    # 이미지 파일을 읽어옵니다.
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식을 사용하므로 RGB로 변환합니다.

    # 마스크를 반투명하게 덧씌웁니다.
    overlay = im.copy()
    for idx in high_score_indices:
        mask = pred_masks[idx].cpu().numpy().astype(bool)
        if pred_classes[idx] == 0:
            overlay[mask] = [255, 0, 0]  # 테이블은 빨간색으로 마스크 부분을 칠합니다.
        elif pred_classes[idx] == 1:
            overlay[mask] = [0, 0, 255]  # 피규어는 파란색으로 마스크 부분을 칠합니다.

    # 원본 이미지와 마스크를 합성합니다.
    cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

    # 이미지를 파일로 저장합니다.
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # 다시 BGR 형식으로 변환합니다.
    cv2.imwrite(output_path, im)

    print(f'Scores above {score_threshold}: {high_scores}')
    print(f'Image saved to {output_path}')

def process_images(input_folder, output_folder,predictor):
    for root, dirs, files in tqdm(os.walk(input_folder)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                im = cv2.imread(image_path)
                outputs = predictor(im)
                visualize_masks_above_threshold(image_path, output_path, outputs, alpha=0.5)
                print(f"Processed {image_path} and saved to {output_path}")

def process_images_gaussian(input_folder, output_folder, predictor, blur_kernel_size=(3,3), sigma=0):
    for root, dirs, files in tqdm(os.walk(input_folder)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 이미지를 흑백으로 읽습니다.
                im = cv2.imread(image_path, 0)

                # Gaussian Blur를 적용합니다.
                blurred_image = cv2.GaussianBlur(im, blur_kernel_size, sigma)

                # 2차원 이미지를 3차원으로 변환합니다.
                blurred_image_colored = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

                # 모델에 적용합니다.
                outputs = predictor(blurred_image_colored)
                visualize_masks_above_threshold(image_path, output_path, outputs, alpha=0.5)
                print(f"Processed {image_path} and saved to {output_path}")

def process_pdf(input_pdf, output_folder, predictor):
    # 임시 디렉터리를 생성하고 사용 후 삭제되도록 설정
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_folder = pdf_to_images(input_pdf, temp_dir)
        process_images(temp_output_folder, output_folder, predictor)

def process_val(input_json_path, output_folder,predictor):
    data_path = '/root/data/training_datasets/val'
    with open(input_json_path) as f:
        validation_data = json.load(f)
    for i in validation_data['images']:
        image_path = os.path.join(data_path,i['file_name'])
        output_path = os.path.join(output_folder, i['file_name'])
        im = cv2.imread(image_path)
        outputs = predictor(im)
        visualize_masks_above_threshold(image_path, output_path, outputs, alpha=0.5)
        print(f"Processed {image_path} and saved to {output_path}")

    
def main(input_type, input_path, output_folder):
    input_json_path = '/root/data/training_datasets/val.json'
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_DFormer_config(cfg)
    config_file = '/root/data/calm04/DFormer/output/config.yaml'
    cfg.merge_from_file(config_file)
    model_weight = '/root/data/calm04/DFormer/output/model_0019999.pth'
    cfg.MODEL.WEIGHTS = model_weight
    predictor = DefaultPredictor(cfg)

    if input_type == 'pdf':
        process_pdf(input_path, output_folder, predictor)
    elif input_type == 'images':
        process_images(input_path, output_folder, predictor)
    elif input_type == 'val':
        process_val(input_json_path,output_folder,predictor)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_type", type=str, choices=['pdf', 'images', 'val'],)
    args = parser.parse_args()
    input_image_folder = '/root/data/calm04/DFormer/dbpia_data/train' #/root/data/calm04/DFormer/mixed_adjust_datasets/val'
    output_image_folder = '/root/data/calm04/DFormer/test_result_val_0527_objectquery_30' #/root/data/calm04/DFormer/dbpia_arxiv_mix0517'

    main(args.input_type,input_image_folder,output_image_folder)# Updated~