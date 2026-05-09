import torch
import numpy as np
import pyiqa
import time
import cv2
from typing import Tuple, Optional, Dict, Union
import os
import warnings
warnings.filterwarnings('ignore')

# 设置缓存目录
CACHE_DIR = os.environ.get('BLUR_DETECTOR_CACHE', '/tmp/cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'huggingface')


class BlurDetector:
    """
    使用ARNIQA-CSIQ指标进行模糊检测的类
    基于大规模数据集分析得出的阈值进行判断
    """
    
    # 基于分析结果的阈值配置
    THRESHOLDS = {
        'severe': 0.5919,    # 严重模糊
        'moderate': 0.7167,  # 中度模糊
        'mild': 0.7403,      # 轻度模糊（最优阈值）
        'conservative': 0.7696,  # 保守阈值（减少误判）
        'aggressive': 0.7443,     # 激进阈值（捕获更多模糊）
        'super': 0.81
    }
    
    def __init__(self,
                 cfg, 
                 device: str = "cuda:0",
                 metric_name: str = 'arniqa-csiq',
                 cache_scores: bool = True,
                 sensitivity: str = 'super'):
        """
        初始化模糊检测器
        
        Args:
            device: 计算设备
            metric_name: 使用的IQA指标名称
            cache_scores: 是否缓存分数以提高性能
            sensitivity: 敏感度设置 ('conservative', 'balanced', 'aggressive')
        """
        self.config = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metric_name = metric_name
        self.sensitivity = sensitivity
        self.cache_scores = cache_scores
        self.score_cache = {} if cache_scores else None
        
        # 初始化ARNIQA-CSIQ指标
        print(f"Initializing {metric_name} on {self.device}...")
        self.metric = pyiqa.create_metric(metric_name, device=self.device)
        self.metric.eval()
        print(f"{metric_name} initialized successfully")
        
        # 设置主阈值
        self._set_threshold(sensitivity)
        
    def _set_threshold(self, sensitivity: str):
        """根据敏感度设置阈值"""
        if sensitivity == 'conservative':
            self.primary_threshold = self.THRESHOLDS['conservative']
        elif sensitivity == 'aggressive':
            self.primary_threshold = self.THRESHOLDS['aggressive']
        elif sensitivity == 'super':
            self.primary_threshold = self.config["blur_threshold"]
        else:  # balanced
            self.primary_threshold = self.THRESHOLDS['mild']
    
    def detect_blur(self, 
                    image_tensor: torch.Tensor, 
                    return_score: bool = False,
                    return_level: bool = False) -> Union[bool, Tuple]:
        """
        检测图像是否模糊
        
        Args:
            image_tensor: 输入图像tensor，支持多种格式
                - [H, W, 3] or [3, H, W] or [1, 3, H, W]
            return_score: 是否返回质量分数
            return_level: 是否返回模糊级别
            
        Returns:
            根据参数返回不同结果：
            - 默认: bool (是否模糊)
            - return_score=True: (is_blurry, score)
            - return_level=True: (is_blurry, score, blur_level)
        """
        # 预处理图像tensor
        processed_tensor = self._preprocess_image(image_tensor)
        
        # 计算分数（使用缓存）
        score = self._compute_score(processed_tensor)
        
        print("score is",score)
        # 判断是否模糊
        is_blurry = score < self.primary_threshold 
        
        # 获取模糊级别
        blur_level = self._get_blur_level(score) if return_level else None
        
        # 根据参数返回结果
        if return_level:
            return is_blurry, score, blur_level
        elif return_score:
            return is_blurry, score
        else:
            return is_blurry
    
    def _preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        预处理图像tensor到统一格式 [1, 3, H, W]
        """
        # 克隆以避免修改原始tensor
        tensor = image_tensor.clone()
        
        # 确保在正确的设备上
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        
        # 处理不同的输入格式
        if len(tensor.shape) == 3:
            if tensor.shape[-1] == 3:  # [H, W, 3]
                tensor = tensor.permute(2, 0, 1)  # -> [3, H, W]
            # tensor现在是 [3, H, W]
            tensor = tensor.unsqueeze(0)  # -> [1, 3, H, W]
        elif len(tensor.shape) == 2:  # 灰度图
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        
        # 确保值在[0, 1]范围
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        return tensor
    
    def _compute_score(self, image_tensor: torch.Tensor) -> float:
        """
        计算图像质量分数
        
        Args:
            image_tensor: 预处理后的图像tensor [1, 3, H, W]
            
        Returns:
            质量分数 (越高越清晰)
        """
        # 生成缓存键
        if self.cache_scores:
            cache_key = self._generate_cache_key(image_tensor)
            if cache_key in self.score_cache:
                return self.score_cache[cache_key]
        
        # 计算分数
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):  # 确保全精度
                score = self.metric(image_tensor).cpu().item()
        
        # 缓存结果
        if self.cache_scores:
            self.score_cache[cache_key] = score
        
        return score
    
    def _generate_cache_key(self, tensor: torch.Tensor) -> int:
        """生成tensor的缓存键"""
        # 使用tensor的部分数据生成哈希
        sample = tensor[0, :, ::10, ::10].cpu().numpy()
        return hash(sample.tobytes())
    
    def _get_blur_level(self, score: float) -> str:
        """
        根据分数获取模糊级别
        
        Returns:
            'severe', 'moderate', 'mild', or 'sharp'
        """
        if score < self.THRESHOLDS['severe']:
            return 'severe'
        elif score < self.THRESHOLDS['moderate']:
            return 'moderate'
        elif score < self.THRESHOLDS['mild']:
            return 'mild'
        else:
            return 'sharp'
    
    def get_blur_degree(self, score: float) -> float:
        """
        计算模糊程度（0-1范围，0=清晰，1=严重模糊）
        """
        if score >= self.THRESHOLDS['mild']:
            return 0.0  # 清晰
        elif score <= self.THRESHOLDS['severe']:
            return 1.0  # 严重模糊
        else:
            # 线性插值
            return 1.0 - (score - self.THRESHOLDS['severe']) / (self.THRESHOLDS['mild'] - self.THRESHOLDS['severe'])
    
    def batch_detect(self, image_tensors: list, batch_size: int = 4) -> list:
        """
        批量检测多张图像
        
        Args:
            image_tensors: 图像tensor列表
            batch_size: 批处理大小
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i+batch_size]
            batch_tensor = torch.stack([self._preprocess_image(img).squeeze(0) for img in batch])
            
            with torch.no_grad():
                scores = self.metric(batch_tensor).cpu().numpy()
            
            for score in scores:
                is_blurry = score < self.primary_threshold
                blur_level = self._get_blur_level(score)
                results.append({
                    'is_blurry': is_blurry,
                    'score': float(score),
                    'blur_level': blur_level,
                    'blur_degree': self.get_blur_degree(score)
                })
        
        return results
    
    def benchmark_speed(self, image_tensor: torch.Tensor, n_iterations: int = 100) -> Dict:
        """
        测试检测速度
        
        Args:
            image_tensor: 测试图像
            n_iterations: 迭代次数
            
        Returns:
            包含时间统计的字典
        """
        processed_tensor = self._preprocess_image(image_tensor)
        
        # 预热
        for _ in range(10):
            _ = self._compute_score(processed_tensor)
        
        # 清空缓存
        if self.cache_scores:
            self.score_cache.clear()
        
        # 计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            _ = self._compute_score(processed_tensor)
            if self.cache_scores:
                self.score_cache.clear()  # 避免缓存影响
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / n_iterations * 1000  # ms
        
        return {
            'avg_time_ms': avg_time,
            'fps': 1000 / avg_time,
            'total_time_s': end_time - start_time,
            'n_iterations': n_iterations
        }
    
    def update_sensitivity(self, sensitivity: str):
        """动态更新敏感度设置"""
        self.sensitivity = sensitivity
        self._set_threshold(sensitivity)
        print(f"Sensitivity updated to '{sensitivity}', threshold: {self.primary_threshold:.4f}")


class ConservativeBlurDetector(BlurDetector):
    """
    保守的模糊检测器，减少误判
    """
    def __init__(self, device: str = "cuda:0", **kwargs):
        super().__init__(device=device, sensitivity='conservative', **kwargs)


class AggressiveBlurDetector(BlurDetector):
    """
    激进的模糊检测器，捕获更多潜在模糊
    """
    def __init__(self, device: str = "cuda:0", **kwargs):
        super().__init__(device=device, sensitivity='aggressive', **kwargs)


class FastBlurDetector:
    """
    快速模糊检测器，使用传统方法作为预筛选
    结合ARNIQA进行精确检测
    """
    
    def __init__(self, device: str = "cuda:0", use_arniqa_for_uncertain: bool = True):
        self.device = device
        self.use_arniqa_for_uncertain = use_arniqa_for_uncertain
        
        if use_arniqa_for_uncertain:
            self.arniqa_detector = BlurDetector(device=device)
    
    def detect_blur_fast(self, image: Union[np.ndarray, torch.Tensor], 
                        fft_threshold: float = 7.0) -> Tuple[bool, float]:
        """
        快速模糊检测（使用FFT）
        """
        # 转换为numpy
        if torch.is_tensor(image):
            img_array = image.cpu().numpy()
        else:
            img_array = image.copy()
        
        # 处理维度
        if len(img_array.shape) == 4:
            img_array = img_array[0]
        
        # FFT检测
        score = self._compute_fft_score(img_array)
        
        # 明确的情况直接返回
        if score < fft_threshold * 0.7:  # 明显模糊
            return True, score
        elif score > fft_threshold * 1.3:  # 明显清晰
            return False, score
        
        # 不确定的情况使用ARNIQA
        if self.use_arniqa_for_uncertain:
            is_blurry, arniqa_score = self.arniqa_detector.detect_blur(image, return_score=True)
            return is_blurry, arniqa_score
        else:
            return score < fft_threshold, score
    
    def _compute_fft_score(self, image: np.ndarray) -> float:
        """计算FFT分数"""
        # 转换为灰度图
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # [C, H, W]
                image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image[:, :, 0]
        else:
            grayscale = image
        
        # 确保uint8
        if grayscale.max() <= 1.0:
            grayscale = (grayscale * 255).astype(np.uint8)
        
        # FFT
        f = np.fft.fft2(grayscale)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # 高频能量
        rows, cols = grayscale.shape
        crow, ccol = rows // 2, cols // 2
        low_freq_radius = min(rows, cols) // 16
        
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), low_freq_radius, 0, -1)
        
        high_freq_energy = magnitude_spectrum * mask
        high_freq_sum = np.sum(high_freq_energy)
        normalized_energy = high_freq_sum / (rows * cols)
        
        return normalized_energy