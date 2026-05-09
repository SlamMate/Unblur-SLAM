# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from thirdparty.EVSSM.models.EVSSM import EVSSM
from torchvision.transforms import functional as F
import torch.nn.functional as nnF
from thirdparty.glorie_slam.motion_filter import MotionFilter
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
from thirdparty.monogs.utils.slam_utils import variance_of_laplacian
from src.utils.common import save_tensor
import cv2
class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.only_tracking = slam.only_tracking
        self.output = slam.save_dir

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        self.every_kf = self.cfg['mapping']['every_keyframe']
        # frontend process
        self.frontend = slam.frontend
        self.online_ba = slam.online_ba
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']

        self.printer:Printer = slam.printer


        # Initialize EVSSM model if fake_sharp is enabled
        self.fake_sharp = self.cfg.get("fake_sharp", False)
        self.sharp_judge = self.cfg.get("sharp_judge", False)
    
    def run(self, stream:BaseDataset):
        print("TRACKING: Starting tracking process")
        '''
        Trigger the tracking process.
        1. check whether there is enough motion between the current frame and last keyframe by motion_filter
        2. use frontend to do local bundle adjustment, to estimate camera pose and depth image, 
            also delete the current keyframe if it is too close to the previous keyframe after local BA.
        3. run online global BA periodically by backend
        4. send the estimated pose and depth to mapper, 
            and wait until the mapper finish its current mapping optimization.
        '''
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0
        init = False
        number_of_kf = 0
        intrinsic = stream.get_intrinsic()
        # for (timestamp, image, _, _) in tqdm(stream):
        
        dataset = self.cfg["dataset"]
        fake_sharp_gt = self.cfg["fake_sharp_gt"]
        sharp_judge = self.cfg["sharp_judge"]
        fake_sharp = self.cfg["fake_sharp"]

        sequence = self.cfg["scene"]
        blur_degree = 0.0

        length = len(stream)

        for i in range(length):
            print(f"TRACKING: Processing frame {i}/{len(stream)}")
            # 将前端输入的图片变为中间曝光时刻的锐利图
            # 只有replica可以直接读gt的锐利图像
            # 初始帧是锐利帧
            if dataset == "replica_blurry" and fake_sharp_gt and i!=0:
                timestamp, image, _, _, _, sharp_image_gt = stream[i]
                # sharp_dir = f"./output/sharp/{dataset}/{sequence}"
                # save_tensor(image, timestamp, sharp_dir, True)
                image = sharp_image_gt
            else:
                timestamp, image, _, _, _ = stream[i]

            # 将该sharp image保存到本地用时间戳，然后作为数据集属性，将路径，然后再读取
            is_blurry = True
            is_mapping = False


            with torch.no_grad():
                # Check there is enough motion
                # 初始化的阶段不要动
                if fake_sharp and i!=0:
                    if self.cfg["exam_blur_score"]:
                        image, blur_degree, is_kf, is_blurry, check_score  = self.motion_filter.track(timestamp, image, intrinsic, fake_sharp, sharp_judge, stream, init)
                        print("check_score is", check_score)
                    else:
                        image, blur_degree, is_kf, is_blurry = self.motion_filter.track(timestamp, image, intrinsic, fake_sharp, sharp_judge, stream, init)
                    if is_kf:
                        if not self.only_tracking and image is not None and is_blurry:
                            sharp_dir = f"./output/sharp/{dataset}/{sequence}"
                            # save_tensor(image, timestamp, sharp_dir)
                            save_tensor(image, timestamp, sharp_dir, True)
                        # 跟踪only模式的可视化选项
                        if self.only_tracking and image is not None and is_blurry:
                            sharp_dir = f"./output/sharp/{dataset}/{sequence}"
                            # save_tensor(image, timestamp, sharp_dir, True)
                        # 模糊了然后如果去模糊失败的回退机制
                        if blur_degree is None and is_blurry and self.cfg["backend_compensation"] and not self.only_tracking and self.cfg["exam_blur_score"]:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                        "timestamp": timestamp, "end": False, "blur_degree": 1.0, "check_score":check_score})
                            self.pipe.recv()
                            is_mapping = True
                        elif blur_degree is None and is_blurry and self.cfg["backend_compensation"] and not self.only_tracking:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                        "timestamp": timestamp, "end": False, "blur_degree": 1.0})
                            self.pipe.recv()
                            is_mapping = True
                        # 没有模糊的判断机制
                        """
                        elif not is_blurry and not self.only_tracking and self.cfg["exam_blur_score"]:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                        "timestamp": timestamp, "end": False, "blur_degree": 0.0, "check_score":check_score})
                            self.pipe.recv()
                            is_mapping = True
                        elif not is_blurry and not self.only_tracking:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                        "timestamp": timestamp, "end": False, "blur_degree": 0.0})
                            self.pipe.recv()
                            is_mapping = True
                        """
                else:
                    self.motion_filter.track(timestamp, image, intrinsic)
                # Local bundle adjustment
                self.frontend()

            curr_kf_idx = self.video.counter.value - 1
            
            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                number_of_kf += 1
                if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                    # Run online global BA every {self.ba_freq} keyframes
                    self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}", FontColor.TRACKER)
                    self.online_ba.dense_ba(2)
                    prev_ba_idx = curr_kf_idx
                if (not self.only_tracking) and (number_of_kf % self.every_kf == 0):

                    # Inform the mapper that the estimation of current pose and depth is finished
                    if not init and self.cfg['clear_init']:
                        if fake_sharp and is_blurry and self.cfg["exam_blur_score"]:
                            self.pipe.send({"is_keyframe": True, "video_idx": 0,
                                            "timestamp": stream[0][0], "end": False, "blur_degree":blur_degree, "check_score":check_score})
                        elif fake_sharp and is_blurry:
                            self.pipe.send({"is_keyframe": True, "video_idx": 0,
                                            "timestamp": stream[0][0], "end": False, "blur_degree":blur_degree})
                        elif fake_sharp and not is_blurry and self.cfg["exam_blur_score"]:
                            self.pipe.send({"is_keyframe": True, "video_idx": 0,
                                            "timestamp": stream[0][0], "end": False, "blur_degree":0.0, "check_score":check_score})
                        elif fake_sharp and not is_blurry:
                            self.pipe.send({"is_keyframe": True, "video_idx": 0,
                                            "timestamp": stream[0][0], "end": False, "blur_degree":0.0})
                        else:
                            self.pipe.send({"is_keyframe": True, "video_idx": 0,
                                            "timestamp": stream[0][0], "end": False})
                        self.pipe.recv()
                        init = True
                    elif not init and self.cfg['warmup_mapper']:
                        print("init_warm_up")
                        print(curr_kf_idx)
                        # 循环发送所有累积的预热关键帧 (0 -> curr_kf_idx)
                        for kf_id in range(curr_kf_idx+1):
                            # 从 video 对象中获取历史关键帧的时间戳
                            # self.video.timestamp 是一个 tensor，存储了 keyframe index -> timestamp 的映射
                            kf_timestamp = int(self.video.timestamp[kf_id].item())
                            
                            print("init_timestamp")
                            print(kf_timestamp)
                            # 构造基础 payload
                            payload = {
                                "is_keyframe": True,
                                "video_idx": kf_id,
                                "timestamp": kf_timestamp,
                                "end": False
                            }
                            # 判断是否为锐利帧，如果为锐利帧则blur_degree=0
                            # 锐利帧的判断往往通过是否有神经网络输出的去模糊图像为准
                            dataset = self.cfg["dataset"]
                            sequence = self.cfg["scene"]
                            sharp_dir = f"./output/sharp/{dataset}/{sequence}"
                            sharp_file = Path(sharp_dir) / f"{kf_timestamp}.pt"
                            if os.path.exists(sharp_file):
                                payload["check_score"] = 0.82
                                payload["blur_degree"] = 0.0
                            else:
                                payload["check_score"] = 0.6
                                payload["blur_degree"] = 0.8
                            self.pipe.send(payload)
                            self.pipe.recv()
                        init = True
                    else:
                        print("follow")
                        print(curr_kf_idx)
                        # deblur没有失败的情况下，则直接传check score出去了
                        if fake_sharp and is_blurry and self.cfg["exam_blur_score"] and not is_mapping:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                        "timestamp": timestamp, "end": False, "blur_degree":blur_degree, "check_score":check_score})
                            self.pipe.recv()
                        elif fake_sharp and is_blurry and not is_mapping:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                        "timestamp": timestamp, "end": False, "blur_degree":blur_degree})
                            self.pipe.recv()
                        elif fake_sharp and not is_blurry and self.cfg["exam_blur_score"] and not is_mapping:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                "timestamp": timestamp, "end": False, "blur_degree":0.0, "check_score":check_score})
                            self.pipe.recv()
                        elif fake_sharp and not is_blurry and not is_mapping:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                                "timestamp": timestamp, "end": False, "blur_degree":0.0})
                            self.pipe.recv()
                        elif not is_mapping:
                            self.pipe.send({"is_keyframe": True, "video_idx": curr_kf_idx,
                                            "timestamp": timestamp, "end": False})
                            self.pipe.recv()

            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()

        if not self.only_tracking:
            if fake_sharp and is_blurry and self.cfg["exam_blur_score"]:
                self.pipe.send({"is_keyframe":True, "video_idx":None, "timestamp":None, "end":True, "blur_degree":blur_degree, "check_score":check_score})
            elif fake_sharp and is_blurry:
                self.pipe.send({"is_keyframe":True, "video_idx":None, "timestamp":None, "end":True, "blur_degree":blur_degree})
            elif fake_sharp and not is_blurry and self.cfg["exam_blur_score"]:
                self.pipe.send({"is_keyframe": True, "video_idx": None,
                                "timestamp": None, "end": True, "blur_degree":0.0, "check_score": check_score})
            elif fake_sharp and not is_blurry:
                self.pipe.send({"is_keyframe": True, "video_idx": None,
                                "timestamp": None, "end": True, "blur_degree":0.0})
            else:
                self.pipe.send({"is_keyframe":True, "video_idx":None, "timestamp":None, "end":True})