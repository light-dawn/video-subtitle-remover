import json
import os
import subprocess
import sys
from collections import deque
from functools import cached_property
from pathlib import Path

# PaddleOCR's CPU inference backend otherwise inherits all logical cores on
# Windows.  Its worker process can create thousands of threads and appear to
# hang on the first frame.  These values are read when Paddle is imported.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "4")
# Avoid a host-availability probe on every first OCR use.  Models are supplied
# from ModelConfig's local model directory.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .common_tools import get_readable_path
from .ocr import get_coordinates
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

# Subtitle scanning is also invoked from the GUI worker thread.  Keep tqdm
# output out of the packaged application's invalid console handle.
TQDM_OUTPUT = open(os.devnull, "w", encoding="utf-8")
GPU_WORKER_MESSAGE_PREFIX = "__VSR_GPU_OCR__"

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(self, video_path, sub_areas=[]):
        self.video_path = video_path
        self.sub_areas = sub_areas
        # GPU OCR additionally provides the original text quadrilaterals. They
        # are kept separate from boxes because range grouping still uses boxes.
        self.frame_polygons = {}
        self._init_sample_step()

    def _init_sample_step(self):
        """根据视频帧率自适应设置采样间隔，保持每秒至少采样8帧"""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            self.SAMPLE_STEP = 4
        elif fps >= 30:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = 2

    @cached_property
    def text_detector(self):
        import paddle
        paddle.set_flags({"FLAGS_paddle_num_threads": 4})
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        model_config = ModelConfig()
        # The Windows RTX 50-series Paddle GPU wheel exposes CUDA through
        # Paddle itself.  Keep a CPU fallback so this source tree remains
        # runnable in a CPU-only environment as well.
        ocr_device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device=ocr_device,
            # PaddleX HPI is not supported in a native Windows installation.
            # CUDAExecutionProvider belongs to ONNX Runtime and is still used
            # elsewhere; it must not force PaddleOCR to require its separate
            # HPI plugin (which fails predictor creation on Windows).
            enable_hpi=False,
        )

    def detect_subtitle(self, img):
        temp_list = []
        results = self.text_detector.predict(img)
        sub_areas = self.sub_areas
        has_areas = sub_areas is not None and len(sub_areas) > 0
        for res in results:
            dt_polys = res['dt_polys']
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list = get_coordinates(dt_polys.tolist())
            if not coordinate_list:
                continue
            if not has_areas:
                temp_list.extend(coordinate_list)
            elif len(sub_areas) == 1:
                # 单区域快速路径（最常见场景）
                s_ymin, s_ymax, s_xmin, s_xmax = sub_areas[0]
                for xmin, xmax, ymin, ymax in coordinate_list:
                    if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                        temp_list.append((xmin, xmax, ymin, ymax))
            else:
                for xmin, xmax, ymin, ymax in coordinate_list:
                    for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
                        if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                            temp_list.append((xmin, xmax, ymin, ymax))
                            break
        return temp_list

    @staticmethod
    def _gpu_ocr_python():
        project_root = Path(__file__).resolve().parents[2]
        python_path = project_root / ".venv-ocr-gpu" / "Scripts" / "python.exe"
        return python_path if python_path.is_file() else None

    @staticmethod
    def _set_detection_progress(sub_remover, current_frame_no, frame_count):
        if not sub_remover:
            return
        sub_remover.progress_total = int(50 * float(current_frame_no) / float(frame_count))
        sub_remover.notify_progress_listeners()

    def _find_subtitle_frame_no_gpu(self, sub_remover, python_path):
        model_config = ModelConfig()
        project_root = Path(__file__).resolve().parents[2]
        command = [
            str(python_path), "-m", "backend.tools.gpu_ocr_worker",
            "--video", get_readable_path(self.video_path),
            "--model-name", model_config.DET_MODEL_NAME,
            "--model-dir", model_config.DET_MODEL_DIR,
            "--sub-areas", json.dumps(self.sub_areas),
            # GPU OCR is fast enough to inspect every frame.  Sampling every
            # third frame can miss a short subtitle or its fade-in/fade-out.
            "--sample-step", "1",
            "--ab-sections", json.dumps(sub_remover.ab_sections if sub_remover else None),
        ]
        if sub_remover:
            sub_remover.progress_base = 0
            sub_remover.progress_span = 50
            sub_remover.append_output("[GPU OCR] " + tr['Main']['ProcessingStartFindingSubtitles'])
        process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        diagnostics = deque(maxlen=20)
        sampled_results = None
        try:
            for line in process.stdout:
                line = line.rstrip()
                if not line.startswith(GPU_WORKER_MESSAGE_PREFIX):
                    diagnostics.append(line)
                    continue
                message = json.loads(line[len(GPU_WORKER_MESSAGE_PREFIX):])
                if message["kind"] == "progress":
                    self._set_detection_progress(sub_remover, message["current"], message["total"])
                elif message["kind"] == "result":
                    raw_results = message["sampled_results"]
                    self.frame_polygons = {
                        int(frame_no): value.get("polygons", [])
                        for frame_no, value in raw_results.items()
                    }
                    sampled_results = {
                        int(frame_no): value["boxes"]
                        for frame_no, value in raw_results.items()
                    }
                elif message["kind"] == "error":
                    diagnostics.append(message["message"])
        finally:
            if process.stdout:
                process.stdout.close()
        return_code = process.wait()
        if return_code or sampled_results is None:
            details = "\n".join(diagnostics)
            raise RuntimeError(f"GPU OCR worker failed (exit code {return_code}). {details}")
        return sampled_results

    def find_subtitle_frame_no(self, sub_remover=None):
        gpu_python = self._gpu_ocr_python()
        if config.hardwareAcceleration.value and gpu_python:
            sampled_results = self._find_subtitle_frame_no_gpu(sub_remover, gpu_python)
            return self._finalize_sampled_results(sampled_results, sub_remover)

        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=TQDM_OUTPUT, desc='Subtitle Finding')
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            # OCR scanning is the first half of a detection-based run.  This
            # must notify the GUI through its process queue; assigning the
            # attribute alone only changes the child process's memory.
            sub_remover.progress_base = 0
            sub_remover.progress_span = 50
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            if not is_frame_number_in_ab_sections(current_frame_no - 1, sub_remover.ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                self._set_detection_progress(sub_remover, current_frame_no, frame_count)
        video_cap.release()
        return self._finalize_sampled_results(sampled_results, sub_remover)

    def _finalize_sampled_results(self, sampled_results, sub_remover):
        # 阶段2：插值填充 — 两个采样帧之间都有字幕时，中间帧也标记为有字幕
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                fill_mask = sampled_results[f]
                for fill_f in range(f + 1, next_f):
                    subtitle_frame_no_box_dict[fill_f] = fill_mask
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.progress_total = 50
            sub_remover.notify_progress_listeners()
            # The inpainting tqdm begins at zero, so map it to the remaining
            # half rather than making the GUI progress jump backwards.
            sub_remover.progress_base = 50
            sub_remover.progress_span = 50
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(start - half, prev_end + 1)
                new_end = min(start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
