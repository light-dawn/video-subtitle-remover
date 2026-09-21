"""GPU-only subtitle detection worker for the isolated Paddle environment."""

import argparse
import json
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
from paddleocr import TextDetection


MESSAGE_PREFIX = "__VSR_GPU_OCR__"


def emit(kind, **payload):
    print(f"{MESSAGE_PREFIX}{json.dumps({'kind': kind, **payload}, ensure_ascii=False)}", flush=True)


def get_coordinates(polygons):
    coordinates = []
    for polygon in polygons:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = polygon
        coordinates.append((max(int(x1), int(x4)), min(int(x2), int(x3)),
                            max(int(y1), int(y2)), min(int(y3), int(y4))))
    return coordinates


def in_selected_sections(frame_no, sections):
    if not sections:
        return True
    return any(start <= frame_no <= end for start, end in sections)


def filter_regions(polygons, sub_areas):
    coordinates = get_coordinates(polygons)
    if not sub_areas:
        return coordinates, polygons
    boxes = []
    matched_polygons = []
    for polygon, (xmin, xmax, ymin, ymax) in zip(polygons, coordinates):
        for s_ymin, s_ymax, s_xmin, s_xmax in sub_areas:
            if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                boxes.append((xmin, xmax, ymin, ymax))
                matched_polygons.append(polygon)
                break
    return boxes, matched_polygons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sub-areas", required=True)
    parser.add_argument("--sample-step", required=True, type=int)
    parser.add_argument("--ab-sections", default="null")
    args = parser.parse_args()

    sub_areas = json.loads(args.sub_areas)
    ab_sections = json.loads(args.ab_sections)
    detector = TextDetection(
        model_name=args.model_name,
        model_dir=args.model_dir,
        device="gpu",
        enable_hpi=False,
    )
    capture = cv2.VideoCapture(args.video)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_results = {}
    current_frame_no = 0
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            current_frame_no += 1
            if (in_selected_sections(current_frame_no - 1, ab_sections)
                    and ((current_frame_no - 1) % args.sample_step == 0 or args.sample_step <= 1)):
                boxes = []
                polygons_for_frame = []
                for result in detector.predict(frame):
                    polygons = result.get("dt_polys")
                    if polygons is not None and len(polygons):
                        result_boxes, result_polygons = filter_regions(polygons.tolist(), sub_areas)
                        boxes.extend(result_boxes)
                        polygons_for_frame.extend(result_polygons)
                if boxes:
                    sampled_results[current_frame_no] = {
                        "boxes": boxes,
                        "polygons": polygons_for_frame,
                    }
            # Reducing IPC traffic is important for long videos while keeping
            # the UI responsive.
            if current_frame_no % 3 == 0 or current_frame_no == frame_count:
                emit("progress", current=current_frame_no, total=frame_count)
        emit("result", sampled_results=sampled_results)
    finally:
        capture.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("error", message=str(error))
        raise
