import os
import glob
import cv2
import numpy as np
import re
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from ultralytics import YOLO


class CellTrack:
    def __init__(self, track_id, mask, centroid, fourier_desc, box):
        self.track_id = track_id
        self.age = 1
        self.time_since_update = 0
        # 轨迹历史，用于绘图
        self.history = deque(maxlen=50)
        self.history.append(centroid)

        self.mask = mask
        self.centroid = centroid
        self.fourier = fourier_desc
        self.box = box  # [x1, y1, x2, y2]


        self.velocity = (0, 0)

    def predict_position(self):

        pred_x = self.centroid[0] + self.velocity[0]
        pred_y = self.centroid[1] + self.velocity[1]
        return (pred_x, pred_y)

    def update(self, new_mask, new_centroid, new_fourier, new_box):

        dx = new_centroid[0] - self.centroid[0]
        dy = new_centroid[1] - self.centroid[1]
        self.velocity = (0.5 * self.velocity[0] + 0.5 * dx,
                         0.5 * self.velocity[1] + 0.5 * dy)

        self.mask = new_mask
        self.centroid = new_centroid
        self.fourier = new_fourier
        self.box = new_box
        self.history.append(new_centroid)
        self.age += 1
        self.time_since_update = 0


class GSCTracker:
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, max_lost=5):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_lost = max_lost

        self.tracks = []
        self.frame_count = 0
        self.next_id = 1

    def _get_centroid(self, mask):

        M = cv2.moments(mask)
        if M["m00"] == 0:
            return (0, 0)  # 异常保护
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def _get_fourier_descriptor(self, mask, num_coeffs=10):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(num_coeffs)

        contour = contours[0][:, 0, :]
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]

        fourier = np.fft.fft(contour_complex)


        if len(fourier) < num_coeffs:
            fourier = np.pad(fourier, (0, num_coeffs - len(fourier)), 'constant')
        else:
            fourier = fourier[:num_coeffs]


        fourier = np.abs(fourier)
        if fourier[0] != 0:
            fourier = fourier / fourier[0]
        return fourier[1:]

    def _calculate_hausdorff(self, mask_a, mask_b):

        pts_a = np.column_stack(np.where(mask_a > 0))
        pts_b = np.column_stack(np.where(mask_b > 0))

        if len(pts_a) == 0 or len(pts_b) == 0:
            return 100.0

        d1 = directed_hausdorff(pts_a, pts_b)[0]
        d2 = directed_hausdorff(pts_b, pts_a)[0]
        return max(d1, d2)

    def update(self, detections):
        self.frame_count += 1

        current_measurements = []
        for det in detections:
            mask = det['mask'].astype(np.uint8)
            current_measurements.append({
                'mask': mask,
                'centroid': self._get_centroid(mask),
                'fourier': self._get_fourier_descriptor(mask),
                'box': det['box']
            })


        if len(self.tracks) == 0:
            for meas in current_measurements:
                self.tracks.append(
                    CellTrack(self.next_id, meas['mask'], meas['centroid'], meas['fourier'], meas['box']))
                self.next_id += 1
            return


        N = len(self.tracks)
        M = len(current_measurements)
        cost_matrix = np.zeros((N, M))

        for i, track in enumerate(self.tracks):
            pred_pos = track.predict_position()

            for j, meas in enumerate(current_measurements):

                dist_check = np.linalg.norm(np.array(pred_pos) - np.array(meas['centroid']))

                if dist_check > 100:
                    cost = 10000.0
                else:

                    motion_dist = dist_check


                    hausdorff_dist = self._calculate_hausdorff(track.mask, meas['mask'])

                    f_len = min(len(track.fourier), len(meas['fourier']))
                    shape_dist = np.sum(np.abs(track.fourier[:f_len] - meas['fourier'][:f_len]))


                    cost = (self.alpha * motion_dist) + (self.beta * hausdorff_dist) + (self.gamma * shape_dist * 100)

                cost_matrix[i, j] = cost


        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        MATCH_THRESHOLD = 200.0

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < MATCH_THRESHOLD:
                meas = current_measurements[c]
                self.tracks[r].update(meas['mask'], meas['centroid'], meas['fourier'], meas['box'])
                assigned_tracks.add(r)
                assigned_dets.add(c)


        for i in range(len(self.tracks) - 1, -1, -1):
            if i not in assigned_tracks:
                self.tracks[i].time_since_update += 1
                if self.tracks[i].time_since_update > self.max_lost:
                    self.tracks.pop(i)


        for j in range(M):
            if j not in assigned_dets:
                meas = current_measurements[j]
                self.tracks.append(
                    CellTrack(self.next_id, meas['mask'], meas['centroid'], meas['fourier'], meas['box']))
                self.next_id += 1


def natural_sort_key(s):

    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]



def run_tracking_folder(model_path, data_folder):
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"正在读取数据: {data_folder}")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(data_folder, ext)))


    image_files.sort(key=natural_sort_key)

    if not image_files:
        print("错误：文件夹中没有找到图片！")
        return

    print(f"找到 {len(image_files)} 张图片，开始追踪...")

    tracker = GSCTracker()


    first_frame = cv2.imread(image_files[0])
    h, w, _ = first_frame.shape
    save_path = os.path.join(data_folder, 'tracking_result.mp4')
    out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

    for idx, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue


        results = model(frame, verbose=False, conf=0.25, iou=0.7)
        result = results[0]

        detections = []
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()  # (N, H_net, W_net)
            boxes_data = result.boxes.xyxy.cpu().numpy()  # (N, 4)

            for i, mask_small in enumerate(masks_data):

                mask_binary = (mask_small * 255).astype(np.uint8)
                mask_full = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

                detections.append({
                    'mask': mask_full,
                    'box': boxes_data[i]
                })


        tracker.update(detections)


        display_frame = frame.copy()


        count = 0
        for track in tracker.tracks:
            if track.time_since_update == 0:
                count += 1

                x1, y1, x2, y2 = map(int, track.box)
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)


                pts = np.array(list(track.history), np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], False, (0, 0, 255), 2)


                cx, cy = track.centroid
                cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(display_frame, str(track.track_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(display_frame, f"Frame: {idx + 1}/{len(image_files)} Cells: {count}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        out_video.write(display_frame)
        cv2.imshow('Glioma Stem Cell Tracking', display_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_video.release()
    cv2.destroyAllWindows()
    print(f"处理完成！结果已保存至: {save_path}")
if __name__ == "__main__":
    model_path = r"D:\py\best1.pt"
    dataset_folder = r"D:\py\jiaozhiliu"

    run_tracking_folder(model_path, dataset_folder)