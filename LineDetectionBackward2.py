import sys
import cv2 as cv
import numpy as np
import math
import socket
import json
import threading
import base64
import time
import os

current_color = "blue"
prev_angle_deg = 0.0
prev_x1 = 0
prev_y1 = 0
last_track_send_time = 0
track_send_interval = 1.0 / 5 
track_send_count = 0
track_send_timer = time.time()
prev_corrected_angle = 0.0

last_mask = None
last_output = None
last_track_vector = "line=[[0,0],[0,0]]"
last_deviation = 0
last_angle_servo = 90
last_contrast = 0

FRAME_DIR = "/var/run"
os.makedirs(FRAME_DIR, exist_ok=True)
current_frame_file = "1.jpg" 
current_mask_file = "mask1.jpg"
frame_toggle = True

last_t_found_time = 0
last_contour_time = 0
angle_offset = 0

skip = 100
output = 0
mask = 0
last_angle = 0
relay_val = 0
stop_line = 0

sign_bias = 0.0
sign_fix = 1.0

PIXELS_PER_METER = 240

def save_frame(frame):
    global current_frame_file
    path = os.path.join(FRAME_DIR, current_frame_file)

    base, ext = os.path.splitext(path)
    tmp_path = base + ".tmp" + ext

    
    ok = cv.imwrite(tmp_path, frame, [int(cv.IMWRITE_JPEG_QUALITY), 30])  
    if not ok:
        return
    os.replace(tmp_path, path)
    current_frame_file = "2.jpg" if current_frame_file == "1.jpg" else "1.jpg"
    
def save_mask(mask):
    global current_mask_file
    path = os.path.join(FRAME_DIR, current_mask_file)

    base, ext = os.path.splitext(path)
    tmp_path = base + ".tmp" + ext

    
    ok = cv.imwrite(tmp_path, mask, [int(cv.IMWRITE_JPEG_QUALITY), 30])  
    if not ok:
        return
    os.replace(tmp_path, path)
    current_mask_file = "mask2.jpg" if current_mask_file == "mask1.jpg" else "mask1.jpg"

def listen_to_server(sock):
    global current_color, active_cam
    while True:
        try:
            data = sock.recv(1024)
            if not data:
                break
            message = data.decode().strip()

            if message.startswith("line.color="):
                current_color = message.split("=")[1].strip()
                print(f"Выбран цвет: {current_color}")     

            
            elif message.startswith("line.cam="):
                try:
                    new_cam = int(message.split("=")[1].strip())
                    if new_cam in (0, 1):
                        active_cam = new_cam
                        print(f"Переключение на камеру {active_cam}")
                except:
                    pass
            
              
        except:
            break

# Аппроксимация точек
def fit_points(xs, ys):
    if len(xs) < 2:
        return None, None, False

    xs = np.array(xs, dtype=np.float64) / 1000.0
    ys = np.array(ys, dtype=np.float64) / 1000.0

    sumX = np.sum(xs)
    sumY = np.sum(ys)
    sumXY = np.sum(xs * ys)
    sumX2 = np.sum(xs * xs)

    xMean = sumX / len(xs)
    yMean = sumY / len(ys)
    denominator = sumX2 - sumX * xMean

    if abs(denominator) < 1e-7:
        return None, None, False

    k = (sumXY - sumX * yMean) / denominator
    b = yMean - k * xMean
    return k, b, True


'''
def detect_t_intersection(mask):
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, threshold=50,
                           minLineLength=int(mask.shape[1] * 0.1), maxLineGap=20)
    if lines is None or len(lines) < 2:
        return False

    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        if abs(angle) < 20 and length >= mask.shape[1] * 0.4:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) < 20 or abs(angle + 90) < 20:
            vertical.append((x1, y1, x2, y2))

    if not (horizontal and vertical):
        return False

    for hx1, hy1, hx2, hy2 in horizontal:
        for vx1, vy1, vx2, vy2 in vertical:
            h_cx = (hx1 + hx2) // 2
            h_cy = (hy1 + hy2) // 2
            v_cx = (vx1 + vx2) // 2
            v_cy = (vy1 + vy2) // 2

            if h_cy > v_cy:
                continue  

            if min(vy1, vy2) < h_cy - 10:
                continue

            if (min(hx1, hx2) < v_cx < max(hx1, hx2)):
                return True  

    return False
'''


def detect_t_intersection(mask):
    HORIZONTAL_MIN_FRAC = 0.50   # мин длина горизонтали
    H_ANGLE_TOL_DEG = 15         # допустимый угол горизонтально
    V_ANGLE_TOL_DEG = 25         # допустимый угол вертикально
    MIN_LINE_LEN_FRAC = 0.15     # minLineLength как доля ширины для Hough
    TOP_IGNORE_FRAC = 0.02       # игнорировать верхние 10% кадра

    h, w = mask.shape[:2]
    start_row = int(h * TOP_IGNORE_FRAC)
    search_mask = mask[start_row:, :]              

    min_line_length = int(w * MIN_LINE_LEN_FRAC)

    lines = cv.HoughLinesP(search_mask, 1, np.pi / 180, threshold=50,
                           minLineLength=min_line_length, maxLineGap=20)
    if lines is None or len(lines) < 2:
        return False

    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Y в координаты исходной маски
        y1 += start_row
        y2 += start_row

        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)

        # Классификация по углу и длине
        if abs(angle) < H_ANGLE_TOL_DEG and length >= w * HORIZONTAL_MIN_FRAC:
            horizontal.append((x1, y1, x2, y2, length, angle))
        elif abs(abs(angle) - 90) < V_ANGLE_TOL_DEG:
            vertical.append((x1, y1, x2, y2, length, angle))

    if not (horizontal and vertical):
        return False

    for hx1, hy1, hx2, hy2, hlen, hangle in horizontal:
        h_x_min = min(hx1, hx2)
        h_x_max = max(hx1, hx2)
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2

        '''
        if h_cy < int(h * 0.02): 
            continue
        '''

        for vx1, vy1, vx2, vy2, vlen, vangle in vertical:
            v_cx = (vx1 + vx2) // 2
            v_cy = (vy1 + vy2) // 2

            # горизонталь должна быть выше (меньше y) вертикали
            if h_cy > v_cy:
                continue

            # верхушка вертикали не должна быть значительно выше горизонтали
            if min(vy1, vy2) < h_cy - 10:
                continue

            # проверяем попадание X вертикали в горизонтальный span
            if (h_x_min <= v_cx <= h_x_max):
                # убедимся, что горизонталь по span >= требуемой доли ширины
                if (h_x_max - h_x_min) >= w * HORIZONTAL_MIN_FRAC:
                    return True

    return False

def process_frame(frame, sock, server_address):
    global skip
    global output
    global mask
    global last_angle
    global relay_val
    '''
    skip = skip + 1
    if (skip < 2):
        return mask, output

    '''
    
    skip = 0
    global prev_angle_deg, prev_x1, prev_y1
    global last_track_send_time, track_send_interval
    global prev_corrected_angle, sign_bias, sign_fix

    global last_mask, last_output, last_track_vector
    global last_deviation, last_angle_servo, last_contrast

    global last_t_found_time
    global last_contour_time

    global angle_offset
    global stop_line

    global PIXELS_PER_METER, CAM_OX, CAM_OY

    frame_height = frame.shape[0]
    roi = frame[int(frame_height * 0.6):, :]

    height = roi.shape[0]
    width = roi.shape[1]


    CAM_OX = width // 2
    CAM_OY = height - 2

    # b2 > (r2 + g2)(k2 - 1)
    
    B, G, R = cv.split(np.float32(roi))
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)
    v_min = 50

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    if current_color == "blue":
        k_blue = 1.35
        B2 = B ** 2
        RG2 = R ** 2 + G ** 2
        threshold_blue = RG2 * (k_blue ** 2 - 1)
        base_mask = (B2 > threshold_blue)

        #hue_mask = (H >= 100) & (H <= 130)
        hue_mask = (H >= 70) & (H <= 160)
        value_mask = V >= v_min

        mask = (base_mask & hue_mask & value_mask).astype(np.uint8) * 255
        mask = (base_mask & hue_mask & value_mask).astype(np.uint8) * 255

    elif current_color == "red":
        k_red = 1.7
        R2 = R ** 2
        GB2 = G ** 2 + B ** 2
        threshold_red = GB2 * (k_red ** 2 - 1)
        base_mask = (R2 > threshold_red)

        hue_mask = ((H >= 0) & (H <= 10)) | ((H >= 170) & (H <= 179))
        value_mask = V >= v_min

        mask = (base_mask & hue_mask & value_mask).astype(np.uint8) * 255

    elif current_color == "green":
        k_green = 1.5
        G2 = G ** 2
        RB2 = R ** 2 + B ** 2
        threshold_green = RB2 * (k_green ** 2 - 1)
        base_mask = (G2 > threshold_green)

        hue_mask = (H >= 40) & (H <= 85)
        value_mask = V >= v_min

        mask = (base_mask & hue_mask & value_mask).astype(np.uint8) * 255
        
    # Морфологическая обработка
    kernel = np.ones((15, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)

    # Вычисление контрастности
    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    mask_inverted = cv.bitwise_not(mask)
    mean_foreground = cv.mean(roi_gray, mask=mask)[0]
    mean_background = cv.mean(roi_gray, mask=mask_inverted)[0]
    contrast = abs(mean_foreground - mean_background) / 255 * 100

    # Поиск контуров с приоритетом по центральной вертикальной оси
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    output = roi.copy()

    angle_servo = 90
    deviation = last_deviation
    angle_deg = 0
    line_vector = "line=[[0,0],[0,0]]"

    # Если найдено больше 1 контура - аппроксимация 
    if len(contours) > 1:
        points = []
        for cnt in contours:
            for p in cnt:
                x, y = p[0]
                points.append((x, y))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        k, b, success = fit_points(xs, ys)
        if success:
            for x in range(roi.shape[1]):
                y = int(k * (x / 1000.0) + b)
                y = int(y * 1000)
                if 0 <= y < roi.shape[0]:
                    cv.circle(mask, (x, y), 1, 255, -1)

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # выбор контура, ближайшего к центральной вертикали
    if contours:
        h, w = roi.shape[:2]
        cx_img = w // 2

        # Параметры фильтрации 
        #edge_margin_px = int(min(w, h) * 0.05)  
        min_area_px = 600  # минимальная площадь контура

        scored = []
        for cnt in contours:
            # пропуск маленьких контуров
            area = cv.contourArea(cnt)
            if area < min_area_px:
                continue

            # проверка касания края кадра
            x, y, bw, bh = cv.boundingRect(cnt)
            x_min, x_max = x, x + bw
            y_min, y_max = y, y + bh

            # не учитываются контуры, которые касаются краёв
            ''' 
            if x_min <= edge_margin_px or y_min <= edge_margin_px or x_max >= (w - edge_margin_px) or y_max >= (h - edge_margin_px):
                continue
            '''

            # центр масс контура 
            M = cv.moments(cnt)
            if M.get("m00", 0) == 0:
                cxx = int(x + bw / 2)
                cyy = int(y + bh / 2)
            else:
                cxx = int(M["m10"] / M["m00"])
                cyy = int(M["m01"] / M["m00"])

            # отклонение от центральной вертикали кадра 
            dist_x = abs(cxx - cx_img)

            score = float(dist_x)

            scored.append((cnt, score))

        # лучший контур по минимуму отклонения 
        if scored:
            scored.sort(key=lambda t: t[1])  
            longest_contour = scored[0][0]
        else:
            longest_contour = max(contours, key=cv.contourArea)

        cv.drawContours(output, [longest_contour], -1, (0, 0, 255), 2)

        # Если контур очень короткий - пропускаем
        if len(longest_contour) < 15:
            now = time.time()
            if now - last_contour_time >= 1.0:
                print("Контур слишком короткий")
                last_contour_time = now
            return mask, output

        # Центр масс
        M = cv.moments(longest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            image_center_x = roi.shape[1] // 2
            deviation = cx - image_center_x
            last_deviation = deviation

        
        pts = longest_contour.reshape(-1, 2)
        
        # Вектор
        bottom_y = np.max(pts[:, 1])
        bottom_candidates = pts[pts[:, 1] == bottom_y]
        x_bottom = int(np.mean(bottom_candidates[:, 0]))
        y_bottom = int(bottom_y)

        top_y = np.min(pts[:, 1])
        top_candidates = pts[pts[:, 1] == top_y]
        x_top = int(np.mean(top_candidates[:, 0]))
        y_top = int(top_y)

        min_separation_px = 5
        if abs(y_bottom - y_top) < min_separation_px and abs(x_bottom - x_top) < min_separation_px:
            try:
                vx, vy, x0, y0 = cv.fitLine(longest_contour, cv.DIST_L2, 0, 0.01, 0.01).flatten()
                math_vx = float(vx)
                math_vy = -float(vy) 
                angle_rad = math.atan2(math_vy, math_vx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360
                if angle_deg > 180: angle_deg -= 180
                cx, cy = int(x0), int(y0)
                vector_length = 100
                x1_vis = int(cx + math_vx * vector_length)
                y1_vis = int(cy - math_vy * vector_length)
                x0_vis, y0_vis = cx, cy
                last_angle = angle_deg
            except Exception:
                angle_deg = last_angle
                x0_vis, y0_vis = x_bottom, y_bottom
                x1_vis, y1_vis = x_top, y_top
        else:
            vx = x_top - x_bottom
            math_vx = float(vx)
            math_vy = float(y_bottom - y_top)  
            angle_rad = math.atan2(math_vy, math_vx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0: angle_deg += 360
            if angle_deg > 180: angle_deg -= 180

            angle_deg = 180.0 - angle_deg

            x0_vis, y0_vis = x_bottom, y_bottom
            vector_length = 80
            norm = max(1.0, np.hypot(math_vx, math_vy))
            x1_vis = int(x0_vis + math_vx * (vector_length / norm))
            y1_vis = int(y0_vis - math_vy * (vector_length / norm))  

        # Сглаживание угла
        
        alpha = 0.2
        angle_deg = prev_angle_deg * (1 - alpha) + angle_deg * alpha
        prev_angle_deg = angle_deg

        line_angle = angle_deg - 90
        
        # ----------------- подготовка сглаженного визуального конца -----------------
        # (уже были в твоём коде — оставляем, но обязательно выполнить до конверсии в метры)
        x1_smooth = int(prev_x1 * (1 - alpha) + x1_vis * alpha)
        y1_smooth = int(prev_y1 * (1 - alpha) + y1_vis * alpha)
        prev_x1, prev_y1 = x1_smooth, y1_smooth

        cv.arrowedLine(output, (int(x0_vis), int(y0_vis)),
                    (int(x1_smooth), int(y1_smooth)),
                    (0, 255, 255), 2, tipLength=0.3)

        # ----------------- Перевод в метры (с явными параметрами) -----------------
        pixels_per_meter = PIXELS_PER_METER   # вынеси выше
        CAM_OX = CAM_OX
        CAM_OY = CAM_OY

        # Преобразование: x_pixel -> meters lateral; y_pixel -> meters forward
        # Важно: в изображении y увеличивается вниз; мы хотим y_forward = (CAM_OY - y_pixel) / ppm
        x0_m = (int(x0_vis) - CAM_OX) / pixels_per_meter
        y0_m = (CAM_OY - int(y0_vis)) / pixels_per_meter
        x1_m = (int(x1_smooth) - CAM_OX) / pixels_per_meter
        y1_m = (CAM_OY - int(y1_smooth)) / pixels_per_meter

        # clamp (в твоих пределах)
        X_MIN, X_MAX = -0.9, 0.9
        Y_MIN, Y_MAX = 0.1, 1.4
        def clamp(val, mn, mx): return max(min(val, mx), mn)
        x0_m = clamp(round(x0_m, 2), X_MIN, X_MAX)
        y0_m = clamp(round(y0_m, 2), Y_MIN, Y_MAX)
        x1_m = clamp(round(x1_m, 2), X_MIN, X_MAX)
        y1_m = clamp(round(y1_m, 2), Y_MIN, Y_MAX)

        # ----------------- Hierarchical search (coarse->fine) -----------------
        # параметры поиска
        L_m = 2.4
        v_sim = 1.0
        dt_sim = 0.05
        max_sim_time = 1.0
        n_sim_steps = int(max_sim_time / dt_sim)
        samples_y = 16
        initial_parts = 7
        refine_parts = 9
        levels = 2
        MAX_STEER_DEG = 45.0
        min_forward = 0.03

        # вынеси функции вверх в модуль (опционально), но для простоты оставим тут
        '''
        def simulate_trajectory_for_sigma(sigma_rad, L=L_m, v=v_sim, dt=dt_sim, max_steps=n_sim_steps):
            v = abs(v)
            x = 0.0; y = 0.0; yaw = 0.0
            traj_x = [x]; traj_y = [y]
            for _ in range(max_steps):
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
                yaw += (v * math.tan(sigma_rad) / L) * dt
                traj_x.append(x); traj_y.append(y)
            return np.array(traj_x), np.array(traj_y)
        '''

        def simulate_trajectory_for_sigma(sigma_rad, L=L_m, v=v_sim, dt=dt_sim, max_steps=n_sim_steps):
            # x_lat = lateral (м), y_fwd = forward (м)
            x_lat = 0.0
            y_fwd = 0.0
            yaw = 0.0
            traj_x = [x_lat]   # lateral
            traj_y = [y_fwd]   # forward
            v_use = abs(v)
            sign = -1.0  # если rear-steer, влияет только на yaw sign
            for _ in range(max_steps):
                # стандартные уравнения, но так, чтобы traj_y = forward, traj_x = lateral
                y_fwd += v_use * math.cos(yaw) * dt   # forward = v*cos(yaw)
                x_lat  += v_use * math.sin(yaw) * dt   # lateral = v*sin(yaw)
                yaw += sign * (v_use * math.tan(sigma_rad) / L) * dt
                traj_x.append(x_lat)
                traj_y.append(y_fwd)
            return np.array(traj_x), np.array(traj_y)



        def interp_x_at_y(traj_x, traj_y, query_y):
            if query_y < traj_y[0] - 1e-9:   # query below start
                return traj_x[0]
            if query_y > traj_y[-1] + 1e-9:
                return None
            return float(np.interp(query_y, traj_y, traj_x))

        def line_x_at_y(x0, y0, x1, y1, query_y):
            ymin = min(y0, y1); ymax = max(y0, y1)
            if query_y < ymin - 1e-9 or query_y > ymax + 1e-9:
                return None
            if abs(y1 - y0) < 1e-9:
                return x0
            t = (query_y - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))

        def compute_area_error_for_sigma_deg(sigma_deg, x0, y0, x1, y1, top_y_limit):
            sigma_rad = math.radians(sigma_deg)
            traj_x, traj_y = simulate_trajectory_for_sigma(sigma_rad)
            y_start = min(y0, y1)
            y_top = min(max(y0, y1), top_y_limit)
            if y_top - y_start < min_forward:
                return 1e6
            ys = np.linspace(y_start, y_top, samples_y)
            dy = (y_top - y_start) / max(1, samples_y - 1)
            area = 0.0
            for qy in ys:
                x_line = line_x_at_y(x0, y0, x1, y1, qy)
                x_pred = interp_x_at_y(traj_x, traj_y, qy)
                if x_line is None or x_pred is None:
                    return 1e5 + max(0.0, (y_top - traj_y[-1])) * 1e3
                area += abs(x_pred - x_line) * dy
            return area

        # top_y_m: верх по высоте (не искать выше)
        top_y_m = max(y0_m, y1_m)

        # search

        '''
        # === Прямая интерпретация угла линии ===
        # Вертикальная линия (90°) → 0°, отклонения влево/вправо ±45°
        line_angle = angle_deg - 90.0  # 90° = вертикаль
        line_angle = np.clip(line_angle, -45.0, 45.0)

        # === Учёт смещения по X (если линия не по центру) ===
        offset_x = (x0_m + x1_m) / 2.0  # среднее смещение линии относительно центра кадра (в метрах)
        k_offset = 50.0  # чувствительность к смещению, подбери эмпирически
        offset_correction = np.clip(k_offset * offset_x, -30.0, 30.0)  # ограничение корректировки


        #print("angle_deg, line_angle:", angle_deg, line_angle)
        #print("x0_vis,y0_vis,x1_smooth,y1_smooth", x0_vis, y0_vis, x1_smooth, y1_smooth)
        #print("x0_m,y0_m,x1_m,y1_m", x0_m, y0_m, x1_m, y1_m)
        #print("CAM_OX,CAM_OY,PPM:", CAM_OX, CAM_OY, PIXELS_PER_METER)


        # Итоговый угол поворота (учёт угла и смещения)
        best_angle = line_angle + offset_correction

        # Сглаживание
        alpha_ctrl = 0.10
        try:
            prev_relay_angle
        except NameError:
            prev_relay_angle = best_angle

        relay_angle_smoothed = prev_relay_angle * (1 - alpha_ctrl) + best_angle * alpha_ctrl
        prev_relay_angle = relay_angle_smoothed

        # Коррекция для заднего рулевого управления
        direction_factor = -1.0

        # Итоговый управляющий угол
        relay_val = float(np.clip(direction_factor * relay_angle_smoothed, -MAX_STEER_DEG, MAX_STEER_DEG))

        print("best_angle, relay_angle_smoothed, relay_val, REAR_STEER:", best_angle, relay_angle_smoothed, relay_val, REAR_STEER)




        '''
        a, b = -MAX_STEER_DEG, MAX_STEER_DEG
        best_angle = 0.0
        for level in range(levels):
            parts = initial_parts if level == 0 else refine_parts
            candidates = np.linspace(a, b, parts)
            best_local = None; best_err = float("inf")
            for cand in candidates:
                err = compute_area_error_for_sigma_deg(cand, x0_m, y0_m, x1_m, y1_m, top_y_m)
                if err < best_err:
                    best_err = err; best_local = cand
            if best_local is None:
                break
            span = (b - a) / parts
            a = max(-MAX_STEER_DEG, best_local - span)
            b = min(MAX_STEER_DEG, best_local + span)
            best_angle = best_local
        

        # final fine search
        span = max(span, 1.0)  # защитное значение
        fine_candidates = np.linspace(max(-MAX_STEER_DEG, best_angle - span/2),
                                    min(MAX_STEER_DEG, best_angle + span/2), 9)
        best_err = float("inf")
        for cand in fine_candidates:
            err = compute_area_error_for_sigma_deg(cand, x0_m, y0_m, x1_m, y1_m, top_y_m)
            if err < best_err:
                best_err = err; best_angle = cand

        

        # сглаживание и формирование relay_val
        alpha_ctrl = 0.10
        
        # alpha_ctrl уже у тебя задан
        try:
            prev_relay_angle
        except NameError:
            prev_relay_angle = best_angle

        relay_angle_smoothed = prev_relay_angle * (1 - alpha_ctrl) + best_angle * alpha_ctrl
        prev_relay_angle = relay_angle_smoothed

        # определяем направление (используй свою переменную скорости или флаг)
        # пример: current_speed — скорость в м/с (может быть + или -)
        direction_factor = -1.0 
        # или: direction_factor = -1.0 if is_reverse else 1.0

        # итоговый управляющий угол, который будем отправлять
        relay_val = float(np.clip(direction_factor * relay_angle_smoothed, -MAX_STEER_DEG, MAX_STEER_DEG))
        

        #cv.putText(output, f"hier_sigma:{relay_val:5.1f}deg err:{best_err:.3f}", (10,24),
                #cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)




        


        # отрисовать предсказанную траекторию для best_angle (надо после вычисления best_angle)
        traj_x, traj_y = simulate_trajectory_for_sigma(math.radians(best_angle))
        # конвертация в пиксели
        pts_px = []





        print("ROI size (w,h):", w, h)
        #print("CAM_OX,CAM_OY,PIXELS_PER_METER:", CAM_OX, CAM_OY, PIXELS_PER_METER)
        #print("traj_x range:", traj_x.min(), traj_x.max(), "traj_y range:", traj_y.min(), traj_y.max())
        # show first mapped pixel points
        mapped = [(int(CAM_OX + tx * PIXELS_PER_METER), int(CAM_OY - ty * PIXELS_PER_METER))
                for tx,ty in zip(traj_x[:10], traj_y[:10])]
        print("mapped first pts:", mapped)

        for tx, ty in zip(traj_x, traj_y):
            px = int(CAM_OX + tx * PIXELS_PER_METER)   # x_pixel = CAM_OX + x_m * ppm
            py = int(CAM_OY - ty * PIXELS_PER_METER)   # y_pixel = CAM_OY - y_m * ppm
            # stop если выходит за пределы кадра
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            pts_px.append((px, py))

        if len(pts_px) >= 2:
            cv.polylines(output, [np.array(pts_px, dtype=np.int32)], False, (255, 0, 0), 2)  # синий - предсказанная траектория
            # и нарисовать ближайшие точки кружком
            for i, (px, py) in enumerate(pts_px[:10]):
                cv.circle(output, (px, py), 2, (255, 0, 0), -1)


        # -------------------------------------------------------------------------













        #steer_angle = line_angle * 5 + 2 * deviation
        #steer_angle = line_angle * 2.5 + 2 * deviation
        #relay_val = -1 * max(-45, min(steer_angle, 45))
        #last_angle = relay_val

        '''
        steer_angle = line_angle + 0.4 * deviation
        relay_val = max(-45, min(steer_angle, 45))
        last_angle = relay_val
        '''

        # Перевод в угол сервы
        #angle_offset = int(angle_deg - 90)
        #angle_offset = max(-60, min(60, angle_offset))
        #angle_servo = 90 + angle_offset

        '''
        # Сглаживание визуального конца вектора
        x1_smooth = int(prev_x1 * (1 - alpha) + x1_vis * alpha)
        y1_smooth = int(prev_y1 * (1 - alpha) + y1_vis * alpha)
        prev_x1, prev_y1 = x1_smooth, y1_smooth

        # Стрелка вектора
        cv.arrowedLine(output, (int(x0_vis), int(y0_vis)),
                    (int(x1_smooth), int(y1_smooth)),
                    (0, 255, 255), 2, tipLength=0.3)

        # Перевод в метры 
        pixels_per_meter = 80
        x0_m = round((int(x0_vis) - 80) / pixels_per_meter, 2)
        y0_m = round((int(y0_vis) - 60) / pixels_per_meter, 2)
        x1_m = round((int(x1_smooth) - 80) / pixels_per_meter, 2)
        y1_m = round((60 - int(y1_smooth)) / pixels_per_meter, 2)

        # clamp 
        X_MIN, X_MAX = -0.9, 0.9
        Y_MIN, Y_MAX = 0.1, 1.4
        def clamp(val, mn, mx): return max(min(val, mx), mn)
        x0_m = clamp(x0_m, X_MIN, X_MAX)
        y0_m = clamp(y0_m, Y_MIN, Y_MAX)
        x1_m = clamp(x1_m, X_MIN, X_MAX)
        y1_m = clamp(y1_m, Y_MIN, Y_MAX)
        '''


        line_vector = json.dumps([[x0_m, y0_m], [x1_m, y1_m]])

    now = time.time()
    if now - last_track_send_time >= track_send_interval:
        message = (
            f"line.pos={deviation}\n"
            #f"line.angle={angle_servo:.2f}\n"
            #f"relay.angle={angle_servo - 90 + deviation / 2:.2f}\n"
            #f"relay.angle={relay_val:.2f}\n"
            f"line.angle={relay_val - 90:.2f}\n"
            f"line.contr={contrast:.2f}\n"
            f"line={line_vector}\n"
            f"line.color={current_color}\n"
            #f"relay.angle={angle_servo:.2f}\n" #!!!!!!!!
            f"line.stop={stop_line:.2f}\n"
            
        )
        sock.send(message.encode())
        last_track_send_time = now

        print("angle deg=", angle_deg)
        print("angle relay=", relay_val)
        print("deviation=", deviation)

        # Подсчёт отправок в секунду
        global track_send_count, track_send_timer
        track_send_count += 1
        if time.time() - track_send_timer >= 1.0:
            print(f"Отправлено line.* сообщений: {track_send_count} в сек.")
            track_send_count = 0
            track_send_timer = time.time()
            
    t_found = detect_t_intersection(mask)
    stop_line = 0
    if t_found:
        now = time.time()
        if now - last_t_found_time >= 1.0:
            print("Обнаружена Т-образная линия")
            last_t_found_time = now
            stop_line = 1

    return mask, output


def main(argv):
    fps_limit = 5  
    frame_interval = 1.0 / fps_limit
    last_time = time.time()
    frame_count = 0
    fps_timer = time.time()
    global active_cam

    server_ip = '127.0.0.1'
    server_port = 7777
    server_address = (server_ip, server_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(server_address)
    except Exception as e:
        print(f"Ошибка подключения к серверу: {e}")
        return -1

    threading.Thread(target=listen_to_server, args=(sock,), daemon=True).start()


    cap0= cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
    cap0.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    if not cap0.isOpened():
        print('Ошибка при открытии веб-камеры 0')
        return -1
        
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, 160)
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, 120)
    
    
    cap1 = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)
    cap1.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    if cap1.isOpened():
        cap1.set(cv.CAP_PROP_FRAME_WIDTH, 160)
        cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 120)
    else:
        print('веб-камера 1 недоступна')
        cap1 = None
    

    # текущая активная камера
    active_cam = 0

    while True:
        # Читаем кадры
        ret0, frame0 = cap0.read()
        
        ret1, frame1 = (False, None)
        if cap1:
            ret1, frame1 = cap1.read()
            if ret1:
                frame1 = cv.resize(frame1, (160, 120))
        
        if not ret0 and not ret1:
            print("Нет сигнала с камер")
            break

        if active_cam == 0 and ret0:
            frame = frame0
        
        elif active_cam == 1 and cap1 and ret1:
            frame = frame1
        
        else:
            continue

        # Ограничение FPS
        now = time.time()
        elapsed = now - last_time
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            last_time += frame_interval
        else:
            last_time = now

        # FPS счётчик
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            print(f"FPS: {frame_count}, активная камера: {active_cam}")
            frame_count = 0
            fps_timer = time.time()

        # обработка
        mask, output = process_frame(frame, sock, server_address)

        save_frame(output)
        save_mask(mask)

    cap0.release()
    
    if cap1:
        cap1.release()
    
    cv.destroyAllWindows()
    sock.close()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
