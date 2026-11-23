import cv2
import threading
import numpy as np

# --- Global State & Caching ---
cam_states = {}

bowtie_cache = {
    'zoom': None,
    'aspect': None,
    'map_x': None,
    'map_y': None
}

bev_cache = {
    'params': None,
    'map_x': None,
    'map_y': None
}

def toggle_dewarp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cam_id = param
        cam_states[cam_id]['dewarp_active'] = not cam_states[cam_id]['dewarp_active']

def dummy_callback(val):
    pass

class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.capture.set(cv2.CAP_PROP_FPS, 10)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.status, self.frame = self.capture.read()
        self.active = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.active:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()

    def get_frame(self):
        return self.status, self.frame

    def stop(self):
        self.active = False
        self.thread.join()
        self.capture.release()

# --- 1. Cylindrical Bowtie Mapping (Original) ---
def get_cylindrical_maps(w, h, zoom, aspect_expand):
    fov = 240
    radius = 1000
    y_squeeze = 1.8
    x_edge = 0.80
    bowtie = 0.30

    out_w = int(w * aspect_expand)
    out_h = out_w // 2
    cx = w / 2.0
    cy = h / 2.0

    if (bowtie_cache['zoom'] == zoom and
        bowtie_cache['aspect'] == aspect_expand and
        bowtie_cache['map_x'] is not None):
        return bowtie_cache['map_x'], bowtie_cache['map_y'], out_w, out_h

    fov_rad = np.radians(fov)
    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))

    x_norm = (x / out_w - 0.5) * 2.0
    base_lon = np.clip(x_norm * (np.pi / zoom), -np.pi, np.pi)
    lon = base_lon / (1.0 + x_edge * (x_norm**2))

    y_cyl = (y / out_h - 0.5) * (np.pi / zoom)
    y_modifier = y_squeeze / (1.0 + bowtie * (base_lon**2))
    y_cyl = y_cyl * y_modifier

    X = np.sin(lon)
    Z = np.cos(lon)

    r_3d = np.sqrt(1.0 + y_cyl**2)
    theta = np.arccos(np.clip(Z / r_3d, -1.0, 1.0))
    R_2d = np.maximum(np.sqrt(X**2 + y_cyl**2), 1e-5)

    f_len = radius / (fov_rad / 2.0)

    map_x = (cx + f_len * theta * (X / R_2d)).astype(np.float32)
    map_y = (cy + f_len * theta * (y_cyl / R_2d)).astype(np.float32)

    map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    bowtie_cache.update({
        'zoom': zoom,
        'aspect': aspect_expand,
        'map_x': map_x,
        'map_y': map_y
    })

    return map_x, map_y, out_w, out_h

# --- 2. Direct 1-Step Ground-Plane (Bird's-Eye View) Mapping ---
def get_bev_maps(w, h, height_m, pitch_deg, ppm, shift_z_m, out_w=700, out_h=700):
    params = (w, h, height_m, pitch_deg, ppm, shift_z_m, out_w, out_h)
    if bev_cache['params'] == params and bev_cache['map_x'] is not None:
        return bev_cache['map_x'], bev_cache['map_y']

    # Match your lens intrinsics
    fov = 240.0
    radius = 1000.0
    cx = w / 2.0
    cy = h / 2.0
    fov_rad = np.radians(fov)
    f_len = radius / (fov_rad / 2.0)

    # Create 2D pixel grid for the top-down output canvas
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32),
                       np.arange(out_h, dtype=np.float32))

    # Convert canvas pixels to physical ground coordinates (meters)
    # Xw: Right (+), Zw: Forward (+), Yw: Downward (+height_m below camera)
    Xw = (u - out_w / 2.0) / ppm
    Zw = ((out_h * 0.85) - v) / ppm + shift_z_m
    Yw = np.full_like(Xw, height_m)

    # 1. Apply camera downward pitch (rotation around X-axis)
    pitch = np.radians(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Yc = Yw * cp - Zw * sp
    Zc = Yw * sp + Zw * cp
    Xc = Xw

    # 2. Project 3D camera-frame rays directly into equidistant fisheye sensor
    r_3d = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta = np.arccos(np.clip(Zc / np.maximum(r_3d, 1e-5), -1.0, 1.0))
    R_2d = np.maximum(np.sqrt(Xc**2 + Yc**2), 1e-5)

    map_x = (cx + f_len * theta * (Xc / R_2d)).astype(np.float32)
    map_y = (cy + f_len * theta * (Yc / R_2d)).astype(np.float32)

    # Mask out any ground rays that fall outside the 240-deg lens FOV
    out_of_fov = theta > (fov_rad / 2.0)
    map_x[out_of_fov] = -1
    map_y[out_of_fov] = -1

    # Convert to fixed-point for ~2x faster cv2.remap
    map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    bev_cache.update({
        'params': params,
        'map_x': map_x,
        'map_y': map_y
    })

    return map_x, map_y
    # --- Main Execution ---
cameras = {}

for i in [1, 2, 3, 0]:
    cam = ThreadedCamera(i)
    if cam.capture.isOpened() and cam.status:
        cameras[i] = cam

        # Window 1: Original Bowtie Window
        bowtie_win = f"Panoramic Bowtie Cam {i}"
        cv2.namedWindow(bowtie_win)
        cv2.createTrackbar("Aspect Expand", bowtie_win, 15, 20, dummy_callback)
        cv2.setTrackbarMin("Aspect Expand", bowtie_win, 7)
        cv2.createTrackbar("Zoom", bowtie_win, 90, 300, dummy_callback)
        cv2.setTrackbarMin("Zoom", bowtie_win, 50)

        cam_states[i] = {'dewarp_active': False}
        cv2.setMouseCallback(bowtie_win, toggle_dewarp, param=i)

        # Window 2: Direct Bird's-Eye View Window
        bev_win = f"Bird's-Eye View Cam {i}"
        cv2.namedWindow(bev_win)
        cv2.createTrackbar("Height (cm)", bev_win, 100, 500, dummy_callback)
        cv2.setTrackbarMin("Height (cm)", bev_win, 10)
        cv2.createTrackbar("Pitch (Deg)", bev_win, 30, 90, dummy_callback)
        cv2.createTrackbar("Scale (px/m)", bev_win, 100, 150, dummy_callback)
        cv2.setTrackbarMin("Scale (px/m)", bev_win, 70)
        cv2.createTrackbar("Shift Z (cm)", bev_win, 300, 600, dummy_callback)

        break
    else:
        cam.stop()

if not cameras:
    raise RuntimeError("Could not find an attached USB camera.")

active_cam_id = list(cameras.keys())[0]
cam = cameras[active_cam_id]
bowtie_win = f"Panoramic Bowtie Cam {active_cam_id}"
bev_win = f"Bird's-Eye View Cam {active_cam_id}"

try:
    while True:
        status, frame = cam.get_frame()
        if not status or frame is None:
            break

        h, w = frame.shape[:2]

        # --- Update Window 1: Bowtie ---
        zoom = max(0.5, cv2.getTrackbarPos("Zoom", bowtie_win) / 100.0)
        aspect_expand = max(0.7, cv2.getTrackbarPos("Aspect Expand", bowtie_win) / 10.0)

        if cam_states[active_cam_id]['dewarp_active']:
            map_x, map_y, out_w, out_h = get_cylindrical_maps(w, h, zoom, aspect_expand)
            display_frame = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            label_status = f"BOWTIE ON | Aspect: {aspect_expand:.1f}x | Zoom: {zoom:.2f}x"
            color = (0, 255, 0)
        else:
            display_frame = frame
            label_status = "BOWTIE OFF (Click to Toggle)"
            color = (0, 0, 255)

        dh, dw = display_frame.shape[:2]
        display = cv2.resize(display_frame, (int(dw * 0.5), int(dh * 0.5)))
        cv2.putText(display, label_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow(bowtie_win, display)

        # --- Update Window 2: Direct Bird's-Eye View ---
        height_m = max(0.1, cv2.getTrackbarPos("Height (cm)", bev_win) / 100.0)
        pitch_deg = cv2.getTrackbarPos("Pitch (Deg)", bev_win)
        ppm = max(70, cv2.getTrackbarPos("Scale (px/m)", bev_win))
        shift_z_m = (cv2.getTrackbarPos("Shift Z (cm)", bev_win) - 300) / 100.0

        bev_map_x, bev_map_y = get_bev_maps(w, h, height_m, pitch_deg, ppm, shift_z_m)
        bev_frame = cv2.remap(
            frame, bev_map_x, bev_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        bev_label = f"H: {height_m:.2f}m | Pitch: {pitch_deg}deg | Scale: {ppm}px/m"
        cv2.putText(bev_frame, bev_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(bev_win, bev_frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
