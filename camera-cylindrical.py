import cv2
import threading
import numpy as np

# --- Global State & Caching ---
cam_states = {}

map_cache = {
    'zoom': None,
    'aspect': None,
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

# --- Cylindrical Bowtie Mapping ---

def get_cylindrical_maps(w, h, zoom, aspect_expand):
    fov = 240
    radius = 1000
    y_squeeze = 1.6
    x_edge = 0.80
    bowtie = 0.30

    out_w = int(w * aspect_expand)
    out_h = out_w // 2
    cx = w / 2.0
    cy = h / 2.0

    if (map_cache['zoom'] == zoom and
        map_cache['aspect'] == aspect_expand and
        map_cache['map_x'] is not None):
        return map_cache['map_x'], map_cache['map_y'], out_w, out_h

    fov_rad = np.radians(fov)
    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))

    x_norm = (x / out_w - 0.5) * 2.0
    base_lon = np.clip(x_norm * (np.pi / zoom), -np.pi, np.pi)
    lon = base_lon / (1.0 + x_edge * (x_norm**2))

    y_cyl = (y / out_h - 0.5) * (np.pi / zoom)
    y_modifier = y_squeeze / (1.0 + bowtie * (base_lon**2))
    y_cyl = y_cyl * y_modifier

    # --- Optimized 3D to Fisheye Pipeline ---
    X = np.sin(lon)
    Z = np.cos(lon)

    # 1. 3D Radius Optimization (X^2 + Z^2 = 1)
    r_3d = np.sqrt(1.0 + y_cyl**2)
    theta = np.arccos(np.clip(Z / r_3d, -1.0, 1.0))

    # 2. 2D Projected Radius (bypasses arctan2 and cos/sin)
    # np.maximum prevents division by zero at the exact center (X=0, Y=0)
    R_2d = np.maximum(np.sqrt(X**2 + y_cyl**2), 1e-5)

    f_len = radius / (fov_rad / 2.0)

    map_x = (cx + f_len * theta * (X / R_2d)).astype(np.float32)
    map_y = (cy + f_len * theta * (y_cyl / R_2d)).astype(np.float32)

    # 3. Fixed-Point Conversion for ~2x cv2.remap performance[cite: 2, 3]
    map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    map_cache.update({
        'zoom': zoom,
        'aspect': aspect_expand,
        'map_x': map_x,
        'map_y': map_y
    })

    return map_x, map_y, out_w, out_h

# --- Main Execution ---
cameras = {}

for i in [1, 2, 3, 0]:
    cam = ThreadedCamera(i)
    if cam.capture.isOpened() and cam.status:
        cameras[i] = cam
        window_name = f"Panoramic Bowtie Cam {i}"
        cv2.namedWindow(window_name)

        cv2.createTrackbar("Aspect Expand", window_name, 15, 20, dummy_callback)
        cv2.setTrackbarMin("Aspect Expand", window_name, 7)

        cv2.createTrackbar("Zoom", window_name, 90, 300, dummy_callback)
        cv2.setTrackbarMin("Zoom", window_name, 50)

        cam_states[i] = {'dewarp_active': False}
        cv2.setMouseCallback(window_name, toggle_dewarp, param=i)
        break
    else:
        cam.stop()

if not cameras:
    raise RuntimeError("Could not find an attached USB camera.")

active_cam_id = list(cameras.keys())[0]
cam = cameras[active_cam_id]
window_name = f"Panoramic Bowtie Cam {active_cam_id}"

try:
    while True:
        status, frame = cam.get_frame()
        if not status or frame is None:
            break

        h, w = frame.shape[:2]

        zoom = max(0.5, cv2.getTrackbarPos("Zoom", window_name) / 100.0)
        aspect_expand = max(0.7, cv2.getTrackbarPos("Aspect Expand", window_name) / 10.0)

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
            # frame.copy() removed; cv2.resize handles the allocation safely
            display_frame = frame
            label_status = "BOWTIE OFF (Click to Toggle)"
            color = (0, 0, 255)

        dh, dw = display_frame.shape[:2]
        new_w, new_h = int(dw * 0.5), int(dh * 0.5)
        display = cv2.resize(display_frame, (new_w, new_h))

        cv2.putText(display, label_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
