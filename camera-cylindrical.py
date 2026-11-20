import cv2
import threading
import numpy as np

# --- Global State & Caching ---
cam_states = {}

map_cache = {
    'fov': None,
    'radius': None,
    'cy': None,
    'zoom': None,
    'aspect': None,
    'map_x': None,
    'map_y': None
}

def toggle_dewarp(event, x, y, flags, param):
    """Mouse callback: Left clicking toggles manual dewarping ON/OFF."""
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

# --- Cylindrical Projection Mapping ---

def get_cylindrical_maps(w, h, fov, radius, cy, zoom, aspect_expand):
    # Expand the destination image size to physically fit the un-scrunched edges[cite: 2]
    out_w = int(w * aspect_expand)

    # Maintain the 2:1 panoramic relationship
    out_h = out_w // 2
    cx = w / 2.0

    if (map_cache['fov'] == fov and map_cache['radius'] == radius and
        map_cache['cy'] == cy and map_cache['zoom'] == zoom and
        map_cache['aspect'] == aspect_expand):
        return map_cache['map_x'], map_cache['map_y'], out_w, out_h

    fov_rad = np.radians(max(1, fov))

    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # Longitude: Rotation around the cylinder
    lon = (x / out_w - 0.5) * (2 * np.pi / zoom)

    # Height: Linear vertical position on the flat cylinder wall
    y_cyl = (y / out_h - 0.5) * (np.pi / zoom)

    # 3D Coordinates on the cylinder
    X = np.sin(lon)
    Z = np.cos(lon)
    Y = y_cyl

    # Calculate the 3D distance from the origin to the point on the cylinder wall
    r_3d = np.sqrt(X**2 + Y**2 + Z**2)

    # Incident angle from the Z-axis (optical axis)
    theta = np.arccos(np.clip(Z / r_3d, -1.0, 1.0))

    # Angle on the 2D sensor plane
    angle = np.arctan2(Y, X)

    # Equidistant Fisheye Mapping (radius = focal_length * theta)
    f_len = radius / (fov_rad / 2.0)
    r = f_len * theta

    # Map back to Cartesian pixels on the raw camera sensor
    map_x = (cx + r * np.cos(angle)).astype(np.float32)
    map_y = (cy + r * np.sin(angle)).astype(np.float32)

    map_cache.update({
        'fov': fov, 'radius': radius, 'cy': cy, 'zoom': zoom, 'aspect': aspect_expand,
        'map_x': map_x, 'map_y': map_y
    })

    return map_x, map_y, out_w, out_h

# --- Main Execution ---
cameras = {}

for i in [1, 2, 3]:
    cam = ThreadedCamera(i)
    if cam.capture.isOpened() and cam.status:
        cameras[i] = cam
        window_name = f"Cylindrical Cam {i}"
        cv2.namedWindow(window_name)

        cv2.createTrackbar("FOV (Deg)", window_name, 260, 400, dummy_callback)
        cv2.setTrackbarMin("FOV (Deg)", window_name, 200)

        cv2.createTrackbar("Radius", window_name, 1200, 2000, dummy_callback)
        cv2.setTrackbarMin("Radius", window_name, 1000)

        cv2.createTrackbar("Center Y", window_name, 540, 1080, dummy_callback)

        # Aspect Expand to widen the output canvas (10 = 1.0x width)[cite: 2]
        cv2.createTrackbar("Aspect Expand", window_name, 15, 40, dummy_callback)

        cv2.createTrackbar("Zoom", window_name, 100, 300, dummy_callback)
        cv2.setTrackbarMin("Zoom", window_name, 10)

        cam_states[i] = {
            'dewarp_active': False
        }
        cv2.setMouseCallback(window_name, toggle_dewarp, param=i)
        break
    else:
        cam.stop()

if not cameras:
    raise RuntimeError("Could not find an attached USB camera.")

active_cam_id = list(cameras.keys())[0]
cam = cameras[active_cam_id]
window_name = f"Cylindrical Cam {active_cam_id}"

try:
    while True:
        status, frame = cam.get_frame()
        if not status or frame is None:
            break

        h, w = frame.shape[:2]

        fov = max(1, cv2.getTrackbarPos("FOV (Deg)", window_name))
        radius = max(1, cv2.getTrackbarPos("Radius", window_name))
        cy = cv2.getTrackbarPos("Center Y", window_name)
        zoom = max(0.1, cv2.getTrackbarPos("Zoom", window_name) / 100.0)

        # Read the width multiplier[cite: 2]
        aspect_expand = max(1.0, cv2.getTrackbarPos("Aspect Expand", window_name) / 10.0)

        if cam_states[active_cam_id]['dewarp_active']:
            map_x, map_y, out_w, out_h = get_cylindrical_maps(w, h, fov, radius, cy, zoom, aspect_expand)

            display_frame = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            # --- Software Panoramic Band ---
            # Crop the top and bottom 30% of the incoming video feed in software
            # This hides the 16:9 sensor tapering and leaves a wide horizontal band
            crop_amount = int(out_h * 0.3)
            display_frame = display_frame[crop_amount:(out_h - crop_amount), :]

            label_status = f"CYLINDRICAL: ON | Zoom: {zoom:.2f}x | Expand: {aspect_expand:.1f}x"
            color = (0, 255, 0)
        else:
            display_frame = frame.copy()
            label_status = "CYLINDRICAL: OFF (Click to Toggle)"
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
