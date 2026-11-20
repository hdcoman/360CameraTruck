import cv2
import threading
import numpy as np

# --- Global State Dictionary to track toggles per camera ---
cam_states = {}

def toggle_dewarp(event, x, y, flags, param):
    """Mouse callback: Left clicking the video feed toggles dewarping ON/OFF."""
    if event == cv2.EVENT_LBUTTONDOWN:
        cam_id = param
        cam_states[cam_id]['dewarp_active'] = not cam_states[cam_id]['dewarp_active']

def dummy_callback(val):
    pass

def unwarp_fisheye(frame, fov_degrees, balance_val, width_mult, out_zoom):
    fov_degrees = max(1, fov_degrees)

    h, w = frame.shape[:2]
    fov_rad = np.radians(fov_degrees)
    cx, cy = w / 2.0, h / 2.0

    focal_length = (h / 2.0) / (fov_rad / 2.0)

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    D = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float64)

    # Calculate the expanded output width
    out_w = int(w * width_mult)

    # Tell OpenCV to calculate the new matrix for the wider canvas
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance_val, new_size=(out_w, h)
    )

    # Scale down the output focal lengths to zoom out and reveal cropped edges
    K_new[0, 0] *= out_zoom
    K_new[1, 1] *= out_zoom

    # Generate the map using the expanded canvas size (out_w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (out_w, h), cv2.CV_16SC2
    )

    return cv2.remap(
        frame, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )


class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        # Force high resolution at the hardware level
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Force low framerate at the hardware level
        self.capture.set(cv2.CAP_PROP_FPS, 10)

        # Keep buffer limit to prevent stale frames
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


# --- Main Execution ---
cameras = {}

# BYPASS LAPTOP CAMERA: Scan starting at index 1 instead of 0
for i in range(1, 10):
    cam = ThreadedCamera(i)
    if cam.capture.isOpened() and cam.status:
        cameras[i] = cam

        window_name = f"Camera {i}"
        cv2.namedWindow(window_name)

        cam_states[i] = {'dewarp_active': False, 'window': window_name}
        cv2.setMouseCallback(window_name, toggle_dewarp, param=i)

        # Defaults updated based on preferences
        cv2.createTrackbar("Scale (%)", window_name, 50, 200, dummy_callback)
        cv2.createTrackbar("FOV (Deg)", window_name, 90, 280, dummy_callback)
        cv2.createTrackbar("Balance (0-100)", window_name, 55, 100, dummy_callback)
        cv2.createTrackbar("Aspect Expand", window_name, 15, 40, dummy_callback) # 15 = 1.5x
        cv2.createTrackbar("Proj Zoom", window_name, 100, 200, dummy_callback) # 100 = 1.0x

        if len(cameras) == 1:
            cv2.createTrackbar("Global FPS", window_name, 10, 60, dummy_callback)
    else:
        cam.stop()

print("Available camera feeds:", list(cameras.keys()))
if not cameras:
    raise RuntimeError("Could not open any camera feeds")

master_window = cam_states[list(cameras.keys())[0]]['window']
target_fps = 10

try:
    while cameras:
        # Use a list of keys so we can safely delete items from the dictionary while looping
        for i in list(cameras.keys()):
            cam = cameras[i]
            window_name = cam_states[i]['window']

            # Detect if the user manually closed this specific window by clicking the 'X'
            try:
                is_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                is_visible = 0

            if is_visible < 1:
                cam.stop()
                del cameras[i]

                # If the closed window was hosting the Global FPS slider, migrate it to the next open window
                if window_name == master_window and cameras:
                    master_window = cam_states[list(cameras.keys())[0]]['window']
                    cv2.createTrackbar("Global FPS", master_window, target_fps, 60, dummy_callback)
                continue

            status, frame = cam.get_frame()

            if not status or frame is None:
                cam.stop()
                del cameras[i]
                cv2.destroyWindow(window_name)

                # Migrate trackbar on stream fail
                if window_name == master_window and cameras:
                    master_window = cam_states[list(cameras.keys())[0]]['window']
                    cv2.createTrackbar("Global FPS", master_window, target_fps, 60, dummy_callback)
                continue

            fov = cv2.getTrackbarPos("FOV (Deg)", window_name)
            balance = cv2.getTrackbarPos("Balance (0-100)", window_name) / 100.0
            scale_percent = cv2.getTrackbarPos("Scale (%)", window_name)
            scale = max(0.01, scale_percent / 100.0)

            # Read the new width multiplier
            width_expand = cv2.getTrackbarPos("Aspect Expand", window_name) / 10.0
            width_expand = max(1.0, width_expand) # Prevent going below original width

            # Read the new zoom value (e.g. 75 becomes 0.75x)
            proj_zoom = cv2.getTrackbarPos("Proj Zoom", window_name) / 100.0
            proj_zoom = max(0.1, proj_zoom) # Prevent division by zero/crashing

            if cam_states[i]['dewarp_active']:
                display_frame = unwarp_fisheye(frame, fov, balance, width_expand, proj_zoom)
                label_status = "DEWARP: ON"
                color = (0, 255, 0)
            else:
                display_frame = frame.copy()
                label_status = "DEWARP: OFF (Click to Toggle)"
                color = (0, 0, 255)

            h, w = display_frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(display_frame, (new_w, new_h))

            cv2.putText(display, label_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            res_text = f"Display: {new_w}x{new_h} | Native: {w}x{h} ({w/h:.2f}:1)"
            cv2.putText(display, res_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)

        if not cameras:
            break

        # Read global FPS safely in case master window was just closed
        try:
            fps_val = cv2.getTrackbarPos("Global FPS", master_window)
            if fps_val > 0:
                target_fps = fps_val
        except cv2.error:
            pass

        loop_delay = int(1000 / target_fps)

        if cv2.waitKey(loop_delay) & 0xFF == ord("q"):
            break
finally:
    for cam in cameras.values():
        cam.stop()
    cv2.destroyAllWindows()
