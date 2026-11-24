import cv2
import threading
import numpy as np

# --- Global State ---
cam_states = {}
MODES = ["RAW", "BOWTIE", "BEV"]

# Slider definitions: (key, label, min_val, max_val, default_val, format_fn)
SLIDER_DEFS = [
    # Left Column (0..3)
    ("aspect",  "Aspect Expand", 7,   20,   10,   lambda v: f"{v/10.0:.1f}x"),
    ("fov",     "FOV (Deg)",     200, 280,  240,  lambda v: f"{v} deg"),
    ("radius",  "Radius (px)",   800, 1300, 1000, lambda v: f"{v} px"),
    ("zoom",    "Bowtie Zoom",   50,  300,  90,   lambda v: f"{v/100.0:.2f}x"),
    # Right Column (4..7)
    ("height",  "BEV Height",    10,  500,  100,  lambda v: f"{v/100.0:.2f} m"),
    ("pitch",   "BEV Pitch",     0,   90,   30,   lambda v: f"{v} deg"),
    ("ppm",     "BEV Scale",     70,  150,  100,  lambda v: f"{v} px/m"),
    ("shift_z", "BEV Shift Z",   0,   600,  300,  lambda v: f"{(v-300)/100.0:+.2f} m"),
]

PANEL_H = 165  # Height of the 2-column control bar below the video


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


# --- Custom 2-Column UI Panel & Mouse Handler ---
def update_slider_from_x(state, slider_idx, mouse_x):
    key, _, min_v, max_v, _, _ = SLIDER_DEFS[slider_idx]
    x0, _, x1, _ = state['slider_rects'][slider_idx]
    ratio = np.clip((mouse_x - x0) / max(1, (x1 - x0)), 0.0, 1.0)
    state['values'][key] = int(round(min_v + ratio * (max_v - min_v)))


def on_mouse(event, x, y, flags, param):
    cam_id = param
    state = cam_states[cam_id]
    video_h = state['video_h']

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Click inside the video area -> cycle view mode
        if y < video_h:
            state['mode_idx'] = (state['mode_idx'] + 1) % len(MODES)
            return

        # 2. Click a mode button in the top bar of the control panel
        for idx, (bx0, by0, bx1, by1) in enumerate(state['mode_rects']):
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                state['mode_idx'] = idx
                return

        # 3. Click/grab one of the 8 sliders in the 2-column grid
        for idx, (sx0, sy0, sx1, sy1) in enumerate(state['slider_rects']):
            if (sx0 - 8) <= x <= (sx1 + 8) and (sy0 - 12) <= y <= (sy1 + 12):
                state['dragging_slider'] = idx
                update_slider_from_x(state, idx, x)
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if state['dragging_slider'] is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            update_slider_from_x(state, state['dragging_slider'], x)

    elif event == cv2.EVENT_LBUTTONUP:
        state['dragging_slider'] = None


def render_control_panel(cam_id, width, video_h):
    """Renders a compact 2-column slider panel + mode selector bar."""
    state = cam_states[cam_id]
    panel = np.full((PANEL_H, width, 3), (32, 32, 36), dtype=np.uint8)
    cv2.line(panel, (0, 0), (width, 0), (80, 80, 90), 1)

    # Top row: Mode selector buttons
    btn_w = 105
    btn_h = 24
    btn_y0 = 8
    state['mode_rects'] = []
    mode_colors = [(70, 70, 210), (50, 175, 70), (40, 180, 210)]

    for idx, mode_name in enumerate(MODES):
        bx0 = 12 + idx * (btn_w + 8)
        bx1 = bx0 + btn_w
        by1 = btn_y0 + btn_h
        state['mode_rects'].append((bx0, video_h + btn_y0, bx1, video_h + by1))

        is_active = (state['mode_idx'] == idx)
        bg = mode_colors[idx] if is_active else (55, 55, 62)
        border = (240, 240, 240) if is_active else (95, 95, 105)
        cv2.rectangle(panel, (bx0, btn_y0), (bx1, by1), bg, -1)
        cv2.rectangle(panel, (bx0, btn_y0), (bx1, by1), border, 1)
        cv2.putText(panel, mode_name, (bx0 + 18, btn_y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, "Click video or buttons to switch view | 'q' to quit",
                (12 + 3 * (btn_w + 8) + 10, btn_y0 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 175, 185), 1, cv2.LINE_AA)

    # 2-Column Slider Grid (4 rows x 2 columns)
    col_gap = 24
    margin_x = 14
    col_w = (width - 2 * margin_x - col_gap) // 2
    row_start_y = 48
    row_step_y = 29

    state['slider_rects'] = []

    for idx, (key, label, min_v, max_v, _, fmt_fn) in enumerate(SLIDER_DEFS):
        col = idx // 4      # 0 = Left column, 1 = Right column
        row = idx % 4       # 0..3

        cx0 = margin_x + col * (col_w + col_gap)
        cy = row_start_y + row * row_step_y

        val = state['values'][key]
        text_str = f"{label}: {fmt_fn(val)}"
        cv2.putText(panel, text_str, (cx0, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 230), 1, cv2.LINE_AA)

        # Slider track sits right below the label inside that column
        track_y = cy + 9
        track_x0 = cx0
        track_x1 = cx0 + col_w
        state['slider_rects'].append((track_x0, video_h + track_y, track_x1, video_h + track_y))

        ratio = (val - min_v) / max(1, (max_v - min_v))
        knob_x = int(track_x0 + ratio * (track_x1 - track_x0))

        # Background track + active fill + knob
        cv2.line(panel, (track_x0, track_y), (track_x1, track_y), (75, 75, 85), 4, cv2.LINE_AA)
        cv2.line(panel, (track_x0, track_y), (knob_x, track_y), (0, 200, 255), 4, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (knob_x, track_y), 6, (40, 40, 45), 1, cv2.LINE_AA)

    return panel


# --- 1. Cylindrical Bowtie Mapping (Per-Camera Cached) ---
def get_cylindrical_maps(cam_id, w, h, zoom, aspect_expand, fov_deg, radius_px):
    cache = cam_states[cam_id]['bowtie_cache']
    params = (w, h, zoom, aspect_expand, fov_deg, radius_px)

    out_w = int(w * aspect_expand)
    out_h = out_w // 2

    if cache['params'] == params and cache['map_x'] is not None:
        return cache['map_x'], cache['map_y'], out_w, out_h

    y_squeeze = 1.8
    x_edge = 0.80
    bowtie = 0.30

    cx = w / 2.0
    cy = h / 2.0
    fov_rad = np.radians(fov_deg)

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

    f_len = radius_px / (fov_rad / 2.0)

    map_x = (cx + f_len * theta * (X / R_2d)).astype(np.float32)
    map_y = (cy + f_len * theta * (y_cyl / R_2d)).astype(np.float32)

    map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    cache.update({
        'params': params,
        'map_x': map_x,
        'map_y': map_y
    })

    return map_x, map_y, out_w, out_h


# --- 2. Direct 1-Step Ground-Plane (Bird's-Eye View) Mapping (Per-Camera Cached) ---
def get_bev_maps(cam_id, w, h, height_m, pitch_deg, ppm, shift_z_m, fov_deg, radius_px, out_w=700, out_h=700):
    cache = cam_states[cam_id]['bev_cache']
    params = (w, h, height_m, pitch_deg, ppm, shift_z_m, fov_deg, radius_px, out_w, out_h)

    if cache['params'] == params and cache['map_x'] is not None:
        return cache['map_x'], cache['map_y']

    cx = w / 2.0
    cy = h / 2.0
    fov_rad = np.radians(fov_deg)
    f_len = radius_px / (fov_rad / 2.0)

    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32),
                       np.arange(out_h, dtype=np.float32))

    Xw = (u - out_w / 2.0) / ppm
    Zw = ((out_h * 0.85) - v) / ppm + shift_z_m
    Yw = np.full_like(Xw, height_m)

    pitch = np.radians(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Yc = Yw * cp - Zw * sp
    Zc = Yw * sp + Zw * cp
    Xc = Xw

    r_3d = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    theta = np.arccos(np.clip(Zc / np.maximum(r_3d, 1e-5), -1.0, 1.0))
    R_2d = np.maximum(np.sqrt(Xc**2 + Yc**2), 1e-5)

    map_x = (cx + f_len * theta * (Xc / R_2d)).astype(np.float32)
    map_y = (cy + f_len * theta * (Yc / R_2d)).astype(np.float32)

    out_of_fov = theta > (fov_rad / 2.0)
    map_x[out_of_fov] = -1
    map_y[out_of_fov] = -1

    map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

    cache.update({
        'params': params,
        'map_x': map_x,
        'map_y': map_y
    })

    return map_x, map_y


# --- Main Execution ---
cameras = {}

# Scan indices 1..9 (skipping 0 to ignore the built-in laptop webcam)
for i in range(1, 10):
    cam = ThreadedCamera(i)
    if cam.capture.isOpened() and cam.status and cam.frame is not None:
        cameras[i] = cam
        win_name = f"USB Camera {i}"
        cv2.namedWindow(win_name)

        cam_states[i] = {
            'mode_idx': 0,  # 0: RAW, 1: BOWTIE, 2: BEV
            'window': win_name,
            'values': {k: default_v for k, _, _, _, default_v, _ in SLIDER_DEFS},
            'dragging_slider': None,
            'slider_rects': [],
            'mode_rects': [],
            'video_h': 540,
            'bowtie_cache': {'params': None, 'map_x': None, 'map_y': None},
            'bev_cache': {'params': None, 'map_x': None, 'map_y': None}
        }
        cv2.setMouseCallback(win_name, on_mouse, param=i)
    else:
        cam.stop()

if not cameras:
    raise RuntimeError("Could not find any attached external USB cameras.")

print(f"Detected external USB cameras on indices: {list(cameras.keys())}")

try:
    while cameras:
        for i, cam in list(cameras.items()):
            win_name = cam_states[i]['window']

            try:
                is_visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                is_visible = 0

            if is_visible < 1:
                cam.stop()
                del cameras[i]
                continue

            status, frame = cam.get_frame()
            if not status or frame is None:
                cam.stop()
                del cameras[i]
                cv2.destroyWindow(win_name)
                continue

            h, w = frame.shape[:2]
            vals = cam_states[i]['values']
            mode = MODES[cam_states[i]['mode_idx']]

            aspect_expand = vals['aspect'] / 10.0
            fov_deg = vals['fov']
            radius_px = vals['radius']

            if mode == "RAW":
                disp_w = int(w * 0.5 * aspect_expand)
                disp_h = int(h * 0.5)
                video_disp = cv2.resize(frame, (disp_w, disp_h))
                label = f"Cam {i} | RAW | Aspect: {aspect_expand:.1f}x"
                color = (0, 0, 255)

            elif mode == "BOWTIE":
                zoom = vals['zoom'] / 100.0
                map_x, map_y, _, _ = get_cylindrical_maps(
                    i, w, h, zoom, aspect_expand, fov_deg, radius_px
                )
                dewarped = cv2.remap(
                    frame, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                dh, dw = dewarped.shape[:2]
                video_disp = cv2.resize(dewarped, (int(dw * 0.5), int(dh * 0.5)))
                label = f"Cam {i} | BOWTIE | Aspect: {aspect_expand:.1f}x | Zoom: {zoom:.2f}x"
                color = (0, 255, 0)

            elif mode == "BEV":
                height_m = vals['height'] / 100.0
                pitch_deg = vals['pitch']
                ppm = vals['ppm']
                shift_z_m = (vals['shift_z'] - 300) / 100.0

                bev_out_w = int(700 * aspect_expand)
                bev_out_h = 700

                bev_map_x, bev_map_y = get_bev_maps(
                    i, w, h, height_m, pitch_deg, ppm, shift_z_m,
                    fov_deg, radius_px, out_w=bev_out_w, out_h=bev_out_h
                )
                video_disp = cv2.remap(
                    frame, bev_map_x, bev_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                label = f"Cam {i} | BEV | Aspect: {aspect_expand:.1f}x | H:{height_m:.2f}m Pitch:{pitch_deg}deg"
                color = (0, 255, 255)

            cv2.putText(video_disp, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            vh, vw = video_disp.shape[:2]
            cam_states[i]['video_h'] = vh

            panel = render_control_panel(i, vw, vh)
            combined = np.vstack([video_disp, panel])

            cv2.imshow(win_name, combined)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
finally:
    for cam in cameras.values():
        cam.stop()
    cv2.destroyAllWindows()
