# 360CameraTruck
import importlib

try:
	cv2 = importlib.import_module("cv2")
except ModuleNotFoundError as error:
	raise RuntimeError(
		"OpenCV is not installed. Install it with: python -m pip install opencv-python"
	) from error

cameras = {}
for index in range(10):
	camera = cv2.VideoCapture(index)
	if camera.isOpened():
		cameras[index] = camera
	else:
		camera.release()

print("Available camera feeds:", list(cameras))
if not cameras:
	raise RuntimeError("Could not open any camera feeds")

try:
	while cameras:
		for index, camera in list(cameras.items()):
			success, frame = camera.read()
			if not success:
				camera.release()
				del cameras[index]
				cv2.destroyWindow(f"USB Camera {index}")
				continue

			cv2.imshow(f"USB Camera {index}", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
finally:
	for camera in cameras.values():
		camera.release()
	cv2.destroyAllWindows()
