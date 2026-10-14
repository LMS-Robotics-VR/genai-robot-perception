import cv2
import numpy as np
import pyrealsense2 as rs


'''
0, 1, 2, 3 for usb camera
"realsense" for realsense camera
'''


def image_capture(camera, export_path, image_extension):
    count = 0

    if type(camera) == int:
        cap = cv2.VideoCapture(camera)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                cv2.imshow("YOLOv8 Inference", frame)

                key = cv2.waitKey(1)
                if key == 113:
                    break
                elif key == 13:
                    frame = cv2.resize(frame, (912,512))
                    cv2.imwrite(export_path+f"/frame{count}.{image_extension}", frame)
                    print("Captured frame")
                    count += 1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


    elif camera == "realsense":

        # realsense camera Class
        class DepthCamera:
            def __init__(self):
                # Configure depth and color streams
                self.pipeline = rs.pipeline()
                config = rs.config()

                # Get device product line for setting a supporting resolution
                pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
                pipeline_profile = config.resolve(pipeline_wrapper)
                device = pipeline_profile.get_device()
                device_product_line = str(device.get_info(rs.camera_info.product_line))

                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

                # Start streaming
                self.pipeline.start(config)

            def get_frame(self):
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                if not depth_frame or not color_frame:
                    return False, None, None
                return True, depth_image, color_image

            def release(self):
                self.pipeline.stop()

        # create instance of depth camera
        depth_camera = DepthCamera()

        while True:
            success, depth_image, frame = depth_camera.get_frame()

            if success:

                cv2.imshow("Intel RealSense D415 feed", frame)

                key = cv2.waitKey(1)
                if key == 113:
                    break
                elif key == 13:
                    cv2.imwrite(export_path+f"/frame{count}.{image_extension}", frame)
                    print("Captured frame")
                    count += 1
            else:
                break

        depth_camera.release()
        cv2.destroyAllWindows()