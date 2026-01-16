import math
import time
import cv2 as cv
import pyrealsense2 as rs
from ultralytics import YOLO
import stretch_body.robot
import numpy as np
import pyttsx3

class ObjectDetectionManipulation():
    def __init__(self, target_object):
        # --- Parameters ---
        self.head_cam_serial_number     = None  # H-cam serial number
        self.gripper_cam_serial_number  = None  # G-cam serial number

        self.start_tilt = math.radians(-30.0)  # -30° downward (head)
        self.forward_direction_pan = math.radians(-90.0)   # -90° (far left) (head) #XXX
        self.backward_direction_pan = math.radians(90.0)  # +90° (far right) (head)
        
        self.pan_step_size = 0.03  # rad (~2°)
        self.pan_speed     = 0.1 # sec

        self.head_FOV_pan  = math.radians(69.4)   # RealSense D435 pan field of view 
        self.head_FOV_tilt = math.radians(42.5)  # RealSense D435 tilt field of view

        self.target_object     = target_object  # Object to detected
        self.target_confirmed  = False
        self.required_frames   = 3  # Required frames to confirm object presence
        self.detected_frames   = 0  # Object detected in # frames

        self.current_pos = 0.0
        self.max_distance = 2.5  # meters (~9ft)
        self.pos_step_size = 0.1  # meters
        self.dist_to_object = 0.0  # Distance Stretch need to move to get close to the object
        self.boundary_reached = False
        self.object_retrieved = False
        self.search_direction = None

        
        self.current_pan              = self.forward_direction_pan
        self.current_tilt             = self.start_tilt
        self.robot_centered_at_object = False
        self.gripper_cam_lowered      = False

        self.SAFE_DISTANCE = 0.4  # meters

        self.head_cam_height                  = 1.0  # H-cam at height 1.0m
        self.gripper_cam_height               = 0.90  # G-cam at height 0.90 m
        self.gripper_image_height             = 480  # Image height in pixels. G-Image shape: 480 x 640
        self.gripper_cam_vertical_FOV_degrees = 58  # Vertical FOV of Realsense d405 is 87° x 58°
        self.gripper_cam_vertical_FOV_radians = self.gripper_cam_vertical_FOV_degrees * (math.pi / 180)

        # --- Depth cam moving average setup ---
        self.depth_buffer    = []
        self.buffer_size     = 5
        self.head_cam_height = 1.0  # meters
        self.table_height    = 0.7  # meters
        self.arm_length      = 0.5  # meters

        self._init_robot()
        self._init_yolo()
        self._get_cam_serial_numbers()
        self._init_realsense_camera(cam="head")  # default H-cam

        # --- Tilt H-cam ---
        self.r.head.move_to('head_pan', self.forward_direction_pan)
        self.r.head.move_to('head_tilt', self.start_tilt)
        self.r.wait_command()
        time.sleep(0.1)
        print('[INFO] Ready. Scanning for target object...')

    def _init_robot(self):
        ''' Initialize Stretch Robot '''
        self.r = stretch_body.robot.Robot()
        print(f"[INFO] Robot connected: {self.r.startup()}")
        self.r.head.move_to('head_pan', 0)
        self.r.head.move_to('head_tilt', 0)
        self.r.arm.move_to(0.0)
        self.r.lift.move_to(0.5)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)
    
    def _init_yolo(self):
        ''' Initialize YOLO object detection model '''
        self.model = YOLO('yolo11n.pt')
        print("[INFO] YOLO model initialized")
    
    def _init_realsense_camera(self, cam):
        ''' Initialize RealSense camera. Default H-cam '''
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        try:
            if cam == "head":
                self.config.enable_device(self.head_cam_serial_number) # Enable head camera
            elif cam == "gripper":
                self.config.enable_device(self.gripper_cam_serial_number) # Enable gripper camera
            
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # Enable Depth stream
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Enable RGB stream
            
            self.pipeline.start(self.config) # Start streaming
            print('[INFO] RealSense camera initialized')
        except Exception as e:
            print("[ERROR]: Invalid camera name. Either 'head' / 'gripper'")
      
    def _get_cam_serial_numbers(self):
        ''' Get serial number of the camera peripherals '''
        ctx = rs.context()
        devices = ctx.query_devices()

        for device in devices:
            if device.get_info(rs.camera_info.name) == "Intel RealSense D435I":
                self.head_cam_serial_number = device.get_info(rs.camera_info.serial_number)
                print(f'[INFO] Head camera serial number: {self.head_cam_serial_number}')
            elif device.get_info(rs.camera_info.name) == "Intel RealSense D405":
                self.gripper_cam_serial_number = device.get_info(rs.camera_info.serial_number)
                print(f'[INFO] Gripper camera serial number: {self.gripper_cam_serial_number}')
            else:
                print('[ERROR] No device found')

    def _calculate_camera_lowering(self, pixel_offset_y):
        ''' Calculate how much to lower the camera in meters to center the object in the frame '''
        real_world_height_per_pixel = (2 * self.gripper_cam_height * math.tan(self.gripper_cam_vertical_FOV_radians / 2)) / self.gripper_image_height
        real_world_displacement = pixel_offset_y * real_world_height_per_pixel
        return real_world_displacement

    def _get_cam_frames(self):
        ''' Get camera frames '''
        frames = self.pipeline.wait_for_frames()  # Wait for frames
        depth_frame = frames.get_depth_frame()  # Depth data
        color_frame = frames.get_color_frame()  # Color data
        return depth_frame, color_frame
    
    def _frames_to_nparray(self, depth_frame, color_frame):
        ''' Convert frames to numpy arrays '''
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def _yolo_detection(self, image):
        ''' Detect object using YOLO model '''
        return self.model.predict(image, conf=0.5, verbose=False)  # 50% confidence

    def _draw_bbox(self, results, image):
        ''' Draw bounding box around the detected object'''
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label == self.target_object:
                self.object_detected = True
                self.detected_frames += 1  # If object detected, increment detected frames

                self.obj_center_x = x1 + (x2 - x1) / 2
                self.obj_center_y = y1 + (y2 - y1) / 2
                self.obj_coords = [x1, x2, y1, y2]

                cv.rectangle(image, (x1, y1), (x2, y2), (238, 130, 238), 2)
                cv.putText(image, f'{label} {conf:.2f}', (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (238, 130, 238), 2)
                
    def _orient_robot(self, current_pan, target_pan, target_tilt):
        ''' Orient the robot '''
        self.r.base.rotate_by(current_pan + target_pan)
        self.r.head.move_to('head_pan', 0.0)
        self.r.head.move_to('head_tilt', target_tilt)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)

    def _translate_robot(self, distance):
        ''' Move robot forward '''
        self.r.base.translate_by(distance)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)

    def _init_gripper_arm(self):
        ''' Initialize arm and gripper for object pickup '''
        self.r.base.rotate_by(1.65) # slightly more than 90 deg for pick up
        self.r.arm.move_to(0.0)
        self.r.lift.move_to(self.gripper_cam_height)
        self.r.push_command()
        self.r.wait_command()
        
        self.r.end_of_arm.move_to('wrist_yaw', 1.0)
        self.r.end_of_arm.move_to('wrist_pitch', 0.0)
        self._open_gripper(100)
        time.sleep(0.1)
    
    def _init_gripper_camera(self):
        ''' Stop other pipelines and enable G-cam '''
        self.pipeline.stop() # Stop H-cam pipeline
        time.sleep(0.1)
        print('[INFO] Head cam pipeline stopped')
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable G-cam
        self.config.enable_device(self.gripper_cam_serial_number)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.pipeline.start(self.config)
        time.sleep(0.1)
        print('[INFO] Gripper cam pipeline started')

    def _lower_arm(self, distance):
        ''' Lower lift '''
        print('[INFO] Lowering arm...')
        self.r.lift.move_to(distance)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)

    def _extend_arm(self, distance):
        ''' Extends arm '''
        print('[INFO] Extending arm...')
        self.r.arm.move_to(distance)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)

    def _open_gripper(self, amount):
        ''' Open gripper '''        
        print('[INFO] Opening gripper...')
        self.r.end_of_arm.move_to('stretch_gripper', amount)
        self.r.wait_command()
        time.sleep(0.1)

    def _close_gripper(self, amount):
        ''' Close gripper '''
        print('[INFO] Closing gripper...')
        self.r.end_of_arm.move_to('stretch_gripper', amount)
        self.r.wait_command()
        time.sleep(0.1)

    def _retract_arm(self, distance):
        ''' Retracts arm '''
        print('[INFO] Retracting arm...')
        self.r.arm.move_to(distance)
        self.r.push_command()
        self.r.wait_command()
        time.sleep(0.1)
    
    def _return_to_home(self):
        print('[INFO] Returning to home...')
        if self.search_direction == 'Forward':
            self.r.base.rotate_by(-0.3)
            self.r.end_of_arm.move_to('wrist_yaw', 0.0)
            self.r.base.push_command()
            self.r.wait_command()
            self._translate_robot((self.current_pos + 0.1) * -1)
        elif self.search_direction == 'Backward':
            self.r.base.rotate_by(-0.3)
            self.r.end_of_arm.move_to('wrist_yaw', 3.0)
            self.r.base.push_command()
            self.r.wait_command()
            self._translate_robot(self.max_distance - self.current_pos + 0.1)
        self.r.push_command()
        self.r.wait_command()

    def _grip_object(self):
        ''' Object manipulation '''
        try:
            self._init_gripper_camera()
            self._init_gripper_arm()

            while True:
                self.depth_frame, self.color_frame = self._get_cam_frames()

                if not self.depth_frame or not self.color_frame:
                    continue

                # Convert frames to numpy arrays
                self.depth_image, self.color_image = self._frames_to_nparray(self.depth_frame, self.color_frame)
                
                # Don't need to rotate frames for G-cam

                self.results = self._yolo_detection(self.color_image) # Detect object in the frame
                self.object_detected = False
                self._draw_bbox(self.results, self.color_image) # Draw bbox

                # Draw a circle at the center of the frame
                cv.circle(self.color_image, (320, 240), 5, (0, 0, 255), -1)
                cv.circle(self.color_image, (int(self.obj_center_x), int(self.obj_center_y)), 5, (0, 255, 255), -1)

                cv.imshow('Gripper cam live', self.color_image)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                if not self.gripper_cam_lowered:
                    # Calculate the vertical displacement (offset) from the center of the frame (320, 240)
                    frame_center_y = 240  # Center of the frame in y direction (half of 480)
                    pixel_offset_y = self.obj_center_y - frame_center_y  # How far the object is from the center

                    # Calculate how much to lower the camera (in meters)
                    lowering_amount = self._calculate_camera_lowering(pixel_offset_y)
                    lowering_amount = round(lowering_amount, 2) - 0.05 #XXX
                    print(f"[INFO] Arm need to be lowered by: {lowering_amount:.3f} meters")
                    
                    if self.target_object != "cell phone":
                        self._lower_arm(self.gripper_cam_height - lowering_amount)
                        self.gripper_cam_lowered = True
                        self._extend_arm(0.4) 
                        self._close_gripper(20)
                        self._retract_arm(0.0)
                        break
                    else:
                        print("I found your cell phone, but unfortunately I can't pick it.")
                        pyttsx3.speak("I found your cell phone, but unfortunately I can't pick it.")
                        time.sleep(1.0)
                        break
        except Exception as e:
            print('[ERROR] with gripping object: %s' %e)   

    def _detect_object(self):
        ''' Scans around and detects the object '''
        try:
            while (self.current_pos <= self.max_distance):
                self.depth_frame, self.color_frame = self._get_cam_frames()

                if not self.depth_frame or not self.color_frame:
                    continue

                # Convert frames to numpy arrays
                self.depth_image, self.color_image = self._frames_to_nparray(self.depth_frame, self.color_frame)

                # Rotate H-cam images
                self.depth_image_rotated = cv.rotate(self.depth_image, cv.ROTATE_90_CLOCKWISE)
                self.color_image_rotated = cv.rotate(self.color_image, cv.ROTATE_90_CLOCKWISE)
                
                # Shape and center of the image
                self.rotated_image_w, self.rotated_image_h = self.color_image_rotated.shape[:2]
                self.center_x, self.center_y = self.rotated_image_h // 2, self.rotated_image_w // 2
                cv.circle(self.color_image_rotated, (self.center_x, self.center_y), 5, (0, 0, 255), -1)  # Red dot at the center for reference

                # Detect object by passing the frame to YOLO
                self.results = self._yolo_detection(self.color_image_rotated)

                self.object_detected = False
                self.obj_center_x = self.obj_center_y = None

                # Draw bounding box around the detected object
                self._draw_bbox(self.results, self.color_image_rotated)
                
                cv.imshow('Head cam live', self.color_image_rotated)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                # Object confirmation
                if not self.object_detected:
                    self.detected_frames = 0
                    self.target_confirmed = False
                
                if self.detected_frames >= self.required_frames:
                    # Center the robot towards the object, if not
                    if not self.robot_centered_at_object:
                        self.target_confirmed = True
                        print('[INFO] Object confirmed')
                        print(f'[INFO] Object center: ({self.obj_center_y}, {self.obj_center_x})')
                        print(f'[INFO] Object coordinates: {self.obj_coords}')
                    
                        # Normalize the object center
                        x_center_norm = self.obj_center_x / self.rotated_image_h
                        y_center_norm = self.obj_center_y / self.rotated_image_w
                        print(f'[INFO] Center normalized: {x_center_norm:.3f}, {y_center_norm:.3f}')

                        # Calculate target pan and tilt to orient the robot towards the object
                        target_pan = (0.5 - x_center_norm) * self.head_FOV_tilt # FOV_tilt, since the image is rotated
                        target_tilt = ((0.5 - y_center_norm) * self.head_FOV_pan) + self.start_tilt
                        print(f'[INFO] Current pan: {self.current_pan:.3f}')
                        print(f'[INFO] Target pan: {target_pan:.3f}')
                        print(f'[INFO] Current tilt: {self.current_tilt:.3f}')
                        print(f'[INFO] Target tilt: {target_tilt:.3f}')

                        self._orient_robot(self.current_pan, target_pan, target_tilt)
                        self.robot_centered_at_object = True
                    else:
                        # Once centered, use depth information to move Stretch close to the object
                        depth_point = self.depth_image_rotated[int(self.obj_center_y), int(self.obj_center_x)]
                        # depth_region = np.mean(self.depth_image_rotated[self.obj_coords[2]:self.obj_coords[3],
                                                                        # self.obj_coords[0]:self.obj_coords[1]]) # Avg dist inside obj bbox
                        
                        if depth_point.size > 0 and np.all(depth_point != 0):  # Ensure region is not empty or invalid
                            dist_to_object = np.mean(depth_point) / 1000.0  # Convert mm to meters
                            print(f"[INFO] Distance to object: {dist_to_object:.3f} meters")

                            self.depth_buffer.append(dist_to_object)
                            # To stabilize depth fluctuations. Avg depth over 5 frames then move forward
                            if len(self.depth_buffer) >= self.buffer_size:
                                avg_depth = np.mean(self.depth_buffer)
                                print(f"[INFO] Average distance to object: {avg_depth:.3f}")

                                horizontal_dist = math.sqrt(avg_depth**2 - (self.head_cam_height - self.table_height)**2) # sqrt(hypotenuse^2 - lift height^2) = ground forward dist
                                print(f'[INFO] Horizontal distance: {horizontal_dist:.3f}')

                                if horizontal_dist > self.arm_length + self.SAFE_DISTANCE:
                                    self._translate_robot(0.1)
                                    self.dist_to_object += 0.1
                                else:
                                    print('[INFO] Object at reach')
                                    print(f'[INFO] Moved {self.dist_to_object} m')
                                    cv.destroyAllWindows()
                                    self._grip_object()
                                    self._return_to_home()
                                    self.object_retrieved = True
                                    print('[INFO] Reached home')
                                    break
                                    # TODO: Return to home
                                
                                self.depth_buffer = []
                        else:
                            print('[ERROR] Depth region empty or invalid')
                elif not self.target_confirmed:
                    self.current_pos += abs(self.pos_step_size)  # Reduce ~2° from current pan
                    self.r.base.translate_by(self.pos_step_size)  # Keep scanning
                    self.r.push_command()
                    self.r.wait_command()
                    time.sleep(0.1)
            self.boundary_reached = True
            print('[INFO] Boundary reached')
        
        except Exception as e:
            print('[ERROR] with detecting object: %s' %e)   


    def _reset(self):
        self.current_pos = 0.0
        self.pos_step_size = -0.1
        self.current_pan = self.backward_direction_pan
        self.r.head.move_to('head_pan', self.backward_direction_pan)
        self.r.wait_command()
        time.sleep(0.1)


    def run(self):
        ''' Run object detection and manipulation '''
        try:
            while True:
                if not self.boundary_reached and not self.object_retrieved:
                    self.search_direction = 'Forward'
                    self._detect_object()
                elif self.boundary_reached and not self.object_retrieved:
                    self._reset()
                    self.search_direction = 'Backward'
                    self._detect_object()
                    break
                else:
                    break
        except KeyboardInterrupt:
            print('[ERROR] User interrupted')
        except Exception as e:
            print('[ERROR] %s' %e)
        finally:
            self.pipeline.stop()
            cv.destroyAllWindows()
            self.r.stop()
            print('[INFO] Robot stopped') 


if __name__ == "__main__":
    target_object = input('Enter the object you want to find (cup/bottle/apple/cell phone): ')
    obj = ObjectDetectionManipulation(target_object)
    obj.run()
