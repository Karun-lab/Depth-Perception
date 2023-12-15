
import copy
import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

import tkinter as tk
import tkinter.simpledialog as simpledialog

def debug():
    for i in range(len(relative_value_list)):
        print("point ",i,"=", calibration_p_list[i])
        print("  relative_value =", relative_value_list[i])
        print("  absolute_value =", absolute_value_list[i])
    if mouse_point is not None and a is not None and b is not None:
        print("a = ", a)
        print("b = ", b)
        print("mouse point = ",mouse_point)
        print("  relative_value = ",depth_map[mouse_point[1],mouse_point[0]])  
        print("  absolute_balue = ",(depth_map[mouse_point[1],mouse_point[0]] * a) + b,"\n")

def relative_value_update():
    global relative_value_list, calibration_p_list
    for i in range(len(relative_value_list)):
        relative_value_list[i] = depth_map[calibration_p_list[i][1],calibration_p_list[i][0]]

def mouse_callback(event, x, y, flags, param):
    # Global variables for holding mouse coordinates, relative distance values, absolute distance values, and calibration coordinates
    global mouse_point
    global relative_value_list, absolute_value_list, calibration_p_list
    global depth_map

    mouse_point = [x, y]

    # left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Enter actual measurement value
        input_data = simpledialog.askstring(
            "",
            "Enter actual measurement value (cm)",
        )
        try:
            absolute_value = int(float(input_data))

            # absolute distance value
            absolute_value_list.append(absolute_value)
            # distance relative value
            relative_value_list.append(depth_map[y][x])
            # calibration coordinates
            calibration_p_list.append([x, y])
        except:
            # non-numeric
            pass


def run_inference(interpreter, image):
    image_width, image_height = image.shape[1], image.shape[0]

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    inputHeight, inputWidth, channels, = input_shape[1], input_shape[2], input_shape[3]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
# and 256 x 256 pixels for the back model
    img_input = cv2.resize(img, (inputWidth,inputHeight),interpolation = cv2.INTER_CUBIC).astype(np.float32)

# Scale input pixel values to -1 to 1
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_input = ((img_input/ 255.0 - mean) / std).astype(np.float32)
    img_input = img_input[np.newaxis,:,:,:]      

    # 推論
#    input_details = interpreter.get_input_details()
#    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])

    
# Remove extra dimensions and resize to input image size
    result = np.squeeze(result)
    result_depth_map = cv2.resize(result, (image_width, image_height))

    return result_depth_map


def linear_approximation(x, y):
    # Find the coefficients a and b of the linear function (y = ax + b) using the least squares method




    a = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - a * x[0]

    n = len(x)
    a1 = ((np.dot(x, y) - y.sum() * x.sum() / n) /
         ((x**2).sum() - x.sum()**2 / n))
    b2 = (y.sum() - a * x.sum()) / n

    return a, b


def draw_info(
    image,
    depth_map_,
    elapsed_time,
    mouse_point_,
    relative_value_list_,
    absolute_value_list_,
    calibration_p_list_,
    a=None,
    b=None,
):
    # Create drawing frame
    rgb_frame = copy.deepcopy(image)
    depth_frame = copy.deepcopy(depth_map_)

   
    # Adjust value range for pseudocolor
    depth_max = depth_frame.max()
    depth_frame = ((depth_frame / depth_max) * 255).astype(np.uint8)
    depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_TURBO)

    # Draw inferred value on mouse pointer
    if mouse_point_ is not None:
        point_x = mouse_point_[0]
        point_y = mouse_point_[1]

        
        # If calibrated, draw in cm notation
        if a is not None and b is not None:

            display_d = "{0:.1f}".format(
                ((depth_map_[point_y][point_x] * a) + b)) + "cm"

            # RGB image
            cv2.circle(rgb_frame, (point_x, point_y),
                       3, (0, 255, 0),
                       thickness=1)
            cv2.putText(rgb_frame, display_d, (point_x, point_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Depth image
            cv2.circle(depth_frame, (point_x, point_y),
                       3, (255, 255, 255),
                       thickness=1)
            cv2.putText(depth_frame, display_d, (point_x, point_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

    
    # Calibration point drawing
    for index, calibration_p in enumerate(calibration_p_list_):
        point_x = calibration_p[0]
        point_y = calibration_p[1]

        # RGB image
        cv2.circle(rgb_frame, (point_x, point_y), 3, (0, 255, 0), thickness=1)
        cv2.putText(
            rgb_frame, "{0:.1f}".format(relative_value_list_[index]) + " : " +
            str(absolute_value_list_[index]) + "cm", (point_x, point_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Depth image
        cv2.circle(depth_frame, (point_x, point_y),
                   3, (255, 255, 255),
                   thickness=1)
        cv2.putText(
            depth_frame, "{0:.1f}".format(relative_value_list_[index]) +
            " : " + str(absolute_value_list_[index]) + "cm",
            (point_x, point_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1, cv2.LINE_AA)

   # Inference time drawing
    # RGB image
    cv2.putText(rgb_frame,
                "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                cv2.LINE_AA)
    # Depth image
    cv2.putText(depth_frame,
                "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)

    return rgb_frame, depth_frame


if __name__ == "__main__":
    
    # Global variables for holding mouse coordinates, relative distance values, absolute distance values, and calibration coordinates
    global mouse_point
    global relative_value_list, absolute_value_list, calibration_p_list
    global depth_map
    mouse_point = None
    relative_value_list, absolute_value_list, calibration_p_list = [], [], []
    depth_map = None

    
    # model load
    interpreter = Interpreter("lite-model_midas_v2_1_small_1_lite_1.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    
    # Camera preparation
    cap = cv2.VideoCapture(0)

    #Tkinter initialization
    root = tk.Tk()
    root.withdraw()

    # Register callback for OpenCV window initialization and mouse operation
    rgb_window_name = 'rgb'
    cv2.namedWindow(rgb_window_name)
    cv2.setMouseCallback(rgb_window_name, mouse_callback)

    depth_window_name = 'depth'
    cv2.namedWindow(depth_window_name)
    cv2.setMouseCallback(depth_window_name, mouse_callback)

    while True:
        start_time = time.time()

        # camera capture
        ret, frame = cap.read()
        if not ret:
            continue

        # Depth estimation
        depth_map = run_inference(interpreter, frame)

        # Linear approximation of relative distance and absolute distance using least squares method
        a, b = None, None
        if len(calibration_p_list) >= 2:
            relative_value_update()
            a, b = linear_approximation(np.array(relative_value_list),
                                        np.array(absolute_value_list))

        elapsed_time = time.time() - start_time

        # Information drawing
        rgb_frame, depth_frame = draw_info(
            frame,
            depth_map,
            elapsed_time,
            mouse_point,
            relative_value_list,
            absolute_value_list,
            calibration_p_list,
            a,
            b,
        )

        debug()

        cv2.imshow(rgb_window_name, rgb_frame)
        cv2.imshow(depth_window_name, depth_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()