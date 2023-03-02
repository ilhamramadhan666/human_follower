from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

import pyzed.sl as sl

cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

DETECT_AFTER = 100
algo = 'kcf'
output_size = (1280,720)
REALSENSE = True
TINY = 0
PLAY_FROM_FILE = 0
occlusion = False

def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def video_capture(frame_queue, darknet_image_queue):
    while(1):
        if zed.grab() == sl.ERROR_CODE.SUCCESS :
            #RGB IMAGE
            # Retrieve the left image in sl.Mat
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            # Convert to RGB
            frame_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)
        
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(image_ocv)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)

        #return image_ocv, img_for_detect
        key = cv2.waitKey(1)
        # print(box)
        if key & 0xFF == ord('q'):
            break
    zed.close()
    #cap.release()

def video_depth_capture():
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
        # RGB IMAGE
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        # Convert to RGB
        frame_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    return image_ocv, img_for_detect, depth_map

def inference(darknet_image_queue, detections_queue, fps_queue):
    while(1):
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, True)
        darknet.free_image(darknet_image)

def detection(img_for_detect):
    darknet_image = img_for_detect
    prev_time = time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))
    darknet.print_detections(detections, True)
    darknet.free_image(darknet_image)

    return detections, fps

def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    #video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while (1):
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            #if not args.dont_show:
            cv2.imshow('Inference', image)
            #if args.out_filename is not None:
            #    video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    #cap.release()
    #zed.close()
    #video.release()
    cv2.destroyAllWindows()

def draw(frame, detections, fps):
    random.seed(3)
    detections_adjusted = []
    if frame is not None:
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        #if not args.dont_show:
        cv2.imshow('Inference', image)
        #if args.out_filename is not None:
        #    video.write(image)
        if cv2.waitKey(fps) == 27:
            break
    #cap.release()
    #zed.close()
    #video.release()
    cv2.destroyAllWindows()

def get_iou(boxA, boxB):
	""" Find iou of detection and tracking boxes
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def get_control(bbox, depth,  Kp_l, Ki_l, Kd_l, Kp_a, Ki_a, Kd_a):
    x, y, w, h  = convert2relative(bbox)

    global i_error_l 
	global i_error_a
	global d_error_l
	global d_error_a

	twist = Twist()

	p_error_l = depth - 1500
	p_error_a = x
	i_error_l += p_error_l
	i_error_a += p_error_a
	curr_d_error_l = p_error_l - d_error_l
	curr_d_error_a = p_error_a - d_error_a

	linear = Kp_l*p_error_l + Ki_l*i_error_l + Kd_l*curr_d_error_l
	angular = Kp_a*p_error_a + Ki_a*i_error_a + Kd_a*curr_d_error_a
	print('linear: {} ,angular: {}  \n'.format(linear,angular))

	if linear > 0.3:
		linear = 0.3

	if angular > 0.3:
		angular = 0.3

	if linear < -0.3:
		linear = -0.3

	if angular < -0.3:
		angular = -0.3

def get_coordinates(box, x, y, x1, y1):
	""" Get co-ordinates of flaged person
	"""
	if len(box) == 0:
#		print('!!!!!!!!No person detected!!!!')
		return
	iou_scores = []
	for i in range(len(box)):
		iou_scores.append(get_iou(box[i],[x,y,x1,y1]))

	index = np.argmax(iou_scores)
#	print(iou_scores, ' ',box, ' ', x, y, x1, y1)

	if np.sum(iou_scores) == 0:
		# print('#'*20, 'No Match found', '#'*20)
		box = np.array(box)
		distance = np.power(((x+x1)/2 - np.array(box[:,0] + box[:,2])/2),2) + np.power(((y+y1)/2 - (box[:,1]+box[:,3])/2), 2)
		index = np.argmin(distance)

	x, y, w, h = box[index][0], box[index][1], (box[index][2]-box[index][0]), (box[index][3]-box[index][1])
	initBB = (x+w//2-50,y+h//3-50,100,100)

	return initBB, (x,y,x+w,y+h)

if __name__ == '__main__':
    # Set Queue
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    # Setting PID constants
    kp_l = .2
	kp_a = .2
	kd_l = 0
	kd_a = 0
    ki_l = 0
    ki_a = 0
    # Variables
    x,y,w,h = 0,0,0,0

    # Setup Zed Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    #init_params.camera_fps = 30  # Set fps at 30

    #open camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
    
    #set runtime params
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    
    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    #declare sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.F32_C1)
    #image_depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    config_file = "./cfg/yolov7.cfg"
    data_file = "./cfg/coco.data"
    weights = "yolov7.weights"
    network, class_names, class_colors = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=1
        )
    
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    #input_path = str2int(args.input)
    #cap = cv2.VideoCapture(input_path)
    
    # For saving video, not used
    video_width = int(image_size.width)
    video_height = int(image_size.height)

    # Threading
    #Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    #Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    #Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

    # Initiate tracker
    if zed.grab() == sl.ERROR_CODE.SUCCESS :
        #RGB IMAGE
        # Retrieve the left image in sl.Mat
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        # Use get_data() to get the numpy array
        image_ocv = image_zed.get_data()
        # Convert to RGB
        frame_rgb = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2RGB)
        
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                               interpolation=cv2.INTER_LINEAR)
    H, W = frame.shape[:2]
    initBB = cv2.selectROI('Frame', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR), fromCenter=False)
    tracker = OPENCV_OBJECT_TRACKERS[algo]()
	tracker.init(frame_resized, initBB)
    # Run
    while(1):
        try:
            frame_number+=1
			frame_ += 1
			
			image_ocv, img_for_detect, depth_map = video_depth_capture()

            if frame_number % DETECT_AFTER == (DETECT_AFTER-1) or not success :

				#img, yolo_box = yolo_output(frame.copy(),model,['person'], confidence, nms_thesh, CUDA, inp_dim)
				#cv2.imshow('yolo', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                detections, fps = detection(img_for_detect)
				label, confidence, bbox = detections 
                bbox = convert2original(img_for_detect, bbox)
				
                initBB, trueBB = get_coordinates(bbox, x, y, x+w, y+h)

				tracker = OPENCV_OBJECT_TRACKERS[algo]()
				tracker.init(img_for_detect, bbox)
				#fps = (frame_)//(time.time()-a)
				frame_ = 0
				#a = time.time()
            success, box = tracker.update(frame)

            if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

				calc_x, calc_z = (x+w/2), depth_map(x+w//2, y+h//2)
				# print('distance: {:2.2f}'.format(calc_z))
				#twist = get_controls(calc_x, calc_z, 1/5, 0, 0.1,-1/500, 0, 0)

            info = [("Tracker", algo),("Success", "Yes" if success else "No"),("FPS", "{:.2f}".format(fps)),]

			for (i, (k, v) ) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            #cv2.imshow("Frame", cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            draw(img_for_detect, detections, fps)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        except NameError:
            pass
        except Exception as e:
            print(e)
    zed.close()