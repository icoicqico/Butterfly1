import tensorflow as tf
import cv2
import numpy as np
import cupy as cp
#load the trained model, I fine tune the pretrained model, perform transfer learning by training the butterfly class.
detect_fn = tf.saved_model.load('butterfly_out/saved_model')
# load the butterfly video
cap = cv2.VideoCapture("in.mp4")
#label list, we only have one class butterfly
category_index = {
    1: {'id': 1, 'name': 'butterfly'},}
# the function to draw the bounding boxes
init_opt = False
def check_movement(flow,frame):
	#1 % of the frame height or width as boundary
	imw_b = int(frame.shape[1]*0.04)
	imh_b = int(frame.shape[0]*0.04)
	# totally moevement threshold
	threshold = 450
	boundary_1 = (flow[imh_b:,:,2])>0
	boundary_2 = (flow[:imh_b,:,2])>0
	boundary_3 = (flow[:,imw_b:,2])>0
	boundary_4 = (flow[:,:imw_b,2])>0
	#print(boundary_1)
	#print(cp.sum(boundary_2))
	if cp.sum(boundary_1) > threshold or cp.sum(boundary_2) > threshold or cp.sum(boundary_3) > threshold or cp.sum(boundary_4) > threshold:
		return True
	else:
		return False

def draw_box(detections,frame):
	# count the number of butterfly detected
	counts = 0
	# get the shape, since the tensorflow output is normalized so we have to multiply by the image shape
	imw = frame.shape[1]
	imh = frame.shape[0]
	# get the result pass the 0.5 score threshold
	idx = detections['detection_scores'][0].numpy() >= 0.75
	# get the result pass the 0.5 score threshold
	boxes = detections['detection_boxes'][0].numpy()[idx]
	#print(detections['detection_boxes'])
	#tracker
	near_bound = False
	print(boxes)
	for box in boxes:
		imw_b = int(frame.shape[1]*0.04)
		imh_b = int(frame.shape[0]*0.04)
		if (imw*box[1])<imw_b or (imw*box[3])>(imw -imw_b) or (imh*box[0])<imh_b or imh*box[2]>(imh -imh_b):
			near_bound = True
		
		
			
		# butterfly count
		counts+=1
		# draw the bounding box
		output = cv2.rectangle(frame,(int(imw*box[1]),int(imh*box[0])),(int(imw*box[3]),int(imh*box[2])),(0,0,255),1)
	# return result frame and the counts
	return output, counts, near_bound
skip_count = 0
prev_counts = 0
in_bound = 0
out_bound = 0
pp_counts = 0
reset_move = 0
wait_frame = 0
check_bound_count = False
out_init = False
while True:
	ret, frame = cap.read()
	skip_count += 1
	if skip_count % 2 ==0:
		skip_count == 0
		continue
	if ret != True:
		continue
	if not init_opt:
		draw_flow = np.zeros_like(frame)
		draw_flow[...,1] = 255
	ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ref_gray_gpu = cv2.cuda_GpuMat(ref_gray)
	if not init_opt:
		fg=ref_gray
		fg_gpu = ref_gray_gpu
	optical_flow_gpu = cv2.cuda.FarnebackOpticalFlow_create()
	flow_gpu_cuda = optical_flow_gpu.calc(fg_gpu, ref_gray_gpu, None)
	optical_flow = flow_gpu_cuda.download().astype(np.float32)	
	#optical_flow = cv2.calcOpticalFlowFarneback(fg, ref_gray, None, 0.5, 3, 5, 3, 3, 2, 2)
	magnitude, angle = cv2.cartToPolar(optical_flow[...,0],optical_flow[...,1])
	draw_flow[...,0] = (180/(np.pi/2))*angle
	draw_flow[...,2] = cv2.normalize(magnitude, None,0 , 255, cv2.NORM_MINMAX)
	#THRESHOLD, filter out small movement, like backgound noise
	idx_ = draw_flow[...,2]<=175
	draw_flow[...,2][idx_]=0
	#check if any movement at the boundary
	b_move = check_movement(draw_flow,frame)

	print(draw_flow.shape)
	draw_flow_show = cv2.cvtColor(draw_flow, cv2.COLOR_HSV2BGR)
	#print(magnitude, angle)
	# convert to tensor format
	input_tensor = np.expand_dims(frame, 0)
	# inference
	detections = detect_fn(input_tensor)
	# call the draw box function
	output, counts,near_bound = draw_box(detections,frame)
	# if some movement at the edge and there is any detected object start to check of the object counts increase or decrease within 5-15 frames
	if b_move == True and near_bound == True:
		check_bound_count = True
	if check_bound_count:
		if wait_frame == 15:
			check_bound_count = False
			wait_frame = 0
		wait_frame+=1
		if counts > prev_counts and wait_frame>5:
			in_bound+=1
			check_bound_count = False
			wait_frame = 0
		if counts < prev_counts and wait_frame>5:
			out_bound+=1
			check_bound_count = False
			wait_frame = 0
	# write the butterfly counts in frame
	output = cv2.putText(output, 'Butterfly in frame: '+str(counts), (20,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
	output = cv2.putText(output, 'in: '+str(in_bound) + "   out:  "+str(out_bound), (frame.shape[1]-450,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
	# show the real time result
	if init_opt:
		fg=ref_gray
		fg_gpu = ref_gray_gpu
	if not init_opt:
		init_opt = True
	#update previous counr number	
	pp_counts = prev_counts
	prev_counts = counts
	cv2.imshow("output",output)
	#initialize video writer for saving result
	if not out_init:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('output.avi', fourcc, 30.0, (output.shape[1],output.shape[0]))
		out_init = True
		
	out.write(output)
	cv2.imshow("flow",draw_flow_show)	
	cv2.waitKey(1)
