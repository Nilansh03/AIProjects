from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import FPS
from mylib import config, thread
import numpy as np
import imutils
import time, dlib, cv2

t0 = time.time()

def run():
	CLASSES = ["person"]

	net = cv2.dnn.readNetFromCaffe('./mobilenet_ssd/MobileNetSSD_deploy.prototxt',
								   './mobilenet_ssd/MobileNetSSD_deploy.caffemodel')
	print("[INFO] Starting the video..")
	vs = cv2.VideoCapture('./videos/People Walking Inside Shopping Mall Stock Footage.mp4')
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=10, maxDistance=100)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either left or right
	totalFrames = 0
	totalRight = 0
	totalLeft = 0
	x = []
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:

		frame = vs.read()
		frame = frame[1]

		if './videos/People Walking Inside Shopping Mall Stock Footage.mp4' != None and frame is None:
			break

		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		rects = []

		if totalFrames % 30 == 0:

			trackers = []

			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > 0.4:
					# extract the index of the class label from the
					# detections list
					idx = int(0)
					print(idx)


					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:

				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))


		cv2.line(frame, (W // 2, 0), (W // 2, H ), (0, 0, 0), 3)

		objects = ct.update(rects)

		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the x-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'left' and positive for 'right')
				y = [c[0] for c in to.centroids]
				direction = centroid[0] - np.mean(y)
				to.centroids.append(centroid)
				if not to.counted:
					if direction < 0 and centroid[0] < W // 2:
						totalLeft += 1
						empty.append(totalLeft)
						to.counted = True
					elif direction > 0 and centroid[0] > W // 2:
						totalRight += 1
						empty1.append(totalRight)
						to.counted = True

					x = []
					x.append(len(empty1) - len(empty))
			# print("Total people inside:", x)

			trackableObjects[objectID] = to
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		info = [
			("Left-Right", totalRight),
			# 	("Total", totalLeft+totalRight)
		]



                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)



		cv2.imshow("Nilansh- ", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("Left-Right People {}".format(totalRight))
	print("Total:{} ".format(objectID))

	# issue 15
	if config.Thread:
		vs.release()
	# close any open windows
	cv2.destroyAllWindows()
run()
