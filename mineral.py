import cv2

def FrameCapture():
	
	vidObj = cv2.VideoCapture(0)
	count = 0
	success = 1

	while success:
		success, image = vidObj.read()
		cv2.imwrite("frame%d.jpg" % count, image)

		count += 1

if __name__ == '__main__':
	FrameCapture()
