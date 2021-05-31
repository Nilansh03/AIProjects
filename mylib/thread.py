import cv2, threading, queue

class ThreadingClass:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
  def _reader(self):
    while True:
      ret, frame = self.cap.read() # read the frames and ---
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except queue.Empty:
          pass
      self.q.put(frame) # --- store them in a queue (instead of the buffer)

  def read(self):
    return self.q.get() # fetch frames from the queue one by one

  def release(self):
    return self.cap.release() # release the hw resource
