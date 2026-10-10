"""Phase 2 perception: static camera → tracker → Kalman filter → ``ObjectState``.

Plain words: look at the camera picture, find the object, guess where it is on the
table and how fast it moves. Hand that guess to the policy in the frozen shape:
``position, velocity, covariance, timestamp, valid``.

Files, in the order data flows:
  camera.py    pixel <-> table point (geometry only)
  track.py     find the object blob in each picture (pixels only)
  kalman.py    smooth the table points into position + velocity
  estimator.py glue: runs the three above once per camera frame
"""
