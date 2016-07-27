pkg load image-acquisition;
cap = videoinput('v4l2', '/dev/video0');
set(cap, 'VideoFormat', 'RGB3');
start(cap);
img = getsnapshot(cap);
imshow(img)
stop(cap);