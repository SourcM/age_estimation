import numpy as np
import cv2
import datetime


class CenterFace(object):
    def __init__(self, landmarks=True):
        self.landmarks = landmarks
        if self.landmarks:
            self.net = cv2.dnn.readNetFromONNX('/content/drive/MyDrive/ACS/crop_faces/centerface_640_640.onnx') #change this please
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, height, width, threshold=0.5):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        begin = datetime.datetime.now()
        if self.landmarks:
            heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
        else:
            heatmap, scale, offset = self.net.forward(["535", "536", "537"])
        end = datetime.datetime.now()
        # print("cpu times = ", end - begin)
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode_fast(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode_fast(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep
    
    def decode_fast(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        sz = size
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)

        #numpy computation
        s0, s1 = np.exp(scale0[c0, c1]) * 4, np.exp(scale1[c0, c1]) * 4
        o0, o1 = offset0[c0, c1], offset1[c0, c1]
        s = heatmap[c0, c1]
        x1, y1 = np.maximum(0, (c1 + o1 + 0.5) * 4 - s1 / 2), np.maximum(0, (c0 + o0 + 0.5) * 4 - s0 / 2)
        x1, y1 = np.minimum(x1, sz[1]), np.minimum(y1, sz[0])
        boxes = np.vstack((x1, y1, np.minimum(x1 + s1, sz[1]), np.minimum(y1 + s0, sz[0]), s)).T
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = self.faster_nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :].copy()

        #landmarks
        lx=np.array([1, 3, 5, 7, 9]) 
        ly = np.array([0, 2, 4, 6, 8])
        landmark_o =landmark[0, :, :, :].copy()

        landmark_o_x = landmark_o[lx,:,:].copy()
        landmark_o_x = landmark_o_x[:, c0, c1] * s1 + x1
        landmark_o_x = landmark_o_x.T

        landmark_o_y = landmark_o[ly,:,:].copy()
        landmark_o_y = landmark_o_y[:, c0, c1] * s0 + y1
        landmark_o_y = landmark_o_y.T

        row_a, col_a = np.shape(landmark_o_x)
        row_b, col_b = np.shape(landmark_o_y)
        all_landmarks = np.ravel([landmark_o_x.T, landmark_o_y.T], 'F').reshape(row_a, col_a+col_b)

        lms = all_landmarks[keep, :].copy()

        return boxes, lms

    def faster_nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return keep

def make_image_square(in_img):
    img = cv2.imread(in_img)
    #get size
    height, width, channels = img.shape
    # print (in_img,height, width, channels)
    # Create a black image
    x = height if height > width else width
    y = height if height > width else width
    square= np.zeros((x,y,3), np.uint8)
    #
    #This does the job
    #
    square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
    
    return  cv2.resize(square, (640, 640))