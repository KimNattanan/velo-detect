import cv2 as cv2
import numpy as np
import time
from flann import FLANN
from kalman_sort import Sort


def compute_flow(gray1, gray2):
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 5)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 5)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75,
                                        levels=3,
                                        winsize=5,
                                        iterations=3,
                                        poly_n=10,
                                        poly_sigma=1.2,
                                        flags=0)
    return flow


def draw_flow(gray, flow, step=16):
    h, w = gray.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def get_flow_viz(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((7, 7))):
    motion_mask = np.uint8(flow_mag > motion_thresh) * 255

    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return motion_mask


def get_contour_detections(mask, ang, angle_thresh=2, thresh=400):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_mask = np.zeros_like(mask)
    angle_thresh = angle_thresh * ang.std()
    detections = []

    for cnt in contours:
        if len(cnt) > 5:
            cnt = cv2.convexHull(cnt)  # Simplify contour by approximating with a convex hull
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
            flow_angle = ang[np.nonzero(temp_mask)]

            if (area > thresh) and (flow_angle.std() < angle_thresh):
                detections.append([x, y, x + w, y + h, area])

    return np.array(detections), contours


def non_max_suppression(boxes, scores, threshold):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            if iou > threshold:
                order.remove(j)
    return keep


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    tvecs = np.swapaxes(tvecs, 1, 2)
    return rvecs, tvecs, trash





ArucoMultiMatrix_path = "./ArucoMultiMatrix.npz"

flow_upd_time = 0.5  # seconds
aruco_sz = 6  # centimeters

velo_lb = 0
velo_ub = 1e18

mask_kernel = np.ones((7, 7), dtype=np.uint8)
bbox_thresh = 400
ellipse_thresh = 400
nms_thresh = 0.1  # non_max_suppression


calib_data_path = ArucoMultiMatrix_path
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_param = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)
aruco_peri = (aruco_sz * 4)

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


class INFO:
    def __init__(self, frame, t):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.t = t
        self.bboxes = np.array([])
        self.bboxes_keep = np.array([])
        # self.bboxes_speed = np.array([])
        # self.bboxes_feature_score = np.array([])
        self.bboxes_speed={}
        self.bboxes_feature_score={}
        self.contour_mask = np.zeros_like(self.gray)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,
                                                                        detectShadows=True)
        self.aruco_ratio=0
        self.tracker=Sort()
        self.ids=set()
        self.centroid={}

    def upd(self, frame, t):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = compute_flow(self.gray, gray)

        # Apply background subtractor
        fg_mask = self.background_subtractor.apply(frame)

        aruco_corners, _, _ = detector.detectMarkers(gray)
        self.aruco_ratio = 0
        if aruco_corners:
            self.aruco_ratio = aruco_peri / cv2.arcLength(aruco_corners[0], True) / (t - self.t) # cm/px.s
            # rVecs, tVecs, _ = my_estimatePoseSingleMarkers(aruco_corners, aruco_sz, cam_mat, dist_coef)
            # aruco_distance = np.sqrt(tVecs[0][0][2] ** 2 + tVecs[0][0][0] ** 2 + tVecs[0][0][1] ** 2)
            # print("aruco_distance =", aruco_distance)
        else: print("no aruco found")

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = get_motion_mask(mag, motion_thresh=motion_thresh, kernel=mask_kernel)

        # Combine motion mask with foreground mask
        combined_mask = cv2.bitwise_and(motion_mask, fg_mask)

        detections, contours = get_contour_detections(combined_mask, ang, thresh=bbox_thresh)

        if detections.shape[0]:
            bboxes = detections[:, :4]
            scores = detections[:, -1]
            keep = non_max_suppression(bboxes, scores, nms_thresh)
            # cur_boxes=bboxes[keep]
            # self.bboxes_speed=np.zeros((len(keep)))
            # self.bboxes_feature_score=np.zeros((len(keep)))
            # if(self.aruco_ratio!=0):
            #     for i, cur_box in enumerate(cur_boxes):
            #         best_iou = 0
            #         best_id = -1
            #         for j, prev_box in enumerate(self.bboxes):
            #             iou = compute_iou(cur_box, prev_box)
            #             if iou > best_iou:
            #                 best_iou = iou
            #                 best_id = j
            #         if best_iou >= match_iou_thresh:
            #             prev_box=self.bboxes[best_id]
            #             c0=((prev_box[0]+prev_box[2])/2, (prev_box[1]+prev_box[3])/2)
            #             c1=((cur_box[0]+cur_box[2])/2, (cur_box[1]+cur_box[3])/2)
            #             self.bboxes_speed[i]=np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2)*self.aruco_ratio
            #         self.bboxes_feature_score[i]=flann.getScore(gray[cur_box[1]:cur_box[3]+1, cur_box[0]:cur_box[2]+1])

            # self.bboxes = cur_boxes

            self.bboxes_keep=bboxes[keep]

            tracked=self.tracker.update(bboxes[keep]).astype(int)
            self.bboxes=[]
            s=set()
            for (x0,y0,x1,y1,id) in tracked:
                if(id != -1):
                    s.add(id)
                    (cx,cy)=((x0+x1)/2,(y0+y1)/2)
                    if(id in self.centroid):
                        (x,y)=self.centroid[id]
                        self.bboxes_speed[id]=np.sqrt( (x-cx)**2+(y-cy)**2 )*self.aruco_ratio
                    else: self.bboxes_speed[id]=0
                    self.centroid[id]=(cx,cy)
                    self.bboxes_feature_score[id]=flann.getScore(gray[y0:y1+1, x0:x1+1])
                    self.bboxes.append([x0,y0,x1,y1,self.bboxes_speed[id],self.bboxes_feature_score[id]])
                else: self.bboxes.append([x0,y0,x1,y1,0,flann.getScore(gray[y0:y1+1, x0:x1+1])])
            
            id_clear=list(self.ids-s)
            for id in id_clear: del(self.centroid[id])
            self.ids=s

        else:
            self.bboxes = np.array([])
            self.bboxes_keep=np.array([])
            # self.bboxes_speed=np.array([])

        self.contour_mask = np.zeros_like(self.gray)
        cv2.drawContours(self.contour_mask, contours, -1, (255,), -1)
        self.gray = gray
        self.t = t

    def get_contour_mask(self):
        return self.contour_mask
    


test_aruco_sz=50
test_aruco_img=cv2.resize(cv2.imread("./aruco.png"),(test_aruco_sz,test_aruco_sz))

def cap_with_aruco(cap):
    ret,img=cap.read()
    img=cv2.resize(img,(640,480))
    img[:10+test_aruco_sz,:10+test_aruco_sz]=[255,255,255]
    img[5:5+test_aruco_sz,5:5+test_aruco_sz]=test_aruco_img
    return (ret,img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("./vid2.mp4")
    info = INFO(cap_with_aruco(cap)[1], time.time())
    window_h, window_w = info.gray.shape[:2]
    motion_thresh = np.c_[np.linspace(0.3, 1, window_h)].repeat(window_w, axis=-1)

    feature_path="./feature_dir"
    flann=FLANN(feature_path,0.6)

    while True:
        ret, cur = cap_with_aruco(cap)
        if not ret:
            break
        cur_t = time.time()
        frame1 = cur

        if cur_t - info.t >= flow_upd_time:
            info.upd(cur, cur_t)

        contour_mask_bgr = cv2.cvtColor(info.get_contour_mask(), cv2.COLOR_GRAY2BGR)
        for (x0, y0, x1, y1) in info.bboxes_keep:
            cv2.rectangle(frame1, (x0, y0), (x1, y1), (0, 0, 200), 2)
        for (x0, y0, x1, y1, speed, feature_score) in info.bboxes:
            cv2.rectangle(frame1, (x0, y0), (x1, y1), (0, 200, 200), 2)
            cv2.putText(frame1,"score: {0}".format(feature_score),(x0-50,y0-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(frame1,"speed: {:.2f}".format(speed),(x0-50,y0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.rectangle(contour_mask_bgr, (x0, y0), (x1, y1), (0, 255, 0), 3)
            cv2.putText(contour_mask_bgr,"score: {0}".format(feature_score),(x0-50,y0-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(contour_mask_bgr,"speed: {:.2f}".format(speed),(x0-50,y0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # Fit ellipses to the contours and get major and minor axes
        contours = cv2.findContours(info.get_contour_mask(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for i, cnt in enumerate(contours):
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                # if(3.14159265*ellipse[1][0]*ellipse[1][1]<=ellipse_thresh): continue
                if(ellipse[1][0]*ellipse[1][1]<=bbox_thresh): continue
                
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # Validate ellipse dimensions
                    cv2.ellipse(contour_mask_bgr, ellipse, (0, 255, 255), 2)
                    cv2.ellipse(frame1, ellipse, (0, 255, 255), 2)
                    center, axes, angle = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    # print("Major Axis:", major_axis)
                    # print("Minor Axis:", minor_axis)

                    # Draw number
                    # cv2.putText(contour_mask_bgr, str(i + 1), (int(center[0]), int(center[1])),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # cv2.putText(frame1, str(i + 1), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #             (255, 255, 255), 2)

        stacked_image = np.hstack((frame1, contour_mask_bgr))
        cv2.imshow("stacked_image", stacked_image)
        # cv2.imshow("Gray Video", info.gray)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()