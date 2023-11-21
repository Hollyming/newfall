from openpose import pyopenpose as op
from sdk import *
from clor import *

def draw_from_numpy(img_ori, skel):
    img = img_ori.copy()
    # print(data.shape)
    cnt = 0
    pairs = [(1, 8), (1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
             (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    body_parts = []
    for kk in range(len(skel)):
        k = skel[kk]
        body_part = (int(k[0]), int(k[1]))
        if body_part != (0, 0):
            cv2.circle(img, body_part, 4, (0, 0, 255), 4)
        body_parts.append(body_part)
    for pair in pairs:
        p1 = body_parts[pair[0]]
        p2 = body_parts[pair[1]]
        if p1 != (0, 0) and p2 != (0, 0):
            cv2.line(img, p1, p2, (0, 0, 255), 4)
    return img


if __name__ == "__main__":
    # openpose Parameter
    params = dict()
    params["model_folder"] = "openpose/models/"
    params["net_resolution"] = "480x-1"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # algorithm parameters
    flag = 0
    fall_num = 0
    fine_num = 0
    isFallDown = 0
    tflag = 0
    bed_state = 0
    bedupnum = 0
    bednum=0
    standup=0

    spd = CalMeanSpeed(rx= -30)
    foldername=r'bedupData'
    bgflag=True
    last_center = np.array([0,0])
    denoise_kernelo = np.ones((3, 3), np.uint8)
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # set tof initial parameters
    tof_cam_idx = 0
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        # print(graph.get_input_devices())  # list of camera device
        tof_cam_idx = graph.get_input_devices().index("DepEye USB Video")
        print("-> find DepEye USB cam: index=%d" % tof_cam_idx)
    except ImportError:
        print("-> Use default cam idx=%d" % tof_cam_idx)
    except ValueError:
        print("-> No DepEye USB cam detected")
        sys.exit(255)

    if platform.system().lower() == 'windows':
        dev = cv2.VideoCapture(tof_cam_idx, cv2.CAP_DSHOW)  #
    else:
        dev = cv2.VideoCapture(tof_cam_idx)  #

    switch_depth_ir_frame(dev)
    isp_dev_set_fps(dev, 30)
    ts_pinfo = time.time()

    while (1):
        # get raw frame
        ret, raw_frame = dev.read()
        if not ret:
            time.sleep(0.1)
            continue

        dep_ir_frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape(480, 1280)
        depth, ir = dep_ir_frame[:, 0::2], dep_ir_frame[:, 1::2]

        # visualize module
        vis_amp = cv2.convertScaleAbs(ir, None, 1/2).astype(np.uint8)
        vis_dep = cv2.convertScaleAbs(depth, None, 1/16).astype(np.uint8)
        vis_image = cv2.merge([vis_dep] * 3)
        vis_inf = cv2.merge([vis_amp] * 3)

        datum = op.Datum()
        datum.cvInputData = vis_inf
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))    
        output = datum.cvOutputData
        
        vis_image = vis_inf
        if datum.poseKeypoints is not None:
            skel = datum.poseKeypoints[0,:15,:2]
            vis_image = draw_from_numpy(vis_image, skel)

        print(np.mean(vis_image))
        cv2.imshow('fallDownDetection', vis_image)

        k = cv2.waitKey(2) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            exit(0)
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()
    dev.release()