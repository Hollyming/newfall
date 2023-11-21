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

        # fall detection module
        img_dep = cv2.resize(depth, (320, 240))
        img_ir = cv2.resize(ir, (320, 240))
        dep0 = depth.copy()
        bg_copy = depth.copy()

        if bgflag :
            dep_bg = img_dep
            ir_bg = img_ir
            cv2.imshow('bg',cv2.convertScaleAbs(img_dep, None, 1 / 16))
        else:
            tofcap.tof_cap(img_dep,img_ir)

            depnb = cv2.convertScaleAbs(img_dep, None, 1 / 16)
            depnb = cv2.merge([depnb] * 3)

            img_amp = cv2.convertScaleAbs(img_ir, None, 1)
            img_fall = cv2.convertScaleAbs(img_dep, None, 1 / 16)

            img_move = mog.apply(img_fall)
            img_move = cv2.morphologyEx(img_move, cv2.MORPH_OPEN, denoise_kernelo, iterations=2)
            img_move = cv2.morphologyEx(img_move, cv2.MORPH_CLOSE, denoise_kernelo, iterations=2)
            
            img_fall = cv2.merge([img_fall] * 3)
            img_fall = cv2.applyColorMap(img_fall, cv2.COLORMAP_RAINBOW)
            panl = np.ones_like(img_fall,dtype = 'uint8')*255
                


            # 去背景
            if dep_bg is not None:
                img_dep[np.abs(img_ir.copy() - ir_bg) < 40] = 0


            img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                                amp_th=50,  # 红外图
                                dmax=4000, dmin=500,  # 深度图
                                # cutx=40, cuty=10  # 图像四周区域
                                )
            img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernelo, iterations=2)
            img_dep[img_hand == 0] = 0
            depnb2 = cv2.convertScaleAbs(img_dep, None, 1 / 16)
            depnb2 = cv2.merge([depnb2] * 3)

            _, markers, stats, centroids = cv2.connectedComponentsWithStats(img_hand)
            area_th = 1500
            detflag = 0
            human_num  = 0

            min_d = 20000

            for i in range(1, len(stats)):
                center = centroids[i].astype('int')
                dd=transxy(center,dep0)[0][2]
                human_area = stats[i][4]*dd*dd

                if human_area > area_th:
                    human_num+=1
                    detflag = 1
                    hum_stat = stats[i]
                    cv2.rectangle(depnb, (hum_stat[0], hum_stat[1]),
                                (hum_stat[0] + hum_stat[2], hum_stat[1] + hum_stat[3]), color=[0, 255, 0],
                                thickness=2)
                    d_center = np.sum(np.fabs(last_center - center))
                    if min_d > d_center:
                        min_d = d_center
                        min_id = i
                    
            if detflag:
                hum_stat = stats[min_id]
                center = centroids[min_id].astype('int')
                last_center = center.copy()
                img_hand[markers != min_id] = 0
                img_dep[img_hand == 0] = 0
                # dc = np.average(img_dep[center[1] - 3:center[1] + 4, center[0] - 3:center[0] + 4])
                # img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                #                     dmax=dc + 500, dmin=dc - 500,  # 深度图
                #                     )

                
                cv2.rectangle(depnb, (hum_stat[0], hum_stat[1]),
                                (hum_stat[0] + hum_stat[2], hum_stat[1] + hum_stat[3]), color=[0, 0, 255],
                                thickness=2)

                cbk = hum_stat[3] / hum_stat[2]

                # 速度满足要求
                vy = spd.meanSpeed(center, dep0)[1]

                if vy > 0.8:
                    flag = 1

                if flag == 1 and cbk<1.3: #(hum_stat[3] < 120 or cbk < 0.85):
                    fall_num = fall_num + 1
                else:
                    fall_num = 0

                if flag == 1 and fall_num == 3:
                    isFallDown = 1
                    # 跌倒，这里加动作
                    #########################

                    # simpleaudio_play('falldown_ring.wav')

                    ##########################
                    fall_num = 0
                    flag = 0

                # 高度 角度要求
                # if hum_stat[3] > 180 or cbk > 1.8:
                if cbk>1.5:
                    fine_num = fine_num + 1
                else:
                    fine_num = 0

                if fine_num == 5:
                    flag = 0
                    fine_num = 0
                    isFallDown = 0

                if cbk < 0.7:
                    bednum+=1
                else:
                    bednum=0
                if bednum >=16:
                    bed_state =1
                    standup=0

                if cbk>1.2:
                    bedupnum += 1
                else:
                    bedupnum=0

                if bedupnum > 50:
                    bed_state = 0
                elif bedupnum > 5 and bed_state== 1:
                    bed_state = 2

            standup+=1
            if standup > 50:
                bed_state = 0

            cv2.putText(panl, "Fall Detect: ", (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
            cv2.putText(panl, "Move Detect:", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
            cv2.putText(panl, "Sleep State:", (20, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)

            if isFallDown == 1:
                cv2.putText(panl, "FALL DOWN!", (60, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                cv2.circle(panl, (40, 50),10, (0, 0, 255), -1)
            else:
                cv2.putText(panl, "FINE!", (60, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
                cv2.circle(panl, (40, 50), 10, (0, 255, 0), -1)

            if (img_move>0).sum()>1000:
                cv2.putText(panl, "Move.", (40, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
            else:
                cv2.putText(panl, "No Move.", (40, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

            if bed_state ==0 :
                cv2.putText(panl, "None.", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
            elif bed_state ==1 :
                cv2.putText(panl, "Sleeping...", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
            else :
                cv2.putText(panl, "Bed UP !", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 1)



            # img_fall = cv2.resize(img_fall, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            depnb = cv2.resize(depnb, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


            img_fall = np.concatenate((panl,img_fall),axis = 0)
            cv2.imshow("Status", img_fall)
            # cv2.imshow('nobackground', depnb)

            # visualize module
            vis_amp = cv2.convertScaleAbs(ir, None, 1/2).astype(np.uint8)
            # vis_dep = cv2.convertScaleAbs(depth, None, 1/16).astype(np.uint8)
            # vis_image = cv2.merge([vis_dep] * 3)
            vis_inf = cv2.merge([vis_amp] * 3)

            datum = op.Datum()
            datum.cvInputData = vis_inf
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))    
            output = datum.cvOutputData
            
            if datum.poseKeypoints is not None:
                skel = datum.poseKeypoints[0,:15,:2]
                bgWithSkel = draw_from_numpy(background, skel)
                cv2.imshow('fallDownDetection', bgWithSkel)
            else:
                cv2.imshow('fallDownDetection', background)


        k = cv2.waitKey(2) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            exit(0)
        elif k == ord('q'):
            break
        elif k == ord('t'):
            tflag ^= 1
        elif k == ord('b') and bgflag:
            bgflag = False
            tofcap = TofCapture(time.strftime("tof%m%d_%H%M%S"), 0, foldername)
            background = cv2.convertScaleAbs(bg_copy, None, 1/16).astype(np.uint8)
            background = cv2.merge([background] * 3)

    cv2.destroyAllWindows()
    dev.release()