import argparse

import PIL
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *
import pickle

my_list = [1, 2, 3, 4, 5]

import os
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def test(result:list, output):
    # data_dir = r'E:\BaiduNetdiskDownload\CCPD2020\CCPD2020\ccpd_green\test'
    # img_list = os.listdir(data_dir)
    # img_list = [os.path.join(data_dir, img) for img in img_list if img.endswith('.jpg')]
    # total = len(img_list)
    # rate:72.39312824610468%
    correct = 0
    for i in range(len(result)):
        print(result[i][0][0])
        id = result[i][0][0].split('-')[-3].split('_')
        brands = result[i][1]
        car_brand = ''
        car_brand += provinces[int(id[0])]
        car_brand += alphabets[int(id[1])]
        car_brand += ads[int(id[2])]
        car_brand += ads[int(id[3])]
        car_brand += ads[int(id[4])]
        car_brand += ads[int(id[5])]
        car_brand += ads[int(id[6])]
        # electral
        # car_brand += ads[int(id[7])]
        # print(car_brand)
        # print(brands)
        if car_brand in brands:
            correct += 1
    print("rate:{}%".format(correct/len(result)*100.))

    with open(output+'\\list.pkl', 'wb') as file:
        pickle.dump(result, file)
    with open(output+'\\test.txt', 'w') as f:
        f.write(str(correct/len(result)*100.))


def detect(save_img=False):
    yolo = YOLO()
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = True
    if classify:
        # modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
        print("load pretrained model successful!")
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    result = []
    ii = 0

    for path, img, im0s, vid_cap in dataset:
        result.append([[], []])
        result[ii][0].append(path.split('\\')[-1])
        # print(result)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred.shape)
        # print(pred)
        # assert 1 < 0
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred,plat_num = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de,lic_plat in zip(det,plat_num):
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls=de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        lb = ""
                        for a,i in enumerate(lic_plat):
                            # if a ==0:
                            #     continue
                            lb += CHARS[int(i)]
                        result[ii][1].append(lb)
                        label = '%s %.2f' % (lb, conf)
                        im0=plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # print(type(im0))
                        img_pil = Image.fromarray(im0) # narray转化为图片
                        im0 = yolo.detect_image(img_pil) #图片才能检测

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))#不打印东西

            # Stream results
            if view_img:
                if isinstance(im0, PIL.Image.Image):
                    # im0 = numpy.array(im0)
                    im0 = np.asarray(im0)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    im0 = np.array(im0) # 图片转化为 narray
                    cv2.imwrite(save_path, im0) #这个地方的im0必须为narray
                    # assert 1 < 0
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    im0 = np.array(im0) # 图片转化为 narray#JMW添加
                    vid_writer.write(im0)
        ii += 1

    if save_txt or save_img:
        # print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    return result

    # print('Done. (%.3fs)' % (time.time() - t0))#不打印一堆东西


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='source', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     opt = parser.parse_args()
#     # print(opt)
#     filePath = r'G:\data\CCPD2020\CCPD2020\ccpd_green'
#     files = os.listdir(filePath)
#     print(files)
#     for file in files:
#         opt.source = filePath+"\\"+file
#         opt.output = filePath+"\\"+file+"_test"
#         with torch.no_grad():
#             if opt.update:  # update all models (to fix SourceChangeWarning)
#                 for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
#                     detect()
#                     create_pretrained(opt.weights, opt.weights)
#             else:
#                 result = detect()
#                 test(result, opt.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
