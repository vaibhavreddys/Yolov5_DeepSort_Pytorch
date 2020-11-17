from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

from getFrame import n_th_frame

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

line_left = None
line_right = None
first_frame = None
line_set = False
def click_event(event, x, y, flags, params):
    global line_right
    global line_left
    global line_set
    if not (line_left is None or line_right is None):
        print ("Points are set")
        line_set = True
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        print ("Left Click event", cv2.EVENT_LBUTTONDOWN)
        line_left = (x, y)
        print ("x", x, "  y", y)
        if line_right is not None:
            cv2.line(first_frame, line_left, line_right, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(first_frame, line_left, 5, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)

    if event == cv2.EVENT_MBUTTONDOWN:
        print ("Right Click event", cv2.EVENT_LBUTTONDOWN)
        line_right = (x, y)
        print ("x", x, " y", y)
        if line_left is not None:
            cv2.line(first_frame, line_left, line_right, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(first_frame, line_right, 5, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)

def point_positions(X, Y):
    if not line_set:
        return 0
    print("\n")
    x1, y1, x2, y2, = line_left[0],line_left[1], line_right[0], line_right[1]
    #print("Line: ", x1, y1, x2, y2)
    v1 = (x2-x1, y2-y1)   # Vector1
    v2 = (x2-X, y2-Y)   # Vector 1
    xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

    if xp > 0:
        #print ("Man Above the line", X, Y)
        return 1
    elif xp < 0:
        #print ("Man Below the line", X, Y)
        return -1
    else:
        print ("WTF!, Man On the line", X, Y)
        #import sys
        #sys.exit(1)
        return 0

def get_line_points(input_source):
    print("Input source:", input_source)
    frame = n_th_frame(input_source)
    if frame is None:
        print ("Invalid frame received")
        return
    print ("Valid frame received")
    global first_frame
    first_frame = frame
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event, frame)
    while not line_set:
        cv2.imshow("Frame", first_frame)
        if cv2.waitKey(1)==13:
            break
    print ("Line is at:", line_left, line_right)
    from time import sleep
    sleep(1)
    cv2.destroyAllWindows()

def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

label_positions = {}
global_positions = {'ENTRY' : 0, 'EXIT' : 0}
start_positions = {}
mid_positions = {}

def update_entry_exit_counter(label, pos, start_pos, mid_pos):
    entry = 1
    exit = -1
    current_position = point_positions(pos[0], pos[1])
    current_mid_position = point_positions(mid_pos[0], mid_pos[1])
    # TODO: Need to implement the moving window technique to elimate few of the corner cases
    # where people stop and move slowly near the prediction border.
    if label in label_positions and label in mid_positions:
        if current_position == entry and ( label_positions[label] == exit or label_positions[label] == 0): # and \
            #(current_mid_position == entry and mid_positions[label] == entry):
                print("Man {}: entry".format(label))
                print ("Label positions: ", label_positions)
                print ("Mid Position: ", mid_positions)
                global_positions.update({'ENTRY' : global_positions['ENTRY'] + 1 })
        if current_position == exit and ( label_positions[label] == entry or label_positions[label] == 0): # and \
            #(current_mid_position == entry and mid_positions[label] == entry):
                global_positions.update({'EXIT' : global_positions['EXIT'] + 1 })
                print ("Man {}: exit".format(label))
                print ("Label positions: ", label_positions)
                print ("Mid Position: ", mid_positions)
    label_positions.update({ label : current_position })
    mid_positions.update({ label : current_mid_position})


def show_stats_on_screen(img):
    print ("Stats:")
    stats_bg_color = (0,51,51)
    stats_text_color = (51, 255, 255)
    x,y,w,h = 5,5,75,10
    font_size, font_thickness = 1.5, 1
    text = "Entry: " + str(global_positions['ENTRY']) + ", Exit: " + str(global_positions['EXIT'])
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (x, y), (x + t_size[0] + 3, y + t_size[1] + 4), stats_bg_color, -1)
    img = cv2.putText(img, text, (x ,y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, font_size, stats_text_color, font_thickness)
    print (global_positions)

def draw_boxes(img, bbox, centroid_tracking, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        cv2.circle(img, (x2, y2), 5, thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)
        cv2.circle(img, (x1, y1), 5, thickness=2, color=(0, 0, 255), lineType=cv2.LINE_AA)
        mid_x, mid_y = (x1 + x2) // 2, (y2 + y1) // 2
        #print("\n(x1, y1)=({},{})".format(x1,y1))
        #print("(MID-X, MID-Y)=({},{})".format(mid_x,mid_y))
        #print("(x2, y2)=({},{})".format(x2,y2))
        cv2.circle(img, (mid_x, mid_y), 5, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        position_tracked = (x2, y2)
        if centroid_tracking:
            position_tracked = (mid_x, mid_y)
        update_entry_exit_counter(label, position_tracked, (x1, y1), (mid_x, mid_y))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, centroid_tracking = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.centroid
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    print("\nBounding box", bbox_xyxy)
                    identities = outputs[:, -1]
                    print("___________Draw_boxes now__________________________________")
                    draw_boxes(im0, bbox_xyxy, centroid_tracking, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:  
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            show_stats_on_screen(im0)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                #cv2.namedWindow(p, cv2.WINDOW_NORMAL)
                im0 = cv2.line(im0, line_left, line_right, thickness=3, color=(0,255,0), lineType=cv2.LINE_AA)
                cv2.circle(im0, line_left, 5, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)
                cv2.circle(im0, line_right, 5, thickness=2, color=(0, 255, 0), lineType=cv2.LINE_AA)
                im0 = cv2.resize(im0, (1080, 980))
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                #print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    #print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.75, help='IOU threshold for NMS') # Check this with 0.75
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--centroid', action='store_true', help='Track using the centroid of the box or else use the bottom right corner')
    # parser.add_argument("--line", nargs='+', type=int)
    parser.add_argument("--line", action='store_true')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print("\nArguments passed: ", args, end="\n\n")
    #sys.exit(0)
    if args.line is not None and args.line: # and (len(args.line) == 4): # args type -> nargs='+'
        get_line_points(args.source)
    else:
        print(" Do not draw line")

    with torch.no_grad():
        detect(args)
