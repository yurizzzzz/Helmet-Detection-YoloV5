import ctypes
import os
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import argparse

# 设置阈值
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
# 设置检测的类别
categories = ["head", "helmet"]


# 遍历检测框并在检测框上标注信息
def draw_boxes(image_raw, result_boxes, result_scores, result_classid):
    for i in range(len(result_boxes)):
        box = result_boxes[i]
        plot_one_box(
            box,
            image_raw,
            label="{}:{:.2f}".format(categories[int(result_classid[i])],
                                     result_scores[i]),
        )
    return image_raw

def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    cv2.putText(img, "FPS: {:.1f}".format(fps), (11, 25), font, 0.5, (32, 32, 32), 4, line)
    cv2.putText(img, "FPS: {:.1f}".format(fps), (10, 25), font, 0.5, (240, 240, 240), 1, line)
    return img


# 检测框标注信息
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    # 检测框边线粗细
    tl = (line_thickness
          or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    # 检测框边线颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 在源图像上框出矩形框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 标注标签即类别信息
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


# YoloV5的TensorRT集成类
class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_file_path):
        # Create a Context on this device
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(
                engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                # The shape of processing images
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    # TensorRT推理检测图像
    def infer(self, image_raw):
        # params：image_raw是输入图像的array类型
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            image_raw)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w)
        # 返回原图像，检测框，检测的置信度，检测类别
        return image_raw, result_boxes, result_scores, result_classid

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2,
                                   cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    # 转换数据表示形式
    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(
            x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:,
              1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:,
              3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:,
              0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:,
              2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred)
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes,
                                      scores,
                                      iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid


def images_detection(args):
    # 加载图片和tensorrt文件
    img = cv2.imread(args.img_dir)
    PLUGIN_LIBRARY = args.lib_dir
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = args.engine_dir
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    # 开始推理检测
    print("start detection!")
    start_time = time.time()
    img, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(
        img)
    end_time = time.time()
    img = draw_boxes(img, result_boxes, result_scores, result_classid)
    cv2.imwrite("result.jpg", img)
    cv2.destroyAllWindows()
    print("finish!, cost: %.2f" % (end_time - start_time))
    yolov5_wrapper.destroy()


def video_detection(args):
    video = cv2.VideoCapture(args.video_dir)
    PLUGIN_LIBRARY = args.lib_dir
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = args.engine_dir
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    print("start detection!")
    full_scrn = False
    while True:
        ret, img = video.read()
        if img is not None:
            timer = cv2.getTickCount()
            img, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(
                img)
            img = draw_boxes(img, result_boxes, result_scores, result_classid)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.imshow("OUTPUT", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print("\rfps: " + str(fps), end="")
        else:
            break
    print("\nfinish!")
    cv2.destroyAllWindows()
    yolov5_wrapper.destroy()


def csiCam_detection(args):
    gst_str = ('nvarguscamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)1920, height=(int)1080, '
               'format=(string)NV12, framerate=(fraction)60/1 ! '
               'nvvidconv flip-method=2 ! '
               'video/x-raw, width=(int)1920, height=(int)1080, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink')
    camera = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    PLUGIN_LIBRARY = args.lib_dir
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = args.engine_dir
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    print("start detection!")
    while True:
        ret, img = camera.read()
        timer = cv2.getTickCount()
        img, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(
            img)
        img = draw_boxes(img, result_boxes, result_scores, result_classid)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.namedWindow("OUTPUT", 0)
        cv2.resizeWindow("OUTPUT", 1920, 1080)
        cv2.imshow("OUTPUT", img)
        print("\rfps: " + str(fps), end="")
        if cv2.waitKey(1) == ord('q'):
            break

    print("\nfinish!")
    camera.release()
    cv2.destroyAllWindows()
    yolov5_wrapper.destroy()


def usbCam_detection(args):
    camera = cv2.VideoCapture(0)
    camera.set(3, 1920)
    camera.set(4, 1080)
    PLUGIN_LIBRARY = args.lib_dir
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = args.engine_dir
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    print("start detection!")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('output.mp4', fourcc, 6, (1920, 1080))
    while True:
        ret, img = camera.read()
        timer = cv2.getTickCount()
        img, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(
            img)
        img = draw_boxes(img, result_boxes, result_scores, result_classid)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        img = show_fps(img, fps)
        out.write(img)
        cv2.namedWindow("OUTPUT", 0)
        cv2.resizeWindow("OUTPUT", 1920, 1080)
        cv2.imshow("OUTPUT", img)
        print("\rfps: " + str(fps), end="")
        if cv2.waitKey(1) == ord('q'):
            break

    print("\nfinish!")
    out.release()
    camera.release()
    cv2.destroyAllWindows()
    yolov5_wrapper.destroy()


def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default='img',
                        help="Choose the type of detection——img/video/csiCam/usbCam")

    parser.add_argument("--img_dir", type=str, default='./1.jpg',
                        help="Location of image file")

    parser.add_argument("--video_dir", type=str, default='./test.mp4',
                        help="Location of video file")

    parser.add_argument("--lib_dir", type=str, default='build/libmyplugins.so',
                        help="Location of libmyplugins.so")

    parser.add_argument("--engine_dir", type=str, default='./best.engine',
                        help="Location of tensorRT engine file")


    return parser.parse_args()

if __name__ == "__main__":
    args = input_args()
    if args.source == 'img':
        images_detection(args)
    if args.source == 'video':
        video_detection(args)
    if args.source == 'csiCam':
        csiCam_detection(args)
    if args.source == 'usbCam':
        usbCam_detection(args)
