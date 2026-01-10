# -------------------------------------------------------------------------
# predict.py (预测入口)
# -------------------------------------------------------------------------
import time
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 引入核心类
from mask_rcnn import MASK_RCNN

if __name__ == "__main__":
    # 初始化模型
    mask_rcnn = MASK_RCNN()

    # -------------------------------------------------------------------------
    #   mode:
    #   'predict'       : 单张图片交互式预测
    #   'dir_predict'   : 遍历文件夹批量预测并保存结果
    #   'video'         : 视频检测
    #   'fps'           : 测速
    # -------------------------------------------------------------------------
    mode = "predict"

    # -------------------------------------------------------------------------
    #   文件夹预测配置 (dir_predict 模式专用)
    # -------------------------------------------------------------------------
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # -------------------------------------------------------------------------
    #   视频配置 (video 模式专用)
    # -------------------------------------------------------------------------
    video_path = 0  # 0表示摄像头，或填视频路径 "xxx.mp4"
    video_save_path = ""
    video_fps = 25.0

    if mode == "predict":
        print("【Predict Mode】请输入图片路径 (支持 jpg, png, tif 等)")
        while True:
            img = input('Input image filename: ')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                # 推理并返回绘图后的 PIL 图片
                r_image = mask_rcnn.detect_image(image)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        print(f"【Dir Predict Mode】Reading from {dir_origin_path}, Saving to {dir_save_path}")
        img_names = os.listdir(dir_origin_path)

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                try:
                    image = Image.open(image_path)
                    r_image = mask_rcnn.detect_image(image)

                    # 保存结果 (可视化结果保存为 png 方便查看)
                    save_name = os.path.splitext(img_name)[0] + ".png"
                    r_image.save(os.path.join(dir_save_path, save_name), quality=95)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    elif mode == "fps":
        test_interval = 100
        # 请准备一张测试图
        img_path = "img/street.jpg"
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found.")
        else:
            img = Image.open(img_path)
            tact_time = mask_rcnn.get_FPS(img, test_interval)
            print(f'{tact_time:.4f} seconds per image, {1 / tact_time:.2f} FPS')

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频）")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            # BGR -> RGB -> PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))

            # Detect
            frame = np.array(mask_rcnn.detect_image(frame))

            # PIL -> RGB -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        capture.release()
        out.release() if video_save_path != "" else None
        cv2.destroyAllWindows()