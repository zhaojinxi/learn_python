import os
import cv2
import time
import imageio


def get_all_img_files(root_path):
    for root, dirs, img_files in os.walk(root_path):
        all_img_files = list(map(lambda path: os.path.join(root, path), img_files))
        all_img_files.sort()
        return all_img_files


def create_gif(gif_name, all_img_files, duration=0.1):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ：
    all_img_files :
    duration :  gif 图像时间间隔
    '''

    frames = list(map(lambda img_file: imageio.imread(img_file), all_img_files))  # 读取 png 图像文件
    # 保存为 gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


if __name__ == '__main__':
    root_path = '/opt/disk1/meteorology_ai/SRAD2018_TRAIN_010/RAD_206482484232530'
    all_img_path = get_all_img_files(root_path)

    # create_gif('/opt/disk1/a.gif', all_img_path, duration=0.1)

    for img_path in all_img_path:
        im = cv2.imread(img_path)
        print(im.shape)
        cv2.imshow('a', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()