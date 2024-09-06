import numpy as np  
import math 
import matplotlib.pyplot as plt
import os
import cv2  # OpenCV computer vision library
import imutils  # Image processing
import dlib  # Library for machine learning, computer vision, and image processing
from imutils import face_utils  # Tools for handling facial data


def show_img(img, bigger=False):
    if bigger:
        plt.figure(figsize=(15,15))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #將色彩空間轉為RGB
    plt.imshow(image_rgb) #在圖表中繪製圖片
    plt.show()  #顯示圖表

def rotate_img(img, angle):
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w+20, h))
    
    return rotate_img

def resize_img(img, size, Rlength):

    scale_percent =  Rlength/size # 要放大縮小幾%
    width = int(img.shape[1] * scale_percent) # 縮放後圖片寬度
    height = int(img.shape[0] * scale_percent) # 縮放後圖片高度
    dim = (width, height) # 圖片形狀 
    
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  #圖像縮放

    return resize_img

def process_coordinate(img_path):
    #輸入圖片
    img = cv2.imread(img_path) #主圖像
    input_img = img.copy()
    righteye=cv2.imread('./source/righteye.png') #右眼
    lefteye=cv2.imread('./source/lefteye.png') #左眼
    mouth=cv2.imread('./source/mouth.png') #嘴吧
    dets = detector(img, 1) #使用detector進行人臉識別,dets為返回的結果

    #尋找臉部68點
    for k,d in enumerate(dets):
        # print("Detection: {} Left: {} Top: {} Right: {} Bottom: {}".format(
        #         k, d.left(), d.top(), d.right(), d.bottom()))
        # Left人臉左邊距離圖片左邊界的距離
        # Right人臉右邊距離圖片左邊界的距離
        # Top人臉上邊距離圖片上邊界的距離
        # Bottom人臉下邊距離圖片上邊界的距離

        shape = predictor(img, d) # 使用predictor進行人臉關鍵點識別,shape為返回的結果

        
        # 繪製特徵點
        for index,pt in enumerate(shape.parts()):
            # print('Part {}: {}'.format(index, pt))
            if index == 37:  # 找出右眼右眼角座標
                lilx = pt.x
                lily = pt.y
            elif index == 40:  # 找出右眼左眼角座標
                lirx = pt.x
                liry = pt.y
            elif index == 43:  # 找出左眼右眼角座標
                rilx = pt.x
                rily = pt.y
            elif index == 46:  # 找出左眼左眼角座標
                rirx = pt.x
                riry = pt.y
            elif index == 49:  # 找出左嘴角座標
                lmx = pt.x
                lmy = pt.y
            elif index == 55:  # 找出右嘴角座標
                rmx = pt.x
                rmy = pt.y
            elif index == 63:  # 找出上嘴唇下座標
                umx = pt.x
                umy = pt.y
            elif index == 67:  # 找出下嘴唇上座標
                dmx = pt.x
                dmy = pt.y
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (255, 0, 0), 2) # 標記特徵點

            font = cv2.FONT_HERSHEY_SIMPLEX # 字體
            cv2.putText(img, str(index+1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA) # 標記特徵點序號
    # print()

    # 右眼眼距
    Rlength=( math.sqrt((abs(lilx-lirx)**2) + (abs(lily-liry)**2)) ) # 兩點距離公式
    # print("右眼眼距:",Rlength)

    # 左眼眼距
    Llength=( math.sqrt((abs(rilx-rirx)**2) + (abs(rily-riry)**2)) )  # 兩點距離公式
    # print("左眼眼距:",Llength)

    # 嘴距
    Mlength=( math.sqrt((abs(lmx-rmx)**2) + (abs(lmy-rmy)**2)) )  # 兩點距離公式
    # print("嘴距:",Mlength)




    # 處理右眼
    # print(righteye.shape) # 印出圖片大小
    # show_img(righteye)  # show出圖片

    righteye=rotate_img(righteye,20) # 旋轉圖片(逆時針20度)
    # print(righteye.shape)  # 印出圖片大小
    # show_img(righteye)  # show出圖片

    righteye=resize_img(righteye,10,Rlength)  # 圖片縮放
    # print(righteye.shape)  # 印出圖片大小
    # show_img(righteye)  # show出圖片


    # 處理左眼
    # print(lefteye.shape)  #印出圖片大小
    # show_img(lefteye)  #show出圖片

    lefteye=resize_img(lefteye,16,Rlength)  #圖片縮放
    # print(lefteye.shape)  #印出圖片大小
    # show_img(lefteye)  #show出圖片


    # 處理嘴巴
    # print(mouth.shape)  #印出圖片大小
    # show_img(mouth)  #show出圖片

    mouth=resize_img(mouth,5.75,Rlength)  #圖片縮放
    # print(mouth.shape)  #印出圖片大小
    # show_img(mouth)  #show出圖片



    # 右眼中心座標
    righteye_x=(lilx+lirx)/2-(righteye.shape[1]*0.54)  # 右眼中心x座標-星爆右眼中心x座標
    righteye_y=(lily+liry)/2-(righteye.shape[0]*0.69)  # 右眼中心y座標-星爆右眼中心y座標
    # print("右眼中心座標",righteye_x,righteye_y)

    # 左眼中心座標
    lefteye_x=(rilx+rirx)/2-(lefteye.shape[1]*0.55)  # 左眼中心x座標-星爆左眼中心x座標
    lefteye_y=(rily+riry)/2-(lefteye.shape[0]*0.7)  # 左眼中心y座標-星爆左眼中心y座標
    # print("左眼中心座標",lefteye_x,lefteye_y)

    # 嘴巴中心座標
    mou_x=(((lmx+rmx)/2-(mouth.shape[1]*0.42))+((umx+dmx)/2-(mouth.shape[1]*0.42)))/2  
    # ((嘴巴左右中心x座標-星爆嘴中心x座標)+(嘴巴上下中心x座標-星爆嘴中心x座標))/2
    mou_y=(((lmy+lmy)/2-(mouth.shape[0]*0.5))+((umy+dmy)/2-(mouth.shape[0]*0.5)))/2   
    # ((嘴巴左右中心y座標-星爆嘴中心y座標)+(嘴巴上下中心y座標-星爆嘴中心y座標))/2
    # print("嘴巴中心座標",mou_x,mou_y)

    # show_img(img)  # show出圖片
    # cv2.imshow('img', img)
    # k = cv2.waitKey()
    # cv2.destroyAllWindows()
    return input_img, lefteye, lefteye_x, lefteye_y, righteye, righteye_x, righteye_y, mouth, mou_x, mou_y

def process_sources(lefteye, righteye, mouth):
    
    # show_img(lefteye)
    # show_img(righteye)

    # 去背右眼
    righteye_hsv = cv2.cvtColor(righteye,cv2.COLOR_RGB2HSV)  # 將色彩空間轉為RGB

    lower_blue=np.array([0,0,0])  # 獲取最小閾值
    upper_blue=np.array([0,255,255])  # 獲取最大閾值

    # cv2.inRange(圖片，低於這個的值圖像值變為0，高於這個的值圖像值變為0)
    mask = cv2.inRange(righteye_hsv, lower_blue, upper_blue)  #創建遮罩
    # show_img(mask) # show出遮罩

    righteye_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)))  # 開運算(先侵蝕後膨脹)
    # show_img(righteye_opening)  # show出圖片

    # 去背左眼
    lefteye_hsv = cv2.cvtColor(lefteye,cv2.COLOR_RGB2HSV)  #將色彩空間轉為RGB

    lower_blue=np.array([0,0,0])  #獲取最小閾值
    upper_blue=np.array([0,255,255])  #獲取最大閾值

    #cv2.inRange(圖片，低於這個的值圖像值變為0，高於這個的值圖像值變為0)
    mask = cv2.inRange(lefteye_hsv, lower_blue, upper_blue)  #創建遮罩
    # show_img(mask)

    lefteye_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))) 
    # show_img(lefteye_opening)  #show出圖片

    #去背嘴巴
    mouth_hsv = cv2.cvtColor(mouth,cv2.COLOR_RGB2HSV)

    lower_blue=np.array([0,0,0])  #獲取最小閾值
    upper_blue=np.array([0,255,255])  #獲取最大閾值

    #cv2.inRange(圖片，低於這個的值圖像值變為0，高於這個的值圖像值變為0)
    mask = cv2.inRange(mouth_hsv, lower_blue, upper_blue)  #創建遮罩
    # show_img(mask)

    mouth_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)))  #開運算(先侵蝕後膨脹)
    # show_img(mouth_opening)  #show出圖片

    # show_img(lefteye_opening)
    # show_img(righteye_opening)
    return lefteye_opening, righteye_opening, mouth_opening

def result(input_img, lefteye, lefteye_x, lefteye_y, righteye, righteye_x, righteye_y, mouth, mou_x, mou_y,lefteye_mask, righteye_mask, mouth_mask):
    
    # show_img(lefteye)
    # show_img(righteye)

    rows,cols,chanel = righteye.shape  

    center = [righteye_x,righteye_y]  #設置前景圖開始位置

    #一格一格跑，遇到黑色就給對應位置的顏色
    for i in range(rows): 
        for j in range(cols): 
            if righteye_mask[i,j]==0:  #代表黑色
                input_img[int(center[1]+i),int(center[0]+j)] =righteye[i,j]  #賦值顏色

    #貼上左眼
    rows,cols,chanel = lefteye.shape

    center = [lefteye_x,lefteye_y]  #設置前景圖開始位置

    #一格一格跑，遇到黑色就給對應位置的顏色
    for i in range(rows): 
        for j in range(cols): 
            if lefteye_mask[i,j]==0:  #代表黑色
                input_img[int(center[1]+i),int(center[0]+j)] =lefteye[i,j]  #賦值顏色

    #貼上嘴巴
    rows,cols,chanel = mouth.shape

    center = [mou_x,mou_y]  #設置前景圖開始位置

    #一格一格跑，遇到黑色就給對應位置的顏色
    for i in range(rows): 
        for j in range(cols): 
            if mouth_mask[i,j]==0:  #代表黑色
                input_img[int(center[1]+i),int(center[0]+j)] =mouth[i,j]  #賦值顏色

    
    return input_img

if __name__ == '__main__':
    
    img_name = input('輸入想星爆的人像圖片名稱: ')
    img_folder = './input_img/'  # 指定圖片所在的資料夾

    # 檢查 jpg 和 png 檔案
    img_extensions = ['.jpg', '.png']
    img_path = None

    # 防護機制：檢查圖片是否存在
    for ext in img_extensions:
        potential_path = f'{img_folder}{img_name}{ext}'
        if os.path.exists(potential_path):
            img_path = potential_path
            break

    if img_path is None:
        print(f"錯誤：在 {img_folder} 中找不到 {img_name} 的 .jpg 或 .png 檔案。")
    else:
        try:
            # 嘗試讀取圖片檔案
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("該檔案無法作為圖片載入。")
            
            # 在此處進行圖片的處理
            print(f"圖片 {img_path} 已成功載入。")
        
        except Exception as e:
            # 捕捉例外情況
            print(f"An error occurred: {e}")

    save_image = input("要儲存圖片嗎？(y/n): ").strip().lower()

    detector = dlib.get_frontal_face_detector() # face detector
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") # feature extractor
    
    input_img, lefteye, lefteye_x, lefteye_y, righteye, righteye_x, righteye_y, mouth, mou_x, mou_y = process_coordinate(img_path)
    lefteye_mask, righteye_mask, mouth_mask = process_sources(lefteye, righteye, mouth)
    ssf_img = result(input_img, lefteye, lefteye_x, lefteye_y, righteye, righteye_x, righteye_y, mouth, mou_x, mou_y, lefteye_mask, righteye_mask, mouth_mask)
    
    if save_image == 'y':
        # 指定儲存圖片的路徑
        save_path = f'result/{img_name}.jpg'  # 替換為儲存圖片的實際路徑
        cv2.imwrite(save_path, ssf_img)
        print(f"圖片已儲存至 {save_path}")
    else:
        print("圖片未儲存")

    show_img(ssf_img)