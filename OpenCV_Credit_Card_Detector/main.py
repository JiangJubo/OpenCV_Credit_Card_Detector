# Credit Card ID Dector Project
# Date: 2022/4/14
import cv2
import numpy as np
import os.path


def cv_show(name, img_r, time):
    cv2.imshow(name, img_r)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def cv_img_resize(imgr, wide):
    img_g = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    hg, wg = img_g.shape
    height = int(wide * (hg/wg))
    imgr = cv2.resize(imgr, (wide, height))
    return imgr


# 分图像检测函数
def cv_credit_card_detector(num):

    # 待测图像的处理
    img_test = img[num]
    img_test = cv_img_resize(img_test, wide=500)
    cv_show('test', img_test, 800)
    img_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    # 礼帽操作
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)
    # x方向梯度
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * (gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    # 两次 闭操作 二值化操作
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 边缘检测
    contours_img, hierarchy_img = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_img = img_test.copy()
    res = cv2.drawContours(draw_img, contours_img, -1, (0, 0, 255), 2)
    cnts = contours_img
    # 轮廓特征选择
    locs = []
    for (i, c) in enumerate(cnts):
        (ix, iy, iw, ih) = cv2.boundingRect(c)
        ar = iw / float(ih)
        if 4.0 > ar > 3.0:
            if 100 > iw > 60:
                if 30 > ih > 20:
                    locs.append((ix, iy, iw, ih))
    locs = sorted(locs, key=lambda ix: ix[0])  # 将内容排序

    # 四块主要轮廓分别识别
    Output = []
    for (i, (gx, gy, gw, gh)) in enumerate(locs):
        group = img_gray[gy - 3:gy + gh + 3, gx - 3:gx + gw + 3]
        group = cv2.morphologyEx(group, cv2.MORPH_CLOSE, minKernel)
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 自适应二值化
        contours_img_min, hierarchy_img_min = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        groupOutput = []
        cnt_min = contours_img_min

        locs_min = []
        for (i, c) in enumerate(cnt_min):
            (ix, iy, iw, ih) = cv2.boundingRect(c)
            if iw > 10:
                locs_min.append((ix, iy, iw, ih))
        locs_min = sorted(locs_min, key=lambda ix: ix[0])  # 将内容排序

        # 得到最小的部分依次识别
        for (i, (gx_m, gy_m, gw_m, gh_m)) in enumerate(locs_min):
            rio = group[gy_m:gy_m + gh_m, gx_m:gx_m + gw_m]
            rio = cv2.resize(rio, (57, 88))
            scores = []
            for dig in p:
                result = cv2.matchTemplate(rio, digit[p[int(dig)]], cv2.TM_CCORR_NORMED)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            m = np.argmin(scores)  # 找到最小的数字的代号
            m = p[m]
            groupOutput.append(str(int(m)))

        # 结果输出
        Output.extend(groupOutput)
        cv2.rectangle(img_test, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 255, 255), 3)
        cv2.putText(img_test, "".join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 结果打印
    print("Credit Card #: {}".format("".join(Output)))
    cv_show('img', img_test, 800)
    # 检测结果保存
    file_pathname_save = 'C:/Users/503/PycharmProjects/CV0414_IDDector2/img_tested/'
    cv2.imwrite(os.path.join(file_pathname_save+"img_tested_"+str(num)+".png"), img_test)


# 文件的输入及预处理
img_ref = cv2.imread('ocr_a_reference.png')
img = []
file_pathname = 'C:/Users/503/PycharmProjects/CV0414_IDDector2/img'
for filename in os.listdir(file_pathname):
    print(filename)
    img.append(cv2.imread(file_pathname+'/'+filename))
print(len(img))

# 模板的操作 将各部分分割
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
ret, thresh_ref = cv2.threshold(gray_ref, 10, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh_ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
draw_img = img_ref.copy()
res_ref = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
x = np.zeros((1, 10))
y = np.zeros((1, 10))
w = np.zeros((1, 10))
h = np.zeros((1, 10))
digit = {}
for i in range(10):
    x[0, i], y[0, i], w[0, i], h[0, i] = cv2.boundingRect(contours[i])
    rio = gray_ref[int(y[0, i]):int(y[0, i] + h[0, i]), int(x[0, i]):int(x[0, i]+ w[0, i])]
    rect_ref = cv2.rectangle(img_ref, (int(x[0, i]), int(y[0, i])), (int(x[0, i] + w[0, i]),
                                                                 int(y[0, i] + h[0, i])), (0, 255, 0), 2)
    digit[i] = rio
p = np.zeros(10)  # 相当于一个字典
for j in range(10):
    for i in range(10):
        if x[0, i] > x[0, j]:
            p[i] = p[i] + 1
cv_show('rect', rect_ref, 500)

# 初始化卷积和
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
minKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

print("Testing...")

# 对所有文件夹中的图像统一检测
for i in range(len(img)):
    cv_credit_card_detector(i)

print("Tested")
