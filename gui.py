from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
from model import *
from tkinter.ttk import Combobox


def changeStyle(img):
    # 根据设定窗口尺寸修改展示图片
    img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_LINEAR)
    # 转化成能显示的格式
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = Image.fromarray(dst)

    show_img = ImageTk.PhotoImage(image=dst)
    return show_img


def openImage():
    global image
    imgPath = askopenfilename()
    if len(imgPath) == 0:
        Label(window, text="路径为空", fg='red').grid(row=4, column=1, sticky='w')
    else:
        imgPath = imgPath.replace("\\", "//")
        try:
            image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            Label(window, text="路径正确", fg='green').grid(row=4, column=1, sticky='w')

            show_img = changeStyle(image)

            origin_img_label = Label(window, image=show_img)
            origin_img_label.image = show_img
            origin_img_label.grid(row=1, column=0)
        except:
            Label(window, text="路径错误", fg='red').grid(row=4, column=1, sticky='w')


def removeSmog():
    algName = varLabel.get()
    if algName == "请选择去雾算法":
        Label(window, text="请选择去雾算法", fg="red").grid(row=4, column=1)
        Label(window, image=empty_img).grid(row=1, column=1)
    else:
        Label(window, text="", width=20).grid(row=4, column=1)

        if algName == values[0]:
            # 使用模型
            DCP_model = DCP(image)
            DCP_model.runAlgorithm(kernel_size=15)

            res_img = DCP_model.getRes()
            res_img = changeStyle(res_img)

            res_img_label = Label(window, image=res_img)
            res_img_label.image = res_img
            res_img_label.grid(row=1, column=1)
        elif algName == values[1]:
            # 使用模型
            Retinex_model = Retinex(image)
            Retinex_model.runAlgorithm()

            res_img = Retinex_model.getRimg()
            res_img = changeStyle(res_img)

            res_img_label = Label(window, image=res_img)
            res_img_label.image = res_img
            res_img_label.grid(row=1, column=1)


if __name__ == "__main__":
    window = Tk()  # 创建一个窗口
    window.title("图像去雾")
    window.geometry("820x600")  # 设置窗口大小

    # 创建一张空白图
    empty_img = Image.new('RGB', (400, 300), (255, 255, 255))
    empty_img = ImageTk.PhotoImage(image=empty_img)
    Label(window, text="图像去雾", font=30).grid(row=0, columnspan=2, sticky="n")
    Label(window, image=empty_img).grid(row=1, column=0, pady=20)
    Label(window, text="原图", font=15).grid(row=2, column=0)
    Label(window, image=empty_img).grid(row=1, column=1)
    Label(window, text="去雾后图像", font=15).grid(row=2, column=1)
    varLabel = StringVar()
    varLabel.set("请选择去雾算法")
    values = ["暗通道先验算法", "Retinex算法"]
    Combobox(window, height=2, width=20, state="readonly", cursor="arrow", textvariable=varLabel, values=values).grid(row=3, column=0)
    Button(window, text="打开图片", width=10, height=2, command=openImage).grid(row=3, column=1, pady=50, sticky="w")
    Button(window, text="图片去雾", width=10, height=2, command=removeSmog).grid(row=3, column=1, pady=50)
    window.mainloop()  # 维持窗口打开状态
