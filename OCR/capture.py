import tkinter
import tkinter.filedialog
import os
import torch
#from PIL import ImageGrab
import pyscreenshot as ImageGrab
import OCR.OCR as OCR
from time import sleep


class MyCapture(object):
    def __init__(self,png,tk_obj):
        #变量X和Y用来记录鼠标左键按下的位置
        self.X = tkinter.IntVar(value=0)
        self.Y = tkinter.IntVar(value=0)
        self.tk_obj = tk_obj.init_window
        self.init_data_Text = tk_obj.init_data_Text
        self.ok = False
        self.matrix = []
        #屏幕尺寸
        screenWidth = self.tk_obj.winfo_screenwidth()
        screenHeight = self.tk_obj.winfo_screenheight()
        #创建顶级组件容器
        self.top = tkinter.Toplevel(self.tk_obj, width=screenWidth, height=screenHeight)
        #不显示最大化、最小化按钮
        self.top.overrideredirect(True)
        self.canvas = tkinter.Canvas(self.top,bg='white', width=screenWidth, height=screenHeight)
        #显示全屏截图，在全屏截图上进行区域截图
        self.image = tkinter.PhotoImage(file=png)
        self.canvas.create_image(screenWidth//2, screenHeight//2, image=self.image)
        #鼠标左键按下的位置
        def onLeftButtonDown(event):
            self.X.set(event.x)
            self.Y.set(event.y)
            #开始截图
            self.sel = True
        self.canvas.bind('<Button-1>', onLeftButtonDown)
        #鼠标左键移动，显示选取的区域
        def onLeftButtonMove(event):
            if not self.sel:
                return
            global lastDraw
            try:
                #删除刚画完的图形，要不然鼠标移动的时候是黑乎乎的一片矩形
                self.canvas.delete(lastDraw)
            except Exception as e:
                pass
            lastDraw = self.canvas.create_rectangle(self.X.get(), self.Y.get(), event.x, event.y, outline='black')
        self.canvas.bind('<B1-Motion>', onLeftButtonMove)
        #获取鼠标左键抬起的位置，保存区域截图
        def onLeftButtonUp(event):
            self.sel = False
            try:
                self.canvas.delete(lastDraw)
            except Exception as e:
                pass
            sleep(0.1)
            #考虑鼠标左键从右下方按下而从左上方抬起的截图
            left, right = sorted([self.X.get(), event.x])
            top, bottom = sorted([self.Y.get(), event.y])
            pic = ImageGrab.grab((left+1, top+1, right, bottom))
            #弹出保存截图对话框
            fileName = "/home/robotsl/workspace/ocr_enhancement/OCR/images/for_ocr.png"
            #关闭当前窗口
            self.top.destroy()
            pic.save(fileName)
            tk_obj.init_window.state('normal')
            result,matrix = OCR.OCR_OR_LOGMAX("log",fileName)
            tk_obj.matrix = matrix
            #sorted,indes = torch.sort(log,-1)
            #tk_obj.set_result(result,"before_enhance")
            tk_obj.init_data_Text.delete('1.0','end')
            #tk_obj.init_data_Text.insert("end","test")
            tk_obj.init_data_Text.insert("end",result)
            tk_obj.result_data_Text.delete('1.0','end')

        self.canvas.bind('<ButtonRelease-1>', onLeftButtonUp)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)
