import tkinter as tk
import tkinter.filedialog
import os
#from PIL import ImageGrab
import pyscreenshot as ImageGrab
from time import sleep
import OCR.capture as capture


class GUI(object):
    def __init__(self):
        self.init_window = tk.Tk()
    #设置窗口
    def set_init_window(self):
        self.init_window.title("OCR增强工具v0.1")           #窗口名
        self.init_window.geometry('600x450+10+10')

        self.init_data_label = tk.Label(self.init_window, text="ocr扫描结果")
        self.init_data_label.grid(row=0, column=0)

        self.result_data_label = tk.Label(self.init_window, text="增强后结果")
        self.result_data_label.grid(row=0, column=15)

        self.init_data_Text = tk.Text(self.init_window, width=37, height=30)  #原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=10, columnspan=10)

        self.result_data_Text = tk.Text(self.init_window, width=37, height=30)  #处理结果展示
        self.result_data_Text.grid(row=1, column=15, rowspan=15, columnspan=10)

        self.enhance_button = tk.Button(self.init_window, text="增强", bg="lightblue", width=8,command=self.command_for_button)#,command=self.str_trans_to_md5)  # 调用内部方法  加()为直接调用
        self.enhance_button.grid(row=1, column=13)

    def command_for_button(self):
        str = self.init_data_Text.get('0.0','end')
        print("waiting for devolpoe",str)
        self.result_data_Text.delete('1.0','end')
        self.result_data_Text.insert("end",str)

    def key(self,event=None):
        print('You pressed Ctrl+Shift+t')
        self.init_window.state('iconic')
        filename = './OCR/images/temp.png'  #这里一定要这样写，不然会出错
        im = ImageGrab.grab()
        im.save(filename)
        im.close()
        #self.init_data_Text.insert("end","test")
        capture.MyCapture(filename,self)
        self.init_window.state('withdrawn')
        #self.init_data_Text.insert("end","test")
        #self.clear_text()



    def gui_start(self):
        self.set_init_window()
        self.init_window.focus_set()
        self.init_window.bind('<Control-Shift-KeyPress-T>', self.key)
        self.init_window.mainloop()

    def set_result(self,string,label):
        if label == "before_enhance":
            self.init_data_Text.delete('1.0','end')
            self.init_data_Text.insert("end",string)
            self.result_data_Text.delete('1.0','end')
        else:
            self.result_data_Text.insert("end",string)
    def clear_text(self):
        self.init_data_Text.delete('1.0','end')
        self.result_data_Text.delete('1.0','end')

def main():
    gui = GUI()
    gui.gui_start()

if __name__ == '__main__':
    main()
