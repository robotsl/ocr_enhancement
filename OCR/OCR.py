# -*- encoding=utf-8 -*-
import pyocr.builders
import time
from PIL import Image,ImageEnhance
import pyocr.builders

class OCR(object):
    """docstring for OCR."""
    def __init__(self):
        self.tool = pyocr.get_available_tools()[0]
        self.builder = pyocr.builders.TextBuilder()
        self.lang = 'eng'
        print("self.tool",self.tool)
    def  pic_orc(self,filename,resize_num,b):
        try:
            print("Begin to ocr!\n")
            im = Image.open(filename)
            im = im.resize((im.width * int(resize_num), im.height * int(resize_num)))
            imgry = im.convert('L')
            sharpness = ImageEnhance.Contrast(imgry)
            sharp_img = sharpness.enhance(b)
            txt = self.tool.image_to_string(sharp_img)#, lang=self.lang,builder=self.builder)
            return txt
        except:
            print("error")
