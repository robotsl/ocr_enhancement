做了以下假设简化：

1. 一级字库（3755 个）的印刷体汉字识别。
2. 常见字体（思源、方正等）的识别。
3. 不考虑数字、字母。





字体图片合成

```shell
python gen_printed_char.py --out_dir ./dataset --font_dir ./chinese_fonts --width 30 --height 30 --margin 4 --rotate 30 --rotate_step 1

```

