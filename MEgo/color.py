from .models import EmotionColor
import re

# rgb to hex code
def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

# input parsed_emo should be list : ex) ['intest']
# from emotion list to overall hexcode : for multiple emotion
def emo_to_hex(parsed_emotion):

    n = len(parsed_emotion)
    r = 0
    g = 0
    b = 0
    for emo in parsed_emotion:
        text_re = re.compile("\w+")
        preprocessed_emo = re.findall(text_re,emo)[0]
        color = EmotionColor.objects.get(emotion__exact=preprocessed_emo)
        print('hello', preprocessed_emo)
        r = r +  (color.r * color.a + 255 * (1 - color.a)) / n
        g = g + (color.g * color.a + 255 * (1 - color.a)) / n
        b = b + (color.b * color.a + 255 * (1 - color.a)) / n
    return rgb_to_hex(r, g, b)
