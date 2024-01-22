from ctypes import *
from bitstring import ConstBitStream
import os

functions = CDLL(os.path.dirname(__file__) + "/huff.so")
functions.huffmanEncode.argtypes = [c_char_p, c_int, c_int]
functions.huffmanEncode.restype = c_char_p
functions.huffmanDecode.argtypes = [c_char_p, c_int, c_int]
functions.huffmanDecode.restype = c_char_p

diller = ["de", "en", "es", "fr", "it", "nl","tr"]

def huffman_encode(metin, dil):
    # text = c_char_p(metin) 
    text = c_char_p(bytes(metin, 'utf8'))
    size = c_int(len(metin)*20)   
    lang = c_int(diller.index(dil))
    output = c_char_p(functions.huffmanEncode(text, size, lang))
    return ConstBitStream(bin=output.value.decode(encoding='utf8', errors='ignore')).tobytes()  # '01000001' -> 65
   

def huffman_decode(veri, dil):
    data = c_char_p(bytes(ConstBitStream(veri).bin, encoding='utf8'))  # 65 -> '01000001'
    size = c_int(len(veri)*2)
    lang = c_int(diller.index(dil))
    output = c_char_p(functions.huffmanDecode(data, size, lang))
    return output.value.decode(encoding='utf8', errors='ignore')
