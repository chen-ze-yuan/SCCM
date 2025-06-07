# 声调符号
# 字母表
letter = list('abcdefghijklmnopqrstuvwxyz')
global shengyuns
global py2sy, sy2py
# 声母表
initials = [
  'b',    'p',    'm',    'f',
  'd',    't',    'n',    'l',
  'g',    'k',    'h',
  'j',    'q',    'x',
  'zh',   'ch',   'sh',   'r',
  'z',    'c',    's',
]
adds=['_a',   '_o',   '_e',   '_y',   '_w',   '_v',]

sheng = initials + adds

# 韵母表
_finals = {
  ' ': ['a',    'o',    'e',    'ai',   'ei',   'ao',   'ou',   'an',   'en',   'ang',  'eng',  'ong', 'er', 'i2', 'i3'],
  'i': ['i1',    'ia',           'ie',                   'iao',  'iou',  'ian',  'in',   'iang', 'ing',  'iong', ],
  'u': ['u',    'ua',   'uo',           'uai',  'uei',                  'uan',  'uen',  'uang', 'ueng',         ],
  'v': ['v',                    've',                                   'van',  'vn',                           ],
}
finals = _finals[' '] + _finals['i'] + _finals['u'] + _finals['v']
yun = finals

initials_finals = initials + finals
#print(initials_finals)
shengyuns = sheng + yun

# 拼音声韵母互转词典
_pinyin_to_initial_final = {}
_initial_final_to_pinyin = {}

# 除'v'行韵母外，其他韵母跟声母的组合
for final in (_finals[' '] + _finals['i'] + _finals['u']):
  # iou，uei，uen前面加声母的时候，写成iu，ui，un
  if final == 'iou':
    _final = 'iu'
  elif final == 'uei':
    _final = 'ui'
  elif final == 'uen':
    _final = 'un'
  else:
    _final = final

  for initial in initials:
    if _final == 'i2':
      if initial in ['z','c','s']:
        pinyin = initial + 'i'
        _pinyin_to_initial_final[pinyin] = [initial, 'i2']
        _initial_final_to_pinyin[initial + _final] = pinyin
        continue
    if _final == 'i3':
      if initial in ['zh','ch','sh']:
        pinyin = initial + 'i'
        _pinyin_to_initial_final[pinyin] = [initial, 'i3']
        _initial_final_to_pinyin[initial + _final] = pinyin
        continue
    if _final == 'i1':
      if initial not in ['zh', 'ch', 'sh','z','c','s']:
        pinyin = initial + 'i'
        _pinyin_to_initial_final[pinyin] = [initial, 'i1']
        _initial_final_to_pinyin[initial + _final] = pinyin
        continue
    if _final == 'er':
      continue
    elif initial in ['j', 'q', 'x'] and _final in ['u', 'uan','un','ue']:
      continue
    pinyin = initial + _final
    _pinyin_to_initial_final[pinyin] = [initial, final]
    _initial_final_to_pinyin[initial + final] = pinyin
# ' '行韵母
for final in _finals[' ']:
  # 前面没有声母的时候
  if final == 'i2' or final == 'i3':
    continue
  if final[0] == 'a':
    pinyin = final
    _pinyin_to_initial_final[pinyin] = ['_a',final]
    _initial_final_to_pinyin['_a' + final] = pinyin
  if final[0] == 'o':
    pinyin = final
    _pinyin_to_initial_final[pinyin] = ['_o',final]
    _initial_final_to_pinyin['_o'+ final] = pinyin
  if final[0] == 'e':
    pinyin = final
    _pinyin_to_initial_final[pinyin] = ['_e',final]
    _initial_final_to_pinyin['_e'+ final] = pinyin

# 'i'行韵母
for final in _finals['i']:
  # 前面没有声母的时候，写成yi，ya，ye，yao，you，yan，yin，yang，ying，yong
  if final == 'i1':
    pinyin = 'y' + 'i'
    _pinyin_to_initial_final[pinyin] = ['_y', final]
    _initial_final_to_pinyin['_y' + final] = pinyin
    continue
  if final in ['in', 'ing']:
    pinyin = 'y' + final
  else:
    pinyin = 'y' + final[1:]
  _pinyin_to_initial_final[pinyin] = ['_y',final]
  _initial_final_to_pinyin['_y'+ final] = pinyin
# 'u'行韵母
for final in _finals['u']:
  # 前面没有声母的时候，写成wu，wa，wo，wai，wei，wan，wen，wang，weng
  if final == 'u':
    pinyin = 'wu'
  else:
    pinyin = 'w' + final[1:]
  _pinyin_to_initial_final[pinyin] = ['_w',final]
  _initial_final_to_pinyin['_w'+ final] = pinyin

# 'v'行韵母
for final in _finals['v']:
  # 前面没有声母的时候，写成yu，yue，yuan，yun
  if final == 'v':
    pinyin = 'yu'
  else:
    pinyin = 'yu' + final[1:]
  _pinyin_to_initial_final[pinyin] = ['_y',final]
  _initial_final_to_pinyin['_y'+final] = pinyin
  # 跟声母j，q，x拼的时候，v写成u （自动替换之前u行韵母跟声母j，q，x的组合，如jun）

  for initial in ['j', 'q', 'x']:
    pinyin = initial + 'u' + final[1:]
    _pinyin_to_initial_final[pinyin] = [initial, final]
    _initial_final_to_pinyin[initial + final] = pinyin

  # 跟声母n，l拼的时候，v写成v（实际上不存在nvan，lvan，nvn，lvn）
  for initial in ['n', 'l']:
    if final == 'v' or final == 've':
      pinyin = initial + final
      _pinyin_to_initial_final[pinyin] = [initial, final]
      _initial_final_to_pinyin[initial + final] = [pinyin]

'''
def pinyin_to_initial_final(pinyin):
  for item in pinyin.strip().split(' '):
    yield _pinyin_to_initial_final.get(item)

def initial_final_to_pinyin(phoneme):
  for item in phoneme.strip().split(' '):
    yield _initial_final_to_pinyin.get(item)

def pinyin_to_shengyun(pinyin):
  return pinyin_to_initial_final(pinyin)

def shengyun_to_pinyin(phoneme):
  return initial_final_to_pinyin(phoneme)
'''
py2sy = _pinyin_to_initial_final
sy2py = _initial_final_to_pinyin

py2sy["yo"]=['_y','o']
sy2py["_yo"]=["yo"]
#print(py2sy["you"])
#print(py2sy["yo"])
#print(sy2py["_yiou"])

#print(py2sy["a"])
#print(sy2py["_aa"])
py2sy["<blank>"] = ["<blank>"]
sy2py["<blank>"] = ["<blank>"]

#print(py2sy["er"])
#print(py2sy["dui"])
#print(sy2py["jvan"])

'''
print(list(py2sy.items())[-10:])
print(list(sy2py.items())[-10:])
print(len(_pinyin_to_initial_final))
print(len(_initial_final_to_pinyin))
print(_pinyin_to_initial_final['yu'])
print(_initial_final_to_pinyin['jvan'])
'''
