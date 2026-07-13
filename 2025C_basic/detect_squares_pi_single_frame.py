import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
from picamera2 import Picamera2

# ==================== 原距离标定（保持不变） ====================
DISTANCE_CALIBRATION = [
    (761.0, 200),(781.5, 195),(801.0, 190),(822.0, 185),(844.0, 180),
    (868.0, 175),(893.0, 170),(919.0, 165),(946.5, 160),(975.0, 155),
    (1006.5, 150),(1042.0, 145),(1075.5, 140),(1114.5, 135),
    (1156.5, 130),(1200.5, 125),(1249.0, 120),(1301.0, 115),
    (1357.0, 110),(1419.5, 105),(1488.5, 100)
]
DISTANCE_CALIBRATION.sort(key=lambda x: x[0])
CALIB_PIXELS = np.array([p for p, _ in DISTANCE_CALIBRATION], np.float32)
CALIB_DISTANCES = np.array([d for _, d in DISTANCE_CALIBRATION], np.float32)

COMPENSATION_TABLE = {
    (100,120): {'normal':1.006,'normal2':1.0035,'slight':1.005,'moderate':1.000,'large':1.000,'circle':0.999},
    (120,140): {'normal':1.005,'normal2':1.0035,'slight':1.004,'moderate':1.009,'large':1.009,'circle':0.999},
    (140,160): {'normal':1.005,'normal2':1.0035,'slight':1.003,'moderate':1.008,'large':1.008,'circle':0.999},
    (160,180): {'normal':1.005,'normal2':1.0034,'slight':1.002,'moderate':1.007,'large':1.007,'circle':0.999},
    (180,200): {'normal':1.0053,'normal2':1.004,'slight':1.001,'moderate':1.006,'large':1.006,'circle':0.999},
    (200,300): {'normal':1.006,'normal2':1.0045,'slight':1.000,'moderate':1.005,'large':1.005,'circle':0.999},
}

def estimate_distance(pixel_height):
    if pixel_height <= CALIB_PIXELS[0]:
        k = (CALIB_DISTANCES[1]-CALIB_DISTANCES[0])/(CALIB_PIXELS[1]-CALIB_PIXELS[0])
        return float(CALIB_DISTANCES[0] + k*(pixel_height-CALIB_PIXELS[0]))
    if pixel_height >= CALIB_PIXELS[-1]:
        k = (CALIB_DISTANCES[-1]-CALIB_DISTANCES[-2])/(CALIB_PIXELS[-1]-CALIB_PIXELS[-2])
        return float(CALIB_DISTANCES[-1] + k*(pixel_height-CALIB_PIXELS[-1]))
    return float(np.interp(pixel_height, CALIB_PIXELS, CALIB_DISTANCES))

def get_compensation(distance_cm):
    for (lo, hi), tbl in COMPENSATION_TABLE.items():
        if lo <= distance_cm < hi:
            return tbl
    if distance_cm < 100:
        return COMPENSATION_TABLE[(100,120)]
    return COMPENSATION_TABLE[(200,300)]

def distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, np.float32)-np.asarray(p2, np.float32)))

def get_outer_frame_params(approx_outer):
    pts = approx_outer.reshape(4,2).astype(np.float32)
    xs = pts[np.argsort(pts[:,0])]
    return distance(xs[0],xs[1]), distance(xs[2],xs[3])

def order_points(pts):
    rect = np.zeros((4,2), np.float32)
    s = pts.sum(axis=1); d = np.diff(pts,axis=1).ravel()
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

# ==================== 新正方形检测器 ====================
# 透视图固定 20 pixel/cm；题目正方形边长 6~12 cm。
SCALE = 20.0
MIN_SIDE = 5.5*SCALE
MAX_SIDE = 12.5*SCALE

@dataclass
class SquareInfo:
    vertices: List[Tuple[int,int]]
    center: Tuple[int,int]
    avg_side_length: float
    score: float
    detection_method: str

all_squares: List[SquareInfo] = []

def reset_square_collection():
    global all_squares
    all_squares = []

def add_square_to_collection(vertices, avg_side, method, center=None, score=0.0):
    if center is None:
        center = tuple(np.mean(np.asarray(vertices),axis=0).astype(int))
    s = SquareInfo([tuple(map(int,p)) for p in vertices], tuple(center),
                   float(avg_side), float(score), method)
    all_squares.append(s)
    return s

def get_all_squares():
    return all_squares

def _cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])

def _poly_mask(shape, pts, shrink=0):
    p = np.asarray(pts,np.float32)
    if shrink:
        c=p.mean(axis=0); p=c+(p-c)*shrink
    m=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(m,np.round(p).astype(np.int32),255)
    return m

def _band_masks(shape, pts, band=4):
    p=np.round(np.asarray(pts)).astype(np.int32)
    line=np.zeros(shape,np.uint8)
    cv2.polylines(line,[p],True,255,1)
    kin=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band+1,2*band+1))
    wide=cv2.dilate(line,kin)
    inside=_poly_mask(shape,pts,0.96)
    return cv2.bitwise_and(wide,inside), cv2.bitwise_and(wide,cv2.bitwise_not(inside))

def _ratio(binary, mask):
    n=cv2.countNonZero(mask)
    return cv2.countNonZero(cv2.bitwise_and(binary,mask))/n if n else 0.0

def _candidate_score(binary, contour_edge, pts, anchor_count):
    h,w=binary.shape
    p=np.asarray(pts,np.float32)
    if np.any(p[:,0]<1) or np.any(p[:,0]>=w-1) or np.any(p[:,1]<1) or np.any(p[:,1]>=h-1):
        return -1.0
    # 实心方块内部原则上全黑；缩小区域避免抗锯齿边缘。
    inner=_poly_mask(binary.shape,p,0.90)
    fill=_ratio(binary,inner)
    if fill < 0.82:
        return -1.0
    # 可见轮廓支持率。重叠边可以消失，所以只作为软分数。
    line=np.zeros(binary.shape,np.uint8)
    cv2.polylines(line,[np.round(p).astype(np.int32)],True,255,2)
    near=cv2.dilate(contour_edge,np.ones((7,7),np.uint8))
    support=_ratio(near,line)
    # 边界内侧应黑；外侧白的比例越高，说明越像真实外露边。
    ib,ob=_band_masks(binary.shape,p,4)
    inside_black=_ratio(binary,ib)
    outside_white=1.0-_ratio(binary,ob)
    score=0.50*fill+0.18*inside_black+0.22*support+0.10*outside_white
    score += min(anchor_count,3)*0.025
    return score

def _corner_distance_count(pts, corner_points, tol=13):
    if len(corner_points)==0: return 0
    c=np.asarray(corner_points,np.float32)
    return sum(float(np.min(np.linalg.norm(c-q,axis=1)))<=tol for q in np.asarray(pts))

def _deduplicate(candidates):
    candidates=sorted(candidates,key=lambda x:x.score,reverse=True)
    kept=[]
    for c in candidates:
        cp=np.asarray(c.vertices,np.float32)
        duplicate=False
        for k in kept:
            kp=np.asarray(k.vertices,np.float32)
            dc=distance(c.center,k.center)
            side=max(c.avg_side_length,k.avg_side_length)
            if dc < 0.16*side and abs(c.avg_side_length-k.avg_side_length)<0.12*side:
                duplicate=True; break
        if not duplicate: kept.append(c)
    return kept

def detect_squares_robust(binary):
    """在透视校正后的 inner 二值图上检测分离/局部重叠的实心正方形。"""
    reset_square_collection()
    b=(binary>0).astype(np.uint8)*255
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>0.45*MIN_SIDE*MIN_SIDE]
    edge=np.zeros_like(b); cv2.drawContours(edge,contours,-1,255,2)
    candidates=[]
    global_lengths=[]
    component_data=[]

    for cnt in contours:
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.006*peri,True).reshape(-1,2).astype(np.float32)
        if len(approx)<4: continue
        # 去除很短的阶梯噪声边。
        clean=[]
        for p in approx:
            if not clean or distance(p,clean[-1])>=8: clean.append(p)
        if len(clean)>=4 and distance(clean[0],clean[-1])<8: clean.pop()
        v=np.asarray(clean,np.float32)
        if len(v)<4: continue
        area=cv2.contourArea(v.reshape(-1,1,2),oriented=True)
        orient=1.0 if area>0 else -1.0
        corners=[]
        for i,p in enumerate(v):
            a=v[(i-1)%len(v)]-p; d=v[(i+1)%len(v)]-p
            la=np.linalg.norm(a); ld=np.linalg.norm(d)
            if la<10 or ld<10: continue
            cosine=float(np.dot(a,d)/(la*ld))
            angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
            # OpenCV 图像坐标中，用轮廓方向判断凸角。
            turn=_cross(v[(i+1)%len(v)]-p, v[(i-1)%len(v)]-p)
            convex=(turn*orient)>0
            if convex and 76<=angle<=104:
                corners.append((p,a/la,d/ld,la,ld))
                for L in (la,ld):
                    if MIN_SIDE<=L<=MAX_SIDE: global_lengths.append(float(L))
        simple_component = False

        # 完整、彼此分离的四边形：minAreaRect 是最稳的主通道。
        rect=cv2.minAreaRect(cnt); box=cv2.boxPoints(rect)
        rw,rh=rect[1]
        if min(rw,rh)>=MIN_SIDE and max(rw,rh)<=MAX_SIDE and min(rw,rh)>0:
            square_err=abs(rw-rh)/max(rw,rh)
            rectangularity=cv2.contourArea(cnt)/(rw*rh)
            if square_err<0.13 and rectangularity>0.86:
                s=(rw+rh)/2
                sc=_candidate_score(b,edge,box,4)
                if sc>0.65:
                    simple_component = True
                    candidates.append(SquareInfo([tuple(map(int,p)) for p in box],
                        tuple(np.mean(box,axis=0).astype(int)),s,sc+0.08,'complete_contour'))
        component_data.append((cnt,v,corners,simple_component))

    # 对长度作小范围聚类，抑制轮廓近似导致的随机候选长度。
    lengths=[]
    for L in sorted(global_lengths):
        if not lengths or abs(L-lengths[-1])>8: lengths.append(L)
        else: lengths[-1]=(lengths[-1]+L)/2

    for cnt,v,corners,simple_component in component_data:
        # 完整独立正方形已经由 minAreaRect 精确处理，不再套用其他方块的
        # 边长，避免在大实心方块内部虚构一个更小方块。
        if simple_component:
            continue
        corner_pts=[x[0] for x in corners]
        for p,u,vdir,lu,lv in corners:
            local=[L for L in (lu,lv) if MIN_SIDE<=L<=MAX_SIDE]
            # 完整边长度 + 全局其他角提供的长度；重叠时某个角的边会被截短。
            trial=[]
            for L in local+lengths:
                if MIN_SIDE<=L<=MAX_SIDE and all(abs(L-x)>6 for x in trial): trial.append(L)
            for s in trial:
                # u、vdir 是从角点沿两邻边向外的单位向量。
                pts=np.array([p,p+u*s,p+(u+vdir)*s,p+vdir*s],np.float32)
                anchors=_corner_distance_count(pts,corner_pts)
                # 至少一个真实凸角（当前 p）；没有第二角时要求分数更高。
                sc=_candidate_score(b,edge,pts,anchors)
                threshold=0.70 if anchors>=2 else 0.78
                if sc>=threshold:
                    candidates.append(SquareInfo([tuple(map(int,q)) for q in pts],
                        tuple(np.mean(pts,axis=0).astype(int)),float(s),sc,'corner_inference'))

    result=_deduplicate(candidates)
    # 再去掉被同中心、更可信小方形解释的弱候选；保留真正不同中心的重叠方块。
    result=[s for s in result if s.score>=0.65]
    all_squares.extend(result)
    return result,b

def draw_square_on_original(img, square, border_pix, M_inv, color, text=None):
    pts=np.asarray(square.vertices,np.float32)
    pts[:,0]+=border_pix; pts[:,1]+=border_pix
    orig=cv2.perspectiveTransform(pts.reshape(-1,1,2),M_inv).reshape(-1,2).astype(np.int32)
    cv2.polylines(img,[orig],True,color,3)
    c=tuple(np.mean(orig,axis=0).astype(int))
    if text:
        cv2.putText(img,text,c,cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)

# ==================== 主处理：透视和 inner 获取逻辑保持原样 ====================
def process_frame(frame):
    img=frame.copy()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,11,2)
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(img,'No A4 paper',(20,80),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2); return img
    a4_outer=max(contours,key=cv2.contourArea)
    peri=cv2.arcLength(a4_outer,True)
    approx=cv2.approxPolyDP(a4_outer,.02*peri,True)
    if len(approx)!=4:
        cv2.putText(img,'Not rectangle',(20,80),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2); return img
    cv2.drawContours(img,[a4_outer],-1,(0,255,0),2)

    left_pix,right_pix=get_outer_frame_params(approx)
    outer_height=(left_pix+right_pix)/2.0
    distance_cm=estimate_distance(outer_height) if outer_height>0 else 0.0
    cv2.putText(img,f'Dist: {distance_cm:.1f}cm',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    # 下列透视变换、尺度及 2 cm 裁剪与 basic_pi.py 完全相同。
    src_pts=order_points(approx.reshape(4,2).astype(np.float32))
    scale=20
    dst_width=int(21*scale); dst_height=int(29.7*scale)
    dst_pts=np.float32([[0,0],[dst_width,0],[dst_width,dst_height],[0,dst_height]])
    M=cv2.getPerspectiveTransform(src_pts,dst_pts)
    M_inv=np.linalg.inv(M)
    warped=cv2.warpPerspective(gray,M,(dst_width,dst_height))
    warped_blur=cv2.GaussianBlur(warped,(5,5),0)

    # 2 cm 裁剪范围保持原逻辑不变；但内部实心图形改从灰度图用 Otsu 提取。
    border_pix=int(2*scale)
    inner_gray=warped_blur[border_pix:dst_height-border_pix,
                           border_pix:dst_width-border_pix]
    otsu_value,inner_binary=cv2.threshold(
        inner_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    inner_binary=cv2.morphologyEx(inner_binary,cv2.MORPH_OPEN,
                                  np.ones((3,3),np.uint8))
    inner_binary=cv2.morphologyEx(inner_binary,cv2.MORPH_CLOSE,
                                  np.ones((5,5),np.uint8))

    # 单帧调试文件：用于定位外框、透视、裁剪和二值化问题。
    cv2.imwrite('debug_01_gray.jpg',gray)
    cv2.imwrite('debug_02_outer_thresh.jpg',thresh)
    cv2.imwrite('debug_03_warped.jpg',warped)
    cv2.imwrite('debug_04_inner_gray.jpg',inner_gray)
    cv2.imwrite('debug_05_inner_binary.jpg',inner_binary)
    print(f'Otsu threshold: {otsu_value:.1f}')

    squares,clean_binary=detect_squares_robust(inner_binary)
    cv2.imwrite('debug_06_clean_binary.jpg',clean_binary)
    if not squares:
        cv2.putText(img,'No valid square',(20,85),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
        return img
    smallest=min(squares,key=lambda s:s.avg_side_length)
    comp=get_compensation(distance_cm)
    # 透视平面已经是厘米标尺，补偿只作最后的小修正。normal2 可实测后关闭。
    side_cm=smallest.avg_side_length/scale*comp['normal2']
    for sq in squares:
        selected=(sq is smallest)
        color=(0,0,255) if selected else (255,0,0)
        txt=f'{sq.avg_side_length/scale:.2f}' if selected else None
        draw_square_on_original(img,sq,border_pix,M_inv,color,txt)
    cv2.putText(img,f'Min square: {side_cm:.2f}cm',(20,85),
                cv2.FONT_HERSHEY_SIMPLEX,.85,(0,255,255),2)
    cv2.putText(img,f'Count: {len(squares)}',(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
    print(f'D={distance_cm:.1f} cm, squares={[round(s.avg_side_length/scale,2) for s in squares]}, min={side_cm:.2f} cm')
    return img

if __name__=='__main__':
    # 单帧调试模式：启动相机后等待曝光稳定，只拍摄和检测一次。
    picam2=Picamera2()
    config=picam2.create_still_configuration(
        main={'size':(1280,720),'format':'RGB888'})
    picam2.configure(config)
    picam2.start()
    print('Camera started; waiting for exposure/AWB...')
    time.sleep(2.0)

    try:
        frame_rgb=picam2.capture_array()
        frame_bgr=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite('debug_00_original.jpg',frame_bgr)
        print(f'Captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}')

        start=time.perf_counter()
        result=process_frame(frame_bgr)
        elapsed=(time.perf_counter()-start)*1000.0
        cv2.imwrite('debug_07_result.jpg',result)
        print(f'Processing time: {elapsed:.1f} ms')
        print('Saved debug_00_original.jpg through debug_07_result.jpg')

        # 有桌面环境时显示；通过 SSH 无显示环境时，注释下面四行即可。
        cv2.imshow('Original',frame_bgr)
        cv2.imshow('Detection result',result)
        print('Press any key to exit.')
        cv2.waitKey(0)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
