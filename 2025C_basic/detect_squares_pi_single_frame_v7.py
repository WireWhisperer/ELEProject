import cv2
import numpy as np
import os
import time
import serial
from dataclasses import dataclass
from typing import List, Tuple
from picamera2 import Picamera2

# ==================== 原距离标定（保持不变） ====================
IMAGES_DIR = 'images'

DISTANCE_CALIBRATION = [
    (232.21, 200),(237.94, 195),(243.98, 190),(250.35, 185),(257.06, 180),
    (264.21, 175),(271.52, 170),(279.79, 165),(288.04, 160),(297.37, 155),
    (306.64, 150),(318.05, 145),(328.08, 140),(341.49, 135),(353.16, 130),
    (367.41, 125),(383.57, 120),(404.31, 115),(421.94, 110),(441.94, 105),(464.26, 100),
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


def find_a4_outer_quad(binary):
    """从所有四边形中选择面积最大的合格 A4 外边缘，避免误取黑框内边缘。"""
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    h,w=binary.shape[:2]
    image_area=float(h*w)
    candidates=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<0.025*image_area or area>0.90*image_area:
            continue
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        if len(approx)!=4 or not cv2.isContourConvex(approx):
            continue
        q=order_points(approx.reshape(4,2).astype(np.float32))
        width=(distance(q[0],q[1])+distance(q[3],q[2]))/2.0
        height=(distance(q[0],q[3])+distance(q[1],q[2]))/2.0
        if min(width,height)<1:
            continue
        ratio=max(width,height)/min(width,height)
        if not 1.20<=ratio<=1.68:
            continue
        rectangularity=area/(width*height)
        if rectangularity<0.68:
            continue
        # A4 为竖放；增加面积权重，同时轻微偏好接近 sqrt(2) 的候选。
        score=area*(1.0-0.20*abs(ratio-np.sqrt(2.0)))
        candidates.append((score,area,cnt,approx,ratio))
    if not candidates:
        return None,None
    # 外边缘面积一定大于同一黑框的内边缘。
    candidates.sort(key=lambda x:x[0],reverse=True)
    best=candidates[0]
    print(f'A4 candidate: area={best[1]:.1f}, aspect={best[4]:.3f}')
    return best[2],best[3]

def estimate_black_border_widths(warped_gray, dark_threshold=130):
    """估计透视图上下左右黑边厚度；正常应接近 40 px。仅用于调试输出。"""
    h,w=warped_gray.shape
    def run(values):
        values=np.asarray(values)
        dark=values<dark_threshold
        # 允许最外侧因角点插值出现少量亮像素，从前 12px 内第一个暗点开始。
        ids=np.flatnonzero(dark[:max(12,len(dark)//5)])
        if len(ids)==0:return 0
        start=int(ids[0]); n=0
        for x in dark[start:]:
            if x:n+=1
            elif n>=3:break
        return n
    return [run(warped_gray[:,w//2]),run(warped_gray[::-1,w//2]),
            run(warped_gray[h//2,:]),run(warped_gray[h//2,::-1])]

# ==================== 新正方形检测器 ====================
# 透视图固定 20 pixel/cm；题目正方形边长 6~12 cm。
SCALE = 20.0
# 尺寸标定：(算法原始测量值cm, 实际边长cm)。
# 当前三点来自 6.0、8.0、10.5 cm 目标；后续可继续加入 7/9/10/11/12 cm 标定点。
SIZE_CALIBRATION = [
    (5.95, 6.00),
    (7.85, 8.00),
    (10.57, 10.50),
]
SIZE_CALIBRATION.sort(key=lambda x: x[0])
SIZE_RAW = np.array([x for x, _ in SIZE_CALIBRATION], dtype=np.float32)
SIZE_REAL = np.array([y for _, y in SIZE_CALIBRATION], dtype=np.float32)

def correct_square_size(raw_cm):
    """按尺寸标定表分段线性插值；超出标定范围时使用端点斜率外推。"""
    raw_cm=float(raw_cm)
    if raw_cm<=SIZE_RAW[0]:
        x1,x2=float(SIZE_RAW[0]),float(SIZE_RAW[1])
        y1,y2=float(SIZE_REAL[0]),float(SIZE_REAL[1])
        return y1+(y2-y1)/(x2-x1)*(raw_cm-x1)
    if raw_cm>=SIZE_RAW[-1]:
        x1,x2=float(SIZE_RAW[-2]),float(SIZE_RAW[-1])
        y1,y2=float(SIZE_REAL[-2]),float(SIZE_REAL[-1])
        return y2+(y2-y1)/(x2-x1)*(raw_cm-x2)
    return float(np.interp(raw_cm,SIZE_RAW,SIZE_REAL))


# A4支架局部遮挡黑边造成的最终尺寸补偿。
# 根据当前7组测试数据，以最小二乘方式拟合得到。
FRAME_COMPENSATION = 0.982

def get_final_square_size(raw_cm):
    """尺寸标定插值后，再应用支架遮挡补偿；仅用于最终显示，不参与几何检测。"""
    calibrated_cm=correct_square_size(raw_cm)
    return float(calibrated_cm*FRAME_COMPENSATION)
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

def merge_close_points(points, threshold=10.0):
    """合并凸包和直角检测中位置接近的重复点。"""
    merged=[]
    for p in points:
        p=np.asarray(p,np.float32)
        hit=-1
        for i,q in enumerate(merged):
            if distance(p,q)<=threshold:
                hit=i; break
        if hit>=0:
            merged[hit]=(merged[hit]+p)*0.5
        else:
            merged.append(p.copy())
    return merged

def generate_diagonal_square_candidates(binary, contour_edge, point_candidates):
    """用两个外露的相对顶点恢复被遮挡一个顶点的正方形。"""
    result=[]
    if len(point_candidates)<2:
        return result
    pts=np.asarray(point_candidates,np.float32)
    dmin=MIN_SIDE*np.sqrt(2.0)
    dmax=MAX_SIDE*np.sqrt(2.0)
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            p1,p2=pts[i],pts[j]
            diag=distance(p1,p2)
            if not (dmin<=diag<=dmax):
                continue
            center=(p1+p2)*0.5
            half=(p2-p1)*0.5
            perp=np.array([-half[1],half[0]],np.float32)
            p3=center+perp
            p4=center-perp
            square=np.array([p1,p3,p2,p4],np.float32)
            cx,cy=np.round(center).astype(int)
            if not (0<=cy<binary.shape[0] and 0<=cx<binary.shape[1]) or binary[cy,cx]==0:
                continue
            anchors=_corner_distance_count(square,pts,tol=14)
            if anchors<2:
                continue
            score=_candidate_score(binary,contour_edge,square,anchors)
            # 对角点有两个真实顶点支持，允许被重叠覆盖的边界不可见。
            if score>=0.67:
                result.append(SquareInfo(
                    [tuple(np.round(q).astype(int)) for q in square],
                    tuple(np.round(center).astype(int)),
                    float(diag/np.sqrt(2.0)),float(score),'opposite_corners'))
    return result

def edge_support_values(contour_edge, vertices, tolerance=5):
    """分别计算候选四条边与真实组合外轮廓的重合比例。"""
    near=cv2.dilate(contour_edge,np.ones((2*tolerance+1,2*tolerance+1),np.uint8))
    p=np.asarray(vertices,np.float32)
    values=[]
    for i in range(4):
        line=np.zeros(contour_edge.shape,np.uint8)
        a=tuple(np.round(p[i]).astype(int)); b=tuple(np.round(p[(i+1)%4]).astype(int))
        cv2.line(line,a,b,255,2,cv2.LINE_AA)
        values.append(_ratio(near,line))
    return values

def candidate_geometry_valid(binary, contour_edge, square):
    """用内部填充和逐边轮廓证据过滤凸包点随意配对产生的假方块。"""
    p=np.asarray(square.vertices,np.float32)
    h,w=binary.shape
    if np.any(p[:,0]<1) or np.any(p[:,0]>=w-1) or np.any(p[:,1]<1) or np.any(p[:,1]>=h-1):
        return False,0.0,[]
    mask=_poly_mask(binary.shape,p,0.97)
    fill=_ratio(binary,mask)
    supports=edge_support_values(contour_edge,p,4)
    ordered=sorted(supports,reverse=True)
    if square.detection_method=='complete_contour':
        valid=fill>=0.82 and np.mean(supports)>=0.52
    elif square.detection_method=='two_visible_edges':
        valid=fill>=0.84 and ordered[0]>=0.48 and ordered[1]>=0.32
    elif square.detection_method=='one_visible_edge':
        valid=fill>=0.90 and ordered[0]>=0.62 and ordered[1]>=0.18
    else:  # opposite_corners
        # 被遮挡时允许两条边消失，但至少两条边必须有真实轮廓证据。
        valid=fill>=0.86 and ordered[0]>=0.42 and ordered[1]>=0.22
    evidence=0.65*fill+0.35*float(np.mean(ordered[:2]))
    return valid,evidence,supports

def polygon_iou(a,b,shape):
    ma=_poly_mask(shape,a,1.0); mb=_poly_mask(shape,b,1.0)
    inter=cv2.countNonZero(cv2.bitwise_and(ma,mb))
    union=cv2.countNonZero(cv2.bitwise_or(ma,mb))
    return inter/union if union else 0.0

def suppress_near_duplicates(candidates,shape):
    """按多边形IoU、中心、边长去除同一方块的多种恢复结果。"""
    ordered=sorted(candidates,key=lambda s:s.score,reverse=True)
    kept=[]
    for c in ordered:
        dup=False
        for k in kept:
            side=max(c.avg_side_length,k.avg_side_length)
            if (distance(c.center,k.center)<0.18*side and
                abs(c.avg_side_length-k.avg_side_length)<0.14*side and
                polygon_iou(c.vertices,k.vertices,shape)>0.65):
                dup=True; break
        if not dup: kept.append(c)
    return kept

def select_square_explanation(binary,candidates,beam_width=48,max_candidates=18):
    """选择最少的一组正方形，使其并集尽可能解释实际黑色区域。"""
    if not candidates:return []
    obs=(binary>0).astype(np.uint8)*255
    obs_n=max(cv2.countNonZero(obs),1)
    # 优先保留边界证据强的候选，限制树莓派搜索规模。
    candidates=sorted(candidates,key=lambda s:s.score,reverse=True)[:max_candidates]
    masks=[_poly_mask(obs.shape,s.vertices,1.0) for s in candidates]
    empty=np.zeros_like(obs)
    def loss(union,ids):
        fn=cv2.countNonZero(cv2.bitwise_and(obs,cv2.bitwise_not(union)))/obs_n
        fp=cv2.countNonZero(cv2.bitwise_and(union,cv2.bitwise_not(obs)))/obs_n
        # 数量惩罚会自动删除位于真实方块并集内部、没有新增解释能力的假候选。
        return fn+1.65*fp+0.018*len(ids)
    states=[(loss(empty,()),empty,())]
    for idx,mask in enumerate(masks):
        expanded=list(states)
        for _,union,ids in states:
            nu=cv2.bitwise_or(union,mask)
            nids=ids+(idx,)
            expanded.append((loss(nu,nids),nu,nids))
        expanded.sort(key=lambda x:x[0])
        # 对相同候选集合规模保留若干方案，避免单一路径过早锁死。
        states=expanded[:beam_width]
    best=min(states,key=lambda x:x[0])
    selected=[candidates[i] for i in best[2]]
    print(f'Global square explanation: candidates={len(candidates)}, selected={len(selected)}, loss={best[0]:.4f}')
    return selected

def detect_squares_robust(binary):
    """在透视校正后的 inner 二值图上检测分离/局部重叠的实心正方形。"""
    reset_square_collection()
    b=(binary>0).astype(np.uint8)*255
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
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
            valid_u=MIN_SIDE<=lu<=MAX_SIDE
            valid_v=MIN_SIDE<=lv<=MAX_SIDE
            trial=[]
            if valid_u and valid_v:
                # 同一个真实正方形角点的两条边应等长；差异过大通常是重叠交点。
                if abs(lu-lv)/max(lu,lv)<=0.18:
                    trial=[(float((lu+lv)/2.0),'two_visible_edges',0.68)]
            elif valid_u:
                trial=[(float(lu),'one_visible_edge',0.82)]
            elif valid_v:
                trial=[(float(lv),'one_visible_edge',0.82)]
            for s,method,base_threshold in trial:
                # u、vdir 是从角点沿两邻边向外的单位向量。
                pts=np.array([p,p+u*s,p+(u+vdir)*s,p+vdir*s],np.float32)
                anchors=_corner_distance_count(pts,corner_pts)
                # 至少一个真实凸角（当前 p）；没有第二角时要求分数更高。
                sc=_candidate_score(b,edge,pts,anchors)
                threshold=max(0.64,base_threshold-(0.04 if anchors>=2 else 0.0))
                if sc>=threshold:
                    candidates.append(SquareInfo([tuple(map(int,q)) for q in pts],
                        tuple(np.mean(pts,axis=0).astype(int)),float(s),sc,method))

        # 对角恢复不能只依赖标准90度角：阈值化后大方块左右尖角可能被削平。
        # 将轮廓凸包顶点与标准直角点合并，再搜索相对顶点。
        hull=cv2.convexHull(cnt,returnPoints=True)
        hull_peri=cv2.arcLength(hull,True)
        hull_approx=cv2.approxPolyDP(hull,0.008*hull_peri,True)
        hull_points=hull_approx.reshape(-1,2).astype(np.float32).tolist()
        right_angle_points=[c[0] for c in corners]
        diagonal_points=merge_close_points(hull_points+right_angle_points,threshold=10.0)
        candidates.extend(generate_diagonal_square_candidates(b,edge,diagonal_points))

    result=_deduplicate(candidates)
    result=[s for s in result if s.score>=0.65 and MIN_SIDE<=s.avg_side_length<=MAX_SIDE]
    verified=[]
    for sq in result:
        valid,evidence,supports=candidate_geometry_valid(b,edge,sq)
        if valid:
            # 将逐边证据融合到原评分，供全局搜索排序。
            sq.score=0.55*sq.score+0.45*evidence
            verified.append(sq)
        else:
            print(f'Rejected: side={sq.avg_side_length/SCALE:.2f}cm method={sq.detection_method} supports={[round(x,2) for x in supports]}')
    verified=suppress_near_duplicates(verified,b.shape)
    result=select_square_explanation(b,verified)
    result=sorted(result,key=lambda s:s.avg_side_length)
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
    a4_outer,approx=find_a4_outer_quad(thresh)
    if a4_outer is None or approx is None:
        cv2.putText(img,'No valid A4 outer frame',(20,80),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        return img, None, None
    # 直接画实际用于透视的四边形，便于确认选中的是黑框外缘。
    cv2.drawContours(img,[approx],-1,(0,255,0),2)

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
    warped_blur=cv2.GaussianBlur(warped,(3,3),0)

    # 2 cm 裁剪范围保持原逻辑不变；但内部实心图形改从灰度图用 Otsu 提取。
    border_pix=int(2*scale)
    inner_gray=warped_blur[border_pix:dst_height-border_pix,
                           border_pix:dst_width-border_pix]
    otsu_value,inner_binary=cv2.threshold(
        inner_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # 不做开运算，避免腐蚀旋转正方形的尖角；仅用小核填补微小断裂。
    inner_binary=cv2.morphologyEx(inner_binary,cv2.MORPH_CLOSE,
                                  np.ones((3,3),np.uint8))

    # 单帧调试文件：用于定位外框、透视、裁剪和二值化问题。
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_01_gray.jpg'),gray)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_02_outer_thresh.jpg'),thresh)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_03_warped.jpg'),warped)
    border_widths=estimate_black_border_widths(warped)
    print(f'Warped black border widths [top,bottom,left,right]: {border_widths}; expected about {border_pix}px')
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_04_inner_gray.jpg'),inner_gray)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_05_inner_binary.jpg'),inner_binary)
    print(f'Otsu threshold: {otsu_value:.1f}')

    squares,clean_binary=detect_squares_robust(inner_binary)
    cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_06_clean_binary.jpg'),clean_binary)
    if not squares:
        cv2.putText(img,'No valid square',(20,85),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
        return img, distance_cm, None
    smallest=min(squares,key=lambda s:s.avg_side_length)
    # 距离尚未标定时不应用距离补偿，避免左上结果与轮廓文字不一致。
    raw_side_cm=smallest.avg_side_length/scale
    side_cm=get_final_square_size(raw_side_cm)
    for sq in squares:
        selected=(sq is smallest)
        color=(0,0,255) if selected else (255,0,0)
        raw_cm=sq.avg_side_length/scale
        corrected_cm=get_final_square_size(raw_cm)
        txt=f'{corrected_cm:.2f}' if selected else None
        draw_square_on_original(img,sq,border_pix,M_inv,color,txt)
    cv2.putText(img,f'Min square: {side_cm:.2f}cm',(20,85),
                cv2.FONT_HERSHEY_SIMPLEX,.85,(0,255,255),2)
    cv2.putText(img,f'Count: {len(squares)}',(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
    print(f'D={distance_cm:.1f} cm, squares={[round(get_final_square_size(s.avg_side_length/scale),2) for s in squares]}, min={side_cm:.2f} cm')
    for i,sq in enumerate(squares,1):
        raw_cm=sq.avg_side_length/scale
        calibrated_cm=correct_square_size(raw_cm)
        final_cm=get_final_square_size(raw_cm)
        print(f'  square[{i}]: raw={raw_cm:.3f}cm, calibrated={calibrated_cm:.3f}cm, final={final_cm:.3f}cm, method={sq.detection_method}, score={sq.score:.3f}')
    return img, distance_cm, side_cm

if __name__=='__main__':
    # 单帧调试模式：启动相机后等待曝光稳定，只拍摄和检测一次。
    picam2=Picamera2()
    config=picam2.create_still_configuration(
        main={'size':(1280,720),'format':'RGB888'})
    picam2.configure(config)
    picam2.start()
    print('Camera started; waiting for exposure/AWB...')
    time.sleep(2.0)

    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)

    try:
        frame_rgb=picam2.capture_array()
        frame_bgr=cv2.cvtColor(frame_rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_00_original.jpg'),frame_bgr)
        print(f'Captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}')

        start=time.perf_counter()
        result, distance_cm, side_cm = process_frame(frame_bgr)
        elapsed=(time.perf_counter()-start)*1000.0
        cv2.imwrite(os.path.join(IMAGES_DIR, 'debug_07_result.jpg'),result)
        print(f'Processing time: {elapsed:.1f} ms')
        print('Saved debug_00_original.jpg through debug_07_result.jpg')

        if distance_cm is not None and side_cm is not None:
            msg = f"D={distance_cm:.1f},x={side_cm:.2f},type=square\r\n"
            ser.write(msg.encode('utf-8'))
            print(f'Sent via serial: {msg.strip()}')

        print('Waiting 1s before exit...')
        time.sleep(1.0)
    finally:
        ser.close()
        picam2.stop()
        cv2.destroyAllWindows()
