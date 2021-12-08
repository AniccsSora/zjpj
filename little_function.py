def bbox_isOverlap(bbox1, bbox2):
    """
    檢查兩個 bbox 是否有重疊。
    ---
    @param bbox1: (x1,y1,x2,y2)
    @param bbox2: (x1,y1,x2,y2)
    @return: boolean
    算法，利用x軸, y軸是否由重疊來達成檢查。
    """
    assert len(bbox1) == len(bbox2)
    assert len(bbox1) == 4

    xaxis_mapping_isOverlap = None
    # X 軸投影判斷 (2端點均不在另一條範圍內。)
    if bbox1[0] < bbox1[2]:
        B1x1, B1x2 = bbox1[0], bbox1[2]
    else:
        B1x1, B1x2 = bbox1[2], bbox1[0]

    if bbox2[0] < bbox2[2]:
        B2x1, B2x2 = bbox2[0], bbox2[2]
    else:
        B2x1, B2x2 = bbox2[2], bbox2[0]

    if (B2x1 < B1x1 < B2x2) or (B2x1 < B1x2 < B2x2):
        # B1 的兩端點是否在 B2的線段內?
        xaxis_mapping_isOverlap = True
    elif (B1x1 < B2x1 < B1x2) or (B1x1 < B2x2 < B1x2):
        # B2 的兩端點是否在 B1的線段內?
        xaxis_mapping_isOverlap = True
    else:
        # 只要有一軸是不重疊的，那麼這兩個框框絕對是不重疊。
        return False

    # x-axis 檢核完畢
    # -------------------------
    # Y 軸投影判斷 (2端點均不在另一條範圍內。)
    if bbox1[1] < bbox1[3]:
        B1y1, B1y2 = bbox1[1], bbox1[3]
    else:
        B1y1, B1y2 = bbox1[3], bbox1[1]

    if bbox2[1] < bbox2[3]:
        B2y1, B2y2 = bbox2[1], bbox2[3]
    else:
        B2y1, B2y2 = bbox2[3], bbox2[1]

    yaxis_mapping_isOverlap = None
    if (B2y1 < B1y1 < B2y2) or (B2y1 < B1y2 < B2y2):
        yaxis_mapping_isOverlap = True
    elif (B1y1 < B2y1 < B1y2) or (B1y1 < B2y2 < B1y2):
        yaxis_mapping_isOverlap = True
    else:
        return False

    # X 軸 Y 軸 的投影皆需要重疊才被視為 bbox 重疊。
    if xaxis_mapping_isOverlap and yaxis_mapping_isOverlap:
        return True
    else:
        return False


def cutting_cube(w_h, cube_size, overlap = 1.0):
    """
    在給定的範圍內切出正方形的框框，回傳座標組
    @param w_h: 寬跟長
    @param cube_size: 方框的大小(正方形)
    @param overlap: 重疊程度。(float)，1: 剛好不重疊。
    @return: 生成器(generate)
    """
    x_shift, y_shift = int(cube_size * overlap), int(cube_size * overlap)
    x, y = 0, 0
    w, h = w_h[0], w_h[1]
    while y+cube_size <= h:
        while x+cube_size <= w:
            yield tuple((x, y, x+cube_size, y+cube_size))
            x = x+x_shift
        y = y+y_shift
        x = 0


def cutting_cube_include_surplus(w_h, cube_size, overlap=0.9):
    """
    Note:
    在給定的範圍內切出正方形的框框，回傳座標組，並且會回傳最後仍然剩下的座標組。
    最後一次的迭帶 stride!=cube_size.
    對任何圖片大小，總有機會在每一 列 or 行 有剩下且不足以補滿cube_size的殘邊，
    這個 function 將會將其納入回傳座標組。
    ---
    @param w_h: 圖片寬跟長:(tuple)。
    @param cube_size: 方框的大小(正方形):(int)。
    @param overlap: 重疊程度。(float)，1: 剛好不重疊。
    @return: 生成器(generate)。
    """
    assert overlap <= 1.0 and overlap >= 0.01
    x_shift, y_shift = int(cube_size * overlap), int(cube_size * overlap)
    x, y = 0, 0
    w, h = w_h[0], w_h[1]
    while y+cube_size <= h:
        while x+cube_size <= w:
            yield tuple((x, y, x+cube_size, y+cube_size))
            x = x+x_shift
        else:
            if x != w-1:  # 去除cube_size整除時會重複最後一個row的問題。
                yield tuple((w-cube_size, y, x + cube_size, y + cube_size))
                x = x + x_shift
        y = y+y_shift
        x = 0
    else:
        while x+cube_size <= w:
            yield tuple((x, h-cube_size, x+cube_size, y+cube_size))
            x = x+x_shift
        else:
            if x != w-1:  # 去除cube_size整除時會重複最後一個row的問題。
                yield tuple((w - cube_size, h-cube_size, x + cube_size, y + cube_size))
                x = x + x_shift

def analysis_yolo_row_data(rows, w ,h):
    """
    回傳 x1,y1,x2,y2 座標組 (int)
    @param rows: yolo format bboxes.
    @return: [tuple(x1,y1,x2,y2), ... ]
    """
    res = []

    for row in rows:
        x, y, bbox_w, bbox_h = row[1:]  # 只拿 bbox 參數
        # 轉成 整數 座標
        x, y, bbox_w, bbox_h = float(x), float(y), float(bbox_w), float(bbox_h)

        x1 = round((x - bbox_w / 2) * w)
        y1 = round((y - bbox_h / 2) * h)
        x2 = round((x + bbox_w / 2) * w)
        y2 = round((y + bbox_h / 2) * h)
        res.append((x1, y1, x2, y2))

    return res


if __name__ == "__main__":
    cube_generator = cutting_cube((100, 100), 10)
    for idx,cube in enumerate(cube_generator):
        print(idx, cube)