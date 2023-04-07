from copy import deepcopy
import numpy as np
import cv2



class CodeBook:
    """
    FRAME_PER_RELOCATION    : int   every # frames, __relocate CodeWord's order
    MNRL_LIM                : int   every background CodeWord should recur at least every {MNRL_LIM} frames.
    """

    FRAME_PER_RELOCATION: int
    MNRL_LIM: int

    def __init__(self, training_frame: int, fpr: int = 20) -> None:
        self.cd_list: list[CodeWord] = []
        self.training_frame = training_frame
        CodeBook.MNRL_LIM = int(training_frame * 0.5)
        CodeBook.FRAME_PER_RELOCATION = fpr
        
    def training(self, seg: np.ndarray, t: int) -> None:  # State: Finish. Need Unit Test
        if not len(self.cd_list) :
            cwd = CodeWord(seg, t)
            self.cd_list.append(cwd)
            return

        if t % CodeBook.FRAME_PER_RELOCATION == 0:
            self.__relocate()

        row, col, _ = seg.shape
        waiting_for_check = np.ones(
            (row, col), dtype=bool
        )  # ? True : Elements waits for checking  False: Elements has been checked.
        for i in range(len(self.cd_list)):
            match_pixel, effective_nan = self.cd_list[i].tr_examine(
                seg, t, waiting_for_check
            )
            except_update = self.cd_list[i].update(
                seg, t, match_pixel
            )  # 沒有更新過得element 
            waiting_for_check = np.bitwise_and(
                np.bitwise_and(waiting_for_check, except_update),
                np.bitwise_not(effective_nan),
            )  # 沒有更新過得element 跟上一次待更新的 element 做and ， 取出還有哪些element 還要match
        
        if np.any(
            waiting_for_check
        ):  # ? waiting_for_check 如果存在一個 True (需要創建新的Cwd)，就創建一個新的
            new_cwd = CodeWord(seg, t, waiting_for_check)
            self.cd_list.append(new_cwd)

    
    def __mnrl_value_update(self) -> None:  # State: Finish. Unit Test Succeed
        for i in range(len(self.cd_list)):
            mnrl = self.cd_list[i].mnrl
            cmp = (
                self.training_frame
                + self.cd_list[i].first_times
                - self.cd_list[i].last_times
                - 1
            )
            self.cd_list[i].mnrl = np.where(cmp > mnrl, cmp, mnrl)

            # ! 會有Nan > 數字的問題，只要有 NAN 與數字相關的比較，回傳是False
            del_pixel = (
                self.cd_list[i].mnrl > CodeBook.MNRL_LIM
            )  # * 捨棄MNRL> MNRL_LIM的數值
            bgr_del = np.dstack((del_pixel, del_pixel, del_pixel))
            self.cd_list[i].avg_bgr = np.where(bgr_del, np.NAN, self.cd_list[i].avg_bgr)
            # * 將MNRL大於等於一半total frame 的 pixel 設成 np.NAN
            for attr_nm, value in vars(self.cd_list[i]).items():
                if attr_nm in CodeWord.ARR_2D_SET:
                    setattr(
                        self.cd_list[i], attr_nm, np.where(del_pixel, np.NAN, value)
                    )
    def __relocate(self) -> None:  # State: Finish. Unit Test Succeed
        for i in range(len(self.cd_list)):
            i_nan = np.isnan(self.cd_list[i].freq)
            if np.any(i_nan):
                self.cd_list[i].freq = np.where(i_nan, 0, self.cd_list[i].freq)

            j = i - 1
            while j > -1:
                cared_pixel = self.cd_list[j].freq < self.cd_list[j + 1].freq
                if not np.any(cared_pixel):
                    break

                cwd = deepcopy(self.cd_list[j + 1])
                bgr_pixel = np.dstack((cared_pixel, cared_pixel, cared_pixel))
                self.cd_list[j + 1].avg_bgr = np.where(
                    bgr_pixel, self.cd_list[j].avg_bgr, self.cd_list[j + 1].avg_bgr
                )
                self.cd_list[j].avg_bgr = np.where(
                    bgr_pixel, cwd.avg_bgr, self.cd_list[j].avg_bgr
                )

                # ? 同時迭代 j CodeWord 及 j+1 CodeWord 內部的 public variable。同時迭代時，變數迭代的順序是一樣的。
                # ?（avg_bgr -> bri_max -> bri_min -> freq -> mnrl -> first_times -> last_times)
                # * 將比較freq 較大的，移到cd_list 中 index 較小的位置
                for (attr_nm1, arr_1), (attr_nm2, arr_2) in zip(
                    vars(self.cd_list[j]).items(), vars(self.cd_list[j + 1]).items()
                ):
                    if attr_nm1 == attr_nm2 and attr_nm1 in CodeWord.ARR_2D_SET:
                        setattr(
                            self.cd_list[j + 1],
                            attr_nm1,
                            np.where(cared_pixel, arr_1, arr_2),
                        )
                        setattr(
                            self.cd_list[j],
                            attr_nm2,
                            np.where(cared_pixel, getattr(cwd, attr_nm2), arr_1),
                        )
                j -= 1
        
        length_cd_list = len(self.cd_list)
        for i in range(length_cd_list):
            self.cd_list[i].freq = np.where(
                self.cd_list[i].freq == 0, np.NAN, self.cd_list[i].freq
            )

    def __delete_empty_cwd(self) -> int:  # State: Finish. Unit Test Succeed 
        """回傳總共刪了多少"""
        total_num = len(self.cd_list)
        for i in range(total_num):
            cared_pixel = np.isnan(self.cd_list[i].mnrl)  # * NAN的區域
            if np.all(cared_pixel):  # 全False => 全部都是 NAN
                self.cd_list = self.cd_list[:i]
                break
        return total_num - len(self.cd_list)

    def temporal_filter(self) -> None:  # State: Finish. Unit Test Succeed
        """filter out codewords which thier mnrl is larger than {MNRL_LIM}"""
        print(f"temporal function start")
        self.__mnrl_value_update()
        self.__relocate()
        del_num = self.__delete_empty_cwd()
        print(f"del num :{del_num}")
        
    def BGS(self, seg: np.ndarray, t:int) -> np.ndarray:  # ! State: Not Yet
        row, col, _ = seg.shape
        potential_fg = np.ones(
            (row, col), dtype=bool
        )
        for i in range(len(self.cd_list)):
            if not np.any(potential_fg):
                break
            match_pixel = self.cd_list[i].test_examine(seg, potential_fg)
            
            except_update = self.cd_list[i].update(
                seg, t, match_pixel.astype(bool)
            )

            potential_fg = np.bitwise_and(
                potential_fg,
                except_update)
            
        return potential_fg
    def get_cb_list(self):
        return self.cd_list
class CodeWord:
    """
    parameter ->
    COLOR_TOLERANCE : float32 color distortion tolerance

    avg_color  ->
    avg_bgr         : float32    average color until the current frame (have 3 channels)

    aux ->
    bri_max             : float32    max brightness
    bri_min             : float32   min brightness
    freq                : float32      frequency with which the CodeWord has occurred
    mnrl                : float32      maximum negative run-length
    first_times           : float32      first access time
    last_times            : float32      min acceess

    """

    ARR_2D_SET = (
        "bri_max",
        "bri_min",
        "freq",
        "mnrl",
        "first_times",
        "last_times",
    )
    COLOR_TOLERANCE = 10.0  # ? NOT CERTAIN NUMBER DESCRIBED
    BRI_ALPHA = 0.65  # ? between 0.4 and 0.7
    BRI_BETA = 1.34  # ? between 1.1 and 1.5

    def __init__(               # State: Finish. (Assume)
        self, seg: np.ndarray, t: int, cared_pixel: np.ndarray = np.array([])
    ) -> None:  
        """Create CodeWord in numpy certain element type"""
        row, col, _ = seg.shape
        if cared_pixel.size != 0:
            bgr_cared = np.dstack((cared_pixel, cared_pixel, cared_pixel))
            self.avg_bgr = np.where(bgr_cared, seg.astype(np.float32), np.NAN)
            bri = self.__bri_calculate(seg)
            self.bri_max = np.where(cared_pixel, bri, np.NAN)
            self.bri_min = np.where(cared_pixel, bri, np.NAN)

            self.freq = np.where(cared_pixel, 1, np.NAN).astype(np.float32)
            self.mnrl = np.where(cared_pixel, t , np.NAN).astype(np.float32)

            self.first_times = np.where(cared_pixel, t, np.NAN).astype(np.float32)
            self.last_times = np.where(cared_pixel, t, np.NAN).astype(np.float32)

        else:
            self.avg_bgr = seg.astype(np.float32)
            self.bri_max = self.__bri_calculate(seg)
            self.bri_min = self.__bri_calculate(seg)  # Copy without reference
            self.freq = np.ones((row, col), dtype=np.float32)
            self.mnrl = np.empty((row, col), dtype=np.float32)
            self.mnrl[:] = t
            self.first_times = np.empty((row, col), dtype=np.float32)
            self.last_times = np.empty((row, col), dtype=np.float32)
            self.first_times[:] = t 
            self.last_times[:] = t

    def tr_examine(  # State: Finish. (Assume)
        self, seg: np.ndarray, t: int, cared_pixel: np.ndarray
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:  # first: match_pixel  2: 2D brightness lim   3: effective_nan
        """During training period, examine whether seg belongs to this CodeWord."""
        effective_nan = np.bitwise_and(np.isnan(self.freq), cared_pixel)
        bri = self.__bri_calculate(seg)

        if np.any(effective_nan):  # 更新 cared_pixel 中與 這個 cwd nan 重疊的部份的 pixel 的數值
            bgr_nan = np.dstack((effective_nan, effective_nan, effective_nan))
            self.avg_bgr = np.where(bgr_nan, seg.astype(np.float32), self.avg_bgr)
            self.bri_max = np.where(effective_nan, bri, self.bri_max)
            self.bri_min = np.where(effective_nan, bri, self.bri_min)

            self.freq = np.where(effective_nan, 1, self.freq).astype(np.float32)
            self.mnrl = np.where(effective_nan, t - 1, self.mnrl).astype(np.float32)

            self.first_times = np.where(effective_nan, t, self.first_times).astype(
                np.float32
            )
            self.last_times = np.where(effective_nan, t, self.last_times).astype(
                np.float32
            )

        not_yet_match = np.bitwise_and(cared_pixel, np.bitwise_not(effective_nan))

        dist = self.__colordist(seg)
        bri_low_lim = CodeWord.BRI_ALPHA * self.bri_max

        bu1 = CodeWord.BRI_BETA * self.bri_max
        bu2 = self.bri_min / CodeWord.BRI_ALPHA
        bri_up_lim = np.where(bu2 < bu1, bu2, bu1)  # ! True 則回傳 bu2 ， False 則回傳 bu1

        matched_pixel = np.bitwise_and(
            np.bitwise_and((bri < bri_up_lim), (bri > bri_low_lim)),
            np.bitwise_and((dist <= CodeWord.COLOR_TOLERANCE), not_yet_match),
        )
        return matched_pixel, effective_nan

    def update(  # State: Finish. (Assume)
        self, seg: np.ndarray, t: int, matched_pixel: np.ndarray) -> np.ndarray:
        bgr_effective = np.dstack((matched_pixel, matched_pixel, matched_pixel))
        bgr_freq = np.dstack((self.freq, self.freq, self.freq))
        bri = self.__bri_calculate(seg)
        self.avg_bgr = np.where(
            bgr_effective,
            (bgr_freq * self.avg_bgr + seg.astype(np.float32)) / (bgr_freq + 1),
            self.avg_bgr,
        )

        self.bri_min = np.where(
            np.bitwise_and(matched_pixel, bri < self.bri_min),
            bri,
            self.bri_min,
        )
        self.bri_max = np.where(
            np.bitwise_and(matched_pixel, bri > self.bri_max),
            bri,
            self.bri_max,
        )
        self.freq = np.where(matched_pixel, self.freq + 1, self.freq)
        self.mnrl = np.where(
            np.bitwise_and(matched_pixel, (t - self.last_times) > self.mnrl),
            t - self.last_times,
            self.mnrl,
        )
        self.last_times = np.where(matched_pixel, t, self.last_times)

        return np.bitwise_not(matched_pixel)  #  更新過的 element 以外的 element

    def test_examine(self, seg: np.ndarray, cared_pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:   # ! State: Not Yet
        """During testing period, examine whether seg belongs to this CodeWord."""
        dist = self.__colordist(seg)
        bri = self.__bri_calculate(seg)

        bri_low_lim = CodeWord.BRI_ALPHA * self.bri_max

        bu1 = CodeWord.BRI_BETA * self.bri_max
        bu2 = self.bri_min / CodeWord.BRI_ALPHA
        bri_up_lim = np.where(bu2 < bu1, bu2, bu1)  
        
        matched_pixel = np.bitwise_and(
            np.bitwise_and((bri < bri_up_lim), (bri > bri_low_lim)),
            np.bitwise_and((dist < CodeWord.COLOR_TOLERANCE), cared_pixel),
        )

        return matched_pixel

    def __bri_calculate(  # State: Unit test successfully
        self, seg: np.ndarray
    ) -> np.ndarray:
        seg = seg.astype(np.float32)
        return np.sqrt(
            np.square(seg[:, :, 0]) + np.square(seg[:, :, 1]) + np.square(seg[:, :, 2])
        )

    def __colordist(self, seg: np.ndarray):  # State: Unit test successfully
        """Calculate color distorion"""
        avg = self.avg_bgr
        avg_seg_element_dot = (
            avg[:, :, 0] * seg[:, :, 0]
            + avg[:, :, 1] * seg[:, :, 1]
            + avg[:, :, 2] * seg[:, :, 2]
        )
        avg_abs_square = np.square(self.__bri_calculate(avg))        
        seg_abs_square = np.square(self.__bri_calculate(seg))
        
        
        p_square = np.square(avg_seg_element_dot) / avg_abs_square
        
        difference = seg_abs_square - p_square
        difference[difference<0] = 0.0 # ? 電腦運算出來的值有寫可能會造成簡出來是負的，比如說當兩個比較的image大小相同時，因運算誤差造成後續在sqrt的時候有可能接收到負值，因此把其排除
        
        distortion = np.sqrt(difference)
        
        return distortion


def divide_frame(  # State: Unit test successfully
    frame: np.ndarray, sv_path: str,  height: int, width: int, num_h: int, num_w: int, t: int
):
    """divide the input frame into {num_h}*{num_w} pictures"""
    from os.path import join
    step_r = height // num_h
    step_c = width // num_w
    module_r = height % num_h
    module_c = width % num_w
    pre_r = 0
    for i in range(1, num_h + 1):
        r = i * step_r if i * step_r != (height - module_r) else height
        pre_c = 0
        for j in range(1, num_w + 1):
            c = j * step_c if j * step_c != (width - module_c) else width
            # print(f"pre_r: {pre_r} r: {r} | pre_c: {pre_c} c: {c}")
            segment = frame[int(pre_r):int(r), int(pre_c):int(c)]
            cv2.imwrite(join(sv_path,f"frame:{t}_{i}_{j}.jpg"), segment)
            pre_c = c
        pre_r = r
def recover_image(path:str, num_h: int, num_w: int, t:int ):
    from os.path import join
    vstack_list = []
    for i in range(1,num_h+1):
        img = cv2.imread(join(path, f"frame:{t}_{i}_1.jpg"))
        for j in range(2,num_w+1):
            fpth = join(path,f"frame:{t}_{i}_{j}.jpg")
            var = cv2.imread(fpth)
            img = cv2.hconcat((img,var))
        vstack_list.append(img)
    
    frame = vstack_list.pop(0)
    while len(vstack_list) != 0:
        frame = cv2.vconcat((frame, vstack_list.pop(0)))
    
    return frame

def capture_video(  # State: Unit test successfully
    v_pth="/home/sean/Desktop/AIVC/CodeBook/highway.mp4",
) -> cv2.VideoCapture:
    """Output : video, total_frame_num , Height, Width"""
    # v_pth = input("Key in the path of the video :")
    from os.path import exists
    if exists(v_pth):
        video = cv2.VideoCapture(v_pth)
    else:
        video = False

    return video, video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)

  



def training(CB:CodeBook, part:str):
    t = time.time()
    for i in range(0, training_frame.value):
        # print(i)
        img = cv2.imread(f"/home/sean/Desktop/AIVC/CodeBook/data/frame:{i}_{part}.jpg")
        # cv2.imshow("img",img)
        # cv2.waitKey()
        CB.training(img, i)

    CB.temporal_filter()
    print(f"{part} After cd_list: {len(CB.cd_list)}\ncost time: {time.time() - t}")
    print(f"cost time: {time.time() - t}")

# * Training & Testing Code Book
if __name__ == "__main__":
    from  multiprocessing.managers import BaseManager, NamespaceProxy
    import multiprocessing

    from lib.Customized_Multiprocess import register_proxy, ObjProxy, MyManager
    register_proxy("CodeBook", CodeBook , ObjProxy)


    
    
    video, total_frame, height, width = capture_video()
    training_frame = multiprocessing.Value('i', 300)
    print(f"Total Frame: {training_frame.value}")
    print(f"CodeWord.BRI_ALPHA = {CodeWord.BRI_ALPHA} | CodeWord.BRI_ALPHA = {CodeWord.BRI_BETA} ")

    import time

    t = time.time()
    with MyManager() as manager:
        # manager.start()
        CB = CodeBook(training_frame.value)
        training(CB, "1_1")


        for i in range(0, 553):
            img = cv2.imread(f"/home/sean/Desktop/AIVC/CodeBook/data/frame:{i}_1_1.jpg")
            cv2.imshow('1_1',img)
            # print(img.shape)
            fg = CB.BGS(img,i)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            # fg =cv2.morphologyEx(fg.astype(np.uint8),cv2.MORPH_OPEN,kernel, iterations=1)
            cv2.imshow("1_1 fg", fg.astype(np.uint8)*255)
            cv2.waitKey(12)
        # manager.shutdown()

# if __name__ == "__main__":
#     import multiprocessing
#     up_lim_cpus =multiprocessing.cpu_count()