from os.path import exists
from sys import exit
from copy import deepcopy
import numpy as np
import cv2


def capture_video(  # State: Unit test successfully
    v_pth="/home/sean/Desktop/AIVC/Codebook/highway.mp4",
) -> cv2.VideoCapture:
    # v_pth = input("Key in the path of the video :")
    if exists(v_pth):
        video = cv2.VideoCapture(v_pth)
    else:
        video = False

    return video


class Codebook:
    """
    FRAME_PER_RELOCATION    : int   every # frames, __relocate codeword's order
    MNRL_LIM                : int   every background codeword should recur at least every {MNRL_LIM} frames.
    """

    FRAME_PER_RELOCATION: int
    MNRL_LIM: int

    def __init__(self, total_frame: int, fpr: int = 20) -> None:
        self.cd_list: list[Codeword] = []
        self.total_frame = total_frame
        Codebook.MNRL_LIM = int(total_frame * 0.5)
        Codebook.FRAME_PER_RELOCATION = fpr

    def training(self, seg: np.array, t: int) -> None:  # State: Finish. Need Unit Test
        if len(self.cd_list) == 0:
            cwd = Codeword(seg, t)
            self.cd_list.append(cwd)
            print("not match. Create a new one")
            return

        if t % Codebook.FRAME_PER_RELOCATION == 0:
            self.__relocate()
            if t % (Codebook.FRAME_PER_RELOCATION * 5) == 0:
                del_num = self.__delete_empty_array()
                print(f"del_num: {del_num}")

        row, col, _ = seg.shape
        waiting_for_check = np.ones(
            (row, col), dtype=bool
        )  # ? True : Elements waits for checking  False: Elements has been checked.

        for i in range(len(self.cd_list)):
            match_pixel, bri_lim, effective_nan = self.cd_list[i].tr_examine(
                seg, t, waiting_for_check
            )

            except_update = self.cd_list[i].update(
                seg, t, match_pixel, bri_lim
            )  # 更新過得element 以外的elelment
            waiting_for_check = np.bitwise_and(
                np.bitwise_and(waiting_for_check, except_update),
                np.bitwise_not(effective_nan),
            )  # 沒有更新過得element 跟上一次待更新的 element 做and ， 取出還有哪些element 還要match

        if np.any(
            waiting_for_check
        ):  # ? waiting_for_check 如果存在一個 True (需要創建新的Cwd)，就創建一個新的
            new_cwd = Codeword(seg, t, waiting_for_check)
            self.cd_list.append(new_cwd)

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

                # ? 同時迭代 j codeword 及 j+1 codeword 內部的 public variable。同時迭代時，變數迭代的順序是一樣的。
                # ?（avg_bgr -> bri_max -> bri_min -> freq -> mnrl -> first_times -> last_times)
                # * 將比較freq 較大的，移到cd_list 中 index 較小的位置
                for (attr_nm1, arr_1), (attr_nm2, arr_2) in zip(
                    vars(self.cd_list[j]).items(), vars(self.cd_list[j + 1]).items()
                ):
                    if attr_nm1 == attr_nm2 and attr_nm1 in Codeword.ARR_2D_SET:
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
        # indx = getattr(self.__relocate, 'indx', False)
        # print(f"__relocate.indx : {indx}")
        length_cd_list = len(self.cd_list)
        for i in range(length_cd_list):
            self.cd_list[i].freq = np.where(
                self.cd_list[i].freq == 0, np.NAN, self.cd_list[i].freq
            )

    def __mnrl_value_update(self) -> None:  # State: Finish. Unit Test Succeed
        for i in range(len(self.cd_list)):
            mnrl = self.cd_list[i].mnrl
            cmp = (
                self.total_frame
                + self.cd_list[i].first_times
                - self.cd_list[i].last_times
                - 1
            )
            self.cd_list[i].mnrl = np.where(cmp > mnrl, cmp, mnrl)

            # ! 會有Nan > 數字的問題，只要有 NAN 與數字相關的比較，回傳是False
            del_pixel = (
                self.cd_list[i].mnrl >= Codebook.MNRL_LIM
            )  # * 捨棄MNRL>= MNRL_LIM的數值
            bgr_del = np.dstack((del_pixel, del_pixel, del_pixel))
            self.cd_list[i].avg_bgr = np.where(bgr_del, np.NAN, self.cd_list[i].avg_bgr)
            # * 將MNRL大於等於一半total frame 的 pixel 設成 np.NAN
            for attr_nm, value in vars(self.cd_list[i]).items():
                if attr_nm in Codeword.ARR_2D_SET:
                    setattr(
                        self.cd_list[i], attr_nm, np.where(del_pixel, np.NAN, value)
                    )

    def __delete_empty_array(self) -> int:  # State: Finish. Unit Test Succeed 
        """回傳總共刪了多少"""
        total_num = len(self.cd_list)
        for i in range(total_num - 1, -1, -1):
            cared_pixel = np.isnan(self.cd_list[i].mnrl)  # * NAN的區域
            if np.all(cared_pixel):  # 全False => 全部都是 NAN
                del (self.cd_list[i])
            else:
                break
        return total_num - len(self.cd_list)

    def temporal_filter(self) -> None:  # State: Finish. Unit Test Succeed
        """filter out codewords which thier mnrl is larger than {MNRL_LIM}"""
        print(f"temporal function start")
        self.__mnrl_value_update()
        self.__relocate()
        del_num = self.__delete_empty_array()
        print(f"del num :{del_num}")
        
    def testing(self) -> None:  # ! State: Not Yet
        pass


class Codeword:
    """
    parameter ->
    COLOR_TOLERANCE : float32 color distortion tolerance

    avg_color  ->
    avg_bgr         : float32    average color until the current frame (have 3 channels)

    aux ->
    bri_max             : float32    max brightness
    bri_min             : float32   min brightness
    freq                : float32      frequency with which the codeword has occurred
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
    BRI_ALPHA = 0.7  # ? between 0.4 and 0.7
    BRI_BETA = 1.1  # ? between 1.1 and 1.5

    def __init__(               # State: Finish. (Assume)
        self, seg: np.array, t: int, cared_pixel: np.array = np.array([])
    ) -> None:  
        """Create codeword in numpy certain element type"""
        row, col, _ = seg.shape
        if cared_pixel.size != 0:
            bgr_cared = np.dstack((cared_pixel, cared_pixel, cared_pixel))
            self.avg_bgr = np.where(bgr_cared, seg.astype(np.float32), np.NAN)
            bri = self.__bri_calculate(seg)
            self.bri_max = np.where(cared_pixel, bri, np.NAN)
            self.bri_min = np.where(cared_pixel, bri, np.NAN)

            self.freq = np.where(cared_pixel, 1, np.NAN).astype(np.float32)
            self.mnrl = np.where(cared_pixel, t - 1, np.NAN).astype(np.float32)

            self.first_times = np.where(cared_pixel, t, np.NAN).astype(np.float32)
            self.last_times = np.where(cared_pixel, t, np.NAN).astype(np.float32)

        else:
            self.avg_bgr = seg.astype(np.float32)
            self.bri_max = self.__bri_calculate(seg)
            self.bri_min = np.copy(self.bri_max)  # Copy without reference
            self.freq = np.ones((row, col), dtype=np.float32)
            self.mnrl = np.empty((row, col), dtype=np.float32)
            self.mnrl[:] = t - 1
            self.first_times = np.empty((row, col), dtype=np.float32)
            self.last_times = np.empty((row, col), dtype=np.float32)
            self.first_times[:] = t
            self.last_times[:] = t

    def tr_examine(  # State: Finish. (Assume)
        self, seg: np.array, t: int, cared_pixel: np.array
    ) -> tuple[
        np.array, np.array, np.array
    ]:  # first: match_pixel  2: 2D brightness lim   3: effective_nan
        """During training period, examine whether seg belongs to this codeword."""
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
        bri_low_lim = Codeword.BRI_ALPHA * self.bri_max

        bu1 = Codeword.BRI_BETA * self.bri_max
        bu2 = self.bri_min / Codeword.BRI_ALPHA
        bri_up_lim = np.where(bu2 < bu1, bu2, bu1)  # ! True 則回傳 bu2 ， False 則回傳 bu1

        bri_lim = np.empty((*bri_up_lim.shape, 2), dtype=np.float32)
        bri_lim[:, :, 0] = bri_up_lim
        bri_lim[:, :, 1] = bri_low_lim
        # print(bri_lim)
        matched_pixel = np.bitwise_and(
            np.bitwise_and((bri < bri_up_lim), (bri > bri_low_lim)),
            np.bitwise_and((dist < Codeword.COLOR_TOLERANCE), not_yet_match),
        )

        return matched_pixel, bri_lim, effective_nan

    def update(  # State: Finish. (Assume)
        self, seg: np.array, t: int, matched_pixel: np.array, bri_lim: np.array
    ) -> np.array:
        bgr_effective = np.dstack((matched_pixel, matched_pixel, matched_pixel))
        bgr_freq = np.dstack((self.freq, self.freq, self.freq))

        self.avg_bgr = np.where(
            bgr_effective,
            (bgr_freq * self.avg_bgr + seg.astype(np.float32)) / (bgr_freq + 1),
            self.avg_bgr,
        )

        self.bri_min = np.where(
            np.bitwise_and(matched_pixel, bri_lim[:, :, 1] < self.bri_min),
            bri_lim[:, :, 1],
            self.bri_min,
        )
        self.bri_max = np.where(
            np.bitwise_and(matched_pixel, bri_lim[:, :, 0] > self.bri_max),
            bri_lim[:, :, 0],
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

    def test_examine(self, seg: np.array, t) -> None:   # ! State: Not Yet
        """During testing period, examine whether seg belongs to this codeword."""
        
        pass

    def __bri_calculate(  # State: Unit test successfully
        self, seg: np.array
    ) -> np.array:
        seg = seg.astype(np.float32)
        return np.sqrt(
            np.square(seg[:, :, 0]) + np.square(seg[:, :, 1]) + np.square(seg[:, :, 2])
        )

    def __colordist(self, seg: np.array):  # State: Unit test successfully
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
        distortion = np.sqrt(seg_abs_square - p_square)

        return distortion


def divide_frame(  # State: Unit test successfully
    frame: np.array, height: int, width: int, num_h: int, num_w: int, t: int
):
    """divide the input frame into {num_h}*{num_w} pictures"""
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
            segment = frame[pre_r:r, pre_c:c]
            cv2.imwrite(f"./data/frame:{t}_{i}_{j}.jpg", segment)
            pre_c = c
        pre_r = r


if __name__ == "__main__":
    video: cv2.VideoCapture = capture_video()
    num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame = 100
    CB = Codebook(frame)
    print(f"Total Frame: {num_frame}")

    import time

    t = time.time()
    for i in range(1, 100):
        img = cv2.imread(f"/home/sean/Desktop/AIVC/Codebook/data/frame:{i}_1_1.jpg")
        # cv2.imshow('1_1',img)
        # cv2.waitKey()
        CB.training(img, i)

    print(len(CB.cd_list))
    CB.temporal_filter()

    print(f"cost time: {time.time() - t}")

# if __name__ == "__main__":
#     video: cv2.VideoCapture = capture_video()
# num_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
# CB = Codebook(num_frame)
# print(f"Total Frame: {num_frame}")
#
# for t in range(1, int(num_frame) + 1):
# ret, frame = video.read()
# if not ret:
# print("end of program")
# break
#
# cv2.imshow("highway", frame)
# divide_frame(
# frame,
# int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
# int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
# 2,
# 3,
# t,
# )
# segment = cv2.imread("./data/1_1.jpg")
# CB.training(segment, t)
# key = cv2.waitKey(33)
# if key == ord("q"):
# break
