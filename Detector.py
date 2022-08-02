import logging
import model as qrcnn_model
import torch
from misc.F import set_logger

set_logger(path='log_txtsave')
logger = logging.getLogger("Detector.py")

class Detector:
    def __init__(self, weight, net):
        """
        @param weight: .pt 路徑
        @param net: 使用的 nn.Module
        """
        self.percentile_pick = 99.99  # 百分位數設定
        self.thick = 2  # bbox bound
        self.overlap = 0.3  # 切割精細度
        # merge 策略 (float)
        self.merge_delta_x = 0.01
        self.merge_delta_y = 0.01
        #
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #
        # ====
        #
        self.weight = weight
        self.net = net
        #
        # ===
        self._do_init()

    def _do_init(self):
        self.net.load_state_dict(torch.load(self.weight))
        logger.info(f"Load weight from {self.weight}")
        if torch.cuda.is_available():
            self.net.cuda()
            logger.info(f"Move net to Cuda.")
        else:
            logger.warning("NOT USING CUDA!")


if __name__ == "__main__":

    Detector(weight="./log_save/20220421_1341_58/weight.pt",
             net=qrcnn_model.QRCode_CNN())