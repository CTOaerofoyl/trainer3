### File: Segmenter.py
import sys
from PyQt5 import QtWidgets
from segmenterApp.segmentation_app import SegmentationApp

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = SegmentationApp()
    ex.show()
    sys.exit(app.exec_())
