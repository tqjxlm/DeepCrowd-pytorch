from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from torch.multiprocessing import Queue
import time


class Viewer(QtGui.QMainWindow):
    """
    A high performance viewer that supports remote rendering (X-forwarding)
    """
    def __init__(self, h, w, buffer: Queue, parent=None):
        super(Viewer, self).__init__(parent)

        self.buffer = buffer

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0, 0, h, w))

        # Image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):
        # Get image
        data = self.buffer.get(timeout=10)
        if data is None:
            return
        self.img.setImage(data)

        # Calculate FPS
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        self.label.setText(tx)

        # Next frame
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1
