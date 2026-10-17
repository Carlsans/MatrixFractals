"""MatrixFractals GUI. Run from the repository root:  python -m gui.app"""
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from gui import bridge  # noqa: E402  (installs the import shims first)


_JOBS = []


class Job(QThread):
    """Runs fn(job) off the UI thread. fn polls job.should_stop() and logs through job.log."""
    log = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        _JOBS.append(self)
        self.fn = fn
        self._stop = threading.Event()

    def should_stop(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self))
        except Exception:
            self.failed.emit(traceback.format_exc())


def to_pixmap(rgb, width=None):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
    return pix.scaledToWidth(width, Qt.SmoothTransformation) if width else pix


class ImageView(QLabel):
    def __init__(self, size=420):
        super().__init__("no image")
        self.setMinimumSize(size, size // 2)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888")

    def show_rgb(self, rgb):
        self.setPixmap(to_pixmap(rgb, max(self.width() - 4, 100)))


class LogBox(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setMaximumHeight(130)

    def add(self, text):
        self.appendPlainText(text)


class Runner:
    """Starts one background job at a time for a tab and wires its signals to a log."""
    def __init__(self, log, buttons=()):
        self.log, self.buttons, self.job = log, buttons, None

    def start(self, fn, on_done=None):
        if self.job is not None and self.job.isRunning():
            return
        self.job = Job(fn)
        self.job.log.connect(self.log.add)
        self.job.failed.connect(lambda tb: self.log.add(tb))
        if on_done:
            self.job.finished_ok.connect(on_done)
        self.job.finished.connect(lambda: [b.setEnabled(True) for b in self.buttons])
        for b in self.buttons:
            b.setEnabled(False)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.stop()

    def running(self):
        return self.job is not None and self.job.isRunning()


class ExploreTab(QWidget):
    matrix_changed = Signal(object)

    def __init__(self, conf):
        super().__init__()
        self.conf, self.matrix, self.volume = conf, None, None
        self.size_spin = QSpinBox(); self.size_spin.setRange(2, 7); self.size_spin.setValue(4)
        self.dims = QComboBox(); self.dims.addItems(bridge.DIMENSION_MODES); self.dims.setCurrentText("medium_def")
        self.iters = QSpinBox(); self.iters.setRange(2, 1000); self.iters.setValue(100)
        self.table = QTableWidget()
        self.table.setToolTip("Cells: a complex number (0.3-1.1j), fn(V) such as -sin(C), or a variable (C, D, Z). "
                              "Rows are multiplied, row results are summed.")
        self.layer = QSlider(Qt.Horizontal); self.layer.valueChanged.connect(self.show_layer)
        self.view = ImageView(); self.score = QLabel("score: -")
        self.log = LogBox()
        self.btn_random = QPushButton("Random matrix"); self.btn_random.clicked.connect(self.randomize)
        self.btn_render = QPushButton("Render volume"); self.btn_render.clicked.connect(self.render)
        btn_load = QPushButton("Open .frac"); btn_load.clicked.connect(self.open_frac)
        btn_save = QPushButton("Save .frac"); btn_save.clicked.connect(self.save_frac)
        btn_png = QPushButton("Export layers (PNG)"); btn_png.clicked.connect(self.export_png)
        self.runner = Runner(self.log, (self.btn_render,))

        form = QFormLayout()
        form.addRow("Matrix size (square)", self.size_spin)
        form.addRow("Resolution (x, y, t)", self.dims)
        form.addRow("Max iterations", self.iters)
        row = QHBoxLayout()
        for b in (self.btn_random, self.btn_render, btn_load, btn_save, btn_png):
            row.addWidget(b)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(self.table); left.addLayout(row)
        right = QVBoxLayout(); right.addWidget(self.view, 1); right.addWidget(self.layer); right.addWidget(self.score)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addLayout(right, 1)
        outer = QVBoxLayout(self); outer.addLayout(top); outer.addWidget(self.log)
        self.randomize()

    def set_matrix(self, matrix):
        self.matrix = matrix
        n = len(matrix.matrix)
        self.table.setRowCount(n); self.table.setColumnCount(n)
        for i, row in enumerate(matrix.matrix):
            for j, item in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(bridge.cell_text(item)))
        self.matrix_changed.emit(matrix)

    def current_matrix(self):
        n = self.table.rowCount()
        cells = [[self.table.item(i, j).text() for j in range(n)] for i in range(n)]
        self.matrix = bridge.matrix_from_cells(self.conf, cells)
        return self.matrix

    def randomize(self):
        self.set_matrix(bridge.random_matrix(self.conf, self.size_spin.value()))

    def render(self):
        try:
            matrix = self.current_matrix()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid cell", str(exc)); return
        dims = bridge.FractalTensors.get_dimensions_by_mode(self.dims.currentText())
        iters = self.iters.value()
        self.log.add(f"rendering {dims} ...")

        def work(job):
            return bridge.render(self.conf, matrix, dims, iters)

        def done(result):
            self.volume, score, _ = result
            self.layer.setRange(0, self.volume.shape[2] - 1)
            self.score.setText(f"x264 size per voxel: {score:.4f}   (the engine's interestingness note)")
            self.matrix_changed.emit(matrix)
            self.show_layer()
        self.runner.start(work, done)

    def show_layer(self):
        if self.volume is not None:
            self.view.show_rgb(bridge.colorize(self.volume[:, :, self.layer.value()]))

    def open_frac(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .frac (pickle: only open your own files)",
                                              str(bridge.IMAGES_DIR), "Fractal (*.frac)")
        if path:
            self.set_matrix(bridge.load_frac(self.conf, path))

    def save_frac(self):
        path = bridge.save_frac(self.current_matrix(), bridge.FractalTensors.generate_fractal_name())
        self.log.add(f"saved {path}")

    def export_png(self):
        if self.volume is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for PNG layers")
        if folder:
            import matplotlib.pyplot as plt
            for k in range(self.volume.shape[2]):
                plt.imsave(Path(folder) / f"layer_{k:04d}.png", bridge.colorize(self.volume[:, :, k]))
            self.log.add(f"wrote {self.volume.shape[2]} layers to {folder}")


class GATab(QWidget):
    send_to_explore = Signal(object)

    def __init__(self, conf):
        super().__init__()
        self.conf, self.ga, self.step_count = conf, bridge.GeneticAlgorithm(conf), 0
        self.pool_spin = QSpinBox(); self.pool_spin.setRange(2, 1000); self.pool_spin.setValue(20)
        self.cross = QDoubleSpinBox(); self.cross.setRange(0, 1); self.cross.setSingleStep(.1); self.cross.setValue(conf.crossing_probability)
        self.shift = QDoubleSpinBox(); self.shift.setRange(0, 1); self.shift.setSingleStep(.05); self.shift.setValue(conf.row_shift_probability)
        self.flip = QDoubleSpinBox(); self.flip.setDecimals(4); self.flip.setRange(0, 1); self.flip.setSingleStep(.001); self.flip.setValue(conf.mutation_flip_probability)
        self.mut = QDoubleSpinBox(); self.mut.setRange(0, 1); self.mut.setSingleStep(.1); self.mut.setValue(conf.mutation_probability)
        self.use_mut = QCheckBox("Legacy: targeted cell mutation (probability above; off in the original config)")
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["name", "note", "matrix"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.log = LogBox()
        self.btn_seed = QPushButton("Seed pool"); self.btn_step = QPushButton("Step"); self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop"); btn_send = QPushButton("Send selected to Explore")
        btn_save = QPushButton("Save top 5 (.frac)")
        self.btn_seed.clicked.connect(self.seed); self.btn_step.clicked.connect(lambda: self.run(1))
        self.btn_run.clicked.connect(lambda: self.run(None)); self.btn_stop.clicked.connect(lambda: self.runner.stop())
        btn_send.clicked.connect(self.send); btn_save.clicked.connect(self.save_top)
        self.runner = Runner(self.log, (self.btn_seed, self.btn_step, self.btn_run))
        form = QFormLayout()
        for label, w in (("Pool size", self.pool_spin), ("Crossing probability", self.cross),
                         ("Row-shift probability", self.shift), ("Flip mutation (auto-tuned by similarity)", self.flip),
                         ("Targeted mutation probability", self.mut)):
            form.addRow(label, w)
        row = QHBoxLayout()
        for b in (self.btn_seed, self.btn_step, self.btn_run, self.btn_stop, btn_send, btn_save):
            row.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addLayout(form); lay.addWidget(self.use_mut); lay.addLayout(row); lay.addWidget(self.table); lay.addWidget(self.log)

    def apply_conf(self):
        c = self.conf
        c.poolsize, c.crossing_probability = self.pool_spin.value(), self.cross.value()
        c.row_shift_probability, c.mutation_flip_probability = self.shift.value(), self.flip.value()
        c.mutation_probability = self.mut.value()

    def seed(self):
        self.apply_conf()
        self.ga = bridge.GeneticAlgorithm(self.conf)
        self.runner.start(lambda job: bridge.seed_pool(self.ga, self.conf, self.conf.poolsize, job=job,
                                                       log=self.runner.job.log.emit),
                          lambda _: self.refresh())

    def run(self, steps):
        if not self.ga.pool:
            self.log.add("seed the pool first"); return
        self.apply_conf()

        def work(job):
            n = 0
            while not job.should_stop() and (steps is None or n < steps):
                bridge.ga_step(self.ga, self.conf, self.use_mut.isChecked(), log=job.log.emit)
                self.step_count += 1; n += 1
        self.runner.start(work, lambda _: self.refresh())

    def refresh(self):
        pool = sorted(self.ga.pool, key=lambda i: i['note'], reverse=True)
        self.table.setRowCount(len(pool))
        for r, item in enumerate(pool):
            self.table.setItem(r, 0, QTableWidgetItem(str(item['fractal']['fractal_name'])))
            self.table.setItem(r, 1, QTableWidgetItem(f"{item['note']:.3f}"))
            self.table.setItem(r, 2, QTableWidgetItem(str(item['fractal']['matrix']).replace("\n", " | ")))
        self.pool_view = pool

    def send(self):
        row = self.table.currentRow()
        if row >= 0:
            self.send_to_explore.emit(self.pool_view[row]['fractal']['matrix'])

    def save_top(self):
        for item in self.ga.pick_top_items(5):
            self.log.add(f"saved {bridge.save_frac(item['fractal']['matrix'], item['fractal']['fractal_name'])}")


class VRTab(QWidget):
    def __init__(self, conf):
        super().__init__()
        self.matrix = None
        self.width_spin = QSpinBox(); self.width_spin.setRange(256, 8192); self.width_spin.setSingleStep(256); self.width_spin.setValue(conf.out_W)
        self.preview_w = QSpinBox(); self.preview_w.setRange(256, 2048); self.preview_w.setValue(768)
        self.duration = QSpinBox(); self.duration.setRange(1, 600); self.duration.setValue(conf.video_length)
        self.fps = QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(conf.fps)
        self.disp = QDoubleSpinBox(); self.disp.setRange(0, .5); self.disp.setSingleStep(.01); self.disp.setValue(conf.eye_disparity)
        self.cmap = QComboBox(); self.cmap.addItems(["gist_earth", "nipy_spectral", "viridis", "plasma", "magma", "turbo"])
        self.label = QLabel("Matrix: use Explore or the GA tab to pick one")
        self.view = ImageView(); self.log = LogBox()
        self.btn_prev = QPushButton("Preview one frame"); self.btn_exp = QPushButton("Export stereo 360 video")
        self.btn_prev.clicked.connect(self.preview); self.btn_exp.clicked.connect(self.export)
        self.runner = Runner(self.log, (self.btn_prev, self.btn_exp))
        form = QFormLayout()
        form.addRow("Preview width", self.preview_w); form.addRow("Video width (out_W)", self.width_spin)
        form.addRow("Duration (s)", self.duration); form.addRow("FPS", self.fps)
        form.addRow("Eye disparity", self.disp); form.addRow("Colormap", self.cmap)
        lay = QVBoxLayout(self)
        lay.addWidget(self.label); lay.addLayout(form)
        row = QHBoxLayout(); row.addWidget(self.btn_prev); row.addWidget(self.btn_exp)
        lay.addLayout(row); lay.addWidget(self.view, 1); lay.addWidget(self.log)

    def set_matrix(self, matrix):
        self.matrix = matrix
        self.label.setText("Matrix:\n" + str(matrix))

    def preview(self):
        if self.matrix is None:
            return
        self.runner.start(lambda job: bridge.vr_preview(self.matrix, self.preview_w.value(), self.cmap.currentText(), self.disp.value()),
                          self.view.show_rgb)

    def export(self):
        if self.matrix is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Video", "vr.mp4", "MP4 (*.mp4)")
        if path:
            self.runner.start(lambda job: bridge.vr_export(self.matrix, path, self.duration.value(), self.fps.value(),
                                                           self.width_spin.value(), self.disp.value(), self.cmap.currentText()),
                              lambda _: self.log.add(f"done: {path} (+ ...360.mp4 with spatial metadata)"))


class LegacyTab(QWidget):
    """Features that main.py no longer calls, or whose code was left on a stale API."""
    matrix_found = Signal(object)

    def __init__(self, conf, explore):
        super().__init__()
        self.conf, self.explore = conf, explore
        self.log = LogBox(); self.view = ImageView()

        search = QGroupBox("Random search (find_fractals_cuda): random matrix -> 24^3 test -> 256x256x64 test -> save")
        self.t1 = QDoubleSpinBox(); self.t1.setRange(0, 1); self.t1.setValue(.8)
        self.t2 = QDoubleSpinBox(); self.t2.setRange(0, 1); self.t2.setValue(.7)
        self.size = QSpinBox(); self.size.setRange(2, 7); self.size.setValue(4)
        b1 = QPushButton("Start search"); b2 = QPushButton("Stop")
        b1.clicked.connect(self.start_search); b2.clicked.connect(lambda: self.r1.stop())
        self.r1 = Runner(self.log, (b1,))
        f = QFormLayout(search); f.addRow("Stage-1 threshold", self.t1); f.addRow("Stage-2 threshold", self.t2)
        f.addRow("Matrix size", self.size); row = QHBoxLayout(); row.addWidget(b1); row.addWidget(b2); f.addRow(row)

        hill = QGroupBox("Hill climb (test_laptop): mutate one matrix, keep the best, restart from a small pool")
        c1 = QPushButton("Start climb"); c2 = QPushButton("Stop")
        c1.clicked.connect(self.start_climb); c2.clicked.connect(lambda: self.r2.stop())
        self.r2 = Runner(self.log, (c1,))
        hl = QHBoxLayout(hill); hl.addWidget(c1); hl.addWidget(c2)

        proj = QGroupBox("Volume projections of a rendered bailout volume (voxeltoVR equirectangular modes)")
        self.mode = QComboBox(); self.mode.addItems(bridge.EQUIRECT_MODES)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0, 1); self.thr.setSingleStep(.01); self.thr.setValue(.15)
        self.pw = QSpinBox(); self.pw.setRange(256, 4096); self.pw.setValue(1024)
        b3 = QPushButton("Project volume from Explore tab"); b3.clicked.connect(self.project)
        self.r3 = Runner(self.log, (b3,))
        note = QLabel("Several of these modes were never finished in the original (black frames, broadcasting errors). "
                      "Failures are reported in the log instead of crashing.")
        note.setWordWrap(True)
        pf = QFormLayout(proj); pf.addRow("Mode", self.mode); pf.addRow("Threshold", self.thr); pf.addRow("Width", self.pw)
        pf.addRow(b3); pf.addRow(note)

        lay = QVBoxLayout(self)
        lay.addWidget(search); lay.addWidget(hill); lay.addWidget(proj); lay.addWidget(self.view, 1); lay.addWidget(self.log)

    def start_search(self):
        self.r1.start(lambda job: bridge.random_search(self.conf, (self.t1.value(), self.t2.value()), job,
                                                       self.r1.job.log.emit, self.size.value()),
                      lambda p: p and self.explore.set_matrix(bridge.load_frac(self.conf, p)))

    def start_climb(self):
        self.r2.start(lambda job: bridge.hill_climb(self.conf, 4, (24, 24, 24), job, self.r2.job.log.emit,
                                                    on_best=lambda m, s: self.matrix_found.emit(m)))

    def project(self):
        if self.explore.volume is None:
            self.log.add("render a volume in the Explore tab first"); return
        self.r3.start(lambda job: bridge.volume_projection(self.explore.volume, self.mode.currentText(), self.pw.value(), threshold=self.thr.value()),
                      self.view.show_rgb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MatrixFractals")
        conf = bridge.default_conf()
        self.explore = ExploreTab(conf); self.ga = GATab(conf); self.vr = VRTab(conf)
        self.legacy = LegacyTab(conf, self.explore)
        tabs = QTabWidget()
        for w, name in ((self.explore, "Explore"), (self.ga, "Genetic algorithm"), (self.vr, "VR export"),
                        (self.legacy, "Legacy / experimental")):
            tabs.addTab(w, name)
        self.setCentralWidget(tabs)
        self.explore.matrix_changed.connect(self.vr.set_matrix)
        self.ga.send_to_explore.connect(self.explore.set_matrix)
        self.legacy.matrix_found.connect(self.explore.set_matrix)
        self.vr.set_matrix(self.explore.matrix)
        self.resize(1100, 800)

    def closeEvent(self, event):
        running = [job for job in _JOBS if job.isRunning()]
        for job in running:
            job.stop()
        # a QThread destroyed while running aborts the process, and terminating one inside CUDA does too
        if not all(job.wait(15000) for job in running):
            import os
            os._exit(0)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
