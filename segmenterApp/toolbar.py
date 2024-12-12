from PyQt5 import QtWidgets, QtCore
import qtawesome as qta

class Toolbar(QtWidgets.QToolBar):
    def __init__(self, parent):
        super().__init__("Toolbar", parent)

        self.parent = parent

        # Capture button
        self.capture_action = QtWidgets.QAction(qta.icon('fa.camera'), '', parent)
        self.capture_action.setToolTip('Capture Image')
        self.capture_action.triggered.connect(parent.capture_image)
        self.addAction(self.capture_action)

        # Segment button
        self.segment_action = QtWidgets.QAction(qta.icon('ri.scissors-cut-fill'), '', parent)
        self.segment_action.setToolTip('Segment with Selected Points')
        self.segment_action.triggered.connect(parent.segment_with_points)
        self.segment_action.setEnabled(False)
        self.addAction(self.segment_action)

        # Auto-segment button
        self.autosegment_action = QtWidgets.QAction(qta.icon('fa.magic'), '', parent)
        self.autosegment_action.setToolTip('Auto-Segment')
        self.autosegment_action.triggered.connect(parent.auto_segment)
        self.autosegment_action.setEnabled(False)
        self.addAction(self.autosegment_action)

        # Finalize button
        self.finalize_action = QtWidgets.QAction(qta.icon('fa.check'), '', parent)
        self.finalize_action.setToolTip('Finalize Object')
        self.finalize_action.triggered.connect(parent.finalize_current_object)
        self.finalize_action.setEnabled(False)
        self.addAction(self.finalize_action)

        # Save button
        self.save_action = QtWidgets.QAction(qta.icon('fa.save'), '', parent)
        self.save_action.setToolTip('Save Segments')
        self.save_action.triggered.connect(parent.save_segments)
        self.save_action.setEnabled(False)
        self.addAction(self.save_action)

        # Undo button
        self.undo_action = QtWidgets.QAction(qta.icon('mdi.undo-variant'), '', parent)
        self.undo_action.setToolTip('Undo Last Action')
        self.undo_action.triggered.connect(parent.undo_action_triggered)
        self.undo_action.setEnabled(False)
        self.addAction(self.undo_action)

        # Redo button
        self.redo_action = QtWidgets.QAction(qta.icon('mdi.redo-variant'), '', parent)
        self.redo_action.setToolTip('Redo Last Action')
        self.redo_action.triggered.connect(parent.redo_action_triggered)
        self.redo_action.setEnabled(False)
        self.addAction(self.redo_action)

        # Reset button
        self.reset_action = QtWidgets.QAction(qta.icon('fa.refresh'), '', parent)
        self.reset_action.setToolTip('Reset All Segments')
        self.reset_action.triggered.connect(parent.reset_segments)
        self.addAction(self.reset_action)

    def set_actions_enabled(self, capture):
        self.segment_action.setEnabled(capture)
        self.autosegment_action.setEnabled(capture)
        self.finalize_action.setEnabled(capture)
        self.save_action.setEnabled(capture)
        self.undo_action.setEnabled(bool(self.parent.undo_stack))
        self.redo_action.setEnabled(bool(self.parent.redo_stack))
    def update_action_states(self, capture):
        self.segment_action.setEnabled(capture)
        self.autosegment_action.setEnabled(capture)
        self.finalize_action.setEnabled(capture)
        self.save_action.setEnabled(capture)
        self.undo_action.setEnabled(bool(self.parent.undo_stack))
        self.redo_action.setEnabled(bool(self.parent.redo_stack))
