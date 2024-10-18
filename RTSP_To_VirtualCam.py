# Enjoy Martins Dabolins

import sys
import cv2
import numpy as np
import av
import time
import queue
import threading
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import logging
import pyvirtualcam


# Configure logging to output to both console and file for better traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("virtual_cam_app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


class VirtualCamSendThread(QThread):
    """
    Thread responsible for sending frames to virtual camera.
    It can either send frames from a queue (RTSP frames) or generate and send test pattern frames.
    Emits a signal after sending each frame for GUI preview.
    """
    virtual_cam_frame_sent_signal = pyqtSignal(np.ndarray)  # Signal with frame data after sending to virtual cam

    def __init__(self):
        super().__init__()
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.frame_rate_n = None
        self.frame_rate_d = None
        self.send_test_pattern = True  # Start with test pattern until stream is connected
        self.test_pattern_resolution = (1920, 1080)  # Width, Height
        self.virtual_cam = None

    def run(self):
        logging.info("VirtualCamSendThread started.")
        while self.running:
            if not self.virtual_cam:
                time.sleep(0.1)  # Wait until virtual camera is initialized
                continue
            if self.send_test_pattern:
                # Generate test pattern
                frame = self.create_test_pattern(*self.test_pattern_resolution)

                # Send test pattern frame to virtual camera
                self.send_frame_to_virtual_cam(frame)

                # Emit signal for GUI preview
                self.virtual_cam_frame_sent_signal.emit(frame)

                # Control the frame rate for test pattern
                self.virtual_cam.sleep_until_next_frame()
            else:
                try:
                    frame = self.queue.get(timeout=1)  # Wait up to 1 second for a frame

                    # Send RTSP frame to virtual camera
                    self.send_frame_to_virtual_cam(frame)

                    # Emit signal for GUI preview
                    self.virtual_cam_frame_sent_signal.emit(frame)
                except queue.Empty:
                    # If no frame is available, continue the loop
                    continue
                except Exception as e:
                    logging.error(f"Exception in VirtualCamSendThread.run: {e}")

        # Clean up
        if self.virtual_cam:
            self.virtual_cam.close()
        logging.info("VirtualCamSendThread stopped.")

    def send_frame_to_virtual_cam(self, frame):
        """
        Sends the frame to the virtual camera.
        """
        try:
            # Ensure frame is in RGB format, pyvirtualcam requires RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Send frame to virtual camera
            self.virtual_cam.send(frame_rgb)

            # Wait until it's time for the next frame
            self.virtual_cam.sleep_until_next_frame()

            logging.debug("Frame sent to virtual camera.")
        except Exception as e:
            logging.error(f"Exception while sending frame to virtual camera: {e}")

    def send_frame(self, frame):
        """
        Enqueue a new frame to be sent to virtual camera.
        If the queue is full, discard the old frame and enqueue the new one.
        """
        if self.send_test_pattern:
            # Do not accept RTSP frames when sending test pattern
            logging.debug("VirtualCamSendThread is in test pattern mode. RTSP frame ignored.")
            return

        try:
            self.queue.put_nowait(frame)
            logging.debug("Frame enqueued to VirtualCamSendThread.")
        except queue.Full:
            try:
                discarded_frame = self.queue.get_nowait()  # Discard the old frame
                logging.debug("Old frame discarded from VirtualCamSendThread queue.")
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
                logging.debug("New frame enqueued to VirtualCamSendThread after discarding old frame.")
            except queue.Full:
                logging.warning("VirtualCamSendThread queue is still full after discarding. Frame not enqueued.")

    def set_frame_rate(self, frame_rate_n, frame_rate_d):
        """
        Initialize the virtual camera with the frame rate.
        """
        self.frame_rate_n = frame_rate_n
        self.frame_rate_d = frame_rate_d
        fps = self.frame_rate_n / self.frame_rate_d
        if not self.virtual_cam:
            try:
                width = self.test_pattern_resolution[0]
                height = self.test_pattern_resolution[1]
                self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
                logging.info(f'Virtual camera started ({self.virtual_cam.device}) with fps: {fps}.')
            except Exception as e:
                logging.error(f"Failed to open virtual camera: {e}")
                self.running = False
        else:
            logging.warning("Virtual camera already initialized. Cannot change fps.")
        logging.info(f"VirtualCamSendThread frame rate set to {self.frame_rate_n}/{self.frame_rate_d}.")

    def enable_test_pattern(self):
        """
        Enable sending test pattern frames to virtual camera.
        """
        if not self.send_test_pattern:
            self.send_test_pattern = True
            logging.info("VirtualCamSendThread switched to test pattern mode.")

    def disable_test_pattern(self):
        """
        Disable sending test pattern frames to virtual camera and resume sending RTSP frames.
        """
        if self.send_test_pattern:
            self.send_test_pattern = False
            logging.info("VirtualCamSendThread switched to RTSP frame mode.")

    def create_test_pattern(self, width, height):
        """
        Generates a simple test pattern (color bars).
        """
        try:
            bar_height = height // 8
            colors = [
                (255, 255, 255),  # White
                (255, 255, 0),    # Yellow
                (0, 255, 255),    # Cyan
                (0, 255, 0),      # Green
                (255, 0, 255),    # Magenta
                (255, 0, 0),      # Red
                (0, 0, 255),      # Blue
                (0, 0, 0)         # Black
            ]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for i, color in enumerate(colors):
                frame[i * bar_height:(i + 1) * bar_height, :] = color
            return frame
        except Exception as e:
            logging.error(f"Exception while creating test pattern: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)  # Return a black frame on error

    def stop(self):
        """
        Stop the VirtualCamSendThread gracefully.
        """
        self.running = False
        self.wait()
        logging.info("VirtualCamSendThread stopped.")


class VideoThread(QThread):
    """
    Thread responsible for reading frames from the RTSP stream.
    Emits frames for GUI display and sends them to VirtualCamSendThread.
    Implements automatic reconnection with exponential backoff.
    """
    frame_ready_signal = pyqtSignal(np.ndarray)  # Signal with RTSP frame data
    virtual_cam_frame_ready_signal = pyqtSignal(np.ndarray)  # Signal with virtual cam frame data for GUI
    connection_lost_signal = pyqtSignal()
    connection_established_signal = pyqtSignal()
    reconnection_attempt_signal = pyqtSignal(int)  # Signal to indicate reconnection attempts

    def __init__(self, rtsp_url, virtual_cam_send_thread):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.virtual_cam_send_thread = virtual_cam_send_thread
        self.running = True
        self.connection_established = False
        self.container = None
        self.lock = threading.Lock()
        self.should_reconnect = True  # Flag to control reconnection attempts
        self.is_reconnecting = False   # Prevent multiple reconnection loops

        # Exponential backoff parameters
        self.initial_wait = 1  # Initial wait time in seconds
        self.max_wait = 32     # Maximum wait time in seconds
        self.current_wait = self.initial_wait

    def run(self):
        while self.running:
            try:
                self.connect_stream()
                if not self.connection_established:
                    # Connection failed, emit connection lost and attempt reconnection
                    if not self.is_reconnecting:
                        self.connection_lost_signal.emit()
                        self.virtual_cam_send_thread.enable_test_pattern()
                        self.is_reconnecting = True
                        self.attempt_reconnection()
                    continue

                # Reset reconnection parameters upon successful connection
                self.current_wait = self.initial_wait

                # Process frames
                for packet in self.container.demux():
                    with self.lock:
                        if not self.running:
                            logging.info("VideoThread stopping.")
                            break
                    if packet.dts is None:
                        continue
                    for frame in packet.decode():
                        with self.lock:
                            if not self.running:
                                logging.info("VideoThread stopping inside decode loop.")
                                break
                        if frame is None:
                            continue
                        img = frame.to_ndarray(format='bgr24')

                        # Emit signal to indicate a new RTSP frame is ready for GUI
                        self.frame_ready_signal.emit(img)

                        # Send the frame to VirtualCamSendThread
                        self.virtual_cam_send_thread.send_frame(img)

                        # Emit signal for virtual cam preview in GUI
                        self.virtual_cam_frame_ready_signal.emit(img)

            except av.AVError as e:
                logging.error(f"Stream error: {e}")
                self.handle_disconnection()
            except Exception as e:
                logging.error(f"Unexpected exception in VideoThread.run: {e}")
                self.handle_disconnection()
            finally:
                # Clean up container
                if self.container:
                    self.container.close()
                    self.container = None

    def connect_stream(self):
        with self.lock:
            if self.container:
                self.container.close()
                self.container = None

            try:
                # Set low-latency options for FFmpeg
                options = {
                    'rtsp_transport': 'tcp',       # Use TCP for RTSP transport
                    'stimeout': '5000000',         # Socket timeout in microseconds
                    'fflags': 'nobuffer',          # Disable buffering
                    'flags': 'low_delay',          # Enable low delay
                    'max_delay': '0',              # No maximum delay
                    'probesize': '32',             # Reduce probing size
                    'analyzeduration': '0'         # Reduce analysis duration
                }
                self.container = av.open(self.rtsp_url, options=options)
                logging.info("Successfully connected to the RTSP stream.")

                # Retrieve and set the actual frame rate from the stream
                stream = self.container.streams.video[0]
                if stream.average_rate:
                    frame_rate_n = stream.average_rate.numerator
                    frame_rate_d = stream.average_rate.denominator
                    logging.info(f"Stream frame rate set to {frame_rate_n}/{frame_rate_d} fps.")
                else:
                    # Fallback to default if average_rate is not available
                    frame_rate_n = 25
                    frame_rate_d = 1
                    logging.info(f"Stream frame rate not available. Using default {frame_rate_n}/{frame_rate_d} fps.")

                # Update the VirtualCamSendThread with the actual frame rate
                self.virtual_cam_send_thread.set_frame_rate(frame_rate_n, frame_rate_d)

                self.connection_established = True
                self.connection_established_signal.emit()

                # Disable test pattern since connection is established
                self.virtual_cam_send_thread.disable_test_pattern()

                # Reset reconnection flags
                self.is_reconnecting = False

            except av.AVError as e:
                logging.error(f"Failed to open stream: {e}")
                self.connection_established = False
                self.container = None
            except Exception as e:
                logging.error(f"Exception in connect_stream: {e}")
                self.connection_established = False
                self.container = None

    def handle_disconnection(self):
        with self.lock:
            if self.connection_established:
                self.connection_lost_signal.emit()
                self.virtual_cam_send_thread.enable_test_pattern()
                self.connection_established = False

            if self.is_reconnecting:
                logging.debug("Already in reconnection mode. Skipping new reconnection attempts.")
                return

            self.is_reconnecting = True

        if self.should_reconnect:
            self.attempt_reconnection()

    def attempt_reconnection(self):
        while self.running and self.should_reconnect and self.is_reconnecting:
            logging.info(f"Attempting to reconnect in {self.current_wait} seconds...")
            self.reconnection_attempt_signal.emit(self.current_wait)
            time.sleep(self.current_wait)
            if not self.running:
                logging.info("Reconnection stopped due to thread stop.")
                break
            self.connect_stream()
            if self.connection_established:
                logging.info("Reconnected to the RTSP stream.")
                break
            # Exponential backoff
            self.current_wait = min(self.current_wait * 2, self.max_wait)
            logging.info(f"Next reconnection attempt in {self.current_wait} seconds.")

    def stop(self):
        with self.lock:
            self.running = False
        self.wait()
        with self.lock:
            if self.container:
                self.container.close()
                self.container = None
            self.should_reconnect = False
            self.is_reconnecting = False
        logging.info("VideoThread stopped.")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTSP to Virtual Camera")

        # Initialize VirtualCamSendThread
        self.virtual_cam_send_thread = VirtualCamSendThread()
        self.virtual_cam_send_thread.virtual_cam_frame_sent_signal.connect(self.update_virtual_cam_preview)  # Connect frames to GUI
        self.virtual_cam_send_thread.start()

        # GUI Elements
        self.rtsp_input = QComboBox(self)
        self.rtsp_input.setEditable(True)
        self.rtsp_input.setInsertPolicy(QComboBox.NoInsert)
        self.rtsp_input.setMinimumWidth(300)
        self.rtsp_input.lineEdit().setPlaceholderText("Enter RTSP URL")

        self.connect_button = QPushButton("Connect", self)

        self.status_label = QLabel("Status: Disconnected", self)
        self.rtsp_label = QLabel(self)
        self.virtual_cam_label = QLabel(self)

        # Layouts
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.rtsp_input)
        input_layout.addWidget(self.connect_button)

        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("RTSP Video Stream"))
        video_layout.addWidget(QLabel("Virtual Camera Frame"))

        display_layout = QHBoxLayout()
        display_layout.addWidget(self.rtsp_label)
        display_layout.addWidget(self.virtual_cam_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(display_layout)

        self.setLayout(main_layout)

        # Flags
        self.is_streaming = False

        # Threads
        self.video_thread = None

        # Signals
        self.connect_button.clicked.connect(self.toggle_stream)

        # Store last 10 RTSP URLs
        self.rtsp_history = deque(maxlen=10)

    def toggle_stream(self):
        if self.is_streaming:
            # Disconnect
            self.disconnect_stream()
        else:
            # Connect
            self.connect_to_stream()

    def connect_to_stream(self):
        logging.info("connect_to_stream called.")
        try:
            rtsp_url = self.rtsp_input.currentText()
            logging.info(f"RTSP URL: {rtsp_url}")
            if not rtsp_url:
                QMessageBox.warning(self, "Input Error", "Please enter a valid RTSP URL.")
                return

            # Stop any existing threads
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
                logging.info("Existing VideoThread stopped.")

            # Create VideoThread and pass the VirtualCamSendThread
            self.video_thread = VideoThread(
                rtsp_url,
                self.virtual_cam_send_thread
            )
            logging.info("VideoThread created.")

            # Connect signals
            self.video_thread.frame_ready_signal.connect(self.update_rtsp_preview)
            self.video_thread.virtual_cam_frame_ready_signal.connect(self.update_virtual_cam_preview)
            self.video_thread.connection_lost_signal.connect(self.on_connection_lost)
            self.video_thread.connection_established_signal.connect(self.on_connection_established)
            self.video_thread.reconnection_attempt_signal.connect(self.on_reconnection_attempt)
            logging.info("VideoThread signals connected.")

            # Start VideoThread
            self.video_thread.start()
            logging.info("VideoThread started.")

            self.is_streaming = True
            self.status_label.setText("Status: Connecting...")
            self.connect_button.setText("Disconnect")

            # Add RTSP URL to history
            if rtsp_url not in self.rtsp_history:
                self.rtsp_history.appendleft(rtsp_url)
                self.rtsp_input.clear()
                self.rtsp_input.addItems(self.rtsp_history)
                self.rtsp_input.setCurrentIndex(0)
            logging.info("RTSP URL added to history.")
        except Exception as e:
            logging.error(f"Exception in connect_to_stream: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def disconnect_stream(self):
        logging.info("disconnect_stream called.")
        self.is_streaming = False  # Set this before stopping the thread
        try:
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
                logging.info("VideoThread stopped and set to None.")
        except Exception as e:
            logging.error(f"Exception while stopping VideoThread: {e}")

        # Clear the virtual camera preview
        self.status_label.setText("Status: Disconnected")
        self.connect_button.setText("Connect")
        self.rtsp_label.clear()
        self.virtual_cam_label.clear()

        # Ensure VirtualCamSendThread is set to send test pattern
        self.virtual_cam_send_thread.enable_test_pattern()

    def update_rtsp_preview(self, frame):
        """
        Slot to update the RTSP video preview in the GUI.
        """
        try:
            qt_img = self.convert_cv_qt(frame)
            self.rtsp_label.setPixmap(qt_img)
        except Exception as e:
            logging.error(f"Exception in update_rtsp_preview: {e}")

    def update_virtual_cam_preview(self, frame):
        """
        Slot to update the virtual camera video preview in the GUI.
        """
        try:
            qt_img = self.convert_cv_qt(frame)
            self.virtual_cam_label.setPixmap(qt_img)
        except Exception as e:
            logging.error(f"Exception in update_virtual_cam_preview: {e}")

    def on_connection_lost(self):
        logging.warning("Connection lost.")
        self.status_label.setText("Status: Connection Lost. Sending Test Pattern.")
        # Enable test pattern in VirtualCamSendThread
        self.virtual_cam_send_thread.enable_test_pattern()

    def on_connection_established(self):
        logging.info("Connection established.")
        self.status_label.setText("Status: Connected")
        # Disable test pattern in VirtualCamSendThread
        self.virtual_cam_send_thread.disable_test_pattern()

    def on_reconnection_attempt(self, wait_time):
        logging.info(f"Reconnection attempt in {wait_time} seconds.")
        self.status_label.setText(f"Status: Reconnecting in {wait_time} seconds...")

    def closeEvent(self, event):
        logging.info("Closing application.")
        try:
            if self.video_thread:
                self.video_thread.stop()
        except Exception as e:
            logging.error(f"Exception while stopping VideoThread: {e}")

        try:
            if self.virtual_cam_send_thread:
                self.virtual_cam_send_thread.stop()
        except Exception as e:
            logging.error(f"Exception while stopping VirtualCamSendThread: {e}")

        event.accept()

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap."""
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            p = QPixmap.fromImage(convert_to_Qt_format)
            p = p.scaled(480, 270, Qt.KeepAspectRatio)
            return p
        except Exception as e:
            logging.error(f"Exception in convert_cv_qt: {e}")
            return QPixmap()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
