import io
import traceback
from typing import Dict, List, Literal, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import viser


class PoseControllerFolder:
    SEPARATOR = " "

    def __init__(self, viewer, server: viser.ViserServer):
        self.viewer = viewer
        self.server = server

        self._setup()

    def _setup(self):
        server = self.server

        with server.gui.add_folder("Pose Controller"):
            self.pose_text = server.gui.add_text(label="Pose", initial_value="")
            self.apply_pose_button = server.gui.add_button(label="Apply Pose")
            self.download_image_button = server.gui.add_button(label="Download Image")
        server.on_client_connect(self._register_camera_hook)
        self.apply_pose_button.on_click(self._apply_camera_pose)
        self.download_image_button.on_click(self._send_image_download_request)

    def _register_camera_hook(self, client: viser.ClientHandle):
        client.camera.on_update(self._update_camera_pose_gui)

    def _update_camera_pose_gui(self, event: viser.GuiEvent):
        camera = event.client.camera
        self.pose_text.value = self.pose2str(camera.wxyz, camera.position)

    def _apply_camera_pose(self, event: viser.GuiEvent):
        with event.client.atomic():
            wxyz, position = self.str2pose(self.pose_text.value)
            event.client.camera.wxyz = wxyz
            event.client.camera.position = position

    def _send_image_download_request(self, event: viser.GuiEvent):
        base64_data = self.render_and_encode(event)
        if base64_data is None:
            return
        filename = "renderer{}".format(".png" if self.viewer.image_format == "png" else ".jpg")
        self.server.send_file_download(filename, base64_data)

    def render_and_encode(self, event: viser.GuiEvent):
        # render image
        output_pkg = self.render_image(event)
        if output_pkg is None:
            return None

        image, jpeg_quality = output_pkg
        _, base64_data = self.encode_image(image, format=self.viewer.image_format, jpeg_quality=jpeg_quality)

        return base64_data

    def render_image(self, event: viser.GuiEvent):
        """modified from internal.viewer.client.ClientThread.render_and_send()"""
        client_thread = self.viewer.clients[event.client.client_id]
        with event.client.atomic():
            max_res, jpeg_quality = client_thread.get_render_options()
            camera = client_thread.get_camera(
                client_thread.client.camera,
                image_size=max_res,
                appearance_id=self.viewer.get_appearance_id_value(),
                time_value=self.viewer.time_slider.value,
                camera_transform=self.viewer.camera_transform,
            ).to_device(self.viewer.device)

        with torch.no_grad():
            try:
                image = client_thread.renderer.get_outputs(camera, scaling_modifier=self.viewer.scaling_modifier.value)
                image = torch.clamp(image, max=1.0)
                image = torch.permute(image, (1, 2, 0))

            except:
                traceback.print_exc()
                return
        return image.cpu().numpy(), jpeg_quality

    def encode_image(self, image: np.ndarray, format: Literal["jpeg", "png"] = "jpeg", jpeg_quality: int = None):
        """modified from viser._scene_api._encode_image_base64()"""
        # to uint8
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            if np.issubdtype(image.dtype, np.integer):
                image = np.clip(image, 0, 255).astype(np.uint8)

        media_type: Literal["image/jpeg", "image/png"]
        with io.BytesIO() as buffer:
            if format == "png":
                media_type = "image/png"
                iio.imwrite(buffer, image, extension=".png")
            elif format == "jpeg":
                media_type = "image/jpeg"
                iio.imwrite(buffer, image[..., :3], extension=".jpg", quality=75 if jpeg_quality is None else jpeg_quality)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # base64_data = base64.b64encode(buffer.getvalue()).decode("ascii")
            base64_data = buffer.getvalue()

        return media_type, base64_data

    def pose2str(self, wxyz: np.ndarray, position: np.ndarray) -> str:
        return self.SEPARATOR.join([f"{x:.5f}" for x in wxyz] + [f"{x:.5f}" for x in position])

    def str2pose(self, s: str) -> Tuple[np.ndarray, np.ndarray]:
        vals = [float(x) for x in s.split(self.SEPARATOR)]
        wxyz = np.array(vals[:4])
        position = np.array(vals[4:])
        return wxyz, position
