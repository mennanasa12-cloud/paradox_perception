#!/usr/bin/env python3
import os
import sys
import time

import cv2
import numpy as np
import torch
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# BGR colors: regolith, crater, rock, mountain, sky
PALETTE_BGR = np.array([[156, 70, 187], [200, 0, 120], [80, 250, 232],
                        [31, 69, 173], [248, 201, 34]], np.uint8)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        defaults = dict(repo_root="~/paradox_perception", seg_weights="", det_weights="",
                        input_topic="/camera/image", seg_size=512,
                        det_conf=0.25, device="cuda")
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        g = lambda k: self.get_parameter(k).value
        path = lambda k: os.path.expanduser(g(k))

        sys.path.insert(0, path("repo_root"))
        from src.perception.segmentation.model import build_model, predict
        from src.perception.detection.detector import LunarDetector

        use_cuda = g("device") == "cuda" and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        self.seg_size = int(g("seg_size"))
        self.seg_predict = predict

        self.seg_model = build_model(5)
        self.seg_model.load_state_dict(torch.load(path("seg_weights"), map_location="cpu"))
        self.seg_model.to(self.device).eval()
        self.detector = LunarDetector(path("det_weights"), conf=float(g("det_conf")),
                                      device=0 if use_cuda else "cpu")

        self.bridge = CvBridge()
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        out_qos = QoSProfile(depth=1)
        self.create_subscription(Image, g("input_topic"), self.on_image, sensor_qos)
        self.pub_seg = self.create_publisher(Image, "/perception/segmentation", out_qos)
        self.pub_det = self.create_publisher(Detection2DArray, "/perception/detections", out_qos)
        self.pub_dbg = self.create_publisher(Image, "/perception/debug_image", out_qos)

        self._warmup()
        self.n, self.t_acc = 0, 0.0
        self.get_logger().info(
            f"perception_node ready on {self.device}, listening on {g('input_topic')}")

    def _warmup(self):
        dummy = np.zeros((480, 640, 3), np.uint8)
        self.segment(dummy)
        self.detector.detect(dummy)

    @torch.no_grad()
    def segment(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.seg_size, self.seg_size), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            mask = self.seg_predict(self.seg_model, t)
        mask = mask[0].to(torch.uint8).cpu().numpy()
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def on_image(self, msg):
        t0 = time.time()
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        mask = self.segment(bgr)
        dets = self.detector.detect(bgr)

        seg_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        seg_msg.header = msg.header
        self.pub_seg.publish(seg_msg)

        arr = Detection2DArray()
        arr.header = msg.header
        for d in dets:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (d.x1 + d.x2) / 2.0
            det.bbox.center.position.y = (d.y1 + d.y2) / 2.0
            det.bbox.size_x = d.x2 - d.x1
            det.bbox.size_y = d.y2 - d.y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.cls_name
            hyp.hypothesis.score = d.conf
            det.results.append(hyp)
            arr.detections.append(det)
        self.pub_det.publish(arr)

        if self.pub_dbg.get_subscription_count() > 0:
            vis = cv2.addWeighted(bgr, 0.6, PALETTE_BGR[mask], 0.4, 0)
            for d in dets:
                col = (0, 255, 0) if d.cls_name == "crater" else (0, 0, 255)
                cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
                cv2.putText(vis, f"{d.cls_name} {d.conf:.2f}", (int(d.x1), max(12, int(d.y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            dbg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            dbg.header = msg.header
            self.pub_dbg.publish(dbg)

        self.n += 1
        self.t_acc += time.time() - t0
        if self.n % 30 == 0:
            self.get_logger().info(
                f"avg {1000 * self.t_acc / self.n:.0f} ms/frame, {len(dets)} detections")


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()